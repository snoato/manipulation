"""Tabletop dataset generation — PDDL problems with motion-planning validation.

For each generated instance:
  1. A random cylinder scene is sampled.
  2. BFS finds the shortest pick-sequence to reach every possible target.
  3. Target is selected with uniform distribution *across plan lengths* so
     trivially-reachable (plan-length-1) targets do not dominate the dataset.
  4. Feasibility is validated with IK + RRT* (GraspPlanner, no hardcoded offsets).
  5. PDDL problem, plan file, and optional visualisation are saved.

Usage:
    python -m manipulation.symbolic.domains.tabletop.generate_data
    python -m manipulation.symbolic.domains.tabletop.generate_data \\
        --num-train 500 --num-test 50 --num-val 50 \\
        --min-objects 3 --max-objects 7 --num-workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from manipulation import FrankaEnvironment, RRTStar
from manipulation.planners.grasp_planner import GraspPlanner, GraspType
from manipulation.symbolic.domains.tabletop.grid_domain import GridDomain
from manipulation.symbolic.domains.tabletop.state_manager import StateManager
from manipulation.symbolic.domains.tabletop.visualization import visualize_grid_state

_HERE   = Path(__file__).parent
_XML    = (
    _HERE / ".." / ".." / ".." / "environments"
    / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
).resolve()
_DOMAIN = (_HERE / "pddl" / "domain.pddl").resolve()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR           = Path("data/tabletop")
_DEFAULT_TRAIN                = 1000
_DEFAULT_TEST                 = 100
_DEFAULT_VAL                  = 100
_DEFAULT_MIN_OBJECTS          = 3
_DEFAULT_MAX_OBJECTS          = 7
_DEFAULT_GRID_WIDTH           = 0.4
_DEFAULT_GRID_HEIGHT          = 0.3
_DEFAULT_CELL_SIZE            = 0.04
_DEFAULT_GRID_OFFSET_X        = 0.05   # calibrated so all cells are IK-reachable
_DEFAULT_GRID_OFFSET_Y        = 0.25   # calibrated so all cells are IK-reachable
_DEFAULT_PLACEMENT_MARGIN     = 1      # free cardinal cells around each cylinder
_DEFAULT_RRT_ITERS            = 1000   # benchmarked: fastest zero-false-negative
_DEFAULT_IK_ITERS             = 100    # benchmarked: fastest zero-false-negative
_DEFAULT_IK_POS_THRESH        = 0.005  # benchmarked: fastest zero-false-negative
_DEFAULT_NUM_WORKERS          = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate tabletop PDDL dataset with motion-planning validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",          type=str,   default=str(_DEFAULT_OUTPUT_DIR))
    p.add_argument("--num-train",           type=int,   default=_DEFAULT_TRAIN)
    p.add_argument("--num-test",            type=int,   default=_DEFAULT_TEST)
    p.add_argument("--num-val",             type=int,   default=_DEFAULT_VAL)
    p.add_argument("--min-objects",         type=int,   default=_DEFAULT_MIN_OBJECTS)
    p.add_argument("--max-objects",         type=int,   default=_DEFAULT_MAX_OBJECTS)
    p.add_argument("--grid-width",          type=float, default=_DEFAULT_GRID_WIDTH)
    p.add_argument("--grid-height",         type=float, default=_DEFAULT_GRID_HEIGHT)
    p.add_argument("--cell-size",           type=float, default=_DEFAULT_CELL_SIZE)
    p.add_argument("--grid-offset-x",       type=float, default=_DEFAULT_GRID_OFFSET_X)
    p.add_argument("--grid-offset-y",       type=float, default=_DEFAULT_GRID_OFFSET_Y)
    p.add_argument("--placement-margin",    type=int,   default=_DEFAULT_PLACEMENT_MARGIN,
                   help="Free cells required in each cardinal direction around each cylinder")
    p.add_argument("--rrt-iters",           type=int,   default=_DEFAULT_RRT_ITERS,
                   help="RRT* iteration budget per planning call")
    p.add_argument("--ik-iters",            type=int,   default=_DEFAULT_IK_ITERS)
    p.add_argument("--ik-pos-thresh",       type=float, default=_DEFAULT_IK_POS_THRESH)
    p.add_argument("--num-workers",         type=int,   default=_DEFAULT_NUM_WORKERS,
                   help="Parallel workers (0 = all CPUs)")
    p.add_argument("--no-viz",              action="store_true",
                   help="Disable visualisation image generation")
    p.add_argument("--seed",                type=int,   default=None)
    p.add_argument("--wandb",               action="store_true")
    p.add_argument("--wandb-project",       type=str,   default="tabletop-data-generation")
    p.add_argument("--wandb-run-name",      type=str,   default=None)
    p.add_argument("--domain-src",          type=str,   default="",
                   help="Override path to domain.pddl")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _patch_fast_step(env: FrankaEnvironment) -> None:
    """Bypass RateLimiter for headless data generation."""
    _dt = env.model.opt.timestep
    def _fast_step():
        mujoco.mj_step(env.model, env.data)
        env.sim_time += _dt
        return _dt
    env.step = _fast_step


def _build_env(args) -> tuple[FrankaEnvironment, RRTStar, GridDomain, StateManager, GraspPlanner]:
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    _patch_fast_step(env)

    env.ik.max_iters     = args.ik_iters
    env.ik.pos_threshold = args.ik_pos_thresh

    planner = RRTStar(env)
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grid = GridDomain(
        model=env.model,
        cell_size=args.cell_size,
        working_area=(args.grid_width, args.grid_height),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=args.grid_offset_x,
        grid_offset_y=args.grid_offset_y,
    )

    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    return env, planner, grid, state_manager, grasp_planner


# ---------------------------------------------------------------------------
# Cylinder placement
# ---------------------------------------------------------------------------

def _sample_state(state_manager: StateManager,
                  n_cylinders: int,
                  placement_margin: int) -> int:
    """Place cylinders with a cardinal-cell margin between them.

    Cells in the margin are not reserved — they may overlap between cylinders —
    but each centre cell must have *placement_margin* free cells in each of the
    four cardinal directions.

    Returns the number of cylinders actually placed.
    """
    n_cylinders = max(1, min(30, int(n_cylinders)))

    for i in range(30):
        state_manager._hide_cylinder(i)

    all_cells = [
        (x, y)
        for x in range(state_manager.grid.cells_x)
        for y in range(state_manager.grid.cells_y)
    ]
    np.random.shuffle(all_cells)

    chosen: list[tuple[int, int]] = []

    def _valid(cx: int, cy: int) -> bool:
        for ox, oy in chosen:
            if ox == cx and abs(oy - cy) <= placement_margin:
                return False
            if oy == cy and abs(ox - cx) <= placement_margin:
                return False
        return True

    for cx, cy in all_cells:
        if _valid(cx, cy):
            chosen.append((cx, cy))
            if len(chosen) >= n_cylinders:
                break

    selected = np.random.choice(30, len(chosen), replace=False)
    for cyl_idx, (cx, cy) in zip(selected, chosen):
        cell_name = f"cell_{cx}_{cy}"
        center_x, center_y = state_manager.grid.cells[cell_name]["center"]
        _, height = state_manager.CYLINDER_SPECS[int(cyl_idx)]
        center_z = state_manager.grid.table_height + height + 0.002
        state_manager._set_cylinder_position(int(cyl_idx), center_x, center_y, center_z)

    # Let cylinders drop the 2mm clearance and reach resting contact (~40 steps),
    # then zero velocities to kill the MuJoCo contact micro-oscillation.
    for _ in range(40):
        state_manager.env.step()
    state_manager.env.reset_velocities()
    mujoco.mj_forward(state_manager.env.model, state_manager.env.data)
    return len(chosen)


# ---------------------------------------------------------------------------
# Grasp feasibility (lightweight, uses collision exceptions — no scene rebuild)
# ---------------------------------------------------------------------------

def _pick_candidate(candidates):
    """Prefer FRONT approach for tall cylinders (same as all other examples)."""
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _validate_pick(
    env: FrankaEnvironment,
    planner: RRTStar,
    grasp_planner: GraspPlanner,
    target_obj: str,
    ignored_objects: list[str] | None = None,
    rrt_iters: int = _DEFAULT_RRT_ITERS,
) -> bool:
    """Check whether *target_obj* can be approached and grasped.

    Uses collision exceptions to virtually remove *ignored_objects* from the
    scene — no cylinders are moved, so the check is fast.  The robot is reset
    to its home configuration before each call.
    """
    saved_exceptions = list(env.collision_exceptions)
    saved_qpos = env.data.qpos.copy()
    saved_qvel = env.data.qvel.copy()
    try:
        env.set_collision_exceptions(list(set((ignored_objects or []) + [target_obj])))

        # Reset robot to home
        env.data.qpos[:8] = env.initial_qpos[:8]
        env.data.ctrl[:8] = env.initial_ctrl[:8]
        env.controller.stop()
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)

        dt = env.model.opt.timestep

        cyl_pos  = env.get_object_position(target_obj)
        half_size = env.get_object_half_size(target_obj)
        cyl_quat = env.get_object_orientation(target_obj)
        candidate = _pick_candidate(
            grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
        )
        if candidate is None:
            return False

        # Approach IK
        env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
        if not env.ik.converge_ik(dt):
            return False

        # Approach RRT*
        path = planner.plan_to_pose(
            candidate.approach_pos, candidate.grasp_quat,
            dt=dt, max_iterations=rrt_iters,
        )
        if path is None:
            return False

        # Execute approach so grasp check starts from the correct config
        env.execute_path(path, planner)
        env.wait_idle(max_steps=5000, settle_steps=30)

        # Grasp IK
        env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
        if not env.ik.converge_ik(dt):
            return False

        # Grasp RRT*
        path = planner.plan_to_pose(
            candidate.grasp_pos, candidate.grasp_quat,
            dt=dt, max_iterations=rrt_iters,
        )
        return path is not None

    except Exception:
        return False
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        env.set_collision_exceptions(saved_exceptions)


# ---------------------------------------------------------------------------
# BFS — shortest plan for one target
# ---------------------------------------------------------------------------

def _find_plan(
    env: FrankaEnvironment,
    planner: RRTStar,
    grasp_planner: GraspPlanner,
    target_obj: str,
    all_objects: list[str],
    cache: dict,
    rrt_iters: int,
) -> list[str] | None:
    """BFS over removal sequences to find shortest plan reaching *target_obj*.

    Returns a list ending with *target_obj* (each element is picked then
    dropped), or None if no plan exists.
    """
    all_set   = frozenset(all_objects)
    start     = all_set
    queue     = deque([(start, [])])
    visited   = {start}

    while queue:
        current, plan = queue.popleft()
        ignored = list(all_set - current)

        key = (current, target_obj)
        ok  = cache.get(key)
        if ok is None:
            ok = _validate_pick(env, planner, grasp_planner, target_obj,
                                ignored, rrt_iters)
            cache[key] = ok
        if ok:
            return plan + [target_obj]

        for obj in sorted(current - {target_obj}):
            key2 = (current, obj)
            ok2  = cache.get(key2)
            if ok2 is None:
                ok2 = _validate_pick(env, planner, grasp_planner, obj,
                                     ignored, rrt_iters)
                cache[key2] = ok2
            if ok2:
                nxt = current - {obj}
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, plan + [obj]))

    return None


# ---------------------------------------------------------------------------
# Target selection with plan-length diversity
# ---------------------------------------------------------------------------

def _select_target(
    env: FrankaEnvironment,
    planner: RRTStar,
    grasp_planner: GraspPlanner,
    cylinders: list[str],
    rrt_iters: int,
) -> tuple[str, list[str]] | tuple[None, None]:
    """Run BFS for every cylinder and pick a target with uniform distribution
    across plan lengths.

    Running BFS for all cylinders shares a single reachability cache, so the
    overhead over running it for just one target is modest.

    Returns (target_cylinder, plan_sequence) or (None, None) if no cylinder
    is reachable.
    """
    cache: dict = {}
    plans: dict[str, list[str] | None] = {}

    for cyl in cylinders:
        plans[cyl] = _find_plan(
            env, planner, grasp_planner, cyl, cylinders, cache, rrt_iters
        )

    # Group feasible cylinders by plan length
    by_length: dict[int, list[str]] = {}
    for cyl, plan in plans.items():
        if plan is not None:
            by_length.setdefault(len(plan), []).append(cyl)

    if not by_length:
        return None, None

    # Sample uniformly across plan lengths, then uniformly within the group.
    # This prevents the dataset from being dominated by trivially accessible
    # (plan-length-1) targets.
    chosen_length = int(np.random.choice(sorted(by_length.keys())))
    target = str(np.random.choice(by_length[chosen_length]))
    return target, plans[target]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _append_metadata(path: Path, args, split: str, config_num: int,
                     requested: int, actual: int, target: str) -> None:
    with open(path, "a") as f:
        f.write("\n; --- generation metadata ---\n")
        f.write(f"; split: {split}\n")
        f.write(f"; config_num: {config_num}\n")
        f.write(f"; requested_cylinders: {requested}\n")
        f.write(f"; actual_cylinders: {actual}\n")
        f.write(f"; target_cylinder: {target}\n")
        f.write(f"; min_objects: {args.min_objects}\n")
        f.write(f"; max_objects: {args.max_objects}\n")
        f.write(f"; grid_width_m: {args.grid_width}\n")
        f.write(f"; grid_height_m: {args.grid_height}\n")
        f.write(f"; cell_size_m: {args.cell_size}\n")
        f.write(f"; grid_offset_x_m: {args.grid_offset_x}\n")
        f.write(f"; grid_offset_y_m: {args.grid_offset_y}\n")
        f.write(f"; placement_margin_cells: {args.placement_margin}\n")
        f.write(f"; rrt_iters: {args.rrt_iters}\n")
        f.write(f"; ik_iters: {args.ik_iters}\n")
        f.write(f"; ik_pos_thresh: {args.ik_pos_thresh}\n")


def _write_plan(path: Path, plan: list[str], cylinder_cells: dict[str, str],
                target: str, cylinders: list[str], split: str,
                config_num: int, gen_sec: float) -> int:
    action_count = len(plan) * 2 - 1
    with open(path, "w") as f:
        for obj in plan[:-1]:
            cell = cylinder_cells.get(obj, "")
            f.write(f"(pick {obj} {cell})\n" if cell else f"(pick {obj})\n")
            f.write(f"(drop {obj})\n")
        last = plan[-1]
        cell = cylinder_cells.get(last, "")
        f.write(f"(pick {last} {cell})\n" if cell else f"(pick {last})\n")
        f.write(f"; Total actions: {action_count}\n")
        f.write(f"; split: {split}\n")
        f.write(f"; config_num: {config_num}\n")
        f.write(f"; Target: {target}\n")
        f.write(f"; Plan: {', '.join(plan)}\n")
        f.write(f"; Objects: {', '.join(cylinders)}\n")
        f.write(f"; Generation time: {gen_sec:.2f}s\n")
    return action_count


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _init_stats() -> dict:
    return {
        "attempts": 0, "accepted": 0,
        "reject_min_objects": 0, "reject_no_plan": 0,
        "requested_hist": Counter(), "actual_hist": Counter(),
        "plan_len_hist": Counter(), "gen_seconds": [],
    }


def _merge_stats(items: list[dict]) -> dict:
    merged = _init_stats()
    for s in items:
        for k in ("attempts", "accepted", "reject_min_objects", "reject_no_plan"):
            merged[k] += s[k]
        for k in ("requested_hist", "actual_hist", "plan_len_hist"):
            merged[k].update(s[k])
        merged["gen_seconds"].extend(s["gen_seconds"])
    return merged


def _print_stats(label: str, stats: dict, wall_time: float) -> None:
    n, ok = stats["attempts"], stats["accepted"]
    print(f"  [{label}] attempts={n} accepted={ok} "
          f"acceptance={ok/max(n,1):.3f}")
    print(f"  [{label}] rejects: min_objects={stats['reject_min_objects']} "
          f"no_plan={stats['reject_no_plan']}")
    print(f"  [{label}] plan_len={dict(sorted(stats['plan_len_hist'].items()))}")
    if stats["gen_seconds"]:
        arr = np.array(stats["gen_seconds"])
        print(f"  [{label}] gen_sec: mean={arr.mean():.2f} "
              f"median={np.median(arr):.2f} p90={np.percentile(arr,90):.2f}")
    print(f"  [{label}] wall_time={wall_time:.1f}s")


# ---------------------------------------------------------------------------
# Core generation loop (single worker)
# ---------------------------------------------------------------------------

def _generate_split(split: str, count: int, args,
                    env: FrankaEnvironment, planner: RRTStar,
                    state_manager: StateManager, grasp_planner: GraspPlanner,
                    output_base: Path, wandb_run=None) -> dict:
    if count <= 0:
        return _init_stats()

    problem_dir = output_base / split
    viz_dir     = output_base / split / "viz"
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    stats      = _init_stats()
    generated  = 0
    t_start    = time.time()

    while generated < count:
        t0 = time.time()
        stats["attempts"] += 1
        config_num = generated + 1

        n_req = int(np.random.randint(args.min_objects, args.max_objects + 1))
        stats["requested_hist"][n_req] += 1

        _sample_state(state_manager, n_cylinders=n_req,
                      placement_margin=args.placement_margin)

        state    = state_manager.ground_state()
        cylinders = sorted(state["cylinders"].keys())
        stats["actual_hist"][len(cylinders)] += 1

        if len(cylinders) < args.min_objects:
            stats["reject_min_objects"] += 1
            continue

        target, plan = _select_target(
            env, planner, grasp_planner, cylinders, args.rrt_iters
        )

        if plan is None:
            stats["reject_no_plan"] += 1
            continue

        # Write PDDL problem
        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}", problem_path,
            goal_string=f"(holding {target})",
        )
        _append_metadata(problem_path, args, split, config_num,
                         requested=n_req, actual=len(cylinders), target=target)

        # Write plan
        cylinder_cells = {k: v[0] for k, v in state["cylinders"].items() if v}
        action_count   = _write_plan(
            problem_dir / f"config_{config_num}.pddl.plan",
            plan, cylinder_cells, target, cylinders,
            split, config_num, time.time() - t_start,
        )
        stats["plan_len_hist"][action_count] += 1

        # Visualise
        if not args.no_viz:
            visualize_grid_state(
                state_manager,
                save_path=viz_dir / f"config_{config_num}.png",
                title=f"{split.capitalize()} {config_num}  plan_len={len(plan)}",
                target_cylinder=target,
            )

        if wandb_run is not None:
            log: dict = {
                f"{split}/n_objects":   len(cylinders),
                f"{split}/plan_length": len(plan),
                f"{split}/progress":    (generated + 1) / count,
            }
            if not args.no_viz:
                viz_p = viz_dir / f"config_{config_num}.png"
                if viz_p.exists():
                    log[f"{split}/visualization"] = wandb.Image(str(viz_p))
            wandb.log(log)

        generated += 1
        stats["accepted"] += 1
        stats["gen_seconds"].append(time.time() - t0)

        if generated % 10 == 0 or generated == 1:
            elapsed = time.time() - t_start
            eta     = elapsed / generated * (count - generated)
            print(f"  [{split}] {generated}/{count}  "
                  f"plan_len={len(plan)}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    _print_stats(split, stats, time.time() - t_start)
    return stats


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker(worker_id: int, split: str, start_idx: int, count: int,
            args_dict: dict, result_queue: mp.Queue) -> None:
    # Reconstruct args
    class _Args:
        pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    seed = (
        args.seed + worker_id * 10_000 + hash(split) % 10_000
        if args.seed is not None
        else int(time.time() * 1000) % (2**31) + worker_id * 10_000
    )
    np.random.seed(seed)

    env, planner, grid, state_manager, grasp_planner = _build_env(args)

    output_base = Path(args.output_dir)
    problem_dir = output_base / split
    viz_dir     = output_base / split / "viz"
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    stats      = _init_stats()
    generated  = 0
    t_start    = time.time()

    while generated < count:
        t0 = time.time()
        stats["attempts"] += 1
        config_num = start_idx + generated + 1

        n_req = int(np.random.randint(args.min_objects, args.max_objects + 1))
        stats["requested_hist"][n_req] += 1

        _sample_state(state_manager, n_cylinders=n_req,
                      placement_margin=args.placement_margin)

        state     = state_manager.ground_state()
        cylinders = sorted(state["cylinders"].keys())
        stats["actual_hist"][len(cylinders)] += 1

        if len(cylinders) < args.min_objects:
            stats["reject_min_objects"] += 1
            continue

        target, plan = _select_target(
            env, planner, grasp_planner, cylinders, args.rrt_iters
        )

        if plan is None:
            stats["reject_no_plan"] += 1
            continue

        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}", problem_path,
            goal_string=f"(holding {target})",
        )
        _append_metadata(problem_path, args, split, config_num,
                         requested=n_req, actual=len(cylinders), target=target)

        cylinder_cells = {k: v[0] for k, v in state["cylinders"].items() if v}
        action_count   = _write_plan(
            problem_dir / f"config_{config_num}.pddl.plan",
            plan, cylinder_cells, target, cylinders,
            split, config_num, time.time() - t_start,
        )
        stats["plan_len_hist"][action_count] += 1

        if not args.no_viz:
            visualize_grid_state(
                state_manager,
                save_path=viz_dir / f"config_{config_num}.png",
                title=f"{split.capitalize()} {config_num}  plan_len={len(plan)}",
                target_cylinder=target,
            )

        generated += 1
        stats["accepted"] += 1
        stats["gen_seconds"].append(time.time() - t0)

        if generated % 10 == 0 or generated == 1:
            elapsed = time.time() - t_start
            print(f"  [W{worker_id}/{split}] {generated}/{count}  "
                  f"plan_len={len(plan)}  elapsed={elapsed:.0f}s")

    env.close()
    result_queue.put({
        "worker_id": worker_id,
        "split":     split,
        "generated": generated,
        "time":      time.time() - t_start,
        "stats":     stats,
    })


def _generate_split_parallel(split: str, count: int, args,
                              num_workers: int) -> dict | None:
    if count <= 0:
        return None

    print(f"\n{'='*65}")
    print(f"  {split}  ({count} examples, {num_workers} workers)")
    print(f"{'='*65}")

    output_base = Path(args.output_dir)
    (output_base / split).mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        (output_base / split / "viz").mkdir(parents=True, exist_ok=True)

    base = count // num_workers
    rem  = count % num_workers
    queue: mp.Queue = mp.Queue()
    procs = []
    idx   = 0
    t0    = time.time()

    for wid in range(num_workers):
        wcount = base + (1 if wid < rem else 0)
        if wcount > 0:
            p = mp.Process(
                target=_worker,
                args=(wid, split, idx, wcount, vars(args), queue),
            )
            procs.append(p)
            p.start()
            idx += wcount

    for p in procs:
        p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())

    wall = time.time() - t0
    merged = _merge_stats([r["stats"] for r in results])
    print(f"  {split} done: {sum(r['generated'] for r in results)} configs "
          f"in {wall:.1f}s")
    _print_stats(split, merged, wall)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    num_workers = args.num_workers or mp.cpu_count()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Copy domain.pddl
    domain_src = Path(args.domain_src) if args.domain_src else _DOMAIN
    domain_dst = output_base / "domain.pddl"
    if domain_src.exists():
        shutil.copy(domain_src, domain_dst)
        print(f"Copied domain.pddl → {domain_dst}")
    else:
        print(f"WARNING: domain.pddl not found at {domain_src}")

    print("=" * 65)
    print("  Tabletop Dataset Generation")
    print("=" * 65)
    print(f"  output:   {output_base}")
    print(f"  splits:   train={args.num_train}  test={args.num_test}  val={args.num_val}")
    print(f"  objects:  {args.min_objects}–{args.max_objects}")
    print(f"  grid:     {args.grid_width}×{args.grid_height}m  "
          f"cell={args.cell_size}m  "
          f"offset=({args.grid_offset_x},{args.grid_offset_y})")
    print(f"  ik:       iters={args.ik_iters}  pos_thresh={args.ik_pos_thresh}")
    print(f"  rrt:      iters={args.rrt_iters}")
    print(f"  workers:  {num_workers}")

    # wandb
    wandb_run = None
    if args.wandb:
        if wandb is None:
            print("WARNING: wandb not installed — skipping")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )

    t_total = time.time()
    all_stats = []

    if num_workers > 1:
        for split, n in [("train", args.num_train),
                          ("test",  args.num_test),
                          ("validation", args.num_val)]:
            s = _generate_split_parallel(split, n, args, num_workers)
            if s:
                all_stats.append(s)
    else:
        env, planner, grid, state_manager, grasp_planner = _build_env(args)
        info = grid.get_grid_info()
        print(f"\n  Grid: {info['grid_dimensions'][0]}×{info['grid_dimensions'][1]} cells  "
              f"({info['cell_size']*100:.1f}cm each)")

        for split, n in [("train", args.num_train),
                          ("test",  args.num_test),
                          ("validation", args.num_val)]:
            s = _generate_split(split, n, args, env, planner,
                                 state_manager, grasp_planner,
                                 output_base, wandb_run)
            all_stats.append(s)

        env.close()

    wall = time.time() - t_total
    total = args.num_train + args.num_test + args.num_val
    print(f"\n{'='*65}")
    print(f"  Done: {total} configs in {wall:.1f}s "
          f"({wall/max(total,1):.2f}s each)")

    if all_stats:
        _print_stats("global", _merge_stats(all_stats), wall)

    if wandb_run is not None:
        wandb.log({"total_time": wall,
                   "avg_per_config": wall / max(total, 1)})
        wandb.finish()


if __name__ == "__main__":
    main()
