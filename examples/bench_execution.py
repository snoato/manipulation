"""Headless execution benchmark: replay generated plans, measure success rate.

Loads PDDL problem files + plan files from a directory (e.g.
/tmp/tabletop_test/train), executes each plan action-by-action using the
specified planning strategy, and reports per-action-type success rates.

Usage::

    cd examples
    # Single strategy
    python bench_execution.py /tmp/tabletop_test/train --strategy baseline
    python bench_execution.py /tmp/tabletop_test/train --strategy combined

    # Compare all strategies (runs each config once per strategy)
    python bench_execution.py /tmp/tabletop_test/train --strategy all

    # Limit to a few configs for a quick sanity check
    python bench_execution.py /tmp/tabletop_test/train --strategy combined --max_configs 5 -v

Strategies
----------
baseline  RRTStar only  (current pddl_interactive behaviour)
retry     RRTStar x2 attempts before giving up
fallback  RRTStar → FeasibilityRRT fallback
via_home  fallback + via-HOME decomposition
combined  retry + fallback + via-HOME  (most robust)
"""

import re
import sys
import time
import argparse
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from manipulation import RRTStar, FeasibilityRRT, ControllerStatus
from manipulation.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from manipulation.planners.robust_planner import RobustPlanner
from manipulation.symbolic.domains.tabletop import GridDomain, StateManager
from manipulation.symbolic.domains.tabletop.env_builder import make_symbolic_builder

# ── Scene / planner constants (match pddl_interactive.py) ─────────────────
_GRID_WIDTH    = 0.4
_GRID_HEIGHT   = 0.3
_CELL_SIZE     = 0.04
_GRID_OFFSET_X = 0.05
_GRID_OFFSET_Y = 0.25

_RRT_ITERS      = 2000
_RRT_ITERS_FINE = 1000
_COL_STEPS      = 20
_CTRL_DELTA     = 0.05
_VEL_THRESHOLD  = 0.02

_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
_HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
_HOME_Q7   = _HOME_QPOS[:7]  # 7-DOF for planning


# ── Environment setup ──────────────────────────────────────────────────────

def _make_headless_env():
    """Create a FrankaEnvironment with the rate limiter bypassed."""
    env = make_symbolic_builder().build_env(rate=200.0)
    grid = GridDomain(
        model=env.model,
        cell_size=_CELL_SIZE,
        working_area=(_GRID_WIDTH, _GRID_HEIGHT),
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
        grid_offset_x=_GRID_OFFSET_X,
        grid_offset_y=_GRID_OFFSET_Y,
    )

    # Bypass the rate limiter — identical to ActionFeasibilityChecker
    _dt = env.model.opt.timestep
    def _fast_step():
        if env._attached is not None:
            env._apply_attachment()
        mujoco.mj_step(env.model, env.data)
        env.sim_time += _dt
        return _dt
    env.step = _fast_step

    return env, grid


def _make_planner(env, strategy: str):
    """Build the appropriate planner (or RobustPlanner wrapper) for *strategy*."""
    primary = RRTStar(env)
    primary.step_size             = 0.2
    primary.goal_sample_rate      = 0.2
    primary.collision_check_steps = _COL_STEPS

    if strategy == "baseline":
        return primary

    fallback = FeasibilityRRT(env)
    fallback.step_size             = 0.2
    fallback.collision_check_steps = _COL_STEPS

    return RobustPlanner(
        primary  = primary,
        fallback = fallback,
        home_q   = _HOME_Q7,
        strategy = strategy,
    )


# ── Execution helpers ──────────────────────────────────────────────────────

def _settle(env, threshold=_VEL_THRESHOLD, max_wait_steps=400):
    for _ in range(max_wait_steps):
        env.step()
        env.controller.step(delta=_CTRL_DELTA)
        if np.linalg.norm(env.data.qvel[:7]) < threshold:
            return


def _wait_idle(env, max_steps=8000, settle_steps=50):
    for _ in range(max_steps):
        env.controller.step(delta=_CTRL_DELTA)
        env.step()
        if env.controller.get_status() == ControllerStatus.IDLE:
            for _ in range(settle_steps):
                env.step()
            return True
    return False


def _pick_candidate(candidates):
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _execute_pick(env, planner, grasp_planner, cyl_name, state_manager):
    """Execute a pick action. Returns (success, failure_phase)."""
    dt = env.model.opt.timestep
    cyl_pos   = env.get_object_position(cyl_name)
    half_size = env.get_object_half_size(cyl_name)
    cyl_quat  = env.get_object_orientation(cyl_name)
    candidates = grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
    cand = _pick_candidate(candidates)
    if cand is None:
        return False, "no_grasp_candidate"

    env.add_collision_exception(cyl_name)

    # Approach
    path = planner.plan_to_pose(cand.approach_pos, cand.grasp_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        env.clear_collision_exceptions()
        return False, "approach_plan_fail"
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    # Grasp descent
    path = planner.plan_to_pose(cand.grasp_pos, cand.grasp_quat,
                                dt=dt, max_iterations=_RRT_ITERS_FINE)
    if path is None:
        env.clear_collision_exceptions()
        return False, "grasp_plan_fail"
    env.execute_path(path, planner, step_size=0.005)
    _wait_idle(env)

    _settle(env)
    env.controller.close_gripper()
    env.attach_object_to_ee(cyl_name)
    env.rest(1.0)

    # Transport to canonical hold pose
    transport_pos, transport_quat = state_manager.get_transport_pose()
    home_full = env.data.qpos.copy()
    home_full[:8] = _HOME_QPOS
    env.ik.update_configuration(home_full)
    env.ik.set_target_position(transport_pos, transport_quat)
    if not env.ik.converge_ik(env.model.opt.timestep):
        env.clear_collision_exceptions()
        return False, "transport_ik_fail"
    transport_q = env.ik.configuration.q[:7]

    # Lift the cylinder straight up before transport so it clears table-height
    # neighbours.  The exception is still active during this short move, so no
    # false positives from the cylinder touching its immediate neighbours at the
    # grasp position.  If the lift fails we skip it and proceed anyway.
    ee_pos  = env.data.site_xpos[env._ee_site_id].copy()
    lift_pos = ee_pos + np.array([0.0, 0.0, 0.12])
    lift_path = planner.plan_to_pose(lift_pos, cand.grasp_quat,
                                     dt=dt, max_iterations=_RRT_ITERS_FINE)
    if lift_path is not None:
        env.execute_path(lift_path, planner, step_size=0.005)
        _wait_idle(env)

    link7_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    saved_body_ids = env._collision_body_ids
    saved_held     = env._collision_held_body
    env._collision_body_ids  = saved_body_ids - {link7_id}
    env._collision_held_body = None

    path = planner.plan(env.data.qpos[:7], transport_q, max_iterations=_RRT_ITERS)

    env._collision_body_ids  = saved_body_ids
    env._collision_held_body = saved_held

    if path is None:
        env.clear_collision_exceptions()
        return False, "transport_plan_fail"

    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)
    env.clear_collision_exceptions()
    return True, "ok"


def _execute_put(env, planner, grasp_planner, grid, cyl_name, cell_id):
    """Execute a put action. Returns (success, failure_phase)."""
    dt = env.model.opt.timestep
    cyl_idx = int(cyl_name.split("_")[1])
    radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
    cx, cy = grid.cells[cell_id]["center"]
    cyl_z      = grid.table_height + half_height + 0.002
    target_pos = np.array([cx, cy, cyl_z])
    half_size  = np.array([radius, radius, half_height])

    candidates = grasp_planner.generate_candidates(target_pos, half_size)
    cand = _pick_candidate(candidates)
    if cand is None:
        return False, "no_place_candidate"

    put_quat     = cand.grasp_quat
    R_put        = quat_to_rotmat(put_quat)
    rel_pos      = env._attached["rel_pos"]
    place_ee_pos = target_pos - R_put @ rel_pos
    approach_pos = place_ee_pos + np.array([0.0, 0.0, grasp_planner.approach_dist])

    if not env.is_collision_free(env.data.qpos[:7]):
        return False, "start_in_collision"

    # Approach above cell
    path = planner.plan_to_pose(approach_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        return False, "put_approach_plan_fail"
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    # Place descent
    path = planner.plan_to_pose(place_ee_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS_FINE)
    if path is None:
        return False, "put_descent_plan_fail"
    env.execute_path(path, planner, step_size=0.005)
    _wait_idle(env)

    _settle(env)
    env.detach_object()
    env.controller.open_gripper()
    env.rest(1.5)

    # Retreat to approach height
    env.add_collision_exception(cyl_name)
    retreat = planner.plan_to_pose(approach_pos, put_quat,
                                   dt=dt, max_iterations=_RRT_ITERS_FINE)
    env.remove_collision_exception(cyl_name)
    if retreat is not None:
        env.execute_path(retreat, planner, step_size=0.01)
        _wait_idle(env)

    return True, "ok"


# ── PDDL parsing ───────────────────────────────────────────────────────────

def _extract_init_section(pddl_text):
    m = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', pddl_text, re.DOTALL | re.IGNORECASE)
    if m is None:
        m = re.search(r'\(:init\s+(.*?)\)', pddl_text, re.DOTALL | re.IGNORECASE)
    if m is None:
        raise ValueError("Could not find (:init ...) section")
    return m.group(1)


def _parse_plan(plan_text):
    actions = []
    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        inner = line.strip("()")
        parts = inner.split()
        if parts[0] == "pick":
            actions.append(("pick", parts[1]))
        elif parts[0] == "put":
            actions.append(("put", parts[1], parts[2]))
    return actions


# ── Single-config runner ───────────────────────────────────────────────────

def _run_config(problem_path: Path, plan_path: Path, strategy: str, verbose: bool):
    """Execute one plan; return list of (action_type, success, phase) dicts."""
    problem_text = problem_path.read_text()
    init_section = _extract_init_section(problem_text)
    actions      = _parse_plan(plan_path.read_text())

    env, grid = _make_headless_env()
    planner   = _make_planner(env, strategy)
    sm        = StateManager(grid, env)
    gp        = GraspPlanner(table_z=grid.table_height)

    sm.init_from_pddl_state(init_section)
    env.data.qpos[:8] = _HOME_QPOS
    env.data.ctrl[:8] = _HOME_CTRL
    env.reset_velocities()
    mujoco.mj_forward(env.model, env.data)

    results = []
    holding: str | None = None  # track gripper state

    for action in actions:
        action_type = action[0]
        t0 = time.perf_counter()

        if action_type == "pick":
            cyl = action[1]
            if holding is not None:
                if verbose:
                    print(f"    SKIP pick {cyl}  (already holding {holding})")
                results.append({"type": "pick", "success": False, "phase": "skipped_holding"})
                continue
            success, phase = _execute_pick(env, planner, gp, cyl, sm)
            elapsed = time.perf_counter() - t0
            if success:
                holding = cyl
            if verbose:
                tag = "PASS" if success else f"FAIL({phase})"
                print(f"    pick {cyl:12s}  {tag}  ({elapsed:.1f}s)")
            results.append({"type": "pick", "success": success, "phase": phase})

        elif action_type == "put":
            cyl, cell = action[1], action[2]
            if holding != cyl:
                if verbose:
                    print(f"    SKIP put  {cyl} → {cell}  (not holding)")
                results.append({"type": "put", "success": False, "phase": "skipped_not_holding"})
                continue
            success, phase = _execute_put(env, planner, gp, grid, cyl, cell)
            elapsed = time.perf_counter() - t0
            if success:
                holding = None
            if verbose:
                tag = "PASS" if success else f"FAIL({phase})"
                print(f"    put  {cyl} → {cell:10s}  {tag}  ({elapsed:.1f}s)")
            results.append({"type": "put", "success": success, "phase": phase})

    return results


# ── Summary reporting ──────────────────────────────────────────────────────

def _print_summary(all_results: list[dict], strategy: str):
    pick_results = [r for r in all_results if r["type"] == "pick"]
    put_results  = [r for r in all_results if r["type"] == "put"]

    def _stats(results):
        n      = len(results)
        passed = sum(r["success"] for r in results)
        if n == 0:
            return 0, 0, {}
        failures: dict[str, int] = {}
        for r in results:
            if not r["success"]:
                failures[r["phase"]] = failures.get(r["phase"], 0) + 1
        return passed, n, failures

    p_pass, p_n, p_fail = _stats(pick_results)
    u_pass, u_n, u_fail = _stats(put_results)
    total_pass = p_pass + u_pass
    total_n    = p_n + u_n

    print(f"\n{'─'*60}")
    print(f"Strategy: {strategy}")
    print(f"  pick : {p_pass}/{p_n} ({100*p_pass/p_n:.0f}%)  failures: {dict(p_fail)}")
    print(f"  put  : {u_pass}/{u_n} ({100*u_pass/u_n:.0f}%)  failures: {dict(u_fail)}")
    pct = 100 * total_pass / total_n if total_n else 0
    print(f"  total: {total_pass}/{total_n} ({pct:.0f}%)")
    print(f"{'─'*60}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Headless execution benchmark")
    parser.add_argument("data_dir", nargs="?", default="/tmp/tabletop_test/train",
                        help="Directory containing *.pddl and *.pddl.plan files")
    parser.add_argument("--strategy", default="baseline",
                        choices=[*RobustPlanner.STRATEGIES, "all"],
                        help="Planning strategy (default: baseline)")
    parser.add_argument("--max_configs", type=int, default=0,
                        help="Limit number of configs tested (0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-action results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    plan_files = sorted(data_dir.glob("*.pddl.plan"))
    if not plan_files:
        print(f"No *.pddl.plan files found in {data_dir}")
        return

    if args.max_configs > 0:
        plan_files = plan_files[: args.max_configs]

    strategies = list(RobustPlanner.STRATEGIES) if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        all_results = []
        t_start = time.perf_counter()
        print(f"\n=== Strategy: {strategy} — {len(plan_files)} configs ===")

        for plan_path in plan_files:
            stem = plan_path.name[: -len(".pddl.plan")]  # e.g. config_1
            prob_path = data_dir / f"{stem}.pddl"
            if not prob_path.exists():
                print(f"  Missing problem file for {plan_path.name}, skipping.")
                continue

            if args.verbose:
                print(f"\n  {stem}:")

            try:
                results = _run_config(prob_path, plan_path, strategy, args.verbose)
                all_results.extend(results)
            except Exception as exc:
                print(f"  ERROR in {stem}: {exc}")

        elapsed = time.perf_counter() - t_start
        _print_summary(all_results, strategy)
        print(f"  wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
