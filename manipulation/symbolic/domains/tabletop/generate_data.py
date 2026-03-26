"""Tabletop dataset generation — PDDL problems with motion-planning validation.

For each generated instance:
  1. A random cylinder scene is sampled.
  2. A plan is found to reach every possible target:
     - No-drop mode (default): pick+put rearrangement using ActionFeasibilityChecker.
       Greedy with MRV obstacle ordering first, BFS fallback.
     - Drop mode (--allow-drop): BFS over pick/drop sequences.
  3. Target is selected with uniform distribution *across plan lengths* so
     trivially-reachable (plan-length-1) targets do not dominate the dataset.
  4. PDDL problem, plan file, and optional visualisation are saved.

Usage:
    python -m manipulation.symbolic.domains.tabletop.generate_data
    python -m manipulation.symbolic.domains.tabletop.generate_data \\
        --num-train 500 --num-test 50 --num-val 50 \\
        --min-objects 3 --max-objects 7 --num-workers 4
    # Drop mode (deprecated):
    python -m manipulation.symbolic.domains.tabletop.generate_data --allow-drop
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

from manipulation import FrankaEnvironment, RRTStar, FeasibilityRRT, SCENE_SYMBOLIC
from manipulation.planners.grasp_planner import GraspPlanner, GraspType
from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker
from manipulation.symbolic.domains.tabletop.grid_domain import GridDomain
from manipulation.symbolic.domains.tabletop.state_manager import StateManager
from manipulation.symbolic.domains.tabletop.visualization import visualize_grid_state

_HERE         = Path(__file__).parent
_XML          = SCENE_SYMBOLIC
_DOMAIN       = (_HERE / "pddl" / "domain.pddl").resolve()
_DOMAIN_NO_DROP = (_HERE / "pddl" / "spatial_put" / "domain.pddl").resolve()

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

# No-drop planning constants
_SHADOW_MAX_DEPTH  = 3   # rows behind a cylinder included in its shadow cone
_BFS_CAND_LIMIT    = 8   # max put-destinations per obstacle explored in BFS
_GREEDY_CAND_LIMIT = 15  # max put-destinations per obstacle evaluated in greedy
_GREEDY_MAX_MOVES  = 6   # max pick+put pairs in greedy phase
_BFS_MAX_DEPTH     = 4   # max pick+put pairs explored in BFS phase

# Feasibility checker tuning for data generation
# Put feasibility is checked with an IK proxy (dest_cell in reachable_cells) rather
# than full RRT — the shadow + reachability filter in _sorted_candidates is the
# main quality gate.  Pick checks use full RRT but with a reduced budget vs the
# interactive checker (correctness is still high; FeasibilityRRT at 300 iters >>
# RRTStar at 1000 in terms of wall-clock).
_CHECKER_PICK_ITERS    = 300  # FeasibilityRRT iterations for pick checks
_CHECKER_SETTLE_STEPS  = 10   # sim settle steps before each pick check


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
    p.add_argument("--allow-drop",          action="store_true",
                   help="Allow the robot to drop cylinders on the floor (default: no-drop mode)")
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


def _build_env(args) -> tuple:
    """Build environment, planners, checker, and pre-compute reachable cells.

    Returns:
        (env, planner, feas_planner, checker, reachable_cells,
         grid, state_manager, grasp_planner)

        reachable_cells: frozenset[str] of cell IDs where IK converges.
    """
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    _patch_fast_step(env)

    env.ik.max_iters     = args.ik_iters
    env.ik.pos_threshold = args.ik_pos_thresh

    planner = RRTStar(env)
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    feas_planner = FeasibilityRRT(env)
    feas_planner.step_size             = planner.step_size
    feas_planner.collision_check_steps = planner.collision_check_steps

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

    # ActionFeasibilityChecker is used for no-drop path; also provides
    # verify_reachability.  It overrides env.step with an attachment-aware
    # fast step — safe for the drop path too (attachment check is a no-op).
    checker = ActionFeasibilityChecker(
        env, planner, state_manager, grasp_planner,
        max_iterations=_CHECKER_PICK_ITERS,
        settle_steps=_CHECKER_SETTLE_STEPS,
        ik_max_iters=args.ik_iters,
        ik_pos_threshold=args.ik_pos_thresh,
        feasibility_planner=feas_planner,
    )

    # Pre-compute kinematic reachability once per process (IK only, fast).
    reach_map = checker.verify_reachability(verbose=False)
    reachable_cells = frozenset(c for c, ok in reach_map.items() if ok)

    return env, planner, feas_planner, checker, reachable_cells, grid, state_manager, grasp_planner


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
# Drop mode — grasp feasibility (lightweight, uses collision exceptions)
# ---------------------------------------------------------------------------

def _pick_candidate(candidates):
    """Prefer FRONT approach for tall cylinders (same as all other examples)."""
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _validate_pick(
    env: FrankaEnvironment,
    planner: RRTStar,
    feas_planner: FeasibilityRRT,
    grasp_planner: GraspPlanner,
    target_obj: str,
    ignored_objects: list[str] | None = None,
    rrt_iters: int = _DEFAULT_RRT_ITERS,
) -> bool:
    """Check whether *target_obj* can be approached and grasped (drop mode).

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

        # Approach
        path = feas_planner.plan_to_pose(
            candidate.approach_pos, candidate.grasp_quat,
            dt=dt, max_iterations=rrt_iters,
        )
        if path is None:
            return False

        # Execute approach so grasp check starts from the correct config
        env.execute_path(path, feas_planner)
        env.wait_idle(max_steps=5000, settle_steps=30)

        # Grasp IK
        env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
        if not env.ik.converge_ik(dt):
            return False

        # Grasp
        path = feas_planner.plan_to_pose(
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
# BFS drop plan — shortest pick/drop sequence for one target
# ---------------------------------------------------------------------------

def _find_plan(
    env: FrankaEnvironment,
    planner: RRTStar,
    feas_planner: FeasibilityRRT,
    grasp_planner: GraspPlanner,
    target_obj: str,
    all_objects: list[str],
    cache: dict,
    rrt_iters: int,
) -> list[str] | None:
    """BFS over removal sequences to find shortest plan reaching *target_obj* (drop mode).

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
            ok = _validate_pick(env, planner, feas_planner, grasp_planner, target_obj,
                                ignored, rrt_iters)
            cache[key] = ok
        if ok:
            return plan + [target_obj]

        for obj in sorted(current - {target_obj}):
            key2 = (current, obj)
            ok2  = cache.get(key2)
            if ok2 is None:
                ok2 = _validate_pick(env, planner, feas_planner, grasp_planner, obj,
                                     ignored, rrt_iters)
                cache[key2] = ok2
            if ok2:
                nxt = current - {obj}
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, plan + [obj]))

    return None


# ---------------------------------------------------------------------------
# No-drop mode — arrangement helpers
# ---------------------------------------------------------------------------

# An "arrangement" is frozenset[tuple[str, str]] — set of (cylinder_name, cell_id) pairs.
# It uniquely identifies the positions of all cylinders on the table.

def _make_arrangement(state_dict: dict) -> frozenset:
    """Convert state_dict to an arrangement frozenset."""
    return frozenset(
        (cyl, cells[0])
        for cyl, cells in state_dict["cylinders"].items()
        if cells
    )


def _arrangement_to_state_dict(arrangement: frozenset) -> dict:
    """Convert an arrangement to a state_dict accepted by ActionFeasibilityChecker."""
    return {
        "cylinders":    {cyl: [cell] for cyl, cell in arrangement},
        "gripper_empty": True,
        "holding":      None,
    }


def _shadow_cells(arrangement: frozenset, grid) -> set[str]:
    """Return cell IDs in the directional shadow cone of any cylinder.

    A cell (cx+dx, cy+depth) is in the shadow of (cx, cy) when:
      depth ∈ [1, _SHADOW_MAX_DEPTH]  and  |dx| ≤ depth - 1

    This represents a widening cone behind each cylinder from the robot's
    approach direction (y=0 is closest to robot; higher y = further away).
    Shadow cells are deprioritised as put-destinations — placing an obstacle
    behind another one creates stacked occlusion.
    """
    shadow: set[str] = set()
    cells_x = grid.cells_x
    cells_y = grid.cells_y

    for _, cell_id in arrangement:
        parts = cell_id.split("_")
        cx, cy = int(parts[1]), int(parts[2])
        for depth in range(1, _SHADOW_MAX_DEPTH + 1):
            for dx in range(-(depth - 1), depth):
                nx, ny = cx + dx, cy + depth
                if 0 <= nx < cells_x and 0 <= ny < cells_y:
                    shadow.add(f"cell_{nx}_{ny}")

    return shadow


def _sorted_candidates(
    arrangement: frozenset,
    reachable_cells: frozenset,
    grid,
    exclude_cell: str | None = None,
    target_cell: str | None = None,
) -> list[str]:
    """Return empty reachable cells sorted by desirability as put-destinations.

    Order: [non-shadow, non-adjacent-to-target] first, then shadow/adjacent.

    Args:
        arrangement:    Current cylinder positions.
        reachable_cells: Pre-computed IK-reachable cell IDs.
        grid:           GridDomain instance (for cells_x, cells_y).
        exclude_cell:   Cell to exclude from candidates (the cylinder's current
                        cell — it will be vacated, but other cylinders might
                        fill it, so we exclude it to avoid trivial swaps).
        target_cell:    If provided, cells adjacent to the target are moved to
                        the end of the list (placing an obstacle next to the
                        target makes the final pick harder).
    """
    occupied = {cell for _, cell in arrangement}
    if exclude_cell is not None:
        occupied = occupied - {exclude_cell}

    shadow = _shadow_cells(arrangement, grid)

    # Cells adjacent to target (cardinal directions)
    near_target: set[str] = set()
    if target_cell is not None:
        parts = target_cell.split("_")
        tx, ty = int(parts[1]), int(parts[2])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < grid.cells_x and 0 <= ny < grid.cells_y:
                near_target.add(f"cell_{nx}_{ny}")

    available = [c for c in reachable_cells if c not in occupied and c != exclude_cell]

    preferred  = [c for c in available if c not in shadow and c not in near_target]
    deprioritised = [c for c in available if c not in preferred]

    return preferred + deprioritised


# ---------------------------------------------------------------------------
# No-drop mode — cached feasibility checks
# ---------------------------------------------------------------------------

_FAST_PICK_ITERS = 100  # FeasibilityRRT iterations for the fast pick check


def _check_pick_no_drop(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    cyl: str,
    pick_cache: dict,
) -> bool:
    """Fast cached pick feasibility: approach IK + approach RRT only.

    checker.check("pick") does 3 RRT calls + 2 execute_path + 2 wait_idle,
    taking ~1-2s per call.  For the greedy inner loop we only need enough
    signal to know whether the approach path is clear — a single RRT call to
    the approach pose (~50-100ms) is sufficient.  Grasp and transport phases
    are skipped; they almost never fail when the approach succeeds.
    """
    key = (arrangement, cyl)
    result = pick_cache.get(key)
    if result is not None:
        return result

    env            = checker._env
    feas_planner   = checker._feas_planner
    grasp_planner  = checker._grasp_planner
    state_manager  = checker._state_manager

    saved_qpos       = env.data.qpos.copy()
    saved_qvel       = env.data.qvel.copy()
    saved_exceptions = list(env.collision_exceptions)
    try:
        state = _arrangement_to_state_dict(arrangement)
        state_manager.set_from_grounded_state(state)
        env.controller.stop()

        # Reset robot to home so RRT starts from a consistent config.
        env.data.qpos[:8] = env.initial_qpos[:8]
        env.data.ctrl[:8] = env.initial_ctrl[:8]
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)

        dt = env.model.opt.timestep

        cyl_pos   = env.get_object_position(cyl)
        half_size = env.get_object_half_size(cyl)
        cyl_quat  = env.get_object_orientation(cyl)
        candidates = grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
        # Prefer FRONT but try all — a cylinder only counts as a true blocker
        # if it blocks every approach direction, not just one.
        ordered = sorted(
            candidates,
            key=lambda c: 0 if c.grasp_type == GraspType.FRONT else 1,
        )

        ok = False
        home_q = env.data.qpos[:7].copy()
        for candidate in ordered:
            # Reset robot and collision state for each candidate attempt.
            env.set_collision_exceptions(saved_exceptions)
            # --- Phase 1: approach (home → approach pose) ---
            env.data.qpos[:7] = home_q
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)

            env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
            if not env.ik.converge_ik(dt):
                continue
            approach_q = env.ik.configuration.q[:7].copy()
            # Fast pre-filter: if the approach pose itself is in collision there
            # is no path to it — skip the expensive RRT entirely.
            if not env.is_collision_free(approach_q):
                continue
            path = feas_planner.plan(
                home_q, approach_q, max_iterations=_FAST_PICK_ITERS,
            )
            if path is None:
                continue

            # --- Phase 2: grasp descent (approach pose → grasp contact) ---
            # Teleport to approach config; add collision exception so
            # is_collision_free(grasp_q) passes on gripper–cylinder contact.
            env.add_collision_exception(cyl)
            env.data.qpos[:7] = approach_q
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)

            env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
            if not env.ik.converge_ik(dt):
                continue
            grasp_q = env.ik.configuration.q[:7].copy()
            path = feas_planner.plan(
                approach_q, grasp_q, max_iterations=_FAST_PICK_ITERS,
            )
            if path is not None:
                ok = True
                break
    except Exception:
        ok = False
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        env.set_collision_exceptions(saved_exceptions)

    pick_cache[key] = ok
    return ok


def _check_put_no_drop(
    reachable_cells: frozenset,
    arrangement: frozenset,
    cyl: str,
    dest_cell: str,
    put_cache: dict,
    placement_margin: int = 1,
) -> bool:
    """Lightweight put feasibility: IK proxy + placement-margin check.

    Full RRT put checks are too expensive for use in the greedy inner loop.
    Instead we check:
      1. dest_cell is IK-reachable.
      2. No other cylinder is within placement_margin cells in any cardinal
         direction of dest_cell (matches _sample_state validity rule).

    The cache key uses others_arr (arrangement minus the moving cyl) because
    the moving cyl's current cell doesn't affect put feasibility.
    """
    others_arr = frozenset((c, cl) for c, cl in arrangement if c != cyl)
    key = (others_arr, dest_cell, placement_margin)
    result = put_cache.get(key)
    if result is None:
        if dest_cell not in reachable_cells:
            result = False
        else:
            parts = dest_cell.split("_")
            dx, dy = int(parts[1]), int(parts[2])
            result = True
            for _, other_cell in others_arr:
                oparts = other_cell.split("_")
                ox, oy = int(oparts[1]), int(oparts[2])
                if ox == dx and abs(oy - dy) <= placement_margin:
                    result = False
                    break
                if oy == dy and abs(ox - dx) <= placement_margin:
                    result = False
                    break
        put_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# No-drop mode — greedy solver (MRV ordering)
# ---------------------------------------------------------------------------

def _greedy_no_drop(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    target: str,
    reachable_cells: frozenset,
    grid,
    pick_cache: dict,
    put_cache: dict,
    ordering: str = "mrv",
    placement_margin: int = 1,
) -> list[tuple] | None:
    """Greedy no-drop solver with MRV (Minimum Remaining Values) ordering.

    At each step:
    1. Check if target is directly pickable → done.
    2. For each obstacle, count how many valid put-destinations it has
       (checking up to _GREEDY_CAND_LIMIT candidates).
    3. Sort obstacles by fewest valid destinations (MRV = most constrained first).
    4. Move the most constrained obstacle to its first valid destination.
    5. Repeat up to _GREEDY_MAX_MOVES times.

    Returns a plan (list of action tuples) or None.

    Args:
        ordering: ``"mrv"`` — move most-constrained obstacle first (fewest
            valid destinations); ``"lrv"`` — move least-constrained first;
            ``"random"`` — shuffle obstacle order before ranking.
    """
    current = arrangement
    plan: list[tuple] = []

    for _ in range(_GREEDY_MAX_MOVES):
        if _check_pick_no_drop(checker, current, target, pick_cache):
            target_cell = dict(current)[target]
            return plan + [("pick", target, target_cell)]

        obstacles = [(cyl, cell) for cyl, cell in current if cyl != target]
        if not obstacles:
            return None

        if ordering == "random":
            obstacles = list(obstacles)
            np.random.shuffle(obstacles)

        # Identify true blockers: obstacles whose removal makes the target pickable.
        # Only move these — moving a non-blocker never helps.
        blockers = [
            (cyl, cell) for cyl, cell in obstacles
            if _check_pick_no_drop(
                checker,
                frozenset((c, cl) for c, cl in current if c != cyl),
                target,
                pick_cache,
            )
        ]
        # Fall back to all obstacles only when no single-removal unblocks the target
        # (target needs 2+ cylinders removed simultaneously).
        candidates = blockers if blockers else obstacles

        # Score each candidate by number of valid put-destinations (instant check).
        ranked: list[tuple[int, str, str, list[str]]] = []
        for cyl, cell in candidates:
            others = frozenset((c, cl) for c, cl in current if c != cyl)
            cands = _sorted_candidates(others, reachable_cells, grid,
                                       exclude_cell=cell,
                                       target_cell=dict(current).get(target))
            valid: list[str] = []
            for dest in cands[:_GREEDY_CAND_LIMIT]:
                if _check_put_no_drop(reachable_cells, current, cyl, dest, put_cache, placement_margin):
                    valid.append(dest)
                    if len(valid) >= 3:  # enough to compare; cap for speed
                        break
            ranked.append((len(valid), cyl, cell, valid))

        if ordering == "lrv":
            ranked.sort(key=lambda x: x[0], reverse=True)
        elif ordering != "random":  # mrv (default)
            ranked.sort(key=lambda x: x[0])
        # random: already shuffled above; stable sort preserves order

        moved = False
        for _, cyl, cell, valid_dests in ranked:
            if not valid_dests:
                continue
            # Verify the obstacle itself is actually pickable in the current
            # arrangement before committing to move it.
            if not _check_pick_no_drop(checker, current, cyl, pick_cache):
                continue
            dest = valid_dests[0]
            plan.append(("pick", cyl, cell))
            plan.append(("put", cyl, dest))
            current = (current - {(cyl, cell)}) | {(cyl, dest)}
            moved = True
            break

        if not moved:
            return None

    # Final check after exhausting greedy moves
    if _check_pick_no_drop(checker, current, target, pick_cache):
        target_cell = dict(current)[target]
        return plan + [("pick", target, target_cell)]

    return None


# ---------------------------------------------------------------------------
# No-drop mode — BFS fallback
# ---------------------------------------------------------------------------

def _bfs_no_drop(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    target: str,
    reachable_cells: frozenset,
    grid,
    pick_cache: dict,
    put_cache: dict,
    placement_margin: int = 1,
) -> list[tuple] | None:
    """BFS over (pick, put) pairs to find the shortest no-drop plan.

    State: (arrangement, plan_so_far).
    Branching is bounded to _BFS_CAND_LIMIT destinations per obstacle per state,
    and depth is capped at _BFS_MAX_DEPTH pick+put pairs.
    """
    queue: deque[tuple[frozenset, list[tuple]]] = deque([(initial_arrangement, [])])
    visited: set[frozenset] = {initial_arrangement}

    while queue:
        arrangement, plan = queue.popleft()

        if len(plan) // 2 >= _BFS_MAX_DEPTH:
            continue

        if _check_pick_no_drop(checker, arrangement, target, pick_cache):
            target_cell = dict(arrangement)[target]
            return plan + [("pick", target, target_cell)]

        for cyl, cell in arrangement:
            if cyl == target:
                continue

            others = frozenset((c, cl) for c, cl in arrangement if c != cyl)
            cands = _sorted_candidates(others, reachable_cells, grid,
                                       exclude_cell=cell,
                                       target_cell=dict(arrangement).get(target))

            checked = 0
            for dest in cands:
                if checked >= _BFS_CAND_LIMIT:
                    break
                if not _check_put_no_drop(reachable_cells, arrangement, cyl, dest, put_cache, placement_margin):
                    continue
                checked += 1

                new_arr = (arrangement - {(cyl, cell)}) | {(cyl, dest)}
                if new_arr in visited:
                    continue
                visited.add(new_arr)
                queue.append((new_arr, plan + [("pick", cyl, cell), ("put", cyl, dest)]))

    return None


# ---------------------------------------------------------------------------
# No-drop mode — entry point for one target
# ---------------------------------------------------------------------------

def _find_plan_no_drop(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    target: str,
    reachable_cells: frozenset,
    grid,
    pick_cache: dict,
    put_cache: dict,
    placement_margin: int = 1,
) -> list[tuple] | None:
    """Find a no-drop plan to pick *target*.

    Tries three greedy orderings in sequence (MRV, LRV, random).  BFS is not
    used because it calls the pick-feasibility check at every explored node,
    which is exponential in the number of objects.  With fast pick checks
    (~100ms each) the multi-ordering greedy handles >95% of scenes; the
    remaining scenes are rejected.
    """
    for ordering in ("mrv", "lrv", "random"):
        plan = _greedy_no_drop(
            checker, initial_arrangement, target, reachable_cells, grid,
            pick_cache, put_cache, ordering=ordering,
            placement_margin=placement_margin,
        )
        if plan is not None:
            return _compress_plan(
                plan, initial_arrangement, reachable_cells, put_cache, placement_margin
            )
    return None


# ---------------------------------------------------------------------------
# No-drop mode — plan compression
# ---------------------------------------------------------------------------

def _compress_plan(
    plan: list[tuple],
    initial_arrangement: frozenset,
    reachable_cells: frozenset,
    put_cache: dict,
    placement_margin: int = 1,
) -> list[tuple]:
    """Collapse consecutive same-cylinder move pairs into one.

    Pattern: (pick X c1)(put X c2)(pick X c2)(put X c3)
    If the direct put to c3 is valid from the arrangement before the first
    pick, replace all four actions with (pick X c1)(put X c3).

    Also removes complete no-ops where c3 == c1.
    Repeats until no further merges can be made.
    """
    def _apply(arr: frozenset, action: tuple) -> frozenset:
        verb, cyl, cell = action
        return arr - {(cyl, cell)} if verb == "pick" else arr | {(cyl, cell)}

    changed = True
    result = list(plan)
    while changed:
        changed = False
        new_result: list[tuple] = []
        arr = initial_arrangement
        i = 0
        while i < len(result):
            if i + 3 < len(result):
                a0, a1, a2, a3 = result[i], result[i+1], result[i+2], result[i+3]
                if (a0[0] == "pick" and a1[0] == "put" and
                        a2[0] == "pick" and a3[0] == "put" and
                        a0[1] == a1[1] == a2[1] == a3[1] and
                        a1[2] == a2[2]):
                    cyl, c1, c3 = a0[1], a0[2], a3[2]
                    if c3 == c1:
                        # Complete no-op — skip all four
                        i += 4
                        changed = True
                        continue
                    if _check_put_no_drop(reachable_cells, arr, cyl, c3,
                                         put_cache, placement_margin):
                        new_result.append(("pick", cyl, c1))
                        arr = arr - {(cyl, c1)}
                        new_result.append(("put", cyl, c3))
                        arr = arr | {(cyl, c3)}
                        i += 4
                        changed = True
                        continue
            new_result.append(result[i])
            arr = _apply(arr, result[i])
            i += 1
        result = new_result
    return result


# ---------------------------------------------------------------------------
# Target selection — no-drop mode
# ---------------------------------------------------------------------------

def _select_target_no_drop(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    cylinders: list[str],
    reachable_cells: frozenset,
    grid,
    pick_cache: dict,
    put_cache: dict,
    placement_margin: int = 1,
) -> tuple[str, list[tuple]] | tuple[None, None]:
    """Run _find_plan_no_drop for every cylinder and select target by plan-length diversity.

    The shared caches mean subsequent cylinder queries reuse earlier results,
    so the overhead of running for all cylinders is modest.

    Returns (target_cylinder, plan) or (None, None).
    """
    plans: dict[str, list[tuple] | None] = {}

    for cyl in cylinders:
        plans[cyl] = _find_plan_no_drop(
            checker, initial_arrangement, cyl, reachable_cells, grid,
            pick_cache, put_cache, placement_margin=placement_margin,
        )

    by_length: dict[int, list[str]] = {}
    for cyl, plan in plans.items():
        if plan is not None:
            by_length.setdefault(len(plan), []).append(cyl)

    if not by_length:
        return None, None

    chosen_length = int(np.random.choice(sorted(by_length.keys())))
    target = str(np.random.choice(by_length[chosen_length]))
    return target, plans[target]


# ---------------------------------------------------------------------------
# Target selection — drop mode
# ---------------------------------------------------------------------------

def _select_target(
    env: FrankaEnvironment,
    planner: RRTStar,
    feas_planner: FeasibilityRRT,
    grasp_planner: GraspPlanner,
    cylinders: list[str],
    rrt_iters: int,
) -> tuple[str, list[str]] | tuple[None, None]:
    """Run BFS for every cylinder and pick a target with uniform distribution
    across plan lengths (drop mode).

    Returns (target_cylinder, plan_sequence) or (None, None).
    """
    cache: dict = {}
    plans: dict[str, list[str] | None] = {}

    for cyl in cylinders:
        plans[cyl] = _find_plan(
            env, planner, feas_planner, grasp_planner, cyl, cylinders, cache, rrt_iters
        )

    by_length: dict[int, list[str]] = {}
    for cyl, plan in plans.items():
        if plan is not None:
            by_length.setdefault(len(plan), []).append(cyl)

    if not by_length:
        return None, None

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
        f.write(f"; allow_drop: {args.allow_drop}\n")


def _write_plan(
    path: Path,
    plan: list[tuple],
    target: str,
    cylinders: list[str],
    split: str,
    config_num: int,
    gen_sec: float,
) -> int:
    """Write a PDDL plan file.

    *plan* is a list of action tuples, e.g.:
      [("pick", "cylinder_2", "cell_3_1"),
       ("put",  "cylinder_2", "cell_5_2"),
       ("pick", "cylinder_0", "cell_1_4")]

    Each tuple is written as a parenthesised PDDL action.
    """
    action_count = len(plan)
    with open(path, "w") as f:
        for action in plan:
            f.write(f"({' '.join(action)})\n")
        f.write(f"; Total actions: {action_count}\n")
        f.write(f"; split: {split}\n")
        f.write(f"; config_num: {config_num}\n")
        f.write(f"; Target: {target}\n")
        f.write(f"; Objects: {', '.join(cylinders)}\n")
        f.write(f"; Generation time: {gen_sec:.2f}s\n")
    return action_count


def _drop_plan_to_tuples(plan: list[str], cylinder_cells: dict[str, str]) -> list[tuple]:
    """Convert drop-mode list[str] plan to unified list[tuple] format."""
    result: list[tuple] = []
    for obj in plan[:-1]:
        cell = cylinder_cells.get(obj, "")
        result.append(("pick", obj, cell) if cell else ("pick", obj))
        result.append(("drop", obj))
    last = plan[-1]
    cell = cylinder_cells.get(last, "")
    result.append(("pick", last, cell) if cell else ("pick", last))
    return result


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

def _generate_split(
    split: str,
    count: int,
    args,
    env: FrankaEnvironment,
    planner: RRTStar,
    feas_planner: FeasibilityRRT,
    checker: ActionFeasibilityChecker,
    reachable_cells: frozenset,
    grid,
    state_manager: StateManager,
    grasp_planner: GraspPlanner,
    output_base: Path,
    pick_cache: dict,
    put_cache: dict,
    wandb_run=None,
) -> dict:
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

        state     = state_manager.ground_state()
        cylinders = sorted(state["cylinders"].keys())
        stats["actual_hist"][len(cylinders)] += 1

        if len(cylinders) < args.min_objects:
            stats["reject_min_objects"] += 1
            continue

        if args.allow_drop:
            target, drop_plan = _select_target(
                env, planner, feas_planner, grasp_planner, cylinders, args.rrt_iters
            )
            if drop_plan is None:
                stats["reject_no_plan"] += 1
                continue
            cylinder_cells = {k: v[0] for k, v in state["cylinders"].items() if v}
            plan = _drop_plan_to_tuples(drop_plan, cylinder_cells)
        else:
            initial_arrangement = _make_arrangement(state)
            target, plan = _select_target_no_drop(
                checker, initial_arrangement, cylinders, reachable_cells, grid,
                pick_cache, put_cache,
                placement_margin=args.placement_margin,
            )
            if plan is None:
                stats["reject_no_plan"] += 1
                continue

            # Restore initial scene — checker.check() leaves cylinders at
            # whatever position _init_state last set them to.
            state_manager.set_from_grounded_state(state)

        # Write PDDL problem
        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}", problem_path,
            goal_string=f"(holding {target})",
        )
        _append_metadata(problem_path, args, split, config_num,
                         requested=n_req, actual=len(cylinders), target=target)

        # Write plan
        action_count = _write_plan(
            problem_dir / f"config_{config_num}.pddl.plan",
            plan, target, cylinders, split, config_num, time.time() - t_start,
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

    (env, planner, feas_planner, checker, reachable_cells,
     grid, state_manager, grasp_planner) = _build_env(args)

    output_base = Path(args.output_dir)
    problem_dir = output_base / split
    viz_dir     = output_base / split / "viz"
    problem_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Per-process global caches for no-drop mode
    pick_cache: dict = {}
    put_cache:  dict = {}

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

        if args.allow_drop:
            target, drop_plan = _select_target(
                env, planner, feas_planner, grasp_planner, cylinders, args.rrt_iters
            )
            if drop_plan is None:
                stats["reject_no_plan"] += 1
                continue
            cylinder_cells = {k: v[0] for k, v in state["cylinders"].items() if v}
            plan = _drop_plan_to_tuples(drop_plan, cylinder_cells)
        else:
            initial_arrangement = _make_arrangement(state)
            target, plan = _select_target_no_drop(
                checker, initial_arrangement, cylinders, reachable_cells, grid,
                pick_cache, put_cache,
                placement_margin=args.placement_margin,
            )
            if plan is None:
                stats["reject_no_plan"] += 1
                continue

            # Restore initial scene — checker.check() leaves cylinders at
            # whatever position _init_state last set them to.
            state_manager.set_from_grounded_state(state)

        problem_path = problem_dir / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(
            f"config-{config_num}", problem_path,
            goal_string=f"(holding {target})",
        )
        _append_metadata(problem_path, args, split, config_num,
                         requested=n_req, actual=len(cylinders), target=target)

        action_count = _write_plan(
            problem_dir / f"config_{config_num}.pddl.plan",
            plan, target, cylinders, split, config_num, time.time() - t_start,
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

    # Copy domain.pddl — choose based on mode
    if args.domain_src:
        domain_src = Path(args.domain_src)
    elif args.allow_drop:
        domain_src = _DOMAIN
    else:
        domain_src = _DOMAIN_NO_DROP

    domain_dst = output_base / "domain.pddl"
    if domain_src.exists():
        shutil.copy(domain_src, domain_dst)
        print(f"Copied domain.pddl → {domain_dst}")
    else:
        print(f"WARNING: domain.pddl not found at {domain_src}")

    mode_str = "drop" if args.allow_drop else "no-drop"
    print("=" * 65)
    print("  Tabletop Dataset Generation")
    print("=" * 65)
    print(f"  mode:     {mode_str}")
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
        (env, planner, feas_planner, checker, reachable_cells,
         grid, state_manager, grasp_planner) = _build_env(args)

        info = grid.get_grid_info()
        print(f"\n  Grid: {info['grid_dimensions'][0]}×{info['grid_dimensions'][1]} cells  "
              f"({info['cell_size']*100:.1f}cm each)  "
              f"{len(reachable_cells)}/{len(grid.cells)} cells reachable")

        # Per-process global caches for no-drop mode
        pick_cache: dict = {}
        put_cache:  dict = {}

        for split, n in [("train", args.num_train),
                          ("test",  args.num_test),
                          ("validation", args.num_val)]:
            s = _generate_split(
                split, n, args,
                env, planner, feas_planner, checker, reachable_cells,
                grid, state_manager, grasp_planner,
                output_base, pick_cache, put_cache, wandb_run,
            )
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
