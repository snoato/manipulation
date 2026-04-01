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
import heapq
from pathlib import Path

import mujoco
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from manipulation import FrankaEnvironment, RRTStar, FeasibilityRRT
from manipulation.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from manipulation.symbolic.domains.tabletop.env_builder import make_symbolic_builder
from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker
from manipulation.symbolic.domains.tabletop.grid_domain import GridDomain
from manipulation.symbolic.domains.tabletop.state_manager import StateManager
from manipulation.symbolic.domains.tabletop.visualization import visualize_grid_state

_HERE         = Path(__file__).parent
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
_BFS_MAX_DEPTH         = 6   # max obstacle pick+put pairs before giving up
_BFS_MAX_PUT_CANDS     = 4   # top-K put destinations tried per obstacle per BFS node

# Feasibility checker tuning for data generation
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
    p.add_argument("--min-plan-len",        type=int,   default=1,
                   help="hard minimum plan length; scenes with shorter plans are rejected")
    p.add_argument("--plan-len-mean",       type=float, default=None,
                   help="target plan length for Gaussian shaping (None = disabled)")
    p.add_argument("--plan-len-std",        type=float, default=2.0,
                   help="std-dev of Gaussian plan-length distribution (used with --plan-len-mean)")
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
    env = make_symbolic_builder().build_env(rate=200.0)
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
        table_geom_name="simple_table_surface",
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
        strict_transport=True,
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


def _put_cell_order(
    checker: ActionFeasibilityChecker,
    target_cell: str,
) -> list[str]:
    """All grid cells sorted by Euclidean distance from *target_cell* (farthest first).

    Prefers cells far from the target to minimise future occlusion of the
    approach corridor.  Uses world-space coordinates rather than a grid-index
    proxy so diagonal distances are represented correctly.

    Proximity-margin and IK feasibility filtering happen in the BFS expansion
    loop via _check_put_no_drop and _check_put_ik respectively.
    """
    grid = checker._state_manager.grid
    tx, ty = grid.cells[target_cell]["center"]

    scored = [
        (np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2), cell_id)
        for cell_id, info in grid.cells.items()
        for cx, cy in (info["center"],)
    ]
    scored.sort(key=lambda x: -x[0])
    return [cell_id for _, cell_id in scored]


# ---------------------------------------------------------------------------
# No-drop mode — cached feasibility checks
# ---------------------------------------------------------------------------

_FAST_PICK_ITERS       = 100  # FeasibilityRRT iterations for the fast pick check
_FAST_PUT_ITERS        = 150  # FeasibilityRRT iterations for the put-from-transport check
_PRECOMPUTE_RRT_ITERS  = 100  # RRT iterations per segment during precompute path validation
_MAX_PICK_NEIGHBORS    = 3    # max cylinders within Manhattan-2 before pick is skipped


def _pick_neighbor_count(arrangement: frozenset, cyl: str, max_dist: int = 2) -> int:
    """Count cylinders within Manhattan distance *max_dist* of *cyl*.

    Pure grid geometry, ~0ms.  Cylinders with high neighbor counts are
    surrounded and almost certainly unpickable; skip them without any
    MuJoCo work.
    """
    arr_dict  = dict(arrangement)
    cyl_cell  = arr_dict.get(cyl)
    if cyl_cell is None:
        return 0
    cx, cy = int(cyl_cell.split("_")[1]), int(cyl_cell.split("_")[2])
    count = 0
    for other_cyl, other_cell in arrangement:
        if other_cyl == cyl:
            continue
        ox, oy = int(other_cell.split("_")[1]), int(other_cell.split("_")[2])
        if abs(ox - cx) + abs(oy - cy) <= max_dist:
            count += 1
    return count


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
    # Fast density pre-filter: too many close neighbours → surrounded → skip.
    if _pick_neighbor_count(arrangement, cyl) >= _MAX_PICK_NEIGHBORS:
        pick_cache[(arrangement, cyl)] = False
        return False

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
    arrangement: frozenset,
    cyl: str,
    dest_cell: str,
    put_cache: dict,
    placement_margin: int = 1,
) -> bool:
    """Placement-margin check: no other cylinder within *placement_margin* cells
    in any cardinal direction of *dest_cell*.

    IK feasibility is handled separately by _check_put_ik; this function is
    purely geometric (no MuJoCo state, ~0 ms).

    The cache key uses others_arr (arrangement minus the moving cyl) because
    the moving cyl's current cell doesn't affect put feasibility.
    """
    others_arr = frozenset((c, cl) for c, cl in arrangement if c != cyl)
    key = (others_arr, dest_cell, placement_margin)
    result = put_cache.get(key)
    if result is None:
        parts = dest_cell.split("_")
        dx, dy = int(parts[1]), int(parts[2])
        result = True
        for _, other_cell in others_arr:
            oparts = other_cell.split("_")
            ox, oy = int(oparts[1]), int(oparts[2])
            if max(abs(ox - dx), abs(oy - dy)) <= placement_margin:
                result = False
                break
        put_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# No-drop mode — fast IK+collision check and 1-on-1 blocker detection
# ---------------------------------------------------------------------------

def _check_approach_ik(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    cyl: str,
    ik_cache: dict,
    front_only: bool = False,
) -> bool:
    """IK convergence + teleport + collision check only (no RRT).

    Returns True if a grasp candidate's approach pose is collision-free.

    front_only=False (default): any approach direction counts — used for
      obstacle movability pre-filtering (can we move this obstacle at all?).
    front_only=True: only the FRONT approach candidate is tested — used for
      blocker detection and the terminal pick check, so that a cylinder that
      blocks the preferred FRONT approach is correctly flagged even if some
      other approach direction would be clear.
    """
    key = (arrangement, cyl, front_only)
    result = ik_cache.get(key)
    if result is not None:
        return result

    env           = checker._env
    grasp_planner = checker._grasp_planner
    state_manager = checker._state_manager

    saved_qpos       = env.data.qpos.copy()
    saved_qvel       = env.data.qvel.copy()
    saved_exceptions = list(env.collision_exceptions)
    try:
        state = _arrangement_to_state_dict(arrangement)
        state_manager.set_from_grounded_state(state)
        env.controller.stop()
        env.data.qpos[:8] = env.initial_qpos[:8]
        env.data.ctrl[:8] = env.initial_ctrl[:8]
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)

        dt        = env.model.opt.timestep
        cyl_pos   = env.get_object_position(cyl)
        half_size = env.get_object_half_size(cyl)
        cyl_quat  = env.get_object_orientation(cyl)
        candidates = grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
        ordered = sorted(candidates, key=lambda c: 0 if c.grasp_type == GraspType.FRONT else 1)
        if front_only:
            ordered = [c for c in ordered if c.grasp_type == GraspType.FRONT]

        ok = False
        home_q = env.data.qpos[:7].copy()
        for candidate in ordered:
            env.set_collision_exceptions(saved_exceptions)
            env.data.qpos[:7] = home_q
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)

            # --- Approach pose (standoff) ---
            env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
            if not env.ik.converge_ik(dt):
                continue
            approach_q = env.ik.configuration.q[:7].copy()
            if not env.is_collision_free(approach_q):
                continue

            # --- Grasp pose (arm fully extended to contact position) ---
            # The approach pose may be clear while the grasp pose (further along
            # the approach axis) hits adjacent cylinders.  Exclude the target
            # cylinder itself since the arm is intentionally in contact with it.
            env.add_collision_exception(cyl)
            env.data.qpos[:7] = approach_q
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)
            env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
            if not env.ik.converge_ik(dt):
                continue
            grasp_q = env.ik.configuration.q[:7].copy()
            if env.is_collision_free(grasp_q):
                ok = True
                break
    except Exception:
        ok = False
    finally:
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        env.set_collision_exceptions(saved_exceptions)

    ik_cache[key] = ok
    return ok


def _direct_blockers_1on1(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    target_cyl: str,
    ik_cache: dict,
) -> list[tuple[str, str]]:
    """Find cylinders that block *target_cyl* in 1-on-1 isolation.

    For each candidate obstacle with cy ≤ target_cy, set up a 2-cylinder
    scene (only target + that obstacle) and check whether the target's
    approach pose is in collision.  Obstacles that block even in isolation
    are true geometric blockers.

    Returns list of (cyl, cell) sorted by cy ascending (closest to robot
    first — most directly in the approach path).
    """
    arrangement_dict = dict(arrangement)
    target_cell = arrangement_dict.get(target_cyl)
    if target_cell is None:
        return []

    tparts = target_cell.split("_")
    ty = int(tparts[2])

    blockers: list[tuple[int, str, str]] = []  # (cy, cyl, cell)
    for cyl, cell in arrangement:
        if cyl == target_cyl:
            continue
        cparts = cell.split("_")
        cy = int(cparts[2])
        if cy > ty:  # behind target — can't block the approach from the robot side
            continue

        # Isolated scene: only target + this one obstacle
        isolated = frozenset([(target_cyl, target_cell), (cyl, cell)])
        if not _check_approach_ik(checker, isolated, target_cyl, ik_cache, front_only=True):
            blockers.append((cy, cyl, cell))

    blockers.sort(key=lambda x: x[0])  # closest to robot first
    return [(cyl, cell) for _, cyl, cell in blockers]


# ---------------------------------------------------------------------------
# No-drop mode — put IK+collision check (fast, no RRT)
# ---------------------------------------------------------------------------

def _check_put_ik(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    cyl: str,
    dest_cell: str,
    put_ik_cache: dict,
) -> bool:
    """IK convergence + collision check for the put approach and place poses.

    Analogous to _check_approach_ik for picks: deterministic, ~5-10 ms,
    no RRT.  Checks:
      1. Transport IK from HOME converges.
      2. IK for approach above dest_cell converges; approach_q is collision-free
         with the held cylinder teleported to its EE-relative position.
      3. IK for place-contact at dest_cell converges; place_q is collision-free.

    *arrangement* must already have *cyl* removed (table state while held).
    Cached by (arrangement, cyl, dest_cell).
    """
    key = (arrangement, cyl, dest_cell)
    result = put_ik_cache.get(key)
    if result is not None:
        return result

    env           = checker._env
    grasp_planner = checker._grasp_planner
    state_manager = checker._state_manager
    grid          = state_manager.grid

    saved_qpos       = env.data.qpos.copy()
    saved_qvel       = env.data.qvel.copy()
    saved_exceptions = list(env.collision_exceptions)
    ok = False
    try:
        # 1. Place table cylinders at their cells (no held cylinder yet).
        state = _arrangement_to_state_dict(arrangement)
        state_manager.set_from_grounded_state(state)
        env.controller.stop()
        mujoco.mj_forward(env.model, env.data)

        dt = env.model.opt.timestep

        # 2. Transport IK from HOME.
        env.ik.update_configuration(env.data.qpos)
        transport_pos, transport_quat = state_manager.get_transport_pose()
        env.ik.set_target_position(transport_pos, transport_quat)
        if not env.ik.converge_ik(dt):
            put_ik_cache[key] = False
            return False
        transport_q = env.ik.configuration.q[:7].copy()

        # 3. Teleport arm to transport_q, place cylinder at grasp-contact position
        #    (so rel_pos is correct ≈ [0,0,GRASP_CONTACT_OFFSET]), then attach.
        from manipulation.planners.grasp_planner import GRASP_CONTACT_OFFSET
        env.data.qpos[:7] = transport_q
        mujoco.mj_forward(env.model, env.data)
        ee_pos_t = env.data.site_xpos[env._ee_site_id].copy()
        ee_mat_t = env.data.site_xmat[env._ee_site_id].reshape(3, 3).copy()
        cyl_idx = int(cyl.split("_")[1])
        cyl_world_pos = ee_pos_t + ee_mat_t @ np.array([0.0, 0.0, GRASP_CONTACT_OFFSET])
        state_manager._set_cylinder_position(cyl_idx, cyl_world_pos[0], cyl_world_pos[1], cyl_world_pos[2])
        mujoco.mj_forward(env.model, env.data)
        env.attach_object_to_ee(cyl)

        # 4. Compute put target geometry.
        _radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
        half_size = np.array([_radius, _radius, half_height])

        cx, cy_coord = grid.cells[dest_cell]["center"]
        cyl_z = grid.table_height + half_height + 0.002
        target_pos = np.array([cx, cy_coord, cyl_z])

        candidates = grasp_planner.generate_candidates(target_pos, half_size)
        front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
        candidate = front if front is not None else (candidates[0] if candidates else None)
        if candidate is None:
            put_ik_cache[key] = False
            return False

        put_quat     = candidate.grasp_quat
        R_put        = quat_to_rotmat(put_quat)
        rel_pos_ee   = env._attached["rel_pos"]
        place_ee_pos = target_pos - R_put @ rel_pos_ee
        approach_pos = place_ee_pos + np.array([0.0, 0.0, grasp_planner.approach_dist])

        # 5. Approach pose: IK from transport_q, then collision check.
        env.data.qpos[:7] = transport_q
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        env.ik.set_target_position(approach_pos, put_quat)
        if not env.ik.converge_ik(dt):
            put_ik_cache[key] = False
            return False
        approach_q = env.ik.configuration.q[:7].copy()
        if not env.is_collision_free(approach_q):
            put_ik_cache[key] = False
            return False

        # 6. Place pose: IK from approach_q, then collision check.
        env.data.qpos[:7] = approach_q
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        env.ik.set_target_position(place_ee_pos, put_quat)
        if not env.ik.converge_ik(dt):
            put_ik_cache[key] = False
            return False
        place_q = env.ik.configuration.q[:7].copy()
        ok = env.is_collision_free(place_q)

    except Exception:
        ok = False
    finally:
        env.detach_object()
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        env.set_collision_exceptions(saved_exceptions)

    put_ik_cache[key] = ok
    return ok


# ---------------------------------------------------------------------------
# No-drop mode — put feasibility from transport pose
# ---------------------------------------------------------------------------

def _check_put_from_transport(
    checker: ActionFeasibilityChecker,
    arrangement: frozenset,
    cyl: str,
    dest_cell: str,
    put_rrt_cache: dict,
    put_ik_cache: dict | None = None,
) -> bool:
    """Check if *cyl* can be placed at *dest_cell* starting from transport pose.

    Mirrors the actual execution sequence:
      transport_q  →  approach above dest_cell  →  place descent

    Uses the feas_planner directly (no settle steps, no execute_path, no
    wait_idle) for speed and uses the same IK seed / collision setup as the
    execution code.

    If *put_ik_cache* is provided, runs _check_put_ik as a fast pre-filter
    before the RRT planning — avoids the stochastic RRT call when the endpoint
    is already provably in collision.

    *arrangement* must already have *cyl* removed (table state while held).
    """
    # Fast IK+collision pre-filter (same check, cheaper than RRT).
    if put_ik_cache is not None:
        if not _check_put_ik(checker, arrangement, cyl, dest_cell, put_ik_cache):
            put_rrt_cache[(arrangement, cyl, dest_cell)] = False
            return False
    key = (arrangement, cyl, dest_cell)
    result = put_rrt_cache.get(key)
    if result is not None:
        return result

    env           = checker._env
    feas_planner  = checker._feas_planner
    grasp_planner = checker._grasp_planner
    state_manager = checker._state_manager
    grid          = state_manager.grid

    saved_qpos       = env.data.qpos.copy()
    saved_qvel       = env.data.qvel.copy()
    saved_exceptions = list(env.collision_exceptions)
    ok = False
    try:
        # 1. Place table cylinders at their cells (no held cylinder yet).
        state = _arrangement_to_state_dict(arrangement)
        state_manager.set_from_grounded_state(state)
        env.controller.stop()
        mujoco.mj_forward(env.model, env.data)

        dt = env.model.opt.timestep

        # 2. Solve transport IK from HOME — same seed as execution.
        env.ik.update_configuration(env.data.qpos)  # qpos is HOME after set_from_grounded_state
        transport_pos, transport_quat = state_manager.get_transport_pose()
        env.ik.set_target_position(transport_pos, transport_quat)
        if not env.ik.converge_ik(dt):
            ok = False
            put_rrt_cache[key] = ok
            return ok
        transport_q = env.ik.configuration.q[:7].copy()

        # 3. Teleport arm to transport_q, place cylinder at grasp-contact position
        #    so rel_pos is correct (~[0,0,GRASP_CONTACT_OFFSET]), then attach.
        from manipulation.planners.grasp_planner import GRASP_CONTACT_OFFSET
        env.data.qpos[:7] = transport_q
        mujoco.mj_forward(env.model, env.data)
        ee_pos_t = env.data.site_xpos[env._ee_site_id].copy()
        ee_mat_t = env.data.site_xmat[env._ee_site_id].reshape(3, 3).copy()
        cyl_idx = int(cyl.split("_")[1])
        cyl_world_pos = ee_pos_t + ee_mat_t @ np.array([0.0, 0.0, GRASP_CONTACT_OFFSET])
        state_manager._set_cylinder_position(cyl_idx, cyl_world_pos[0], cyl_world_pos[1], cyl_world_pos[2])
        mujoco.mj_forward(env.model, env.data)
        env.attach_object_to_ee(cyl)  # now rel_pos ≈ [0,0,GRASP_CONTACT_OFFSET]

        # 4. Compute put target geometry (same as execution and _check_put).
        _radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
        half_size = np.array([_radius, _radius, half_height])

        cx, cy_coord = grid.cells[dest_cell]["center"]
        cyl_z = grid.table_height + half_height + 0.002
        target_pos = np.array([cx, cy_coord, cyl_z])

        candidates = grasp_planner.generate_candidates(target_pos, half_size)
        front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
        candidate = front if front is not None else (candidates[0] if candidates else None)
        if candidate is None:
            ok = False
            put_rrt_cache[key] = ok
            return ok

        put_quat     = candidate.grasp_quat
        R_put        = quat_to_rotmat(put_quat)
        rel_pos_ee   = env._attached["rel_pos"]
        place_ee_pos = target_pos - R_put @ rel_pos_ee
        approach_pos = place_ee_pos + np.array([0.0, 0.0, grasp_planner.approach_dist])

        # 5. IK + RRT: transport_q → approach pose above dest_cell.
        env.data.qpos[:7] = transport_q
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        env.ik.set_target_position(approach_pos, put_quat)
        if not env.ik.converge_ik(dt):
            ok = False
            put_rrt_cache[key] = ok
            return ok
        approach_q = env.ik.configuration.q[:7].copy()
        if not env.is_collision_free(approach_q):
            ok = False
            put_rrt_cache[key] = ok
            return ok
        path = feas_planner.plan(transport_q, approach_q, max_iterations=_FAST_PUT_ITERS)
        if path is None:
            ok = False
            put_rrt_cache[key] = ok
            return ok

        # 6. IK + RRT: approach pose → place-contact descent.
        env.data.qpos[:7] = approach_q
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        env.ik.set_target_position(place_ee_pos, put_quat)
        if not env.ik.converge_ik(dt):
            ok = False
            put_rrt_cache[key] = ok
            return ok
        place_q = env.ik.configuration.q[:7].copy()
        path = feas_planner.plan(approach_q, place_q, max_iterations=_FAST_PUT_ITERS)
        ok = path is not None

    except Exception:
        ok = False
    finally:
        env.detach_object()
        env.data.qpos[:] = saved_qpos
        env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(env.model, env.data)
        env.set_collision_exceptions(saved_exceptions)

    put_rrt_cache[key] = ok
    return ok


# ---------------------------------------------------------------------------
# No-drop mode — upfront put-candidate precomputation
# ---------------------------------------------------------------------------

def _precompute_put_candidates(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    placement_margin: int = 1,
) -> dict[str, list[tuple[str, float, float]]]:
    """Pre-compute valid put destinations for every cylinder in the scene.

    Called ONCE per scene (target-independent).  For each cylinder:
      - Sets up the MuJoCo scene ONCE (all other cylinders present, this one removed).
      - Solves transport IK once from HOME.
      - Attaches the cylinder to the EE.
      - Checks placement margin + approach-corridor + IK + RRT for each free cell.
      - Stores (cell_id, world_x, world_y) for each valid cell so callers can
        re-sort by any target without re-running MuJoCo.

    Returns  {cyl_name: [(cell_id, cx, cy), ...]  sorted by world_y DESC}.

    The caller (``_find_plan_no_drop``) re-sorts by  world_y + dist_from_target
    for the specific target of that BFS run.
    """
    env           = checker._env
    grasp_planner = checker._grasp_planner
    state_manager = checker._state_manager
    grid          = state_manager.grid

    result: dict[str, list[tuple[str, float, float]]] = {}
    all_cell_ids = list(grid.cells.keys())

    n_cyls = len(initial_arrangement)
    for idx_c, (focus_cyl, focus_cell) in enumerate(initial_arrangement):
        others_arr    = frozenset((c, cl) for c, cl in initial_arrangement if c != focus_cyl)
        occupied_cells = {cl for _, cl in others_arr}

        saved_qpos       = env.data.qpos.copy()
        saved_qvel       = env.data.qvel.copy()
        saved_exceptions = list(env.collision_exceptions)

        valid_cells: list[tuple[float, str, float, float]] = []  # (world_y, cell_id, cx, cy_coord)

        try:
            # ── Scene setup ONCE per cylinder ──────────────────────────────
            state = _arrangement_to_state_dict(others_arr)
            state_manager.set_from_grounded_state(state)
            env.controller.stop()
            env.data.qpos[:8] = env.initial_qpos[:8]
            env.data.ctrl[:8] = env.initial_ctrl[:8]
            mujoco.mj_forward(env.model, env.data)

            dt = env.model.opt.timestep

            # Transport IK once
            env.ik.update_configuration(env.data.qpos)
            transport_pos, transport_quat = state_manager.get_transport_pose()
            env.ik.set_target_position(transport_pos, transport_quat)
            if not env.ik.converge_ik(dt):
                result[focus_cyl] = []
                print(f"  [precompute {idx_c+1}/{n_cyls}] {focus_cyl}: transport IK failed")
                continue
            transport_q = env.ik.configuration.q[:7].copy()

            # Place cylinder at grasp-contact position so rel_pos is correct,
            # then attach. (Hidden cylinder at [100,0,50] would give wrong rel_pos.)
            from manipulation.planners.grasp_planner import GRASP_CONTACT_OFFSET
            env.data.qpos[:7] = transport_q
            mujoco.mj_forward(env.model, env.data)
            ee_pos_t = env.data.site_xpos[env._ee_site_id].copy()
            ee_mat_t = env.data.site_xmat[env._ee_site_id].reshape(3, 3).copy()
            cyl_idx  = int(focus_cyl.split("_")[1])
            cyl_world_pos = ee_pos_t + ee_mat_t @ np.array([0.0, 0.0, GRASP_CONTACT_OFFSET])
            state_manager._set_cylinder_position(
                cyl_idx, cyl_world_pos[0], cyl_world_pos[1], cyl_world_pos[2]
            )
            mujoco.mj_forward(env.model, env.data)
            env.attach_object_to_ee(focus_cyl)

            _radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
            half_size = np.array([_radius, _radius, half_height])

            # ── Check each free cell ───────────────────────────────────────
            for cell_id in all_cell_ids:
                if cell_id in occupied_cells or cell_id == focus_cell:
                    continue

                # Placement margin (pure geometry, ~0ms)
                parts = cell_id.split("_")
                dx, dy_grid = int(parts[1]), int(parts[2])
                margin_ok = True
                for _, other_cell in others_arr:
                    oparts = other_cell.split("_")
                    ox, oy = int(oparts[1]), int(oparts[2])
                    if max(abs(ox - dx), abs(oy - dy_grid)) <= placement_margin:
                        margin_ok = False
                        break
                if not margin_ok:
                    continue

                # Approach-corridor check (~0ms, deterministic).
                # For a FRONT approach the arm travels in the +y direction from
                # the robot to the destination.  Any cylinder in the same column
                # (same x) at a y-value strictly between the robot edge (y=0)
                # and the destination (y=dy_grid - margin) lies directly in the
                # approach path and will block it.  This rules out "clearly
                # infeasible" puts without any IK or RRT calls.
                corridor_blocked = any(
                    ox == dx and 0 <= oy < dy_grid - placement_margin
                    for _, other_cell in others_arr
                    for ox, oy in [(int(other_cell.split("_")[1]),
                                    int(other_cell.split("_")[2]))]
                )
                if corridor_blocked:
                    continue

                # Target geometry
                cx, cy_coord = grid.cells[cell_id]["center"]
                cyl_z   = grid.table_height + half_height + 0.002
                tgt_pos = np.array([cx, cy_coord, cyl_z])

                candidates = grasp_planner.generate_candidates(tgt_pos, half_size)
                front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
                candidate = front if front is not None else (candidates[0] if candidates else None)
                if candidate is None:
                    continue

                put_quat     = candidate.grasp_quat
                R_put        = quat_to_rotmat(put_quat)
                rel_pos_ee   = env._attached["rel_pos"]
                place_ee_pos = tgt_pos - R_put @ rel_pos_ee
                approach_pos = place_ee_pos + np.array([0.0, 0.0, grasp_planner.approach_dist])

                # IK approach from transport_q + collision check
                env.data.qpos[:7] = transport_q
                mujoco.mj_forward(env.model, env.data)
                env.ik.update_configuration(env.data.qpos)
                env.ik.set_target_position(approach_pos, put_quat)
                if not env.ik.converge_ik(dt):
                    continue
                approach_q = env.ik.configuration.q[:7].copy()
                if not env.is_collision_free(approach_q):
                    continue

                # IK place from approach_q + collision check
                env.data.qpos[:7] = approach_q
                mujoco.mj_forward(env.model, env.data)
                env.ik.update_configuration(env.data.qpos)
                env.ik.set_target_position(place_ee_pos, put_quat)
                if not env.ik.converge_ik(dt):
                    continue
                place_q = env.ik.configuration.q[:7].copy()
                if not env.is_collision_free(place_q):
                    continue

                # Direct placement check: teleport held cylinder to EXACTLY
                # tgt_pos (no IK orientation error) and verify no collision.
                # This is the most reliable guard for "does it fit here?".
                held_info = env._attached
                if held_info is not None:
                    cyl_qadr = held_info["qadr"]
                    saved_cyl_qpos = env.data.qpos[cyl_qadr:cyl_qadr + 7].copy()
                    env.data.qpos[cyl_qadr:cyl_qadr + 3] = tgt_pos
                    env.data.qpos[cyl_qadr + 3:cyl_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
                    mujoco.mj_forward(env.model, env.data)
                    placement_clear = env.check_collisions()
                    env.data.qpos[cyl_qadr:cyl_qadr + 7] = saved_cyl_qpos
                    mujoco.mj_forward(env.model, env.data)
                    if not placement_clear:
                        continue

                # Fast path-existence check: short RRT for each segment.
                # The scene is already set up (cylinder attached, others placed), so
                # this reuses the current state — no extra set_from_grounded_state cost.
                # feas_planner.plan() saves/restores qpos internally.
                feas_planner = checker._feas_planner
                if feas_planner.plan(
                    transport_q, approach_q, max_iterations=_PRECOMPUTE_RRT_ITERS
                ) is None:
                    continue
                if feas_planner.plan(
                    approach_q, place_q, max_iterations=_PRECOMPUTE_RRT_ITERS
                ) is None:
                    continue

                # Store world coords so caller can re-sort per target cheaply.
                valid_cells.append((cy_coord, cell_id, cx, cy_coord))

        except Exception as e:
            print(f"  [precompute {idx_c+1}/{n_cyls}] {focus_cyl}: exception: {e}")
        finally:
            env.detach_object()
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            mujoco.mj_forward(env.model, env.data)
            env.set_collision_exceptions(saved_exceptions)

        valid_cells.sort(key=lambda x: -x[0])  # world_y DESC as default order
        result[focus_cyl] = [(cell_id, cx, cy_c) for _, cell_id, cx, cy_c in valid_cells]
        print(f"  [precompute {idx_c+1}/{n_cyls}] {focus_cyl}: "
              f"{len(result[focus_cyl])} valid put destinations")

    return result


# ---------------------------------------------------------------------------
# No-drop mode — RRT plan validation
# ---------------------------------------------------------------------------

def _validate_plan_rrt(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    plan: list[tuple],
    put_rrt_cache: dict,
    put_ik_cache: dict | None = None,
    pick_cache: dict | None = None,
) -> tuple[bool, int, dict]:
    """Validate every action in *plan* with full RRT checks via the checker.

    Returns ``(True, -1, {})`` if all actions pass.
    Returns ``(False, fail_step, fail_info)`` on the first failure.

    ``fail_info`` always contains:
      "type"         : "pick" | "put"
      "cyl"          : cylinder name
      "arrangement"  : table arrangement at the failing step
      "plan_prefix"  : plan actions before the *pick* that precedes this action

    For "put" failures it also contains:
      "dest"                  : the failing destination cell
      "arrangement_before_pick": arrangement before the pick of this cylinder
    """
    arrangement = initial_arrangement
    # arr_history[i] = table arrangement just before executing plan[i]
    arr_history: list[frozenset] = [initial_arrangement]

    for i, action in enumerate(plan):
        if action[0] == "pick":
            cyl, cell = action[1], action[2]
            if pick_cache is not None:
                ok = _check_pick_no_drop(checker, arrangement, cyl, pick_cache)
                reason = "pick_no_drop_fail"
            else:
                state = _arrangement_to_state_dict(arrangement)
                ok, timing = checker.check("pick", state, cylinder_name=cyl)
                reason = timing.get("reason", "?")
            if not ok:
                return False, i, {
                    "type":        "pick",
                    "cyl":         cyl,
                    "cell":        cell,
                    "arrangement": arrangement,
                    "plan_prefix": list(plan[:i]),
                    "reason":      reason,
                }
            arrangement = arrangement - {(cyl, cell)}

        elif action[0] == "put":
            cyl, dest = action[1], action[2]
            # arrangement here has cyl already removed (picked in previous step).
            # Use _check_put_from_transport: starts the arm at transport_q (same
            # as execution) and checks transport_q → approach → place via RRT.
            ok = _check_put_from_transport(
                checker, arrangement, cyl, dest, put_rrt_cache, put_ik_cache
            )
            if not ok:
                return False, i, {
                    "type":                  "put",
                    "cyl":                   cyl,
                    "dest":                  dest,
                    "arrangement":           arrangement,     # table while cyl is held
                    "arrangement_before_pick": arr_history[i - 1],
                    "plan_prefix":           list(plan[:i - 1]),
                    "reason":                "rrt_put_fail",
                }
            arrangement = arrangement | {(cyl, dest)}

        arr_history.append(arrangement)

    return True, -1, {}


# ---------------------------------------------------------------------------
# No-drop mode — best-first search solver
# ---------------------------------------------------------------------------

def _find_plan_no_drop(
    checker: ActionFeasibilityChecker,
    initial_arrangement: frozenset,
    target: str,
    reachable_cells: frozenset,
    pick_cache: dict,
    put_cache: dict,
    ik_cache: dict,
    placement_margin: int = 1,
    max_seconds: float | None = None,
    precomputed_puts: dict | None = None,
) -> list[tuple] | None:
    """Best-first search over pick+put pairs to find a no-drop plan for *target*.

    Node priority is determined by the 1-on-1 blocker chain:
      0 — moving a direct 1-on-1 blocker of the target (closest to robot first)
      1 — moving a blocker of a blocker (secondary chain)
      2 — any other pickable obstacle (BFS fallback behaviour)

    Within the same priority level nodes expand in insertion order (FIFO),
    which gives BFS semantics as a natural fallback when the heuristic runs
    out of directed moves.

    Obstacle movability is pre-filtered with the cheap IK+collision check
    (_check_approach_ik); the definitive pick and target feasibility checks
    still use the full IK+collision+RRT path (_check_pick_no_drop).
    """
    target_cell_initial = dict(initial_arrangement).get(target)
    if target_cell_initial is None:
        return None

    t_start = time.time()

    # Pre-compute valid put destinations once, upfront.
    # If not provided by caller, compute now.
    if precomputed_puts is None:
        print(f"  [BFS] precomputing put candidates for {len(initial_arrangement)} cylinders …")
        precomputed_puts = _precompute_put_candidates(
            checker, initial_arrangement, placement_margin
        )
        elapsed = time.time() - t_start
        total_cands = sum(len(v) for v in precomputed_puts.values())
        print(f"  [BFS] precompute done in {elapsed:.1f}s — "
              f"{total_cands} total candidates across {len(precomputed_puts)} cylinders")

    # Re-sort candidates for this specific target: world_y + dist_from_target.
    # Precompute stores (cell_id, cx, cy) — sorting is O(K) per cylinder, ~0ms.
    target_cell_info = checker._state_manager.grid.cells.get(target_cell_initial, {})
    tx, ty = target_cell_info.get("center", (0.0, 0.0))
    sorted_puts: dict[str, list[str]] = {}
    for cyl, cands in precomputed_puts.items():
        ranked = sorted(
            cands,
            key=lambda t: -(t[2] + np.sqrt((t[1] - tx) ** 2 + (t[2] - ty) ** 2)),
        )
        sorted_puts[cyl] = [cell_id for cell_id, _, _ in ranked]
    precomputed_puts = sorted_puts

    # Heap entries: (priority, cy, counter, arrangement, plan)
    # priority: 0=direct blocker, 1=secondary blocker, 2=other
    # cy: lower = closer to robot = more urgent within same priority
    # counter: insertion order for FIFO tie-breaking within same (priority, cy)
    counter = 0
    heap: list = [(0, 0, counter, initial_arrangement, [])]
    visited: set[frozenset] = {initial_arrangement}

    # RRT-confirmed infeasible put destinations: (table_arrangement_while_held, dest_cell)
    # "table_arrangement_while_held" = arrangement with the moving cylinder already removed.
    bad_put_dests: set[tuple] = set()

    # Cache for _check_put_from_transport results (keyed by (arrangement, cyl, dest)).
    put_rrt_cache: dict = {}
    # put_ik_cache is kept for the pick-failure fallback path that calls _check_put_ik.
    put_ik_cache: dict = {}

    _DEBUG_INTERVAL = 25  # print progress every N nodes

    nodes_explored = 0
    while heap:
        if max_seconds is not None and (time.time() - t_start) > max_seconds:
            print(f"  [BFS] target={target} TIMEOUT after {max_seconds:.0f}s "
                  f"nodes={nodes_explored} heap={len(heap)}")
            return None

        _, _, _, arrangement, plan = heapq.heappop(heap)
        depth = len(plan) // 2
        nodes_explored += 1

        if nodes_explored % _DEBUG_INTERVAL == 0:
            elapsed = time.time() - t_start
            print(f"  [BFS] target={target} nodes={nodes_explored} "
                  f"heap={len(heap)} depth={depth} "
                  f"put_rrt={len(put_rrt_cache)} t={elapsed:.1f}s")

        # 1-on-1 blocker analysis — computed first because it uses isolation
        # keys that cache well, and its result lets us skip the expensive
        # pick check (~100ms) whenever blockers are still present.
        direct = _direct_blockers_1on1(checker, arrangement, target, ik_cache)
        direct_cyls = {cyl for cyl, _ in direct}

        # No direct blockers remain — check if approach is clear in the full scene.
        # Use the fast IK+collision check (5ms, deterministic) rather than
        # the stochastic RRT pick check (100ms), which was creating false
        # negatives for back-row cylinders and killing long-plan candidates.
        # If the approach is IK-feasible and collision-free with all remaining
        # cylinders present, the pick is achievable — the plan is valid.
        if not direct_cyls:
            # Also require target not to be surrounded — direct blockers may
            # not capture cylinders behind/diagonal to the target.
            if (_pick_neighbor_count(arrangement, target) < _MAX_PICK_NEIGHBORS
                    and _check_approach_ik(checker, arrangement, target, ik_cache, front_only=True)):
                target_cell = dict(arrangement)[target]
                candidate_plan = plan + [("pick", target, target_cell)]

                # Validate every action with full RRT — the static check only
                # verifies endpoint reachability, not path existence.
                rrt_ok, fail_step, fail_info = _validate_plan_rrt(
                    checker, initial_arrangement, candidate_plan,
                    put_rrt_cache, put_ik_cache, pick_cache
                )
                if rrt_ok:
                    return candidate_plan

                # ── Feed RRT failure back into BFS ─────────────────────────
                if fail_info["type"] == "put":
                    # Blacklist this (table_state, destination) pair so it is
                    # skipped in all future expansions.
                    bad_put_dests.add((fail_info["arrangement"], fail_info["dest"]))

                    # Push the arrangement just before the failed pick+put so
                    # the BFS can try different put destinations for that move.
                    arr_before = fail_info["arrangement_before_pick"]
                    prefix     = fail_info["plan_prefix"]
                    depth_before = len(prefix) // 2
                    if depth_before < _BFS_MAX_DEPTH:
                        # Do NOT guard with visited — we want to re-expand this
                        # arrangement with the updated bad_put_dests blacklist.
                        counter += 1
                        heapq.heappush(heap, (1, 0, counter, arr_before, prefix))

                elif fail_info["type"] == "pick":
                    # The static check passed but RRT couldn't find a path.
                    # Use leave-one-out RRT to identify which cylinders block the
                    # trajectory (not just the endpoint).
                    arrangement_failed = fail_info["arrangement"]
                    prefix             = fail_info["plan_prefix"]
                    cyl_failed         = fail_info["cyl"]
                    depth_before       = len(prefix) // 2

                    if depth_before < _BFS_MAX_DEPTH:
                        for pb_cyl, pb_cell in arrangement_failed:
                            if pb_cyl == cyl_failed:
                                continue
                            reduced = arrangement_failed - {(pb_cyl, pb_cell)}
                            state   = _arrangement_to_state_dict(reduced)
                            pick_ok, _ = checker.check("pick", state, cylinder_name=cyl_failed)
                            if not pick_ok:
                                continue
                            # pb_cyl is a trajectory blocker — push moves for it
                            pb_cy      = int(pb_cell.split("_")[2])
                            pb_others  = frozenset(
                                (c, cl) for c, cl in arrangement_failed if c != pb_cyl
                            )
                            pb_put_count = 0
                            pb_cands = precomputed_puts.get(pb_cyl, [])
                            for dest in pb_cands:
                                if pb_put_count >= _BFS_MAX_PUT_CANDS:
                                    break
                                if (pb_others, dest) in bad_put_dests:
                                    continue
                                if not _check_put_no_drop(
                                    arrangement_failed,
                                    pb_cyl, dest, put_cache, placement_margin
                                ):
                                    continue
                                pb_put_count += 1
                                new_arr = (
                                    (arrangement_failed - {(pb_cyl, pb_cell)})
                                    | {(pb_cyl, dest)}
                                )
                                if new_arr in visited:
                                    continue
                                visited.add(new_arr)
                                counter += 1
                                heapq.heappush(heap, (
                                    1, pb_cy, counter, new_arr,
                                    prefix + [
                                        ("pick", pb_cyl, pb_cell),
                                        ("put",  pb_cyl, dest),
                                    ],
                                ))

            # Approach is blocked in the full scene despite no 1-on-1 blocker —
            # combination block; can't help by moving obstacles, dead end.
            continue

        # Prune: if more blockers remain than budget allows, no solution possible.
        if depth + len(direct_cyls) > _BFS_MAX_DEPTH or depth >= _BFS_MAX_DEPTH:
            continue

        # Chain walking: focus on ONE cylinder per expansion.
        # "direct" is sorted by cy ascending (closest to robot first).
        # Walk down the chain: if the closest direct blocker is itself blocked,
        # recurse one level into its blocker rather than expanding everything.
        arrangement_dict = dict(arrangement)

        focus_cyl: str | None = None
        focus_cell: str | None = None
        focus_priority: int = 0

        candidate_cyl, candidate_cell = direct[0]
        isolated = frozenset([(candidate_cyl, candidate_cell)])
        if _check_approach_ik(checker, isolated, candidate_cyl, ik_cache):
            # Closest direct blocker is movable — focus on it.
            focus_cyl, focus_cell, focus_priority = candidate_cyl, candidate_cell, 0
        else:
            # Closest direct blocker is itself blocked — walk one level deeper.
            sub_blockers = _direct_blockers_1on1(
                checker, arrangement, candidate_cyl, ik_cache
            )
            for s_cyl, s_cell in sub_blockers:
                isolated_s = frozenset([(s_cyl, s_cell)])
                if _check_approach_ik(checker, isolated_s, s_cyl, ik_cache):
                    focus_cyl, focus_cell, focus_priority = s_cyl, s_cell, 1
                    break

        if focus_cyl is None:
            continue  # no movable cylinder found — dead end

        cparts = focus_cell.split("_")
        cy = int(cparts[2])

        # Density guard: if focus_cyl is surrounded by ≥ _MAX_PICK_NEIGHBORS
        # cylinders in the current full arrangement it is almost certainly
        # unpickable — skip without generating any pick+put pairs.
        if _pick_neighbor_count(arrangement, focus_cyl) >= _MAX_PICK_NEIGHBORS:
            continue

        focus_others = frozenset(
            (c, cl) for c, cl in arrangement if c != focus_cyl
        )
        # Use precomputed put candidates (IK+collision already verified upfront).
        # Only check current-arrangement occupancy/margin via _check_put_no_drop.
        put_count = 0
        cands = precomputed_puts.get(focus_cyl, [])
        for dest in cands:
            if put_count >= _BFS_MAX_PUT_CANDS:
                break
            if (focus_others, dest) in bad_put_dests:
                continue  # RRT-confirmed infeasible — skip
            if not _check_put_no_drop(
                arrangement, focus_cyl, dest, put_cache, placement_margin
            ):
                continue
            put_count += 1
            new_arr = (arrangement - {(focus_cyl, focus_cell)}) | {(focus_cyl, dest)}
            if new_arr in visited:
                continue
            visited.add(new_arr)
            counter += 1
            heapq.heappush(heap, (
                focus_priority, cy, counter,
                new_arr, plan + [("pick", focus_cyl, focus_cell), ("put", focus_cyl, dest)],
            ))

    return None


# ---------------------------------------------------------------------------
# No-drop mode — plan compression
# ---------------------------------------------------------------------------

def _compress_plan(
    plan: list[tuple],
    initial_arrangement: frozenset,
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
                    if _check_put_no_drop(arr, cyl, c3,
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
    pick_cache: dict,
    put_cache: dict,
    ik_cache: dict,
    placement_margin: int = 1,
    precomputed_puts: dict | None = None,
) -> tuple[str, list[tuple]] | tuple[None, None]:
    """Pick one uniformly random target cylinder and find a no-drop plan for it.

    Returns (target_cylinder, plan) or (None, None) if no plan exists.
    All filtering (min_plan_len, Gaussian shaping) is handled by the caller.
    Pass precomputed_puts to avoid re-running the precompute for each target.
    """
    target = str(np.random.choice(cylinders))
    plan = _find_plan_no_drop(
        checker, initial_arrangement, target, reachable_cells,
        pick_cache, put_cache, ik_cache, placement_margin=placement_margin,
        precomputed_puts=precomputed_puts,
    )
    return (target, plan) if plan is not None else (None, None)


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
        "reject_plan_len": 0, "reject_distribution": 0,
        "requested_hist": Counter(), "actual_hist": Counter(),
        "plan_len_hist": Counter(), "gen_seconds": [],
    }


def _merge_stats(items: list[dict]) -> dict:
    merged = _init_stats()
    for s in items:
        for k in ("attempts", "accepted", "reject_min_objects", "reject_no_plan",
                  "reject_plan_len", "reject_distribution"):
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
          f"no_plan={stats['reject_no_plan']} "
          f"plan_len={stats['reject_plan_len']} "
          f"distribution={stats['reject_distribution']}")
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
    ik_cache: dict,
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
            # Precompute once per scene; _select_target_no_drop re-sorts per target.
            scene_precomputed = _precompute_put_candidates(
                checker, initial_arrangement,
                placement_margin=args.placement_margin,
            )
            target, plan = _select_target_no_drop(
                checker, initial_arrangement, cylinders, reachable_cells,
                pick_cache, put_cache, ik_cache,
                placement_margin=args.placement_margin,
                precomputed_puts=scene_precomputed,
            )
            if plan is None:
                stats["reject_no_plan"] += 1
                continue

            # Hard minimum plan length filter.
            if len(plan) < args.min_plan_len:
                stats["reject_plan_len"] += 1
                continue

            # Gaussian distribution shaping (optional).
            # Acceptance probability peaks at plan_len == plan_len_mean and
            # falls off as exp(-0.5 * ((k - mean) / std)^2), so plans near
            # the target length are always kept and distant lengths are
            # down-sampled proportionally.
            if args.plan_len_mean is not None:
                k = len(plan)
                p = np.exp(
                    -0.5 * ((k - args.plan_len_mean) / args.plan_len_std) ** 2
                )
                if np.random.random() > p:
                    stats["reject_distribution"] += 1
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
    ik_cache:   dict = {}

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
            # Precompute once per scene; _select_target_no_drop re-sorts per target.
            scene_precomputed = _precompute_put_candidates(
                checker, initial_arrangement,
                placement_margin=args.placement_margin,
            )
            target, plan = _select_target_no_drop(
                checker, initial_arrangement, cylinders, reachable_cells,
                pick_cache, put_cache, ik_cache,
                placement_margin=args.placement_margin,
                precomputed_puts=scene_precomputed,
            )
            if plan is None:
                stats["reject_no_plan"] += 1
                continue

            # Hard minimum plan length filter.
            if len(plan) < args.min_plan_len:
                stats["reject_plan_len"] += 1
                continue

            # Gaussian distribution shaping (optional).
            # Acceptance probability peaks at plan_len == plan_len_mean and
            # falls off as exp(-0.5 * ((k - mean) / std)^2), so plans near
            # the target length are always kept and distant lengths are
            # down-sampled proportionally.
            if args.plan_len_mean is not None:
                k = len(plan)
                p = np.exp(
                    -0.5 * ((k - args.plan_len_mean) / args.plan_len_std) ** 2
                )
                if np.random.random() > p:
                    stats["reject_distribution"] += 1
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
        ik_cache:   dict = {}

        for split, n in [("train", args.num_train),
                          ("test",  args.num_test),
                          ("validation", args.num_val)]:
            s = _generate_split(
                split, n, args,
                env, planner, feas_planner, checker, reachable_cells,
                grid, state_manager, grasp_planner,
                output_base, pick_cache, put_cache, ik_cache, wandb_run,
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
