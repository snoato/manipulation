"""Live demo: headless BFS planning followed by full physical execution.

For each episode:
  1. A random scene is sampled.
  2. BFS finds the shortest pick sequence to reach a target (headless, fast).
  3. The MuJoCo viewer opens and the robot executes every step of the plan:
       - For each blocker: approach → grasp → lift → discard
       - For the target:   approach → grasp → lift → hold → release → home
  4. Repeat indefinitely (close the viewer window to stop).

Usage:
    python examples/demo_solve.py
    python examples/demo_solve.py --objects 5 --seed 42
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from tampanda import FrankaEnvironment, RRTStar, SCENE_SYMBOLIC
from tampanda.planners.grasp_planner import GraspPlanner, GraspType
from tampanda.symbolic.domains.tabletop.grid_domain import GridDomain
from tampanda.symbolic.domains.tabletop.state_manager import StateManager
from tampanda.symbolic.domains.tabletop.generate_data import (
    _patch_fast_step, _sample_state, _select_target,
)

_XML = SCENE_SYMBOLIC

# Discard position — off to the right of the table
_DISCARD_POS  = np.array([0.55, 0.0, 0.65])
_DISCARD_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # pointing down

RRT_ITERS     = 1000
IK_ITERS      = 100
IK_POS_THRESH = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_candidate(candidates):
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _go_home(env: FrankaEnvironment, planner: RRTStar, viewer):
    """Return robot to home configuration."""
    home = env.initial_qpos[:7].copy()
    path = planner.plan(env.data.qpos[:7].copy(), home)
    if path:
        env.execute_path(path, planner, step_size=0.02)
        _wait(env, viewer)


def _wait(env: FrankaEnvironment, viewer, extra_s: float = 0.0):
    """Step until idle, then optionally wait extra seconds."""
    env.wait_idle(max_steps=8000, settle_steps=50)
    if extra_s > 0:
        env.rest(extra_s)
    if viewer is not None:
        viewer.sync()


def _execute_pick(env: FrankaEnvironment, planner: RRTStar,
                  grasp_planner: GraspPlanner,
                  cyl_name: str, viewer, dt: float) -> bool:
    """Approach → grasp a cylinder. Returns True on success."""
    cyl_pos   = env.get_object_position(cyl_name)
    half_size = env.get_object_half_size(cyl_name)
    cyl_quat  = env.get_object_orientation(cyl_name)
    candidate = _pick_candidate(
        grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
    )
    if candidate is None:
        print(f"    [{cyl_name}] no grasp candidate")
        return False

    env.add_collision_exception(cyl_name)

    # Approach
    print(f"    [{cyl_name}] planning approach...")
    path = planner.plan_to_pose(candidate.approach_pos, candidate.grasp_quat,
                                dt=dt, max_iterations=RRT_ITERS)
    if path is None:
        print(f"    [{cyl_name}] approach planning failed")
        env.clear_collision_exceptions()
        return False
    env.execute_path(path, planner, step_size=0.02)
    _wait(env, viewer)

    # Grasp
    print(f"    [{cyl_name}] planning grasp contact...")
    path = planner.plan_to_pose(candidate.grasp_pos, candidate.grasp_quat,
                                dt=dt, max_iterations=RRT_ITERS)
    if path is None:
        print(f"    [{cyl_name}] grasp planning failed")
        env.clear_collision_exceptions()
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait(env, viewer)

    # Close gripper
    print(f"    [{cyl_name}] closing gripper...")
    env.controller.close_gripper()
    _wait(env, viewer, extra_s=0.5)
    return True


def _lift(env: FrankaEnvironment, planner: RRTStar, cyl_name: str,
          viewer, dt: float, height: float = 0.15):
    """Lift straight up by `height` metres from current EE position."""
    cyl_pos  = env.get_object_position(cyl_name)
    lift_pos  = cyl_pos + np.array([0.0, 0.0, height])
    # Use same orientation as current grasp
    quat = env.data.mocap_quat[0].copy()
    print(f"    [{cyl_name}] lifting...")
    path = planner.plan_to_pose(lift_pos, quat, dt=dt, max_iterations=RRT_ITERS)
    if path:
        env.execute_path(path, planner, step_size=0.02)
        _wait(env, viewer)


def _discard(env: FrankaEnvironment, planner: RRTStar,
             cyl_name: str, viewer, dt: float):
    """Move to discard zone and open gripper."""
    print(f"    [{cyl_name}] moving to discard zone...")
    path = planner.plan_to_pose(_DISCARD_POS, _DISCARD_QUAT,
                                dt=dt, max_iterations=RRT_ITERS)
    if path:
        env.execute_path(path, planner, step_size=0.02)
        _wait(env, viewer)

    print(f"    [{cyl_name}] dropping...")
    env.controller.open_gripper()
    _wait(env, viewer, extra_s=0.5)
    env.clear_collision_exceptions()


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_episode(env: FrankaEnvironment, planner: RRTStar,
                grid: GridDomain, state_manager: StateManager,
                grasp_planner: GraspPlanner,
                n_objects: int, episode: int) -> bool:
    """Plan headlessly, then execute with viewer. Returns False to stop."""

    # ---- 1. Headless planning ----------------------------------------
    original_step = env.step
    _patch_fast_step(env)
    env.ik.max_iters     = IK_ITERS
    env.ik.pos_threshold = IK_POS_THRESH

    print(f"\n{'='*65}")
    print(f"  Episode {episode}  —  sampling {n_objects} cylinders...")
    t0 = time.perf_counter()

    _sample_state(state_manager, n_cylinders=n_objects, placement_margin=1)
    state    = state_manager.ground_state()
    cylinders = sorted(state["cylinders"].keys())
    print(f"  Placed {len(cylinders)} cylinders: {', '.join(cylinders)}")

    print(f"  Searching for plan (headless)...")
    target, plan = _select_target(env, planner, grasp_planner, cylinders, RRT_ITERS)

    t_plan = (time.perf_counter() - t0) * 1000
    if plan is None:
        print(f"  No feasible plan found ({t_plan:.0f}ms) — skipping episode.")
        env.step = original_step
        return True

    print(f"  Plan found in {t_plan:.0f}ms:  {' → '.join(plan)}")
    print(f"  Target: {target}  (plan length: {len(plan)} picks)")

    # ---- 2. Restore scene & switch to real-time step -----------------
    # Re-sample the same scene (planning may have moved the robot/cylinders)
    _sample_state(state_manager, n_cylinders=n_objects, placement_margin=1)
    # Actually re-use the grounded state that was planned against
    state_manager.set_from_grounded_state(state)
    mujoco.mj_forward(env.model, env.data)

    env.step = original_step  # restore rate-limited step for execution

    # Reset robot to home
    env.data.qpos[:8] = env.initial_qpos[:8]
    env.data.ctrl[:8] = env.initial_qpos[:8]
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)

    dt = env.model.opt.timestep

    # ---- 3. Execute plan in viewer -----------------------------------
    print(f"\n  Executing plan in viewer...")
    with env.launch_viewer() as viewer:
        viewer.sync()
        time.sleep(1.0)  # brief pause so user can see the initial scene

        for i, cyl_name in enumerate(plan):
            is_target = (i == len(plan) - 1)
            label = "TARGET" if is_target else f"blocker {i+1}/{len(plan)-1}"
            print(f"\n  Step {i+1}/{len(plan)}: picking {cyl_name} ({label})")

            ok = _execute_pick(env, planner, grasp_planner, cyl_name, viewer, dt)

            if not ok:
                print(f"  Execution failed at {cyl_name} — aborting episode.")
                break

            _lift(env, planner, cyl_name, viewer, dt)

            if is_target:
                print(f"\n  *** GOAL REACHED: holding {target} ***")
                time.sleep(2.0)
                print(f"  Releasing and returning home...")
                env.controller.open_gripper()
                _wait(env, viewer, extra_s=0.5)
                env.clear_collision_exceptions()
            else:
                _discard(env, planner, cyl_name, viewer, dt)

            _go_home(env, planner, viewer)

        if not viewer.is_running():
            return False

        time.sleep(1.5)

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--objects", type=int, default=4,
                    help="Number of cylinders per episode")
    ap.add_argument("--seed",    type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("Building environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    env.ik.max_iters     = IK_ITERS
    env.ik.pos_threshold = IK_POS_THRESH

    planner = RRTStar(env)
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )

    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    print(f"Grid: {grid.cells_x}×{grid.cells_y} cells  "
          f"(objects per episode: {args.objects})")
    print("Close the viewer window between episodes to stop.\n")

    episode = 0
    while True:
        episode += 1
        keep_going = run_episode(
            env, planner, grid, state_manager, grasp_planner,
            n_objects=args.objects, episode=episode,
        )
        if not keep_going:
            break

    print(f"\nDone after {episode} episode(s).")


if __name__ == "__main__":
    main()
