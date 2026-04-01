"""RRT*-based grasping using GraspPlanner for geometry-aware grasp candidates.

GraspPlanner features:
  - Reads object geometry (half-extents) to compute valid grasp widths
  - Generates and ranks multiple candidates (top-down X/Y, front approach)
  - Accounts for table clearance and gripper opening limits
  - Aligns finger axis with block orientation to minimise required opening
"""

import time

import numpy as np

from tampanda import RRTStar, ControllerStatus, GraspPlanner
from tampanda.planners.grasp_planner import GraspType
from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder

_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
_HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])


def _pick_candidate(candidates):
    """Select best candidate for tall thin cylinders.

    GraspPlanner scores top-down highest (+20), but for tall cylinders the
    top-down grasp point lands inside the cylinder body (contact at body
    centre ≠ top surface) and the round surface makes it mechanically
    unstable.  A front (side) approach contacts at the correct height and
    grips around the cylinder properly.
    """
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def main():
    env = make_symbolic_builder().build_env(rate=200.0)

    planner = RRTStar(env)
    planner.max_iterations   = 1000
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )
    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    with env.launch_viewer() as viewer:
        targets   = []
        target    = None
        candidate = None

        while viewer.is_running():
            if target is None and len(targets) == 0:
                state_manager.sample_random_state(n_cylinders=3)
                env.data.qpos[:8] = _HOME_QPOS
                env.data.ctrl[:8] = _HOME_CTRL
                env.reset_velocities()
                env.forward()
                env.rest(2.0)
                targets = sorted(state_manager.ground_state()["cylinders"].keys())

            dt = env.step()

            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target     = targets[0]
                target_pos = env.get_object_position(target)
                half_size  = env.get_object_half_size(target)
                block_quat = env.get_object_orientation(target)
                candidates = grasp_planner.generate_candidates(target_pos, half_size, block_quat)
                candidate  = _pick_candidate(candidates)
                step = 0
                if candidate:
                    print(f"Grasp candidate: {candidate.grasp_type.value}  score={candidate.score:.0f}")
                else:
                    print(f"No valid grasp candidate for {target}, skipping.")

            if target is not None and candidate is not None:
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        print(f"Planning approach to {target}...")
                        path = planner.plan_to_pose(
                            candidate.approach_pos, candidate.grasp_quat,
                            dt=dt, max_iterations=2000
                        )
                        if path is not None:
                            print("Path found! Executing...")
                            env.execute_path(path, planner)
                        else:
                            print("Failed to plan approach path!")
                            step = 7

                    if step == 2:
                        print("Planning grasp...")
                        env.add_collision_exception(target)
                        path = planner.plan_to_pose(
                            candidate.grasp_pos, candidate.grasp_quat,
                            dt=dt, max_iterations=1000
                        )
                        if path is not None:
                            env.execute_path(path, planner)
                        else:
                            print("Failed to plan grasp path!")
                            env.clear_collision_exceptions()
                            step = 7

                    if step == 3:
                        print("Closing gripper...")
                        env.controller.close_gripper()

                    if step == 4:
                        print("Planning lift...")
                        env.attach_object_to_ee(target)
                        path = planner.plan_to_pose(
                            candidate.lift_pos, candidate.grasp_quat,
                            dt=dt, max_iterations=2000
                        )
                        if path is not None:
                            env.execute_path(path, planner)
                        else:
                            print("Failed to plan lift path!")
                            step = 7

                    if step == 5:
                        print("Planning dropoff...")
                        dropoff_pos, dropoff_quat = env.get_dropoff_pose()
                        path = planner.plan_to_pose(
                            dropoff_pos, dropoff_quat,
                            dt=dt, max_iterations=3000
                        )
                        if path is not None:
                            env.execute_path(path, planner)
                        else:
                            print("Failed to plan dropoff path!")
                            step = 7

                    if step == 6:
                        print("Opening gripper...")
                        env.detach_object()
                        env.clear_collision_exceptions()
                        env.controller.open_gripper()

                    if step == 7:
                        target    = None
                        candidate = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            time.sleep(2.0)

                if step == 3 or step == 6:
                    env.rest(2.0)

            elif target is not None and candidate is None:
                # No valid grasp found — skip to next target
                target = None
                if len(targets) > 1:
                    targets = targets[1:]
                else:
                    targets = []

            env.controller.step()


if __name__ == "__main__":
    main()
