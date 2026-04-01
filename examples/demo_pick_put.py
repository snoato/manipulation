"""Demo: pick-and-put loop within a fixed randomized scene.

One scene is sampled at startup. The robot then picks a random cylinder
and places it at a random empty grid cell, repeating indefinitely within
that same scene (cylinder positions update after each place).

Usage::

    cd examples
    python demo_pick_put.py
"""

import numpy as np

from tampanda import ControllerStatus, FrankaEnvironment, RRTStar, SCENE_SYMBOLIC
from tampanda.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from tampanda.symbolic.domains.tabletop.grid_domain import GridDomain
from tampanda.symbolic.domains.tabletop.state_manager import StateManager

_XML            = SCENE_SYMBOLIC
_GRID_WIDTH     = 0.4
_GRID_HEIGHT    = 0.3
_CELL_SIZE      = 0.04
_GRID_OFFSET_X  = 0.05
_GRID_OFFSET_Y  = 0.25
_N_CYLINDERS    = 4
_RRT_ITERS      = 2000
_RRT_ITERS_FINE = 1000   # descent/place phases

# Controller advance delta — tighter than default (0.1) to avoid overshoot.
_CTRL_DELTA = 0.05

# Arm-settle velocity threshold (rad/s, sum over all joints).
_VEL_THRESHOLD = 0.02

# Collision-check intermediate points per RRT edge.
# 5 = fast (original), 10 = tighter (catches narrow gaps, ~2× slower planning).
_COLLISION_CHECK_STEPS = 10


def _pick_candidate(candidates):
    """Prefer FRONT approach for tall thin cylinders."""
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _settle(env, threshold: float = _VEL_THRESHOLD, max_wait: float = 2.0):
    """Step physics until arm velocity drops below threshold (or max_wait expires)."""
    steps = int(max_wait / env.rate.dt)
    for _ in range(steps):
        env.step()
        if np.linalg.norm(env.data.qvel[:7]) < threshold:
            return


def _abort_hold(env):
    """Recovery after a failure while holding a cylinder.

    Detaches the object, opens the gripper so the cylinder falls free,
    clears collision exceptions, and waits for everything to settle.
    """
    env.detach_object()
    env.controller.open_gripper()
    env.clear_collision_exceptions()
    env.rest(1.5)   # let the released cylinder settle on the table


def main():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    planner = RRTStar(env)
    planner.step_size             = 0.2
    planner.goal_sample_rate      = 0.2
    planner.collision_check_steps = _COLLISION_CHECK_STEPS

    grid = GridDomain(
        model=env.model,
        cell_size=_CELL_SIZE,
        working_area=(_GRID_WIDTH, _GRID_HEIGHT),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=_GRID_OFFSET_X,
        grid_offset_y=_GRID_OFFSET_Y,
    )
    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
    home_ctrl = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])

    # --- Sample one scene and reset arm to home ---
    print("Sampling scene...")
    state_manager.sample_random_state(n_cylinders=_N_CYLINDERS)
    env.data.qpos[:8] = home_qpos
    env.data.ctrl[:8] = home_ctrl
    env.reset_velocities()

    # --- State machine ---
    # Phases: "init" | "approach" | "grasp" | "close" | "lift"
    #       | "put_approach" | "put_place" | "open"
    phase            = "init"
    target_cyl       = None
    target_cell      = None
    pick_cand        = None
    put_approach_pos = None   # EE standoff position for put
    place_ee_pos     = None   # EE position that lands cylinder exactly on cell centre
    put_quat         = None   # EE orientation for put

    with env.launch_viewer() as viewer:
        while viewer.is_running():
            dt = env.step()
            env.controller.step(delta=_CTRL_DELTA)

            if env.controller.get_status() != ControllerStatus.IDLE:
                continue

            # ----------------------------------------------------------------
            # INIT: ground current scene state, choose cylinder + target cell
            # ----------------------------------------------------------------
            if phase == "init":
                state = state_manager.ground_state()
                cylinders = list(state["cylinders"].keys())
                if not cylinders:
                    print("No cylinders in scene — check setup")
                    continue

                target_cyl = np.random.choice(cylinders)

                occupied_cells = {c for cells in state["cylinders"].values() for c in cells}
                empty_cells    = [c for c in grid.cells if c not in occupied_cells]
                if not empty_cells:
                    print("No empty cells — all cells occupied")
                    continue

                target_cell = np.random.choice(empty_cells)
                print(f"\nPick: {target_cyl}   →   Put: {target_cell}")
                phase = "approach"

            # ----------------------------------------------------------------
            # APPROACH: plan RRT to standoff above the cylinder
            # ----------------------------------------------------------------
            elif phase == "approach":
                cyl_pos   = env.get_object_position(target_cyl)
                half_size = env.get_object_half_size(target_cyl)
                cyl_quat  = env.get_object_orientation(target_cyl)
                candidates = grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
                pick_cand  = _pick_candidate(candidates)

                if pick_cand is None:
                    print("  No grasp candidate — retrying")
                    phase = "init"
                    continue

                env.add_collision_exception(target_cyl)
                print(f"  Planning pick approach...")
                path = planner.plan_to_pose(
                    pick_cand.approach_pos, pick_cand.grasp_quat,
                    dt=dt, max_iterations=_RRT_ITERS,
                )
                if path is None:
                    print("  Approach planning failed — retrying")
                    env.clear_collision_exceptions()
                    phase = "init"
                    continue
                env.execute_path(path, planner, step_size=0.01)
                phase = "grasp"

            # ----------------------------------------------------------------
            # GRASP: descend to contact pose (slow, fine step)
            # ----------------------------------------------------------------
            elif phase == "grasp":
                print(f"  Planning grasp descent...")
                path = planner.plan_to_pose(
                    pick_cand.grasp_pos, pick_cand.grasp_quat,
                    dt=dt, max_iterations=_RRT_ITERS_FINE,
                )
                if path is None:
                    print("  Grasp planning failed — retrying")
                    env.clear_collision_exceptions()
                    phase = "init"
                    continue
                env.execute_path(path, planner, step_size=0.005)
                phase = "close"

            # ----------------------------------------------------------------
            # CLOSE: wait for arm to settle, then close gripper + attach
            # ----------------------------------------------------------------
            elif phase == "close":
                _settle(env)
                print(f"  Closing gripper, attaching {target_cyl}...")
                env.controller.close_gripper()
                env.attach_object_to_ee(target_cyl)
                env.rest(1.0)
                phase = "lift"

            # ----------------------------------------------------------------
            # LIFT: raise cylinder clear of other objects
            # ----------------------------------------------------------------
            elif phase == "lift":
                print(f"  Planning lift...")
                path = planner.plan_to_pose(
                    pick_cand.lift_pos, pick_cand.grasp_quat,
                    dt=dt, max_iterations=_RRT_ITERS,
                )
                if path is None:
                    print("  Lift planning failed — releasing and resetting")
                    _abort_hold(env)
                    phase = "init"
                    continue
                env.execute_path(path, planner, step_size=0.01)
                phase = "put_approach"

            # ----------------------------------------------------------------
            # PUT APPROACH: plan RRT to standoff above target cell.
            #
            # We compute the EE position precisely from the stored kinematic
            # attachment offset so the cylinder lands exactly on the cell centre:
            #   cylinder_world = ee_pos + R_ee @ rel_pos
            #   → place_ee_pos = target_pos - R_put @ rel_pos
            # ----------------------------------------------------------------
            elif phase == "put_approach":
                cyl_idx = int(target_cyl.split("_")[1])
                radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
                cx, cy = grid.cells[target_cell]["center"]
                cyl_z      = grid.table_height + half_height + 0.002
                target_pos = np.array([cx, cy, cyl_z])
                half_size  = np.array([radius, radius, half_height])

                # Choose best grasp orientation for the target cell.
                candidates = grasp_planner.generate_candidates(target_pos, half_size)
                put_cand_raw = _pick_candidate(candidates)

                if put_cand_raw is None:
                    print("  No place candidate — releasing and resetting")
                    _abort_hold(env)
                    phase = "init"
                    continue

                # Precise EE position: use the actual stored relative offset
                # instead of GraspPlanner's contact-offset approximation.
                put_quat    = put_cand_raw.grasp_quat
                R_put       = quat_to_rotmat(put_quat)
                rel_pos     = env._attached["rel_pos"]   # cylinder centre in EE frame
                hand_z      = R_put[:, 2]
                place_ee_pos     = target_pos - R_put @ rel_pos
                put_approach_pos = place_ee_pos - hand_z * grasp_planner.approach_dist

                print(f"  Planning move to above {target_cell}...")
                path = planner.plan_to_pose(
                    put_approach_pos, put_quat,
                    dt=dt, max_iterations=_RRT_ITERS,
                )
                if path is None:
                    print("  Put approach planning failed — releasing and resetting")
                    _abort_hold(env)
                    phase = "init"
                    continue
                env.execute_path(path, planner, step_size=0.01)
                phase = "put_place"

            # ----------------------------------------------------------------
            # PUT PLACE: slow descent to exact place position
            # ----------------------------------------------------------------
            elif phase == "put_place":
                print(f"  Planning place descent...")
                path = planner.plan_to_pose(
                    place_ee_pos, put_quat,
                    dt=dt, max_iterations=_RRT_ITERS_FINE,
                )
                if path is None:
                    print("  Place descent planning failed — releasing and resetting")
                    _abort_hold(env)
                    phase = "init"
                    continue
                env.execute_path(path, planner, step_size=0.005)
                phase = "open"

            # ----------------------------------------------------------------
            # OPEN: wait for arm to settle, then detach + open gripper
            # ----------------------------------------------------------------
            elif phase == "open":
                _settle(env)
                print(f"  Placing {target_cyl} at {target_cell} — opening gripper...")
                env.detach_object()
                env.controller.open_gripper()
                env.clear_collision_exceptions()
                env.rest(1.5)
                print(f"  Done.")
                phase = "init"


if __name__ == "__main__":
    main()
