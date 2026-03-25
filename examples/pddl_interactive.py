"""Interactive PDDL-driven manipulation demo.

Loads a PDDL domain + problem file, shows the scene in the MuJoCo viewer,
and prompts the user to specify actions one at a time.  Each action is first
validated through ActionFeasibilityChecker (the same API used by external
planners), then executed in the simulation.

The feasibility check runs on a separate headless environment so it never
corrupts the main simulation state.

Usage::

    cd examples
    python pddl_interactive.py <domain.pddl> <problem.pddl>

Commands at the interactive prompt::

    pick <cylinder_name>            pick up a cylinder
    put  <cylinder_name> <cell_id>  place the held cylinder at a grid cell
    drop                            drop the held cylinder in place
    state                           reprint the current symbolic state
    quit | q                        exit

Example session::

    > pick cylinder_10
    > put cylinder_10 cell_2_3
    > quit
"""

import re
import argparse
from pathlib import Path

import numpy as np
import mujoco

from manipulation import FrankaEnvironment, RRTStar, SCENE_SYMBOLIC, ControllerStatus
from manipulation.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from manipulation.symbolic.domains.tabletop import GridDomain, StateManager
from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker

# ── Scene configuration ────────────────────────────────────────────────────
_XML           = SCENE_SYMBOLIC
_GRID_WIDTH    = 0.4
_GRID_HEIGHT   = 0.3
_CELL_SIZE     = 0.04
_GRID_OFFSET_X = 0.05
_GRID_OFFSET_Y = 0.25

# ── Motion-planner settings ────────────────────────────────────────────────
_RRT_ITERS      = 2000
_RRT_ITERS_FINE = 1000
_CTRL_DELTA     = 0.05
_VEL_THRESHOLD  = 0.02
_COL_STEPS      = 10

_HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
_HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])


# ── Helpers ────────────────────────────────────────────────────────────────

def _pick_candidate(candidates):
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


def _settle(env, threshold=_VEL_THRESHOLD, max_wait=2.0):
    steps = int(max_wait / env.rate.dt)
    for _ in range(steps):
        env.step()
        env.controller.step(delta=_CTRL_DELTA)
        if np.linalg.norm(env.data.qvel[:7]) < threshold:
            return


def _wait_idle(env):
    """Step until the controller is IDLE, then settle briefly."""
    for _ in range(8000):
        env.controller.step(delta=_CTRL_DELTA)
        env.step()
        if env.controller.get_status() == ControllerStatus.IDLE:
            for _ in range(50):
                env.step()
            return


def _abort_hold(env):
    env.detach_object()
    env.controller.open_gripper()
    env.clear_collision_exceptions()
    env.rest(1.5)


def _print_state(state):
    print("\n── Symbolic state ──────────────────────────────────────")
    holding = state.get("holding")
    print(f"  Holding : {holding if holding else '(none)'}")
    cylinders = state.get("cylinders", {})
    print(f"  On table: {len(cylinders)} cylinder(s)")
    for cyl, cells in sorted(cylinders.items()):
        print(f"    {cyl:12s}  →  {', '.join(cells)}")
    print("────────────────────────────────────────────────────────\n")


def _print_timing(ok, timing):
    tag = "FEASIBLE" if ok else f"INFEASIBLE  reason={timing['reason']}"
    print(f"  [{tag}]  total={timing['total_ms']:.0f} ms", end="")
    sub = {k: v for k, v in timing.items()
           if k.endswith("_ms") and k != "total_ms" and isinstance(v, float)}
    if sub:
        parts = "  " + "  ".join(f"{k}={v:.0f}" for k, v in sub.items())
        print(parts, end="")
    print()


def _extract_init_section(pddl_text):
    m = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', pddl_text, re.DOTALL | re.IGNORECASE)
    if m is None:
        # Fallback: grab everything inside (:init ...)
        m = re.search(r'\(:init\s+(.*?)\)', pddl_text, re.DOTALL | re.IGNORECASE)
    if m is None:
        raise ValueError("Could not find (:init ...) section in PDDL problem file")
    return m.group(1)


def _make_env_and_grid():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    grid = GridDomain(
        model=env.model,
        cell_size=_CELL_SIZE,
        working_area=(_GRID_WIDTH, _GRID_HEIGHT),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=_GRID_OFFSET_X,
        grid_offset_y=_GRID_OFFSET_Y,
    )
    return env, grid


# ── Action executors ───────────────────────────────────────────────────────

def _execute_pick(env, planner, grasp_planner, cyl_name):
    dt = env.model.opt.timestep
    cyl_pos   = env.get_object_position(cyl_name)
    half_size = env.get_object_half_size(cyl_name)
    cyl_quat  = env.get_object_orientation(cyl_name)
    candidates = grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
    cand = _pick_candidate(candidates)
    if cand is None:
        print("  No grasp candidate.")
        return False

    env.add_collision_exception(cyl_name)

    print("  Planning approach ...")
    path = planner.plan_to_pose(cand.approach_pos, cand.grasp_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        print("  Approach planning failed.")
        env.clear_collision_exceptions()
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    print("  Planning grasp descent ...")
    path = planner.plan_to_pose(cand.grasp_pos, cand.grasp_quat,
                                dt=dt, max_iterations=_RRT_ITERS_FINE)
    if path is None:
        print("  Grasp planning failed.")
        env.clear_collision_exceptions()
        return False
    env.execute_path(path, planner, step_size=0.005)
    _wait_idle(env)

    _settle(env)
    print("  Closing gripper ...")
    env.controller.close_gripper()
    env.attach_object_to_ee(cyl_name)
    env.rest(1.0)

    print("  Planning lift ...")
    path = planner.plan_to_pose(cand.lift_pos, cand.grasp_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        print("  Lift failed — releasing.")
        _abort_hold(env)
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    env.clear_collision_exceptions()
    return True


def _execute_put(env, planner, grasp_planner, grid, cyl_name, cell_id):
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
        print("  No place candidate — releasing.")
        _abort_hold(env)
        return False

    put_quat     = cand.grasp_quat
    R_put        = quat_to_rotmat(put_quat)
    rel_pos      = env._attached["rel_pos"]
    place_ee_pos = target_pos - R_put @ rel_pos
    approach_pos = place_ee_pos - R_put[:, 2] * grasp_planner.approach_dist

    print(f"  Planning move above {cell_id} ...")
    path = planner.plan_to_pose(approach_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        print("  Put approach failed — releasing.")
        _abort_hold(env)
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    print("  Planning place descent ...")
    path = planner.plan_to_pose(place_ee_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS_FINE)
    if path is None:
        print("  Place descent failed — releasing.")
        _abort_hold(env)
        return False
    env.execute_path(path, planner, step_size=0.005)
    _wait_idle(env)

    _settle(env)
    print("  Opening gripper ...")
    env.detach_object()
    env.controller.open_gripper()
    env.rest(1.5)
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive PDDL-driven manipulation demo")
    parser.add_argument("domain",  help="Path to PDDL domain file (not parsed, for reference)")
    parser.add_argument("problem", help="Path to PDDL problem file")
    args = parser.parse_args()

    problem_text = Path(args.problem).read_text()
    init_section = _extract_init_section(problem_text)

    # Warn if problem contains a holding predicate (not yet handled)
    if re.search(r'\(holding\s+\w+\)', init_section, re.IGNORECASE):
        print("Warning: (:init ...) contains (holding ...) — gripper state "
              "will be treated as empty for now.")

    # ── Main env: viewer + execution ───────────────────────────────────────
    env, grid = _make_env_and_grid()
    planner = RRTStar(env)
    planner.step_size             = 0.2
    planner.goal_sample_rate      = 0.2
    planner.collision_check_steps = _COL_STEPS

    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=grid.table_height)

    # Load initial scene from PDDL
    state_manager.init_from_pddl_state(init_section)
    env.data.qpos[:8] = _HOME_QPOS
    env.data.ctrl[:8] = _HOME_CTRL
    env.reset_velocities()
    mujoco.mj_forward(env.model, env.data)

    # ── Check env: headless feasibility checker ────────────────────────────
    check_env, check_grid = _make_env_and_grid()
    check_sm      = StateManager(check_grid, check_env)
    check_planner = RRTStar(check_env)
    check_gp      = GraspPlanner(table_z=check_grid.table_height)
    checker = ActionFeasibilityChecker(
        check_env, check_planner, check_sm, check_gp,
        max_iterations=1000,
    )

    print(f"\nLoaded: {Path(args.problem).name}")
    print("Opening viewer — type actions in this terminal.\n")

    with env.launch_viewer() as viewer:
        # Let scene settle visually
        env.rest(0.5)

        while viewer.is_running():
            # Ground the current symbolic state from the simulation
            state = state_manager.ground_state()
            _print_state(state)

            # ── User prompt ────────────────────────────────────────────────
            try:
                cmd = input("Action [pick <cyl> | put <cyl> <cell> | drop | state | quit]: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not viewer.is_running():
                break

            parts = cmd.split()
            if not parts:
                # Keep viewer alive while user thinks
                for _ in range(100):
                    env.step()
                    env.controller.step(delta=_CTRL_DELTA)
                continue

            action = parts[0].lower()

            # ── quit ───────────────────────────────────────────────────────
            if action in ("quit", "q"):
                break

            # ── state ──────────────────────────────────────────────────────
            elif action == "state":
                continue  # state is printed at the top of every loop

            # ── pick ───────────────────────────────────────────────────────
            elif action == "pick":
                if len(parts) != 2:
                    print("  Usage: pick <cylinder_name>")
                    continue
                cyl = parts[1]
                if cyl not in state["cylinders"]:
                    print(f"  '{cyl}' is not on the table "
                          f"(available: {list(state['cylinders'])})")
                    continue
                if state["holding"] is not None:
                    print(f"  Gripper already holds '{state['holding']}' — put it down first.")
                    continue

                print(f"\nChecking feasibility: pick({cyl}) ...")
                ok, timing = checker.check("pick", state, cylinder_name=cyl)
                _print_timing(ok, timing)
                if not ok:
                    continue

                print(f"Executing pick({cyl}) ...")
                if _execute_pick(env, planner, grasp_planner, cyl):
                    state_manager.gripper_holding = cyl
                    print(f"  pick({cyl}) done.\n")
                else:
                    print(f"  pick({cyl}) execution failed.\n")

            # ── put ────────────────────────────────────────────────────────
            elif action == "put":
                if len(parts) != 3:
                    print("  Usage: put <cylinder_name> <cell_id>")
                    continue
                cyl, cell_id = parts[1], parts[2]
                if state["holding"] != cyl:
                    print(f"  Not holding '{cyl}' (holding: {state['holding']}).")
                    continue
                if cell_id not in grid.cells:
                    print(f"  '{cell_id}' is not a valid cell.")
                    continue

                print(f"\nChecking feasibility: put({cyl}, {cell_id}) ...")
                ok, timing = checker.check("put", state,
                                           cylinder_name=cyl, target_cell=cell_id)
                _print_timing(ok, timing)
                if not ok:
                    continue

                print(f"Executing put({cyl}, {cell_id}) ...")
                if _execute_put(env, planner, grasp_planner, grid, cyl, cell_id):
                    state_manager.gripper_holding = None
                    print(f"  put({cyl}, {cell_id}) done.\n")
                else:
                    print(f"  put({cyl}, {cell_id}) execution failed.\n")

            # ── drop ───────────────────────────────────────────────────────
            elif action == "drop":
                ok, timing = checker.check("drop", state)
                _print_timing(ok, timing)
                if not ok:
                    continue
                print("Dropping ...")
                _abort_hold(env)
                state_manager.gripper_holding = None
                print("  drop done.\n")

            else:
                print(f"  Unknown action '{action}'. "
                      "Use: pick | put | drop | state | quit")


if __name__ == "__main__":
    main()
