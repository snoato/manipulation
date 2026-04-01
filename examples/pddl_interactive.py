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
    python pddl_interactive.py <domain.pddl> <problem.pddl> --plan <plan.pddl.plan>

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

try:
    import readline as _rl
    # libedit (macOS default) uses a different bind syntax
    if "libedit" in _rl.__doc__ or "":
        _rl.parse_and_bind("bind ^I rl_complete")
    else:
        _rl.parse_and_bind("tab: complete")
    _rl.set_history_length(500)
    _READLINE = True
except ImportError:
    _READLINE = False


class _Completer:
    """Tab-completer for the interactive prompt.

    Completes:
      - action keywords (first token)
      - cylinder names after ``pick`` / ``put``
      - cell IDs after ``put <cyl>``
    """

    _ACTIONS = ["pick", "put", "drop", "state", "quit"]

    def __init__(self, grid):
        self._grid = grid
        self._state: dict = {}

    def update(self, state: dict):
        self._state = state

    def complete(self, text: str, idx: int):
        buf   = _rl.get_line_buffer() if _READLINE else ""
        parts = buf.lstrip().split()
        # Are we still completing the first token?
        first_token_done = bool(parts) and buf.endswith(" ")

        candidates: list[str] = []

        if not parts or (len(parts) == 1 and not first_token_done):
            candidates = [a for a in self._ACTIONS if a.startswith(text)]

        elif parts[0] == "pick":
            if len(parts) == 1 or (len(parts) == 2 and not first_token_done):
                cyls = sorted(self._state.get("cylinders", {}).keys())
                candidates = [c for c in cyls if c.startswith(text)]

        elif parts[0] == "put":
            if len(parts) == 1 or (len(parts) == 2 and not first_token_done):
                held = self._state.get("holding")
                candidates = [held] if held and held.startswith(text) else []
            elif len(parts) == 2 or (len(parts) == 3 and not first_token_done):
                cells = sorted(self._grid.cells.keys())
                candidates = [c for c in cells if c.startswith(text)]

        return candidates[idx] if idx < len(candidates) else None

from manipulation import RRTStar, FeasibilityRRT, ControllerStatus
from manipulation.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from manipulation.planners.robust_planner import RobustPlanner
from manipulation.symbolic.domains.tabletop import GridDomain, StateManager
from manipulation.symbolic.domains.tabletop.env_builder import make_symbolic_builder
from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker

# ── Scene configuration ────────────────────────────────────────────────────
_GRID_WIDTH    = 0.4
_GRID_HEIGHT   = 0.3
_CELL_SIZE     = 0.04
_GRID_OFFSET_X = 0.05
_GRID_OFFSET_Y = 0.25

# ── Motion-planner settings ────────────────────────────────────────────────
_RRT_ITERS      = 2000
_RRT_ITERS_FINE = 1000
# FeasibilityRRT is ~4x faster per call than RRTStar, so we can afford a
# proportionally larger budget while staying within the same wall-clock time.
# 4000 iters here ≈ old RRTStar at 1000 iters in cost, but 2x the coverage
# of the execution planner — fewer false negatives on tight configurations.
_FEAS_ITERS     = 4000
_CTRL_DELTA     = 0.05
_VEL_THRESHOLD  = 0.02
_COL_STEPS      = 20

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
    return env, grid


# ── Action executors ───────────────────────────────────────────────────────

def _execute_pick(env, planner, grasp_planner, cyl_name, state_manager):
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

    print("  Moving to transport pose ...")
    transport_pos, transport_quat = state_manager.get_transport_pose()
    # Seed IK from HOME (same seed used by set_from_grounded_state in the
    # feasibility checker) so both converge to the same joint configuration.
    home_full = env.data.qpos.copy()
    home_full[:8] = _HOME_QPOS
    env.ik.update_configuration(home_full)
    env.ik.set_target_position(transport_pos, transport_quat)
    if not env.ik.converge_ik(env.model.opt.timestep):
        print("  IK for transport pose failed — releasing.")
        _abort_hold(env)
        return False
    transport_q = env.ik.configuration.q[:7]

    # Lift the cylinder straight up before transport so it clears table-height
    # neighbours.  The collision exception is still active during this short
    # move, so no false positives from the cylinder touching immediate
    # neighbours at the grasp position.  If the lift fails we skip it and
    # proceed with transport planning anyway.
    ee_pos   = env.data.site_xpos[env._ee_site_id].copy()
    lift_pos = ee_pos + np.array([0.0, 0.0, 0.12])
    lift_path = planner.plan_to_pose(lift_pos, cand.grasp_quat,
                                     dt=dt, max_iterations=_RRT_ITERS_FINE)
    if lift_path is not None:
        env.execute_path(lift_path, planner, step_size=0.005)
        _wait_idle(env)

    # During transport planning temporarily disable link7 collision and the
    # held-body teleportation.  The cylinder exception already covers
    # cylinder–cylinder contacts; _collision_held_body would re-add the held
    # cylinder at the current EE position on every is_collision_free call,
    # which can cause false positives right after grasping.
    link7_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    saved_body_ids = env._collision_body_ids
    saved_held     = env._collision_held_body
    env._collision_body_ids  = saved_body_ids - {link7_id}
    env._collision_held_body = None

    path = planner.plan(env.data.qpos[:7], transport_q, max_iterations=_RRT_ITERS)

    env._collision_body_ids  = saved_body_ids
    env._collision_held_body = saved_held

    if path is None:
        print("  Transport planning failed.")
        env.clear_collision_exceptions()
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    env.clear_collision_exceptions()
    return True


def _go_to_transport(env, planner, state_manager):
    """Move the arm to the canonical FRONT-grasp transport pose.

    Must be called after a successful pick so that the put feasibility check
    and execution both start from the same arm configuration.
    Returns True if the arm reached the transport pose, False otherwise.
    """
    transport_pos, transport_quat = state_manager.get_transport_pose()

    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(transport_pos, transport_quat)
    if not env.ik.converge_ik(env.model.opt.timestep):
        print("  IK for transport pose failed — skipping move.")
        return False

    transport_q = env.ik.configuration.q[:7]

    # At lift height the held cylinder can be level with neighbouring table
    # cylinders. Add a collision exception so that cylinder-cylinder contacts
    # at the lift position don't immediately fail is_collision_free(start).
    # We remove the exception once the arm reaches transport — from there the
    # cylinder is 35 cm above the table and subsequent put planning can
    # correctly flag cylinder-environment contacts via _collision_body_ids.
    cyl_name = env._attached["body_name"]
    env.add_collision_exception(cyl_name)

    print("  Planning move to transport pose ...")
    path = planner.plan(env.data.qpos[:7], transport_q, max_iterations=_RRT_ITERS)
    env.remove_collision_exception(cyl_name)   # clear before put planning
    if path is None:
        print("  Transport pose planning failed.")
        return False

    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)
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
    cand = _pick_candidate(candidates)   # FRONT preferred — keeps cylinder vertical
    if cand is None:
        print("  No place candidate.")
        return False

    # rel_pos is the cylinder centre in the EE frame, recorded at FRONT-pick time.
    # place_ee_pos: EE position such that cylinder lands exactly at target_pos.
    # approach_pos: 12 cm *above* place_ee_pos (not behind it in Y) so the held
    #               cylinder is ~32 cm above the table and clears all table objects.
    put_quat     = cand.grasp_quat
    R_put        = quat_to_rotmat(put_quat)
    rel_pos      = env._attached["rel_pos"]
    place_ee_pos = target_pos - R_put @ rel_pos
    approach_pos = place_ee_pos + np.array([0.0, 0.0, grasp_planner.approach_dist])

    # If the start config is in collision (transport failed, arm stuck at table
    # level) RRT will reject it immediately.  Detect this and give a clear hint.
    if not env.is_collision_free(env.data.qpos[:7]):
        print("  Put approach: start config in collision (transport failed earlier?).")
        print("  Use 'drop' to release the cylinder and try again.")
        return False

    print(f"  Planning move above {cell_id} ...")
    path = planner.plan_to_pose(approach_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS)
    if path is None:
        print("  Put approach failed — still holding.")
        return False
    env.execute_path(path, planner, step_size=0.01)
    _wait_idle(env)

    print("  Planning place descent ...")
    path = planner.plan_to_pose(place_ee_pos, put_quat,
                                dt=dt, max_iterations=_RRT_ITERS_FINE)
    if path is None:
        print("  Place descent failed — still holding.")
        return False
    env.execute_path(path, planner, step_size=0.005)
    _wait_idle(env)

    _settle(env)
    print("  Opening gripper ...")
    env.detach_object()
    env.controller.open_gripper()
    env.rest(1.5)

    # Retreat: lift back to approach height so the arm is clear for the next
    # planning step.  Add a collision exception for the just-placed cylinder
    # because the gripper is still very close to it right after release.
    print("  Pulling back ...")
    env.add_collision_exception(cyl_name)
    retreat_path = planner.plan_to_pose(approach_pos, put_quat,
                                        dt=dt, max_iterations=_RRT_ITERS_FINE)
    env.remove_collision_exception(cyl_name)
    if retreat_path is not None:
        env.execute_path(retreat_path, planner, step_size=0.01)
        _wait_idle(env)
    else:
        print("  Pullback planning failed — continuing anyway.")
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def _parse_plan_file(path: str) -> list[tuple[str, ...]]:
    """Parse a .pddl.plan file into a list of action tuples.

    Lines like ``(pick cylinder_0 cell_7_6)`` become ``('pick', 'cylinder_0')``.
    Lines like ``(put  cylinder_0 cell_2_3)`` become ``('put', 'cylinder_0', 'cell_2_3')``.
    Comment lines (starting with ``;``) and blank lines are skipped.
    """
    actions = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # Strip surrounding parens
        inner = line.strip("()")
        parts = inner.split()
        if parts[0] == "pick":
            actions.append(("pick", parts[1]))          # cell arg unused at execution
        elif parts[0] == "put":
            actions.append(("put", parts[1], parts[2]))
    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Interactive PDDL-driven manipulation demo")
    parser.add_argument("domain",  help="Path to PDDL domain file (not parsed, for reference)")
    parser.add_argument("problem", help="Path to PDDL problem file")
    parser.add_argument("--plan",  help="Path to .pddl.plan file for automatic execution")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to pause between actions in auto mode (default: 1.0)")
    parser.add_argument("--strategy", default="combined",
                        choices=RobustPlanner.STRATEGIES,
                        help="Execution planning strategy (default: combined)")
    args = parser.parse_args()

    problem_text = Path(args.problem).read_text()
    init_section = _extract_init_section(problem_text)

    # Warn if problem contains a holding predicate (not yet handled)
    if re.search(r'\(holding\s+\w+\)', init_section, re.IGNORECASE):
        print("Warning: (:init ...) contains (holding ...) — gripper state "
              "will be treated as empty for now.")

    # ── Main env: viewer + execution ───────────────────────────────────────
    env, grid = _make_env_and_grid()
    _primary = RRTStar(env)
    _primary.step_size             = 0.2
    _primary.goal_sample_rate      = 0.2
    _primary.collision_check_steps = _COL_STEPS
    if args.strategy == "baseline":
        planner = _primary
    else:
        _exec_fallback = FeasibilityRRT(env)
        _exec_fallback.step_size             = 0.2
        _exec_fallback.collision_check_steps = _COL_STEPS
        planner = RobustPlanner(
            primary  = _primary,
            fallback = _exec_fallback,
            home_q   = _HOME_QPOS[:7],
            strategy = args.strategy,
        )
    print(f"Execution strategy: {args.strategy}")

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
    check_planner.step_size             = 0.2
    check_planner.goal_sample_rate      = 0.2
    check_planner.collision_check_steps = _COL_STEPS
    check_feas_planner = FeasibilityRRT(check_env)
    check_feas_planner.step_size             = 0.2
    check_feas_planner.collision_check_steps = _COL_STEPS
    check_gp      = GraspPlanner(table_z=check_grid.table_height)
    checker = ActionFeasibilityChecker(
        check_env, check_planner, check_sm, check_gp,
        max_iterations=_FEAS_ITERS,
        feasibility_planner=check_feas_planner,
    )

    print(f"\nLoaded: {Path(args.problem).name}")

    # ── Readline: history + tab-completion ────────────────────────────────
    completer = _Completer(grid)
    if _READLINE:
        _rl.set_completer(completer.complete)
        _rl.set_completer_delims(" \t")  # only split on whitespace

    # ── Auto mode: execute a plan file non-interactively ──────────────────
    if args.plan:
        plan_actions = _parse_plan_file(args.plan)
        print(f"Auto-executing {len(plan_actions)} actions from {Path(args.plan).name}")
        n_ok = 0
        with env.launch_viewer() as viewer:
            env.rest(0.5)
            for i, action in enumerate(plan_actions):
                if not viewer.is_running():
                    break
                print(f"\n[{i+1}/{len(plan_actions)}] {' '.join(action)}")
                if action[0] == "pick":
                    cyl = action[1]
                    ok = _execute_pick(env, planner, grasp_planner, cyl, state_manager)
                    if ok:
                        state_manager.gripper_holding = cyl
                        print(f"  pick({cyl}) OK")
                        n_ok += 1
                    elif env._attached is not None:
                        state_manager.gripper_holding = cyl
                        print(f"  pick({cyl}) grasped but transport failed")
                    else:
                        print(f"  pick({cyl}) FAILED — aborting plan")
                        break
                elif action[0] == "put":
                    cyl, cell_id = action[1], action[2]
                    ok = _execute_put(env, planner, grasp_planner, grid, cyl, cell_id)
                    if ok:
                        state_manager.gripper_holding = None
                        print(f"  put({cyl}, {cell_id}) OK")
                        n_ok += 1
                    else:
                        print(f"  put({cyl}, {cell_id}) FAILED — aborting plan")
                        break
                if args.delay > 0:
                    env.rest(args.delay)
            else:
                print(f"\n{'='*50}")
                print(f"Plan COMPLETE — {n_ok}/{len(plan_actions)} actions succeeded.")
                print(f"{'='*50}")
                env.rest(3.0)   # pause so user can see final state
                return
            print(f"\n{'='*50}")
            print(f"Plan FAILED at action {n_ok+1}/{len(plan_actions)}.")
            print(f"{'='*50}")
            env.rest(3.0)
        return

    print("Opening viewer — type actions in this terminal.\n")

    with env.launch_viewer() as viewer:
        # Let scene settle visually
        env.rest(0.5)

        while viewer.is_running():
            # Ground the current symbolic state from the simulation
            state = state_manager.ground_state()
            _print_state(state)
            completer.update(state)   # refresh autocomplete candidates

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
                if _execute_pick(env, planner, grasp_planner, cyl, state_manager):
                    state_manager.gripper_holding = cyl
                    print(f"  pick({cyl}) done.\n")
                elif env._attached is not None:
                    # Grasped but failed to reach transport — still holding.
                    state_manager.gripper_holding = cyl
                    print(f"  pick({cyl}): grasped but transport failed — use put or drop.\n")
                else:
                    print(f"  pick({cyl}) failed.\n")

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
