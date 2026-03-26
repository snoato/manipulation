"""Action feasibility checker for the tabletop symbolic domain.

Given a symbolic state and an action, determines whether the action is
physically executable by the robot — using IK to solve the target poses and
RRT* to verify collision-free paths exist.

The checker runs fully headless (rate limiter bypassed) for speed.

Usage::

    from manipulation import FrankaEnvironment, RRTStar
    from manipulation.planners.grasp_planner import GraspPlanner
    from manipulation.symbolic.domains.tabletop import GridDomain, StateManager
    from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker

    env     = FrankaEnvironment(xml_path)
    planner = RRTStar(env)
    grid    = GridDomain(model=env.model, ...)
    sm      = StateManager(grid, env)
    gp      = GraspPlanner(table_z=grid.table_height)

    checker = ActionFeasibilityChecker(env, planner, sm, gp)

    state  = sm.ground_state()   # or build manually
    ok, t  = checker.check("pick", state, cylinder_name="cylinder_3")
    print(ok, t)
"""

import time

import mujoco
import numpy as np

from manipulation.planners.grasp_planner import GraspPlanner, GraspType, quat_to_rotmat
from manipulation.symbolic.domains.tabletop.state_manager import StateManager


class ActionFeasibilityChecker:
    """Checks whether pick / drop actions are physically feasible.

    Feasibility for **pick** is defined as:
      1. GraspPlanner produces at least one candidate.
      2. IK converges for the approach pose.
      3. RRT* finds a collision-free path to the approach pose.
      4. IK converges for the grasp-contact pose.
      5. RRT* finds a collision-free path from approach to grasp-contact.

    Feasibility for **drop** is defined as: the gripper is currently holding
    something.  The PDDL ``drop`` action discards the held cylinder at no
    specific location, so no path planning is required.

    Args:
        env:           FrankaEnvironment instance.
        planner:       RRTStar instance.
        state_manager: StateManager instance.
        grasp_planner: GraspPlanner instance.
        max_iterations: RRT* iteration budget per planning call.
        settle_steps:   Headless steps to run after state initialisation.
    """

    def __init__(self, env, planner, state_manager, grasp_planner: GraspPlanner,
                 max_iterations: int = 1000, settle_steps: int = 60,
                 ik_max_iters: int = 100, ik_pos_threshold: float = 0.005,
                 feasibility_planner=None):
        self._env           = env
        self._planner       = planner
        # Use a dedicated feasibility planner (e.g. FeasibilityRRT) for all
        # plan / plan_to_pose calls inside the checker.  Falls back to the
        # main planner if none is provided (backward-compatible default).
        self._feas_planner  = feasibility_planner if feasibility_planner is not None else planner
        self._state_manager = state_manager
        self._grasp_planner = grasp_planner
        self.max_iterations = max_iterations
        self._settle_steps  = settle_steps

        # Tuned for feasibility checking (benchmarked: fastest zero-false-negative combo)
        env.ik.max_iters      = ik_max_iters
        env.ik.pos_threshold  = ik_pos_threshold

        # Bypass the rate limiter — checker is always headless
        _dt = env.model.opt.timestep
        def _fast_step():
            if env._attached is not None:
                env._apply_attachment()
            mujoco.mj_step(env.model, env.data)
            env.sim_time += _dt
            return _dt
        env.step = _fast_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, action_name: str, state_dict: dict,
              cylinder_name: str | None = None,
              target_cell: str | None = None) -> tuple[bool, dict]:
        """Check feasibility of an action in a given symbolic state.

        Args:
            action_name:   ``"pick"``, ``"drop"``, or ``"put"``.
            state_dict:    Symbolic state dict (same format as
                           ``StateManager.ground_state()``).
            cylinder_name: Required for ``"pick"`` and ``"put"``.
            target_cell:   Required for ``"put"`` — cell ID to place the
                           cylinder into (e.g. ``"cell_2_3"``).

        Returns:
            ``(feasible, timing)`` where *timing* is a dict with per-phase
            wall-clock times in milliseconds and a ``"reason"`` string.
        """
        if action_name == "pick":
            if cylinder_name is None:
                raise ValueError("cylinder_name is required for action 'pick'")
            return self._check_pick(state_dict, cylinder_name)
        elif action_name == "drop":
            return self._check_drop(state_dict)
        elif action_name == "put":
            if cylinder_name is None:
                raise ValueError("cylinder_name is required for action 'put'")
            if target_cell is None:
                raise ValueError("target_cell is required for action 'put'")
            return self._check_put(state_dict, cylinder_name, target_cell)
        else:
            raise ValueError(f"Unknown action '{action_name}'. Expected 'pick', 'drop', or 'put'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self, state_dict: dict):
        """Load symbolic state into sim and settle."""
        self._state_manager.set_from_grounded_state(state_dict)
        self._env.controller.stop()
        mujoco.mj_forward(self._env.model, self._env.data)
        self._env.ik.update_configuration(self._env.data.qpos)
        for _ in range(self._settle_steps):
            self._env.step()

    def _wait_idle(self, max_steps: int = 5000) -> bool:
        return self._env.wait_idle(max_steps=max_steps, settle_steps=30)

    @staticmethod
    def _pick_candidate(candidates):
        """Prefer FRONT (side) approach for tall thin cylinders.

        Top-down grasps place the contact point at the body centroid, which
        for tall cylinders lies deep inside the body — causing the gripper to
        crash.  A front approach contacts at the correct height and grips
        around the cylinder properly.
        """
        front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
        return front if front is not None else (candidates[0] if candidates else None)


    # ------------------------------------------------------------------
    # Pick feasibility
    # ------------------------------------------------------------------

    def _check_pick(self, state_dict: dict, cylinder_name: str) -> tuple[bool, dict]:
        t_total = time.perf_counter()
        timing: dict = {}

        # 1. Initialise simulation state
        self._init_state(state_dict)

        dt = self._env.model.opt.timestep

        # 2. Compute grasp candidates
        cyl_pos  = self._env.get_object_position(cylinder_name)
        half_size = self._env.get_object_half_size(cylinder_name)
        cyl_quat = self._env.get_object_orientation(cylinder_name)
        candidates = self._grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
        candidate  = self._pick_candidate(candidates)

        if candidate is None:
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "no_grasp_candidate"}

        # Store for caller (e.g. visualization)
        timing["grasp_pos"]  = candidate.grasp_pos
        timing["grasp_quat"] = candidate.grasp_quat

        # 3. IK — approach pose
        t0 = time.perf_counter()
        self._env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
        ik_ok = self._env.ik.converge_ik(dt)
        timing["ik_approach_ms"] = (time.perf_counter() - t0) * 1000

        if not ik_ok:
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "ik_approach_fail"}

        # 4. RRT* — approach path
        t0 = time.perf_counter()
        path = self._feas_planner.plan_to_pose(
            candidate.approach_pos, candidate.grasp_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_approach_ms"] = (time.perf_counter() - t0) * 1000

        if path is None:
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "rrt_approach_fail"}

        # Execute approach so grasp planning starts from the correct config
        self._env.execute_path(path, self._feas_planner)
        self._wait_idle()

        # 5. IK — grasp-contact pose (cylinder is exception: gripper may touch it)
        self._env.add_collision_exception(cylinder_name)
        t0 = time.perf_counter()
        self._env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
        ik_ok = self._env.ik.converge_ik(dt)
        timing["ik_grasp_ms"] = (time.perf_counter() - t0) * 1000

        if not ik_ok:
            self._env.clear_collision_exceptions()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "ik_grasp_fail"}

        # 6. RRT* — grasp-contact path
        t0 = time.perf_counter()
        path = self._feas_planner.plan_to_pose(
            candidate.grasp_pos, candidate.grasp_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_grasp_ms"] = (time.perf_counter() - t0) * 1000

        if path is None:
            self._env.clear_collision_exceptions()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "rrt_grasp_fail"}

        # Execute grasp so transport check starts from the real post-grasp config.
        # Keep the cylinder exception active: fingers touch the cylinder at the
        # grasp position, so is_collision_free(start) would fail without it.
        self._env.execute_path(path, self._feas_planner)
        self._wait_idle()
        # (exception still active)

        # 7. IK — transport pose (seed from HOME, same as execution)
        t0 = time.perf_counter()
        home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
        home_full = self._env.data.qpos.copy()
        home_full[:8] = home_qpos
        self._env.ik.update_configuration(home_full)
        transport_pos, transport_quat = self._state_manager.get_transport_pose()
        self._env.ik.set_target_position(transport_pos, transport_quat)
        ik_ok = self._env.ik.converge_ik(dt)
        timing["ik_transport_ms"] = (time.perf_counter() - t0) * 1000

        if not ik_ok:
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "ik_transport_fail"}

        transport_q = self._env.ik.configuration.q[:7].copy()

        # 8. RRT* — transport path (link7 excluded, same as execution).
        # Retry once on failure: transport RRT has high variance and a path
        # that exists may not be found within the per-check iteration budget.
        link7_id = mujoco.mj_name2id(
            self._env.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
        )
        saved_ids  = self._env._collision_body_ids
        saved_held = self._env._collision_held_body
        self._env._collision_body_ids = saved_ids - {link7_id}
        self._env._collision_held_body = None
        t0 = time.perf_counter()
        path = self._feas_planner.plan(
            self._env.data.qpos[:7], transport_q,
            max_iterations=self.max_iterations,
        )
        if path is None:
            path = self._feas_planner.plan(
                self._env.data.qpos[:7], transport_q,
                max_iterations=self.max_iterations,
            )
        timing["rrt_transport_ms"] = (time.perf_counter() - t0) * 1000
        self._env._collision_body_ids = saved_ids
        self._env._collision_held_body = saved_held
        self._env.clear_collision_exceptions()

        feasible = path is not None
        timing["total_ms"] = (time.perf_counter() - t_total) * 1000
        timing["reason"]   = "ok" if feasible else "rrt_transport_fail"

        return feasible, timing

    # ------------------------------------------------------------------
    # Grid reachability verification
    # ------------------------------------------------------------------

    def verify_reachability(self, test_cylinder: str = "cylinder_0",
                            verbose: bool = True) -> dict[str, bool]:
        """Check which grid cells are kinematically reachable.

        Places *test_cylinder* at each cell centre one at a time and checks
        whether IK converges for the best grasp candidate.  RRT* is not run —
        this is purely a kinematic reachability check, not a path-planning one,
        so it is fast (one IK solve per cell).

        Args:
            test_cylinder: Name of a cylinder body present in the scene to use
                           as the test object (default: ``"cylinder_0"``).
            verbose:       Print a summary row per cell.

        Returns:
            Dict mapping ``cell_id → reachable (bool)``.
        """
        grid   = self._state_manager.grid
        n_cells = len(grid.cells)
        results: dict[str, bool] = {}

        if verbose:
            print(f"Verifying reachability for {n_cells} cells "
                  f"(IK only, test object: {test_cylinder})...")

        dt = self._env.model.opt.timestep

        for cell_id, cell_info in grid.cells.items():
            state = {
                "cylinders":    {test_cylinder: [cell_id]},
                "gripper_empty": True,
                "holding":      None,
            }
            self._init_state(state)

            cyl_pos   = self._env.get_object_position(test_cylinder)
            half_size = self._env.get_object_half_size(test_cylinder)
            cyl_quat  = self._env.get_object_orientation(test_cylinder)
            candidates = self._grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
            candidate  = self._pick_candidate(candidates)

            if candidate is None:
                reachable = False
            else:
                self._env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
                reachable = self._env.ik.converge_ik(dt)

            results[cell_id] = reachable

            if verbose:
                cx, cy = cell_info["center"]
                tag = "OK  " if reachable else "FAIL"
                print(f"  {cell_id:<14} ({cx:.3f}, {cy:.3f})  {tag}")

        n_ok = sum(results.values())
        if verbose:
            print(f"\nResult: {n_ok}/{n_cells} cells reachable.")

        return results

    # ------------------------------------------------------------------
    # Put feasibility
    # ------------------------------------------------------------------

    def _check_put(self, state_dict: dict, cylinder_name: str,
                   target_cell: str) -> tuple[bool, dict]:
        """Check feasibility of placing *cylinder_name* (currently held) at *target_cell*.

        Feasibility for **put** is defined as:
          1. GraspPlanner produces at least one candidate for the target position.
          2. IK converges for the pre-place standoff pose above the target cell.
          3. RRT* finds a collision-free path to the standoff.
          4. IK converges for the place-contact pose (gripper descended to cell).
          5. RRT* finds a collision-free path from standoff to place contact.

        The held cylinder is hidden and added as a collision exception — it
        moves with the gripper and will not block the arm's path.
        """
        t_total = time.perf_counter()
        timing: dict = {}

        # 1. Load state.  set_from_grounded_state (called by _init_state) sees
        #    'holding' and:
        #      • places the cylinder physically at the EE
        #      • calls env.set_collision_held_body() so every subsequent
        #        is_collision_free check teleports it to EE_pos + R_ee @ rel_pos
        #    The cylinder is therefore a real collision object during RRT* —
        #    the planner will detect if the carried cylinder clips other objects.
        cylinders_on_table = {
            k: v for k, v in state_dict.get("cylinders", {}).items()
            if k != cylinder_name
        }
        put_state = {**state_dict, "cylinders": cylinders_on_table,
                     "holding": cylinder_name}
        self._init_state(put_state)

        cyl_idx = int(cylinder_name.split("_")[1])
        radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
        half_size  = np.array([radius, radius, half_height])
        dt = self._env.model.opt.timestep

        # 2. Compute target position: cell centre at cylinder resting height.
        grid = self._state_manager.grid
        cx, cy = grid.cells[target_cell]["center"]
        cyl_z      = grid.table_height + half_height + 0.002
        target_pos = np.array([cx, cy, cyl_z])

        # 3. Generate place candidates — FRONT preferred so the cylinder stays
        #    vertical (rel_pos was recorded in the FRONT EE frame at pick time).
        candidates = self._grasp_planner.generate_candidates(target_pos, half_size)
        candidate  = self._pick_candidate(candidates)

        def _cleanup():
            self._env.clear_collision_held_body()
            self._env.detach_object()

        if candidate is None:
            _cleanup()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "no_place_candidate"}

        # Compute EE positions using the stored EE-frame offset so the cylinder
        # lands exactly on the cell centre.
        put_quat     = candidate.grasp_quat
        R_put        = quat_to_rotmat(put_quat)
        rel_pos_ee   = self._env._attached["rel_pos"]
        place_ee_pos = target_pos - R_put @ rel_pos_ee
        # Approach from directly above (not from behind in Y) so the held
        # cylinder is ~12 cm above resting height during the approach, clear
        # of all table cylinders.
        approach_pos = place_ee_pos + np.array([0.0, 0.0,
                                                self._grasp_planner.approach_dist])

        timing["place_pos"]  = place_ee_pos
        timing["place_quat"] = put_quat

        # 4. IK — pre-place standoff (above target cell)
        t0 = time.perf_counter()
        self._env.ik.set_target_position(approach_pos, put_quat)
        ik_ok = self._env.ik.converge_ik(dt)
        timing["ik_approach_ms"] = (time.perf_counter() - t0) * 1000

        if not ik_ok:
            _cleanup()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "ik_approach_fail"}

        # 5. RRT* — path to pre-place standoff
        t0 = time.perf_counter()
        path = self._feas_planner.plan_to_pose(
            approach_pos, put_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_approach_ms"] = (time.perf_counter() - t0) * 1000

        if path is None:
            _cleanup()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "rrt_approach_fail"}

        # Execute standoff path so place-descent check starts from correct config.
        self._env.execute_path(path, self._feas_planner)
        self._wait_idle()

        # 6. IK — place-contact pose (descend to cylinder resting height)
        t0 = time.perf_counter()
        self._env.ik.set_target_position(place_ee_pos, put_quat)
        ik_ok = self._env.ik.converge_ik(dt)
        timing["ik_place_ms"] = (time.perf_counter() - t0) * 1000

        if not ik_ok:
            _cleanup()
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "ik_place_fail"}

        # 7. RRT* — descent to place-contact
        t0 = time.perf_counter()
        path = self._feas_planner.plan_to_pose(
            place_ee_pos, put_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_place_ms"] = (time.perf_counter() - t0) * 1000

        _cleanup()

        feasible = path is not None
        timing["total_ms"] = (time.perf_counter() - t_total) * 1000
        timing["reason"]   = "ok" if feasible else "rrt_place_fail"
        return feasible, timing

    # ------------------------------------------------------------------
    # Drop feasibility
    # ------------------------------------------------------------------

    def _check_drop(self, state_dict: dict) -> tuple[bool, dict]:
        """Drop is feasible whenever the gripper is holding something.

        The PDDL ``drop`` action has no placement target — the held cylinder
        is simply discarded.  No path planning is needed.
        """
        holding  = state_dict.get("holding")
        feasible = holding is not None
        timing   = {
            "total_ms": 0.0,
            "reason": "ok" if feasible else "not_holding",
        }
        return feasible, timing
