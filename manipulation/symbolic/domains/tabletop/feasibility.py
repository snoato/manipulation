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

from manipulation.planners.grasp_planner import GraspPlanner, GraspType


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
                 ik_max_iters: int = 100, ik_pos_threshold: float = 0.005):
        self._env           = env
        self._planner       = planner
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
            mujoco.mj_step(env.model, env.data)
            env.sim_time += _dt
            return _dt
        env.step = _fast_step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, action_name: str, state_dict: dict,
              cylinder_name: str | None = None) -> tuple[bool, dict]:
        """Check feasibility of an action in a given symbolic state.

        Args:
            action_name:   ``"pick"`` or ``"drop"``.
            state_dict:    Symbolic state dict (same format as
                           ``StateManager.ground_state()``).
            cylinder_name: Required for ``"pick"``.

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
        else:
            raise ValueError(f"Unknown action '{action_name}'. Expected 'pick' or 'drop'.")

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
        path = self._planner.plan_to_pose(
            candidate.approach_pos, candidate.grasp_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_approach_ms"] = (time.perf_counter() - t0) * 1000

        if path is None:
            timing["total_ms"] = (time.perf_counter() - t_total) * 1000
            return False, {**timing, "reason": "rrt_approach_fail"}

        # Execute approach so grasp planning starts from the correct config
        self._env.execute_path(path, self._planner)
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
        path = self._planner.plan_to_pose(
            candidate.grasp_pos, candidate.grasp_quat,
            dt=dt, max_iterations=self.max_iterations,
        )
        timing["rrt_grasp_ms"] = (time.perf_counter() - t0) * 1000

        self._env.clear_collision_exceptions()

        feasible = path is not None
        timing["total_ms"] = (time.perf_counter() - t_total) * 1000
        timing["reason"]   = "ok" if feasible else "rrt_grasp_fail"

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
