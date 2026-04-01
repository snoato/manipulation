"""Robust execution planner: cascade of RRTStar → FeasibilityRRT → via-HOME.

When RRTStar fails within its iteration budget, the cascade tries the faster
bidirectional FeasibilityRRT (which finds paths more reliably in tight spaces),
then as a last resort decomposes the problem via the HOME configuration (a
globally well-connected waypoint).

Drop-in replacement for RRTStar at call sites that use plan() / plan_to_pose()
/ smooth_path().

Strategies
----------
``'baseline'``  — primary RRTStar only (original behavior, no overhead).
``'fallback'``  — primary → FeasibilityRRT fallback.
``'via_home'``  — primary → fallback → via-HOME decomposition.
``'retry'``     — retry primary once, then fallback → via-HOME.
``'combined'``  — retry + fallback + via-HOME (most robust).
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np


class RobustPlanner:
    """Wraps a primary planner with fallback strategies on planning failure.

    Parameters
    ----------
    primary:
        Main motion planner (RRTStar).  Its ``smooth_path`` is always used
        so paths from the fallback planner are still post-processed for quality.
    fallback:
        Secondary planner (FeasibilityRRT).  Used after primary fails.
        Must share the *same* environment instance as primary.
    home_q:
        Home configuration (7-DOF).  Used as via-waypoint when direct paths
        fail.  If None, via-HOME decomposition is disabled.
    strategy:
        One of ``'baseline'``, ``'fallback'``, ``'via_home'``, ``'retry'``,
        ``'combined'``.
    """

    STRATEGIES = ("baseline", "fallback", "via_home", "retry", "combined")

    def __init__(
        self,
        primary,
        fallback=None,
        home_q: Optional[np.ndarray] = None,
        strategy: str = "combined",
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose from {self.STRATEGIES}.")
        self.primary  = primary
        self.fallback = fallback if fallback is not None else primary
        self.home_q   = home_q
        self.strategy = strategy

        # Mirror the most-accessed attributes so callers can treat RobustPlanner
        # as a drop-in for either primary or fallback planner.
        self.env        = primary.env
        self.step_size  = primary.step_size
        self.goal_threshold = primary.goal_threshold
        self.joint_limits_low  = primary.joint_limits_low
        self.joint_limits_high = primary.joint_limits_high

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start_config: np.ndarray,
        goal_config:  np.ndarray,
        max_iterations: Optional[int] = None,
    ) -> Optional[list]:
        iters = max_iterations or self.primary.max_iterations

        # 1. Primary (RRTStar)
        n_primary = 2 if self.strategy in ("retry", "combined") else 1
        for _ in range(n_primary):
            path = self.primary.plan(start_config, goal_config, iters)
            if path is not None:
                return path

        if self.strategy == "baseline":
            return None

        # 2. Bidirectional fallback (FeasibilityRRT)
        if self.fallback is not self.primary and self.strategy in (
            "fallback", "via_home", "retry", "combined"
        ):
            path = self.fallback.plan(start_config, goal_config, iters)
            if path is not None:
                return path

        # 3. Via-HOME decomposition
        if self.home_q is not None and self.strategy in ("via_home", "combined"):
            path = self._plan_via_home(start_config, goal_config, iters)
            if path is not None:
                return path

        return None

    def plan_to_pose(
        self,
        target_pos:     np.ndarray,
        target_quat:    np.ndarray,
        dt:             float = 0.01,
        max_iterations: Optional[int] = None,
        max_ik_retries: int = 3,
    ) -> Optional[list]:
        iters = max_iterations or self.primary.max_iterations

        # 1. Primary plan_to_pose (IK + RRTStar)
        n_primary = 2 if self.strategy in ("retry", "combined") else 1
        for _ in range(n_primary):
            path = self.primary.plan_to_pose(
                target_pos, target_quat, dt, iters, max_ik_retries
            )
            if path is not None:
                return path

        if self.strategy == "baseline":
            return None

        # 2. Bidirectional fallback plan_to_pose
        if self.fallback is not self.primary and self.strategy in (
            "fallback", "via_home", "retry", "combined"
        ):
            path = self.fallback.plan_to_pose(
                target_pos, target_quat, dt, iters, max_ik_retries
            )
            if path is not None:
                return path

        # 3. Via-HOME: solve IK for goal, decompose via HOME
        if self.home_q is not None and self.strategy in ("via_home", "combined"):
            start_config = self.env.data.qpos[:7].copy()
            goal_config  = self._solve_ik(
                target_pos, target_quat, dt, max_ik_retries
            )
            if goal_config is not None:
                path = self._plan_via_home(start_config, goal_config, iters)
                if path is not None:
                    return path

        return None

    def smooth_path(self, path: list) -> list:
        """Always use the primary planner's smoother for path quality."""
        return self.primary.smooth_path(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _plan_via_home(
        self,
        start_config: np.ndarray,
        goal_config:  np.ndarray,
        max_iterations: int,
    ) -> Optional[list]:
        """Plan start→HOME→goal, using fallback planner for both legs."""
        home = self.home_q
        if home is None:
            return None
        if not self.env.is_collision_free(home):
            return None
        p1 = self.fallback.plan(start_config, home, max_iterations)
        if p1 is None:
            return None
        p2 = self.fallback.plan(home, goal_config, max_iterations)
        if p2 is None:
            return None
        return p1 + p2[1:]  # drop duplicate HOME node at junction

    def _solve_ik(
        self,
        target_pos:     np.ndarray,
        target_quat:    np.ndarray,
        dt:             float,
        max_ik_retries: int,
    ) -> Optional[np.ndarray]:
        """IK-solve target pose; restore qpos regardless; return goal_q or None.

        Seeds tried (in order):
          0 — current config
          1 — 30 % toward joint-range midpoint
          2 — 60 % toward joint-range midpoint
          3 — home_q (if available), giving a qualitatively different wrist
              pose that may avoid collisions near the target.
        """
        env          = self.env
        start_config = env.data.qpos[:7].copy()
        lo = self.joint_limits_low
        hi = self.joint_limits_high
        neutral = (lo + hi) / 2.0

        # Build seed list; home_q is appended when available.
        seeds = [start_config]
        for k in range(1, max_ik_retries):
            alpha = 0.3 * k
            seeds.append((1.0 - alpha) * start_config + alpha * neutral)
        if self.home_q is not None:
            seeds.append(self.home_q)

        for seed in seeds:
            env.data.qpos[:7] = seed
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)

            env.ik.set_target_position(target_pos, target_quat)
            if env.ik.converge_ik(dt):
                goal_config = env.ik.configuration.q[:7].copy()
                env.data.qpos[:7] = start_config
                mujoco.mj_forward(env.model, env.data)
                return goal_config

        env.data.qpos[:7] = start_config
        mujoco.mj_forward(env.model, env.data)
        return None
