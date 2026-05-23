"""Linear-IK chain planner.

Replaces RRT* for short, low-clutter Cartesian moves (typically descent
to grasp, post-grasp lift, place descent).  Approach:

1. Lerp the EE position straight-line in Cartesian space, ``n_substeps``
   waypoints from the current EE pose to the target.
2. For each waypoint, IK-solve to a joint config.  Validate it's
   collision-free.
3. Validate the joint-space interpolation between the previous and
   current substep config — between two collision-free IK solutions
   the lerp can swing a finger through a neighbour, so we sample the
   straight line in joint space at fine resolution.
4. Append the config to the path.

If any substep fails (IK can't converge, config in collision, joint
interpolation in collision), return ``None`` and the caller can fall
back to RRT*.

Why this is useful for the multi-grid domains
---------------------------------------------

RRT*'s randomness hurts when you need a deterministic short move.  For
post-grasp lifts and descents inside a closed-top shelf — moves of
5-12 cm — the joint-space corridor is narrow (the wrist is near a
joint limit), and RRT* often can't sample its way through.  A
straight-line Cartesian path with IK at every substep is the natural
fit: the EE goes where you want, the joint trajectory is smooth, no
RNG.

This is the same pattern used in plan2policy's
``_cartesian_ik_chain_to``.
"""
from __future__ import annotations

from typing import List, Optional

import mujoco
import numpy as np


class LinearIKPlanner:
    """Cartesian straight-line + IK chain planner.

    Args:
        env:                FrankaEnvironment.
        n_substeps:         Cartesian-space waypoints between current
                            EE pose and target (default 8).  Smaller =
                            faster, larger = finer collision check.
        joint_check_steps:  Sub-divisions per joint-space transition
                            for collision validation (default 8).
        collision_tolerance: Allowed penetration depth in metres for
                            the joint-space check (default -0.002 — up
                            to 2 mm of overlap is tolerated to avoid
                            sub-mm physics jitter rejecting valid
                            configs).
    """

    def __init__(
        self,
        env,
        n_substeps: int = 8,
        joint_check_steps: int = 8,
        collision_tolerance: float = -0.002,
    ):
        self.env = env
        self.n_substeps = n_substeps
        self.joint_check_steps = joint_check_steps
        self.collision_tolerance = collision_tolerance
        self._ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def plan_joint_lerp(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.005,
        n_substeps: int = 16,
    ) -> Optional[List[np.ndarray]]:
        """Solve IK once at the target, then lerp joints from current
        to that goal config — no Cartesian sub-IK.

        Right tool for **short, same-orientation moves** where the
        approach's IK config (current ``qpos``) is already a good seed
        for the goal (e.g., descent inside a shelf cavity, post-grasp
        lift).  Avoids the basin-flip risk of Cartesian-substep IK:
        IK runs once, the joint trajectory is a straight line in
        joint space, and we collision-check every interpolated point.

        Returns ``[start_q, ..., goal_q]`` or ``None`` if IK doesn't
        converge or the joint lerp passes through a collision.
        """
        env = self.env
        qpos_save = env.data.qpos.copy()
        qvel_save = env.data.qvel.copy()
        try:
            start_q = env.data.qpos[:7].copy()
            env.ik.update_configuration(env.data.qpos)
            env.ik.set_target_position(target_pos, target_quat)
            if not env.ik.converge_ik(dt):
                return None
            goal_q = env.ik.configuration.q[:7].copy()

            if not self._config_collision_free(goal_q):
                return None
            if not self._segment_collision_free_n(start_q, goal_q,
                                                    n_substeps):
                return None

            # Build an evenly-spaced path so the controller has
            # well-distributed waypoints to follow.
            path: List[np.ndarray] = [start_q.copy()]
            for k in range(1, n_substeps + 1):
                alpha = k / n_substeps
                path.append((1 - alpha) * start_q + alpha * goal_q)
            return path
        finally:
            env.data.qpos[:] = qpos_save
            env.data.qvel[:] = qvel_save
            mujoco.mj_forward(env.model, env.data)

    def _segment_collision_free_n(
        self, q_a: np.ndarray, q_b: np.ndarray, n: int,
    ) -> bool:
        env = self.env
        for j in range(1, n + 1):
            alpha = j / n
            q_mid = (1 - alpha) * q_a + alpha * q_b
            if not env.is_collision_free(q_mid):
                return False
        return True

    def plan_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.005,
        n_substeps: Optional[int] = None,
        slerp_orientation: bool = True,
    ) -> Optional[List[np.ndarray]]:
        """Plan a straight-line Cartesian path from the current EE pose
        to ``(target_pos, target_quat)``.  Returns a list of joint
        configs (7-dof each) that the controller can execute, or
        ``None`` if any substep fails IK / collision validation.

        Args:
            slerp_orientation: When True (default), the EE orientation
                is SLERP-interpolated from the current EE quat to
                ``target_quat`` across substeps.  Useful when the
                approach involves a big rotation (palm-down →
                palm-+y) — IK at each substep sees a small
                incremental change instead of a discontinuity.
                Set False if the caller explicitly wants the EE to
                hold ``target_quat`` from the first substep.

        State is fully restored before returning regardless of outcome.
        """
        if n_substeps is None:
            n_substeps = self.n_substeps
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")

        env = self.env
        qpos_save = env.data.qpos.copy()
        qvel_save = env.data.qvel.copy()

        try:
            cur_ee = env.data.site_xpos[self._ee_site_id].copy()
            cur_q = env.data.qpos[:7].copy()
            target = np.asarray(target_pos, dtype=float)
            target_q4 = np.asarray(target_quat, dtype=float)
            target_q4 = target_q4 / np.linalg.norm(target_q4)
            delta = target - cur_ee

            # Current EE quat (mocap-frame at the attachment site).
            cur_quat = np.empty(4)
            cur_mat = env.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()
            mujoco.mju_mat2Quat(cur_quat, cur_mat.flatten())
            cur_quat = cur_quat / np.linalg.norm(cur_quat)

            path: List[np.ndarray] = [cur_q.copy()]
            prev_q = cur_q.copy()

            for k in range(1, n_substeps + 1):
                alpha = k / n_substeps
                sub_pos = cur_ee + alpha * delta
                if slerp_orientation:
                    sub_quat = _slerp(cur_quat, target_q4, alpha)
                else:
                    sub_quat = target_q4

                # Restore env to prev_q for IK seed continuity.
                env.data.qpos[:7] = prev_q
                env.data.qvel[:] = 0.0
                mujoco.mj_forward(env.model, env.data)
                env.ik.update_configuration(env.data.qpos)
                env.ik.set_target_position(sub_pos, sub_quat)
                if not env.ik.converge_ik(dt):
                    return None
                q_sub = env.ik.configuration.q[:7].copy()

                # Validate substep config is collision-free.
                if not self._config_collision_free(q_sub):
                    return None

                # Validate joint-space transition prev_q → q_sub at fine
                # resolution.  Catches finger-swings through neighbours.
                if not self._segment_collision_free(prev_q, q_sub):
                    return None

                path.append(q_sub)
                prev_q = q_sub

            return path
        finally:
            env.data.qpos[:] = qpos_save
            env.data.qvel[:] = qvel_save
            mujoco.mj_forward(env.model, env.data)

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _config_collision_free(self, q_arm: np.ndarray) -> bool:
        """Check the arm config is collision-free.  Honours
        ``env._collision_held_body`` so a held block tags along."""
        env = self.env
        return env.is_collision_free(q_arm)

    def _segment_collision_free(
        self, q_a: np.ndarray, q_b: np.ndarray,
    ) -> bool:
        """Sample the straight line between two configs at fine
        resolution and check every intermediate config for collision.
        ``q_a`` is assumed already-checked (and is the start of the
        segment); we test j ∈ [1, joint_check_steps]."""
        env = self.env
        for j in range(1, self.joint_check_steps + 1):
            alpha = j / self.joint_check_steps
            q_mid = (1 - alpha) * q_a + alpha * q_b
            if not env.is_collision_free(q_mid):
                return False
        return True


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Both inputs are MuJoCo WXYZ quaternions; output is also WXYZ.
    Falls back to linear interpolation when the two quats are nearly
    parallel (dot product close to 1).
    """
    q0 = np.asarray(q0, dtype=float); q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    # Pick the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1
