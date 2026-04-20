"""Gymnasium wrappers that augment TampandaGymEnv with planning utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import gymnasium

from tampanda.gym.base_env import TampandaGymEnv


class ExpertActionWrapper(gymnasium.Wrapper):
    """Adds ``expert_action()`` to any ``TampandaGymEnv``.

    The expert uses the symbolic plan stored in ``info["symbolic_plan"]`` to
    determine which high-level action to execute next, then delegates to a
    ``PickPlaceExecutor`` to compute the corresponding joint target.

    Useful for:
    * Behavioural cloning / imitation learning dataset collection.
    * DAGGER: interleave learner rollouts with expert corrections.
    * Sanity-checking the planning pipeline inside the Gym loop.

    Args:
        env: A ``TampandaGymEnv`` (or subclass) instance.
        planner: Pre-built ``RRTStar``.  Created lazily on first call if ``None``.
        grasp_planner: Pre-built ``GraspPlanner``.  Created lazily if ``None``.
        table_z: Table surface height used by ``GraspPlanner``.  Inferred from
            the scene if ``None``.
    """

    def __init__(
        self,
        env: TampandaGymEnv,
        planner=None,
        grasp_planner=None,
        table_z: Optional[float] = None,
    ):
        super().__init__(env)
        self._planner = planner
        self._grasp_planner = grasp_planner
        self._table_z = table_z
        self._executor = None
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium passthrough
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        self._executor = None  # rebuild after reset (scene may have changed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Expert interface
    # ------------------------------------------------------------------

    def expert_action(self) -> Optional[np.ndarray]:
        """Return the next joint-space action from the symbolic expert.

        Determines the current plan step, resolves the PDDL action to a motion
        primitive via ``PickPlaceExecutor``, and returns the resulting joint
        target as a numpy array matching ``self.action_space``.

        Returns ``None`` if no plan is available or the plan is exhausted.
        """
        plan = self._last_info.get("symbolic_plan")
        if not plan:
            return None

        plan_step = self._last_info.get("plan_step", 0)
        if plan_step >= len(plan):
            return None

        action_name, params = plan[plan_step]
        sim = self.env.sim  # underlying FrankaEnvironment

        executor = self._get_executor(sim)
        if executor is None:
            return None

        # Execute the motion primitive and read resulting joint positions
        bridge = self.env.bridge
        if bridge is not None:
            try:
                bridge.execute_action(action_name, *params)
            except Exception:
                pass

        # Return current joint configuration as the "expert action"
        q = sim.data.qpos[:7].astype(np.float32).copy()
        # Map back to [-1, 1] for joint_target action space
        from tampanda.gym.base_env import _Q_LOW, _Q_HIGH
        q_norm = 2.0 * (q - _Q_LOW) / (_Q_HIGH - _Q_LOW) - 1.0
        q_norm = np.clip(q_norm, -1.0, 1.0)

        if self.env._include_gripper:
            gripper = np.array([1.0], np.float32)  # open after action
            return np.concatenate([q_norm, gripper])
        return q_norm

    def _get_executor(self, sim):
        if self._executor is not None:
            return self._executor

        try:
            from tampanda.planners.rrt_star import RRTStar
            from tampanda.planners.grasp_planner import GraspPlanner
            from tampanda.planners.pick_place import PickPlaceExecutor

            if self._planner is None:
                self._planner = RRTStar(sim)
            if self._grasp_planner is None:
                table_z = self._table_z or 0.27
                self._grasp_planner = GraspPlanner(table_z=table_z)
            self._executor = PickPlaceExecutor(
                sim, self._planner, self._grasp_planner, use_attachment=True
            )
        except Exception:
            return None

        return self._executor


class SymbolicRewardWrapper(gymnasium.RewardWrapper):
    """Replace the environment reward with one based on symbolic plan progress.

    Each time a new predicate in ``bridge_goals`` becomes True the agent
    receives ``+goal_bonus``.  The underlying dense reward is scaled by
    ``dense_scale`` and added on top.

    Useful for shaping sparse reward tasks without fully hand-crafting a reward.

    Args:
        env: A ``TampandaGymEnv`` instance with ``bridge_factory`` set.
        goal_bonus: Reward bonus per newly satisfied goal predicate.
        dense_scale: Scale factor for the underlying env reward.
    """

    def __init__(self, env: TampandaGymEnv, goal_bonus: float = 1.0, dense_scale: float = 0.1):
        super().__init__(env)
        self._goal_bonus = goal_bonus
        self._dense_scale = dense_scale
        self._prev_satisfied: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_satisfied = self._count_satisfied(info.get("symbolic_state"))
        return obs, info

    def reward(self, reward: float) -> float:
        # reward() is called by RewardWrapper.step() with the raw reward
        return reward * self._dense_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward * self._dense_scale

        sym_state = info.get("symbolic_state")
        satisfied = self._count_satisfied(sym_state)
        shaped += (satisfied - self._prev_satisfied) * self._goal_bonus
        self._prev_satisfied = satisfied

        return obs, shaped, terminated, truncated, info

    def _count_satisfied(self, sym_state) -> int:
        if sym_state is None or not self.env._bridge_goals:
            return 0
        count = 0
        for goal in self.env._bridge_goals:
            if isinstance(goal, tuple) and goal[0] != "not":
                if sym_state.get(goal, False):
                    count += 1
        return count


class PseudoGraspWrapper(gymnasium.Wrapper):
    """Kinematic grasp attachment driven by gripper action + proximity.

    The Franka physics model can in principle produce enough friction force to
    grasp, but in practice RL policies rarely discover the precise closed-loop
    contact sequence needed.  This wrapper bridges the gap: when the policy
    commands a close (``gripper_action < close_threshold``) and the EE is
    within ``grasp_threshold`` of a graspable object, it calls
    ``sim.attach_object_to_ee()`` so the object follows the EE kinematically.
    On open (``gripper_action > open_threshold``) the object is released via
    ``sim.detach_object()``.

    Everything else — observations, rewards, goal structure — passes through
    unchanged, so this wrapper is transparent to SB3 and HER.

    Args:
        env: A ``TampandaGymEnv`` (or subclass / wrapped) instance.
        grasp_threshold: Max EE-to-object-centre distance (m) to allow attach.
        close_threshold: Gripper action value below which the gripper is
            considered "closing" (action is in ``[-1, 1]``).
        open_threshold: Gripper action value above which the gripper is
            considered "opening".
    """

    def __init__(
        self,
        env,
        grasp_threshold: float = 0.08,
        close_threshold: float = -0.3,
        open_threshold: float = 0.3,
    ):
        super().__init__(env)
        self._grasp_threshold = grasp_threshold
        self._close_threshold = close_threshold
        self._open_threshold = open_threshold
        self._grasped_object: Optional[str] = None

    def reset(self, **kwargs):
        # Release any held object before the sim resets
        if self._grasped_object is not None:
            try:
                self._base_sim().detach_object()
            except Exception:
                pass
            self._grasped_object = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Gripper dim is always the last action element when include_gripper=True
        gripper_cmd = float(action[-1])
        sim = self._base_sim()

        if self._grasped_object is None:
            if gripper_cmd < self._close_threshold:
                self._try_attach(sim)
        else:
            if gripper_cmd > self._open_threshold:
                sim.detach_object()
                self._grasped_object = None

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _base_sim(self):
        """Walk down wrapper stack to reach the TampandaGymEnv's sim."""
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "sim"):
                return env.sim
            env = env.env
        return env.sim

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "sim"):
                return env
            env = env.env
        return env

    def _try_attach(self, sim):
        base = self._base_env()
        ee_pos = base.get_ee_pos()
        for name in base.object_names:
            obj_pos = sim.get_object_position(name)
            if np.linalg.norm(ee_pos - obj_pos) < self._grasp_threshold:
                try:
                    sim.attach_object_to_ee(name)
                    self._grasped_object = name
                except Exception:
                    pass
                return
