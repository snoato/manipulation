"""TampandaGoalEnv — GoalEnv subclass with HER-compatible goal structure."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium
from gymnasium import spaces

from tampanda.gym.base_env import TampandaGymEnv


class TampandaGoalEnv(TampandaGymEnv):
    """Goal-conditioned variant of :class:`TampandaGymEnv`.

    Wraps the observation as ``Dict(observation, achieved_goal, desired_goal)``
    and implements ``compute_reward`` — the two requirements for Stable-Baselines3's
    ``HerReplayBuffer``.

    Two goal types are supported:

    * ``"object_pose"`` — achieved/desired goal is the flattened XYZ positions of
      ``goal_objects``.  ``compute_reward`` returns ``0`` (success) or ``-1``
      (failure) based on L2 distance < ``goal_threshold``.

    * ``"symbolic_predicates"`` — achieved/desired goal is a boolean vector of
      all PDDL predicate truth values grounded by the bridge.  ``compute_reward``
      returns ``-hamming(achieved, desired)`` normalised to ``[-1, 0]``.  Requires
      ``bridge_factory`` and ``bridge_objects`` to be set.

    Args:
        goal_type: ``"object_pose"`` or ``"symbolic_predicates"``.
        goal_objects: Body names whose positions form the goal vector
            (``"object_pose"`` mode only).  Defaults to all inferred objects.
        goal_threshold: Distance below which the goal is considered reached
            (``"object_pose"`` mode, metres).
        goal_target_sampler: Optional callable ``(sim, rng) -> np.ndarray``
            that produces a desired goal at each reset.  If ``None``, a random
            position within ``[goal_pos_low, goal_pos_high]`` is used.
        goal_pos_low: Lower bound for the default random goal sampler.
        goal_pos_high: Upper bound for the default random goal sampler.
        **kwargs: All remaining arguments forwarded to :class:`TampandaGymEnv`.
    """

    def __init__(
        self,
        *,
        goal_type: str = "object_pose",
        goal_objects: Optional[List[str]] = None,
        goal_threshold: float = 0.02,
        goal_target_sampler: Optional[Callable] = None,
        goal_pos_low: Tuple[float, float, float] = (0.25, -0.3, 0.3),
        goal_pos_high: Tuple[float, float, float] = (0.65,  0.3, 0.6),
        **kwargs,
    ):
        super().__init__(**kwargs)

        if goal_type not in ("object_pose", "symbolic_predicates"):
            raise ValueError(
                f"Unknown goal_type {goal_type!r}. "
                "Valid: 'object_pose', 'symbolic_predicates'."
            )

        self._goal_type = goal_type
        self._goal_threshold = goal_threshold
        self._goal_target_sampler = goal_target_sampler
        self._goal_pos_low = np.array(goal_pos_low, np.float32)
        self._goal_pos_high = np.array(goal_pos_high, np.float32)

        # goal_objects defaults to all inferred object names
        self._goal_objects: List[str] = (
            goal_objects if goal_objects is not None else list(self._object_names)
        )

        # Compute goal dimension
        if goal_type == "object_pose":
            self._goal_dim = len(self._goal_objects) * 3  # XYZ per object
        else:
            # Count grounded predicate slots by enumerating type-consistent
            # argument combinations — no predicate functions are called so
            # env=None is safe here.
            self._goal_dim = self._count_predicate_slots(
                kwargs.get("bridge_factory"),
                kwargs.get("bridge_objects") or {},
            )

        # Override observation_space to include goal structure
        self._rebuild_goal_obs_space()

        # Runtime goal state
        self._desired_goal: Optional[np.ndarray] = None
        self._predicate_keys: Optional[List[Tuple]] = None

    # ------------------------------------------------------------------
    # Gymnasium / HER API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Run base reset (builds bridge, settles physics)
        obs_base, info = super().reset(seed=seed, options=options)

        # For symbolic mode, lock in predicate key ordering from the live bridge.
        # _goal_dim is already correct (set at __init__), so no space rebuild needed.
        if self._goal_type == "symbolic_predicates":
            if self._bridge is None:
                raise RuntimeError(
                    "goal_type='symbolic_predicates' requires bridge_factory to be set."
                )
            state = self._bridge.ground_state(self._bridge_objects)
            self._predicate_keys = sorted(state.keys())

        # Sample desired goal
        self._desired_goal = self._sample_desired_goal()

        return self._wrap_obs(obs_base), info

    def step(self, action: np.ndarray):
        obs_base, reward_base, terminated, truncated, info = super().step(action)

        sym_state = info.get("symbolic_state")
        achieved = self._compute_achieved_goal(sym_state)
        success = float(self.compute_reward(achieved, self._desired_goal, info)) == 0.0
        # shaped reward from reward_fn + success bonus; HER relabelling still
        # uses compute_reward (sparse 0/-1) for synthetic transitions
        reward = float(reward_base) + (5.0 if success else 0.0)
        terminated = success

        obs = self._wrap_obs(obs_base, achieved)
        info["achieved_goal"] = achieved
        info["desired_goal"] = self._desired_goal
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Any,
    ) -> np.ndarray:
        """Batched reward compatible with HerReplayBuffer.

        Returns ``0.0`` on success, ``-1.0`` otherwise.  Works for both single
        samples and batches (HER calls this with arrays).
        """
        if self._goal_type == "object_pose":
            dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return (dist < self._goal_threshold).astype(np.float32) - 1.0

        # symbolic_predicates: fraction of predicates matching
        match = (achieved_goal == desired_goal).all(axis=-1)
        return match.astype(np.float32) - 1.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_desired_goal(self) -> np.ndarray:
        if self._goal_type == "object_pose":
            if self._goal_target_sampler is not None:
                return self._goal_target_sampler(self._sim, self.np_random).astype(np.float32)
            # Random positions for each goal object
            parts = []
            for _ in self._goal_objects:
                pos = self.np_random.uniform(self._goal_pos_low, self._goal_pos_high)
                parts.append(pos.astype(np.float32))
            return np.concatenate(parts)

        # symbolic_predicates: desired goal comes from bridge_goals
        if self._bridge is None or self._predicate_keys is None:
            return np.zeros(self._goal_dim, np.float32)
        # Build desired predicate vector from bridge_goals list
        desired = np.zeros(len(self._predicate_keys), np.float32)
        if self._bridge_goals:
            for goal in self._bridge_goals:
                if isinstance(goal, tuple) and goal[0] != "not":
                    key = goal
                    if key in self._predicate_keys:
                        desired[self._predicate_keys.index(key)] = 1.0
        return desired

    def _compute_achieved_goal(self, sym_state) -> np.ndarray:
        if self._goal_type == "object_pose":
            parts = [
                self._sim.get_object_position(n).astype(np.float32)
                for n in self._goal_objects
            ]
            return np.concatenate(parts) if parts else np.zeros(self._goal_dim, np.float32)

        # symbolic_predicates
        if sym_state is None or self._predicate_keys is None:
            return np.zeros(self._goal_dim, np.float32)
        return np.array(
            [float(sym_state.get(k, False)) for k in self._predicate_keys], np.float32
        )

    def _wrap_obs(
        self,
        base_obs: Dict[str, np.ndarray],
        achieved_goal: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        if achieved_goal is None:
            sym_state = (
                self._bridge.ground_state(self._bridge_objects)
                if self._bridge is not None
                else None
            )
            achieved_goal = self._compute_achieved_goal(sym_state)

        # Flatten base obs into a single "observation" vector for Dict GoalEnv
        flat_parts = []
        for v in base_obs.values():
            flat_parts.append(v.ravel().astype(np.float32))
        flat_obs = np.concatenate(flat_parts) if flat_parts else np.zeros((0,), np.float32)

        return {
            "observation":   flat_obs,
            "achieved_goal": achieved_goal,
            "desired_goal":  self._desired_goal if self._desired_goal is not None
                             else np.zeros(self._goal_dim, np.float32),
        }

    @staticmethod
    def _count_predicate_slots(bridge_factory, bridge_objects: dict) -> int:
        """Count grounded predicate slots without evaluating any predicate function.

        Calls ``bridge_factory()`` once (safe with ``env=None`` since predicates
        are only registered, never called during construction) then enumerates
        all type-consistent argument combinations.
        """
        import itertools

        if bridge_factory is None:
            return 1  # fallback — symbolic mode needs a factory

        bridge = bridge_factory()
        count = 0
        all_pred_names = list(bridge._code_predicates.keys()) + list(bridge._fluent_names)
        for pred_name in all_pred_names:
            param_types = [
                p.type.name for p in bridge._up_fluents[pred_name].signature
            ]
            if not param_types:
                count += 1
            else:
                count += len(list(itertools.product(
                    *[bridge_objects.get(t, []) for t in param_types]
                )))
        return max(count, 1)

    def _rebuild_goal_obs_space(self):
        """Construct the Dict(observation, achieved_goal, desired_goal) space."""
        # Flatten base observation space to determine total obs dim
        flat_dim = 0
        for sp in self.observation_space.spaces.values():
            flat_dim += int(np.prod(sp.shape))

        goal_space = spaces.Box(-np.inf, np.inf, (self._goal_dim,), np.float32)

        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, (flat_dim,), np.float32),
            "achieved_goal": goal_space,
            "desired_goal":  goal_space,
        })
