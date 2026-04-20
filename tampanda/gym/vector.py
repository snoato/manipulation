"""Spawn-safe vectorised environment factory for TAMPanda Gym envs."""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Dict, List, Optional, Type

import gymnasium
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


def make_vec_env(
    env_fn: Callable[[], gymnasium.Env],
    n_envs: int = 4,
    vec_env_cls: Optional[Type] = None,
    env_kwargs: Optional[Dict] = None,
) -> gymnasium.vector.VectorEnv:
    """Create a vectorised TAMPanda environment.

    Each worker gets its own MuJoCo instance and — if ``bridge_factory`` is
    used — its own ``DomainBridge`` instance, so there is no shared mutable
    state between workers.

    ``AsyncVectorEnv`` workers are spawned (not forked) to avoid MuJoCo's
    internal thread pool being duplicated across processes, which can cause
    deadlocks on macOS and subtle numerical divergence on Linux.

    Args:
        env_fn: Zero-argument callable that returns a fresh ``TampandaGymEnv``
            (or subclass).  Called once per worker.
        n_envs: Number of parallel environments.
        vec_env_cls: ``AsyncVectorEnv`` (default) or ``SyncVectorEnv``.
            Use ``SyncVectorEnv`` for debugging — it runs all envs in the
            calling process.
        env_kwargs: Ignored (for API symmetry with SB3's ``make_vec_env``).
            Pass all arguments via ``env_fn`` instead (e.g. via a lambda or
            ``functools.partial``).

    Returns:
        A ``gymnasium.vector.VectorEnv`` wrapping ``n_envs`` instances.

    Example::

        from functools import partial
        from tampanda.gym import TampandaGymEnv, make_vec_env

        def make_env():
            builder = ArmSceneBuilder()
            ...
            return TampandaGymEnv(builder, obs=["joints", "ee_pose"])

        vec_env = make_vec_env(make_env, n_envs=8)
    """
    if vec_env_cls is None:
        vec_env_cls = AsyncVectorEnv

    fns: List[Callable] = [env_fn for _ in range(n_envs)]

    if vec_env_cls is AsyncVectorEnv:
        ctx = mp.get_context("spawn")
        return AsyncVectorEnv(fns, context=ctx)

    return vec_env_cls(fns)
