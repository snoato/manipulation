"""Parallel yaw probing for multilevel_blocks (Phase 4).

For ``put_upright`` and ``_put_top_down`` the chain serially tries K
grasp-yaw candidates at the high-above / column-align phase.  Each
probe runs Cartesian-substep IK (``plan_to_pose`` with n_substeps≈20),
which is most of the per-check wall-clock — the unconverged probes
each spend mink to its max_iters cap (~100-200 ms) before being
discarded.

This module provides a persistent multiprocessing pool whose workers
each hold their own env + executor + LinearIKPlanner.  The pool's
``first_success_lerp`` / ``first_success_plan`` methods dispatch one
yaw probe per worker and return the first success — wall-clock per
phase drops from ``K × per_probe`` to ``per_probe + dispatch_overhead``.

Workers are kept warm across many ``check_action`` calls inside a
single rgnet worker process.  State sync (``sync_state``) broadcasts
the symbolic state to every worker once per ``check_action``; per-yaw
probes only send the small ``(arm_qpos, target, quat)`` payload.

Architectural notes:

* Each worker process has its OWN MuJoCo env and mink IK instance.
  Memory cost: ~30-50 MB per worker.
* The pool is constructed by :func:`_make_executor` (in
  feasibility.py) when ``fast=True`` AND ``n_yaw_workers > 1``.
  When omitted (default), the executor falls back to serial yaw
  probing — no behaviour change.
* Workers run the SAME ``MultilevelBlocksConfig`` as the main env, so
  they see the same grid, hand_capsule disabled, etc.

Public API::

    pool = MultilevelBlocksYawPool(config, n_workers=4)
    pool.sync_state(state_dict)            # once per check_action
    path, used_quat = pool.first_success_lerp(
        arm_qpos, target_pos, quats, n_substeps,
    )
    pool.close()                             # at shutdown
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Worker side — runs inside each pool subprocess.
# ---------------------------------------------------------------------------


_W: Dict[str, Any] = {}


def _worker_init(config_dict: dict) -> None:
    """Build env + executor + LinearIKPlanner in this worker process."""
    import mujoco
    from tampanda.symbolic.domains.multilevel_blocks import (
        MultilevelBlocksConfig, make_multilevel_blocks_builder,
        make_multilevel_blocks_bridge,
    )
    from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
        _make_executor,
    )
    cfg = MultilevelBlocksConfig(**config_dict)
    scratch = tempfile.TemporaryDirectory(prefix=f"yawpool_w{os.getpid()}_")
    builder, ws, cfg = make_multilevel_blocks_builder(
        scratch_dir=Path(scratch.name), config=cfg,
    )
    env = builder.build_env(rate=10000.0)
    bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
    executor = _make_executor(env, ws, cfg, fast=True)

    _W["scratch"] = scratch
    _W["env"] = env
    _W["ws"] = ws
    _W["cfg"] = cfg
    _W["executor"] = executor
    _W["lik"] = executor.lik
    _W["bridge"] = bridge
    _W["mujoco"] = mujoco


def _worker_sync_state(state_dict: dict) -> bool:
    """Restore the worker's env to ``state_dict`` (block placements,
    held-* fluents).  Called once per :meth:`MultilevelBlocksYawPool.sync_state`.
    """
    from tampanda.symbolic.domains.multilevel_blocks.state import (
        restore_state,
    )
    try:
        restore_state(
            _W["env"], _W["ws"], _W["cfg"], state_dict,
            on_held="attach", executor=_W["executor"],
        )
        return True
    except Exception:
        return False


def _worker_probe_lerp(
    payload: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
) -> Tuple[bool, Optional[List[np.ndarray]], np.ndarray]:
    """One yaw probe via ``plan_joint_lerp``.  Payload:
    ``(arm_qpos, target_pos, quat, n_substeps)``.
    """
    arm_qpos, target_pos, quat, n_substeps = payload
    env = _W["env"]
    lik = _W["lik"]
    mujoco = _W["mujoco"]

    env.data.qpos[:7] = arm_qpos
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)

    path = lik.plan_joint_lerp(target_pos, quat, n_substeps=n_substeps)
    return (path is not None, path, quat)


def _worker_probe_plan(
    payload: Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool],
) -> Tuple[bool, Optional[List[np.ndarray]], np.ndarray]:
    """One yaw probe via ``plan_to_pose`` (Cartesian-substep IK).
    Payload: ``(arm_qpos, target_pos, quat, n_substeps, slerp_orientation)``."""
    arm_qpos, target_pos, quat, n_substeps, slerp_orientation = payload
    env = _W["env"]
    lik = _W["lik"]
    mujoco = _W["mujoco"]

    env.data.qpos[:7] = arm_qpos
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)

    path = lik.plan_to_pose(target_pos, quat,
                                  slerp_orientation=slerp_orientation,
                                  n_substeps=n_substeps)
    return (path is not None, path, quat)


# ---------------------------------------------------------------------------
# Main-process pool wrapper.
# ---------------------------------------------------------------------------


class MultilevelBlocksYawPool:
    """Persistent worker pool for parallel yaw probing.

    Construction is expensive (~1.5 s × n_workers).  Reuse a single
    instance across all ``check_action`` calls in a process.
    """

    def __init__(self, config, n_workers: int = 4,
                     start_method: Optional[str] = None):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        # Convert config dataclass to plain dict so it survives spawn.
        if hasattr(config, "__dataclass_fields__"):
            cfg_dict = asdict(config)
        else:
            cfg_dict = dict(config)
        self._cfg_dict = cfg_dict

        ctx = (mp.get_context(start_method) if start_method
                  else mp.get_context())
        self._pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(cfg_dict,),
        )

    def sync_state(self, state_dict: dict) -> bool:
        """Broadcast ``state_dict`` to every worker — they each call
        ``restore_state`` with it so subsequent probes see the same
        block placements + held state as the main executor.

        Call once per ``check_action`` after the main executor's
        ``restore_state`` has run.
        """
        # apply_async one per worker — we can't broadcast directly with
        # a Pool, but `map` over `range(n_workers)` is close enough
        # (each worker processes one item).  Caveat: workers can share
        # work, so to guarantee EVERY worker syncs we use n_workers
        # tasks AND chunksize=1.
        results = self._pool.map(
            _worker_sync_state,
            [state_dict] * self.n_workers,
            chunksize=1,
        )
        return all(results)

    def first_success_lerp(
        self,
        arm_qpos: np.ndarray,
        target_pos: np.ndarray,
        quats: Sequence[np.ndarray],
        n_substeps: int = 16,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """Dispatch one ``plan_joint_lerp`` probe per quat in parallel.
        Returns ``(path, used_quat)`` for the first success or ``None``
        if every probe fails."""
        if not quats:
            return None
        payloads = [
            (arm_qpos.copy(), target_pos.copy(), q.copy(), n_substeps)
            for q in quats
        ]
        # imap_unordered yields results AS workers finish; we take the
        # first success.  Remaining work continues but we ignore it —
        # the pool reclaims those tasks naturally.
        for ok, path, used_quat in self._pool.imap_unordered(
                _worker_probe_lerp, payloads, chunksize=1,
        ):
            if ok:
                return path, used_quat
        return None

    def first_success_plan(
        self,
        arm_qpos: np.ndarray,
        target_pos: np.ndarray,
        quats: Sequence[np.ndarray],
        n_substeps: int = 20,
        slerp_orientation: bool = False,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """Dispatch one ``plan_to_pose`` probe per quat in parallel.
        Returns the first success or ``None``."""
        if not quats:
            return None
        payloads = [
            (arm_qpos.copy(), target_pos.copy(), q.copy(),
             n_substeps, slerp_orientation)
            for q in quats
        ]
        for ok, path, used_quat in self._pool.imap_unordered(
                _worker_probe_plan, payloads, chunksize=1,
        ):
            if ok:
                return path, used_quat
        return None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
