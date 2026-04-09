"""Worker pool for parallel MuJoCo collision checking.

Each worker owns a private FrankaEnvironment instance so MuJoCo state is
never shared between processes.

Scene state (object positions, held body, collision exceptions) is passed
directly with every check task so worker-side global state is never stale.
``set_scene()`` snapshots the environment locally; the snapshot is then
embedded in each ``check_edges_parallel`` / ``check_configs_parallel`` call.
This avoids a race condition where ``pool.map`` with N tasks for N workers
does not guarantee each worker processes exactly one task, which could leave
some workers with uninitialised ``_scene_exc = None``.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Module-level worker state (one per worker process)
# ---------------------------------------------------------------------------

_env            = None
_steps: int     = 5
_order: np.ndarray = None


def _noop(_) -> None:
    """No-op used as a barrier to confirm a worker has finished its initializer."""
    pass


def _make_binary_order(steps: int) -> np.ndarray:
    """Indices 1..steps in binary-subdivision order (index 0 excluded)."""
    order: list[int] = []
    queue: list[tuple[int, int]] = [(1, steps)]
    while queue:
        lo, hi = queue.pop(0)
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        order.append(mid)
        queue.append((lo, mid - 1))
        queue.append((mid + 1, hi))
    return np.array(order, dtype=np.int32)


def _worker_init(xml_path: str, collision_check_steps: int) -> None:
    global _env, _steps, _order
    from tampanda.environments.franka_env import FrankaEnvironment
    _env = FrankaEnvironment(xml_path)
    _steps = collision_check_steps
    _order = _make_binary_order(collision_check_steps)


# ---------------------------------------------------------------------------
# Per-check tasks — scene snapshot embedded in args
# ---------------------------------------------------------------------------

def _check_single_config_task(args: tuple) -> bool:
    """Check one arm configuration. Scene snapshot included in args."""
    scene_qpos, scene_held, scene_exc, config = args
    _env.data.qpos[7:] = scene_qpos
    _env._collision_held_body    = scene_held
    _env._collision_exception_ids = scene_exc
    return _env.is_collision_free_no_restore(config)


def _check_edge_task(args: tuple) -> bool:
    """Check edge c1→c2 with binary-subdivision. Scene snapshot included in args."""
    scene_qpos, scene_held, scene_exc, c1, c2 = args
    env = _env
    inv   = 1.0 / _steps
    delta = c2 - c1
    env._collision_held_body    = scene_held
    env._collision_exception_ids = scene_exc
    for idx in _order:
        env.data.qpos[7:] = scene_qpos   # reset objects before each check
        config = c1 + (idx * inv) * delta
        if not env.is_collision_free_no_restore(config):
            return False
    return True


# ---------------------------------------------------------------------------
# Public pool class
# ---------------------------------------------------------------------------

class CollisionWorkerPool:
    """Persistent pool of MuJoCo worker processes for parallel collision checks.

    Typical usage per planning query::

        pool.set_scene(env)               # snapshot scene state locally
        pool.check_edges_parallel(edges)  # scene embedded in each task

    ``set_scene`` no longer broadcasts to workers; it only caches a snapshot
    on this object.  The snapshot is then passed with every check task, so
    each worker always operates on the correct scene regardless of which
    worker picks up a given task.

    Args:
        xml_path: Path to the scene XML (must remain on disk while pool is alive).
        n_workers: Number of worker processes to spawn.
        collision_check_steps: Intermediate configs per edge (binary subdivision).
    """

    def __init__(
        self,
        xml_path: str,
        n_workers: int = 4,
        collision_check_steps: int = 5,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(xml_path, collision_check_steps),
        )
        # Barrier: block until every worker has completed _worker_init.
        # ctx.Pool() returns before initializers finish, so without this,
        # check tasks can land on workers still spinning up.
        self._pool.map(_noop, range(n_workers), chunksize=1)
        self.n_workers = n_workers
        self.collision_check_steps = collision_check_steps
        self._scene_snapshot: dict | None = None

    # ------------------------------------------------------------------
    # Scene sync — call once per planning query
    # ------------------------------------------------------------------

    def set_scene(self, env) -> None:
        """Snapshot current scene state for use in subsequent check calls.

        Unlike the previous implementation, this does NOT broadcast to workers.
        The snapshot is passed directly with every check task, so all workers
        always receive the correct scene regardless of task distribution.
        """
        held = env._collision_held_body
        self._scene_snapshot = {
            "qpos": env.data.qpos[7:].copy(),   # object positions only
            "held_body": {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in held.items()
            } if held else None,
            "exception_ids": set(env._collision_exception_ids),
        }

    # ------------------------------------------------------------------
    # Parallel checks — scene embedded in each task
    # ------------------------------------------------------------------

    def check_configs_parallel(self, configs: list[np.ndarray]) -> list[bool]:
        """Check N arm configurations in parallel; returns N booleans."""
        if not configs:
            return []
        if self._scene_snapshot is None:
            raise RuntimeError(
                "CollisionWorkerPool: call set_scene() before check_configs_parallel()"
            )
        s = self._scene_snapshot
        args = [(s["qpos"], s["held_body"], s["exception_ids"], c) for c in configs]
        return self._pool.map(_check_single_config_task, args)

    def check_edges_parallel(
        self, edges: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[bool]:
        """Check N edges in parallel (full binary-subdivision per edge); returns N booleans."""
        if not edges:
            return []
        if self._scene_snapshot is None:
            raise RuntimeError(
                "CollisionWorkerPool: call set_scene() before check_edges_parallel()"
            )
        s = self._scene_snapshot
        args = [(s["qpos"], s["held_body"], s["exception_ids"], c1, c2) for c1, c2 in edges]
        return self._pool.map(_check_edge_task, args)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
