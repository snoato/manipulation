"""Multiprocessing pool wrapper around the feasibility checker.

For BFS data generation the dominant cost is per-action feasibility
checking.  Each worker builds its own MuJoCo env + executor once (~1.5 s)
and then services many (state, action) probes (~70-470 ms each), so the
pool keeps workers alive across the whole batch.

Public API::

    from tampanda.symbolic.domains.multilevel_blocks.parallel import (
        ParallelFeasibilityChecker,
    )

    checker = ParallelFeasibilityChecker(
        n_workers=8, config=cfg, fast=True,
    )
    results = checker.check_batch([(state1, action1), (state2, action2), ...])
    checker.close()

State dicts and action tuples are picklable; only primitives go through
the queue.  The result list preserves input order.

Notes:
* The :class:`MultilevelBlocksConfig` must be picklable — it is, since
  it's a dataclass of primitives.
* Each worker materialises its own scene XML into a temporary
  directory; cleaned up at pool shutdown.
* Pool uses the ``spawn`` start method on macOS / Windows by default; on
  Linux ``fork`` is faster.  We default to whatever the user's
  ``multiprocessing`` is set to.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
)


# ---------------------------------------------------------------------------
# Worker-process state.
# ---------------------------------------------------------------------------
# These globals are populated by the per-worker initialiser and reused
# across every task that lands on the same worker process.

_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(config_dict: Dict[str, Any], fast: bool) -> None:
    """Per-worker setup: build a fresh env + bridge + executor.

    Runs ONCE per worker process at pool startup.  All subsequent tasks
    on this worker reuse the same env / executor.
    """
    # Reconstruct the config dataclass from its dict form.  Pickling
    # the dataclass directly works too but the dict form is more robust
    # to schema changes between client and worker.
    from tampanda.symbolic.domains.multilevel_blocks import (
        make_multilevel_blocks_builder, make_multilevel_blocks_bridge,
    )
    from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
        _make_executor,
    )

    cfg = MultilevelBlocksConfig(**config_dict)
    scratch = tempfile.TemporaryDirectory(prefix="parfeas_")
    builder, ws, cfg = make_multilevel_blocks_builder(
        scratch_dir=Path(scratch.name), config=cfg,
    )
    env = builder.build_env(rate=10000.0)
    bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
    executor = _make_executor(env, ws, cfg, fast=fast)

    _WORKER_STATE["scratch"] = scratch
    _WORKER_STATE["env"] = env
    _WORKER_STATE["ws"] = ws
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["bridge"] = bridge
    _WORKER_STATE["objects"] = objects
    _WORKER_STATE["executor"] = executor
    _WORKER_STATE["fast"] = fast


def _layout_to_state(
    layout: Dict[str, List[str]],
    held: Optional[Tuple[str, str]] = None,
) -> Dict[Tuple, bool]:
    """Reconstruct the minimal ``state`` dict needed by ``restore_state``.

    The full ``bridge.ground_state(...)`` output is huge (millions of
    keys including static adjacency and shape predicates) and pickling
    it dominates worker dispatch cost.  ``restore_state`` only reads
    the ``(in, block, cell)`` entries and any held-* fluents, so we can
    rebuild a tiny equivalent state from a compact layout.
    """
    state: Dict[Tuple, bool] = {}
    for block, cell_ids in layout.items():
        for cid in cell_ids:
            state[("in", block, cid)] = True
    if held is not None:
        fluent, block = held
        state[(fluent, block)] = True
    return state


def _worker_check_action(
    payload: Tuple[Dict[str, List[str]], Optional[Tuple], Tuple],
) -> Dict[str, Any]:
    """Run a single (layout, held, action) check on this worker.

    ``layout`` is a ``{block: [cell_ids]}`` map; ``held`` is
    ``(fluent_name, block_name)`` or ``None``.  Both are tiny picklables
    that avoid the 26 MB ground-state blob.
    """
    from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
        check_action,
    )
    import os as _os
    layout, held, action = payload
    state = _layout_to_state(layout, held)
    t = time.perf_counter()
    res = check_action(
        _WORKER_STATE["env"],
        _WORKER_STATE["ws"],
        _WORKER_STATE["cfg"],
        state,
        action,
        fast=_WORKER_STATE["fast"],
        executor=_WORKER_STATE["executor"],
    )
    res["_worker_pid"] = _os.getpid()
    res["_worker_t"] = time.perf_counter() - t
    return {k: v for k, v in res.items() if k != "executor"}


def _worker_check_sequence(
    payload: Tuple[Dict[str, List[str]], Optional[Tuple], List[Tuple]],
) -> Dict[str, Any]:
    """Run a (layout, held, [action,...]) sequence on this worker."""
    from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
        check_action_sequence,
    )
    layout, held, actions = payload
    state = _layout_to_state(layout, held)
    res = check_action_sequence(
        _WORKER_STATE["env"],
        _WORKER_STATE["ws"],
        _WORKER_STATE["cfg"],
        state,
        list(actions),
        fast=_WORKER_STATE["fast"],
    )
    return res


# ---------------------------------------------------------------------------
# Public pool wrapper.
# ---------------------------------------------------------------------------


class ParallelFeasibilityChecker:
    """Persistent pool of feasibility workers.

    Each worker holds its own MuJoCo env + executor.  ``check_batch``
    distributes ``(state, action)`` payloads across workers and returns
    a list of per-payload result dicts in input order.

    Construction is expensive (~1.5 s per worker) so reuse a single
    instance across all BFS iterations.

    Use as a context manager to guarantee :meth:`close` runs::

        with ParallelFeasibilityChecker(n_workers=8, config=cfg) as ck:
            results = ck.check_batch(items)
    """

    def __init__(
        self,
        n_workers: int,
        config: MultilevelBlocksConfig,
        *,
        fast: bool = True,
        start_method: Optional[str] = None,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.fast = fast

        # Use a dict-of-primitives for config so it survives pickle
        # across process boundaries even if the dataclass schema
        # diverges between client + worker.
        from dataclasses import fields, asdict
        # Some MultilevelBlocksConfig fields are tuples (table_pos);
        # asdict handles those.
        self._config_dict = asdict(config)

        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self._pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(self._config_dict, fast),
        )

    def check_batch(
        self,
        items: List[Tuple[Dict[str, List[str]], Optional[Tuple], Tuple]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        """Check a batch of ``(layout, held, action)`` tuples.

        ``layout`` is a ``{block_name: [cell_ids]}`` dict (compact
        representation of the symbolic state).  ``held`` is
        ``(fluent_name, block_name)`` or ``None``.  ``action`` is the
        usual ``(action_name, *args)`` tuple.

        Avoids pickling the full ground-state dict (~26 MB) on every
        task — sends only the layout (a few hundred bytes).

        Returns results in input order.
        """
        return self._pool.map(_worker_check_action, items, chunksize=chunksize)

    def check_sequence_batch(
        self,
        items: List[Tuple[Dict[str, List[str]], Optional[Tuple], List[Tuple]]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        """Check a batch of ``(layout, held, [action, ...])`` sequences."""
        return self._pool.map(_worker_check_sequence, items,
                                  chunksize=chunksize)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
