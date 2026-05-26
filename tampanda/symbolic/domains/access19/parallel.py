"""Multiprocessing pool wrapper around the access-19 feasibility checker.

For BFS / oracle-with-backtracking data generation the dominant cost is
per-action feasibility checking.  Each worker builds its own MuJoCo env
+ chain pick/put functions + executor once (~1.5 s) and then services
many (state, action) probes (~0.1–1 s each).

Public API::

    from tampanda.symbolic.domains.access19.parallel import (
        ParallelFeasibilityChecker,
    )

    with ParallelFeasibilityChecker(n_workers=8, fast=True) as ck:
        results = ck.check_batch([(layout1, held1, action1), ...])

Compact ``layout`` dicts are sent to workers (not the full ground-state)
to keep per-task pickle cost low.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(fast: bool) -> None:
    """Per-worker setup: build env + chains + executor + home pose.

    Runs ONCE per worker process at pool startup.  All subsequent tasks
    on this worker reuse the same env / chains.
    """
    from tampanda.symbolic.domains.access19 import (
        apply_runtime_tweaks, make_access19_builder,
        make_access19_pick_fn, make_access19_put_fn,
    )
    from tampanda.symbolic.domains.access19.reachability import (
        _build_executor, _solve_access19_staging,
    )
    from tampanda.planners.grasp_planner import GraspType
    from tampanda.planners.linear_ik import LinearIKPlanner
    import mujoco
    import numpy as np

    scratch = tempfile.TemporaryDirectory(prefix="parfeas_access19_")
    builder, ws, cfg = make_access19_builder(scratch_dir=Path(scratch.name))
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    cube_half = float(env.get_object_half_size("ooi")[2])
    table_z = ws["shelf_interior"].level_z - cube_half
    executor = _build_executor(env, table_z=table_z,
                                       allowed_types=[GraspType.FRONT])

    shelf_home = _solve_access19_staging(env, ws, cfg)
    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
    pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                            cube_half_z=cube_half, lik=lik,
                                            home_qpos=shelf_home)
    put_fn = make_access19_put_fn(env, executor, ws, cfg,
                                          cube_half_z=cube_half, lik=lik,
                                          home_qpos=shelf_home)

    object_names = [f"blocker_{i}" for i in range(18)] + ["ooi"]

    _WORKER_STATE.update({
        "scratch": scratch,
        "env": env,
        "ws": ws,
        "cfg": cfg,
        "executor": executor,
        "pick_fn": pick_fn,
        "put_fn": put_fn,
        "home_qpos": shelf_home,
        "object_names": object_names,
        "fast": fast,
    })


def _layout_to_state(
    layout: Dict[str, str],
    held: Optional[str] = None,
) -> Dict[Tuple, bool]:
    """Reconstruct the minimal ``state`` dict needed by ``restore_state``.

    The bridge's ``ground_state`` includes every (occupied/empty) pair
    for the cell × object cross product (~3000 entries); pickling that
    per-task dominates dispatch cost.  ``restore_state`` only needs
    the ``(occupied cell obj)`` truths and any held fluent, so a
    compact layout suffices.

    Args:
        layout: ``{obj_name: cell_id}`` for placed objects.  Objects
            absent from the dict are considered parked.
        held: held object name, or ``None``.
    """
    state: Dict[Tuple, bool] = {}
    for obj, cell_id in layout.items():
        state[("occupied", cell_id, obj)] = True
    if held is not None:
        state[("holding", held)] = True
    return state


def _worker_check_action(
    payload: Tuple[Dict[str, str], Optional[str], Tuple],
) -> Dict[str, Any]:
    """Run a single (layout, held, action) check on this worker."""
    from tampanda.symbolic.domains.access19.feasibility import check_action

    layout, held, action = payload
    state = _layout_to_state(layout, held)
    t = time.perf_counter()
    res = check_action(
        _WORKER_STATE["env"], _WORKER_STATE["ws"], _WORKER_STATE["cfg"],
        state, action, _WORKER_STATE["object_names"],
        _WORKER_STATE["pick_fn"], _WORKER_STATE["put_fn"],
        executor=_WORKER_STATE["executor"],
        fast=_WORKER_STATE["fast"],
        home_qpos=_WORKER_STATE["home_qpos"],
    )
    res["_worker_pid"] = os.getpid()
    res["_worker_t"] = time.perf_counter() - t
    return res


def _worker_check_sequence(
    payload: Tuple[Dict[str, str], Optional[str], List[Tuple]],
) -> Dict[str, Any]:
    """Run a (layout, held, [action, ...]) sequence on this worker."""
    from tampanda.symbolic.domains.access19.feasibility import (
        check_action_sequence,
    )
    layout, held, actions = payload
    state = _layout_to_state(layout, held)
    return check_action_sequence(
        _WORKER_STATE["env"], _WORKER_STATE["ws"], _WORKER_STATE["cfg"],
        state, list(actions), _WORKER_STATE["object_names"],
        _WORKER_STATE["pick_fn"], _WORKER_STATE["put_fn"],
        executor=_WORKER_STATE["executor"],
        fast=_WORKER_STATE["fast"],
        home_qpos=_WORKER_STATE["home_qpos"],
    )


class ParallelFeasibilityChecker:
    """Persistent pool of access-19 feasibility workers.

    Use as a context manager so :meth:`close` runs on exit::

        with ParallelFeasibilityChecker(n_workers=8, fast=True) as ck:
            results = ck.check_batch(items)
    """

    def __init__(
        self,
        n_workers: int,
        *,
        fast: bool = True,
        start_method: Optional[str] = None,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.fast = fast

        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self._pool = ctx.Pool(
            n_workers,
            initializer=_worker_init,
            initargs=(fast,),
        )

    def check_batch(
        self,
        items: List[Tuple[Dict[str, str], Optional[str], Tuple]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        """Check a batch of ``(layout, held, action)`` tuples.

        ``layout`` is a ``{obj_name: cell_id}`` dict; ``held`` is the
        held object name or ``None``.  Returns results in input order.
        """
        return self._pool.map(_worker_check_action, items,
                                  chunksize=chunksize)

    def check_sequence_batch(
        self,
        items: List[Tuple[Dict[str, str], Optional[str], List[Tuple]]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        """Check a batch of action sequences."""
        return self._pool.map(_worker_check_sequence, items,
                                  chunksize=chunksize)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
