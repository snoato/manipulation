"""Multiprocessing pool around the confined_shelf feasibility checker.

For the rearrangement search / data generation the dominant cost is
per-action feasibility checking.  Each worker builds its own MuJoCo env
+ executor + LinearIK planner once (~1.5 s) and then services many
``(layout, held, action)`` probes.

Unlike access-19's fixed 19-object roster, confined_shelf has a variable
cylinder count, so the pool takes the ``ConfinedShelfConfig`` (and
optional colour groups) and rebuilds the matching scene in every worker.

Pass ``start_method="spawn"`` when constructing from inside a CUDA /
torch process (rgnet workers) — ``fork`` corrupts a CUDA context.

Public API::

    from tampanda.symbolic.domains.confined_shelf.parallel import (
        ParallelFeasibilityChecker,
    )

    with ParallelFeasibilityChecker(8, cfg, fast=True) as ck:
        results = ck.check_batch([(layout1, held1, action1), ...])
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tampanda.symbolic.domains.confined_shelf.env_builder import (
    ConfinedShelfConfig,
)


_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(cfg: ConfinedShelfConfig,
                 cylinder_color_groups: Optional[List[int]],
                 fast: bool) -> None:
    """Per-worker setup: build env + executor + LinearIK once."""
    from tampanda.symbolic.domains.confined_shelf.env_builder import (
        STAGING_HOME_QPOS, apply_runtime_tweaks, make_confined_shelf_builder,
    )
    from tampanda.symbolic.domains.confined_shelf.bridge import (
        make_confined_shelf_bridge,
    )
    from tampanda.symbolic.domains.confined_shelf.reachability import (
        _build_executor,
    )
    from tampanda.planners.linear_ik import LinearIKPlanner

    groups = (cylinder_color_groups
              if cylinder_color_groups is not None
              else [i % cfg.n_color_groups for i in range(cfg.n_cylinders)])
    color_names = ["red", "green", "yellow", "blue", "purple", "orange"]
    cyl_colors = [color_names[g] for g in groups]

    scratch = tempfile.TemporaryDirectory(prefix="parfeas_cshelf_")
    builder, ws, cfg = make_confined_shelf_builder(
        Path(scratch.name), cfg, cylinder_color_groups=groups)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    table_z = ws["shelf_interior"].level_z - cfg.cylinder_half_height
    executor = _build_executor(env, table_z)
    make_confined_shelf_bridge(env, ws, cfg, cyl_colors, executor=executor)
    lik = LinearIKPlanner(env, n_substeps=8, joint_check_steps=8)

    _WORKER_STATE.update({
        "scratch": scratch,
        "env": env,
        "ws": ws,
        "cfg": cfg,
        "executor": executor,
        "lik": lik,
        "home_qpos": STAGING_HOME_QPOS.copy(),
        "cylinder_names": [f"cyl_{i}" for i in range(cfg.n_cylinders)],
        "fast": fast,
    })


def _layout_to_state(
    layout: Dict[str, str],
    held: Optional[str] = None,
) -> Dict[Tuple, bool]:
    """Compact ``{cyl: cell_id}`` (+ held) → minimal ``restore_state`` dict."""
    state: Dict[Tuple, bool] = {}
    for cyl, cell_id in layout.items():
        state[("occupied", cell_id, cyl)] = True
    if held is not None:
        state[("holding", held)] = True
    return state


def _worker_check_action(
    payload: Tuple[Dict[str, str], Optional[str], Tuple],
) -> Dict[str, Any]:
    from tampanda.symbolic.domains.confined_shelf.feasibility import (
        check_action,
    )
    layout, held, action = payload
    state = _layout_to_state(layout, held)
    t = time.perf_counter()
    res = check_action(
        _WORKER_STATE["env"], _WORKER_STATE["ws"], _WORKER_STATE["cfg"],
        state, action, _WORKER_STATE["cylinder_names"],
        executor=_WORKER_STATE["executor"], lik=_WORKER_STATE["lik"],
        fast=_WORKER_STATE["fast"], home_qpos=_WORKER_STATE["home_qpos"],
    )
    res["_worker_pid"] = os.getpid()
    res["_worker_t"] = time.perf_counter() - t
    return res


def _worker_check_sequence(
    payload: Tuple[Dict[str, str], Optional[str], List[Tuple]],
) -> Dict[str, Any]:
    from tampanda.symbolic.domains.confined_shelf.feasibility import (
        check_action_sequence,
    )
    layout, held, actions = payload
    state = _layout_to_state(layout, held)
    return check_action_sequence(
        _WORKER_STATE["env"], _WORKER_STATE["ws"], _WORKER_STATE["cfg"],
        state, list(actions), _WORKER_STATE["cylinder_names"],
        executor=_WORKER_STATE["executor"], lik=_WORKER_STATE["lik"],
        fast=_WORKER_STATE["fast"], home_qpos=_WORKER_STATE["home_qpos"],
    )


class ParallelFeasibilityChecker:
    """Persistent pool of confined_shelf feasibility workers.

    Args:
        n_workers: pool size.
        config: ``ConfinedShelfConfig`` — workers rebuild this exact
            scene (cylinder count, grid, dimensions).
        cylinder_color_groups: optional per-cylinder colour group; the
            colours don't affect feasibility but the worker needs a
            consistent roster.
        fast: FAST (prefilter + LinearIK) vs FULL (RRT* + settle).
        start_method: ``"spawn"`` from inside a torch/CUDA process.
    """

    def __init__(
        self,
        n_workers: int,
        config: ConfinedShelfConfig,
        *,
        cylinder_color_groups: Optional[List[int]] = None,
        fast: bool = True,
        start_method: Optional[str] = None,
    ):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.fast = fast
        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self._pool = ctx.Pool(
            n_workers, initializer=_worker_init,
            initargs=(config, cylinder_color_groups, fast),
        )

    def check_batch(
        self,
        items: List[Tuple[Dict[str, str], Optional[str], Tuple]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        """Check ``(layout, held, action)`` tuples; results in input order."""
        return self._pool.map(_worker_check_action, items, chunksize=chunksize)

    def check_sequence_batch(
        self,
        items: List[Tuple[Dict[str, str], Optional[str], List[Tuple]]],
        *,
        chunksize: int = 1,
    ) -> List[Dict[str, Any]]:
        return self._pool.map(_worker_check_sequence, items,
                              chunksize=chunksize)

    def close(self) -> None:
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
