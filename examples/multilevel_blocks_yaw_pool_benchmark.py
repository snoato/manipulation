"""Measure put-upright per-check cost with the Phase 4 yaw pool.

Builds main env + executor + a MultilevelBlocksYawPool with N workers,
then runs put-upright (held-upright -> stack_L0/L1 at a cached-seed
cell) repeatedly with the pool both ON and OFF for comparison.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    _make_executor,
    check_action,
)
from tampanda.symbolic.domains.multilevel_blocks.parallel_yaw import (
    MultilevelBlocksYawPool,
)


N_CHECKS = 10
N_WORKERS = 8


def _measure(env, ws, cfg, executor, state, action, n=N_CHECKS):
    # warm-up
    check_action(env, ws, cfg, state, action, fast=True, executor=executor)
    t0 = time.perf_counter()
    last = None
    for _ in range(n):
        last = check_action(env, ws, cfg, state, action, fast=True,
                                executor=executor)
    return (time.perf_counter() - t0) / n, last


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    with tempfile.TemporaryDirectory(prefix="yawpool_bench_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)

        # Construct pool (~1.5s × n_workers setup).
        print(f"Building yaw pool with {N_WORKERS} workers ...")
        t_pool0 = time.perf_counter()
        pool = MultilevelBlocksYawPool(cfg, n_workers=N_WORKERS)
        pool_setup_s = time.perf_counter() - t_pool0
        print(f"  pool setup: {pool_setup_s:.1f}s\n")

        # Two executors: one WITHOUT pool, one WITH.
        executor_serial = _make_executor(env, ws, cfg, fast=True)
        executor_parallel = _make_executor(env, ws, cfg, fast=True,
                                                  yaw_pool=pool)

        # Put-upright state: held-upright + relevant adjacency
        # Sync pool to this state (broadcast restore_state) before
        # parallel measurement.  Need to do this PER unique state.
        state = {("held-upright", "oblong_0"): True}
        action = ("put-upright", "oblong_0", "stack_L0__7_3", "stack_L1__7_3")

        # Serial baseline
        print("Measuring SERIAL (no pool) ...")
        t_serial, res_serial = _measure(env, ws, cfg, executor_serial,
                                                state, action)
        print(f"  per-check: {t_serial*1000:.0f} ms  success={res_serial['success']}")

        # Parallel: need to sync workers' state to current main state.
        # Since each check_action calls restore_state on the main env,
        # we sync the pool BEFORE each (via a small modification — or
        # call pool.sync_state with the (in...) state per call).
        # For this benchmark we sync once with the held-* state.
        pool.sync_state(state)

        print("Measuring PARALLEL (yaw pool ON) ...")
        t_par, res_par = _measure(env, ws, cfg, executor_parallel,
                                            state, action)
        print(f"  per-check: {t_par*1000:.0f} ms  success={res_par['success']}")

        print()
        if t_par > 0:
            speedup = t_serial / t_par
            print(f"Speedup: {speedup:.2f}x  ({t_serial*1000:.0f} ms -> "
                      f"{t_par*1000:.0f} ms)")

        pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
