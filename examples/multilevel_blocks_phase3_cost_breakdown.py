"""Per-component cost breakdown for the fast feasibility check after
Phase 1+2+3 work.

Builds env + executor once, then runs N identical pick-cube checks.
Reports:
  * setup_s — one-time env build + executor + bridge init.
  * total_check_s / per_check_s — wall-clock per check, amortised.
  * mj_forward_calls_per_check + mj_forward_time_per_check.
  * mink_ik_calls_per_check (approx, via instrumented wrapper).
  * prefilter_time_per_check (negligible — direct call).
  * Python-overhead bucket = per_check - mj_forward - mink_ik.
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


N_CHECKS = 5

# Mink IK call counter — patch the executor's mink instance.
_mink_state = {"calls": 0, "time": 0.0}


def _patch_mink(env):
    orig_converge = env.ik.converge_ik
    def patched(dt):
        t = time.perf_counter()
        out = orig_converge(dt)
        _mink_state["time"] += time.perf_counter() - t
        _mink_state["calls"] += 1
        return out
    env.ik.converge_ik = patched
    return orig_converge


def _patch_mj_forward():
    state = {"calls": 0, "time": 0.0, "orig": mujoco.mj_forward}
    def patched(m, d):
        t = time.perf_counter()
        state["orig"](m, d)
        state["time"] += time.perf_counter() - t
        state["calls"] += 1
    mujoco.mj_forward = patched
    return state


def _unpatch_mj_forward(state):
    mujoco.mj_forward = state["orig"]


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)

    with tempfile.TemporaryDirectory(prefix="phase3_cost_") as scratch:
        # ---- setup (one-time) ----
        t_setup0 = time.perf_counter()
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        executor = _make_executor(env, ws, cfg, fast=True)
        setup_s = time.perf_counter() - t_setup0

        # ---- patch counters ----
        _patch_mink(env)
        mj_state = _patch_mj_forward()

        # ---- N identical checks (amortise setup) ----
        # Single cube on parts table; pick it.  All other 37 blocks parked.
        state_dict = {("in", "cube_0", "parts__7_7"): True}
        action = ("pick-cube", "cube_0", "parts__7_7")

        # Warm up — first check pays cache misses; discard.
        check_action(env, ws, cfg, state_dict, action, fast=True,
                          executor=executor)
        _mink_state["calls"] = 0
        _mink_state["time"] = 0.0
        mj_state["calls"] = 0
        mj_state["time"] = 0.0

        t_run0 = time.perf_counter()
        for _ in range(N_CHECKS):
            check_action(env, ws, cfg, state_dict, action, fast=True,
                              executor=executor)
        total_check_s = time.perf_counter() - t_run0

        _unpatch_mj_forward(mj_state)

        per_check_s = total_check_s / N_CHECKS
        per_check_mj_calls = mj_state["calls"] / N_CHECKS
        per_check_mj_time = mj_state["time"] / N_CHECKS
        per_check_ik_calls = _mink_state["calls"] / N_CHECKS
        per_check_ik_time = _mink_state["time"] / N_CHECKS
        per_check_other = per_check_s - per_check_mj_time - per_check_ik_time

        print(f"scene: nbody={env.model.nbody} ngeom={env.model.ngeom}")
        print()
        print(f"setup (one-time):        {setup_s*1000:6.0f} ms")
        print(f"checks measured:         {N_CHECKS} (warm-up discarded)")
        print()
        print(f"per-check breakdown (fast-mode pick-cube, 1 placed / 37 parked):")
        print(f"  total wall-clock:      {per_check_s*1000:6.0f} ms")
        print(f"    mj_forward          {per_check_mj_time*1000:6.0f} ms  "
                  f"({per_check_mj_calls:6.1f} calls × "
                  f"{per_check_mj_time/max(per_check_mj_calls,1)*1000:.2f} ms)")
        print(f"    mink converge_ik    {per_check_ik_time*1000:6.0f} ms  "
                  f"({per_check_ik_calls:6.1f} calls × "
                  f"{per_check_ik_time/max(per_check_ik_calls,1)*1000:.2f} ms)")
        print(f"    other (Python +     {per_check_other*1000:6.0f} ms")
        print(f"      restore_state +")
        print(f"      prefilter +")
        print(f"      mink solve_ik internals)")
        print()
        share_mj = per_check_mj_time / per_check_s * 100
        share_ik = per_check_ik_time / per_check_s * 100
        share_other = per_check_other / per_check_s * 100
        print(f"share: mj_forward={share_mj:.1f}%  "
                  f"mink_ik={share_ik:.1f}%  "
                  f"other={share_other:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
