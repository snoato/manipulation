"""Per-action cost breakdown after Phase 1+2+3.

Measures fast-mode per-check wall-clock for representative action types
on the rgnet 38-block scene.  Amortises setup over N checks per case.

Cases:
  * pick-cube                                — 1-cell, top-down, 4 yaws
  * put-cube  (from held-cube)              — 1-cell, top-down, 4 yaws
  * pick-flat-x                              — 2-cell, top-down, 2 yaws
  * put-flat-x (from held-flat-x)           — 2-cell, top-down, 2 yaws
  * pick-upright                             — 2-cell, front, 8 yaws
  * put-upright (from held-upright)         — 2-cell, front + traverse
  * put-long-upright (from held-upright)    — 3-cell upright (heaviest)
  * pick-cube INFEAS (jaws blocked)         — prefilter rejection

Reports per case:
  total wall-clock, mj_forward share, mink_ik share, other.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

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

_mink_state = {"calls": 0, "time": 0.0}
_mj_state = {"calls": 0, "time": 0.0, "orig": None}


def _patch_mink(env):
    orig = env.ik.converge_ik
    def patched(dt):
        t = time.perf_counter()
        out = orig(dt)
        _mink_state["time"] += time.perf_counter() - t
        _mink_state["calls"] += 1
        return out
    env.ik.converge_ik = patched


def _patch_mj_forward():
    _mj_state["orig"] = mujoco.mj_forward
    def patched(m, d):
        t = time.perf_counter()
        _mj_state["orig"](m, d)
        _mj_state["time"] += time.perf_counter() - t
        _mj_state["calls"] += 1
    mujoco.mj_forward = patched


def _unpatch_mj_forward():
    mujoco.mj_forward = _mj_state["orig"]


def _reset_counters():
    _mink_state["calls"] = 0
    _mink_state["time"] = 0.0
    _mj_state["calls"] = 0
    _mj_state["time"] = 0.0


def _measure(env, ws, cfg, executor, label: str,
                state: Dict, action: Tuple,
                n: int = N_CHECKS) -> Dict:
    # warmup (discard)
    res = check_action(env, ws, cfg, state, action, fast=True,
                            executor=executor)
    _reset_counters()
    t0 = time.perf_counter()
    last_res = None
    for _ in range(n):
        last_res = check_action(env, ws, cfg, state, action, fast=True,
                                       executor=executor)
    elapsed = (time.perf_counter() - t0) / n
    return {
        "label": label,
        "action_type": action[0],
        "success": last_res["success"],
        "error": last_res.get("error"),
        "per_check_s": elapsed,
        "mj_calls": _mj_state["calls"] / n,
        "mj_time_s": _mj_state["time"] / n,
        "ik_calls": _mink_state["calls"] / n,
        "ik_time_s": _mink_state["time"] / n,
    }


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    with tempfile.TemporaryDirectory(prefix="phase3_costs_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        executor = _make_executor(env, ws, cfg, fast=True)

        _patch_mink(env)
        _patch_mj_forward()

        cases = []

        # --- pick-cube (FEAS) ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "pick-cube (1 cube on parts)",
            {("in", "cube_0", "parts__7_7"): True},
            ("pick-cube", "cube_0", "parts__7_7"),
        ))

        # --- put-cube from held-cube ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "put-cube to stack_L0__5_5 (held-cube)",
            {("held-cube", "cube_0"): True},
            ("put-cube", "cube_0", "stack_L0__5_5"),
        ))

        # --- pick-flat-x on parts ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "pick-flat-x at parts__3_5/parts__4_5",
            {("in", "oblong_0", "parts__3_5"): True,
             ("in", "oblong_0", "parts__4_5"): True},
            ("pick-flat-x", "oblong_0", "parts__3_5", "parts__4_5"),
        ))

        # --- put-flat-x to stack from held ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "put-flat-x to stack_L0__3_4/stack_L0__4_4 (held)",
            {("held-flat-x", "oblong_0"): True},
            ("put-flat-x", "oblong_0", "stack_L0__3_4", "stack_L0__4_4"),
        ))

        # --- pick-upright from a stack column ---
        # Need an upright block already on the stack at L0+L1.
        cases.append(_measure(
            env, ws, cfg, executor,
            "pick-upright at stack_L0__5_5/stack_L1__5_5",
            {("in", "oblong_0", "stack_L0__5_5"): True,
             ("in", "oblong_0", "stack_L1__5_5"): True},
            ("pick-upright", "oblong_0", "stack_L0__5_5", "stack_L1__5_5"),
        ))

        # --- put-upright (the heavy one) ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "put-upright to stack_L0__5_5/stack_L1__5_5 (held)",
            {("held-upright", "oblong_0"): True},
            ("put-upright", "oblong_0", "stack_L0__5_5", "stack_L1__5_5"),
        ))

        # --- put-long-upright ---
        cases.append(_measure(
            env, ws, cfg, executor,
            "put-long-upright to L0/L1/L2 at (5,5) (held)",
            {("held-upright", "long_0"): True},
            ("put-long-upright", "long_0",
             "stack_L0__5_5", "stack_L1__5_5", "stack_L2__5_5"),
        ))

        # --- pick-cube INFEAS via prefilter ---
        infeas_state = {
            ("in", "cube_0", "stack_L0__5_5"): True,
            ("in", "cube_1", "stack_L0__4_5"): True,   # W
            ("in", "cube_2", "stack_L0__6_5"): True,   # E
            ("in", "cube_3", "stack_L0__5_4"): True,   # S
            ("in", "cube_4", "stack_L0__5_6"): True,   # N
        }
        cases.append(_measure(
            env, ws, cfg, executor,
            "pick-cube INFEAS (all 4 neighbours, prefilter rejects)",
            infeas_state,
            ("pick-cube", "cube_0", "stack_L0__5_5"),
        ))

        _unpatch_mj_forward()

        # ---- Report ----
        print(f"scene: nbody={env.model.nbody}  amortised over {N_CHECKS} checks/case")
        print()
        hdr = ("label", "ok", "total_ms", "mj_ms", "mj_n",
                  "ik_ms", "ik_n", "other_ms", "%mj", "%ik")
        widths = (54, 2, 8, 6, 5, 6, 4, 8, 4, 4)
        line = " ".join(f"{h:>{w}s}" if isinstance(h, str)
                              else f"{h:>{w}d}"
                              for h, w in zip(hdr, widths))
        print(line)
        print("-" * (sum(widths) + len(widths)))
        for r in cases:
            t = r["per_check_s"]
            mj = r["mj_time_s"]
            ik = r["ik_time_s"]
            other = t - mj - ik
            row = (
                r["label"][:54],
                "✓" if r["success"] else "✗",
                t * 1000,
                mj * 1000,
                r["mj_calls"],
                ik * 1000,
                r["ik_calls"],
                other * 1000,
                mj / max(t, 1e-9) * 100,
                ik / max(t, 1e-9) * 100,
            )
            print(f"{row[0]:54s} {row[1]:>2s} "
                      f"{row[2]:>8.1f} {row[3]:>6.1f} {row[4]:>5.0f} "
                      f"{row[5]:>6.1f} {row[6]:>4.0f} {row[7]:>8.1f} "
                      f"{row[8]:>4.0f} {row[9]:>4.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
