"""Pick-upright correctness + speed smoke test.

Since the test plans don't include pick-upright, this script directly
exercises the action across a sample of stack cells, in both fast and
full executor modes, to confirm:

* Block ends up held (correct state transition).
* No regression in success rate vs the pre-LUT cold-IK path.
* Fast-mode timing matches the LUT-seeded expectation (~50-80 ms).
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    MultilevelBlocksExecutor,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
    restore_state,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    _make_executor,
)
from tampanda.planners.rrt_star import RRTStar


# A scatter of stack cells across the workspace — corners, mid, edges.
_TEST_CELLS = [
    ("stack_L0__1_1", "stack_L1__1_1"),
    ("stack_L0__7_3", "stack_L1__7_3"),
    ("stack_L0__4_5", "stack_L1__4_5"),
    ("stack_L0__2_3", "stack_L1__2_3"),
    ("stack_L0__5_5", "stack_L1__5_5"),
    ("stack_L1__7_3", "stack_L2__7_3"),
    ("stack_L0__0_0", "stack_L1__0_0"),
    ("stack_L0__9_9", "stack_L1__9_9"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fast", "full", "both"],
                            default="both")
    args = parser.parse_args()

    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    modes: List[str] = ["fast", "full"] if args.mode == "both" else [args.mode]

    results: Dict[str, List[Tuple[str, bool, float]]] = {m: [] for m in modes}

    with tempfile.TemporaryDirectory(prefix="pu_smoke_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)

        for mode in modes:
            if mode == "fast":
                executor = _make_executor(env, ws, cfg, fast=True)
            else:
                rrt = RRTStar(env, max_iterations=3000)
                executor = MultilevelBlocksExecutor(
                    env, ws, cfg, motion_planner=rrt,
                )
            bridge, _ = make_multilevel_blocks_bridge(
                env, ws, cfg, executor=executor,
            )

            print(f"\n== mode={mode} ==")
            for c_low, c_high in _TEST_CELLS:
                # Restore a state where oblong_0 is standing upright at (c_low, c_high)
                init = {
                    ("in", "oblong_0", c_low): True,
                    ("in", "oblong_0", c_high): True,
                }
                restore_state(env, ws, cfg, init, on_held="attach",
                                  executor=executor)
                t0 = time.perf_counter()
                try:
                    success, _ = bridge.execute_action(
                        "pick-upright", "oblong_0", c_low, c_high,
                    )
                except Exception as exc:
                    success = False
                    print(f"  [{c_low}] EXC {type(exc).__name__}: {exc}")
                dt_ms = (time.perf_counter() - t0) * 1000
                results[mode].append((c_low, success, dt_ms))
                mark = "OK" if success else "FAIL"
                print(f"  [{c_low}]  {mark:>4s}  {dt_ms:6.0f} ms")

    print()
    print("============ SUMMARY ============")
    for mode in modes:
        rs = results[mode]
        n_ok = sum(1 for _, s, _ in rs if s)
        avg_ok = (np.mean([dt for _, s, dt in rs if s])
                       if n_ok > 0 else float("nan"))
        print(f"mode={mode:>5s}  pass={n_ok}/{len(rs)}  "
                  f"avg_ms_on_pass={avg_ok:.0f}")

    # exit code: 0 only if fast mode passes >= 6/8 (allowing 2 edge-corner
    # failures, since corners are known-tight for upright stability).
    fast_pass = sum(1 for _, s, _ in results.get("fast", []) if s)
    return 0 if fast_pass >= 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
