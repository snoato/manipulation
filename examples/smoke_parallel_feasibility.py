"""Smoke test: ParallelFeasibilityChecker with N workers.

Generates a batch of (state, action) tuples, runs them single-process
(baseline) and via the parallel pool, reports speedup + agreement.
"""
from __future__ import annotations

import argparse
import tempfile
import pathlib
import time
from typing import List, Tuple, Dict

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
    oblong_block_name,
    ParallelFeasibilityChecker,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    check_action, _make_executor,
)
from tampanda.symbolic.workspace import Cell


def _build_items(n_items: int) -> List[Tuple[Dict[str, List[str]], None, Tuple]]:
    """Build a batch of ``(layout, held, action)`` tuples for testing."""
    items = []
    for i in range(n_items):
        ix, iy = i % 8, (i // 8) % 8
        cell = Cell("stack_L0", ix, iy)
        layout = {cube_block_name(0): [cell.id]}
        action = ("pick-cube", cube_block_name(0), cell.id)
        items.append((layout, None, action))
    return items


def _run_single(cfg: MultilevelBlocksConfig,
                  items: List[Tuple[Dict[str, List[str]], None, Tuple]]
                  ) -> Tuple[List[Dict], float]:
    """Reference single-process pass — uses the same compact layout API
    via _layout_to_state."""
    from tampanda.symbolic.domains.multilevel_blocks.parallel import (
        _layout_to_state,
    )
    with tempfile.TemporaryDirectory() as td:
        builder, ws, cfg2 = make_multilevel_blocks_builder(
            scratch_dir=pathlib.Path(td), config=cfg)
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg2)
        executor = _make_executor(env, ws, cfg2, fast=True)
        t = time.perf_counter()
        results = []
        for layout, held, action in items:
            state = _layout_to_state(layout, held)
            r = check_action(env, ws, cfg2, state, action,
                                  fast=True, executor=executor)
            results.append(r)
        return results, time.perf_counter() - t


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--start-method", type=str, default=None,
                          choices=[None, "fork", "spawn", "forkserver"])
    args = parser.parse_args()

    cfg = MultilevelBlocksConfig(n_cubes=2, n_oblong=2, n_long=2)
    items = _build_items(args.n_items)
    print(f"Built {len(items)} items.")

    # Single-process baseline
    print(f"\n=== Single-process baseline ===")
    single_results, single_t = _run_single(cfg, items)
    print(f"  total={single_t:.2f} s  "
              f"({single_t/len(items)*1000:.0f} ms/item)")

    # Parallel
    print(f"\n=== Parallel (n_workers={args.n_workers}) ===")
    t = time.perf_counter()
    with ParallelFeasibilityChecker(
        n_workers=args.n_workers, config=cfg, fast=True,
        start_method=args.start_method,
    ) as ck:
        startup = time.perf_counter() - t
        print(f"  startup: {startup:.2f} s")
        t2 = time.perf_counter()
        par_results = ck.check_batch(
            items, chunksize=max(1, len(items) // args.n_workers))
        par_t = time.perf_counter() - t2
        print(f"  check:   {par_t:.2f} s  "
                  f"({par_t/len(items)*1000:.0f} ms/item)")
    print(f"  speedup: {single_t/par_t:.1f}x (vs single, excl. startup)")
    print(f"  speedup: {single_t/(par_t+startup):.1f}x "
              f"(incl. startup)")

    # Agreement
    agree = sum(1 for s, p in zip(single_results, par_results)
                    if s["success"] == p["success"])
    print(f"\nAGREEMENT (single-action): "
              f"{agree}/{len(items)} ({100*agree/len(items):.1f}%)")

    # Also exercise check_sequence_batch
    seq_items = [(layout, held, [action,
                                          ("put-cube", action[1], "stack_L0__4_4")])
                      for layout, held, action in items[:32]]
    t = time.perf_counter()
    with ParallelFeasibilityChecker(
        n_workers=args.n_workers, config=cfg, fast=True,
        start_method=args.start_method,
    ) as ck:
        seq_results = ck.check_sequence_batch(seq_items)
    seq_t = time.perf_counter() - t
    print(f"\nSEQUENCE BATCH ({len(seq_items)} items, n_workers={args.n_workers}): "
              f"{seq_t:.2f} s  ({seq_t/len(seq_items)*1000:.0f} ms/item)")
    print(f"  successes: {sum(1 for r in seq_results if r['success'])}/{len(seq_items)}")

    # Per-worker timing breakdown
    from collections import defaultdict
    pid_times = defaultdict(list)
    for r in par_results:
        pid_times[r.get("_worker_pid", -1)].append(r.get("_worker_t", 0))
    print(f"\nPer-worker breakdown:")
    for pid, times in pid_times.items():
        print(f"  pid={pid}  n={len(times)}  "
                  f"mean={sum(times)/len(times)*1000:.0f} ms  "
                  f"min={min(times)*1000:.0f} ms  "
                  f"max={max(times)*1000:.0f} ms  "
                  f"first={times[0]*1000:.0f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
