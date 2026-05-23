"""Compare check_fast vs check_full on a multilevel_blocks action sample.

For each (state, action) test case:
  * Restore MuJoCo state from the PDDL ground state.
  * Run check_fast.
  * Restore again.
  * Run check_full (ground truth).
  * Record (fast_result, full_result, fast_time, full_time).

Outputs:
  * Confusion matrix (Fast says X, Full says Y).
  * Runtime stats per shape.
  * List of disagreement cases (false positives are the dangerous ones).

The default test set covers:
  * cube / flat-x / flat-y picks at corners, edges, centre of L0
  * upright picks across L0/L1 at corners
  * long-x / long-y picks at L0 centre
  * INFEASIBLE setups: pick blocked by stacked obstacle above

Use ``--cases N`` to limit the sample size, ``--full-only`` /
``--fast-only`` to skip the comparison.
"""
from __future__ import annotations

import argparse
import tempfile
import pathlib
from typing import Dict, List

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
    oblong_block_name,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    check_action, check_action_sequence, _make_executor,
)
from tampanda.symbolic.workspace import Cell


def _build_layout_and_state(env, ws, cfg, bridge, objects,
                                  blocks: Dict[str, List[Cell]]):
    """Place blocks at the requested cells, return the grounded state."""
    all_names = ([cube_block_name(i) for i in range(cfg.n_cubes)]
                  + [oblong_block_name(i) for i in range(cfg.n_oblong)]
                  + [long_block_name(i) for i in range(cfg.n_long)])
    for n in all_names:
        env.set_object_pose(n, [100.0, 0.0, 0.05])
    for block, cells in blocks.items():
        poses = np.array([ws.pose_for(c) for c in cells])
        centroid = poses.mean(axis=0)
        if len(cells) == 1:
            quat = [1, 0, 0, 0]
        elif len({c.region for c in cells}) > 1:
            quat = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
        elif len({c.ix for c in cells}) > 1:
            quat = [1, 0, 0, 0]
        else:
            quat = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        env.set_object_pose(block, centroid, quat)
    env.forward()
    return bridge.ground_state(objects)


def _test_cases() -> List[Dict]:
    cases: List[Dict] = []

    # Cube picks at corners + centre of L0.
    for ix, iy in [(0, 0), (7, 0), (0, 7), (7, 7), (4, 4),
                       (0, 4), (4, 0), (4, 7), (7, 4)]:
        c = Cell("stack_L0", ix, iy)
        cases.append({
            "label": f"pick-cube L0({ix},{iy})",
            "shape": "cube",
            "blocks": {cube_block_name(0): [c]},
            "action": ("pick-cube", cube_block_name(0), c.id),
        })

    # Flat-x picks.
    for ix, iy in [(0, 0), (5, 0), (0, 4), (5, 7), (3, 3)]:
        c1 = Cell("stack_L0", ix, iy)
        c2 = Cell("stack_L0", ix + 1, iy)
        cases.append({
            "label": f"pick-flat-x L0 ({ix},{iy})-({ix+1},{iy})",
            "shape": "flat-x",
            "blocks": {oblong_block_name(0): [c1, c2]},
            "action": ("pick-flat-x", oblong_block_name(0), c1.id, c2.id),
        })

    # Flat-y picks.
    for ix, iy in [(0, 0), (7, 0), (4, 3), (3, 5), (0, 6)]:
        c1 = Cell("stack_L0", ix, iy)
        c2 = Cell("stack_L0", ix, iy + 1)
        cases.append({
            "label": f"pick-flat-y L0 ({ix},{iy})-({ix},{iy+1})",
            "shape": "flat-y",
            "blocks": {oblong_block_name(0): [c1, c2]},
            "action": ("pick-flat-y", oblong_block_name(0), c1.id, c2.id),
        })

    # Upright picks.
    for ix, iy in [(0, 0), (7, 0), (4, 4), (3, 7)]:
        c_low = Cell("stack_L0", ix, iy)
        c_high = Cell("stack_L1", ix, iy)
        cases.append({
            "label": f"pick-upright L0/L1 ({ix},{iy})",
            "shape": "upright",
            "blocks": {oblong_block_name(0): [c_low, c_high]},
            "action": ("pick-upright", oblong_block_name(0),
                          c_low.id, c_high.id),
        })

    # Long-x picks at L0.
    for ix, iy in [(0, 0), (4, 4), (5, 3)]:
        cs = [Cell("stack_L0", ix + i, iy) for i in range(3)]
        cases.append({
            "label": f"pick-long-x L0 ({ix},{iy})..({ix+2},{iy})",
            "shape": "long-x",
            "blocks": {long_block_name(0): cs},
            "action": ("pick-long-x", long_block_name(0),
                          cs[0].id, cs[1].id, cs[2].id),
        })

    # Long-y picks at L0.
    for ix, iy in [(0, 0), (4, 4), (3, 5)]:
        cs = [Cell("stack_L0", ix, iy + i) for i in range(3)]
        cases.append({
            "label": f"pick-long-y L0 ({ix},{iy})..({ix},{iy+2})",
            "shape": "long-y",
            "blocks": {long_block_name(0): cs},
            "action": ("pick-long-y", long_block_name(0),
                          cs[0].id, cs[1].id, cs[2].id),
        })

    # Infeasible cases — block under a stack tower.
    for ix, iy in [(4, 4), (0, 0)]:
        c = Cell("stack_L0", ix, iy)
        c_above = Cell("stack_L1", ix, iy)
        cases.append({
            "label": f"INFEASIBLE pick-cube L0({ix},{iy}) under tower",
            "shape": "cube",
            "blocks": {cube_block_name(0): [c],
                          cube_block_name(1): [c_above]},
            "action": ("pick-cube", cube_block_name(0), c.id),
        })

    # ============================================================
    # PUT cases — each is a pick+put sequence.  The pick sets up the
    # held state; the put is the action under test.
    # ============================================================

    # Cube pick+put: source at L0 (3,3), target at various L0 cells.
    for tgt_ix, tgt_iy in [(0, 0), (7, 0), (4, 4), (5, 6)]:
        src = Cell("stack_L0", 3, 3)
        tgt = Cell("stack_L0", tgt_ix, tgt_iy)
        cases.append({
            "label": f"pick+put-cube L0(3,3)->L0({tgt_ix},{tgt_iy})",
            "shape": "put-cube",
            "blocks": {cube_block_name(0): [src]},
            "action": ("pick-cube", cube_block_name(0), src.id),
            "put_target": ("put-cube", cube_block_name(0), tgt.id),
        })

    # Flat-x pick+put: source flat-x at L0, target at L0 different cells.
    for tgt_ix, tgt_iy in [(2, 4), (5, 2), (0, 6)]:
        s1 = Cell("stack_L0", 0, 0); s2 = Cell("stack_L0", 1, 0)
        t1 = Cell("stack_L0", tgt_ix, tgt_iy)
        t2 = Cell("stack_L0", tgt_ix + 1, tgt_iy)
        cases.append({
            "label": f"pick+put-flat-x L0(0,0)-(1,0)->L0({tgt_ix},{tgt_iy})",
            "shape": "put-flat-x",
            "blocks": {oblong_block_name(0): [s1, s2]},
            "action": ("pick-flat-x", oblong_block_name(0), s1.id, s2.id),
            "put_target": ("put-flat-x", oblong_block_name(0), t1.id, t2.id),
        })

    # Flat-y pick+put.
    for tgt_ix, tgt_iy in [(2, 3), (5, 0), (0, 5)]:
        s1 = Cell("stack_L0", 0, 0); s2 = Cell("stack_L0", 0, 1)
        t1 = Cell("stack_L0", tgt_ix, tgt_iy)
        t2 = Cell("stack_L0", tgt_ix, tgt_iy + 1)
        cases.append({
            "label": f"pick+put-flat-y L0(0,0)-(0,1)->L0({tgt_ix},{tgt_iy})",
            "shape": "put-flat-y",
            "blocks": {oblong_block_name(0): [s1, s2]},
            "action": ("pick-flat-y", oblong_block_name(0), s1.id, s2.id),
            "put_target": ("put-flat-y", oblong_block_name(0), t1.id, t2.id),
        })

    # Upright pick+put.  Source upright at L0/L1 (3,3); target at
    # various corners.
    for tgt_ix, tgt_iy in [(0, 0), (4, 4), (5, 3)]:
        s_low = Cell("stack_L0", 3, 3); s_high = Cell("stack_L1", 3, 3)
        t_low = Cell("stack_L0", tgt_ix, tgt_iy)
        t_high = Cell("stack_L1", tgt_ix, tgt_iy)
        cases.append({
            "label": f"pick+put-upright L0/L1(3,3)->L0/L1({tgt_ix},{tgt_iy})",
            "shape": "put-upright",
            "blocks": {oblong_block_name(0): [s_low, s_high]},
            "action": ("pick-upright", oblong_block_name(0),
                          s_low.id, s_high.id),
            "put_target": ("put-upright", oblong_block_name(0),
                                t_low.id, t_high.id),
        })

    # Long-x pick+put.
    for tgt_ix, tgt_iy in [(0, 4), (4, 4)]:
        src_cells = [Cell("stack_L0", i, 0) for i in range(3)]
        tgt_cells = [Cell("stack_L0", tgt_ix + i, tgt_iy) for i in range(3)]
        cases.append({
            "label": f"pick+put-long-x L0(0..2,0)->L0({tgt_ix}..{tgt_ix+2},{tgt_iy})",
            "shape": "put-long-x",
            "blocks": {long_block_name(0): src_cells},
            "action": ("pick-long-x", long_block_name(0),
                          *(c.id for c in src_cells)),
            "put_target": ("put-long-x", long_block_name(0),
                                *(c.id for c in tgt_cells)),
        })

    # Long-y pick+put.
    for tgt_ix, tgt_iy in [(3, 0), (5, 4)]:
        src_cells = [Cell("stack_L0", 0, i) for i in range(3)]
        tgt_cells = [Cell("stack_L0", tgt_ix, tgt_iy + i) for i in range(3)]
        cases.append({
            "label": f"pick+put-long-y L0(0,0..2)->L0({tgt_ix},{tgt_iy}..{tgt_iy+2})",
            "shape": "put-long-y",
            "blocks": {long_block_name(0): src_cells},
            "action": ("pick-long-y", long_block_name(0),
                          *(c.id for c in src_cells)),
            "put_target": ("put-long-y", long_block_name(0),
                                *(c.id for c in tgt_cells)),
        })

    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=None,
                          help="Limit test sample size")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--fast-only", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        cfg = MultilevelBlocksConfig(n_cubes=2, n_oblong=2, n_long=2)
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=pathlib.Path(td), config=cfg)
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)

        cases = _test_cases()
        if args.cases is not None:
            cases = cases[: args.cases]

        # Pre-build both executors and reuse across cases.  This is the
        # realistic BFS pattern: one executor lives for the lifetime of
        # the run, and `restore_state` resets the world between calls.
        fast_executor = (None if args.full_only
                              else _make_executor(env, ws, cfg, fast=True))
        full_executor = (None if args.fast_only
                              else _make_executor(env, ws, cfg, fast=False))

        rows: List[Dict] = []
        for i, case in enumerate(cases):
            label = case["label"]
            row: Dict = {"label": label, "shape": case["shape"]}
            put_target = case.get("put_target")

            if not args.full_only:
                state = _build_layout_and_state(env, ws, cfg, bridge,
                                                      objects, case["blocks"])
                if put_target is not None:
                    res = check_action_sequence(
                        env, ws, cfg, state,
                        [case["action"], put_target],
                        fast=True)
                else:
                    res = check_action(env, ws, cfg, state, case["action"],
                                             fast=True, executor=fast_executor)
                row["fast_ok"] = res["success"]
                row["fast_t"] = res["elapsed_s"]
                row["fast_err"] = res.get("error") or (
                    None if res["success"] else
                    next((r["error"] for r in res.get("per_action", [])
                              if not r["success"]), None))

            if not args.fast_only:
                state = _build_layout_and_state(env, ws, cfg, bridge,
                                                      objects, case["blocks"])
                if put_target is not None:
                    res = check_action_sequence(
                        env, ws, cfg, state,
                        [case["action"], put_target],
                        fast=False)
                else:
                    res = check_action(env, ws, cfg, state, case["action"],
                                             fast=False, executor=full_executor)
                row["full_ok"] = res["success"]
                row["full_t"] = res["elapsed_s"]
                row["full_err"] = res.get("error") or (
                    None if res["success"] else
                    next((r["error"] for r in res.get("per_action", [])
                              if not r["success"]), None))

            rows.append(row)
            f_ok = row.get("fast_ok")
            u_ok = row.get("full_ok")
            f_str = ("OK" if f_ok else "FAIL") + f" {row.get('fast_t', 0):.2f}s" \
                if f_ok is not None else "--"
            u_str = ("OK" if u_ok else "FAIL") + f" {row.get('full_t', 0):.2f}s" \
                if u_ok is not None else "--"
            agree = "==" if f_ok == u_ok else "!="
            print(f"  [{i+1:3d}/{len(cases)}] {label:50s}  "
                      f"fast={f_str:14s}  full={u_str:14s}  {agree}")

        if not (args.fast_only or args.full_only):
            tp = sum(1 for r in rows if r["fast_ok"] and r["full_ok"])
            tn = sum(1 for r in rows
                          if not r["fast_ok"] and not r["full_ok"])
            fp = sum(1 for r in rows
                          if r["fast_ok"] and not r["full_ok"])
            fn = sum(1 for r in rows
                          if not r["fast_ok"] and r["full_ok"])
            agree = tp + tn
            total = len(rows)
            fast_t = sum(r["fast_t"] for r in rows)
            full_t = sum(r["full_t"] for r in rows)
            print()
            print("=" * 70)
            print(f"AGREEMENT:   {agree}/{total} ({100 * agree / total:.1f}%)")
            print(f"  TP (fast=T, full=T): {tp:4d}")
            print(f"  TN (fast=F, full=F): {tn:4d}")
            print(f"  FP (fast=T, full=F): {fp:4d}  <- DANGEROUS")
            print(f"  FN (fast=F, full=T): {fn:4d}  <- conservative (OK)")
            print(f"TIMING: fast={fast_t:.1f}s ({fast_t/total*1000:.0f}ms/case)  "
                      f"full={full_t:.1f}s ({full_t/total*1000:.0f}ms/case)  "
                      f"speedup={full_t/max(fast_t, 1e-9):.1f}x")

            # Per-shape breakdown
            shapes = sorted({r["shape"] for r in rows})
            print()
            print("PER-SHAPE breakdown:")
            print(f"  {'shape':<10s} {'cases':>5s}  {'agree':>6s}  "
                      f"{'fp':>3s}  {'fn':>3s}  {'fast_ms':>8s}  {'full_ms':>8s}")
            for shape in shapes:
                srows = [r for r in rows if r["shape"] == shape]
                s_tp = sum(1 for r in srows if r["fast_ok"] and r["full_ok"])
                s_tn = sum(1 for r in srows if not r["fast_ok"] and not r["full_ok"])
                s_fp = sum(1 for r in srows if r["fast_ok"] and not r["full_ok"])
                s_fn = sum(1 for r in srows if not r["fast_ok"] and r["full_ok"])
                s_fast = sum(r["fast_t"] for r in srows) / len(srows)
                s_full = sum(r["full_t"] for r in srows) / len(srows)
                print(f"  {shape:<10s} {len(srows):>5d}  "
                          f"{s_tp + s_tn:>6d}  {s_fp:>3d}  {s_fn:>3d}  "
                          f"{s_fast * 1000:>8.0f}  {s_full * 1000:>8.0f}")

            if fp:
                print("\n=== False positives (fast wrongly said OK) ===")
                for r in rows:
                    if r["fast_ok"] and not r["full_ok"]:
                        print(f"  {r['label']}  full err={r['full_err']}")
            if fn:
                print("\n=== False negatives (fast wrongly said FAIL) ===")
                for r in rows:
                    if not r["fast_ok"] and r["full_ok"]:
                        print(f"  {r['label']}  fast err={r['fast_err']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
