"""Dataset generation for the dense-YCB tabletop-access fork.

Samples cluttered middle-deck layouts (OoI + occluders + clutter, packed
tight on the 3 cm grid with multi-cell footprints), solves each with the
feasibility-guided planner, and writes a PDDL problem + plan per instance.

* Per-action FAST checks (canonical restore) are the soundness gate; a
  10% FULL spot-check guards against FAST leniency.  Unsolvable / FULL-
  failing samples are reject+resampled.
* Problems emit a **dynamic per-problem cell roster** (plan-touched cells
  + a 1-ring margin) — not the full grid — to keep the GNN object list
  small, plus within-region ``(adjacent …)`` edges and per-object
  ``fp_<W>x<H>`` markers.

CLI::

    python -m tampanda.symbolic.domains.tabletop_access_ycb.generate_data \
        --output-dir data/access_ycb --train 300 --val 50 --eval 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell

from tampanda.symbolic.domains.tabletop_access_ycb.setup import build_setup
from tampanda.symbolic.domains.tabletop_access_ycb.planner import solve, make_fast_oracle
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import (
    footprint_blocks_pick, check_action_sequence,
)
from tampanda.symbolic.domains.tabletop_access_ycb import pddl_gen as P

# Fraction of accepted plans re-validated under FULL physics before writing
# (reject+resample on failure).  Dense-scene FAST is ~1.7%-optimistic on
# picks (examples/ta_ycb_stress.py), so a short plan has a non-trivial
# chance of containing a FAST-only action — default to 1.0 (every plan
# FULL-validated → every written plan is FULL-executable).  Lower it (e.g.
# 0.1) only for fast local iteration.
_DEFAULT_FULL_CHECK_FRAC = 1.0


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def _valid(region) -> set:
    return {(c.ix, c.iy) for c in region.cells()}


def _fits(fp, ax, ay, valid) -> bool:
    return all((ax + dx, ay + dy) in valid for dx, dy in fp.offsets)


def _cells_of(fp, anchor: Cell) -> set:
    return {(c.ix, c.iy) for c in fp.cells_at(anchor)}


def sample_layout(ws, footprints, rng, n_occ: int, n_clutter: int,
                  blocker_pool: List[str]) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    """Place OoI + occluders (in its front corridor) + clutter, non-overlapping.

    Returns ``(source_layout, goal)`` (anchor cell ids) or None on failure.
    """
    mid = ws["middle_deck"]
    top = ws["top_deck"]
    vmid, vtop = _valid(mid), _valid(top)
    fp_ooi = footprints["ooi"]

    # OoI: leave a few front rows for occluders.
    ooi_choices = [(ax, ay) for ay in range(2, mid.cells_y)
                   for ax in range(mid.cells_x) if _fits(fp_ooi, ax, ay, vmid)]
    if not ooi_choices:
        return None
    ax, ay = ooi_choices[rng.integers(len(ooi_choices))]
    ooi_anchor = Cell("middle_deck", ax, ay)
    occupied = _cells_of(fp_ooi, ooi_anchor)
    layout = {"ooi": ooi_anchor.id}

    pool = list(blocker_pool)
    rng.shuffle(pool)

    def place(obj, want_block) -> bool:
        fp = footprints[obj]
        cands = []
        for cy in range(mid.cells_y):
            for cx in range(mid.cells_x):
                if not _fits(fp, cx, cy, vmid):
                    continue
                cset = _cells_of(fp, Cell("middle_deck", cx, cy))
                if cset & occupied:
                    continue
                anc = Cell("middle_deck", cx, cy)
                blk = footprint_blocks_pick(anc, fp, ooi_anchor, fp_ooi)
                if want_block and not blk:
                    continue
                if (not want_block) and blk:
                    continue
                cands.append((anc, cset))
        if not cands:
            return False
        anc, cset = cands[rng.integers(len(cands))]
        layout[obj] = anc.id
        occupied.update(cset)
        return True

    for _ in range(n_occ):
        if not pool:
            break
        for obj in list(pool):
            if place(obj, want_block=True):
                pool.remove(obj)
                break
    for _ in range(n_clutter):
        if not pool:
            break
        for obj in list(pool):
            if place(obj, want_block=False):
                pool.remove(obj)
                break

    # goal: a top-deck anchor where the OoI fits.
    goal_choices = [(gx, gy) for gy in range(top.cells_y)
                    for gx in range(top.cells_x) if _fits(fp_ooi, gx, gy, vtop)]
    if not goal_choices:
        return None
    gx, gy = goal_choices[rng.integers(len(goal_choices))]
    return layout, {"ooi": Cell("top_deck", gx, gy).id}


# ----------------------------------------------------------------------
# PDDL writing (dynamic per-problem cell roster)
# ----------------------------------------------------------------------

def _plan_cells(ws, footprints, source_layout, plan, goal) -> List[str]:
    """Cells the plan touches (source + every action anchor + goal) plus a
    1-ring margin, restricted to valid region cells."""
    touched: set = set()

    def add_anchor(obj, anchor_id):
        a = Cell.parse(anchor_id)
        for c in footprints[obj].cells_at(a):
            touched.add((c.region, c.ix, c.iy))

    for o, aid in source_layout.items():
        add_anchor(o, aid)
    for kind, obj, aid in plan:
        add_anchor(obj, aid)
    for o, aid in goal.items():
        add_anchor(o, aid)

    valid = {r: _valid(ws[r]) for r in ws.regions}
    out: set = set()
    for region, ix, iy in touched:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (ix + dx, iy + dy) in valid[region]:
                    out.add(f"{region}__{ix + dx}_{iy + dy}")
    return sorted(out)


def write_instance(outdir: Path, name: str, ws, footprints,
                   source_layout, goal, plan, meta) -> None:
    movables = sorted(source_layout.keys())
    cells = _plan_cells(ws, footprints, source_layout, plan, goal)
    fp_markers = {o: (footprints[o].dx, footprints[o].dy) for o in movables}
    src_occ = {o: [c.id for c in footprints[o].cells_at(Cell.parse(a))]
               for o, a in source_layout.items()}
    goal_occ = {o: [c.id for c in footprints[o].cells_at(Cell.parse(a))]
                for o, a in goal.items()}
    prob = P.problem_pddl(name, cells, movables, fp_markers, src_occ, goal_occ)
    (outdir / f"{name}.pddl").write_text(prob)

    # Plan steps as grounded per-size PDDL actions (match the domain); the
    # tampanda execution tuple (kind, obj, anchor) is recoverable from the
    # action name prefix + obj + first cell.
    lines = [P.grounded_plan_action(kind, obj, Cell.parse(aid), footprints[obj])
             for kind, obj, aid in plan]
    lines.append(f"; cost = {len(plan)} (unit cost)")
    lines += [f"; {k}: {v}" for k, v in meta.items()]
    (outdir / f"{name}.pddl.plan").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def _counts(split: str, rng) -> Tuple[int, int]:
    if split == "eval_ood":
        return int(rng.integers(1, 4)), int(rng.integers(3, 8))   # denser
    return int(rng.integers(1, 4)), int(rng.integers(0, 3))


def generate_split(split: str, count: int, outdir: Path, setup, oracle, rng,
                   full_check_frac: float = _DEFAULT_FULL_CHECK_FRAC,
                   start_index: int = 0):
    outdir.mkdir(parents=True, exist_ok=True)
    ws, fps = setup.workspace, setup.footprints
    blockers = [o for o in setup.config.object_ids if o != "ooi"]
    made = attempts = full_checked = full_failed = 0
    t0 = time.time()
    while made < count:
        attempts += 1
        if attempts > count * 40 + 200:
            raise RuntimeError(f"{split}: too many resamples ({attempts})")
        n_occ, n_clutter = _counts(split, rng)
        sampled = sample_layout(ws, fps, rng, n_occ, n_clutter, blockers)
        if sampled is None:
            continue
        layout, goal = sampled
        plan = solve(ws, fps, layout, goal, oracle)
        if not plan:
            continue
        if rng.random() < full_check_frac:
            ok, _ = check_action_sequence(
                setup.env, ws, setup.executor, setup.pick_fn, setup.put_fn, fps,
                layout, plan, list(setup.config.object_ids),
                fast=False, home_qpos=setup.home_qpos)
            full_checked += 1
            if not ok:
                full_failed += 1
                continue
        name = f"{split}_{start_index + made:04d}"
        meta = {"split": split, "n_occ": n_occ, "n_clutter": n_clutter,
                "n_objects": len(layout), "plan_len": len(plan)}
        write_instance(outdir, name, ws, fps, layout, goal, plan, meta)
        made += 1
        if made % 10 == 0:
            print(f"  {split}: {made}/{count} (attempts={attempts}, "
                  f"full-checked={full_checked}, {time.time()-t0:.0f}s)")
    print(f"{split}: {made} instances, {attempts} attempts, "
          f"{full_checked} FULL-checked ({full_failed} FULL-rejected), "
          f"{time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("data/access_ycb"))
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-check-frac", type=float, default=_DEFAULT_FULL_CHECK_FRAC,
                    help="fraction of plans re-validated under FULL before "
                         "writing (default 1.0 → every written plan is "
                         "FULL-executable; lower for fast local iteration)")
    ap.add_argument("--start-index", type=int, default=0,
                    help="offset for the NNNN filename index, so SLURM-array "
                         "shards write non-colliding names into shared split dirs")
    args = ap.parse_args()

    import tempfile
    setup = build_setup(Path(tempfile.mkdtemp(prefix="ta_ycb_gen_")))
    oracle = make_fast_oracle(setup)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({(fp.dx, fp.dy) for fp in setup.footprints.values()})
    P.write_domain_pddl(sizes, args.output_dir / "domain.pddl")
    print(f"wrote domain.pddl ({len(sizes)} footprint sizes)")

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.train), ("val", args.val), ("eval_ood", args.eval)):
        if n > 0:
            generate_split(split, n, args.output_dir / split, setup, oracle, rng,
                           full_check_frac=args.full_check_frac,
                           start_index=args.start_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
