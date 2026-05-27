"""FAST==FULL agreement + timing stress harness for the dense-YCB fork.

Samples dense multi-object layouts and probes per-action feasibility under
BOTH FAST and FULL from the SAME canonical pre-state, recording agreement
and per-check wall time by object count.  This is the regime most likely
to expose FAST-optimism (the single-pick+put roster sweep was narrow).

  python examples/ta_ycb_stress.py [--layouts 30] [--seed 0]

Reports:
  * agreement %  and the dangerous FAST-accept / FULL-reject count
  * FAST and FULL per-check ms (median), split by feasible/infeasible and
    bucketed by object count
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from tampanda.symbolic.domains.tabletop_access_ycb.setup import build_setup
from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import check_action
from tampanda.symbolic.domains.tabletop_access_ycb.generate_data import sample_layout
from tampanda.symbolic.workspace import Cell


def _state(layout, footprints, held=None):
    st = {}
    for o, aid in layout.items():
        for c in footprints[o].cells_at(Cell.parse(aid)):
            st[("occupied", c.id, o)] = True
    if held is not None:
        st[("holding", held)] = True
    return st


def _timed_check(setup, state, action, fast):
    fps = setup.footprints
    restore_state(setup.env, setup.workspace, state, list(setup.config.object_ids),
                  fps, executor=setup.executor, home_qpos=setup.home_qpos)
    t0 = time.perf_counter()
    ok = check_action(setup.env, setup.workspace, setup.executor,
                      setup.pick_fn, setup.put_fn, fps, action, fast=fast)
    return ok, (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup = build_setup(Path(tempfile.mkdtemp(prefix="ta_ycb_stress_")))
    ws, fps = setup.workspace, setup.footprints
    top = ws["top_deck"]
    vtop = {(c.ix, c.iy) for c in top.cells()}
    rng = np.random.default_rng(args.seed)
    blockers = [o for o in setup.config.object_ids if o != "ooi"]

    probes = []  # (n_obj, kind, fast_ok, full_ok, fast_ms, full_ms)
    t_start = time.time()
    made = 0
    while made < args.layouts:
        n_occ = int(rng.integers(1, 4))
        n_clutter = int(rng.integers(3, 8))           # dense
        s = sample_layout(ws, fps, rng, n_occ, n_clutter, blockers)
        if s is None:
            continue
        layout, goal = s
        made += 1
        present = list(layout.keys())
        n_obj = len(present)

        # pick probes: up to 3 random present objects
        for obj in rng.permutation(present)[:3]:
            obj = str(obj)
            act = ("pick", obj, layout[obj])
            st = _state(layout, fps)
            fok, fms = _timed_check(setup, st, act, True)
            gok, gms = _timed_check(setup, st, act, False)
            probes.append((n_obj, "pick", fok, gok, fms, gms))

        # put probe: hold one present object, put to a random top anchor it fits
        obj = str(rng.permutation(present)[0])
        fp = fps[obj]
        fits = [(ax, ay) for ay in range(top.cells_y) for ax in range(top.cells_x)
                if all((ax + dx, ay + dy) in vtop for dx, dy in fp.offsets)]
        if fits:
            ax, ay = fits[rng.integers(len(fits))]
            anchor = Cell("top_deck", ax, ay)
            rest = {o: c for o, c in layout.items() if o != obj}
            st = _state(rest, fps, held=obj)
            act = ("put", obj, anchor.id)
            fok, fms = _timed_check(setup, st, act, True)
            gok, gms = _timed_check(setup, st, act, False)
            probes.append((n_obj, "put", fok, gok, fms, gms))

    # ---- report ----
    n = len(probes)
    agree = sum(1 for p in probes if p[2] == p[3])
    fast_opt = [p for p in probes if p[2] and not p[3]]     # FAST yes, FULL no
    fast_pess = [p for p in probes if (not p[2]) and p[3]]  # FAST no, FULL yes
    print(f"\n{made} dense layouts, {n} per-action probes "
          f"({time.time()-t_start:.0f}s)\n")
    print(f"FAST==FULL agreement: {agree}/{n} ({100*agree/n:.1f}%)")
    print(f"  FAST-accept / FULL-reject (unsound, the risk): {len(fast_opt)}")
    print(f"  FAST-reject / FULL-accept (conservative):      {len(fast_pess)}")
    if fast_opt:
        from collections import Counter
        print(f"    by kind: {Counter(p[1] for p in fast_opt)}")

    def med(xs):
        return statistics.median(xs) if xs else float("nan")

    feas = [p for p in probes if p[3]]      # FULL-feasible
    infeas = [p for p in probes if not p[3]]
    print(f"\nper-check ms (median):")
    print(f"  {'outcome':<18}{'n':>5}{'FAST':>8}{'FULL':>9}{'speedup':>9}")
    for label, grp in (("FULL-feasible", feas), ("FULL-infeasible", infeas)):
        if grp:
            fa, fu = med([p[4] for p in grp]), med([p[5] for p in grp])
            print(f"  {label:<18}{len(grp):>5}{fa:>8.1f}{fu:>9.1f}{fu/fa:>8.1f}x")

    print(f"\nFAST ms (median) by object count:")
    buckets = {"2-4": [], "5-6": [], "7-9": []}
    for p in probes:
        k = "2-4" if p[0] <= 4 else "5-6" if p[0] <= 6 else "7-9"
        buckets[k].append(p[4])
    for k, xs in buckets.items():
        if xs:
            print(f"  {k} objs: {med(xs):.1f} ms  (n={len(xs)})")


if __name__ == "__main__":
    main()
