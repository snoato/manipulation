"""Diagnose FAST-accept / FULL-reject picks (the unsound cases).

Reproduces dense layouts, finds picks FAST accepts but FULL rejects, and
for each instruments the FULL attempt: which chain phase bails, and
whether a neighbour object is knocked (drift-contact) vs the chain failing
to plan from a drifted config.  Informs the conservative FAST fix.

  python examples/ta_ycb_diagnose_optimism.py [--max-cases 3] [--seed 0]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import mujoco

from tampanda.symbolic.domains.tabletop_access_ycb.setup import build_setup
from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import check_action
from tampanda.symbolic.domains.tabletop_access_ycb.generate_data import sample_layout
from tampanda.symbolic.workspace import Cell


def _state(layout, fps):
    st = {}
    for o, aid in layout.items():
        for c in fps[o].cells_at(Cell.parse(aid)):
            st[("occupied", c.id, o)] = True
    return st


def _positions(env, ids):
    return {o: env.get_object_pose(o)[0].copy() for o in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup = build_setup(Path(tempfile.mkdtemp(prefix="ta_ycb_diag_")))
    ws, fps = setup.workspace, setup.footprints
    oid_all = list(setup.config.object_ids)
    blockers = [o for o in oid_all if o != "ooi"]
    rng = np.random.default_rng(args.seed)

    found = 0
    layouts = 0
    while found < args.max_cases and layouts < 60:
        layouts += 1
        s = sample_layout(ws, fps, rng, int(rng.integers(1, 4)),
                          int(rng.integers(3, 8)), blockers)
        if s is None:
            continue
        layout, _ = s
        present = list(layout.keys())
        for obj in present:
            act = ("pick", obj, layout[obj])
            st = _state(layout, fps)
            # FAST
            restore_state(setup.env, ws, st, oid_all, footprints=fps,
                          executor=setup.executor, home_qpos=setup.home_qpos)
            fok = check_action(setup.env, ws, setup.executor, setup.pick_fn,
                               setup.put_fn, fps, act, fast=True)
            if not fok:
                continue
            # FULL (instrumented)
            restore_state(setup.env, ws, st, oid_all, footprints=fps,
                          executor=setup.executor, home_qpos=setup.home_qpos)
            before = _positions(setup.env, present)
            print(f"\n--- candidate: pick {obj} @ {layout[obj]} "
                  f"({len(present)} objs) ---")
            print("  FULL chain output:")
            gok = check_action(setup.env, ws, setup.executor, setup.pick_fn,
                               setup.put_fn, fps, act, fast=False)
            if gok:
                continue  # FAST and FULL agree (feasible) — not a case
            found += 1
            after = _positions(setup.env, present)
            print(f"  >>> UNSOUND: FAST=accept, FULL=reject")
            tgt_d = float(np.linalg.norm(after[obj][:2] - before[obj][:2])) * 1000
            print(f"  TARGET {obj} moved {tgt_d:.0f}mm "
                  f"({'pushed' if tgt_d > 5 else 'stable'})")
            # neighbour displacement (exclude the picked obj + parked)
            disp = []
            for o in present:
                if o == obj:
                    continue
                d = float(np.linalg.norm(after[o][:2] - before[o][:2]))
                if d > 0.003:
                    disp.append((o, d * 1000, layout[o]))
            disp.sort(key=lambda x: -x[1])
            if disp:
                print(f"  neighbours knocked (xy mm): "
                      + ", ".join(f"{o}+{d:.0f}mm@{c}" for o, d, c in disp[:4]))
            else:
                print("  no neighbour moved >3mm — failure is plan-from-drift, "
                      "not a knocked neighbour")
            if found >= args.max_cases:
                break

    print(f"\nscanned {layouts} layouts, found {found} unsound pick(s)")


if __name__ == "__main__":
    main()
