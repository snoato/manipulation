"""Validation / contract harness for the dense-YCB tabletop-access fork.

Sections (default: scene + occupancy + feasibility; --reachmap is slower):

  --scene        build the scene; confirm meshes load + mass overridden to
                 0.05 kg; settle each object on a middle_deck cell.
  --occupancy    greedily pack the roster into non-overlapping multi-cell
                 footprints; restore_state -> ground_state round-trip must
                 reproduce the occupancy exactly; held-restore attaches;
                 a generated PDDL problem parses.
  --feasibility  per object: pick from middle_deck + put to top_deck under
                 FAST and FULL; FAST must equal FULL (the soundness gate).
  --reachmap     per-anchor pick/put feasibility map (FAST) over both
                 regions; identifies unreachable cells.

Physics only — no GL backend needed on macOS.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import mujoco

from tampanda.symbolic.domains.tabletop_access_ycb import (
    make_tabletop_access_ycb_builder, apply_runtime_tweaks,
)
from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    compute_all_footprints,
)
from tampanda.symbolic.domains.tabletop_access_ycb.bridge import (
    make_tabletop_access_ycb_bridge,
)
from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state
from tampanda.symbolic.domains.tabletop_access_ycb.setup import build_setup
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import (
    check_action_sequence,
)
from tampanda.symbolic.domains.tabletop_access_ycb import pddl_gen as P
from tampanda.symbolic.workspace import Cell


# ----------------------------------------------------------------------

def scene_section():
    print("== scene + mass override ==")
    b, ws, cfg = make_tabletop_access_ycb_builder(Path(tempfile.mkdtemp()))
    env = b.build_env(rate=10000.0)

    def mass(o):
        # body_mass is what apply_runtime_tweaks sets and what mj_crb reads
        # each step for free-body dynamics; body_subtreemass is a cached
        # field not recomputed on a runtime body_mass write.
        return float(env.model.body_mass[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, o)])

    raw = {o: mass(o) for o in cfg.object_ids}
    apply_runtime_tweaks(env, cfg)
    ok = all(abs(mass(o) - cfg.match_mass) < 1e-6 for o in cfg.object_ids)
    print(f"  {len(cfg.object_ids)} movables; raw mass "
          f"{min(raw.values()):.2f}-{max(raw.values()):.2f} kg -> all {cfg.match_mass} kg: "
          f"{'PASS' if ok else 'FAIL'}")

    region = ws["middle_deck"]
    cells = list(region.cells())
    target = cells[len(cells) // 2]
    settled = 0
    for o in cfg.object_ids:
        half = env.get_object_half_size(o).astype(float)
        x, y, _ = region.pose_for(target)
        env.set_object_pose(o, np.array([x, y, region.level_z + half[2] + 0.002]),
                            np.array([1.0, 0, 0, 0]))
        mujoco.mj_forward(env.model, env.data)
        for _ in range(120):
            mujoco.mj_step(env.model, env.data)
        if env.get_object_position(o)[2] > region.level_z - 0.05:
            settled += 1
        env.set_object_pose(o, np.array([cfg.hide_far_x, 0, 0.1]))
        mujoco.mj_forward(env.model, env.data)
    print(f"  settle on deck: {settled}/{len(cfg.object_ids)} rest stably")
    return ok and settled == len(cfg.object_ids)


def _greedy_pack(region, fps, object_ids):
    free = {(c.ix, c.iy) for c in region.cells()}
    anchors = {}
    for oid in object_ids:
        fp = fps[oid]
        for iy in range(region.cells_y):
            for ix in range(region.cells_x):
                cells = [(ix + dx, iy + dy) for dx, dy in fp.offsets]
                if all(c in free for c in cells):
                    anchors[oid] = Cell(region.name, ix, iy)
                    free -= set(cells)
                    break
            if oid in anchors:
                break
        if oid not in anchors:
            raise RuntimeError(f"cannot pack {oid}")
    return anchors


def occupancy_section():
    print("== occupancy round-trip ==")
    b, ws, cfg = make_tabletop_access_ycb_builder(Path(tempfile.mkdtemp()))
    env = b.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    fps = compute_all_footprints(env, cfg.object_ids, cfg.cell_size)
    bridge, objects = make_tabletop_access_ycb_bridge(env, ws, cfg, cfg.object_ids, fps)

    region = ws["middle_deck"]
    anchors = _greedy_pack(region, fps, cfg.object_ids)
    expected = {("occupied", c.id, o): True
                for o, a in anchors.items() for c in fps[o].cells_at(a)}
    restore_state(env, ws, expected, list(cfg.object_ids), fps)
    grounded = bridge.ground_state(objects)
    got = {k for k, v in grounded.items() if v and k[0] == "occupied"}
    exp = set(expected)
    occ_ok = got == exp
    occ_cells = {k[1] for k in got}
    empties = {k[1] for k, v in grounded.items() if v and k[0] == "empty"}
    empty_ok = not ((set(objects["cell"]) - occ_cells) - empties) and not (occ_cells & empties)
    print(f"  occupied facts {len(exp)}: round-trip {'PASS' if occ_ok else 'FAIL'}; "
          f"empty-consistent {'PASS' if empty_ok else 'FAIL'}")

    held = "ooi"
    hstate = {("occupied", c.id, o): True for o, a in anchors.items() if o != held
              for c in fps[o].cells_at(a)}
    hstate[("holding", held)] = True
    info = restore_state(env, ws, hstate, list(cfg.object_ids), fps)
    held_ok = info["held"] == held and getattr(env, "_attached", None) is not None
    print(f"  held restore: {'PASS' if held_ok else 'FAIL'}")

    src = {o: [c.id for c in fps[o].cells_at(a)] for o, a in anchors.items()}
    goal = {"ooi": [c.id for c in fps["ooi"].cells_at(Cell("top_deck", 4, 1))]}
    fpm = {o: (fps[o].dx, fps[o].dy) for o in cfg.object_ids}
    prob = P.problem_pddl("v", objects["cell"], list(cfg.object_ids), fpm, src, goal)
    from unified_planning.io import PDDLReader
    dom = (Path(P.__file__).parent / "pddl" / "domain.pddl").read_text()
    up = PDDLReader().parse_problem_string(dom, prob)
    pddl_ok = len(list(up.actions)) > 0
    print(f"  PDDL problem parse: {'PASS' if pddl_ok else 'FAIL'}")
    return occ_ok and empty_ok and held_ok and pddl_ok


def feasibility_section(s):
    print("== feasibility FAST==FULL (pick middle + put top) ==")
    mid, top = Cell("middle_deck", 4, 0), Cell("top_deck", 4, 1)
    allok = True
    for obj in s.config.object_ids:
        res = {}
        for fast in (True, False):
            ok, _ = check_action_sequence(
                s.env, s.workspace, s.executor, s.pick_fn, s.put_fn, s.footprints,
                {obj: mid.id}, [("pick", obj, mid.id), ("put", obj, top.id)],
                list(s.config.object_ids), fast=fast, home_qpos=s.home_qpos)
            res[fast] = ok
        agree = res[True] == res[False]
        allok &= (agree and res[False])
        print(f"  {obj:<18} FAST={res[True]!s:<5} FULL={res[False]!s:<5} "
              f"{'OK' if agree and res[False] else 'MISMATCH/FAIL'}")
    return allok


def reachmap_section(s):
    print("== reachability map (FAST) ==")
    fp = s.footprints["ooi"]
    for region_name, kind in (("middle_deck", "pick"), ("top_deck", "put")):
        region = s.workspace[region_name]
        valid = {(c.ix, c.iy) for c in region.cells()}
        res = {}
        for ay in range(region.cells_y):
            for ax in range(region.cells_x):
                if not all((ax + dx, ay + dy) in valid for dx, dy in fp.offsets):
                    continue
                anchor = Cell(region_name, ax, ay)
                src = {"ooi": anchor.id} if kind == "pick" else {}
                actions = [(kind, "ooi", anchor.id)]
                ok, _ = check_action_sequence(
                    s.env, s.workspace, s.executor, s.pick_fn, s.put_fn,
                    s.footprints, src, actions, list(s.config.object_ids),
                    fast=True, home_qpos=s.home_qpos)
                res[(ax, ay)] = ok
        good = sum(v for v in res.values())
        print(f"  [{region_name}] {kind}: {good}/{len(res)} anchors feasible")
        for ay in range(region.cells_y - 1, -1, -1):
            print("   " + "".join("." if res.get((ax, ay)) else
                                  (" " if (ax, ay) not in res else "X")
                                  for ax in range(region.cells_x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--occupancy", action="store_true")
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--reachmap", action="store_true")
    args = ap.parse_args()
    default = not (args.scene or args.occupancy or args.feasibility or args.reachmap)

    t0 = time.time()
    if args.scene or default:
        scene_section()
    if args.occupancy or default:
        occupancy_section()
    if args.feasibility or args.reachmap or default:
        s = build_setup(Path(tempfile.mkdtemp(prefix="ta_ycb_val_")))
        if args.feasibility or default:
            feasibility_section(s)
        if args.reachmap:
            reachmap_section(s)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
