"""Show a generated dense-YCB dataset: summary table + before/after renders.

Usage: python examples/ta_ycb_show_dataset.py [dataset_dir]

Prints a per-instance summary (objects / occluders / clutter / plan length)
and renders initial (cluttered middle deck) vs final (OoI delivered to the
top deck, blockers relocated) for a couple of instances -> ta_ycb_dataset.png.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import mujoco

from tampanda.symbolic.domains.tabletop_access_ycb.setup import build_setup
from tampanda.symbolic.domains.tabletop_access_ycb.state import (
    restore_state, ground_to_object_anchors,
)
from tampanda.symbolic.workspace import Cell

_OCC = re.compile(r"\(occupied\s+(\S+__\d+_\d+)\s+(\w+)\)")


def parse_source(pddl_path):
    txt = pddl_path.read_text()
    init = txt.split("(:init")[1].split("(:goal")[0]
    state = {("occupied", c, o): True for c, o in _OCC.findall(init)}
    return state


def parse_plan(plan_path):
    steps = []
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue
        toks = line.strip("()").split()
        kind = "pick" if toks[0].startswith("pick") else "put"
        steps.append((kind, toks[1], toks[2]))   # (kind, obj, anchor=first cell)
    return steps


def final_layout(source_anchors, plan):
    layout = {o: a.id for o, a in source_anchors.items() if a is not None}
    for kind, obj, anchor in plan:
        if kind == "pick":
            layout.pop(obj, None)
        else:
            layout[obj] = anchor
    return layout


def render(setup, layout, cam):
    fps = setup.footprints
    state = {}
    for o, aid in layout.items():
        for c in fps[o].cells_at(Cell.parse(aid)):
            state[("occupied", c.id, o)] = True
    restore_state(setup.env, setup.workspace, state, list(setup.config.object_ids), fps)
    # Stow the arm hard left + folded so it clears the camera's view of the
    # shelf front.
    setup.env.data.qpos[:7] = [2.9, 0.6, 0.0, -2.4, 0.0, 1.4, 0.8]
    setup.env.data.qvel[:] = 0.0
    mujoco.mj_forward(setup.env.model, setup.env.data)
    r = mujoco.Renderer(setup.env.model, 540, 720)
    r.update_scene(setup.env.data, cam)
    img = r.render()
    r.close()
    return img


def main():
    ddir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/access_ycb_demo")
    insts = sorted(ddir.rglob("*.pddl"))
    insts = [p for p in insts if p.name != "domain.pddl"]

    # summary table
    print(f"{'instance':<22}{'objs':>5}{'occ':>5}{'clut':>6}{'plan':>6}")
    print("-" * 44)
    meta = {}
    for p in insts:
        txt = p.with_suffix(".pddl.plan").read_text()
        g = {k: v for k, v in re.findall(r";\s*(\w+):\s*(\S+)", txt)}
        meta[p] = g
        print(f"{p.parent.name + '/' + p.stem:<22}"
              f"{g.get('n_objects','?'):>5}{g.get('n_occ','?'):>5}"
              f"{g.get('n_clutter','?'):>6}{g.get('plan_len','?'):>6}")

    # pick one moderate train + the densest eval_ood for rendering
    train = [p for p in insts if p.parent.name == "train"]
    evald = [p for p in insts if p.parent.name == "eval_ood"]
    evald.sort(key=lambda p: int(meta[p].get("n_objects", 0)), reverse=True)
    chosen = ([train[0]] if train else []) + ([evald[0]] if evald else [])

    setup = build_setup(Path(tempfile.mkdtemp(prefix="ta_ycb_show_")))
    sx, sy, sz = setup.config.shelf_pos
    setup.env.model.vis.global_.offwidth = 720
    setup.env.model.vis.global_.offheight = 540
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [sx, sy, sz + 0.42]
    cam.distance = 1.15
    cam.azimuth = -90
    cam.elevation = -28

    rows = []
    for p in chosen:
        src = ground_to_object_anchors(parse_source(p), list(setup.config.object_ids))
        plan = parse_plan(p.with_suffix(".pddl.plan"))
        before = render(setup, {o: a.id for o, a in src.items() if a}, cam)
        after = render(setup, final_layout(src, plan), cam)
        gap = np.full((before.shape[0], 8, 3), 255, np.uint8)
        rows.append(np.hstack([before, gap, after]))
        print(f"\nrendered {p.parent.name}/{p.stem}: "
              f"{meta[p].get('n_objects')} objs, plan {meta[p].get('plan_len')} "
              f"(left=initial clutter, right=after delivery)")

    vgap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    composite = rows[0] if len(rows) == 1 else np.vstack([rows[0], vgap, rows[1]])
    out = "ta_ycb_dataset.png"
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out, composite)
    except Exception:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(out, composite)
    print(f"\nwrote {out}  (rows=instances, cols: initial | final)")


if __name__ == "__main__":
    main()
