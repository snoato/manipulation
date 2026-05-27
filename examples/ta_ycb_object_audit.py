"""Object-selection audit for the dense-YCB tabletop-access fork.

Two things, the rationale behind the validated roster:

  --measure   measure every locally-cached YCB object (collision-AABB
              dims -> graspable width; rest-stability after a settle;
              MuJoCo-computed raw mass) and verdict it for the roster
              criteria (graspable < 8 cm, stable, no handle, fits shelf).

  --footprints  build the roster scene and print each object's overlap
              footprint pattern (dx x dy + ASCII) at the domain cell size.

Default: run both.  Physics only (no GL backend needed on macOS).

Background: at the 3 cm grid every object fully fills its bounding
rectangle (round approximation only frees corners for objects >= ~5-7
cells across — none here); the multi-cell footprint code still produces
the honest overlap set.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import mujoco

from tampanda.scenes import ArmSceneBuilder, TABLE_SYMBOLIC_TEMPLATE
from tampanda.scenes.assets.cache import AssetCache
from tampanda.symbolic.domains.tabletop_access_ycb import (
    make_tabletop_access_ycb_builder, apply_runtime_tweaks,
)
from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    compute_all_footprints, ascii_pattern,
)

TABLE_Z = 0.27
MAX_GRIPPER_WIDTH = 0.08
SHELF_USABLE_H = 0.18
HANDLE_OBJECTS = {"mug", "pitcher_base"}
HIDE = [100.0, 0.0, 0.10]


def cached_ycb_names():
    base = AssetCache().path("ycb", "")
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and (p / ".ok").exists())


def measure():
    names = cached_ycb_names()
    if not names:
        print("No cached YCB objects.")
        return
    b = ArmSceneBuilder()
    b.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)
    b.add_object("table", name="simple_table", pos=[0.0, 0.4, 0.0],
                 quat=[0.0, 0.0, 0.0, 1.0])
    for n in names:
        b.add_resource(f"ycb_{n}", {"type": "ycb", "name": n})
        b.add_object(f"ycb_{n}", name=n, pos=HIDE)
    env = b.build_env(rate=10000.0)

    hdr = (f"{'object':<22}{'full XYZ cm':<20}{'minW':>6}{'H':>6}"
           f"{'raw_kg':>8}{'drift':>7}{'tilt':>7}  verdict")
    print(hdr); print("-" * len(hdr))
    for n in names:
        half = env.get_object_half_size(n).astype(float)
        full = 2 * half
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
        mass = float(env.model.body_subtreemass[bid])
        # settle on the table
        z = TABLE_Z + half[2] + 0.002
        env.set_object_pose(n, np.array([0.4, 0.4, z]), np.array([1, 0, 0, 0.0]))
        mujoco.mj_forward(env.model, env.data)
        for _ in range(150):
            mujoco.mj_step(env.model, env.data)
        pos, quat = env.get_object_pose(n)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        tilt = np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1, 1)))
        drift = float(np.hypot(pos[0] - 0.4, pos[1] - 0.4))
        minw = min(full[0], full[1])
        if n in HANDLE_OBJECTS:
            v = "EXCLUDE handle"
        elif pos[2] < TABLE_Z - 0.02:
            v = "EXCLUDE fell"
        elif minw > MAX_GRIPPER_WIDTH:
            v = "EXCLUDE wide"
        elif drift > 0.03 or tilt > 25:
            v = "EXCLUDE roller"
        elif full[2] > SHELF_USABLE_H:
            v = "tall"
        else:
            v = "OK"
        env.set_object_pose(n, np.array(HIDE, float))
        mujoco.mj_forward(env.model, env.data)
        dims = "x".join(f"{c*100:.1f}" for c in full)
        print(f"{n:<22}{dims:<20}{minw*100:>6.1f}{full[2]*100:>6.1f}"
              f"{mass:>8.2f}{drift*1000:>7.1f}{tilt:>7.1f}  {v}")


def footprints():
    b, ws, cfg = make_tabletop_access_ycb_builder(Path(tempfile.mkdtemp()))
    env = b.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    fps = compute_all_footprints(env, cfg.object_ids, cfg.cell_size)
    print(f"\ncell_size = {cfg.cell_size} m  (roster footprints)\n")
    schemas = {}
    for oid in cfg.object_ids:
        fp = fps[oid]
        schemas.setdefault(fp.key, []).append(oid)
        print(f"  {oid:<20} {fp.dx}x{fp.dy}  {fp.n_cells} cells  [{fp.key}]")
    print(f"\n{len(schemas)} distinct footprint schemas:")
    for key, members in schemas.items():
        print(f"\n[{key}] {', '.join(members)}")
        print(ascii_pattern(fps[members[0]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--footprints", action="store_true")
    args = ap.parse_args()
    both = not (args.measure or args.footprints)
    if args.measure or both:
        measure()
    if args.footprints or both:
        footprints()


if __name__ == "__main__":
    main()
