"""Render the dense-YCB scene with a spread of objects on the middle deck.

Visual sanity check for Phase 1 — confirms meshes load, look right, and
sit on the shelf.  Writes ta_ycb_scene.png.  Throwaway diagnostic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import mujoco

from tampanda.symbolic.domains.tabletop_access_ycb import (
    make_tabletop_access_ycb_builder, apply_runtime_tweaks,
)
from tampanda.symbolic.workspace import Cell


def main():
    scratch = Path(tempfile.mkdtemp(prefix="ta_ycb_render_"))
    b, ws, cfg = make_tabletop_access_ycb_builder(scratch)
    env = b.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    region = ws["middle_deck"]
    surf = region.level_z
    # Spread objects across the deck on a coarse sub-lattice (every 4 cells
    # in x) so they don't overlap; settle.
    anchors = [(1, 1), (5, 2), (9, 1), (3, 6), (7, 6), (11, 4),
               (1, 8), (5, 8), (9, 8), (11, 8), (3, 3)]
    for (oid, (ix, iy)) in zip(cfg.object_ids, anchors):
        half = env.get_object_half_size(oid).astype(float)
        x, y, _ = region.pose_for(Cell(region.name, ix, iy))
        env.set_object_pose(oid, np.array([x, y, surf + half[2] + 0.003]),
                            np.array([1.0, 0, 0, 0]))
    mujoco.mj_forward(env.model, env.data)
    for _ in range(200):
        mujoco.mj_step(env.model, env.data)

    sx, sy, sz = cfg.shelf_pos
    env.model.vis.global_.offwidth = 1100
    env.model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(env.model, 720, 1100)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [sx, sy, sz + 0.32]
    cam.distance = 1.5
    cam.azimuth = -90      # look from -y (robot side, into the open front)
    cam.elevation = -18
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, cam)
    img = renderer.render()

    try:
        import imageio.v2 as imageio
        imageio.imwrite("ta_ycb_scene.png", img)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave("ta_ycb_scene.png", img)
    print(f"wrote ta_ycb_scene.png  {img.shape}")


if __name__ == "__main__":
    main()
