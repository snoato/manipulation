"""Sweep access19 shelf placement and record arm-pose animations.

Mirror of ``sweep_access_shelf.py`` for the access-19 (closed-cubicle)
variant.  For each candidate ``table_pos = (sx, sy, 0)`` and a
hypothetical ``top_grid_cells = 7`` (so the top deck mirrors the
7×7 interior footprint), IK-solve the palm-+y canonical grasp pose
at every cell of:

* ``shelf_top`` (7×7 = 49 cells)
* the 21 cube cells of ``shelf_interior`` (ix ∈ {1, 3, 5})

Per placement: a video cycling through every cell with the IK
result and a JSON entry recording per-cell convergence + error.
Use the totals to choose a final ``table_pos`` + ``top_grid_cells``.

Usage::

    python examples/sweep_access19_shelf.py \\
        --x 0.30 0.35 0.40 \\
        --y 0.35 0.40 0.45 \\
        --top-grid-cells 7 \\
        --output-dir /tmp/access19_sweep \\
        --frames-per-cell 8
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tampanda.symbolic.domains.tabletop_access import (
    apply_runtime_tweaks,
    make_access19_builder,
)
from tampanda.symbolic.workspace import Cell


_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])

_RENDER_W = 640
_RENDER_H = 480


def _staging_seed_q() -> np.ndarray:
    """Same canonical Franka-home-with-q[0]=π/2 seed used by access
    and access19 staging IK."""
    return np.array(
        [np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
    )


def _ik_at_cell(env, cell_pos: np.ndarray) -> Tuple[bool, np.ndarray, float, int]:
    """Try both FRONT-quat basins (parallel-jaw symmetry); return
    the basin that converges with smallest error.  Returns
    ``(any_converged, q7, pos_err_m, basin_idx)``."""
    seed = _staging_seed_q()
    best = (False, seed, float("inf"), -1)
    for bi, q in enumerate((_FRONT_QUAT, _FRONT_QUAT_FLIPPED)):
        env.data.qpos[:7] = seed
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        env.ik.set_target_position(cell_pos, q)
        env.ik.pos_threshold = 0.005
        env.ik.ori_threshold = 5e-3
        ok = env.ik.converge_ik(0.005)
        err = env.ik.ee_task.compute_error(env.ik.configuration)
        e = float(np.linalg.norm(err[:3]))
        if e < best[2]:
            best = (ok, env.ik.configuration.q[:7].copy(), e, bi)
    return best


def _setup_camera(model, shelf_pos: Tuple[float, float, float]):
    sx, sy, _ = shelf_pos
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth = 110
    cam.elevation = -20
    cam.distance = 2.0
    cam.lookat[:] = [sx * 0.6, sy * 0.6, 0.45]
    return cam


def _annotate(img: np.ndarray, lines: List[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 0), font=font,
                   stroke_width=1, stroke_fill=(0, 0, 0))
        y += 18
    return np.asarray(pil)


def sweep_placement(sx: float, sy: float,
                      output_dir: Path,
                      top_grid_cells: int,
                      frames_per_cell: int,
                      fps: int) -> Dict:
    import imageio.v3 as iio

    with tempfile.TemporaryDirectory() as scratch_str:
        builder, ws, cfg = make_access19_builder(
            scratch_dir=Path(scratch_str),
            table_pos=(sx, sy, 0.0),
            top_grid_cells=top_grid_cells,
        )
        env = builder.build_env(rate=10000.0)
        apply_runtime_tweaks(env, cfg)

        renderer = mujoco.Renderer(env.model, height=_RENDER_H,
                                      width=_RENDER_W)
        cam = _setup_camera(env.model, (sx, sy, 0.0))

        # Park every object so the arm is unobstructed visually.
        parked = np.array([cfg.hide_far_x, 0.0, 0.05])
        for n in [f"blocker_{i}" for i in range(18)] + ["ooi"]:
            env.set_object_pose(n, parked)
        env.forward()

        # Build the cell list: all top-deck cells + cube columns of
        # the interior.
        top = ws["shelf_top"]
        interior = ws["shelf_interior"]
        cells: List[Cell] = []
        for iy in range(top.cells_y):
            for ix in range(top.cells_x):
                cells.append(Cell("shelf_top", ix, iy))
        for iy in range(interior.cells_y):
            for ix in (1, 3, 5):
                cells.append(Cell("shelf_interior", ix, iy))

        frames: List[np.ndarray] = []
        per_cell: List[Dict] = []
        for cell in cells:
            region = ws[cell.region]
            cx, cy, cz = region.pose_for(cell)
            target = np.array([cx, cy, cz])
            ok, q7, err, basin = _ik_at_cell(env, target)
            per_cell.append({
                "cell": cell.id,
                "target_xyz": target.tolist(),
                "ik_converged": ok,
                "pos_err_mm": err * 1000.0,
                "basin": basin,
                "q7": q7.tolist(),
            })
            env.data.qpos[:7] = q7
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            status = (f"IK ok b={basin} ({err*1000:.1f} mm)"
                       if ok else f"IK FAIL ({err*1000:.1f} mm)")
            lines = [
                f"table=({sx:.2f}, {sy:.2f})  top={top_grid_cells}x{top_grid_cells}",
                f"cell={cell.id}",
                f"target=({cx:.3f}, {cy:.3f}, {cz:.3f})",
                status,
            ]
            renderer.update_scene(env.data, camera=cam)
            img = renderer.render()
            img = _annotate(img, lines)
            for _ in range(frames_per_cell):
                frames.append(img)

        renderer.close()

        out_path = output_dir / f"sweep19_{sx:.2f}_{sy:.2f}_top{top_grid_cells}.mp4"
        iio.imwrite(out_path, frames, fps=fps,
                     codec="libx264", quality=8)

        n_top = sum(1 for c in per_cell
                       if c["cell"].startswith("shelf_top__")
                       and c["ik_converged"])
        n_int = sum(1 for c in per_cell
                       if c["cell"].startswith("shelf_interior__")
                       and c["ik_converged"])
        return {
            "table_pos": (sx, sy, 0.0),
            "top_grid_cells": top_grid_cells,
            "n_top_cells": top.cells_x * top.cells_y,
            "n_top_ik_ok": n_top,
            "n_interior_cube_cells": 3 * interior.cells_y,
            "n_interior_ik_ok": n_int,
            "per_cell": per_cell,
            "video": str(out_path),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40])
    p.add_argument("--y", type=float, nargs="+",
                    default=[0.35, 0.40, 0.45])
    p.add_argument("--top-grid-cells", type=int, default=7)
    p.add_argument("--output-dir", type=Path,
                    default=Path("/tmp/access19_sweep"))
    p.add_argument("--frames-per-cell", type=int, default=8)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep grid: x={args.x}  y={args.y}  "
          f"top={args.top_grid_cells}x{args.top_grid_cells}  "
          f"=> {len(args.x) * len(args.y)} placements")

    summary: List[Dict] = []
    for sx in args.x:
        for sy in args.y:
            print(f"\n--- placement ({sx:.2f}, {sy:.2f}) ---")
            try:
                r = sweep_placement(sx, sy, args.output_dir,
                                      args.top_grid_cells,
                                      args.frames_per_cell, args.fps)
            except Exception as e:
                print(f"  CRASHED: {type(e).__name__}: {e}")
                continue
            print(f"  top IK ok: {r['n_top_ik_ok']} / {r['n_top_cells']}")
            print(f"  interior cube IK ok: "
                  f"{r['n_interior_ik_ok']} / {r['n_interior_cube_cells']}")
            print(f"  video: {r['video']}")
            summary.append(r)

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"{'table_pos':<14} {'top ok':>10}  {'interior ok':>13}")
    for r in summary:
        sx, sy, _ = r["table_pos"]
        print(f"({sx:.2f}, {sy:.2f})    "
              f"{r['n_top_ik_ok']:>3} / {r['n_top_cells']:<3}    "
              f"{r['n_interior_ik_ok']:>3} / {r['n_interior_cube_cells']}")

    import json
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nFull data: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
