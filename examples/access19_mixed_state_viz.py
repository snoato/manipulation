"""Render a state-restoration snapshot with blockers on BOTH the
``shelf_interior`` (lower cubicle) AND the ``shelf_top`` (upper deck).

Builds a synthetic PDDL ground-state that mixes lower and upper cells,
calls :func:`restore_state`, and dumps a 4-pane PNG (front / iso /
side / top) to ``access19_mixed_state.png``.

Usage::

    python examples/access19_mixed_state_viz.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
)
from tampanda.symbolic.domains.access19.state import restore_state


_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]


def _render(env, lookat, azimuth, elevation, distance,
            width=1280, height=800) -> np.ndarray:
    if env.model.vis.global_.offwidth < width:
        env.model.vis.global_.offwidth = width
    if env.model.vis.global_.offheight < height:
        env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.lookat = np.array(lookat)
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    renderer.disable_depth_rendering()
    renderer.update_scene(env.data, camera=cam)
    img = renderer.render().copy()
    renderer.close()
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (40, 40, 40), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _grid(images: List[np.ndarray]) -> np.ndarray:
    top = np.hstack(images[:2])
    bot = np.hstack(images[2:4])
    return np.vstack([top, bot])


def _build_mixed_state() -> Tuple[Dict[Tuple, bool], List[Tuple[str, str]]]:
    """Construct a state with blockers on BOTH shelves + held OoI.

    Coverage goals:
      * Interior cells across all three cube columns (ix=1, 3, 5) and
        a mix of depths (iy=0, 2, 3, 4).
      * Top-deck cells across multiple positions (front, mid, back).
      * Several blockers PARKED (off-scene at hide_far_x=100) — should
        not appear in any view.
    """
    placements: List[Tuple[str, str]] = [
        ("blocker_0",  "shelf_interior__1_0"),
        ("blocker_1",  "shelf_interior__3_2"),
        ("blocker_2",  "shelf_interior__5_3"),
        ("blocker_3",  "shelf_interior__1_4"),
        ("blocker_4",  "shelf_interior__5_0"),
        ("blocker_5",  "shelf_top__0_0"),
        ("blocker_6",  "shelf_top__2_1"),
        ("blocker_7",  "shelf_top__4_3"),
        ("blocker_8",  "shelf_top__6_5"),
        ("blocker_9",  "shelf_top__3_6"),
        ("ooi",        "shelf_top__1_2"),
    ]
    state: Dict[Tuple, bool] = {}
    for obj, cell in placements:
        state[("occupied", cell, obj)] = True
    return state, placements


def main() -> int:
    print("=== access-19 mixed-state restoration viz ===\n")

    with tempfile.TemporaryDirectory(prefix="access19_mixedviz_") as td:
        builder, ws, cfg = make_access19_builder(scratch_dir=Path(td))
        env = builder.build_env(rate=10000.0)
        apply_runtime_tweaks(env, cfg)

        state, placements = _build_mixed_state()
        print(f"  placing {len(placements)} objects across both shelves "
              f"({sum(1 for _, c in placements if 'interior' in c)} "
              f"interior, {sum(1 for _, c in placements if 'top' in c)} "
              f"top); {len(_OBJECT_NAMES) - len(placements)} parked.\n")

        for obj, cell in placements:
            tag = "  interior" if "interior" in cell else "  top deck"
            print(f"{tag}   {obj:<12} → {cell}")
        print()

        info = restore_state(env, ws, cfg, state, _OBJECT_NAMES)
        print(f"  restore_state → placed={len(info['placed'])} "
              f"parked={len(info['parked'])} held={info['held']!r}")
        # Settle visuals (no controller; cubes are at canonical poses).
        mujoco.mj_forward(env.model, env.data)

        # Render four angles.  ``lookat`` straddles the cubicle (front
        # face at y≈0.42) so both deck and interior are visible.
        lookat = (0.35, 0.55, 0.55)
        front = _render(env, lookat, azimuth=90,  elevation=-10, distance=1.6)
        iso   = _render(env, lookat, azimuth=125, elevation=-22, distance=1.6)
        side  = _render(env, lookat, azimuth=0,   elevation=-10, distance=1.6)
        top   = _render(env, lookat, azimuth=90,  elevation=-85, distance=1.4)

        composite = _grid([
            _label(front, "front (-y face)"),
            _label(iso,   "iso"),
            _label(side,  "side (+x)"),
            _label(top,   "top (overhead)"),
        ])
        out_path = Path("access19_mixed_state.png").resolve()
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        print(f"\n  wrote: {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
