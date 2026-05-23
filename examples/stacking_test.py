"""Stacking + stability test for multilevel_blocks executor.

Builds several tower shapes at several stack-table locations, picking
each cube from a parts source cell.  For every tower we measure:

* **Placement drift** — per cube, the xy distance between the actual
  block pose immediately after ``put_cube`` and the target cell xy.
* **Settle drift** — per cube, the xyz distance between the post-place
  pose and the pose after ``--settle-steps`` of physics with no
  controller input.  Captures both gravity-driven settling AND
  unstable stacks that topple after release.
* **Visual snapshot** — a rendered PNG of the final scene from a fixed
  camera, saved to ``--out-dir``.

Tower shapes:
    1x1xH   single-base tower of height H (1 cube per level)
    2x2xH   2x2-base tower of height H (4 cubes per level)

Locations (default): center, left edge, right edge, front edge, back
edge.  Configurable via ``--locations``.

Usage::

    python examples/stacking_test.py --stack-cells 8 8 5 --out-dir stacks/
    python examples/stacking_test.py --heights 3 --shapes 1x1 --locations center
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.planners.rrt_star import RRTStar
from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_builder,
    oblong_block_name,
)
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    MultilevelBlocksExecutor,
    _HOME_NEUTRAL_Q,
)
from tampanda.symbolic.workspace import Cell


def stack_cell_id(level: int, ix: int, iy: int) -> str:
    return f"stack_L{level}__{ix}_{iy}"


def parts_cell_id(ix: int, iy: int) -> str:
    return f"parts__{ix}_{iy}"


# ---------------------------------------------------------------------------
# Tower templates — return List[Dict] where each dict is a placement spec:
#   {"shape": "cube" | "flat-x" | "flat-y" | "long-x",
#    "block": <block name, e.g. cube_0 / oblong_0 / long_0>,
#    "target_cells": [<cell ids the block occupies, in order>]}
# Build order is the list order; first item placed first.
# ---------------------------------------------------------------------------

def tower_1x1(height: int, ix: int, iy: int) -> List[Dict]:
    """Single-column tower; one cube per level."""
    return [
        {"shape": "cube", "block": cube_block_name(lv),
         "target_cells": [stack_cell_id(lv, ix, iy)]}
        for lv in range(height)
    ]


def tower_2x2(height: int, ix: int, iy: int) -> List[Dict]:
    """2x2 base tower; 4 cubes per level.

    Note: this template traditionally FAILS at cube_2 placement because
    the Franka hand body can't fit between already-placed neighbour cubes.
    Kept for the audit; see cantilever / pyramid templates for working
    multi-block towers using oblong + long blocks.
    """
    out: List[Dict] = []
    cube_idx = 0
    for lv in range(height):
        for dx in (0, 1):
            for dy in (0, 1):
                out.append({
                    "shape": "cube",
                    "block": cube_block_name(cube_idx),
                    "target_cells": [stack_cell_id(lv, ix + dx, iy + dy)],
                })
                cube_idx += 1
    return out


def tower_long_x(height: int, ix: int, iy: int) -> List[Dict]:
    """3x1 long blocks stacked H high.

    Each level = one long-x block spanning 3 cells in x centred on
    (ix, iy).  No adjacency issues — one block per level.
    """
    return [
        {"shape": "long-x", "block": long_block_name(lv),
         "target_cells": [stack_cell_id(lv, ix - 1, iy),
                           stack_cell_id(lv, ix, iy),
                           stack_cell_id(lv, ix + 1, iy)]}
        for lv in range(height)
    ]


def tower_cantilever(ix: int, iy: int) -> List[Dict]:
    """2 cubes at L0 corners + 1 long-x bridge at L1.

    Exercises the 2-of-3 support rule: bridge spans (ix-1, iy),
    (ix, iy), (ix+1, iy) at L1.  Below at L0: (ix-1, iy) and
    (ix+1, iy) are supported (cubes); (ix, iy) is empty.
    Standard support rule rejects this; 2-of-3 relaxation allows it.
    """
    return [
        {"shape": "cube", "block": cube_block_name(0),
         "target_cells": [stack_cell_id(0, ix - 1, iy)]},
        {"shape": "cube", "block": cube_block_name(1),
         "target_cells": [stack_cell_id(0, ix + 1, iy)]},
        {"shape": "long-x", "block": long_block_name(0),
         "target_cells": [stack_cell_id(1, ix - 1, iy),
                           stack_cell_id(1, ix, iy),
                           stack_cell_id(1, ix + 1, iy)]},
    ]


def tower_upright_bridges(ix: int, iy: int) -> List[Dict]:
    """4 × 2x1 upright at L0+L1 corners + 2 × 3x1 flat-y bridges at L2.

    Built by the executor (NOT teleported).  Each upright is picked
    flat-x from parts, transformed in-hand to upright, then placed
    spanning L0+L1 at its corner cell.  The bridges then sit on the
    upright tops at L2.

    Anchor (ix, iy) is the bottom-left of the 3x3 xy footprint:
      L0+L1 corners — uprights at (ix, iy), (ix+2, iy),
                                  (ix, iy+2), (ix+2, iy+2)
      L2 bridges — long-y at columns ix and ix+2, spanning y∈[iy, iy+2]

    The 2-of-3 support rule applies: each bridge's L1 below has
    (ix±, iy) + (ix±, iy+2) supported (upright tops), (ix±, iy+1) empty.
    """
    # Place the BACK row (+y, far from robot) first.  If we did front-
    # row first, then putting a back-row upright would require the
    # gripper (which enters FRONT_Y from -y) to swing OVER the already-
    # placed front uprights — the joint-lerp filter rejects those poses
    # because intermediate configs clip into the placed blocks.
    #
    # Each block has dedicated source_cells on the parts table so the
    # init-on-parts mode can pre-place the full kit visibly before the
    # build begins.  Layout on parts (15x15 grid) — DENSE: 1-cell-gap
    # between blocks in the close-axis direction of each pick (the
    # gripper pre-closes to a 40 mm finger gap before descent, so the
    # finger pad outer edges sit ±28 mm from the EE, leaving 17 mm
    # clearance at 60 mm centre-to-centre):
    #
    #   iy=0  : (0,0)-(1,0)  oblong_0     (3,0)-(4,0)  oblong_1
    #   iy=2  : (0,2)-(1,2)  oblong_2     (3,2)-(4,2)  oblong_3
    #   iy=4..6 : (0,4..6)   long_0       (2,4..6)     long_1   (both flat-y)
    return [
        # Back row at iy+2 of the stack.
        {"shape": "upright", "block": oblong_block_name(0),
         "source_cells": [parts_cell_id(0, 0), parts_cell_id(1, 0)],
         "target_cells": [stack_cell_id(0, ix, iy + 2),
                           stack_cell_id(1, ix, iy + 2)]},
        {"shape": "upright", "block": oblong_block_name(1),
         "source_cells": [parts_cell_id(3, 0), parts_cell_id(4, 0)],
         "target_cells": [stack_cell_id(0, ix + 2, iy + 2),
                           stack_cell_id(1, ix + 2, iy + 2)]},
        # Front row at iy of the stack.
        {"shape": "upright", "block": oblong_block_name(2),
         "source_cells": [parts_cell_id(0, 2), parts_cell_id(1, 2)],
         "target_cells": [stack_cell_id(0, ix, iy),
                           stack_cell_id(1, ix, iy)]},
        {"shape": "upright", "block": oblong_block_name(3),
         "source_cells": [parts_cell_id(3, 2), parts_cell_id(4, 2)],
         "target_cells": [stack_cell_id(0, ix + 2, iy),
                           stack_cell_id(1, ix + 2, iy)]},
        # L2 bridges — 2 long-y blocks at the same x as each upright column.
        {"shape": "long-y", "block": long_block_name(0),
         "source_cells": [parts_cell_id(0, 4), parts_cell_id(0, 5),
                          parts_cell_id(0, 6)],
         "target_cells": [stack_cell_id(2, ix, iy),
                           stack_cell_id(2, ix, iy + 1),
                           stack_cell_id(2, ix, iy + 2)]},
        {"shape": "long-y", "block": long_block_name(1),
         "source_cells": [parts_cell_id(2, 4), parts_cell_id(2, 5),
                          parts_cell_id(2, 6)],
         "target_cells": [stack_cell_id(2, ix + 2, iy),
                           stack_cell_id(2, ix + 2, iy + 1),
                           stack_cell_id(2, ix + 2, iy + 2)]},
    ]


def tower_pyramid(ix: int, iy: int) -> List[Dict]:
    """Pyramidal 3-2-1 tower: long → oblong → cube.

    L0: long-x at (ix-1, iy)+(ix, iy)+(ix+1, iy)  (3 cells)
    L1: oblong flat-x at (ix-1, iy)+(ix, iy)     (2 cells, left-shifted)
    L2: cube at (ix-1, iy)                       (1 cell)

    The upper blocks are progressively smaller and offset to share the
    same anchor at (ix-1, iy) — keeps each upper level supported by the
    layer below.
    """
    return [
        {"shape": "long-x", "block": long_block_name(0),
         "target_cells": [stack_cell_id(0, ix - 1, iy),
                           stack_cell_id(0, ix, iy),
                           stack_cell_id(0, ix + 1, iy)]},
        {"shape": "flat-x", "block": oblong_block_name(0),
         "target_cells": [stack_cell_id(1, ix - 1, iy),
                           stack_cell_id(1, ix, iy)]},
        {"shape": "cube", "block": cube_block_name(0),
         "target_cells": [stack_cell_id(2, ix - 1, iy)]},
    ]


class StackingTester:
    """Build towers cube-by-cube and measure drift + stability."""

    def __init__(self, stack_cells: Tuple[int, int, int],
                  parts_cells: Tuple[int, int],
                  n_cubes: int,
                  n_oblong: int,
                  n_long: int,
                  cube_size: Optional[float],
                  hand_capsule_radius: Optional[float],
                  hand_capsule_disabled: bool,
                  verbose: bool):
        kwargs: Dict = dict(stack_grid_cells=stack_cells,
                                parts_grid_cells=parts_cells,
                                n_cubes=n_cubes,
                                n_oblong=n_oblong,
                                n_long=n_long)
        if cube_size is not None:
            kwargs["cube_half_extent"] = cube_size / 2
        cfg = MultilevelBlocksConfig(**kwargs)
        self._scratch = tempfile.TemporaryDirectory(prefix="stacking_test_")
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(self._scratch.name), config=cfg,
        )
        self.cfg = cfg
        self.ws = ws
        self.env = builder.build_env(rate=10000.0)

        # Optionally shrink or fully disable the hand_capsule collision
        # geom.  The 4cm-radius / 6cm-half-length capsule is the
        # planner's conservative approximation of the Franka hand;
        # for dense structures (upright_bridges et al) the AXIAL extent
        # (10cm half-length+cap) reaches across row gaps into placed
        # neighbour blocks, rejecting feasible plans.  Disabling it
        # falls back to the hand_c mesh (which is checked anyway during
        # real contact), so real collisions are still detected — we
        # just stop the conservative bounding capsule from over-
        # rejecting valid plans.
        if hand_capsule_disabled:
            # contype=0, conaffinity=0 → MuJoCo skips this geom in
            # broad-phase collision detection.
            for gid in range(self.env.model.ngeom):
                if self.env.model.geom(gid).name == "hand_capsule":
                    self.env.model.geom_contype[gid] = 0
                    self.env.model.geom_conaffinity[gid] = 0
                    break
        elif hand_capsule_radius is not None:
            for gid in range(self.env.model.ngeom):
                if self.env.model.geom(gid).name == "hand_capsule":
                    self.env.model.geom_size[gid][0] = float(
                        hand_capsule_radius)
                    break

        self._cube_names = [cube_block_name(i) for i in range(cfg.n_cubes)]
        self._oblong_names = [oblong_block_name(i)
                                  for i in range(cfg.n_oblong)]
        self._long_names = [long_block_name(i) for i in range(cfg.n_long)]
        self._all_names = (self._cube_names + self._oblong_names
                                + self._long_names)

        rrt = RRTStar(self.env, max_iterations=3000)
        self.executor = MultilevelBlocksExecutor(
            self.env, self.ws, cfg, motion_planner=rrt, verbose=verbose,
        )

        # Renderer for post-tower snapshots.
        self.env.model.vis.global_.offwidth = max(
            1280, self.env.model.vis.global_.offwidth)
        self.env.model.vis.global_.offheight = max(
            720, self.env.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(self.env.model,
                                                height=720, width=1280)
        # Primary camera — aimed at the stack table for build-progress
        # snapshots.
        self._cam = mujoco.MjvCamera()
        self._cam.azimuth = 45
        self._cam.elevation = -25
        self._cam.distance = 1.6
        sp = cfg.stack_table_pos
        self._cam.lookat[:] = [sp[0], sp[1], sp[2] + 0.30]

        # Wide camera — shows BOTH tables; used for the initial-kit
        # snapshot when init-on-parts mode is active.
        self._cam_wide = mujoco.MjvCamera()
        self._cam_wide.azimuth = 90
        self._cam_wide.elevation = -45
        self._cam_wide.distance = 2.2
        pp = cfg.parts_table_pos
        self._cam_wide.lookat[:] = [
            (sp[0] + pp[0]) / 2,
            (sp[1] + pp[1]) / 2,
            0.30,
        ]

        self._reset_all()

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass
        try:
            self._scratch.cleanup()
        except Exception:
            pass

    # ---- helpers -------------------------------------------------------

    def _park_all(self) -> None:
        for i, name in enumerate(self._all_names):
            parked = np.array([self.cfg.hide_far_x + 0.15 * i, 0.0, 0.05])
            self.env.set_object_pose(name, parked)
        self.env.reset_velocities()
        self.env.forward()

    def _reset_arm(self) -> None:
        if self.executor._held_block is not None:
            try:
                self.executor._detach_open()
            except Exception:
                self.executor._held_block = None
                self.executor._held_offset = np.zeros(3)
        self.env.data.qpos[:7] = _HOME_NEUTRAL_Q
        if self.env.data.qpos.size >= 9:
            self.env.data.qpos[7] = 0.04
            self.env.data.qpos[8] = 0.04
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.reset_velocities()
        self.env.forward()

    def _reset_all(self) -> None:
        self._park_all()
        self._reset_arm()

    def _cell_world(self, cell_id: str) -> np.ndarray:
        return np.asarray(self.ws.pose_for(Cell.parse(cell_id)),
                              dtype=float)

    def _block_pos(self, name: str) -> np.ndarray:
        pos, _ = self.env.get_object_pose(name)
        return np.asarray(pos)

    # ---- physics-settle + render --------------------------------------

    def _settle(self, n_steps: int) -> None:
        """Run physics WITHOUT controller updates so blocks settle on
        their own.  Captures stack instability — the arm holds its
        current ctrl, the blocks experience only gravity + contacts.
        """
        for _ in range(n_steps):
            self.env.step()

    def _snapshot_wide(self, output_path: Path) -> None:
        """Render with the wide camera (both tables visible)."""
        self._renderer.update_scene(self.env.data, camera=self._cam_wide)
        img = self._renderer.render()
        try:
            import cv2  # type: ignore
            cv2.imwrite(str(output_path), img[:, :, ::-1])
        except Exception:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.imsave(str(output_path), img)
            except Exception as e:
                print(f"  WARN: could not save wide snapshot "
                          f"{output_path}: {e}")

    def _snapshot(self, output_path: Path) -> None:
        self._renderer.update_scene(self.env.data, camera=self._cam)
        img = self._renderer.render()
        # Save via OpenCV-compatible RGB→BGR + imwrite, fallback to
        # matplotlib if cv2 unavailable.
        try:
            import cv2  # type: ignore
            cv2.imwrite(str(output_path), img[:, :, ::-1])
        except Exception:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.imsave(str(output_path), img)
            except Exception as e:
                print(f"  WARN: could not save snapshot to {output_path}: {e}")

    # ---- tower build ---------------------------------------------------

    # ---- shape → (place-helper, source-cells, pick-fn, put-fn) -----------

    def _shape_sources(self, shape: str) -> List[str]:
        """Default parts cells where a block of `shape` is teleported."""
        if shape == "cube":
            return [parts_cell_id(0, 0)]
        if shape == "flat-x":
            return [parts_cell_id(0, 0), parts_cell_id(1, 0)]
        if shape == "flat-y":
            return [parts_cell_id(0, 0), parts_cell_id(0, 1)]
        if shape == "long-x":
            return [parts_cell_id(0, 0), parts_cell_id(1, 0),
                      parts_cell_id(2, 0)]
        if shape == "long-y":
            return [parts_cell_id(0, 0), parts_cell_id(0, 1),
                      parts_cell_id(0, 2)]
        if shape == "upright":
            # Source is FLAT-X on parts; executor transforms it to upright
            # in-hand before put_upright at the target.
            return [parts_cell_id(0, 0), parts_cell_id(1, 0)]
        raise ValueError(f"unknown shape: {shape!r}")

    def _place_at_source(self, block: str, shape: str,
                              sources: List[str]) -> None:
        """Teleport `block` to its source cells on the parts table."""
        if shape == "cube":
            pos = self._cell_world(sources[0])
            self.env.set_object_pose(block, pos)
        elif shape in ("flat-x", "long-x", "upright"):
            # Upright source is flat-x on parts (then transformed in-hand).
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            p1 = self._cell_world(sources[0])
            pN = self._cell_world(sources[-1])
            self.env.set_object_pose(block, (p1 + pN) / 2, quat)
        elif shape in ("flat-y", "long-y"):
            quat = np.array([0.7071068, 0.0, 0.0, 0.7071068])
            p1 = self._cell_world(sources[0])
            pN = self._cell_world(sources[-1])
            self.env.set_object_pose(block, (p1 + pN) / 2, quat)
        self.env.reset_velocities()
        self.env.forward()

    def _do_pick_put(self, block: str, shape: str,
                          sources: List[str],
                          target_cells: List[str]) -> Tuple[bool, str]:
        """Dispatch to the right pick/put pair on the executor.

        Returns (ok, phase) where phase ∈ {"pick", "put", "done"}.
        """
        if shape == "cube":
            if not self.executor.pick_cube(block, sources[0]):
                return False, "pick"
            if not self.executor.put_cube(block, target_cells[0]):
                return False, "put"
        elif shape == "flat-x":
            if not self.executor.pick_flat_x(block, sources[0], sources[1]):
                return False, "pick"
            if not self.executor.put_flat_x(block, target_cells[0],
                                                  target_cells[1]):
                return False, "put"
        elif shape == "flat-y":
            if not self.executor.pick_flat_y(block, sources[0], sources[1]):
                return False, "pick"
            if not self.executor.put_flat_y(block, target_cells[0],
                                                  target_cells[1]):
                return False, "put"
        elif shape == "long-x":
            if not self.executor.pick_long_x(block, sources[0], sources[1],
                                                    sources[2]):
                return False, "pick"
            if not self.executor.put_long_x(block, target_cells[0],
                                                  target_cells[1],
                                                  target_cells[2]):
                return False, "put"
        elif shape == "long-y":
            if not self.executor.pick_long_y(block, sources[0], sources[1],
                                                    sources[2]):
                return False, "pick"
            if not self.executor.put_long_y(block, target_cells[0],
                                                  target_cells[1],
                                                  target_cells[2]):
                return False, "put"
        elif shape == "upright":
            # pick flat-x from parts → in-hand transform → put upright.
            if not self.executor.pick_flat_x(block, sources[0],
                                                    sources[1]):
                return False, "pick"
            if not self.executor.make_upright_from_x(block):
                return False, "make_upright"
            if not self.executor.put_upright(block, target_cells[0],
                                                  target_cells[1]):
                return False, "put"
        else:
            raise ValueError(f"unknown shape: {shape!r}")
        return True, "done"

    def _expected_centroid(self, shape: str,
                                target_cells: List[str]) -> np.ndarray:
        """World xy(z) where the block centroid should land."""
        if shape == "cube":
            return self._cell_world(target_cells[0])
        # Multi-cell (flat / long): centroid is midpoint of first + last cell.
        return (self._cell_world(target_cells[0])
                    + self._cell_world(target_cells[-1])) / 2

    # ---- tower build ---------------------------------------------------

    def build_tower(self,
                       placements: List[Dict],
                       label: str,
                       out_dir: Path,
                       settle_steps: int,
                       per_step_snapshots: bool = False,
                       init_on_parts: bool = False) -> Dict:
        """Build a tower one block at a time.

        `placements` is a list of dicts with keys ``shape``, ``block``,
        ``target_cells``.  Optionally each may have ``source_cells`` —
        if present, the block is picked from those cells; otherwise the
        default _shape_sources(shape) is used.

        When ``init_on_parts`` is True, all blocks are pre-placed at
        their source cells (requires unique source_cells per block)
        BEFORE the build loop starts, and an ``{label}_initial.png``
        wide-view snapshot is rendered.  Otherwise blocks are teleported
        one at a time per build step (default).

        When ``per_step_snapshots`` is True, save a snapshot to
        ``out_dir/{label}_step{i:02d}_{block}.png`` after each block,
        and ``..._step{i:02d}_FAILED_{block}.png`` at the failure point.
        """
        self._reset_all()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _sources_for(p: Dict) -> List[str]:
            return p.get("source_cells") or self._shape_sources(p["shape"])

        # Optional init-on-parts: pre-place all blocks at their source
        # cells so the full kit is visible before the build.
        if init_on_parts:
            for p in placements:
                self._place_at_source(p["block"], p["shape"],
                                            _sources_for(p))
            initial_path = out_dir / f"{label}_initial.png"
            self._snapshot_wide(initial_path)
            print(f"  initial-kit snapshot → {initial_path.name}")

        per_block: List[Dict] = []
        post_place_names: List[str] = []
        post_place_targets: List[np.ndarray] = []

        for step_idx, p in enumerate(placements):
            shape = p["shape"]
            block = p["block"]
            target_cells = p["target_cells"]
            sources = _sources_for(p)
            target_xyz = self._expected_centroid(shape, target_cells)

            # In init-on-parts mode, the block is already at the source.
            # Otherwise teleport it just-in-time before the pick.
            if not init_on_parts:
                self._place_at_source(block, shape, sources)

            t0 = time.time()
            try:
                ok, phase = self._do_pick_put(block, shape, sources,
                                                       target_cells)
                if not ok:
                    per_block.append({
                        "block": block, "shape": shape,
                        "target": "+".join(target_cells),
                        "ok": False, "phase": phase,
                        "t": time.time() - t0,
                    })
                    if per_step_snapshots:
                        fail_path = out_dir / (
                            f"{label}_step{step_idx:02d}_"
                            f"FAILED_{block}.png")
                        self._snapshot(fail_path)
                        print(f"    (failure snapshot → {fail_path.name})")
                    break
            except Exception as e:
                per_block.append({
                    "block": block, "shape": shape,
                    "target": "+".join(target_cells),
                    "ok": False, "phase": "exception",
                    "t": time.time() - t0,
                    "exc": f"{type(e).__name__}: {e}",
                })
                if per_step_snapshots:
                    fail_path = out_dir / (
                        f"{label}_step{step_idx:02d}_"
                        f"EXC_{block}.png")
                    self._snapshot(fail_path)
                break

            actual = self._block_pos(block)
            drift_xy = float(np.linalg.norm(actual[:2] - target_xyz[:2]))
            per_block.append({
                "block": block, "shape": shape,
                "target": "+".join(target_cells),
                "ok": True, "phase": "done",
                "t": time.time() - t0,
                "drift_xy": drift_xy,
                "actual_xyz": actual.tolist(),
                "target_xyz": target_xyz.tolist(),
            })
            post_place_names.append(block)
            post_place_targets.append(target_xyz)
            if per_step_snapshots:
                step_path = out_dir / (
                    f"{label}_step{step_idx:02d}_{block}.png")
                self._snapshot(step_path)

        # All placed (or partial).  Run physics settle + measure drift.
        pre_settle: List[np.ndarray] = [self._block_pos(n).copy()
                                                for n in post_place_names]
        self._settle(settle_steps)
        settle_drift: List[float] = []
        post_settle_positions: List[np.ndarray] = []
        for n, pre in zip(post_place_names, pre_settle):
            post = self._block_pos(n)
            settle_drift.append(float(np.linalg.norm(post - pre)))
            post_settle_positions.append(post.copy())

        # Snapshot.
        snap_path = out_dir / f"{label}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot(snap_path)

        all_placed = all(r["ok"] for r in per_block)
        max_place_drift = max(
            (r["drift_xy"] for r in per_block if r["ok"]), default=0.0)
        max_settle = max(settle_drift, default=0.0)

        return {
            "label": label,
            "ok": all_placed,
            "n_blocks": len(placements),
            "per_block": per_block,
            "settle_drift_per_block": settle_drift,
            "max_place_drift_xy": max_place_drift,
            "max_settle_drift": max_settle,
            "snapshot": str(snap_path),
        }


# --- location selection ---------------------------------------------------

def location_options(cells_x: int, cells_y: int,
                          footprint: Tuple[int, int]) -> Dict[str, Tuple[int, int]]:
    """Named (ix, iy) anchor points for a tower of `footprint` size."""
    fx, fy = footprint
    max_ix = cells_x - fx
    max_iy = cells_y - fy
    cx = max_ix // 2
    cy = max_iy // 2
    return {
        "center":      (cx, cy),
        "left_edge":   (0, cy),
        "right_edge":  (max_ix, cy),
        "front_edge":  (cx, 0),
        "back_edge":   (cx, max_iy),
        "corner_fl":   (0, 0),                  # front-left
        "corner_fr":   (max_ix, 0),             # front-right
        "corner_bl":   (0, max_iy),             # back-left
        "corner_br":   (max_ix, max_iy),        # back-right
    }


# --- summary --------------------------------------------------------------

def summarise(results: List[Dict],
                  drift_warn: float,
                  settle_warn: float) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'label':<26s} {'blocks':>6s} {'built':>6s} "
              f"{'place(mm)':>10s} {'settle(mm)':>11s}")
    print("-" * 70)
    for r in results:
        built = "YES" if r["ok"] else "NO"
        place_mm = r.get("max_place_drift_xy", 0.0) * 1000.0
        settle_mm = r.get("max_settle_drift", 0.0) * 1000.0
        flags = ""
        if place_mm > drift_warn * 1000.0:
            flags += "!"
        if settle_mm > settle_warn * 1000.0:
            flags += "*"
        n_blocks = r.get("n_blocks", r.get("n_cubes", 0))
        print(f"  {r['label']:<24s} {n_blocks:>6d} {built:>6s}  "
                  f"{place_mm:>8.1f}   {settle_mm:>9.1f}  {flags}")
    print(f"\n  ! = place drift > {drift_warn*1000:.0f} mm  "
              f"* = settle drift > {settle_warn*1000:.0f} mm")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stack-cells", nargs=3, type=int, default=[8, 8, 5],
                       metavar=("X", "Y", "Z"))
    p.add_argument("--parts-cells", nargs=2, type=int, default=[15, 15],
                       metavar=("X", "Y"))
    p.add_argument("--cube-size", type=float, default=None)
    p.add_argument("--n-cubes", type=int, default=20,
                       help="cubes instantiated (must be >= the largest "
                            "tower's cube count; default 20 covers 2x2x4)")
    p.add_argument("--n-oblong", type=int, default=4,
                       help="oblong blocks; need >=4 for upright_bridges "
                            "(4 uprights), >=1 for pyramid")
    p.add_argument("--n-long", type=int, default=4,
                       help="3x1 long blocks; need >=H for long_x at "
                            "height H, or >=2 for upright_bridges / "
                            "cantilever / pyramid")
    p.add_argument("--shapes", nargs="+", default=["1x1", "2x2"],
                       choices=["1x1", "2x2", "long_x",
                                  "cantilever", "pyramid",
                                  "upright_bridges"])
    p.add_argument("--heights", nargs="+", type=int, default=[3, 4],
                       help="heights apply to 1x1, 2x2, long_x.  "
                            "Ignored by cantilever (height=2) and "
                            "pyramid (height=3).")
    p.add_argument("--locations", nargs="+",
                       default=["center", "left_edge", "right_edge",
                                  "front_edge", "back_edge"])
    p.add_argument("--settle-steps", type=int, default=400,
                       help="physics steps after last placement (no "
                            "controller input) to detect topple")
    p.add_argument("--drift-warn", type=float, default=0.010,
                       help="place-drift threshold (m) for ! flag")
    p.add_argument("--settle-warn", type=float, default=0.005,
                       help="settle-drift threshold (m) for * flag")
    p.add_argument("--out-dir", type=str, default="stacking_test_out")
    p.add_argument("--results-json", type=str, default=None)
    p.add_argument("--hand-capsule-radius", type=float, default=0.02,
                       help="Override the Franka hand_capsule collision "
                            "radius (default 0.02 m).  The MJCF default "
                            "is 0.04 m which is overly conservative for "
                            "dense neighbour configurations (e.g., "
                            "upright_bridges).  Set to 0.04 to disable "
                            "the shrink.  Note: radius shrink only "
                            "helps PERPENDICULAR clearance.  For dense "
                            "structures where the capsule's AXIAL "
                            "extent (~10cm half-length+cap) reaches "
                            "across row gaps, use --disable-hand-capsule "
                            "instead.")
    p.add_argument("--disable-hand-capsule", action="store_true",
                       help="Disable the hand_capsule collision geom "
                            "entirely (contype=conaffinity=0).  Real "
                            "hand-vs-object collisions are still "
                            "detected via the hand_c mesh; only the "
                            "conservative bounding capsule is removed. "
                            "Required for upright_bridges and other "
                            "dense structures whose axial-capsule "
                            "extent over-rejects valid plans.")
    p.add_argument("--per-step-snapshots", action="store_true",
                       help="Render a PNG after each successful block "
                            "placement and at any failure point.  Saved "
                            "as {label}_stepNN_{block}.png in --out-dir.")
    p.add_argument("--init-on-parts", action="store_true",
                       help="Pre-place ALL blocks on the parts table at "
                            "the start (each block at dedicated source "
                            "cells defined by the template).  Renders "
                            "{label}_initial.png with a wide camera "
                            "showing both tables.  Requires the chosen "
                            "tower template to provide source_cells per "
                            "block (currently: upright_bridges).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    stack_cells = tuple(args.stack_cells)
    parts_cells = tuple(args.parts_cells)
    out_dir = Path(args.out_dir)

    tester = StackingTester(stack_cells, parts_cells, args.n_cubes,
                                  args.n_oblong, args.n_long,
                                  args.cube_size,
                                  args.hand_capsule_radius,
                                  args.disable_hand_capsule,
                                  args.verbose)

    results: List[Dict] = []

    def _footprint_for(shape: str) -> Tuple[int, int]:
        # Used only for location_options to keep the tower in-bounds.
        if shape == "1x1":
            return (1, 1)
        if shape == "2x2":
            return (2, 2)
        if shape == "upright_bridges":
            return (3, 3)  # 3x3 xy footprint
        # long_x / cantilever / pyramid span 3 cells in x and 1 in y.
        # Templates use ix-1/ix/ix+1, so leave room on both sides.
        return (3, 1)

    def _build_placements(shape: str, height: int,
                                ix: int, iy: int) -> Tuple[List[Dict], int]:
        """Return (placements, effective_height) for this shape."""
        if shape == "1x1":
            return tower_1x1(height, ix, iy), height
        if shape == "2x2":
            return tower_2x2(height, ix, iy), height
        if shape == "long_x":
            return tower_long_x(height, ix, iy), height
        if shape == "cantilever":
            return tower_cantilever(ix, iy), 2  # fixed H=2
        if shape == "pyramid":
            return tower_pyramid(ix, iy), 3     # fixed H=3
        if shape == "upright_bridges":
            return tower_upright_bridges(ix, iy), 3  # fixed H=3 (L0+L1+L2)
        raise ValueError(f"unknown shape: {shape!r}")

    try:
        # Build (shape, height, location) test matrix.
        cells_x, cells_y, _ = stack_cells
        for shape in args.shapes:
            footprint = _footprint_for(shape)
            locs = location_options(cells_x, cells_y, footprint)
            # For 3-wide footprints (centred), shift the anchor by +1 in
            # x so location_options' "(0, _)" maps to ix=1 (which has
            # room for the ix-1 left edge of the long block).
            fixed_height = shape in ("cantilever", "pyramid",
                                          "upright_bridges")
            heights = [None] if fixed_height else args.heights
            for height in heights:
                for loc_name in args.locations:
                    if loc_name not in locs:
                        print(f"  WARN: unknown location {loc_name!r}; skipping")
                        continue
                    ix, iy = locs[loc_name]
                    # 3-wide templates (long_x/cantilever/pyramid) use
                    # ix-1/ix/ix+1; shift anchor to keep edges in-bounds.
                    # upright_bridges uses anchor as bottom-left of a 3x3
                    # footprint — location_options already does this, no
                    # shift needed.
                    if footprint == (3, 1):
                        ix = max(1, ix)
                        ix = min(cells_x - 2, ix)
                    cells, eff_h = _build_placements(shape, height or 0,
                                                              ix, iy)
                    if fixed_height:
                        label = f"{shape}_{loc_name}"
                    else:
                        label = f"{shape}x{height}_{loc_name}"
                    print(f"\n--- {label}  (anchor=({ix},{iy}), "
                              f"{len(cells)} blocks) ---")
                    r = tester.build_tower(cells, label, out_dir,
                                                  args.settle_steps,
                                                  per_step_snapshots=
                                                      args.per_step_snapshots,
                                                  init_on_parts=
                                                      args.init_on_parts)
                    results.append(r)
                    if r["ok"]:
                        print(f"  built OK; "
                                  f"max_place_drift={r['max_place_drift_xy']*1000:.1f}mm  "
                                  f"max_settle_drift={r['max_settle_drift']*1000:.1f}mm  "
                                  f"snapshot={r['snapshot']}")
                    else:
                        first_fail = next(
                            (c for c in r.get("per_block", []) if not c["ok"]),
                            None)
                        if first_fail:
                            extra = (f" exc=({first_fail.get('exc', '')})"
                                          if "exc" in first_fail else "")
                            print(f"  build FAILED at {first_fail['target']} "
                                      f"phase={first_fail['phase']}{extra}")
                        else:
                            print(f"  build FAILED: {r.get('error', '?')}")
    except KeyboardInterrupt:
        print("\nInterrupted; printing partial summary.")
    finally:
        summarise(results, args.drift_warn, args.settle_warn)
        if args.results_json:
            Path(args.results_json).write_text(json.dumps(results, indent=2))
            print(f"\nResults JSON written to {args.results_json}")
        tester.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
