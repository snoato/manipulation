"""L0 reliability test for multilevel_blocks executor.

Runs the full matrix of pick-from-parts → put-on-stack-L0 tests:
- cube at every L0 cell                 (cells_x * cells_y)
- flat-x oblong at every adjacent pair  ((cells_x-1) * cells_y)
- flat-y oblong at every adjacent pair  (cells_x * (cells_y-1))
- upright oblong at every L0 cell       (cells_x * cells_y)

Usage::

    python examples/reliability_l0.py --stack-cells 8 8 5
    python examples/reliability_l0.py --shapes cube upright --output out.json

Tests are independent: each starts by parking every block, placing the
test block at a fixed parts source cell, and resetting the arm to
neutral HOME with an open gripper.  Success = executor reports True for
every step AND the block ends up within ``--xy-tol`` of the target xy.
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


# Quat for an oblong lying flat with its long axis along world-x (identity).
_QUAT_FLAT_X = np.array([1.0, 0.0, 0.0, 0.0])
# 90 deg around z: long axis along world-y.
_QUAT_FLAT_Y = np.array([0.7071068, 0.0, 0.0, 0.7071068])


def stack_cell(level: int, ix: int, iy: int) -> str:
    return f"stack_L{level}__{ix}_{iy}"


def parts_cell(ix: int, iy: int) -> str:
    return f"parts__{ix}_{iy}"


class Tester:
    """Builds the env once, then runs many independent test cases."""

    def __init__(self, stack_cells: Tuple[int, int, int],
                  parts_cells: Tuple[int, int],
                  cube_size: Optional[float],
                  n_cubes: int,
                  n_oblong: int,
                  n_long: int,
                  verbose: bool):
        # n_oblong / n_long default to 0 in MultilevelBlocksConfig — the
        # test needs at least one body of each type or addressing
        # ``oblong_0`` / ``long_0`` crashes.  n_cubes bumped from
        # the config default (8) to support elevated-source tests that
        # pre-place support stacks below the test cube.
        kwargs: Dict = dict(stack_grid_cells=stack_cells,
                                parts_grid_cells=parts_cells,
                                n_cubes=n_cubes,
                                n_oblong=n_oblong,
                                n_long=n_long)
        if cube_size is not None:
            kwargs["cube_half_extent"] = cube_size / 2
        cfg = MultilevelBlocksConfig(**kwargs)
        self._scratch = tempfile.TemporaryDirectory(prefix="reliability_l0_")
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(self._scratch.name), config=cfg,
        )
        self.cfg = cfg
        self.ws = ws
        self.env = builder.build_env(rate=10000.0)

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

        self._park_all()
        self._reset_arm()

    def close(self) -> None:
        try:
            self._scratch.cleanup()
        except Exception:
            pass

    # ---- reset helpers ------------------------------------------------

    def _park_all(self) -> None:
        for i, name in enumerate(self._all_names):
            parked = np.array([self.cfg.hide_far_x + 0.15 * i, 0.0, 0.05])
            self.env.set_object_pose(name, parked)
        self.env.reset_velocities()
        self.env.forward()

    def _reset_arm(self) -> None:
        # Force-detach any held block (the executor's attach_body is what
        # keeps the gripper holding it; a failed put may leave _held_block
        # set with the body still attached).
        if self.executor._held_block is not None:
            try:
                self.executor._detach_open()
            except Exception:
                self.executor._held_block = None
                self.executor._held_offset = np.zeros(3)
        self.env.data.qpos[:7] = _HOME_NEUTRAL_Q
        # Gripper open (both fingers).
        if self.env.data.qpos.size >= 9:
            self.env.data.qpos[7] = 0.04
            self.env.data.qpos[8] = 0.04
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.reset_velocities()
        self.env.forward()

    def reset_between(self) -> None:
        self._park_all()
        self._reset_arm()

    # ---- placement helpers --------------------------------------------

    def _cell_world(self, cell_id: str) -> np.ndarray:
        return np.asarray(self.ws.pose_for(Cell.parse(cell_id)),
                              dtype=float)

    def place_cube(self, name: str, cell_id: str) -> np.ndarray:
        pos = self._cell_world(cell_id)
        self.env.set_object_pose(name, pos)
        self.env.reset_velocities()
        self.env.forward()
        return pos

    def place_flat_x(self, name: str, c1: str, c2: str) -> np.ndarray:
        p1 = self._cell_world(c1)
        p2 = self._cell_world(c2)
        centre = (p1 + p2) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()
        return centre

    def place_flat_y(self, name: str, c1: str, c2: str) -> np.ndarray:
        p1 = self._cell_world(c1)
        p2 = self._cell_world(c2)
        centre = (p1 + p2) / 2
        self.env.set_object_pose(name, centre, _QUAT_FLAT_Y)
        self.env.reset_velocities()
        self.env.forward()
        return centre

    # ---- support stacks ------------------------------------------------

    def _support_cells_for(self, source_cells: List[str],
                                shape: str) -> List[str]:
        """Compute support cell IDs needed below an elevated stack source.

        For stack source at level L_k > 0, the source block falls to L0
        during the pick chain unless a support stack is placed below.
        Returns the list of cells where cube supports must be teleported.
        Empty list if source is on parts or already at L0.

        shape determines which xy cells need support:
          - flat shapes (cube, flat-x/y, long-x/y) → all source xy cells
          - vertical shapes (upright, long-upright) → single xy (c_low's)
        """
        cells = [Cell.parse(c) for c in source_cells]
        if not all(c.region.startswith("stack_L") for c in cells):
            return []  # parts source — no support needed

        if shape in ("cube", "flat-x", "flat-y", "long-x", "long-y"):
            lvls = [int(c.region.split("_L")[1]) for c in cells]
            src_level = min(lvls)
            xy_cells = [(c.ix, c.iy) for c in cells]
        elif shape in ("upright", "long-upright"):
            # First cell is c_low for upright / long-upright signatures.
            c_low = cells[0]
            src_level = int(c_low.region.split("_L")[1])
            xy_cells = [(c_low.ix, c_low.iy)]
        else:
            raise ValueError(f"unknown shape {shape!r}")

        if src_level == 0:
            return []

        support_cells = []
        for ix, iy in xy_cells:
            for lvl in range(src_level):
                support_cells.append(f"stack_L{lvl}__{ix}_{iy}")
        return support_cells

    def _place_supports(self, support_cells: List[str]) -> None:
        """Teleport cube supports into the given cells.

        Uses cube_1, cube_2, ... — cube_0 is reserved for the test block.
        Raises if there aren't enough cubes in the config.
        """
        if not support_cells:
            return
        n_needed = len(support_cells)
        n_avail = self.cfg.n_cubes - 1
        if n_needed > n_avail:
            raise RuntimeError(
                f"Need {n_needed} support cubes but only {n_avail} "
                f"available (n_cubes={self.cfg.n_cubes}).  Increase --n-cubes."
            )
        for i, cell_id in enumerate(support_cells):
            support_block = cube_block_name(i + 1)
            self.env.set_object_pose(support_block,
                                            self._cell_world(cell_id))
        self.env.reset_velocities()
        self.env.forward()

    # ---- verification --------------------------------------------------

    def _verify_xy(self, name: str, expected_pos: np.ndarray,
                       xy_tol: float) -> Tuple[bool, float]:
        actual_pos, _ = self.env.get_object_pose(name)
        actual_pos = np.asarray(actual_pos)
        err = float(np.linalg.norm(actual_pos[:2] - expected_pos[:2]))
        return (err <= xy_tol), err

    # ---- test cases ----------------------------------------------------

    def test_cube(self, target: str, xy_tol: float,
                       source: Optional[str] = None) -> Dict:
        """pick cube_0 from `source` (default parts__0_0) → put on `target`."""
        src = source or parts_cell(0, 0)
        self.reset_between()
        block = cube_block_name(0)
        self._place_supports(self._support_cells_for([src], "cube"))
        self.place_cube(block, src)

        t0 = time.time()
        try:
            ok_pick = self.executor.pick_cube(block, src)
            if not ok_pick:
                return {"ok": False, "phase": "pick",
                            "t": time.time() - t0}
            ok_put = self.executor.put_cube(block, target)
            if not ok_put:
                return {"ok": False, "phase": "put",
                            "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        expected = self._cell_world(target)
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_flat_x(self, c1: str, c2: str, xy_tol: float,
                          source1: Optional[str] = None,
                          source2: Optional[str] = None) -> Dict:
        src1 = source1 or parts_cell(0, 0)
        src2 = source2 or parts_cell(1, 0)
        self.reset_between()
        block = oblong_block_name(0)
        self._place_supports(
            self._support_cells_for([src1, src2], "flat-x"))
        self.place_flat_x(block, src1, src2)

        t0 = time.time()
        try:
            if not self.executor.pick_flat_x(block, src1, src2):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.put_flat_x(block, c1, c2):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        expected = (self._cell_world(c1) + self._cell_world(c2)) / 2
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_flat_y(self, c1: str, c2: str, xy_tol: float,
                          source1: Optional[str] = None,
                          source2: Optional[str] = None) -> Dict:
        src1 = source1 or parts_cell(0, 0)
        src2 = source2 or parts_cell(0, 1)
        self.reset_between()
        block = oblong_block_name(0)
        self._place_supports(
            self._support_cells_for([src1, src2], "flat-y"))
        self.place_flat_y(block, src1, src2)

        t0 = time.time()
        try:
            if not self.executor.pick_flat_y(block, src1, src2):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.put_flat_y(block, c1, c2):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        expected = (self._cell_world(c1) + self._cell_world(c2)) / 2
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_long_x(self, c1: str, c2: str, c3: str,
                          xy_tol: float,
                          source1: Optional[str] = None,
                          source2: Optional[str] = None,
                          source3: Optional[str] = None) -> Dict:
        """pick 3×1 long flat-x from source triple → put at target triple."""
        src1 = source1 or parts_cell(0, 0)
        src2 = source2 or parts_cell(1, 0)
        src3 = source3 or parts_cell(2, 0)
        self.reset_between()
        block = long_block_name(0)
        self._place_supports(
            self._support_cells_for([src1, src2, src3], "long-x"))
        # Place the long block at the midpoint of src1..src3.
        p1 = self._cell_world(src1)
        p3 = self._cell_world(src3)
        centre = (p1 + p3) / 2
        self.env.set_object_pose(block, centre, _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()

        t0 = time.time()
        try:
            if not self.executor.pick_long_x(block, src1, src2, src3):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.put_long_x(block, c1, c2, c3):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        expected = (self._cell_world(c1) + self._cell_world(c3)) / 2
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_long_y(self, c1: str, c2: str, c3: str,
                          xy_tol: float,
                          source1: Optional[str] = None,
                          source2: Optional[str] = None,
                          source3: Optional[str] = None) -> Dict:
        """pick 3×1 long flat-y from source triple → put at target triple."""
        src1 = source1 or parts_cell(0, 0)
        src2 = source2 or parts_cell(0, 1)
        src3 = source3 or parts_cell(0, 2)
        self.reset_between()
        block = long_block_name(0)
        self._place_supports(
            self._support_cells_for([src1, src2, src3], "long-y"))
        p1 = self._cell_world(src1)
        p3 = self._cell_world(src3)
        centre = (p1 + p3) / 2
        self.env.set_object_pose(block, centre, _QUAT_FLAT_Y)
        self.env.reset_velocities()
        self.env.forward()

        t0 = time.time()
        try:
            if not self.executor.pick_long_y(block, src1, src2, src3):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.put_long_y(block, c1, c2, c3):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        expected = (self._cell_world(c1) + self._cell_world(c3)) / 2
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_long_upright(self, c_low: str, c_mid: str, c_high: str,
                                xy_tol: float) -> Dict:
        """pick flat-x from parts → make_long_upright_from_x → put_long_upright."""
        self.reset_between()
        block = long_block_name(0)
        src1 = parts_cell(0, 0)
        src2 = parts_cell(1, 0)
        src3 = parts_cell(2, 0)
        p1 = self._cell_world(src1)
        p3 = self._cell_world(src3)
        centre = (p1 + p3) / 2
        self.env.set_object_pose(block, centre, _QUAT_FLAT_X)
        self.env.reset_velocities()
        self.env.forward()

        t0 = time.time()
        try:
            if not self.executor.pick_long_x(block, src1, src2, src3):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.make_long_upright_from_x(block):
                return {"ok": False, "phase": "make_upright",
                            "t": time.time() - t0}
            if not self.executor.put_long_upright(block, c_low, c_mid, c_high):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        # Upright 3×1 centroid sits at c_mid (midway between c_low and c_high
        # in z).  Verify against c_mid xy.
        expected = self._cell_world(c_mid)
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }

    def test_upright(self, c_low: str, c_high: str, xy_tol: float) -> Dict:
        """pick flat-x from parts → make-upright-from-x → put-upright."""
        self.reset_between()
        block = oblong_block_name(0)
        src1 = parts_cell(0, 0)
        src2 = parts_cell(1, 0)
        self.place_flat_x(block, src1, src2)

        t0 = time.time()
        try:
            if not self.executor.pick_flat_x(block, src1, src2):
                return {"ok": False, "phase": "pick", "t": time.time() - t0}
            if not self.executor.make_upright_from_x(block):
                return {"ok": False, "phase": "make_upright",
                            "t": time.time() - t0}
            if not self.executor.put_upright(block, c_low, c_high):
                return {"ok": False, "phase": "put", "t": time.time() - t0}
        except Exception as e:
            return {"ok": False, "phase": "exception",
                        "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}

        # Upright block centroid sits between c_low and c_high (z-axis).
        expected = (self._cell_world(c_low) + self._cell_world(c_high)) / 2
        verified, err = self._verify_xy(block, expected, xy_tol)
        return {
            "ok": verified, "phase": "done" if verified else "verify",
            "t": time.time() - t0, "xy_err": err,
        }


def summarise(results: Dict[str, List[Dict]]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for shape, rs in results.items():
        if not rs:
            continue
        total = len(rs)
        passed = sum(1 for r in rs if r["ok"])
        pct = 100.0 * passed / total if total else 0.0
        print(f"  {shape:14s}: {passed:4d}/{total:4d}  ({pct:5.1f}%)")
        # Per-transit breakdown if any result has a "transit" key.
        transits = sorted({r.get("transit", "parts2stack") for r in rs})
        if len(transits) > 1:
            for tr in transits:
                tr_rs = [r for r in rs
                          if r.get("transit", "parts2stack") == tr]
                tr_pass = sum(1 for r in tr_rs if r["ok"])
                tr_total = len(tr_rs)
                print(f"      {tr}: {tr_pass:3d}/{tr_total:3d} "
                          f"({100.0*tr_pass/tr_total:5.1f}%)")
        # Per-level breakdown if any result has a "level" key.
        levels = sorted({r.get("level") for r in rs if "level" in r})
        if levels and len(levels) > 1:
            for L in levels:
                lvl_rs = [r for r in rs if r.get("level") == L]
                lvl_pass = sum(1 for r in lvl_rs if r["ok"])
                lvl_total = len(lvl_rs)
                print(f"      L{L}: {lvl_pass:3d}/{lvl_total:3d} "
                          f"({100.0*lvl_pass/lvl_total:5.1f}%)")
        # Bucket failures by phase
        phases: Dict[str, int] = {}
        for r in rs:
            if not r["ok"]:
                phases[r["phase"]] = phases.get(r["phase"], 0) + 1
        if phases:
            for p, n in sorted(phases.items()):
                print(f"      fail@{p}: {n}")
        # Print one example exception per shape (if any).
        sample_exc = next((r["exc"] for r in rs
                              if not r["ok"] and "exc" in r), None)
        if sample_exc is not None:
            print(f"      sample exception: {sample_exc}")
    grand_total = sum(len(rs) for rs in results.values())
    grand_pass = sum(1 for rs in results.values() for r in rs if r["ok"])
    if grand_total:
        print(f"\n  TOTAL: {grand_pass}/{grand_total} "
                  f"({100.0 * grand_pass / grand_total:.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stack-cells", nargs=3, type=int, default=[8, 8, 5],
                       metavar=("X", "Y", "Z"))
    p.add_argument("--parts-cells", nargs=2, type=int, default=[15, 15],
                       metavar=("X", "Y"))
    p.add_argument("--cube-size", type=float, default=None,
                       help="cube side length (m); default uses config default")
    p.add_argument("--n-cubes", type=int, default=16,
                       help="how many cubes to instantiate.  Default 16 covers "
                            "cube_0 (test) + 12 support cubes for the worst "
                            "case (long-x source at L4 = 3 cells × 4 levels). "
                            "Reduce for faster startup if you're only running "
                            "low-level shapes.")
    p.add_argument("--n-oblong", type=int, default=1,
                       help="how many oblong blocks to instantiate (need >=1 "
                            "for flat / upright tests; default 1)")
    p.add_argument("--n-long", type=int, default=1,
                       help="how many 3x1 long blocks to instantiate (need "
                            ">=1 for long_x / long_y / long_upright; default 1)")
    p.add_argument("--shapes", nargs="+",
                       default=["cube", "flat_x", "flat_y", "upright"],
                       choices=["cube", "flat_x", "flat_y", "upright",
                                  "long_x", "long_y", "long_upright"])
    p.add_argument("--levels", nargs="+", type=int, default=[0],
                       help="Stack levels to test put-at-Lk for each shape. "
                            "Default [0] (L0 only).  For upright the level is "
                            "interpreted as the LOW cell, so --levels 0 1 2 3 "
                            "places upright at (L0,L1), (L1,L2), (L2,L3), "
                            "(L3,L4).  Higher-level puts have no support and "
                            "the block falls to L0; xy verification still "
                            "fires (block lands directly below target xy).")
    p.add_argument("--transits", nargs="+",
                       default=["parts2stack"],
                       choices=["parts2stack", "stack2parts",
                                  "stack2stack", "parts2parts"],
                       help="Source→target transit pattern(s) to test. "
                            "parts2stack (default) is the existing matrix.  "
                            "stack2parts mirrors it (source on stack, target "
                            "on parts__0_0).  stack2stack / parts2parts use "
                            "small corner-pair samples to keep test count "
                            "manageable.")
    p.add_argument("--xy-tol", type=float, default=0.015,
                       help="xy tolerance (m) for success verification")
    p.add_argument("--output", type=str, default=None,
                       help="JSON file for per-test results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    stack_cells = tuple(args.stack_cells)
    parts_cells = tuple(args.parts_cells)
    cells_x, cells_y, _ = stack_cells

    tester = Tester(stack_cells, parts_cells, args.cube_size,
                          args.n_cubes, args.n_oblong, args.n_long,
                          args.verbose)

    results: Dict[str, List[Dict]] = {
        "cube": [], "flat_x": [], "flat_y": [], "upright": [],
        "long_x": [], "long_y": [], "long_upright": [],
    }

    cells_z = stack_cells[2]
    # Filter requested levels to those that fit the stack height.
    levels_flat = [L for L in args.levels if 0 <= L < cells_z]
    # Upright spans (L, L+1) — needs L < cells_z-1.
    levels_upright = [L for L in args.levels if 0 <= L < cells_z - 1]
    # Long upright spans (L, L+1, L+2) — needs L < cells_z-2.
    levels_long_upright = [L for L in args.levels if 0 <= L < cells_z - 2]

    do_p2s = "parts2stack" in args.transits
    do_s2p = "stack2parts" in args.transits
    do_s2s = "stack2stack" in args.transits
    do_p2p = "parts2parts" in args.transits

    # Sample of "representative" cells for the small-sample transits
    # (stack2stack / parts2parts).  Use the 4 corners + centre — these
    # are the historically reach-edge cells where the FRONT chain has
    # struggled, so they're the most informative sample.
    def _stack_sample():
        return [(0, 0), (cells_x - 1, 0),
                  (0, cells_y - 1), (cells_x - 1, cells_y - 1),
                  (cells_x // 2, cells_y // 2)]

    parts_cx, parts_cy = parts_cells
    def _parts_sample():
        return [(0, 0), (parts_cx - 1, 0),
                  (0, parts_cy - 1), (parts_cx - 1, parts_cy - 1),
                  (parts_cx // 2, parts_cy // 2)]

    try:
        # ====================================================================
        # parts → stack  (default; the canonical pick-from-supply test)
        # ====================================================================
        if "cube" in args.shapes and do_p2s:
            print(f"\n--- cube parts→stack: "
                      f"{cells_x * cells_y * len(levels_flat)} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y):
                        target = stack_cell(level, ix, iy)
                        r = tester.test_cube(target, args.xy_tol)
                        r["target"] = target
                        r["level"] = level
                        results["cube"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  cube {target:20s}  {tag:14s}  "
                                  f"t={r['t']:5.1f}s")

        if "flat_x" in args.shapes and do_p2s:
            n = (cells_x - 1) * cells_y * len(levels_flat)
            print(f"\n--- flat_x parts→stack: {n} tests (levels {levels_flat}) ---")
            for level in levels_flat:
                for iy in range(cells_y):
                    for ix in range(cells_x - 1):
                        c1 = stack_cell(level, ix, iy)
                        c2 = stack_cell(level, ix + 1, iy)
                        r = tester.test_flat_x(c1, c2, args.xy_tol)
                        r["target"] = f"{c1}+{c2}"
                        r["level"] = level
                        results["flat_x"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  flat_x {c1:20s}+ {c2:20s}  {tag:14s}  "
                                  f"t={r['t']:5.1f}s")

        if "flat_y" in args.shapes and do_p2s:
            n = cells_x * (cells_y - 1) * len(levels_flat)
            print(f"\n--- flat_y parts→stack: {n} tests (levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y - 1):
                        c1 = stack_cell(level, ix, iy)
                        c2 = stack_cell(level, ix, iy + 1)
                        r = tester.test_flat_y(c1, c2, args.xy_tol)
                        r["target"] = f"{c1}+{c2}"
                        r["level"] = level
                        results["flat_y"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  flat_y {c1:20s}+ {c2:20s}  {tag:14s}  "
                                  f"t={r['t']:5.1f}s")

        if "upright" in args.shapes and do_p2s:
            n = cells_x * cells_y * len(levels_upright)
            print(f"\n--- upright parts→stack: {n} tests "
                      f"(low levels {levels_upright}) ---")
            for L_low in levels_upright:
                for ix in range(cells_x):
                    for iy in range(cells_y):
                        c_low = stack_cell(L_low, ix, iy)
                        c_high = stack_cell(L_low + 1, ix, iy)
                        r = tester.test_upright(c_low, c_high, args.xy_tol)
                        r["target"] = f"{c_low}+{c_high}"
                        r["level"] = L_low
                        results["upright"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  upright {c_low:20s}+ {c_high:20s}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        if "long_x" in args.shapes and do_p2s:
            n = (cells_x - 2) * cells_y * len(levels_flat)
            print(f"\n--- long_x parts→stack: {n} tests (levels {levels_flat}) ---")
            for level in levels_flat:
                for iy in range(cells_y):
                    for ix in range(cells_x - 2):
                        c1 = stack_cell(level, ix, iy)
                        c2 = stack_cell(level, ix + 1, iy)
                        c3 = stack_cell(level, ix + 2, iy)
                        r = tester.test_long_x(c1, c2, c3, args.xy_tol)
                        r["target"] = f"{c1}+{c2}+{c3}"
                        r["level"] = level
                        results["long_x"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  long_x {c1:20s}..{c3:20s}  {tag:14s}  "
                                  f"t={r['t']:5.1f}s")

        if "long_y" in args.shapes and do_p2s:
            n = cells_x * (cells_y - 2) * len(levels_flat)
            print(f"\n--- long_y parts→stack: {n} tests (levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y - 2):
                        c1 = stack_cell(level, ix, iy)
                        c2 = stack_cell(level, ix, iy + 1)
                        c3 = stack_cell(level, ix, iy + 2)
                        r = tester.test_long_y(c1, c2, c3, args.xy_tol)
                        r["target"] = f"{c1}+{c2}+{c3}"
                        r["level"] = level
                        results["long_y"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  long_y {c1:20s}..{c3:20s}  {tag:14s}  "
                                  f"t={r['t']:5.1f}s")

        if "long_upright" in args.shapes and do_p2s:
            if not levels_long_upright:
                print(f"\nWARN: long_upright needs >= 3 levels above the "
                          f"requested L_low; none of {args.levels} fits. "
                          f"Skipping.")
            else:
                n = cells_x * cells_y * len(levels_long_upright)
                print(f"\n--- long_upright: {n} tests "
                          f"(low levels {levels_long_upright}) ---")
                for L_low in levels_long_upright:
                    for ix in range(cells_x):
                        for iy in range(cells_y):
                            c_low = stack_cell(L_low, ix, iy)
                            c_mid = stack_cell(L_low + 1, ix, iy)
                            c_high = stack_cell(L_low + 2, ix, iy)
                            r = tester.test_long_upright(c_low, c_mid,
                                                                  c_high,
                                                                  args.xy_tol)
                            r["target"] = f"{c_low}+{c_mid}+{c_high}"
                            r["level"] = L_low
                            results["long_upright"].append(r)
                            tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                            if "exc" in r:
                                tag += f"  ({r['exc']})"
                            print(f"  long_upright {c_low:20s}..{c_high:20s}  "
                                      f"{tag:14s}  t={r['t']:5.1f}s")

        # ====================================================================
        # stack → parts  (mirror; pick from stack, put on parts__0_0)
        # ====================================================================
        # Stack source over the same matrix as parts2stack uses for target;
        # target on parts is fixed to (0, 0) (cube) / (0,0)+(1,0) (flat-x) /
        # (0,0)+(0,1) (flat-y) / (0,0)+(1,0)+(2,0) (long-x) / etc.  Upright
        # variants are skipped — putting upright on parts isn't supported
        # (parts is single-level; upright spans 2+ levels).

        if "cube" in args.shapes and do_s2p:
            n = cells_x * cells_y * len(levels_flat)
            print(f"\n--- cube stack→parts: {n} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y):
                        source = stack_cell(level, ix, iy)
                        target = parts_cell(0, 0)
                        r = tester.test_cube(target, args.xy_tol,
                                                  source=source)
                        r["transit"] = "stack2parts"
                        r["source"] = source
                        r["target"] = target
                        r["level"] = level
                        results["cube"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  cube  s2p  src={source:20s}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        if "flat_x" in args.shapes and do_s2p:
            n = (cells_x - 1) * cells_y * len(levels_flat)
            print(f"\n--- flat_x stack→parts: {n} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for iy in range(cells_y):
                    for ix in range(cells_x - 1):
                        s1 = stack_cell(level, ix, iy)
                        s2 = stack_cell(level, ix + 1, iy)
                        t1 = parts_cell(0, 0)
                        t2 = parts_cell(1, 0)
                        r = tester.test_flat_x(t1, t2, args.xy_tol,
                                                    source1=s1, source2=s2)
                        r["transit"] = "stack2parts"
                        r["source"] = f"{s1}+{s2}"
                        r["target"] = f"{t1}+{t2}"
                        r["level"] = level
                        results["flat_x"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  flat_x  s2p  src={s1}+{s2}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        if "flat_y" in args.shapes and do_s2p:
            n = cells_x * (cells_y - 1) * len(levels_flat)
            print(f"\n--- flat_y stack→parts: {n} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y - 1):
                        s1 = stack_cell(level, ix, iy)
                        s2 = stack_cell(level, ix, iy + 1)
                        t1 = parts_cell(0, 0)
                        t2 = parts_cell(0, 1)
                        r = tester.test_flat_y(t1, t2, args.xy_tol,
                                                    source1=s1, source2=s2)
                        r["transit"] = "stack2parts"
                        r["source"] = f"{s1}+{s2}"
                        r["target"] = f"{t1}+{t2}"
                        r["level"] = level
                        results["flat_y"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  flat_y  s2p  src={s1}+{s2}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        if "long_x" in args.shapes and do_s2p:
            n = (cells_x - 2) * cells_y * len(levels_flat)
            print(f"\n--- long_x stack→parts: {n} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for iy in range(cells_y):
                    for ix in range(cells_x - 2):
                        s1 = stack_cell(level, ix, iy)
                        s2 = stack_cell(level, ix + 1, iy)
                        s3 = stack_cell(level, ix + 2, iy)
                        t1 = parts_cell(0, 0)
                        t2 = parts_cell(1, 0)
                        t3 = parts_cell(2, 0)
                        r = tester.test_long_x(t1, t2, t3, args.xy_tol,
                                                    source1=s1, source2=s2,
                                                    source3=s3)
                        r["transit"] = "stack2parts"
                        r["source"] = f"{s1}..{s3}"
                        r["target"] = f"{t1}..{t3}"
                        r["level"] = level
                        results["long_x"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  long_x  s2p  src={s1}..{s3}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        if "long_y" in args.shapes and do_s2p:
            n = cells_x * (cells_y - 2) * len(levels_flat)
            print(f"\n--- long_y stack→parts: {n} tests "
                      f"(levels {levels_flat}) ---")
            for level in levels_flat:
                for ix in range(cells_x):
                    for iy in range(cells_y - 2):
                        s1 = stack_cell(level, ix, iy)
                        s2 = stack_cell(level, ix, iy + 1)
                        s3 = stack_cell(level, ix, iy + 2)
                        t1 = parts_cell(0, 0)
                        t2 = parts_cell(0, 1)
                        t3 = parts_cell(0, 2)
                        r = tester.test_long_y(t1, t2, t3, args.xy_tol,
                                                    source1=s1, source2=s2,
                                                    source3=s3)
                        r["transit"] = "stack2parts"
                        r["source"] = f"{s1}..{s3}"
                        r["target"] = f"{t1}..{t3}"
                        r["level"] = level
                        results["long_y"].append(r)
                        tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                        if "exc" in r:
                            tag += f"  ({r['exc']})"
                        print(f"  long_y  s2p  src={s1}..{s3}  "
                                  f"{tag:14s}  t={r['t']:5.1f}s")

        # ====================================================================
        # stack → stack  (same-table rearrangement, corner-pair sample)
        # ====================================================================
        if "cube" in args.shapes and do_s2s:
            samples = _stack_sample()
            pairs = [(s, t) for s in samples for t in samples if s != t]
            print(f"\n--- cube stack→stack: {len(pairs) * len(levels_flat)} "
                      f"tests (levels {levels_flat}, "
                      f"{len(samples)} corner/centre sample) ---")
            for level in levels_flat:
                for (six, siy), (tix, tiy) in pairs:
                    source = stack_cell(level, six, siy)
                    target = stack_cell(level, tix, tiy)
                    r = tester.test_cube(target, args.xy_tol, source=source)
                    r["transit"] = "stack2stack"
                    r["source"] = source
                    r["target"] = target
                    r["level"] = level
                    results["cube"].append(r)
                    tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                    if "exc" in r:
                        tag += f"  ({r['exc']})"
                    print(f"  cube  s2s  {source:20s} → {target:20s}  "
                              f"{tag:14s}  t={r['t']:5.1f}s")

        # ====================================================================
        # parts → parts  (same-table rearrangement, corner-pair sample)
        # ====================================================================
        if "cube" in args.shapes and do_p2p:
            samples = _parts_sample()
            pairs = [(s, t) for s in samples for t in samples if s != t]
            print(f"\n--- cube parts→parts: {len(pairs)} tests "
                      f"({len(samples)} corner/centre sample) ---")
            for (six, siy), (tix, tiy) in pairs:
                source = parts_cell(six, siy)
                target = parts_cell(tix, tiy)
                r = tester.test_cube(target, args.xy_tol, source=source)
                r["transit"] = "parts2parts"
                r["source"] = source
                r["target"] = target
                results["cube"].append(r)
                tag = "OK" if r["ok"] else f"FAIL@{r['phase']}"
                if "exc" in r:
                    tag += f"  ({r['exc']})"
                print(f"  cube  p2p  {source:20s} → {target:20s}  "
                          f"{tag:14s}  t={r['t']:5.1f}s")

    except KeyboardInterrupt:
        print("\nInterrupted; printing partial summary.")
    finally:
        summarise(results)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.output}")
        tester.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
