"""Multi-block structure reliability test for multilevel_blocks.

Builds non-trivial pre-existing structures by TELEPORTING blocks into
position (no MP — represents the planner's mid-plan state), settles
physics, verifies stability, then runs a small set of test actions
ON or AROUND the structure to check that the executor handles
neighbour-block scenarios cleanly.

Default structure ("2x1_bridges") exercises the 2-of-3 support rule
that makes 3×1 long blocks distinctive:

    L0  layout (top-down view, x→right, y→up):

           x=0  1  2  3  4  5  6  7
        y=7  .  .  .  .  .  .  .  .
           6  .  .  .  .  .  .  .  .
           5  .  .  .  .  .  .  .  .
           4  .  .  .  .  .  .  .  .
           3  .  [ob2 ]  [ob3 ]  .  .
           2  .  .  .  .  .  .  .  .
           1  .  [ob0 ]  [ob1 ]  .  .
           0  .  .  .  .  .  .  .  .

        ob0 = oblong_0 flat-x at L0__(1,1)+(2,1)
        ob1 = oblong_1 flat-x at L0__(3,1)+(4,1)
        ob2 = oblong_2 flat-x at L0__(1,3)+(2,3)
        ob3 = oblong_3 flat-x at L0__(3,3)+(4,3)

    L1 bridges (flat-y, span the gap at y=2 with 2-of-3 support):

        long_0 at L1__(1,1)+(1,2)+(1,3)  bridges ob0 and ob2
        long_1 at L1__(3,1)+(3,2)+(3,3)  bridges ob1 and ob3

        Each long block has L0(_,1) and L0(_,3) supported, L0(_,2)
        empty.  Standard support rule would reject this (needs all 3
        below); the 2-of-3 relaxation enables the bridge.

Usage::

    mjpython examples/structure_test.py --out-dir struct_out/
    python examples/structure_test.py --no-actions   # build + render only
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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


_QUAT_FLAT_X = np.array([1.0, 0.0, 0.0, 0.0])
_QUAT_FLAT_Y = np.array([0.7071068, 0.0, 0.0, 0.7071068])
# Rotation of -π/2 around world +y maps the block's body +x (default
# long axis) onto world +z, giving the upright orientation.
_QUAT_UPRIGHT = np.array([0.7071068, 0.0, -0.7071068, 0.0])


def stack_cell_id(level: int, ix: int, iy: int) -> str:
    return f"stack_L{level}__{ix}_{iy}"


def parts_cell_id(ix: int, iy: int) -> str:
    return f"parts__{ix}_{iy}"


# ---------------------------------------------------------------------------
# Structure templates
# ---------------------------------------------------------------------------

def structure_2x1_bridges() -> List[Dict]:
    """4 × 2×1 flat-x at L0 + 2 × 3×1 flat-y bridges at L1.

    The user's spec — corners at (1,1), (3,1), (1,3), (3,3); bridges span
    y-gaps and rely on the 2-of-3 support rule.
    """
    return [
        # L0 base — 4 × 2×1 oblong flat-x
        {"block": oblong_block_name(0), "type": "flat-x",
         "cells": [stack_cell_id(0, 1, 1), stack_cell_id(0, 2, 1)]},
        {"block": oblong_block_name(1), "type": "flat-x",
         "cells": [stack_cell_id(0, 3, 1), stack_cell_id(0, 4, 1)]},
        {"block": oblong_block_name(2), "type": "flat-x",
         "cells": [stack_cell_id(0, 1, 3), stack_cell_id(0, 2, 3)]},
        {"block": oblong_block_name(3), "type": "flat-x",
         "cells": [stack_cell_id(0, 3, 3), stack_cell_id(0, 4, 3)]},
        # L1 bridges — 2 × 3×1 long flat-y (2-of-3 support)
        {"block": long_block_name(0), "type": "flat-y",
         "cells": [stack_cell_id(1, 1, 1), stack_cell_id(1, 1, 2),
                    stack_cell_id(1, 1, 3)]},
        {"block": long_block_name(1), "type": "flat-y",
         "cells": [stack_cell_id(1, 3, 1), stack_cell_id(1, 3, 2),
                    stack_cell_id(1, 3, 3)]},
    ]


def structure_2x1_upright_bridges() -> List[Dict]:
    """4 × 2×1 UPRIGHT at L0+L1 corners + 2 × 3×1 flat-y bridges at L2.

    Same topology as ``2x1_bridges`` but the base oblongs stand on end,
    elevating the bridges by one cube.  The 2-of-3 support rule still
    applies (middle L1 cell below each bridge is empty since the
    uprights only occupy (1,1), (3,1), (1,3), (3,3) — NOT (1,2) or
    (3,2)).

    Visually taller; the bridges sit at L2.z ≈ table_top + 70 mm.
    """
    return [
        # L0+L1 corners — 4 × 2×1 upright
        {"block": oblong_block_name(0), "type": "upright",
         "cells": [stack_cell_id(0, 1, 1), stack_cell_id(1, 1, 1)]},
        {"block": oblong_block_name(1), "type": "upright",
         "cells": [stack_cell_id(0, 3, 1), stack_cell_id(1, 3, 1)]},
        {"block": oblong_block_name(2), "type": "upright",
         "cells": [stack_cell_id(0, 1, 3), stack_cell_id(1, 1, 3)]},
        {"block": oblong_block_name(3), "type": "upright",
         "cells": [stack_cell_id(0, 3, 3), stack_cell_id(1, 3, 3)]},
        # L2 bridges — 2 × 3×1 long flat-y (sit on the upright tops at L1)
        {"block": long_block_name(0), "type": "flat-y",
         "cells": [stack_cell_id(2, 1, 1), stack_cell_id(2, 1, 2),
                    stack_cell_id(2, 1, 3)]},
        {"block": long_block_name(1), "type": "flat-y",
         "cells": [stack_cell_id(2, 3, 1), stack_cell_id(2, 3, 2),
                    stack_cell_id(2, 3, 3)]},
    ]


STRUCTURES = {
    "2x1_bridges": structure_2x1_bridges,
    "2x1_upright_bridges": structure_2x1_upright_bridges,
}


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class StructureTester:
    def __init__(self, stack_cells: Tuple[int, int, int],
                  parts_cells: Tuple[int, int],
                  n_cubes: int,
                  n_oblong: int,
                  n_long: int,
                  verbose: bool):
        cfg = MultilevelBlocksConfig(
            stack_grid_cells=stack_cells,
            parts_grid_cells=parts_cells,
            n_cubes=n_cubes, n_oblong=n_oblong, n_long=n_long,
        )
        self._scratch = tempfile.TemporaryDirectory(prefix="structure_test_")
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

        # Renderer for snapshots.
        self.env.model.vis.global_.offwidth = max(
            1280, self.env.model.vis.global_.offwidth)
        self.env.model.vis.global_.offheight = max(
            720, self.env.model.vis.global_.offheight)
        self._renderer = mujoco.Renderer(self.env.model,
                                                height=720, width=1280)
        self._cam = mujoco.MjvCamera()
        self._cam.azimuth = 35
        self._cam.elevation = -28
        self._cam.distance = 1.6
        sp = cfg.stack_table_pos
        self._cam.lookat[:] = [sp[0], sp[1], sp[2] + 0.30]

        self._park_all()
        self._reset_arm()

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass
        try:
            self._scratch.cleanup()
        except Exception:
            pass

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

    def _cell_world(self, cell_id: str) -> np.ndarray:
        return np.asarray(self.ws.pose_for(Cell.parse(cell_id)),
                              dtype=float)

    def _block_pos(self, name: str) -> np.ndarray:
        pos, _ = self.env.get_object_pose(name)
        return np.asarray(pos)

    def _snapshot(self, output_path: Path) -> None:
        self._renderer.update_scene(self.env.data, camera=self._cam)
        img = self._renderer.render()
        try:
            import cv2  # type: ignore
            cv2.imwrite(str(output_path), img[:, :, ::-1])
        except Exception:
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.imsave(str(output_path), img)
            except Exception as e:
                print(f"  WARN: could not save snapshot {output_path}: {e}")

    # ---- structure build ---------------------------------------------------

    def _expected_centre(self, placement: Dict) -> np.ndarray:
        """World centroid where the block SHOULD land after teleport."""
        cells = placement["cells"]
        if placement["type"] == "cube":
            return self._cell_world(cells[0])
        # flat-x / flat-y / upright: centroid is midpoint of first and last.
        return (self._cell_world(cells[0])
                    + self._cell_world(cells[-1])) / 2

    def build_structure(self, placements: List[Dict]) -> None:
        """Teleport each block into the specified pose."""
        self._park_all()
        self._reset_arm()
        for p in placements:
            block = p["block"]
            ttype = p["type"]
            centre = self._expected_centre(p)
            if ttype == "cube":
                self.env.set_object_pose(block, centre)
            elif ttype == "flat-x":
                self.env.set_object_pose(block, centre, _QUAT_FLAT_X)
            elif ttype == "flat-y":
                self.env.set_object_pose(block, centre, _QUAT_FLAT_Y)
            elif ttype == "upright":
                self.env.set_object_pose(block, centre, _QUAT_UPRIGHT)
            else:
                raise ValueError(f"unknown placement type: {ttype}")
        self.env.reset_velocities()
        self.env.forward()

    def settle(self, n_steps: int = 400) -> None:
        for _ in range(n_steps):
            self.env.step()

    def verify_structure(self, placements: List[Dict],
                              drift_tol: float = 0.005) -> Dict:
        """Compare current block positions to expected centres."""
        results = []
        max_drift = 0.0
        for p in placements:
            block = p["block"]
            expected = self._expected_centre(p)
            actual = self._block_pos(block)
            drift = float(np.linalg.norm(actual - expected))
            ok = drift <= drift_tol
            max_drift = max(max_drift, drift)
            results.append({
                "block": block, "drift": drift, "ok": ok,
                "expected": expected.tolist(),
                "actual": actual.tolist(),
            })
        all_ok = all(r["ok"] for r in results)
        return {"ok": all_ok, "max_drift": max_drift,
                  "per_block": results}

    # ---- test actions ------------------------------------------------------

    def action_pick_long_y(self, block: str,
                                c1: str, c2: str, c3: str) -> Dict:
        """Pick a flat-y long block from the structure."""
        t0 = time.time()
        try:
            ok = self.executor.pick_long_y(block, c1, c2, c3)
        except Exception as e:
            return {"ok": False, "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}
        return {"ok": ok, "t": time.time() - t0}

    def action_put_long_y(self, block: str,
                                c1: str, c2: str, c3: str) -> Dict:
        t0 = time.time()
        try:
            ok = self.executor.put_long_y(block, c1, c2, c3)
        except Exception as e:
            return {"ok": False, "t": time.time() - t0,
                        "exc": f"{type(e).__name__}: {e}"}
        return {"ok": ok, "t": time.time() - t0}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_structure(t: StructureTester, struct_name: str,
                       out_dir: Path, run_actions: bool,
                       settle_steps: int, drift_tol: float) -> Dict:
    """Build, settle, verify, optionally run test actions."""
    placements = STRUCTURES[struct_name]()
    print(f"\n=== {struct_name} ({len(placements)} blocks) ===")

    t.build_structure(placements)
    t._snapshot(out_dir / f"{struct_name}_built.png")
    print(f"  built; snapshot → {struct_name}_built.png")

    t.settle(settle_steps)
    verif = t.verify_structure(placements, drift_tol)
    t._snapshot(out_dir / f"{struct_name}_settled.png")
    print(f"  settled {settle_steps} steps; max_drift="
              f"{verif['max_drift']*1000:.1f}mm  "
              f"{'OK' if verif['ok'] else 'DRIFT'}  "
              f"snapshot → {struct_name}_settled.png")
    if not verif["ok"]:
        for r in verif["per_block"]:
            if not r["ok"]:
                print(f"    DRIFT {r['block']}: "
                          f"{r['drift']*1000:.1f}mm")

    actions_result: List[Dict] = []
    if run_actions:
        # Test 1: pick the long_0 bridge.  Cells are extracted from the
        # template (NOT hardcoded) so the same action works for both
        # the flat-base 2x1_bridges (L1 bridge) and the upright-base
        # 2x1_upright_bridges (L2 bridge).
        long_0_placement = next(
            (p for p in placements
                 if p["block"] == long_block_name(0)), None)
        if long_0_placement is None or len(long_0_placement["cells"]) != 3:
            print("\n  no long_0 with 3-cell footprint in this structure; "
                      "skipping pick/put actions")
            return {
                "structure": struct_name,
                "placements": placements,
                "settle_verify": verif,
                "actions": [],
            }
        c1, c2, c3 = long_0_placement["cells"]
        print(f"\n  action: pick long_0 from bridge at "
                  f"{c1}..{c3}")
        r = t.action_pick_long_y(long_block_name(0), c1, c2, c3)
        actions_result.append({"name": "pick_long_y bridge", **r})
        if "exc" in r:
            print(f"    FAIL@exc: {r['exc']}  t={r['t']:.1f}s")
        else:
            print(f"    {'OK' if r['ok'] else 'FAIL'}  t={r['t']:.1f}s")

        # Test 2: put it back on the parts table.
        if r["ok"]:
            p1 = parts_cell_id(0, 0)
            p2 = parts_cell_id(0, 1)
            p3 = parts_cell_id(0, 2)
            print(f"  action: put long_0 onto parts {p1}..{p3}")
            r2 = t.action_put_long_y(long_block_name(0), p1, p2, p3)
            actions_result.append({"name": "put_long_y parts", **r2})
            if "exc" in r2:
                print(f"    FAIL@exc: {r2['exc']}  t={r2['t']:.1f}s")
            else:
                print(f"    {'OK' if r2['ok'] else 'FAIL'}  t={r2['t']:.1f}s")
            t._snapshot(out_dir / f"{struct_name}_after_pick_put.png")
            print(f"    snapshot → {struct_name}_after_pick_put.png")

        # Verify the rest of the structure didn't move.
        # The placement-list verify includes long_0 which we just moved;
        # drop it from the check.
        remaining = [p for p in placements
                          if p["block"] != long_block_name(0)]
        verif2 = t.verify_structure(remaining, drift_tol)
        print(f"  post-action structure drift: "
                  f"{verif2['max_drift']*1000:.1f}mm  "
                  f"{'OK' if verif2['ok'] else 'DISTURBED'}")
        if not verif2["ok"]:
            for r in verif2["per_block"]:
                if not r["ok"]:
                    print(f"    DRIFT {r['block']}: "
                              f"{r['drift']*1000:.1f}mm")

    return {
        "structure": struct_name,
        "placements": placements,
        "settle_verify": verif,
        "actions": actions_result,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stack-cells", nargs=3, type=int, default=[8, 8, 5])
    p.add_argument("--parts-cells", nargs=2, type=int, default=[15, 15])
    p.add_argument("--n-cubes", type=int, default=4)
    p.add_argument("--n-oblong", type=int, default=4)
    p.add_argument("--n-long", type=int, default=2)
    p.add_argument("--structure", default="2x1_bridges",
                       choices=list(STRUCTURES.keys()))
    p.add_argument("--no-actions", action="store_true",
                       help="Build + render + verify stability only; skip "
                            "the pick/put test actions.")
    p.add_argument("--settle-steps", type=int, default=400)
    p.add_argument("--drift-tol", type=float, default=0.005,
                       help="Allowed xyz drift per block during physics "
                            "settle.  Default 5 mm.")
    p.add_argument("--out-dir", type=str, default="structure_test_out")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tester = StructureTester(
        tuple(args.stack_cells), tuple(args.parts_cells),
        args.n_cubes, args.n_oblong, args.n_long, args.verbose,
    )

    try:
        result = run_structure(
            tester, args.structure, out_dir,
            run_actions=not args.no_actions,
            settle_steps=args.settle_steps,
            drift_tol=args.drift_tol,
        )
    finally:
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2,
                                                          default=str))
            print(f"\nResults JSON written to {args.output}")
        tester.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
