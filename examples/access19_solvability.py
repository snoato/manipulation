"""Phase 0 — Access-19 18-blocker solvability proof.

Builds the canonical 18-blocker + 1-OoI access-19 scene, then runs a
hand-crafted row-by-row clearance plan through the EXISTING access-19
pick_fn / put_fn / generic PickPlaceExecutor execution path.  If this
succeeds end-to-end, the 18-blocker scenario is genuinely solvable
with the current executor — green-lights the data-gen pipeline.

Plan (38 actions):

  Phase 1 (36 actions) — empty the cubicle, dump on top deck:
    for iy_interior in 0..5:                     # front-to-back
        for col_ix in (1, 3, 5):
            pick blocker(col_ix, iy_interior)
            put  blocker → top(col_ix, 6 - iy_interior)   # back-to-front

  Phase 2 (2 actions) — move OoI to top deck:
    pick ooi from shelf_interior(3, 6)           # back of middle column
    put  ooi at  shelf_top(3, 0)                 # front-centre of top deck

Why this ordering:
  * Phase 1 dump back-to-front on top: each subsequent put's
    target_y < every previous put's y, so the Cartesian traverse at
    safe_z never crosses an occupied top cell (8 cm safe_z margin is
    exactly one cube height — no clearance above a stacked cube).
  * OoI goes to top(3, 0): col_3 is the only col where iy_top=0 is
    still empty after Phase 1, and an iy_top=0 put traverses
    forward-only — doesn't cross any iy_top>=1 occupied cell.

Phase 3 (returning blockers to lower shelf) attempted; fails on a
real executor limitation — the chain's pick_deck for off-centre
columns interpenetrates the OoI cube on the deck because the hand_c
body extends ~5 cm below the EE site, overlapping cube tops at
safe_z.  Needs either a wider top deck or a chain redesign with
waypoint-based traversal around occupied cells.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import mujoco

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
    make_tabletop_access_bridge, set_objects_at_cells,
)
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.workspace import Cell
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true",
                              help="Open MuJoCo passive viewer (requires "
                                       "mjpython on macOS) at real-time rate.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        print("=== Phase 0: 18-blocker solvability proof ===\n")
        builder, ws, cfg = make_access19_builder(scratch_dir=Path(td))
        # Viewer mode: drop rate to 240 Hz so each mj_step ~= real time.
        env = builder.build_env(rate=240.0 if args.viz else 10000.0)
        if args.viz:
            env.launch_viewer()
        apply_runtime_tweaks(env, cfg)

        object_ids = [f"blocker_{i}" for i in range(18)] + ["ooi"]

        # Canonical layout: 3 cube columns at ix in {1, 3, 5}, rows 0..5
        # filled with blockers, OoI at (3, 6) (back of middle column).
        layout = {}
        col_ix = [1, 3, 5]
        for ci, ix in enumerate(col_ix):
            for iy in range(6):
                layout[f"blocker_{ci * 6 + iy}"] = Cell("shelf_interior", ix, iy)
        layout["ooi"] = Cell("shelf_interior", 3, 6)
        set_objects_at_cells(env, ws, cfg, layout, object_ids)
        print(f"Placed 18 blockers + 1 OoI per canonical access-19 layout.")

        # Build executor (FRONT grasps only) + pick_fn (column-aligned).
        cube_half = float(env.get_object_half_size("ooi")[2])
        table_z = ws["shelf_interior"].level_z - cube_half
        executor = _build_executor(env, table_z=table_z,
                                          allowed_types=[GraspType.FRONT])

        # Staging home pose + LinearIK for the pick chain.
        shelf_home = _solve_access19_staging(env, ws, cfg)

        def _set_home():
            env.data.qpos[: len(shelf_home)] = shelf_home
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)

        def _half(name: str) -> np.ndarray:
            return np.asarray(env.get_object_half_size(name))

        lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)

        pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                                  cube_half_z=cube_half, lik=lik)
        put_fn  = make_access19_put_fn(env, executor, ws, cfg,
                                                cube_half_z=cube_half, lik=lik)

        actions: List[tuple] = []

        # Phase 1: clear cubicle → dump on top deck (col-aligned, B2F).
        for iy in range(6):
            for ci, ix in enumerate(col_ix):
                blocker = f"blocker_{ci * 6 + iy}"
                src = Cell("shelf_interior", ix, iy)
                dst = Cell("shelf_top", ix, 6 - iy)
                actions.append(("pick", blocker, src))
                actions.append(("put",  blocker, dst))

        # Phase 2: OoI to top deck at (3, 0) — front-centre, the only
        # free col_3 cell after Phase 1's col-aligned dumps.
        actions.append(("pick", "ooi", Cell("shelf_interior", 3, 6)))
        actions.append(("put",  "ooi", Cell("shelf_top", 3, 0)))
        print(f"Hand-coded plan: {len(actions)} actions "
                  f"({len(actions) // 2} pick+put pairs).\n")

        # Execute via chains.make_access19_pick_fn / make_access19_put_fn.
        # Reset arm to staging-home BEFORE each action so the chain's
        # withdraw step starts from a known clean pose (the chain
        # assumes the gripper is at staging home — see DESIGN note).
        # Both fn's have signature fn(obj_name, cell_id, world_pos) -> bool.
        n_ok = 0
        t_start = time.time()
        for i, (verb, name, cell) in enumerate(actions):
            _set_home()
            if args.viz and env.viewer is not None:
                env.viewer.sync()
                time.sleep(0.15)
            t0 = time.time()
            try:
                if verb == "pick":
                    pos, _ = env.get_object_pose(name)
                    ok = pick_fn(name, cell.id, np.asarray(pos))
                else:
                    target_pose = ws.pose_for(cell)
                    ok = put_fn(name, cell.id, np.asarray(target_pose))
                err = None
            except Exception as exc:
                ok = False
                err = f"{type(exc).__name__}: {exc}"
            dt = time.time() - t0
            flag = "OK  " if ok else "FAIL"
            print(f"  [{i+1:3d}/{len(actions)}] {flag} {verb:4s} {name:12s} "
                      f"{cell.id:28s} t={dt:5.2f}s  {err or ''}")
            if not ok:
                print(f"\nPhase 0 FAILED at action {i+1}.  "
                          f"Total elapsed: {time.time() - t_start:.1f}s.")
                return 1
            n_ok += 1

        elapsed = time.time() - t_start
        ooi_pos, _ = env.get_object_pose("ooi")
        goal_pose = ws.pose_for(Cell("shelf_top", 3, 0))
        drift = float(np.linalg.norm(np.asarray(ooi_pos)[:2]
                                              - np.asarray(goal_pose)[:2]))
        print(f"\nAll {n_ok} actions OK in {elapsed:.1f}s "
                  f"({elapsed/n_ok:.1f}s/action).")
        print(f"OoI final xy drift from goal cell: {drift*1000:.1f} mm "
                  f"({'PASS' if drift < 0.03 else 'WARN — large drift'}).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
