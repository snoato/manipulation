"""End-to-end smoke for multi-level blocks (Kulshrestha CoRL-2023).

Sim-dependent (requires MuJoCo).  Builds the tabletop scene with 6
small wooden blocks scattered across the table, plans a 3-block tower
goal at the centre of the table, and renders snapshots of the plan
progression.

Run::

    python examples/multilevel_blocks_smoke.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np
from PIL import Image

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    default_initial_layout,
    make_multilevel_blocks_bridge,
    make_multilevel_blocks_builder,
    set_blocks_at_cells,
    surface_cell_id,
    tower_goal,
)
from tampanda.symbolic.workspace import Cell, Workspace


_VIZ_DIR = Path(__file__).resolve().parents[1] / "viz" / "multilevel_blocks_smoke"
_W, _H = 640, 480


def _print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _add_cameras(builder, target_world: List[float]) -> None:
    """Two diagonal angles around the table — both stay clear of the robot."""
    from tampanda.scenes.builder import _look_at_xyaxes

    front_pos = [target_world[0] + 0.30, target_world[1] - 0.35, target_world[2] + 0.20]
    side_pos  = [target_world[0] + 0.10, target_world[1] - 0.50, target_world[2] + 0.35]
    builder.add_camera(
        "into_scene",
        pos=front_pos,
        xyaxes=_look_at_xyaxes(front_pos, target_world),
        fovy=55.0,
    )
    builder.add_camera(
        "side_scene",
        pos=side_pos,
        xyaxes=_look_at_xyaxes(side_pos, target_world),
        fovy=55.0,
    )


def _render(env, label: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_name in ("into_scene", "side_scene"):
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            continue
        with mujoco.Renderer(env.model, height=_H, width=_W) as renderer:
            renderer.update_scene(env.data, camera=cam_id)
            img = renderer.render()
        out = out_dir / f"{label}__{cam_name}.png"
        Image.fromarray(img).save(out)
        print(f"  saved {out.relative_to(out_dir.parent.parent)}")


def _replay_action(env, workspace: Workspace, cfg: MultilevelBlocksConfig,
                   action: str, params, hide_xyz) -> None:
    """Symbolic replay: ``put`` resolves the cell pose; ``pick`` parks
    the held block off-screen until its next put."""
    if action == "pick":
        env.set_object_pose(params[0], hide_xyz)
        env.reset_velocities()
        env.forward()
        return
    block_name = params[0]
    cell_id = params[1]
    cell = workspace.cell(cell_id)
    if cell.region == "table_top":
        cx, cy, cz = workspace.region_of(cell).pose_for(cell)
    else:
        # Surface cell — resolve dynamically from parent block's pose.
        parent_idx = int(cell.region.split("_")[1])
        parent_pos, _ = env.get_object_pose(f"block_{parent_idx}")
        cx, cy = float(parent_pos[0]), float(parent_pos[1])
        cz = float(parent_pos[2]) + cfg.cube_size
    env.set_object_pose(block_name, np.array([cx, cy, cz]))
    env.reset_velocities()
    env.forward()


def main() -> None:
    cfg = MultilevelBlocksConfig(n_blocks=6, table_grid=(5, 5))
    with tempfile.TemporaryDirectory(prefix="multilevel_blocks_smoke_") as scratch:
        builder, workspace, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        cam_target = [
            cfg.table_pos[0],
            cfg.table_pos[1],
            cfg.table_pos[2] + 0.27 + 4 * cfg.cube_size,  # roughly tower-top z
        ]
        _add_cameras(builder, cam_target)
        env = builder.build_env(rate=10000.0)

    initial = default_initial_layout(cfg)
    set_blocks_at_cells(env, workspace, cfg, initial)

    _print_section("Workspace")
    print(f"  table {workspace['table_top']}")
    print(f"  + {cfg.n_blocks} 1×1 surface regions (one per block)")

    _print_section("Initial state")
    print(f"  6 blocks scattered along the front of the table")
    _render(env, "0_initial", _VIZ_DIR)

    _print_section("Goal — 3-block tower at table centre")
    bridge, objects = make_multilevel_blocks_bridge(env, workspace, cfg)
    base_cell = Cell("table_top", 2, 2)   # centre of 5×5 table
    # Build a 3-block tower: block_2 (bottom), block_1 (middle), block_0 (top).
    tower = [2, 1, 0]
    goals = tower_goal(tower, base_cell)
    print(f"  tower order (bottom→top): {[f'block_{i}' for i in tower]}")
    print(f"  base cell: {base_cell.id}")
    print(f"  {len(goals)} goal literals")

    _print_section("Plan")
    plan = bridge.plan(objects, goals=goals)
    if plan is None:
        print("  UNSAT — planner returned no plan")
        return
    print(f"  plan length = {len(plan)} actions")
    for i, (action, params) in enumerate(plan):
        print(f"    {i:>2}: ({action} {' '.join(params)})")

    _print_section("Replay snapshots")
    hide_xyz = np.array([cfg.hide_far_x, 0.0, cfg.cube_half_extent])
    snapshots = {
        len(plan) // 4,
        len(plan) // 2,
        max(0, len(plan) - 2),
    }
    for step_idx, (action, params) in enumerate(plan):
        _replay_action(env, workspace, cfg, action, params, hide_xyz)
        if step_idx in snapshots:
            _render(env, f"1_step_{step_idx:02d}", _VIZ_DIR)
    _render(env, "2_final", _VIZ_DIR)

    _print_section(f"OK — PNGs in {_VIZ_DIR}")


if __name__ == "__main__":
    main()
