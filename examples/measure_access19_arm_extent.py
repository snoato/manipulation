"""Measure the Franka arm's swept volume when reaching every cube
cell of access19, then derive the minimum side-wall and top-wall
positions that stay clear of the arm.

Procedure (per the user's spec):

1. Build access19 with the **walls stripped** (only the back wall
   and floor remain — sides and top are open).
2. For each cube column (ix=1, 3, 5), define ONE approach pose just
   in front of the cubicle aligned with that column.
3. For each row iy=0..6 in the column, use linear-IK chain to:
     - move from staging to the column-aligned approach
     - descend to the cube's grasp pose
     - lift back up
   At every joint waypoint, query the world XYZ of every robot
   link and update the running bounding box.
4. Print the resulting bounding box (in shelf-body frame) and the
   wall-position recommendations:
     side walls at ``±(max(|min_x|, max_x) + safety)``
     top wall    at ``max_z + safety``

The output is then used to set ``side_margin`` and
``interior_height_z`` in the access19 builder.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.scenes.assets.asset_set import (
    Asset, AssetSet, Shelf, make_generic_boxes,
)
from tampanda.scenes import ArmSceneBuilder, TABLE_TEMPLATE
from tampanda.symbolic.workspace import Cell, GridRegion, Workspace


_ROBOT_LINKS = ("link0", "link1", "link2", "link3", "link4",
                "link5", "link6", "link7", "hand",
                "left_finger", "right_finger")

# FRONT quat (palm-+y, fingers along world ±X) and its 180°-rotation
# around hand-Z (the approach axis).  For a parallel-jaw gripper
# without finger-side preference, these are physically equivalent —
# but they lie in DIFFERENT IK basins, so trying both unlocks cells
# where the primary basin can't reach.
_FRONT_QUAT_PRIMARY = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])
_FRONT_QUATS = (_FRONT_QUAT_PRIMARY, _FRONT_QUAT_FLIPPED)


@dataclass
class ArmExtent:
    """Bounding box of robot-link world positions while INSIDE the
    cubicle's footprint.  Anchored to the cubicle's front-face y so we
    only count the arm-portion that actually has to fit between the
    cubicle walls.
    """
    min_xyz: np.ndarray = field(default_factory=lambda: np.full(3, +np.inf))
    max_xyz: np.ndarray = field(default_factory=lambda: np.full(3, -np.inf))

    def update(self, env, body_ids: List[int],
               front_face_y: float, ceiling_z: float) -> None:
        # Only sample links whose y > front face (they're inside the
        # cubicle footprint) AND whose z < ceiling_z + 0.20 (so we
        # don't track the elbow when it hangs above the cubicle top).
        for bid in body_ids:
            pos = env.data.xpos[bid]
            if pos[1] < front_face_y:
                continue
            if pos[2] > ceiling_z + 0.20:
                continue
            for d in range(3):
                if pos[d] < self.min_xyz[d]:
                    self.min_xyz[d] = float(pos[d])
                if pos[d] > self.max_xyz[d]:
                    self.max_xyz[d] = float(pos[d])

    def union(self, other: "ArmExtent") -> "ArmExtent":
        return ArmExtent(
            min_xyz=np.minimum(self.min_xyz, other.min_xyz),
            max_xyz=np.maximum(self.max_xyz, other.max_xyz),
        )


def _resolve_robot_body_ids(env) -> List[int]:
    out = []
    for n in _ROBOT_LINKS:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid >= 0:
            out.append(bid)
    return out


def _build_wall_stripped_access19(scratch: Path,
                                    table_x: float = 0.30,
                                    table_y: float = 0.40,
                                    pedestal: float = 0.40,
                                    cube_half: float = 0.040,
                                    cell_size: float = 0.06,
                                    n_red_rows: int = 6,
                                    interior_height: float = 0.30,
                                    wall_thickness: float = 0.012):
    """Assemble access19's geometry but with side + top walls REMOVED.
    Only the back wall and floor remain.  No table — shelf is mounted
    on the world floor via the shelf's pedestal."""
    n_cols = 3
    cells_x = 2 * n_cols + 1
    cells_y = n_red_rows + 1
    region_extent_x = cells_x * cell_size
    region_extent_y = cells_y * cell_size
    front_margin = 0.02
    back_margin = 0.04
    side_margin = 0.10  # generous — we're measuring, not enforcing
    interior_x = region_extent_x + 2 * side_margin
    interior_y = region_extent_y + front_margin + back_margin

    # Strip ALL walls except the back (+y) — measurement only.
    # The back wall is kept so the arm has a meaningful "deep limit"
    # to push toward.  Floor is stripped because the wrist dips below
    # the cube z when reaching deep cells (the actual cubicle floor
    # needs to be BELOW the wrist's minimum z).
    shelf = Shelf(
        asset_id="access19_strip",
        interior_size=(interior_x, interior_y, interior_height),
        wall_thickness=wall_thickness,
        open_faces=("-y", "+z", "+x", "-x", "-z"),
        pedestal_height=pedestal,
    )

    cube_assets: List[Asset] = []
    cube_half_extents = (cube_half / 2 * 2, cube_half / 2 * 2, cube_half)
    for i in range(18):
        cube_assets.append(Asset(
            asset_id=f"blocker_{i}",
            half_extents=(0.020, 0.020, cube_half),
            color=(0.78, 0.18, 0.18, 1.0),
        ))
    ooi_asset = Asset(asset_id="ooi", half_extents=(0.020, 0.020, cube_half),
                      color=(0.20, 0.40, 0.85, 1.0))
    aset = AssetSet([shelf, *cube_assets, ooi_asset])

    scratch.mkdir(parents=True, exist_ok=True)
    for a in aset:
        path = scratch / f"{a.asset_id}.xml"
        path.write_text(a.render_template_xml())

    b = ArmSceneBuilder()
    b.add_resource("table", str(TABLE_TEMPLATE))
    for a in aset:
        b.add_resource(a.asset_id, str(scratch / f"{a.asset_id}.xml"))

    # Mount shelf on its own pedestal directly on the world floor —
    # NO TABLE.  Shelf body z = pedestal + wall + interior_z/2.
    shelf_body_z = pedestal + wall_thickness + interior_height / 2
    b.add_object(shelf.asset_id, name="shelf",
                 pos=[table_x, table_y, shelf_body_z])

    parked = [100.0, 0.0, 0.05]
    for a in cube_assets + [ooi_asset]:
        b.add_object(a.asset_id, name=a.asset_id, pos=parked)

    env = b.build_env(rate=10000.0)
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3

    region_origin = (table_x - region_extent_x / 2,
                     table_y - region_extent_y / 2)
    interior_floor_world_z = shelf_body_z - interior_height / 2
    cell_level_z = interior_floor_world_z + cube_half

    workspace = Workspace([
        GridRegion(
            name="shelf_interior",
            origin=region_origin,
            extent=(region_extent_x, region_extent_y),
            cell_size=cell_size,
            level_z=cell_level_z,
            access_modes=("front",),
        ),
    ])
    return env, workspace, shelf_body_z, interior_floor_world_z, cell_size


def _solve_palm_y_ik(env, target_pos: np.ndarray, seed_q7: np.ndarray
                       ) -> Tuple[bool, np.ndarray]:
    target_quat = np.array([-0.5, 0.5, 0.5, 0.5])
    env.data.qpos[:7] = seed_q7
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(target_pos, target_quat)
    ok = env.ik.converge_ik(0.005)
    return ok, env.ik.configuration.q[:7].copy()


def measure(table_y: float = 0.40, table_x: float = 0.30) -> None:
    cube_half = 0.040
    cell_size = 0.06
    n_red_rows = 6

    with tempfile.TemporaryDirectory() as scratch_str:
        scratch = Path(scratch_str)
        env, ws, shelf_body_z, floor_z, cell_size = (
            _build_wall_stripped_access19(scratch,
                                            table_x=table_x,
                                            table_y=table_y,
                                            cube_half=cube_half,
                                            cell_size=cell_size,
                                            n_red_rows=n_red_rows)
        )
        print(f"\n>>> table_x={table_x:.3f}, table_y={table_y:.3f} <<<")
        body_ids = _resolve_robot_body_ids(env)
        region = ws["shelf_interior"]

        # Staging: roughly aligned with shelf centre x, IN FRONT of
        # front face (y=table_y - region_extent_y/2 - front_margin).
        front_face_y = region.origin[1] - 0.02
        staging_seed = np.array(
            [np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
        )
        # Solve a generic palm-+y staging pose — used only as IK seed.
        staging_target = np.array([region.origin[0] + region.extent[0] / 2,
                                    front_face_y - 0.06,
                                    region.level_z + 0.05])
        ok, staging_q = _solve_palm_y_ik(env, staging_target, staging_seed)
        if not ok:
            print(f"WARN: staging IK didn't fully converge")
        env.data.qpos[:7] = staging_q
        mujoco.mj_forward(env.model, env.data)

        col_ix = (1, 3, 5)
        lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)

        total_extent = ArmExtent()
        per_column: List[Tuple[int, ArmExtent, int]] = []   # (ix, extent, n_pass)

        for ix in col_ix:
            print(f"\n=== column ix={ix} ===")
            col_extent = ArmExtent()
            n_pass_col = 0

            # Approach pose: aligned with this column, OUTSIDE the
            # front face by 6 cm.
            col_x = region.origin[0] + (ix + 0.5) * cell_size
            approach_pos = np.array([col_x,
                                      front_face_y - 0.06,
                                      region.level_z + 0.02])

            # Reset to staging seed for each column.
            env.data.qpos[:7] = staging_q
            mujoco.mj_forward(env.model, env.data)

            # Phase 1: lerp to column-aligned approach.  Try both
            # gripper-invariant quats (180° flip around hand-Z) —
            # different IK basins can succeed where one fails.
            path = None
            chosen_quat = _FRONT_QUAT_PRIMARY
            for q in _FRONT_QUATS:
                path = lik.plan_joint_lerp(approach_pos, q,
                                            dt=0.005, n_substeps=20)
                if path is not None:
                    chosen_quat = q
                    break
            if path is None:
                print(f"  approach to column {ix} failed (both quats)")
                continue
            for q in path:
                env.data.qpos[:7] = q
                mujoco.mj_forward(env.model, env.data)
                col_extent.update(env, body_ids,
                                   front_face_y, region.level_z + 0.20)
            env.data.qpos[:7] = path[-1]
            mujoco.mj_forward(env.model, env.data)
            print(f"  approach OK, col_extent so far x∈[{col_extent.min_xyz[0]:.3f},"
                  f"{col_extent.max_xyz[0]:.3f}] z∈[{col_extent.min_xyz[2]:.3f},"
                  f"{col_extent.max_xyz[2]:.3f}]")

            # Phase 2: row-by-row deeper descent.  Each row starts
            # from the PREVIOUS row's grasp pose (or the column
            # approach if it's the first row), so each lerp covers
            # only one cell-size (6 cm) of in-cubicle motion.
            prev_q = env.data.qpos[:7].copy()
            for iy in range(n_red_rows + 1):
                cell_x, cell_y, cell_z = region.pose_for(Cell("shelf_interior",
                                                                ix, iy))
                grasp_pos = np.array([cell_x, cell_y, cell_z])
                env.data.qpos[:7] = prev_q
                mujoco.mj_forward(env.model, env.data)
                # Try both invariant quats — flip-around-hand-Z is
                # the same physical grasp but different IK basin.
                p = None
                for q in _FRONT_QUATS:
                    p = lik.plan_joint_lerp(grasp_pos, q,
                                              dt=0.005, n_substeps=8)
                    if p is not None:
                        break
                if p is None:
                    # Diagnose
                    env.ik.update_configuration(env.data.qpos)
                    env.ik.set_target_position(grasp_pos,
                                                 np.array([-0.5, 0.5, 0.5, 0.5]))
                    ok = env.ik.converge_ik(0.005)
                    if not ok:
                        print(f"  iy={iy} descent FAIL: IK at grasp pose")
                    else:
                        gq = env.ik.configuration.q[:7].copy()
                        env.data.qpos[:7] = gq
                        mujoco.mj_forward(env.model, env.data)
                        cf = env.check_collisions()
                        env.data.qpos[:7] = prev_q
                        mujoco.mj_forward(env.model, env.data)
                        if not cf:
                            for ci in range(env.data.ncon):
                                con = env.data.contact[ci]
                                if con.dist >= 0.001: continue
                                b1 = env.model.geom_bodyid[con.geom1]
                                b2 = env.model.geom_bodyid[con.geom2]
                                if b1 in env._collision_exception_ids or b2 in env._collision_exception_ids: continue
                                r1 = b1 in env._collision_body_ids
                                r2 = b2 in env._collision_body_ids
                                if r1 == r2: continue
                                n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                                n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                                print(f"  iy={iy} descent FAIL: goal collision {n1}↔{n2}")
                                break
                            else:
                                print(f"  iy={iy} descent FAIL: lerp collision")
                        else:
                            print(f"  iy={iy} descent FAIL: lerp collision (goal CF)")
                    continue
                for q in p:
                    env.data.qpos[:7] = q
                    mujoco.mj_forward(env.model, env.data)
                    col_extent.update(env, body_ids,
                                   front_face_y, region.level_z + 0.20)
                # Tiny lift to clear the cube — try both quats again.
                lift_pos = grasp_pos + np.array([0.0, 0.0, 0.04])
                p_lift = None
                for q in _FRONT_QUATS:
                    p_lift = lik.plan_joint_lerp(lift_pos, q,
                                                   dt=0.005, n_substeps=6)
                    if p_lift is not None:
                        break
                if p_lift is not None:
                    for q in p_lift:
                        env.data.qpos[:7] = q
                        mujoco.mj_forward(env.model, env.data)
                        col_extent.update(env, body_ids,
                                   front_face_y, region.level_z + 0.20)
                    n_pass_col += 1
                    print(f"  iy={iy} OK")
                    prev_q = p_lift[-1]  # next row's start
                else:
                    print(f"  iy={iy} lift FAIL")
                    prev_q = p[-1]

            print(f"  column {ix} extent: x∈[{col_extent.min_xyz[0]:.3f},"
                  f"{col_extent.max_xyz[0]:.3f}], z∈[{col_extent.min_xyz[2]:.3f},"
                  f"{col_extent.max_xyz[2]:.3f}], passed {n_pass_col}/{n_red_rows + 1}")
            per_column.append((ix, col_extent, n_pass_col))
            total_extent = total_extent.union(col_extent)

        # ---- Report ----
        print("\n" + "=" * 70)
        print(" arm bounding box across all (column, row) reaches:")
        print(f"   x ∈ [{total_extent.min_xyz[0]:+.3f}, {total_extent.max_xyz[0]:+.3f}]")
        print(f"   y ∈ [{total_extent.min_xyz[1]:+.3f}, {total_extent.max_xyz[1]:+.3f}]")
        print(f"   z ∈ [{total_extent.min_xyz[2]:+.3f}, {total_extent.max_xyz[2]:+.3f}]")
        print(f" floor z = {floor_z:.3f}, shelf body z = {shelf_body_z:.3f}")
        # Required wall positions (in shelf-body frame x/z):
        shelf_x = 0.30  # default — shelf x in world frame
        shelf_z = shelf_body_z
        margin = 0.02
        side_x_max_needed = max(0.0, total_extent.max_xyz[0] - shelf_x) + margin
        side_x_min_needed = max(0.0, shelf_x - total_extent.min_xyz[0]) + margin
        top_z_needed = max(0.0, total_extent.max_xyz[2] - shelf_z) + margin
        print(" required clearance from shelf body centre:")
        print(f"   side walls: at least ±{max(side_x_max_needed, side_x_min_needed):.3f} m")
        print(f"   top wall:   at least  +{top_z_needed:.3f} m above body z")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="sweep table_x and table_y to find a config "
                              "that gets every cell reachable")
    args = parser.parse_args()
    if args.sweep:
        # Tiny shifts to nudge problem cells out of singularities.
        for ty in (0.38, 0.40, 0.41, 0.42, 0.43, 0.45):
            for tx in (0.25, 0.28, 0.30, 0.32, 0.35):
                measure(table_y=ty, table_x=tx)
    else:
        measure()
