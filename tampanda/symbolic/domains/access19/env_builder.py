"""SceneBuilder factory for the ``access-19`` HAL scene.

Forked from ``tabletop_access`` so the deck-style cubicle + top
deck variant can evolve independently of the 3-tier ``access``
scene.

Exposes:
* :class:`Access19Config` — runtime tweaks + shelf placement params.
* :func:`make_access19_builder` — returns ``(builder, workspace, cfg)``.
* :func:`apply_runtime_tweaks` — shrinks the Franka ``hand_capsule``
  to clear the closed-top cubicle walls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from tampanda.scenes import ArmSceneBuilder, TABLE_TEMPLATE
from tampanda.scenes.assets.asset_set import Asset, AssetSet, Shelf
from tampanda.symbolic.workspace import GridRegion, Workspace


@dataclass(frozen=True)
class Access19Config:
    """Runtime parameters for the access-19 scene.

    Attributes:
        scene_variant: Always ``"access-19"`` (kept for compatibility
            with bridge factories that may dispatch on the field).
        n_objects:     Total movables (18 blockers + 1 OoI = 19).
        cell_size:     Grid pitch in metres.
        target_id:     Asset id of the distinguished target.
        hide_far_x:    Sentinel x for parked objects.
        shelf_pos:     World XYZ of the shelf body origin.
        hand_capsule_radius_override: Runtime shrink of the Franka
            ``hand_capsule`` geom (the 4 cm wrist guard).  At default
            radius the capsule clips the closed-top cubicle's side
            walls; 2 cm makes all 49 interior cells reachable.
    """
    scene_variant: str = "access-19"
    n_objects: int = 19
    cell_size: float = 0.060
    target_id: str = "ooi"
    hide_far_x: float = 100.0
    shelf_pos: Tuple[float, float, float] = (0.35, 0.40, 0.0)
    hand_capsule_radius_override: Optional[float] = 0.02


def make_access19_builder(
    scratch_dir: Path,
    # 4×4×8 cm cubes — z half-extent ≥ 0.040 keeps the FRONT grasp's
    # link7-clearance landing (floor + 5.5 cm) inside the block body.
    cube_half_extents: Tuple[float, float, float] = (0.020, 0.020, 0.040),
    n_red: int = 18,
    # 20 mm channel widens the column pitch to 6 cm so off-centre
    # columns at ix=1,5 land outside the Franka palm-+y dead zone.
    inter_row_gap: float = 0.020,
    # 24 cm interior cavity: cube top sits at floor+0.08; the arm's
    # wrist max-z while reaching is +9.5 cm above the cube top, but
    # the forearm/elbow can ride higher than the wrist, so the
    # original 12 cm headroom let the forearm clip the top wall on
    # FRONT grasps.  0.28 m gives 20 cm headroom (clearly clears
    # the elbow) but raises top_deck so far that pick_deck IK fails
    # at the +x edge columns.  0.24 m (16 cm headroom) is the
    # compromise — 4 cm more clearance than original, top_deck only
    # 4 cm higher so safe_z stays reachable.
    interior_height_z: float = 0.24,
    wall_thickness: float = 0.012,
    # 40 cm pedestal places the cubicle floor at world z ≈ 0.41 —
    # in the Franka palm-+y reach band (z ∈ [0.45, 0.55]).
    pedestal_height: float = 0.40,
    use_table: bool = False,
    # Validated by ``examples/measure_access19_arm_extent.py``:
    # (0.35, 0.40) gives 21/21 reachable cells using column-aligned
    # approach + row-by-row descent + gripper-invariance.  +/- 1 cm
    # shifts can reintroduce wrist singularities at edge cells.
    table_pos: Tuple[float, float, float] = (0.35, 0.40, 0.0),
    side_margin: float = 0.10,
    # ``shelf_top`` placement region: 7 × 7 cells covering the
    # shelf's footprint (mirrors the 7 × 7 interior).  All 49 cells
    # reachable by both pick_deck and put_deck at the validated
    # table_pos.
    top_grid_cells: int = 7,
) -> Tuple[ArmSceneBuilder, Workspace, Access19Config]:
    """Build the HAL ``access-19`` scene.

    Confined-cubicle shelf (open in -y only — robot-facing).  18 red
    cubes in 3 columns × 6 rows packed tight in depth + 1 blue OoI at
    the back of the middle column, behind the 6 reds.

    Two placement grids: shelf interior (where the cubes start) and
    shelf top (the OoI's goal placement).

    Args:
        scratch_dir: Directory where templates are materialised.
        cube_half_extents: Half-extents of every cube; must be cubic
            (lx == ly).
        n_red: Number of red blocker cubes.  Total = n_red + 1.
        inter_row_gap: Depth gap for MuJoCo contact stability.
        interior_height_z: Shelf interior cavity height.
        wall_thickness: Wall plate thickness.
        table_pos: World position of the table body.
    """
    n_objects = n_red + 1
    cube_d = 2 * cube_half_extents[0]
    cube_w = 2 * cube_half_extents[1]
    n_cols = 3
    n_red_rows = n_red // n_cols
    if n_red_rows * n_cols != n_red:
        raise ValueError(
            f"n_red ({n_red}) must be divisible by 3 for the 3-column layout"
        )
    if abs(cube_d - cube_w) > 1e-9:
        raise ValueError(
            f"cube_d ({cube_d}) must equal cube_w ({cube_w}) — only "
            f"square footprints are supported in v1."
        )

    cell_size = cube_d + inter_row_gap
    cells_x = 2 * n_cols + 1            # width — 3 cube cols + 4 channels
    cells_y = n_red_rows + 1            # depth — rows + back-row OoI

    region_extent_x = cells_x * cell_size
    region_extent_y = cells_y * cell_size

    front_margin = 0.02
    back_margin  = 0.04
    interior_x = region_extent_x + 2 * side_margin
    interior_y = region_extent_y + front_margin + back_margin

    base_offset_z = 0.27 if use_table else 0.0
    shelf_body_z = (table_pos[2] + base_offset_z + pedestal_height
                     + wall_thickness + interior_height_z / 2)
    cfg = Access19Config(
        n_objects=n_objects,
        cell_size=cell_size,
        shelf_pos=(table_pos[0], table_pos[1], shelf_body_z),
    )

    shelf = Shelf(
        asset_id="a19_shelf",
        interior_size=(interior_x, interior_y, interior_height_z),
        wall_thickness=wall_thickness,
        open_faces=("-y",),
        pedestal_height=pedestal_height,
    )

    red_assets: List[Asset] = []
    for i in range(n_red):
        red_assets.append(Asset(
            asset_id=f"blocker_{i}",
            half_extents=cube_half_extents,
            color=(0.78, 0.18, 0.18, 1.0),
        ))
    target_asset = Asset(
        asset_id="ooi",
        half_extents=cube_half_extents,
        color=(0.20, 0.40, 0.85, 1.0),
    )
    aset = AssetSet([shelf, *red_assets, target_asset])

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for a in aset:
        path = scratch_dir / f"{a.asset_id}.xml"
        path.write_text(a.render_template_xml())

    b = ArmSceneBuilder()
    if use_table:
        b.add_resource("table", str(TABLE_TEMPLATE))
    for a in aset:
        b.add_resource(a.asset_id, str(scratch_dir / f"{a.asset_id}.xml"))

    if use_table:
        b.add_object("table", name="simple_table", pos=list(table_pos))
        b.add_object(shelf.asset_id, name="shelf",
                       pos=[0.0, 0.0, cfg.shelf_pos[2] - table_pos[2]],
                       relative_to="simple_table")
    else:
        b.add_object(shelf.asset_id, name="shelf",
                       pos=list(cfg.shelf_pos))

    parked_z = cube_half_extents[2] + 0.005
    for a in red_assets + [target_asset]:
        b.add_object(a.asset_id, name=a.asset_id,
                       pos=[cfg.hide_far_x, 0.0, parked_z])

    sx_world = cfg.shelf_pos[0]
    sy_world = cfg.shelf_pos[1]
    region_origin = (
        sx_world - region_extent_x / 2,
        sy_world - region_extent_y / 2,
    )

    interior_floor_world_z = cfg.shelf_pos[2] - interior_height_z / 2
    interior_level_z = interior_floor_world_z + cube_half_extents[2]

    top_deck_top_z = cfg.shelf_pos[2] + interior_height_z / 2 + wall_thickness
    top_level_z = top_deck_top_z + cube_half_extents[2]

    top_extent = (top_grid_cells * cell_size, top_grid_cells * cell_size)
    top_origin = (sx_world - top_extent[0] / 2,
                   sy_world - top_extent[1] / 2)

    # Two diagonal-corner cells fail regardless of hand_capsule
    # shrink: (cells_x-1, cells_y-1) is past the Franka's reach
    # envelope; (0, 0) is an IK basin singularity for column-aligned
    # approach.  Confirmed by check_executability.py --mode full.
    cells_x_si = int(round(region_extent_x / cell_size))
    cells_y_si = int(round(region_extent_y / cell_size))
    si_excl: frozenset = frozenset()
    if cells_x_si >= 1 and cells_y_si >= 1:
        si_excl = frozenset({(0, 0), (cells_x_si - 1, cells_y_si - 1)})
    workspace = Workspace([
        GridRegion(
            name="shelf_interior",
            origin=region_origin,
            extent=(region_extent_x, region_extent_y),
            cell_size=cell_size,
            level_z=interior_level_z,
            access_modes=("front",),
            excluded_cells=si_excl,
        ),
        GridRegion(
            name="shelf_top",
            origin=top_origin,
            extent=top_extent,
            cell_size=cell_size,
            level_z=top_level_z,
            access_modes=("top_down", "front", "back", "left", "right"),
        ),
    ])
    return b, workspace, cfg


def apply_runtime_tweaks(env, cfg: Access19Config) -> None:
    """Shrink the Franka ``hand_capsule`` to ``cfg.hand_capsule_radius_override``.

    At the default 4 cm the capsule clips the closed-top cubicle's
    side walls on the corner cells; with the 2 cm shrink, all 49
    cells of the 7×7 interior become reachable.  Call AFTER
    ``builder.build_env(...)`` and BEFORE the first interaction.
    """
    import mujoco
    radius = cfg.hand_capsule_radius_override
    if radius is None:
        return
    for gid in range(env.model.ngeom):
        if env.model.geom(gid).name == "hand_capsule":
            env.model.geom_size[gid][0] = float(radius)
            return
    raise RuntimeError("hand_capsule geom not found on env.model")
