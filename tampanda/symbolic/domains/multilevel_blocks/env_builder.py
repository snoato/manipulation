"""SceneBuilder factory for the multi-level wooden-blocks domain.

The Kulshrestha CoRL-2023 puzzle is a tabletop block-arrangement task:
N small blocks start scattered in a flat "parts" area and must be
rearranged into target multi-level structures in a separate "stack"
area.  This implementation uses **two physical tables**, the parts
table behind the robot and the stack table directly in front.

Workspace structure:

* ``parts`` — a 2D :class:`GridRegion` covering the parts table; blocks
  rest at z = ``parts_table_top``.  No stacking on this table.
* ``stack_L0`` ... ``stack_L4`` — five 2D :class:`GridRegion`s, one per
  vertical level on the stack table.  Cell ``stack_Lk__ix_iy`` denotes
  position ``(ix, iy)`` at level ``k`` (0-indexed from the table top).

A 1×1 cube occupies one cell.  A 2×1 oblong block occupies two cells
— horizontally adjacent at the same level (``flat-x`` / ``flat-y``
orientation) or vertically adjacent across two levels at the same
``(ix, iy)`` (``upright`` orientation).

Cells are identified globally as ``"<region>__<ix>_<iy>"`` per the
workspace convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tampanda.scenes import ArmSceneBuilder
from tampanda.scenes.assets.asset_set import Asset, AssetSet
from tampanda.symbolic.workspace import GridRegion, Workspace


# Painted-wooden palette — green, red, yellow, blue, natural tan.
_KULSHRESTHA_PALETTE: Tuple[Tuple[float, float, float, float], ...] = (
    (0.20, 0.55, 0.30, 1.0),   # green
    (0.85, 0.20, 0.20, 1.0),   # red
    (0.95, 0.80, 0.25, 1.0),   # yellow
    (0.25, 0.45, 0.80, 1.0),   # blue
    (0.78, 0.65, 0.45, 1.0),   # natural / tan
)


# Workspace-position constants — locked in after the dual-table Phase 0
# probe.  Both grids 100% reachable; the parts table sits BEHIND the
# robot so the arm rotates q[0] by ±π/2 to access either workspace.
# See /tmp/sweep_stack_table.py and /tmp/sweep_dual_table.py history
# in the feat/multigrid-domains branch.
_STACK_TABLE_POS = (0.00, 0.50, 0.00)
_PARTS_TABLE_POS = (0.00, -0.45, 0.00)
_STACK_GRID_CELLS = (10, 10, 5)
_PARTS_GRID_CELLS = (15, 15)
_CUBE_HALF_EXTENT = 0.015                # 30 mm cubes


@dataclass(frozen=True)
class MultilevelBlocksConfig:
    """Tunables for :func:`make_multilevel_blocks_builder`.

    Two tables: ``stack`` in front of the robot for vertical structure
    building, ``parts`` behind the robot for the flat block layout.
    Geometry locked in by the Phase 0 dual-table probe — 100% reach on
    both grids.

    Attributes:
        n_cubes:               Number of 1×1 cube blocks.
        n_oblong:              Number of 2×1 oblong blocks.
        n_long:                Number of 3×1 long blocks.  Same shape
                               family as oblong but 3 cells along the
                               long axis.  Symbolic support rule: only
                               needs ≥2 of 3 cells underneath to be
                               supported (enforced in PDDL, not here).
        cube_half_extent:      Half edge length of a unit cell, in
                               metres.  Block size derives from this:
                               cubes are ``2*cube_half_extent`` on each
                               side, oblong blocks are ``4*cube_half_extent``
                               long, ``2*cube_half_extent`` wide and tall.
                               Long (3×1) blocks are ``6*cube_half_extent``
                               long, same width and height as oblong.
        stack_table_pos:       World XYZ of the stack table body origin.
        stack_grid_cells:      ``(cells_x, cells_y, cells_z)`` of the
                               3D stack region.  Modelled as
                               ``cells_z`` stacked 2D grids.
        parts_table_pos:       World XYZ of the parts table body origin.
        parts_grid_cells:      ``(cells_x, cells_y)`` of the 2D parts
                               region.
        table_margin:          Margin around each grid for the table top
                               geometry; the table is sized to
                               grid_extent + 2 * table_margin.
        hide_far_x:            Sentinel x for parked blocks.
    """
    n_cubes: int = 8
    n_oblong: int = 0
    n_long: int = 0
    cube_half_extent: float = _CUBE_HALF_EXTENT
    stack_table_pos: Tuple[float, float, float] = _STACK_TABLE_POS
    stack_grid_cells: Tuple[int, int, int] = _STACK_GRID_CELLS
    parts_table_pos: Tuple[float, float, float] = _PARTS_TABLE_POS
    parts_grid_cells: Tuple[int, int] = _PARTS_GRID_CELLS
    table_margin: float = 0.025
    hide_far_x: float = 100.0

    def __post_init__(self) -> None:
        if self.n_cubes < 0 or self.n_oblong < 0 or self.n_long < 0:
            raise ValueError("block counts must be non-negative")
        if self.n_cubes + self.n_oblong + self.n_long <= 0:
            raise ValueError("must have at least one block")
        if self.cube_half_extent <= 0:
            raise ValueError("cube_half_extent must be > 0")
        if any(c < 1 for c in self.stack_grid_cells):
            raise ValueError("stack_grid_cells dimensions must be >= 1")
        if any(c < 1 for c in self.parts_grid_cells):
            raise ValueError("parts_grid_cells dimensions must be >= 1")

    @property
    def n_blocks(self) -> int:
        return self.n_cubes + self.n_oblong + self.n_long

    @property
    def cube_size(self) -> float:
        return 2 * self.cube_half_extent

    @property
    def stack_levels(self) -> int:
        return self.stack_grid_cells[2]


def cube_block_name(idx: int) -> str:
    return f"cube_{idx}"


def oblong_block_name(idx: int) -> str:
    return f"oblong_{idx}"


def long_block_name(idx: int) -> str:
    """Name for a 3×1 long block."""
    return f"long_{idx}"


def stack_region_name(level: int) -> str:
    """Region name for the 2D grid at stack level ``level``."""
    return f"stack_L{level}"


def _emit_table_xml(half_x: float, half_y: float,
                      scratch: Path, name: str,
                      body_z: float = 0.0) -> Path:
    """Write a custom-sized table MJCF and return its path.

    Top face at z=0.27 in body-local coordinates, matching the standard
    ``table.xml`` template so existing tools (cameras, grasp planner,
    etc.) see a consistent surface height.  Legs are positioned at the
    outer corners of the given half-extent and procedurally sized so
    they reach the world floor (z=0) regardless of where the body
    origin is.  ``body_z`` is the world z at which the table body will
    be placed; legs span body-local z=0.24 down to z=-body_z.
    """
    leg_half_h = (0.24 + body_z) / 2.0
    leg_cz = 0.12 - body_z / 2.0
    xml = f"""<body pos="0 0 0">
  <joint name="_freejoint" type="free"/>
  <!-- table top: {half_x*200:.1f} x {half_y*200:.1f} cm, 4cm thick.
       Top face at z=0.27 in body-local coords. -->
  <geom name="_surface" type="box" size="{half_x:.4f} {half_y:.4f} 0.02"
        pos="0 0 0.25" rgba="0.55 0.55 0.55 1"
        contype="1" conaffinity="1"/>
  <!-- legs (2.4 cm sq, length grows with body_z so the feet touch the
       world floor at z=0) at outer corners -->
  <geom type="box" size="0.012 0.012 {leg_half_h:.4f}"
        pos=" {half_x - 0.03:.4f}  {half_y - 0.03:.4f} {leg_cz:.4f}"
        rgba="0.3 0.3 0.3 1"/>
  <geom type="box" size="0.012 0.012 {leg_half_h:.4f}"
        pos="{-half_x + 0.03:.4f}  {half_y - 0.03:.4f} {leg_cz:.4f}"
        rgba="0.3 0.3 0.3 1"/>
  <geom type="box" size="0.012 0.012 {leg_half_h:.4f}"
        pos=" {half_x - 0.03:.4f} {-half_y + 0.03:.4f} {leg_cz:.4f}"
        rgba="0.3 0.3 0.3 1"/>
  <geom type="box" size="0.012 0.012 {leg_half_h:.4f}"
        pos="{-half_x + 0.03:.4f} {-half_y + 0.03:.4f} {leg_cz:.4f}"
        rgba="0.3 0.3 0.3 1"/>
</body>
"""
    path = scratch / f"_table_{name}.xml"
    path.write_text(xml, encoding="utf-8")
    return path


# Local table-top z constant — matches the body-local z of the surface
# geom emitted by ``_emit_table_xml`` above.  Used to compute the
# world-frame level_z for every grid region.
_TABLE_TOP_LOCAL_Z = 0.27


def make_multilevel_blocks_builder(
    scratch_dir: Path,
    config: Optional[MultilevelBlocksConfig] = None,
    block_colors: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Tuple[ArmSceneBuilder, Workspace, MultilevelBlocksConfig]:
    """Build the multi-level blocks scene.

    Returns ``(builder, workspace, config)``.  The workspace contains
    ``cfg.stack_levels + 1`` regions: one 2D ``parts`` region plus one
    2D ``stack_L<k>`` region per vertical level on the stack table.
    """
    cfg = config or MultilevelBlocksConfig()
    palette = list(block_colors) if block_colors else list(_KULSHRESTHA_PALETTE)

    cube_size = cfg.cube_size
    cube_half = cfg.cube_half_extent

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # ---- Block assets ---------------------------------------------------
    block_assets: List[Asset] = []
    for i in range(cfg.n_cubes):
        block_assets.append(Asset(
            asset_id=cube_block_name(i),
            half_extents=(cube_half,) * 3,
            color=palette[i % len(palette)],
        ))
    # Oblong = 2*cube_size long, cube_size wide, cube_size tall (default
    # orientation is long-axis-along-x; the bridge sets the runtime
    # quat for flat-y / upright placements).
    for i in range(cfg.n_oblong):
        block_assets.append(Asset(
            asset_id=oblong_block_name(i),
            half_extents=(cube_size, cube_half, cube_half),
            color=palette[(cfg.n_cubes + i) % len(palette)],
        ))
    # Long = 3*cube_size long, cube_size wide, cube_size tall.  Same
    # default orientation as oblong (long-axis-along-x); rotated at
    # runtime via the bridge for flat-y / upright placements.
    for i in range(cfg.n_long):
        block_assets.append(Asset(
            asset_id=long_block_name(i),
            half_extents=(1.5 * cube_size, cube_half, cube_half),
            color=palette[(cfg.n_cubes + cfg.n_oblong + i) % len(palette)],
        ))
    aset = AssetSet(block_assets)
    for a in aset:
        path = scratch_dir / f"{a.asset_id}.xml"
        path.write_text(a.render_template_xml(), encoding="utf-8")

    # ---- Table XMLs (custom-sized per grid) -----------------------------
    stack_table_half = (
        cfg.stack_grid_cells[0] * cube_size / 2 + cfg.table_margin,
        cfg.stack_grid_cells[1] * cube_size / 2 + cfg.table_margin,
    )
    parts_table_half = (
        cfg.parts_grid_cells[0] * cube_size / 2 + cfg.table_margin,
        cfg.parts_grid_cells[1] * cube_size / 2 + cfg.table_margin,
    )
    stack_table_path = _emit_table_xml(*stack_table_half, scratch_dir, "stack",
                                            body_z=cfg.stack_table_pos[2])
    parts_table_path = _emit_table_xml(*parts_table_half, scratch_dir, "parts",
                                            body_z=cfg.parts_table_pos[2])

    b = ArmSceneBuilder()
    b.add_resource("stack_table", str(stack_table_path))
    b.add_resource("parts_table", str(parts_table_path))
    for a in aset:
        b.add_resource(a.asset_id, str(scratch_dir / f"{a.asset_id}.xml"))

    b.add_object("stack_table", name="stack_table",
                  pos=list(cfg.stack_table_pos))
    b.add_object("parts_table", name="parts_table",
                  pos=list(cfg.parts_table_pos))

    parked_z = cube_half + 0.005
    for a in block_assets:
        b.add_object(a.asset_id, name=a.asset_id,
                     pos=[cfg.hide_far_x, 0.0, parked_z])

    # ---- Workspace ------------------------------------------------------
    sx, sy, sz = cfg.stack_table_pos
    stack_cells_x, stack_cells_y, stack_cells_z = cfg.stack_grid_cells
    stack_extent = (stack_cells_x * cube_size, stack_cells_y * cube_size)
    stack_origin = (sx - stack_extent[0] / 2, sy - stack_extent[1] / 2)
    stack_table_top_z = sz + _TABLE_TOP_LOCAL_Z

    stack_regions: List[GridRegion] = []
    for level in range(stack_cells_z):
        # Cell ``stack_L<level>`` cube centres sit at
        # ``stack_table_top_z + (level + 0.5) * cube_size``.
        level_z = stack_table_top_z + (level + 0.5) * cube_size
        stack_regions.append(GridRegion(
            name=stack_region_name(level),
            origin=stack_origin,
            extent=stack_extent,
            cell_size=cube_size,
            level_z=level_z,
            access_modes=("top_down",),
        ))

    px, py, pz = cfg.parts_table_pos
    parts_cells_x, parts_cells_y = cfg.parts_grid_cells
    parts_extent = (parts_cells_x * cube_size, parts_cells_y * cube_size)
    parts_origin = (px - parts_extent[0] / 2, py - parts_extent[1] / 2)
    parts_table_top_z = pz + _TABLE_TOP_LOCAL_Z
    parts_level_z = parts_table_top_z + 0.5 * cube_size
    parts_region = GridRegion(
        name="parts",
        origin=parts_origin,
        extent=parts_extent,
        cell_size=cube_size,
        level_z=parts_level_z,
        access_modes=("top_down",),
    )

    workspace = Workspace([parts_region, *stack_regions])
    return b, workspace, cfg
