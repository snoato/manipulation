"""SceneBuilder factory for the confined-shelf (Wang ICAPS-2022) domain.

Builds a wide closed cubicle (one open face) on top of the existing
``simple_table`` body, populated with up to 16 identical cylinders in
K colour groups.  The defining feature of Wang's setup is **lazy
in-place rearrangement**: cylinders never leave the shelf — the goal
is to permute them inside the cubicle into a target arrangement (e.g.,
colour-by-column).

Geometry (sized to the Franka's palm-+y reach, not Wang's robot):

* **9 columns wide × 4 rows deep** = 36 grid cells; the two
  back-right deep cells (8, 2) and (8, 3) are excluded as
  kinematically unreachable, leaving **34 reachable cells**.  The
  4-deep grid gives front-occlusion blocking chains up to 4 long
  (i.e. up to 3 relocations per column) for non-monotone difficulty.
* Default 5 cylinders; cylinders spawn on the even-x sublattice so
  every lateral neighbour cell is an empty gripper-buffer (a
  cylinder in a directly-adjacent cell blocks the FRONT grasp — see
  the empirical clearance probe).
* Drink-bottle cylinders (3 cm radius × 20 cm tall) — graspable from
  the side via FRONT only.
* Closed top, single open face on -y (the robot side).

The factory:

1. Creates a :class:`Shelf` with explicit margins inside the walls so
   the gripper has clearance for FRONT grasps at edge cells.
2. Materialises the shelf template into a caller-provided scratch
   directory and registers it with the SceneBuilder.
3. Registers the existing thin-cylinder template and instantiates N
   cylinders parked off-screen.
4. Constructs the :class:`Workspace` with one :class:`GridRegion`
   ``shelf_interior`` whose footprint matches the n_grid_x × n_grid_y
   placement grid (smaller than the shelf interior — the margins are
   gripper-clearance space).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.scenes import (
    ArmSceneBuilder,
    TABLE_TEMPLATE,
    WORKSPACE_X_OFFSET,
    WORKSPACE_Y_OFFSET,
)
from tampanda.scenes.assets.asset_set import AssetSet, Shelf


def _render_bottle_cylinder_xml(radius: float, half_height: float) -> str:
    """Render a parametric cylinder template — 0.5 L bottle proxy.

    The default ``CYLINDER_THIN_TEMPLATE`` hard-codes radius 1.25 cm
    and half-height 8 cm, which doesn't match Wang's bottle-sized
    objects.  We materialise a fresh template per configuration so the
    visual mesh + collision geom share the dataclass dimensions.
    """
    # Diaginertia for a uniform cylinder (mass m, radius r, height h):
    # Ixx = Iyy = m/12 (3 r^2 + h^2);  Izz = m r^2 / 2.
    # We bump up Ixx/Iyy slightly to stop tipping under contact, like
    # the legacy thin-cylinder template does.
    h = 2 * half_height
    m = 0.20
    ixx = m / 12 * (3 * radius * radius + h * h) * 1.5
    izz = m * radius * radius / 2 * 1.5
    return (
        f'<body pos="0 0 0">\n'
        f'  <joint name="_freejoint" type="free"/>\n'
        f'  <inertial mass="{m:.6g}" pos="0 0 0" '
        f'diaginertia="{ixx:.6g} {ixx:.6g} {izz:.6g}"/>\n'
        f'  <geom name="_geom" type="cylinder" '
        f'size="{radius:.6g} {half_height:.6g}" rgba="0.85 0.20 0.20 1" '
        f'solimp="0.99 0.99 0.01" solref="0.02 1" condim="4"/>\n'
        f'</body>\n'
    )
from tampanda.symbolic.workspace import Cell, GridRegion, Workspace


# Palm-+y staging-home arm config, IK-solved just outside the open -y
# face at (table_x, 0.16, level_z+0.02).  Keeps the EE OUTSIDE the shelf
# cavity so it never overlaps a centre-column cylinder.  Validated to
# reach all 34 cells of the 9×4 grid (check_executability --mode full)
# and to give 31/31 FAST/FULL feasibility agreement.  Single source of
# truth for the pipeline (reachability, parallel, generate_data).
STAGING_HOME_QPOS: np.ndarray = np.array(
    [1.1952, -0.6873, -1.0123, -2.3270, 1.1278, 2.0441, -1.1123, 0.04, 0.04]
)


def excluded_cells(n_grid_x: int, n_grid_y: int) -> frozenset:
    """Cells the kinematic stack can't reliably reach.  The far-back-right
    column (ix = n_grid_x-1 = 8) sits just past the Franka's palm-+y reach
    for the deep rows: iy=2 on the 9×3 grid and both iy=2,3 on 9×4.
    Confirmed by ``check_executability.py confined_shelf --mode full`` and
    ``examples/cs_probe_4th_row.py``.  Single source of truth — used by the
    builder and the goal generator.
    """
    if n_grid_x == 9 and n_grid_y == 3:
        return frozenset({(8, 2)})
    if n_grid_x == 9 and n_grid_y == 4:
        # Only the far-back-right corner is genuinely unreachable (pick
        # AND put confirmed by the per-cell sweeps + check_executability).
        return frozenset({(8, 2), (8, 3)})
    return frozenset()


# Default colour palette for cylinder groups (matches the viz preview).
_DEFAULT_COLORS: Tuple[Tuple[float, float, float, float], ...] = (
    (0.85, 0.20, 0.20, 1.0),  # red
    (0.30, 0.70, 0.30, 1.0),  # green
    (0.95, 0.78, 0.20, 1.0),  # yellow
    (0.20, 0.45, 0.85, 1.0),  # blue
    (0.65, 0.30, 0.75, 1.0),  # purple
    (0.95, 0.55, 0.20, 1.0),  # orange
)


@dataclass(frozen=True)
class ConfinedShelfConfig:
    """Tunables for :func:`make_confined_shelf_builder`.

    Attributes:
        n_grid_x:        Cylinder grid width (cells along world x).
                         Default 6 — Wang's published max.
        n_grid_y:        Cylinder grid depth (cells along world y).
                         Default 3 — Wang's published max.
        n_cylinders:     Number of cylinders.  Must be ≤ n_grid_x ×
                         n_grid_y (else there's no place to put them);
                         to leave room for rearrangement, prefer
                         n_cylinders ≤ n_grid_x*n_grid_y - 2.
        n_color_groups:  How many colour groups partition the cylinders.
        cylinder_radius: Cylinder XY half-width.
        cylinder_half_height: Cylinder Z half-height (so total = 2× this).
        cell_size:       Grid resolution.  Must be ≥ 2*cylinder_radius
                         plus a small slop for the gripper finger
                         thickness.
        interior_height_z: Shelf cavity height.  Must clear the cylinder
                         tops (2*cylinder_half_height) plus the gripper
                         hand body.
        wall_thickness:  Shelf wall plate thickness.
        front_margin:    Y-clearance between the open -y face and the
                         front-most grid row, for gripper standoff.
        back_margin:     Y-clearance between the back wall and the
                         back-most grid row.  Sized for the 12-cm hand
                         capsule when reaching the back row top-down.
        side_margin:     X-clearance between side walls and end columns.
        open_face:       Robot-facing open face.  Default ``"-y"``.
        table_pos:       Table body world position.
        hide_far_x:      Sentinel x for parked cylinders.
    """
    # ---- Grid (cylinder layout) ------------------------------------------
    # 9 columns × 4 rows = 36 cells (back-right deep cells (8,2),(8,3)
    # excluded → 34 reachable; confirmed by check_executability and
    # examples/cs_probe_4th_row.py).  Cylinders spawn on the even-x
    # sublattice so every lateral neighbour is an empty gripper-buffer
    # — a directly-adjacent (distance-1) cylinder blocks the FRONT
    # grasp; distance-2 is clear (empirical clearance probe).  The
    # 4-deep grid gives front-occlusion blocking chains up to 4 long,
    # the lever for non-monotone OOD difficulty.  Wang's ICAPS-2022
    # setup uses a finer lattice; we use the densest pitch the Franka
    # can grasp at after the ``hand_capsule`` shrink.
    n_grid_x: int = 9
    n_grid_y: int = 4
    n_cylinders: int = 5
    n_color_groups: int = 4
    # ---- Cylinder dimensions: 0.5-litre soft-drink bottle proxy ----------
    # Radius 3 cm → 6 cm diameter; half-height 10 cm → 20 cm tall.
    # Matches Wang's cylinders-as-bottles abstraction.
    cylinder_radius: float = 0.030
    cylinder_half_height: float = 0.10
    # Cell size = 7 cm — 1 cm of cell-slack vs the 6 cm cylinder
    # diameter, giving the gripper fingers a sub-cm gap to thread into
    # the cell.  Tighter than the previous 10 cm default; works
    # because we shrink the Franka ``hand_capsule`` (radius 4 cm → 2
    # cm) at build time so the wrist no longer overlaps a held
    # cylinder during palm-+y descent.
    cell_size: float = 0.07
    # ---- Shelf cavity ---------------------------------------------------
    # Interior height: 20 cm bottles + 4 cm post-grasp lift + 10 cm
    # headroom for the wrist (link7 dips below the hand on FRONT
    # grasps).  At 0.30 the lift clipped the ceiling on a few cells;
    # 0.34 gives clean clearance per the feasibility map.
    interior_height_z: float = 0.34
    wall_thickness: float = 0.012
    front_margin: float = 0.04
    back_margin: float = 0.12
    # Side margin sized at 8 cm — covers the hand-capsule (radius 4 cm)
    # AND the link7 capsule (radius 5.5 cm) plus a contact margin.
    # 5 cm was empirically too tight for end columns on Wang's wide
    # shelf (the wrist clipped the side wall during the descent
    # joint-lerp on cell (5, 1)).  See feasibility map.
    side_margin: float = 0.08
    open_face: str = "-y"
    # Pedestal lifts cylinder centres into the IK convergence band
    # (z ≈ 0.51 with 20 cm bottles), well inside the green zone for
    # palm-+y reach.
    pedestal_height: float = 0.13
    # Shelf positioned so the 9-cell grid fits the Franka's palm-+y
    # reach zone after the ``hand_capsule`` shrink: cells span world
    # x ∈ [0.17, 0.73].  Below 0.17 the q1 dead zone bites; above
    # 0.73 reach starts breaking down.  ``table_pos[0] = 0.45``
    # centres the 9-cell span on this corridor.
    table_pos: Tuple[float, float, float] = (0.45,
                                              WORKSPACE_Y_OFFSET, 0.0)
    # Runtime shrink of the Franka ``hand_capsule`` geom (the 4 cm
    # wrist guard) — set to 0.02 m (2 cm) to keep the wrist out of
    # the cylinder's body during palm-+y descent.  Without this, the
    # wrist's 4 cm capsule overlaps the 6 cm cylinder during the
    # final grasp pose and pushes it forward by 4-18 cm under contact
    # forces.  Set to ``None`` to disable.
    hand_capsule_radius_override: Optional[float] = 0.02
    hide_far_x: float = 100.0

    def __post_init__(self) -> None:
        if self.n_cylinders <= 0:
            raise ValueError("n_cylinders must be > 0")
        if self.n_color_groups <= 0:
            raise ValueError("n_color_groups must be > 0")
        if self.n_grid_x <= 0 or self.n_grid_y <= 0:
            raise ValueError("n_grid_x and n_grid_y must be > 0")
        if self.n_cylinders > self.n_grid_x * self.n_grid_y:
            raise ValueError(
                f"n_cylinders ({self.n_cylinders}) > grid capacity "
                f"({self.n_grid_x * self.n_grid_y})"
            )
        if self.cell_size < 2 * self.cylinder_radius:
            raise ValueError(
                f"cell_size ({self.cell_size}) must be >= cylinder diameter "
                f"({2 * self.cylinder_radius})"
            )

    @property
    def interior_size(self) -> Tuple[float, float, float]:
        """Derived shelf interior dimensions.  Grid + margins."""
        return (
            self.n_grid_x * self.cell_size + 2 * self.side_margin,
            self.n_grid_y * self.cell_size + self.front_margin + self.back_margin,
            self.interior_height_z,
        )


def make_confined_shelf_builder(
    scratch_dir: Path,
    config: Optional[ConfinedShelfConfig] = None,
    colors: Optional[List[Tuple[float, float, float, float]]] = None,
    cylinder_color_groups: Optional[List[int]] = None,
) -> Tuple[ArmSceneBuilder, Workspace, ConfinedShelfConfig]:
    """Build a Wang-style confined-shelf scene.

    Returns ``(builder, workspace, config)``.  The builder is ready to
    call ``build_env(rate=...)`` on; the workspace has one
    :class:`GridRegion` ``shelf_interior`` of size
    ``n_grid_x × n_grid_y``.
    """
    cfg = config or ConfinedShelfConfig()
    palette = list(colors) if colors else list(_DEFAULT_COLORS)

    if cylinder_color_groups is None:
        cylinder_color_groups = [i % cfg.n_color_groups for i in range(cfg.n_cylinders)]
    if len(cylinder_color_groups) != cfg.n_cylinders:
        raise ValueError(
            f"cylinder_color_groups length {len(cylinder_color_groups)} != "
            f"n_cylinders {cfg.n_cylinders}"
        )
    for g in cylinder_color_groups:
        if g < 0 or g >= len(palette):
            raise ValueError(
                f"colour-group index {g} out of range for palette of size {len(palette)}"
            )

    sx, sy, sz = cfg.interior_size

    # ---- Shelf asset (static body) ----------------------------------------
    shelf = Shelf(
        asset_id="confined_shelf",
        interior_size=(sx, sy, sz),
        wall_thickness=cfg.wall_thickness,
        open_faces=(cfg.open_face,),
        pedestal_height=cfg.pedestal_height,
    )
    aset = AssetSet([shelf])

    # ---- Builder, base templates -----------------------------------------
    b = ArmSceneBuilder()
    b.add_resource("table", str(TABLE_TEMPLATE))

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shelf_path = scratch_dir / f"{shelf.asset_id}.xml"
    shelf_path.write_text(shelf.render_template_xml())
    b.add_resource(shelf.asset_id, str(shelf_path))

    # Parametric bottle-cylinder template — sized from the config so the
    # visual cylinder matches the cell layout.
    bottle_path = scratch_dir / "bottle_cylinder.xml"
    bottle_path.write_text(_render_bottle_cylinder_xml(
        cfg.cylinder_radius, cfg.cylinder_half_height,
    ))
    b.add_resource("cyl_thin", str(bottle_path))

    # ---- Static fixtures: table + shelf ----------------------------------
    table_top_z = 0.27
    b.add_object("table", name="simple_table", pos=list(cfg.table_pos))

    # The shelf is positioned so its **interior centre along y** sits at
    # ``table_pos.y + (back_margin - front_margin) / 2`` — that places
    # the cylinder grid centred on ``table_pos.y`` while keeping the
    # back wall ``back_margin`` away from the back-most row.
    shelf_y_offset = (cfg.back_margin - cfg.front_margin) / 2
    shelf_z_local = (table_top_z + cfg.pedestal_height
                     + cfg.wall_thickness + sz / 2)
    b.add_object(shelf.asset_id, name="shelf",
                 pos=[0.0, shelf_y_offset, shelf_z_local],
                 relative_to="simple_table")

    # ---- Cylinder pool, parked off-screen --------------------------------
    parked_z = cfg.cylinder_half_height
    for i in range(cfg.n_cylinders):
        rgba = list(palette[cylinder_color_groups[i]])
        b.add_object(
            "cyl_thin",
            name=f"cyl_{i}",
            pos=[cfg.hide_far_x, 0.0, parked_z],
            rgba=rgba,
        )

    # ---- Workspace: one GridRegion sized exactly to the cylinder grid ----
    region_extent_x = cfg.n_grid_x * cfg.cell_size
    region_extent_y = cfg.n_grid_y * cfg.cell_size
    region_origin = (
        cfg.table_pos[0] - region_extent_x / 2,
        cfg.table_pos[1] - region_extent_y / 2,
    )

    shelf_centre_world_z = (cfg.table_pos[2] + table_top_z
                             + cfg.pedestal_height
                             + cfg.wall_thickness + sz / 2)
    interior_floor_world_z = shelf_centre_world_z - sz / 2
    cyl_centre_world_z = interior_floor_world_z + cfg.cylinder_half_height

    excl = excluded_cells(cfg.n_grid_x, cfg.n_grid_y)

    workspace = Workspace([
        GridRegion(
            name="shelf_interior",
            origin=region_origin,
            extent=(region_extent_x, region_extent_y),
            cell_size=cfg.cell_size,
            level_z=cyl_centre_world_z,
            access_modes=("front",),
            excluded_cells=excl,
        )
    ])

    return b, workspace, cfg


# ---------------------------------------------------------------------------
# Layout / goal helpers — Wang's lazy in-place rearrangement
# ---------------------------------------------------------------------------

def default_initial_layout(
    cfg: ConfinedShelfConfig,
    cylinder_color_groups: Optional[List[int]] = None,
) -> Dict[str, Cell]:
    """Stage cylinders on the even-x sublattice so every neighbour cell
    stays empty as a gripper-buffer.

    With ``cell_size`` close to the cylinder diameter, cylinders in
    adjacent cells would touch and the gripper jaws would have no
    room to close around one without pushing its neighbour.  Spawning
    only at even ``ix`` (and walking front-to-back through ``iy``)
    keeps every cylinder surrounded by empty buffer cells.

    Raises ``ValueError`` if the buffer-respecting capacity of the
    grid is smaller than ``n_cylinders``.
    """
    even_x = list(range(0, cfg.n_grid_x, 2))
    cells = [(ix, iy) for iy in range(cfg.n_grid_y) for ix in even_x]
    if cfg.n_cylinders > len(cells):
        raise ValueError(
            f"n_cylinders ({cfg.n_cylinders}) > buffer-respecting "
            f"capacity ({len(cells)}) of the {cfg.n_grid_x}×"
            f"{cfg.n_grid_y} grid"
        )
    return {
        f"cyl_{i}": Cell("shelf_interior", ix, iy)
        for i, (ix, iy) in enumerate(cells[: cfg.n_cylinders])
    }


def apply_runtime_tweaks(env, cfg: ConfinedShelfConfig) -> None:
    """Apply runtime model tweaks the static MJCF can't express.

    Currently only one tweak: shrink the Franka ``hand_capsule`` (the
    4 cm wrist guard) to ``cfg.hand_capsule_radius_override``.  At its
    default 4 cm radius the capsule overlaps a held cylinder's body
    during palm-+y grasps and physics pushes the cylinder forward by
    several cm during the descent.  Shrinking to 2 cm keeps the
    wrist outside the cylinder's footprint and picks land clean.

    Call AFTER ``builder.build_env(...)`` and BEFORE the first
    interaction with ``env``.  Idempotent.
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


def default_color_sort_goal(
    cfg: ConfinedShelfConfig,
    cylinder_color_groups: List[int],
    region_name: str = "shelf_interior",
) -> List[Tuple]:
    """Goal: colour-by-column — each colour's cylinders share a column.

    Colours are assigned to **non-adjacent even columns** (0, 2, 4, …)
    because a FRONT grasp can't reach a cylinder whose lateral neighbour
    is occupied (the empirical clearance rule): two colours in adjacent
    columns would be mutually unplaceable.  Each colour fills its column
    front-to-back; a colour with more cylinders than the column has
    rows spills into the next free even column.  Excluded cells (e.g.
    the back-right deep corner) are skipped.

    Returns concrete ``("occupied", cell_id, cyl)`` goal literals — a
    fully-specified target arrangement.  Raises ``ValueError`` if the
    cylinders don't fit the available even columns.
    """
    even_cols = list(range(0, cfg.n_grid_x, 2))
    excl = excluded_cells(cfg.n_grid_x, cfg.n_grid_y)

    by_color: Dict[int, List[int]] = {}
    for i, g in enumerate(cylinder_color_groups):
        by_color.setdefault(g, []).append(i)

    col_queue = list(even_cols)

    def _rows(col: int) -> List[int]:
        return [iy for iy in range(cfg.n_grid_y) if (col, iy) not in excl]

    def _next_col() -> int:
        if not col_queue:
            raise ValueError(
                "confined_shelf colour-by-column goal: cylinders/colours "
                f"exceed the {len(even_cols)} available even columns")
        return col_queue.pop(0)

    goals: List[Tuple] = []
    for g in sorted(by_color):                 # each colour starts a fresh column
        col = _next_col()
        rows = _rows(col)
        ri = 0
        for cyl_idx in by_color[g]:
            if ri >= len(rows):                # colour overflows -> next even column
                col = _next_col()
                rows = _rows(col)
                ri = 0
            goals.append((
                "occupied", Cell(region_name, col, rows[ri]).id,
                f"cyl_{cyl_idx}",
            ))
            ri += 1
    return goals
