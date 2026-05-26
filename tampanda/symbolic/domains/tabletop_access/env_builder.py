"""SceneBuilder factories for the tabletop-access (Bouhsain HAL 2025) domain.

Two scene variants share the same PDDL domain:

* :func:`make_access_builder` — ``access`` problem.  Free-standing
  3-tier shelf (no table); YCB-proxy items occupy the middle deck;
  three placement grids: floor / middle deck / top deck.
* :func:`make_access19_builder` — ``access-19`` problem.  Deck-style
  open-tunnel shelf on a table; generic same-sized cubes; two placement
  grids: shelf interior / shelf top deck.

Both factories expose the same return tuple
``(builder, workspace, config)`` so :func:`make_tabletop_access_bridge`
can consume either uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tampanda.scenes import (
    ArmSceneBuilder, TABLE_TEMPLATE,
    WORKSPACE_X_OFFSET, WORKSPACE_Y_OFFSET,
)
from tampanda.scenes.assets.asset_set import (
    Asset,
    AssetSet,
    MultiTierShelf,
    Shelf,
    make_generic_boxes,
    make_ycb_proxy,
)
from tampanda.symbolic.workspace import GridRegion, Workspace


# ----------------------------------------------------------------------
# Common config — shared by both scene variants.
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TabletopAccessConfig:
    """Shared parameters for tabletop-access scenes.

    Attributes:
        scene_variant:  ``"access"`` or ``"access-19"``.
        n_objects:      Total number of movables (target + blockers).
        cell_size:      Grid resolution in metres — typically tuned to the
                        smallest box's xy half-extent so each box fits
                        in one cell.
        target_id:      Asset id of the distinguished target.
        hide_far_x:     Sentinel x for parked objects.
        shelf_pos:      World XYZ of the shelf body origin.
    """
    scene_variant: str
    n_objects: int
    cell_size: float = 0.045
    target_id: str = "ooi"
    hide_far_x: float = 100.0
    shelf_pos: Tuple[float, float, float] = (0.55, 0.0, 0.0)
    # Runtime shrink of the Franka ``hand_capsule`` geom (the 4 cm
    # wrist guard) — for closed-top variants the default radius
    # clips the side / top walls when the wrist swings during the
    # joint-lerp descent.  Shrinking to 2 cm matches what confined_shelf
    # and confined_pickonly do.  ``None`` disables.
    hand_capsule_radius_override: Optional[float] = 0.02


# ----------------------------------------------------------------------
# access (3-tier free-standing shelf, YCB-proxy items)
# ----------------------------------------------------------------------

def make_access_builder(
    scratch_dir: Path,
    ycb_items: Optional[List[str]] = None,
    target_item: str = "meat_can",
    deck_size: Tuple[float, float] = (0.60, 0.45),
    # Paper-faithful 2-plate shelf (HAL 2025):
    #   * shelf body at the world floor (z=0, no pedestal).
    #   * lower plate ("middle_deck") at world z=0.30 — within the
    #     densest part of the palm-+y reachable region per the Phase A
    #     map.  This is the "middle shelf" where YCB items rest in
    #     the paper photo.
    #   * upper plate ("top_deck") at world z=0.55 — front cells
    #     reachable; some back cells past reach (excluded via mask).
    #   * legs extend from world floor (z=0) up to the top plate.
    #   * "floor_left" / "floor_right" tiers are the L/R compartments
    #     under the lower plate, between the legs — items in them sit
    #     on the WORLD FLOOR (no bottom plate; the world floor is the
    #     resting surface).  The vertical separator wall splits them.
    #
    # Earlier defaults (base_height=0.45, deck_levels=(0.28, 0.42))
    # added a 45 cm pedestal and an unreachable top deck.  Removed
    # because the paper photo shows the shelf legs go directly to the
    # wooden floor with the spam can sitting alongside.
    deck_levels: Tuple[float, ...] = (0.30, 0.55),
    cell_size: float = 0.06,
    base_height: float = 0.0,
    shelf_pos: Optional[Tuple[float, float, float]] = None,
    n_uniform_blockers: Optional[int] = None,
    blocker_half: Tuple[float, float, float] = (0.030, 0.030, 0.045),
) -> Tuple[ArmSceneBuilder, Workspace, TabletopAccessConfig]:
    """Build the HAL ``access`` scene.

    A **free-standing 3-layer shelf** (no table): floor + level-1 +
    top-level.  The bottom (floor) compartment has a vertical separator
    splitting it into ``floor_left`` / ``floor_right`` halves — varied
    YCB-proxy objects (potentially occluded by the front lip / by each
    other) start in one half, and the goal is to place them in the
    correct positions across the three levels.

    Layout (post +y rotation):

    * Shelf body at ``shelf_pos = (0, 0.45, 0)`` — directly in front of
      the robot.  The deck's long axis (``deck_size[0]``, world x)
      spreads side-to-side; the short axis (world y) is the depth.
    * 4 placement regions: ``floor_left``, ``floor_right``,
      ``middle_deck``, ``top_deck``.  Each is a :class:`GridRegion`.
    * Floor compartments are accessed from the front (open -y face);
      the upper decks are accessible from above and from the front.

    Args:
        scratch_dir:    Directory where templates are materialised; must
                        outlive the subsequent ``builder.build_env()``.
        ycb_items:      YCB-proxy item names.  Defaults to a 5-item mix
                        matching the paper figure.
        target_item:    Which ycb_items entry is the OoI.  Renamed to
                        ``ooi`` in the scene so the bridge identifies it.
        deck_size:      ``(lx, ly)`` of each shelf deck slab.  Default
                        50 × 30 cm (wide enough for several YCB items
                        per row, shallow enough that all cells reach).
        deck_levels:    Z heights of the deck-top surfaces above the
                        shelf base.  Default ``(0.18, 0.40)`` — a
                        18 cm tall bottom compartment + 22 cm middle.
        cell_size:      Grid resolution; default 10 cm to fit the
                        bigger YCB items.
        shelf_pos:      World position of the shelf body origin.

    Returns:
        ``(builder, workspace, config)``.
    """
    if n_uniform_blockers is None:
        ycb_items = ycb_items or [
            # Short items only (no taller than ~5 cm) so each fits the
            # floor compartment with gripper clearance.
            "meat_can", "tomato_soup_can", "tuna_can",
            "gelatin_box", "pudding_box",
        ]
        if target_item not in ycb_items:
            raise ValueError(
                f"target_item {target_item!r} not in ycb_items {ycb_items}")
        n_objects = len(ycb_items)
    else:
        # Dataset roster: OoI + N uniform generic blocker boxes — all short
        # and graspable (like access19's cubes), enough for many clutter.
        n_objects = n_uniform_blockers + 1

    # Shelf body z = ``base_height`` so the bottom (floor) compartment
    # sits at that height in world frame.  Legs extend downward to the
    # world floor (z=0).  Default 45 cm puts the floor in the Franka's
    # comfortable reach.
    if shelf_pos is None:
        # ``(0.25, 0.70, 0)`` — optimal from the placement sweep
        # (``examples/sweep_access_shelf.py``): 64/64 middle/top
        # deck cells + 8/8 floor-left + 8/8 floor-right cells IK-
        # converge from the canonical palm-+y staging seed.  Other
        # placements lose 1–24 cells (see ``summary.json`` from the
        # sweep).  Tested with ``deck_size=(0.60, 0.45)`` and
        # ``base_height=0``.
        shelf_pos = (0.25, 0.70, base_height)

    cfg = TabletopAccessConfig(
        scene_variant="access",
        n_objects=n_objects,
        cell_size=cell_size,
        shelf_pos=shelf_pos,
    )

    shelf = MultiTierShelf(
        asset_id="access_shelf",
        deck_size=deck_size,
        deck_thickness=0.012,
        deck_levels=deck_levels,
        leg_size=(0.018, 0.018),
        leg_inset=0.020,
        floor_separator=True,
        floor_separator_thickness=0.010,
        base_height=base_height,
        # No bottom plate — items in the floor compartments sit on
        # the WORLD FLOOR, matching the paper layout (shelf legs run
        # to the wooden floor; the spam can sits on that same floor
        # beside the shelf, not on a raised plate).
        bottom_plate=False,
    )
    if n_uniform_blockers is None:
        ycb_assets = list(make_ycb_proxy(ycb_items))
        renamed: List[Asset] = []
        for a in ycb_assets:
            if a.asset_id == target_item:
                renamed.append(Asset(
                    asset_id="ooi",
                    half_extents=a.half_extents,
                    color=a.color,
                    label=f"target ({target_item})",
                ))
            else:
                renamed.append(a)
    else:
        renamed = [Asset(asset_id="ooi", half_extents=blocker_half,
                         color=(0.15, 0.40, 0.90, 1.0), label="target (ooi)")]
        renamed += list(make_generic_boxes(
            "blocker", [blocker_half] * n_uniform_blockers))
    aset = AssetSet([shelf, *renamed])

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for a in aset:
        path = scratch_dir / f"{a.asset_id}.xml"
        path.write_text(a.render_template_xml())

    b = ArmSceneBuilder()
    for a in aset:
        b.add_resource(a.asset_id, str(scratch_dir / f"{a.asset_id}.xml"))

    # Free-standing shelf — no table.  Body origin sits at world floor.
    b.add_object(shelf.asset_id, name="shelf", pos=list(cfg.shelf_pos))

    # Park all objects off-screen.
    parked_z = max(a.half_extents[2] for a in renamed) + 0.005
    for a in renamed:
        b.add_object(a.asset_id, name=a.asset_id,
                     pos=[cfg.hide_far_x, 0.0, parked_z])

    # ---- Workspace: 4 grids (floor split by the separator + 2 decks) ----
    sx_world, sy_world, sz_world = cfg.shelf_pos
    dx, dy = deck_size

    # Compute the leg-inside extent so the floor grid never overlaps
    # the corner posts.  Items placed at the leftmost / rightmost cell
    # otherwise straddle the legs (visible as a clipping artifact).
    # ``shelf.leg_size[0]`` is the full leg X-thickness.
    leg_inside_x = dx / 2 - shelf.leg_inset - shelf.leg_size[0]
    leg_inside_y = dy / 2 - shelf.leg_inset - shelf.leg_size[1]
    # Symmetric hand clearance: the floor grids stay one
    # ``hand_clearance`` away from BOTH the corner legs (outer edge)
    # and the floor separator (inner edge).  Asymmetry between the
    # two sides is what made floor_right unreachable while
    # floor_left was fine.
    sep_clear = 0.08
    # ``hand_clearance`` pushes the placement grid INWARD from the
    # corner legs so that an FRONT-grasping hand approaching the
    # outer columns doesn't clip the leg post.  Validated by single-
    # cell IK probes: a 4 cm offset gives the Franka hand_capsule
    # ~5 cm clearance at the outermost cube column.
    hand_clearance = 0.08
    floor_half_extent_x = leg_inside_x - sep_clear - hand_clearance
    floor_extent_y = 2 * leg_inside_y - hand_clearance
    floor_left_origin = (sx_world - leg_inside_x + hand_clearance,
                          sy_world - leg_inside_y + hand_clearance / 2)
    floor_right_origin = (sx_world + sep_clear,
                           sy_world - leg_inside_y + hand_clearance / 2)
    # Upper decks have the same leg-clearance requirement.
    upper_origin = (sx_world - leg_inside_x + hand_clearance,
                     sy_world - leg_inside_y + hand_clearance / 2)
    upper_extent = (2 * (leg_inside_x - hand_clearance),
                     2 * leg_inside_y - hand_clearance)

    # ``level_z`` for each region is the world z of an item's CENTRE
    # when it rests on that region's surface.  Adding half a typical
    # YCB item height (~5 cm) to the surface z so that
    # ``env.set_object_pose(name, [x, y, level_z])`` (which sets the
    # free-joint BODY position to ``level_z``) lands the item sitting
    # on the surface, not interpenetrating it.  Mirrors access19's
    # ``interior_level_z = floor_z + cube_half_extents[2]`` convention.
    _ITEM_HALF_Z = 0.05  # representative YCB half-height
    floor_z = sz_world + _ITEM_HALF_Z          # world floor surface = sz_world
    middle_z = sz_world + deck_levels[0] + _ITEM_HALF_Z
    top_z = sz_world + deck_levels[1] + _ITEM_HALF_Z

    # ``excluded_cells`` flags grid positions geometrically valid but
    # kinematically out of reach.  The numbers come from
    # ``examples/check_executability.py --mode full`` against the
    # default geometry: any change to ``deck_size`` / ``cell_size`` /
    # ``hand_clearance`` / ``deck_levels`` should trigger a re-run to
    # confirm these remain accurate.
    #
    # NOTE: the ``top_excl`` empirical set below was tuned at the
    # OLD geometry (``deck_levels=(0.28, 0.42)``, top at world z=0.87).
    # With the new lower decks (top at z=0.70) the mask over-excludes
    # — fewer back-row cells are actually past the reach envelope.
    # Re-running ``check_executability --domain tabletop_access:access
    # --mode full`` will tighten this; until then we conservatively
    # keep the old mask so the bridge never proposes a cell we
    # haven't validated.
    cells_x_floor = int(round(floor_half_extent_x / cell_size))
    cells_y_floor = int(round(floor_extent_y / cell_size))
    cells_x_top = int(round(upper_extent[0] / cell_size))
    cells_y_top = int(round(upper_extent[1] / cell_size))
    # Floor compartments: column ix=1 of each 2-wide compartment is
    # unreachable (gripper clips the floor separator or extends past
    # the reach envelope on the +x side).  The deepest two cells of
    # ix=0 in floor_left ALSO fail — the arm has to fit under the
    # middle deck at z ≈ 0.11 while wrapping around the left-side
    # legs, and the joint-space corridor narrows past iy=2.  Floor_
    # right's ix=0 is fully reachable.  Set empirically against the
    # new geometry (base_height=0, deck_levels=(0.30, 0.55)) via
    # ``check_executability --domain tabletop_access:access --mode
    # full``.
    floor_left_excl: frozenset = frozenset()
    floor_right_excl: frozenset = frozenset()
    if cells_x_floor == 2:
        # ix=1 always excluded.
        base_excl = {(1, iy) for iy in range(cells_y_floor)}
        # At the optimal shelf placement ``(0.25, 0.70)`` the
        # placement-sweep (``examples/sweep_access_shelf.py``) shows
        # all floor_left ix=0 rows IK-converge.  At other placements
        # back-row cells may fail — the conservative mask refreshes
        # itself empirically if the shelf is moved.
        floor_left_excl = frozenset(base_excl)
        floor_right_excl = frozenset(base_excl)
    # Top deck: empirical at new geometry — full 23/23 reachable
    # subset, conservative mask kept from the old z=0.87 measurement
    # until a tighter sweep is run.  Likely over-excludes back cells.
    # At ``shelf_pos=(0.25, 0.70)`` the placement-sweep showed all
    # top-deck cells in rows iy 0..3 IK-convergent.  The back-row
    # corners (iy=4) past the Franka reach envelope at z = top-deck
    # remain unreachable — empirically just the two right-corner
    # cells.  Refresh after any shelf geometry change.
    top_excl: frozenset = frozenset()
    if cells_x_top == 6 and cells_y_top == 5:
        top_excl = frozenset({(4, 4), (5, 4)})
    workspace = Workspace([
        GridRegion(
            name="floor_left",
            origin=floor_left_origin,
            extent=(floor_half_extent_x, floor_extent_y),
            cell_size=cell_size,
            level_z=floor_z,
            # Floor compartments are reached from the front.
            access_modes=("front",),
            excluded_cells=floor_left_excl,
        ),
        GridRegion(
            name="floor_right",
            origin=floor_right_origin,
            extent=(floor_half_extent_x, floor_extent_y),
            cell_size=cell_size,
            level_z=floor_z,
            access_modes=("front",),
            excluded_cells=floor_right_excl,
        ),
        GridRegion(
            name="middle_deck",
            origin=upper_origin,
            extent=upper_extent,
            cell_size=cell_size,
            level_z=middle_z,
            # Middle deck has open sides + open front; top-down OK iff
            # there's deck-clearance.
            access_modes=("front", "top_down"),
        ),
        GridRegion(
            name="top_deck",
            origin=upper_origin,
            extent=upper_extent,
            cell_size=cell_size,
            level_z=top_z,
            access_modes=("top_down", "front"),
            excluded_cells=top_excl,
        ),
    ])
    return b, workspace, cfg


# ---------------------------------------------------------------------------
# Layout / goal helpers for HAL ``access``
# ---------------------------------------------------------------------------

def access_default_initial_layout(
    workspace: Workspace,
    object_ids: List[str],
    target_id: str = "ooi",
    rng: Optional["np.random.Generator"] = None,
    min_separation: int = 1,
):
    """Spawn all objects in randomized non-adjacent ``middle_deck`` cells.

    Matches the HAL paper figure layout: items are arranged on the
    middle shelf (the lower of the 2 plates).  Each call places the
    target plus the other objects at a fresh random subset of
    ``middle_deck`` cells, with a minimum cell-grid separation between
    any two placed items so YCB items wider than a single cell don't
    interpenetrate.

    Args:
        workspace: The Workspace returned by ``make_access_builder``.
        object_ids: All movable object names (including the target).
        target_id: Name of the OoI (default ``"ooi"``).
        rng: Optional ``np.random.Generator`` for deterministic
            shuffles.  When ``None``, uses ``np.random.default_rng()``
            with fresh entropy.
        min_separation: Chebyshev (max-of-|dx|,|dy|) cell distance
            below which two placed items are considered adjacent and
            rejected.  Default 1 → no 8-neighbour cell may host
            another item; effectively ≥2 cell-pitches between item
            centres.  Set ``0`` for the legacy dense layout.
    """
    import numpy as np
    from tampanda.symbolic.workspace import Cell

    region = workspace["middle_deck"]
    available = [(c.ix, c.iy) for c in region.cells()]
    if not available:
        raise ValueError("middle_deck has no reachable cells")
    if len(object_ids) > len(available):
        raise ValueError(
            f"middle_deck has {len(available)} reachable cells but "
            f"layout requires {len(object_ids)} objects"
        )

    if rng is None:
        rng = np.random.default_rng()

    def _far_enough(picked, candidate):
        if min_separation <= 0:
            return True
        cix, ciy = candidate
        for ix, iy in picked:
            if max(abs(ix - cix), abs(iy - ciy)) <= min_separation:
                return False
        return True

    # Greedy non-adjacent sampling: shuffle, then walk and keep each
    # cell that's far enough from already-chosen ones.  If too few
    # cells fit the separation constraint, raise.
    for attempt in range(50):
        order = rng.permutation(len(available)).tolist()
        picked: List[Tuple[int, int]] = []
        for idx in order:
            cand = available[idx]
            if _far_enough(picked, cand):
                picked.append(cand)
                if len(picked) == len(object_ids):
                    break
        if len(picked) == len(object_ids):
            break
    else:
        raise ValueError(
            f"middle_deck has {len(available)} cells but no layout fits "
            f"{len(object_ids)} items with min_separation={min_separation}"
        )

    placements: Dict[str, Cell] = {}
    placements[target_id] = Cell(region.name, *picked[0])
    other_ids = [n for n in object_ids if n != target_id]
    for name, (ix, iy) in zip(other_ids, picked[1:]):
        placements[name] = Cell(region.name, ix, iy)
    return placements


def access_default_goal(
    workspace: Workspace,
    target_id: str = "ooi",
):
    """Goal: target ends up on the top deck (the "right position").

    Simple version of the paper's task — extract the target from the
    cluttered bottom compartment and place it on the top deck.
    """
    from tampanda.symbolic.workspace import Cell

    top = workspace["top_deck"]
    centre = Cell(top.name, top.cells_x // 2, top.cells_y // 2)
    return [("occupied", centre.id, target_id)]


# ----------------------------------------------------------------------
# access-19 (deck-style shelf, generic cubes, 2 grids)
# ----------------------------------------------------------------------

def make_access19_builder(
    scratch_dir: Path,
    # 4×4×8 cm cubes — z half-extent ≥ 0.040 keeps the FRONT grasp's
    # link7-clearance landing (floor + 5.5 cm) inside the block body.
    cube_half_extents: Tuple[float, float, float] = (0.020, 0.020, 0.040),
    n_red: int = 18,
    # 20 mm channel widens the column pitch to 6 cm so off-centre
    # columns at ix=1,5 land outside the Franka palm-+y dead zone.
    inter_row_gap: float = 0.020,
    # 24 cm tall interior: cube top sits at floor+0.08; the arm's
    # wrist max-z while reaching is +9.5 cm above the cube top, but
    # the forearm/elbow can ride higher than the wrist, so the
    # original 12 cm headroom let the forearm clip the top wall on
    # FRONT grasps.  0.28 m would give 20 cm headroom (clearly
    # clears the elbow) but raises the top deck so far that
    # pick_deck IK fails at the +x edge columns (col_5 target at
    # safe_z = top_deck + 0.08 lands near the Franka's max reach).
    # 0.24 m (16 cm headroom) is the compromise — 4 cm more clearance
    # than the original, top_deck only 4 cm higher so safe_z stays
    # reachable.
    interior_height_z: float = 0.24,
    wall_thickness: float = 0.012,
    # 40 cm pedestal places the cubicle floor at world z ≈ 0.41 —
    # in the Franka palm-+y reach band (z ∈ [0.45, 0.55]).
    pedestal_height: float = 0.40,
    # Free-standing shelf — NO table.  The pedestal + back wall +
    # floor are integral to the shelf body.
    use_table: bool = False,
    # Validated by ``examples/measure_access19_arm_extent.py``:
    # (0.35, 0.40) gives 21/21 reachable cells using column-aligned
    # approach + row-by-row descent + gripper-invariance.  +/- 1 cm
    # shifts can reintroduce point-like wrist singularities at edge
    # cells, so don't override casually.
    table_pos: Tuple[float, float, float] = (0.35, 0.40, 0.0),
    # Side margin — arm extends ~8 cm past the end columns when
    # reaching them; 10 cm gives a safe contact margin.
    side_margin: float = 0.10,
    # ``shelf_top`` placement region: 7 × 7 cells covering the shelf's
    # footprint (mirrors the 7 × 7 interior).  Chain-based reachability
    # check at ``table_pos = (0.35, 0.40)`` confirms 49/49 cells are
    # reachable by both pick and put — see /tmp checks in the
    # feat/multigrid-domains branch history.  3 × 3 was the original
    # conservative default; expanded so the planner has room to park
    # blockers + the OoI without competing for the same cells.
    top_grid_cells: int = 7,
) -> Tuple[ArmSceneBuilder, Workspace, TabletopAccessConfig]:
    """Build the HAL ``access-19`` scene.

    Confined-cubicle shelf (open in -x only — robot-facing).  18 red
    cubes in 3 columns × 6 rows packed tight in depth + 1 blue OoI at
    the back of the middle column, behind the 6 reds.  Cubes touch
    along the depth axis (paper-faithful — only front-row cubes are
    accessible until the front rows are removed).  Columns have a
    one-cell-wide gripping channel between them so the gripper fingers
    fit alongside any column without bumping its neighbours.

    Two placement grids: shelf interior (where the cubes start) and
    shelf top (the OoI's goal placement).

    Args:
        scratch_dir: Directory where templates are materialised.
        cube_half_extents: Half-extents of every cube; must be cubic
            (lx == ly).
        n_red: Number of red blocker cubes.  Total = n_red + 1.
        inter_row_gap: Tiny depth gap for MuJoCo contact stability.
            Default 1 mm — visually the cubes look like they touch.
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

    # GridRegion uses one cell_size for both axes.  Cubes occupy single
    # cells; columns are interleaved with one-cell-wide channels for
    # gripper-finger clearance, so column pitch = 2 × cell_size.
    cell_size = cube_d + inter_row_gap

    # +y orientation: the robot reaches in +y, so cube columns spread
    # along world x (cells_x = 7 = 3 cube cols + 4 channels) and rows
    # extend along world y / depth (cells_y = n_red_rows + 1).
    cells_x = 2 * n_cols + 1            # width — 3 cube cols + 4 channels
    cells_y = n_red_rows + 1            # depth — rows + back-row OoI

    region_extent_x = cells_x * cell_size  # world x = width
    region_extent_y = cells_y * cell_size  # world y = depth

    # access-19 only ever uses FRONT grasps (closed top → no top-down).
    front_margin = 0.02
    back_margin  = 0.04
    interior_x = region_extent_x + 2 * side_margin
    interior_y = region_extent_y + front_margin + back_margin

    # Free-standing shelf — no table.  Shelf body z = pedestal +
    # wall_thickness + interior_height/2 (so the pedestal sits
    # flush on the world floor at z=0).  When ``use_table`` is True,
    # the shelf is mounted on top of a 27-cm-tall table instead.
    base_offset_z = 0.27 if use_table else 0.0
    shelf_body_z = (table_pos[2] + base_offset_z + pedestal_height
                     + wall_thickness + interior_height_z / 2)
    cfg = TabletopAccessConfig(
        scene_variant="access-19",
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
        # Free-standing shelf — pedestal sits directly on world floor.
        b.add_object(shelf.asset_id, name="shelf",
                     pos=list(cfg.shelf_pos))

    # Park all objects off-screen.
    parked_z = cube_half_extents[2] + 0.005
    for a in red_assets + [target_asset]:
        b.add_object(a.asset_id, name=a.asset_id,
                     pos=[cfg.hide_far_x, 0.0, parked_z])

    # ---- Workspace: 2 grids — shelf interior + shelf top deck ----------
    sx_world = cfg.shelf_pos[0]
    sy_world = cfg.shelf_pos[1]
    # Region covers exactly the cube cells, centred in the shelf interior.
    region_origin = (
        sx_world - region_extent_x / 2,
        sy_world - region_extent_y / 2,
    )

    # Interior grid: cubes rest on the shelf interior floor.
    interior_floor_world_z = cfg.shelf_pos[2] - interior_height_z / 2
    interior_level_z = interior_floor_world_z + cube_half_extents[2]

    # Top-deck grid: cubes rest on the +z exterior face of the top wall.
    top_deck_top_z = cfg.shelf_pos[2] + interior_height_z / 2 + wall_thickness
    top_level_z = top_deck_top_z + cube_half_extents[2]

    # Shelf-top region — small grid centred on the shelf, just big
    # enough for the planning puzzle (a few blocker parking spots
    # + the OoI's distinguished placement).  9 cells (3 × 3) is
    # plenty.
    top_extent = (top_grid_cells * cell_size, top_grid_cells * cell_size)
    top_origin = (sx_world - top_extent[0] / 2,
                   sy_world - top_extent[1] / 2)

    # Two diagonal-corner cells fail regardless of hand_capsule
    # shrink: (cells_x-1, cells_y-1) is past the Franka's 0.85 m
    # reach envelope (world distance ~0.91 m from base), and (0, 0)
    # is at an IK basin singularity for the column-aligned approach
    # the custom ``pick_fn`` uses.  Confirmed by ``check_executability.py
    # --mode full``.
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
            # Single open face — the robot-facing -y side.
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


def apply_runtime_tweaks(env, cfg: TabletopAccessConfig) -> None:
    """Apply runtime model tweaks the static MJCF can't express.

    Shrinks the Franka ``hand_capsule`` to
    ``cfg.hand_capsule_radius_override`` (2 cm by default).  At the
    default 4 cm the capsule clips the closed-top cubicle's side
    walls on the corner cells of ``access-19``; with the shrink, all
    49 cells of the 7×7 interior become reachable.  No-op on the
    open 3-tier ``access`` variant — the floor compartments have
    enough side-margin already — but harmless to apply.

    Call AFTER ``builder.build_env(...)`` and BEFORE the first
    interaction.  Idempotent.
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
