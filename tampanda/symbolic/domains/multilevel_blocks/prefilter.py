"""Geometric pre-filters for multilevel_blocks actions.

Cheap (microsecond) rejection of actions that are GEOMETRICALLY impossible
given the current occupancy state — before the executor's expensive
IK + collision-check chain runs.  Conservative: a filter only returns
``INFEASIBLE`` when CERTAIN no yaw / approach could possibly work.  False
positives (filter says UNKNOWN, executor later says no) are fine and
fall through to the full check.  False negatives (filter says
INFEASIBLE, executor would have said yes) MUST NOT occur — they would
silently drop feasible plans.

Wired into :func:`check_action`: if any filter returns ``INFEASIBLE``,
the executor chain is skipped and a ``success=False, error="prefilter:<rule>"``
result is returned immediately.

Rule families
-------------

**Jaw-clearance**: parallel-jaw gripper has two jaws on opposite sides
of the held block.  For a pick / put to succeed, AT LEAST ONE yaw
orientation must have BOTH jaw sides clear of immediate cell
neighbours.  A cube has 2 distinct yaws (90° / 0°); a flat-x / flat-y
oblong has 1 grasp orientation (jaws perpendicular to the long axis);
a long-x / long-y has 1; an upright oblong / long has 4 distinct
front-grasp yaws (the block has 4-fold lateral symmetry).  Each rule
fires only when ALL yaws are blocked.

**Descent-column-clear**: a put-upright / put-long-upright drops the
held block down a vertical column (the gripper descends from
``traverse_z`` to ``c-high + 4 cm``).  If the destination column at
levels ABOVE the put's top cell is occupied at the same ``(ix, iy)``,
the wrist body clips on the way down.

The filters are pure functions of the symbolic ``state`` dict (the
same one ``restore_state`` consumes).  Cell occupancy is read from
the ``(in block cell)`` and ``(empty cell)`` predicates if present;
absence in ``(in ...)`` is interpreted as empty.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from tampanda.symbolic.workspace import Cell

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
)


# ---------------------------------------------------------------------------
# Cell helpers — region-aware (ix, iy, level) arithmetic
# ---------------------------------------------------------------------------


def _parse(cell_id: str) -> Cell:
    """``Cell.parse`` thin wrapper for in-module brevity."""
    return Cell.parse(cell_id)


def _occupied_cells(state: Dict[Tuple, bool]) -> Set[str]:
    """Return the set of cell ids that have at least one ``(in b c)`` atom
    set True in the state.  Cells absent from ``(in ...)`` are treated as
    empty (matches what ``restore_state`` does — empty by default)."""
    return {
        key[2]
        for key, val in state.items()
        if val and isinstance(key, tuple) and len(key) == 3 and key[0] == "in"
    }


# Cardinal-direction offsets (ix, iy) for "neighbour at same region/level".
_NSEW: Tuple[Tuple[str, int, int], ...] = (
    ("n",  0, +1),
    ("s",  0, -1),
    ("e", +1,  0),
    ("w", -1,  0),
)


def _neigh_at_same_region(cell: Cell, dx: int, dy: int) -> str:
    """Cell id at offset (dx, dy) in the same region."""
    return f"{cell.region}__{cell.ix + dx}_{cell.iy + dy}"


def _column_above(cell: Cell, n_stack_levels: int) -> Tuple[str, ...]:
    """Stack cells directly above ``cell`` up to the top stack level.

    Returns an empty tuple if ``cell`` is not a stack cell or is already
    at the top level.
    """
    if not cell.region.startswith("stack_L"):
        return ()
    level = int(cell.region.split("_L")[1])
    return tuple(
        f"stack_L{k}__{cell.ix}_{cell.iy}"
        for k in range(level + 1, n_stack_levels)
    )


# ---------------------------------------------------------------------------
# Per-shape jaw-clearance predicates
# ---------------------------------------------------------------------------


def _cube_jaws_clear(cell: Cell, occupied: Set[str]) -> bool:
    """Returns True iff at least one yaw (0 or 90 °) has both jaw sides
    clear of immediate cardinal neighbours.

    Cube has 2 distinct yaw choices (parallel-jaw symmetry collapses 4
    to 2):
        yaw-0  → jaws along ±y → clear iff (N empty) AND (S empty)
        yaw-90 → jaws along ±x → clear iff (E empty) AND (W empty)
    """
    n = _neigh_at_same_region(cell, 0, +1) in occupied
    s = _neigh_at_same_region(cell, 0, -1) in occupied
    e = _neigh_at_same_region(cell, +1, 0) in occupied
    w = _neigh_at_same_region(cell, -1, 0) in occupied
    yaw0_clear = (not n) and (not s)
    yaw90_clear = (not e) and (not w)
    return yaw0_clear or yaw90_clear


def _flat_x_jaws_clear(cell_west: Cell, cell_east: Cell,
                            occupied: Set[str]) -> bool:
    """flat-x oblong (long axis along +x).  Only one grasp orientation —
    jaws along ±y at the block centroid.  Clear iff both long-side rows
    (entirely +y of the block, entirely -y) have no occupants directly
    adjacent.

    Long-side neighbours: (cell_west.iy ± 1) over BOTH west and east
    cells.  Filter when BOTH +y and -y rows have at least one occupant
    each (no jaw clearance possible).
    """
    n_block = (_neigh_at_same_region(cell_west, 0, +1) in occupied
                  or _neigh_at_same_region(cell_east, 0, +1) in occupied)
    s_block = (_neigh_at_same_region(cell_west, 0, -1) in occupied
                  or _neigh_at_same_region(cell_east, 0, -1) in occupied)
    return not (n_block and s_block)


def _flat_y_jaws_clear(cell_south: Cell, cell_north: Cell,
                            occupied: Set[str]) -> bool:
    """flat-y oblong (long axis along +y).  Jaws along ±x at centroid.
    Symmetric to flat-x with axes swapped."""
    e_block = (_neigh_at_same_region(cell_south, +1, 0) in occupied
                  or _neigh_at_same_region(cell_north, +1, 0) in occupied)
    w_block = (_neigh_at_same_region(cell_south, -1, 0) in occupied
                  or _neigh_at_same_region(cell_north, -1, 0) in occupied)
    return not (e_block and w_block)


def _long_x_jaws_clear(c1: Cell, c2: Cell, c3: Cell,
                            occupied: Set[str]) -> bool:
    """long-x (3 cells along +x).  Jaws along ±y at the centroid (= c2).
    Same logic as flat-x but with 3 cells to consider.  Long-axis end
    neighbours `(c1.west)` and `(c3.east)` could clip the wrist body
    extension — apply that as a secondary filter."""
    n_block = (
        _neigh_at_same_region(c1, 0, +1) in occupied
        or _neigh_at_same_region(c2, 0, +1) in occupied
        or _neigh_at_same_region(c3, 0, +1) in occupied
    )
    s_block = (
        _neigh_at_same_region(c1, 0, -1) in occupied
        or _neigh_at_same_region(c2, 0, -1) in occupied
        or _neigh_at_same_region(c3, 0, -1) in occupied
    )
    end_block = (
        _neigh_at_same_region(c1, -1, 0) in occupied
        and _neigh_at_same_region(c3, +1, 0) in occupied
    )
    return (not (n_block and s_block)) and (not end_block)


def _long_y_jaws_clear(c1: Cell, c2: Cell, c3: Cell,
                            occupied: Set[str]) -> bool:
    """long-y: symmetric to long-x along the y-axis."""
    e_block = (
        _neigh_at_same_region(c1, +1, 0) in occupied
        or _neigh_at_same_region(c2, +1, 0) in occupied
        or _neigh_at_same_region(c3, +1, 0) in occupied
    )
    w_block = (
        _neigh_at_same_region(c1, -1, 0) in occupied
        or _neigh_at_same_region(c2, -1, 0) in occupied
        or _neigh_at_same_region(c3, -1, 0) in occupied
    )
    end_block = (
        _neigh_at_same_region(c1, 0, -1) in occupied
        and _neigh_at_same_region(c3, 0, +1) in occupied
    )
    return (not (e_block and w_block)) and (not end_block)


def _upright_jaws_and_column_clear(c_low: Cell, c_high: Cell,
                                          occupied: Set[str],
                                          n_stack_levels: int = 5) -> bool:
    """Upright oblong / long: front-grasp from horizontal direction.

    Jaws close along the block's long horizontal axis (perpendicular to
    the block's vertical orientation).  Upright has 4-fold lateral
    symmetry → 4 yaws.  Each yaw has jaws along one of ±x or ±y.

    Pre-filter rejects iff ALL 4 yaws have at least one jaw side blocked
    AT THE LOWER CELL.  Equivalently: all of (N, S, E, W) of c_low are
    occupied at the same level.

    ALSO check the column ABOVE c_high (descent path): if any cell in
    that column is occupied, the wrist clips during traverse-descent.
    """
    n = _neigh_at_same_region(c_low, 0, +1) in occupied
    s = _neigh_at_same_region(c_low, 0, -1) in occupied
    e = _neigh_at_same_region(c_low, +1, 0) in occupied
    w = _neigh_at_same_region(c_low, -1, 0) in occupied
    jaws_all_blocked = n and s and e and w
    if jaws_all_blocked:
        return False
    # Descent column above the top cell of the upright block.
    for above_id in _column_above(c_high, n_stack_levels):
        if above_id in occupied:
            return False
    return True


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


# Decision sentinels.
INFEASIBLE = "infeasible"
UNKNOWN = "unknown"


def filter_action(state: Dict[Tuple, bool], action: Tuple,
                       config: Optional[MultilevelBlocksConfig] = None,
                       ) -> Tuple[str, Optional[str]]:
    """Return ``(decision, reason)``.

    decision ∈ {INFEASIBLE, UNKNOWN}.  When INFEASIBLE, ``reason`` names
    the rule that fired.  When UNKNOWN, ``reason`` is None.

    The geometric filters cover pick-cube / pick-flat-x / pick-flat-y /
    pick-long-x / pick-long-y / pick-upright / pick-long-upright on the
    pick side, and the same families on the put side.  In-hand
    transforms (make-* / turn-*) always return UNKNOWN — no static
    geometric obstruction is meaningful for them.
    """
    if not action:
        return UNKNOWN, None
    name = action[0]
    occupied = _occupied_cells(state)
    n_stack = config.stack_grid_cells[2] if config is not None else 5

    # ---- PICK ----------------------------------------------------------
    if name == "pick-cube" and len(action) >= 3:
        c = _parse(action[2])
        if not _cube_jaws_clear(c, occupied):
            return INFEASIBLE, "pick-cube:no-yaw-clear"
        return UNKNOWN, None

    if name == "pick-flat-x" and len(action) >= 4:
        c1, c2 = _parse(action[2]), _parse(action[3])
        west, east = (c1, c2) if c1.ix < c2.ix else (c2, c1)
        if not _flat_x_jaws_clear(west, east, occupied):
            return INFEASIBLE, "pick-flat-x:jaws-blocked"
        return UNKNOWN, None

    if name == "pick-flat-y" and len(action) >= 4:
        c1, c2 = _parse(action[2]), _parse(action[3])
        south, north = (c1, c2) if c1.iy < c2.iy else (c2, c1)
        if not _flat_y_jaws_clear(south, north, occupied):
            return INFEASIBLE, "pick-flat-y:jaws-blocked"
        return UNKNOWN, None

    if name == "pick-long-x" and len(action) >= 5:
        cs = sorted([_parse(a) for a in action[2:5]], key=lambda c: c.ix)
        if not _long_x_jaws_clear(*cs, occupied):
            return INFEASIBLE, "pick-long-x:jaws-or-end-blocked"
        return UNKNOWN, None

    if name == "pick-long-y" and len(action) >= 5:
        cs = sorted([_parse(a) for a in action[2:5]], key=lambda c: c.iy)
        if not _long_y_jaws_clear(*cs, occupied):
            return INFEASIBLE, "pick-long-y:jaws-or-end-blocked"
        return UNKNOWN, None

    if name == "pick-upright" and len(action) >= 4:
        c_low, c_high = _parse(action[2]), _parse(action[3])
        if not _upright_jaws_and_column_clear(c_low, c_high, occupied,
                                                          n_stack_levels=n_stack):
            return INFEASIBLE, "pick-upright:jaws-or-column-blocked"
        return UNKNOWN, None

    if name == "pick-long-upright" and len(action) >= 5:
        cs = [_parse(a) for a in action[2:5]]
        # Long-upright spans 3 vertical levels of the stack column — by
        # definition every cell in the trio is a stack cell.  A GNN
        # candidate that includes a parts cell (region == "parts") is
        # geometrically impossible; short-circuit instead of crashing
        # on the int(c.region.split("_L")[1]) below.
        if not all(c.region.startswith("stack_L") for c in cs):
            return INFEASIBLE, "pick-long-upright:non-stack-cell"
        cs.sort(key=lambda c: int(c.region.split("_L")[1]))
        c_low, _c_mid, c_high = cs
        if not _upright_jaws_and_column_clear(c_low, c_high, occupied,
                                                          n_stack_levels=n_stack):
            return INFEASIBLE, "pick-long-upright:jaws-or-column-blocked"
        return UNKNOWN, None

    # ---- PUT -----------------------------------------------------------
    if name == "put-cube" and len(action) >= 3:
        c = _parse(action[2])
        if not _cube_jaws_clear(c, occupied):
            return INFEASIBLE, "put-cube:no-yaw-clear"
        return UNKNOWN, None

    if name == "put-flat-x" and len(action) >= 4:
        c1, c2 = _parse(action[2]), _parse(action[3])
        west, east = (c1, c2) if c1.ix < c2.ix else (c2, c1)
        if not _flat_x_jaws_clear(west, east, occupied):
            return INFEASIBLE, "put-flat-x:jaws-blocked"
        return UNKNOWN, None

    if name == "put-flat-y" and len(action) >= 4:
        c1, c2 = _parse(action[2]), _parse(action[3])
        south, north = (c1, c2) if c1.iy < c2.iy else (c2, c1)
        if not _flat_y_jaws_clear(south, north, occupied):
            return INFEASIBLE, "put-flat-y:jaws-blocked"
        return UNKNOWN, None

    if name == "put-long-x" and len(action) >= 5:
        cs = sorted([_parse(a) for a in action[2:5]], key=lambda c: c.ix)
        if not _long_x_jaws_clear(*cs, occupied):
            return INFEASIBLE, "put-long-x:jaws-or-end-blocked"
        return UNKNOWN, None

    if name == "put-long-y" and len(action) >= 5:
        cs = sorted([_parse(a) for a in action[2:5]], key=lambda c: c.iy)
        if not _long_y_jaws_clear(*cs, occupied):
            return INFEASIBLE, "put-long-y:jaws-or-end-blocked"
        return UNKNOWN, None

    if name == "put-upright" and len(action) >= 4:
        c_low, c_high = _parse(action[2]), _parse(action[3])
        if not _upright_jaws_and_column_clear(c_low, c_high, occupied,
                                                          n_stack_levels=n_stack):
            return INFEASIBLE, "put-upright:jaws-or-column-blocked"
        return UNKNOWN, None

    if name == "put-long-upright" and len(action) >= 5:
        cs = [_parse(a) for a in action[2:5]]
        # See pick-long-upright above — same guard against non-stack cells.
        if not all(c.region.startswith("stack_L") for c in cs):
            return INFEASIBLE, "put-long-upright:non-stack-cell"
        cs.sort(key=lambda c: int(c.region.split("_L")[1]))
        c_low, _c_mid, c_high = cs
        if not _upright_jaws_and_column_clear(c_low, c_high, occupied,
                                                          n_stack_levels=n_stack):
            return INFEASIBLE, "put-long-upright:jaws-or-column-blocked"
        return UNKNOWN, None

    # Transforms, unknown actions: no static filter.
    return UNKNOWN, None
