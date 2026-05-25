"""Instance templates for the access-19 data-gen pipeline.

Each template returns a :class:`Template` describing:

* ``source_placements`` — ``(obj_name, cell_id)`` tuples saying where
  every object STARTS.  Always a subset of the 18 cube-column cells
  (ix in {1, 3, 5}, iy in {0..5}) plus the OoI at one of the back
  cells.  Off-column cells (ix in {0, 2, 4, 6}) are never used as
  sources — the gripper geometry can't pick from there.
* ``goal_placements`` — ``(obj_name, cell_id)`` saying where each
  object must END UP for the instance to be solved.  Typically the
  OoI moves to a top-deck cell; blockers either stay where they are
  (OoI-only goal) or return to their source cells (full-return goal).
* ``metadata`` — template name + sampling params for traceability.

Templates available:

* :func:`front_row_subset(n, rng, *, return_blockers)` — random
  subset of size ``n`` of the front-row cube cells.  Goal optionally
  requires return.
* :func:`dense_front(n_rows, rng, *, return_blockers)` — fills the
  first ``n_rows`` of every cube column.  Mirrors the canonical
  layouts at the difficulty progression L1/L2/L3.
* :func:`scattered_subset(n, rng, *, return_blockers)` — random
  scatter across the 18-cell mask.  Same density as front_row_subset
  but distributed in iy too.
* :func:`canonical_18(*, return_blockers)` — the canonical full
  18-blocker eval layout.
* :func:`gotcha_corridor_jam(rng, *, return_blockers)` — hand-crafted
  config that forces the planner to move the OoI multiple times.
* :func:`mirror_x(template)` / :func:`mirror_iy(template)` — symmetry
  ops to multiply instances per template at no extra design cost.

All templates respect the "subset of 18 cube cells" constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell


_Placement = Tuple[str, str]      # (obj_name, cell_id)


@dataclass(frozen=True)
class Template:
    """Access-19 instance spec.

    ``source_placements`` and ``goal_placements`` are aligned by
    ``obj_name`` only for the objects present in both — if an object
    appears in ``source`` but not in ``goal``, its final cell is
    unconstrained (the goal-checker will accept any placement).
    """
    name: str
    source_placements: List[_Placement]
    goal_placements: List[_Placement]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants — the 18 cube-column cells, OoI back cells.
# ---------------------------------------------------------------------------

_CUBE_COLS = (1, 3, 5)
_CUBE_ROWS = (0, 1, 2, 3, 4, 5)
_OOI_BACK_CELLS = tuple(
    Cell("shelf_interior", ix, 6).id for ix in _CUBE_COLS
)
_OOI_DEFAULT_GOAL = Cell("shelf_top", 3, 3).id


def _cube_cells() -> List[str]:
    """Every cell in the 18-cell cube mask."""
    return [Cell("shelf_interior", ix, iy).id
                for ix in _CUBE_COLS for iy in _CUBE_ROWS]


def _name_blockers(cells: List[str]) -> List[_Placement]:
    """Assign canonical names ``blocker_0``, ``blocker_1`` ... in the
    order ``cells`` provides."""
    return [(f"blocker_{i}", c) for i, c in enumerate(cells)]


def _placements_to_layout(placements: List[_Placement]) -> Dict[str, str]:
    return dict(placements)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def front_row_subset(
    n: int, rng: np.random.Generator, *,
    return_blockers: bool = False,
    ooi_cell: str = _OOI_BACK_CELLS[1],     # default = (3, 6)
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """Pick ``n`` random cells from the front row (iy=0) of cube cols.

    Mirrors the simplest non-trivial access-19 problem: a few
    blockers blocking the OoI's column-3 row-step path.
    """
    if not 0 <= n <= len(_CUBE_COLS):
        raise ValueError(f"front_row_subset n must be 0..3, got {n}")
    pool = [Cell("shelf_interior", ix, 0).id for ix in _CUBE_COLS]
    chosen = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
    source = _name_blockers([pool[i] for i in chosen])
    source.append(("ooi", ooi_cell))
    if return_blockers:
        goal = list(source)        # blockers return; OoI moves to top
        goal_dict = {o: c for o, c in goal}
        goal_dict["ooi"] = ooi_goal
        goal = list(goal_dict.items())
    else:
        goal = [("ooi", ooi_goal)]
    return Template(
        name=f"front_row_subset_n{n}{'_return' if return_blockers else ''}",
        source_placements=source,
        goal_placements=goal,
        metadata={"n": n, "return": return_blockers, "ooi_cell": ooi_cell},
    )


def dense_front(
    n_rows: int, rng: np.random.Generator, *,
    return_blockers: bool = False,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """Fill the first ``n_rows`` rows of every cube column.

    L1 ≈ dense_front(1), L2 ≈ dense_front(2), L3 ≈ dense_front(3..4),
    L4 = canonical_18 (which is dense_front(6) — handled separately).
    """
    if not 1 <= n_rows <= 6:
        raise ValueError(f"dense_front n_rows must be 1..6, got {n_rows}")
    cells: List[str] = []
    for iy in range(n_rows):
        for ix in _CUBE_COLS:
            cells.append(Cell("shelf_interior", ix, iy).id)
    source = _name_blockers(cells)
    source.append(("ooi", ooi_cell))
    if return_blockers:
        goal_dict = {o: c for o, c in source}
        goal_dict["ooi"] = ooi_goal
        goal = list(goal_dict.items())
    else:
        goal = [("ooi", ooi_goal)]
    return Template(
        name=f"dense_front_rows{n_rows}{'_return' if return_blockers else ''}",
        source_placements=source,
        goal_placements=goal,
        metadata={"n_rows": n_rows, "return": return_blockers,
                       "ooi_cell": ooi_cell},
    )


def scattered_subset(
    n: int, rng: np.random.Generator, *,
    return_blockers: bool = False,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """Random scatter of ``n`` blockers across the 18-cell mask.

    Stress-tests planner generality — the planner can't rely on
    "front-row clears first" heuristics.
    """
    pool = _cube_cells()
    if not 0 <= n <= len(pool):
        raise ValueError(f"scattered_subset n must be 0..{len(pool)}, got {n}")
    chosen_idx = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
    cells = [pool[i] for i in chosen_idx]
    source = _name_blockers(cells)
    source.append(("ooi", ooi_cell))
    if return_blockers:
        goal_dict = {o: c for o, c in source}
        goal_dict["ooi"] = ooi_goal
        goal = list(goal_dict.items())
    else:
        goal = [("ooi", ooi_goal)]
    return Template(
        name=f"scattered_subset_n{n}{'_return' if return_blockers else ''}",
        source_placements=source,
        goal_placements=goal,
        metadata={"n": n, "return": return_blockers, "ooi_cell": ooi_cell},
    )


def canonical_18(
    *,
    return_blockers: bool = False,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """The canonical 18-blocker layout used in the headline eval scenario.

    Set ``return_blockers=True`` for the hardest setting (every blocker
    must end up at its original cell).
    """
    return dense_front(6, np.random.default_rng(),
                              return_blockers=return_blockers,
                              ooi_cell=ooi_cell, ooi_goal=ooi_goal)


def canonical_12(
    *,
    return_blockers: bool = True,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """12-blocker compact subset of ``canonical_18``.

    Used for the **Option-B training distribution** — the trained
    model sees layouts up to 12 blockers, and the held-out evals
    test generalisation to the full 18-blocker layout.  Compact
    dense block (front 4 rows × 3 cube columns) so the structure
    mirrors a clean subset of canonical_18 without adversarial
    placement variation.

    Returns ``dense_front(4)`` under the hood.  Plan length with
    ``return_blockers=True``: ~26-30 actions.
    """
    return dense_front(4, np.random.default_rng(),
                              return_blockers=return_blockers,
                              ooi_cell=ooi_cell, ooi_goal=ooi_goal)


def gotcha_corridor_jam(
    rng: np.random.Generator, *,
    return_blockers: bool = True,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """Hand-crafted layout that FORCES the planner to multi-move the OoI.

    Blockers at the front row of col 3 plus the back row of col 3.
    The OoI's goal (top (3, 3)) is reachable, but returning the back-
    row blockers requires the OoI to leave (3, 3) temporarily.
    """
    cells = [
        Cell("shelf_interior", 3, 0).id,    # blocks col-3 row-step
        Cell("shelf_interior", 3, 4).id,    # back-of-col, needs return
        Cell("shelf_interior", 3, 5).id,    # back-of-col, needs return
    ]
    source = _name_blockers(cells)
    source.append(("ooi", ooi_cell))
    if return_blockers:
        goal_dict = {o: c for o, c in source}
        goal_dict["ooi"] = ooi_goal
        goal = list(goal_dict.items())
    else:
        goal = [("ooi", ooi_goal)]
    return Template(
        name="gotcha_corridor_jam"
                  + ("_return" if return_blockers else ""),
        source_placements=source,
        goal_placements=goal,
        metadata={"return": return_blockers, "ooi_cell": ooi_cell},
    )


# ---------------------------------------------------------------------------
# Symmetry operators
# ---------------------------------------------------------------------------


def _mirror_cell_x(cell_id: str) -> str:
    """Reflect a cell across the col-3 axis (ix → 6 - ix).  No-op for
    cells already at ix=3.  Preserves region + iy."""
    cell = Cell.parse(cell_id)
    new_ix = 6 - cell.ix
    return Cell(cell.region, new_ix, cell.iy).id


def _mirror_cell_iy(cell_id: str, *, max_iy: int = 6) -> str:
    """Reflect a cell across the mid-iy axis (iy → max_iy - iy)."""
    cell = Cell.parse(cell_id)
    new_iy = max_iy - cell.iy
    return Cell(cell.region, cell.ix, new_iy).id


def mirror_x(template: Template) -> Template:
    """Generate the col-3-axis-mirrored variant of a template.

    Source cells with ix=1 ↔ ix=5; the OoI's cell flips too.  Goal
    cells flip in the same way.  Object names are kept (so the same
    blocker_X now sits at the mirrored cell).
    """
    src = [(obj, _mirror_cell_x(cid)) for obj, cid in template.source_placements]
    goal = [(obj, _mirror_cell_x(cid)) for obj, cid in template.goal_placements]
    return Template(
        name=f"{template.name}_mx",
        source_placements=src,
        goal_placements=goal,
        metadata={**template.metadata, "mirrored": "x"},
    )


def mirror_iy(template: Template) -> Template:
    """Generate the iy-mirrored variant (only legal when no source
    cell is at iy=6 — i.e., no OoI at back — since mirror would
    flip it to iy=0, conflicting with a blocker.  Returns ``None``
    when this happens.)"""
    # Compute mirrored OoI cell; reject if it lands on a blocker cell.
    src_dict = dict(template.source_placements)
    mirrored = {obj: _mirror_cell_iy(cid)
                       for obj, cid in template.source_placements}
    if len(set(mirrored.values())) != len(mirrored):
        # Collision after mirroring — drop this variant.
        raise ValueError("mirror_iy: mirrored layout would have duplicates")
    src = list(mirrored.items())
    goal_mirror = {obj: _mirror_cell_iy(cid)
                          if Cell.parse(cid).region == "shelf_interior"
                          else cid
                          for obj, cid in template.goal_placements}
    goal = list(goal_mirror.items())
    return Template(
        name=f"{template.name}_my",
        source_placements=src,
        goal_placements=goal,
        metadata={**template.metadata, "mirrored": "iy"},
    )


# ---------------------------------------------------------------------------
# Helpers exposed for the generator / planner glue.
# ---------------------------------------------------------------------------


def source_layout(template: Template) -> Dict[str, str]:
    """Return the source as a ``{obj: cell_id}`` dict for restore_state."""
    return _placements_to_layout(template.source_placements)


def goal_layout(template: Template) -> Dict[str, str]:
    """Return the goal as a ``{obj: cell_id}`` dict for the planner."""
    return _placements_to_layout(template.goal_placements)
