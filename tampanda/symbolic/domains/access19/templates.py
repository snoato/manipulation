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


def dense_front_n(
    n_blockers: int, rng: np.random.Generator, *,
    return_blockers: bool = False,
    ooi_cell: str = _OOI_BACK_CELLS[1],
    ooi_goal: str = _OOI_DEFAULT_GOAL,
) -> Template:
    """Dense-front pattern with arbitrary blocker count in ``[1, 18]``.

    Fills cube columns row-by-row from the front (iy=0).  Full rows
    take all 3 cube columns; the final (partial) row picks columns in
    the order ``(col_1, col_3, col_5)``.  Counterpart of
    :func:`dense_front` but parameterised by total blocker count
    rather than full-row count — needed for v4's L4 sampling
    ``n_blockers ∈ [10, 14]`` (between canonical_12 and canonical_15).
    """
    if not 1 <= n_blockers <= 18:
        raise ValueError(
            f"dense_front_n n_blockers must be 1..18, got {n_blockers}")
    cells: List[str] = []
    full_rows, partial = divmod(n_blockers, 3)
    for iy in range(full_rows):
        for ix in _CUBE_COLS:
            cells.append(Cell("shelf_interior", ix, iy).id)
    if partial:
        for ix in _CUBE_COLS[:partial]:
            cells.append(Cell("shelf_interior", ix, full_rows).id)
    source = _name_blockers(cells)
    source.append(("ooi", ooi_cell))
    if return_blockers:
        goal_dict = {o: c for o, c in source}
        goal_dict["ooi"] = ooi_goal
        goal = list(goal_dict.items())
    else:
        goal = [("ooi", ooi_goal)]
    return Template(
        name=f"dense_front_n{n_blockers}"
                  f"{'_return' if return_blockers else ''}",
        source_placements=source,
        goal_placements=goal,
        metadata={"n_blockers": n_blockers, "return": return_blockers,
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


def easy_l0(rng: np.random.Generator) -> Template:
    """Simple L0 problem with varied OoI placement.

    Diversifies the L0 distribution beyond the back-row-OoI pattern
    in :func:`front_row_subset`.  Randomly samples:

    * **OoI start**: interior (cube column ``ix ∈ {1,3,5}``, any
      ``iy``) OR top deck (any ``ix, iy``).  50/50.
    * **OoI goal**: random deck cell distinct from start.
    * **Blocker** (50/50): a single non-blocking blocker at a random
      valid cell — if interior, *not* in front of the OoI's column;
      if on deck, not at the OoI or goal cell.

    Plan length: 2 actions (pick + put) when no blocker or
    non-blocking blocker; we deliberately reject configurations that
    would require clearing.
    """
    interior_cells = [Cell("shelf_interior", ix, iy).id
                          for ix in _CUBE_COLS for iy in range(7)]
    deck_cells = [Cell("shelf_top", ix, iy).id
                      for ix in range(7) for iy in range(7)]

    # OoI start.
    ooi_interior = bool(rng.integers(0, 2))
    if ooi_interior:
        ooi_cell = str(rng.choice(interior_cells))
        ooi_col, ooi_row = (int(t) for t in ooi_cell.split("__")[1].split("_"))
    else:
        ooi_cell = str(rng.choice(deck_cells))
        ooi_col = ooi_row = None

    # Goal: random deck cell distinct from start.
    goal_cell = ooi_cell
    while goal_cell == ooi_cell:
        goal_cell = str(rng.choice(deck_cells))

    source_placements: List[_Placement] = []
    has_blocker = bool(rng.integers(0, 2))
    if has_blocker:
        # Try to find a non-blocking blocker position.
        for _attempt in range(30):
            if bool(rng.integers(0, 2)):
                bcell = str(rng.choice(interior_cells))
                bix, biy = (int(t) for t in bcell.split("__")[1].split("_"))
                if ooi_interior and bix == ooi_col and biy < ooi_row:
                    continue  # would block OoI's row-step approach
                if bcell == ooi_cell:
                    continue
            else:
                bcell = str(rng.choice(deck_cells))
                if bcell in (ooi_cell, goal_cell):
                    continue
            source_placements.append(("blocker_0", bcell))
            break

    source_placements.append(("ooi", ooi_cell))

    return Template(
        name="easy_l0",
        source_placements=source_placements,
        goal_placements=[("ooi", goal_cell)],
        metadata={
            "ooi_start": "interior" if ooi_interior else "deck",
            "has_blocker": len(source_placements) > 1,
        },
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


def permute_blockers(
    template: "Template",
    plan: List[Tuple],
    perm: Dict[str, str],
) -> Tuple["Template", List[Tuple]]:
    """Apply a blocker-label permutation to a template + plan.

    ``perm`` maps ``"blocker_X" → "blocker_Y"``.  Objects not present
    in ``perm`` (e.g., ``"ooi"``) pass through unchanged.

    Why useful: the planner's plan is permutation-invariant on blocker
    identities — relabelling blocker_3 ↔ blocker_7 in the init,
    goal, AND every plan action gives a structurally-identical valid
    problem.  v4.5's curated generation runs the planner once per
    base layout and produces many training instances via this cheap
    relabel.  GNN sees varied label patterns over the same underlying
    structure.
    """
    def remap(obj):
        return perm.get(obj, obj)
    new_source = [(remap(o), c) for o, c in template.source_placements]
    new_goal = [(remap(o), c) for o, c in template.goal_placements]
    new_plan: List[Tuple] = []
    for action in plan:
        verb, obj, *rest = action
        new_plan.append((verb, remap(obj), *rest))
    new_tpl = Template(
        name=f"{template.name}_perm",
        source_placements=new_source,
        goal_placements=new_goal,
        metadata={**template.metadata, "permuted": True},
    )
    return new_tpl, new_plan


def mirror_plan_x(plan: List[Tuple]) -> List[Tuple]:
    """Apply ``_mirror_cell_x`` to each plan action's cell.

    Companion to :func:`mirror_x`: the template's source/goal cells
    flip; the plan must flip with them so each action targets the
    mirrored cell.
    """
    out: List[Tuple] = []
    for action in plan:
        verb, obj, cell, *rest = action
        out.append((verb, obj, _mirror_cell_x(cell), *rest))
    return out


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
