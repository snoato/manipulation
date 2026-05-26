"""Layout templates for the tabletop_access:access task.

Constructs instances of the HAL `access` problem: an object-of-interest
(OoI) starts somewhere on the shelf with blockers in front of it, and the
goal is to deliver the OoI to a top-deck cell.

Blocker placement uses the verified blocking rule
(``feasibility.structurally_blocks`` = the gripper's swept volume):

* **occluders** sit in the OoI's OWN column, in front of it — a clean
  front-to-back-clearable stack.  (Adjacent-column blockers can't be used
  as occluders: an adjacent cell at the OoI's row mutually-deadlocks with
  the OoI, and at a column-cell's row mutually-deadlocks with it.)
* **clutter** are extra blockers placed where they occlude neither the
  OoI nor any occluder, so they need not be moved — they just enlarge the
  scene (more graph nodes) for the generalization axis.

Train uses 1..3 blockers (all occluders); the OOD eval uses 6 blockers
(occluders capped at the column depth, the rest clutter).  Blockers-only
count; the OoI is extra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace
from tampanda.symbolic.domains.tabletop_access.feasibility import structurally_blocks

_Placement = Tuple[str, str]
_SOURCE_REGIONS = ("middle_deck", "floor_left", "floor_right")
_GOAL_REGION = "top_deck"
_MAX_OCCLUDERS = 3          # >3 deep needs too many reliable top scratch cells


@dataclass(frozen=True)
class Template:
    name: str
    source_placements: List[_Placement]
    goal_placements: List[_Placement]
    metadata: dict = field(default_factory=dict)


def source_layout(t: Template) -> Dict[str, str]:
    return dict(t.source_placements)


def goal_layout(t: Template) -> Dict[str, str]:
    return dict(t.goal_placements)


def goal_cell(workspace: Workspace) -> str:
    top = workspace[_GOAL_REGION]
    return Cell(_GOAL_REGION, top.cells_x // 2, top.cells_y // 2).id


def occluding_cells(workspace: Workspace, ooi_cell_id: str) -> List[str]:
    """OoI's same-column front cells (front-to-back) — the cleanly
    clearable occluder stack.  Restricted to reachable cells."""
    ooi = Cell.parse(ooi_cell_id)
    region = workspace[ooi.region]
    reachable = {(c.ix, c.iy) for c in region.cells()}
    return [Cell(ooi.region, ooi.ix, iy).id
            for iy in range(ooi.iy) if (ooi.ix, iy) in reachable]


def _clutter_cells(workspace: Workspace, ooi_cell_id: str,
                   occ_cells: List[str], n: int,
                   rng: np.random.Generator) -> List[str]:
    """``n`` free cells (across source regions) that occlude neither the
    OoI nor any occluder — safe non-blocking clutter."""
    taken = set(occ_cells) | {ooi_cell_id}
    cands: List[str] = []
    for rname in _SOURCE_REGIONS:
        try:
            region = workspace[rname]
        except KeyError:
            continue
        for c in region.cells():
            if c.id in taken:
                continue
            if structurally_blocks(c.id, ooi_cell_id):
                continue
            if any(structurally_blocks(c.id, oc) for oc in occ_cells):
                continue
            cands.append(c.id)
    idx = rng.permutation(len(cands))[:n].tolist()
    return [cands[i] for i in idx]


def make_instance(
    workspace: Workspace, ooi_cell_id: str, n_occ: int, n_clutter: int,
    *, blocker_names: List[str], rng: np.random.Generator,
    ooi_goal: Optional[str] = None, goal_clutter: bool = False,
) -> Template:
    """OoI + ``n_occ`` occluders + ``n_clutter`` non-blocking clutter, plus
    optionally one blocker sitting AT the OoI's goal cell (target-placement
    clutter — the solver must clear the goal before delivering)."""
    occ = occluding_cells(workspace, ooi_cell_id)
    if len(occ) < n_occ:
        raise ValueError(f"OoI {ooi_cell_id} column has {len(occ)} front cells, "
                         f"need {n_occ}")
    occ_cells = occ[:n_occ]
    goal = ooi_goal or goal_cell(workspace)
    clutter = _clutter_cells(workspace, ooi_cell_id, occ_cells, n_clutter, rng)
    if len(clutter) < n_clutter:
        raise ValueError(f"only {len(clutter)} clutter cells, need {n_clutter}")
    cells = list(occ_cells)
    if goal_clutter:
        cells.append(goal)              # a blocker already occupies the target
    cells += clutter
    if len(blocker_names) < len(cells):
        raise ValueError(f"need {len(cells)} blocker names, got {len(blocker_names)}")
    source = [(blocker_names[i], cells[i]) for i in range(len(cells))]
    source.append(("ooi", ooi_cell_id))
    return Template(
        name=(f"access_k{len(cells)}_occ{n_occ}"
              f"{'_gc' if goal_clutter else ''}_{Cell.parse(ooi_cell_id).region}"),
        source_placements=source,
        goal_placements=[("ooi", goal)],
        metadata={"k": len(cells), "n_occ": n_occ, "n_clutter": n_clutter,
                  "goal_clutter": goal_clutter, "ooi_cell": ooi_cell_id},
    )


def sample_by_counts(workspace: Workspace, rng: np.random.Generator,
                     *, n_occ: int, n_clutter: int, blocker_names: List[str],
                     goal_clutter: bool = False) -> Template:
    """Structured front-blocker instance with explicit occluder/clutter
    counts: a random OoI cell (anywhere with column depth >= n_occ;
    interior columns for wide regions / deep stacks) + occluders + clutter."""
    cands: List[str] = []
    for rname in _SOURCE_REGIONS:
        try:
            region = workspace[rname]
        except KeyError:
            continue
        # Deep occluder stacks only fit reliably in an interior middle column.
        if n_occ >= 4 and rname != "middle_deck":
            continue
        interior_only = region.cells_x >= 4
        for c in region.cells():
            if interior_only and not (1 <= c.ix <= region.cells_x - 2):
                continue
            if c.iy >= n_occ and len(occluding_cells(workspace, c.id)) >= n_occ:
                cands.append(c.id)
    if not cands:
        raise ValueError(f"no source cell supports {n_occ} occluders")
    ooi_cell = cands[int(rng.integers(len(cands)))]
    return make_instance(workspace, ooi_cell, n_occ, n_clutter,
                         blocker_names=blocker_names, rng=rng,
                         goal_clutter=goal_clutter)


def sample(workspace: Workspace, k: int, rng: np.random.Generator,
           *, blocker_names: List[str], goal_clutter: bool = False) -> Template:
    """Convenience: ``k`` total blockers = up to ``_MAX_OCCLUDERS`` occluders,
    the rest clutter (one optionally at the goal)."""
    n_occ = min(k, _MAX_OCCLUDERS)
    rest = k - n_occ
    gc = bool(goal_clutter and rest > 0)
    return sample_by_counts(workspace, rng, n_occ=n_occ,
                            n_clutter=rest - (1 if gc else 0),
                            blocker_names=blocker_names, goal_clutter=gc)


def random_layout(workspace: Workspace, n_blockers: int,
                  rng: np.random.Generator, *, blocker_names: List[str]):
    """Random OoI + ``n_blockers`` blockers (>=1 occluding the OoI), for
    dataset variety.  Returns ``(source_layout, goal)`` dicts.  Solvability
    is decided by the search planner + feasibility (caller resamples)."""
    cells: List[str] = []
    for r in _SOURCE_REGIONS:
        try:
            cells.extend(c.id for c in workspace[r].cells())
        except KeyError:
            continue
    for _ in range(300):
        idx = rng.choice(len(cells), size=n_blockers + 1, replace=False)
        chosen = [cells[i] for i in idx]
        ooi, blk = chosen[0], chosen[1:]
        if any(structurally_blocks(b, ooi) for b in blk):
            layout = {"ooi": ooi}
            layout.update({blocker_names[i]: b for i, b in enumerate(blk)})
            return layout, {"ooi": goal_cell(workspace)}
    raise RuntimeError("no random layout with an occluder found")
