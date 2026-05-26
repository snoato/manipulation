"""Feasibility-guided plan construction for the tabletop_access:access task.

No return-all, so we don't need a full optimal planner — a greedy solver
guided by the verified blocking rule and validated by the FAST feasibility
oracle suffices, and it handles both the structured front-blocker templates
AND random layouts (for dataset variety):

  while the OoI can't be delivered:
    relocate a pickable "in the way" blocker (transitive closure of what
    occludes the OoI) to a feasibility-VERIFIED scratch cell that re-blocks
    nothing needed;
  then pick the OoI and put it at the goal.

Every action is FAST-feasible by construction (the oracle is consulted
before committing it), so the returned plan replays cleanly; ``generate_data``
adds FULL spot-checks.  Returns ``None`` if it gets stuck (caller resamples).

The oracle is ``feasible(layout, held, action) -> bool`` where ``layout`` is
``{obj: cell}`` of placed objects and ``held`` is the held obj or None — the
caller restores that symbolic state and runs the FAST chain check.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from tampanda.symbolic.workspace import Cell, Workspace
from tampanda.symbolic.domains.tabletop_access.feasibility import (
    structurally_blocks, blocks_put,
)

Action = Tuple[str, str, str]
Feasible = Callable[[Dict[str, str], Optional[str], Action], bool]
# Relocation scratch spans ALL regions — the top deck alone is too cramped
# to hold several wide boxes without their put-paths colliding; the middle
# deck (30 cells) gives ample room.  Order: roomy middle first, then floor,
# then top.
_RELOCATE_REGIONS = ("middle_deck", "floor_left", "floor_right", "top_deck")


def _relocation_cells(workspace: Workspace, ooi_cell: str,
                      goal_cell: str) -> List[str]:
    """Free scratch cells across regions that re-block neither the OoI pick
    nor the goal put.  Deepest top-deck row excluded (unreliable put while
    holding); ordered roomy-region- and front-first."""
    ooi = Cell.parse(ooi_cell)
    out: List[Cell] = []
    for ri, rname in enumerate(_RELOCATE_REGIONS):
        try:
            region = workspace[rname]
        except KeyError:
            continue
        back = region.cells_y - 1
        for c in region.cells():
            if c.id in (goal_cell, ooi_cell):
                continue
            if rname == "top_deck" and c.iy >= back:
                continue
            if blocks_put(c.id, goal_cell):
                continue
            if structurally_blocks(c.id, ooi_cell):
                continue
            out.append((ri, c))
    out.sort(key=lambda rc: (rc[0], rc[1].iy, abs(rc[1].ix - ooi.ix) * -1))
    return [c.id for _, c in out]


def _must_move(layout: Dict[str, str], ooi_cell: str, goal_cell: str) -> set:
    """Transitive set of blockers that must move: those occluding the OoI
    pick, those AT or occluding the goal cell (target-placement clutter),
    plus anything occluding a must-move blocker's own pick."""
    blockers = {o: c for o, c in layout.items() if o != "ooi"}
    way = {o for o, c in blockers.items()
           if structurally_blocks(c, ooi_cell)
           or c == goal_cell or structurally_blocks(c, goal_cell)}
    changed = True
    while changed:
        changed = False
        for o, c in blockers.items():
            if o in way:
                continue
            if any(structurally_blocks(c, blockers[w]) for w in way):
                way.add(o)
                changed = True
    return way


def solve(workspace: Workspace, source_layout: Dict[str, str],
          goal: Dict[str, str], feasible: Feasible,
          *, max_steps: int = 60) -> Optional[List[Action]]:
    ooi_cell = source_layout["ooi"]
    goal_cell = goal["ooi"]
    layout = dict(source_layout)
    scratch = _relocation_cells(workspace, ooi_cell, goal_cell)
    plan: List[Action] = []

    for _ in range(max_steps):
        # 1. Deliver the OoI if both pick and the goal put are feasible.
        if feasible(layout, None, ("pick", "ooi", ooi_cell)):
            rest = {o: c for o, c in layout.items() if o != "ooi"}
            if feasible(rest, "ooi", ("put", "ooi", goal_cell)):
                return plan + [("pick", "ooi", ooi_cell),
                               ("put", "ooi", goal_cell)]

        # 2. Otherwise relocate a pickable blocker that's in the way (of the
        #    OoI pick OR the goal put).
        way = _must_move(layout, ooi_cell, goal_cell)
        if not way:
            return None        # OoI unblocked but undeliverable — give up
        order = sorted(way, key=lambda o: (Cell.parse(layout[o]).iy,
                                           Cell.parse(layout[o]).ix))
        occupied = set(layout.values())
        moved = False
        for o in order:
            if not feasible(layout, None, ("pick", o, layout[o])):
                continue
            rest = {oo: cc for oo, cc in layout.items() if oo != o}
            for dst in scratch:
                if dst in occupied or structurally_blocks(dst, ooi_cell):
                    continue
                if feasible(rest, o, ("put", o, dst)):
                    plan += [("pick", o, layout[o]), ("put", o, dst)]
                    layout[o] = dst
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return None        # nothing pickable+relocatable — stuck
    return None
