"""Feasibility-guided plan construction for the dense-YCB access task.

Greedy clear-and-deliver, identical in spirit to the parent
``tabletop_access.planner`` but footprint-aware and over two regions
(``middle_deck`` scratch + ``top_deck`` goal):

  while the OoI can't be delivered:
    relocate a pickable blocker whose footprint occludes the OoI pick (or
    sits on / blocks the goal) to a feasibility-VERIFIED scratch anchor
    that re-blocks nothing needed;
  then pick the OoI and put it at the goal anchor.

Every action is FAST-feasible by construction (the oracle is consulted
before committing it).  Returns ``None`` if stuck (caller resamples).

``feasible(layout, held, action) -> bool`` — ``layout`` is
``{obj: anchor_cell_id}``, ``held`` the held obj or None; the caller
restores that symbolic state and runs the FAST chain check.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.tabletop_access_ycb.footprint import ObjectFootprint
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import (
    footprint_blocks_pick, footprint_blocks_put, footprint_overlap,
)

Action = Tuple[str, str, str]
Feasible = Callable[[Dict[str, str], Optional[str], Action], bool]

_RELOCATE_REGIONS = ("middle_deck", "top_deck")  # roomy middle first


def _valid_cells(region) -> set:
    return {(c.ix, c.iy) for c in region.cells()}


def _occupied(layout: Dict[str, str], footprints: Dict[str, ObjectFootprint],
              exclude: Optional[str] = None) -> Dict[str, set]:
    occ: Dict[str, set] = {}
    for o, aid in layout.items():
        if o == exclude:
            continue
        a = Cell.parse(aid)
        for c in footprints[o].cells_at(a):
            occ.setdefault(c.region, set()).add((c.ix, c.iy))
    return occ


def _free_anchors(workspace, fp, layout, footprints, ooi_a, fp_ooi,
                  goal_a, fp_goal, exclude) -> List[Cell]:
    """Anchors where ``fp`` fits, overlaps nothing else, and re-blocks
    neither the OoI pick nor the goal put.  Roomy-region- and front-first."""
    occ = _occupied(layout, footprints, exclude=exclude)
    out: List[Tuple[int, Cell]] = []
    for ri, rname in enumerate(_RELOCATE_REGIONS):
        try:
            region = workspace[rname]
        except KeyError:
            continue
        valid = _valid_cells(region)
        rocc = occ.get(rname, set())
        for ay in range(region.cells_y):
            for ax in range(region.cells_x):
                cells = {(ax + dx, ay + dy) for dx, dy in fp.offsets}
                if not cells <= valid or cells & rocc:
                    continue
                anchor = Cell(rname, ax, ay)
                if footprint_blocks_pick(anchor, fp, ooi_a, fp_ooi):
                    continue
                if (footprint_blocks_put(anchor, fp, goal_a, fp_goal)
                        or footprint_overlap(anchor, fp, goal_a, fp_goal)):
                    continue
                out.append((ri, anchor))
    out.sort(key=lambda ra: (ra[0], ra[1].iy, ra[1].ix))
    return [a for _, a in out]


def _must_move(layout, footprints, ooi_a, fp_ooi, goal_a, fp_goal) -> set:
    """Transitive set of blockers that must move: occluding the OoI pick,
    sitting on / blocking the goal, or occluding a must-move blocker's pick."""
    blockers = {o: Cell.parse(c) for o, c in layout.items() if o != "ooi"}
    way = set()
    for o, a in blockers.items():
        fp = footprints[o]
        if (footprint_blocks_pick(a, fp, ooi_a, fp_ooi)
                or footprint_overlap(a, fp, goal_a, fp_goal)
                or footprint_blocks_put(a, fp, goal_a, fp_goal)):
            way.add(o)
    changed = True
    while changed:
        changed = False
        for o, a in blockers.items():
            if o in way:
                continue
            if any(footprint_blocks_pick(a, footprints[o], blockers[w], footprints[w])
                   for w in way):
                way.add(o)
                changed = True
    return way


def solve(workspace: Workspace, footprints: Dict[str, ObjectFootprint],
          source_layout: Dict[str, str], goal: Dict[str, str],
          feasible: Feasible, *, max_steps: int = 80) -> Optional[List[Action]]:
    ooi_a = Cell.parse(source_layout["ooi"])
    goal_a = Cell.parse(goal["ooi"])
    fp_ooi = footprints["ooi"]
    layout = dict(source_layout)
    plan: List[Action] = []

    for _ in range(max_steps):
        # 1. Deliver if both pick and goal put are feasible.
        if feasible(layout, None, ("pick", "ooi", source_layout["ooi"])):
            rest = {o: c for o, c in layout.items() if o != "ooi"}
            if feasible(rest, "ooi", ("put", "ooi", goal["ooi"])):
                return plan + [("pick", "ooi", source_layout["ooi"]),
                               ("put", "ooi", goal["ooi"])]

        # 2. Relocate a pickable must-move blocker.
        way = _must_move(layout, footprints, ooi_a, fp_ooi, goal_a, fp_ooi)
        if not way:
            return None
        order = sorted(way, key=lambda o: (Cell.parse(layout[o]).iy,
                                           Cell.parse(layout[o]).ix))
        moved = False
        for o in order:
            if not feasible(layout, None, ("pick", o, layout[o])):
                continue
            rest = {oo: cc for oo, cc in layout.items() if oo != o}
            for dst in _free_anchors(workspace, footprints[o], layout, footprints,
                                     ooi_a, fp_ooi, goal_a, fp_ooi, exclude=o):
                if feasible(rest, o, ("put", o, dst.id)):
                    plan += [("pick", o, layout[o]), ("put", o, dst.id)]
                    layout[o] = dst.id
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return None
    return None


def make_fast_oracle(setup):
    """Build ``feasible(layout, held, action)`` over the FAST chain check.

    Restores the canonical symbolic state (footprint occupancy) then runs
    one FAST ``check_action`` — history-independent, so plans built with it
    replay cleanly.
    """
    from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state
    from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import check_action

    object_ids = list(setup.config.object_ids)

    def feasible(layout: Dict[str, str], held: Optional[str], action: Action) -> bool:
        state: Dict[Tuple, bool] = {}
        for o, aid in layout.items():
            a = Cell.parse(aid)
            for c in setup.footprints[o].cells_at(a):
                state[("occupied", c.id, o)] = True
        if held is not None:
            state[("holding", held)] = True
        restore_state(setup.env, setup.workspace, state, object_ids,
                      setup.footprints, executor=setup.executor,
                      home_qpos=setup.home_qpos)
        return check_action(setup.env, setup.workspace, setup.executor,
                            setup.pick_fn, setup.put_fn, setup.footprints,
                            action, fast=True)

    return feasible
