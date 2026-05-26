"""Feasibility-aware rearrangement search for confined_shelf.

Transforms an initial cylinder arrangement into a goal arrangement using
``move(cyl, to_cell)`` = ``pick(cyl) + put(cyl, to_cell)`` macro-steps,
where each move must satisfy the empirically-grounded FRONT-grasp
blocking model (front-column occlusion + lateral-neighbour clearance +
cell reachability — see ``feasibility.prefilter_reject``).  This is the
symbolic, sim-free model; generated reference plans are re-validated in
sim by ``feasibility.check_action_sequence``.

Two solvers, mirroring Wang ICAPS-2022:

* :func:`solve_monotone` — DFS that moves each cylinder **at most once**,
  directly to its goal (the paper's monotone solver).  Returns a plan iff
  the instance is monotone.
* :func:`solve_with_buffers` — best-first search that additionally allows
  relocating a blocking cylinder to a buffer cell (the paper's
  perturbation).  Used when no monotone plan exists.

:func:`solve` tries monotone first, then buffers, and classifies the
result.  ``RearrangePlan.monotone`` is the train/test difficulty axis.

PDDL has NO knowledge of this blocking — it only enforces occupied/empty
pick/put preconditions.  All reachability reasoning lives here.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tampanda.symbolic.workspace import Cell, GridRegion

Arr = Dict[str, Tuple[int, int]]            # {cyl: (ix, iy)}
Move = Tuple[str, Tuple[int, int], Tuple[int, int]]   # (cyl, from, to)


# ---------------------------------------------------------------------------
# Blocking model (sim-free; matches feasibility.prefilter_reject)
# ---------------------------------------------------------------------------


def _reachable(region: GridRegion, ix: int, iy: int) -> bool:
    return (0 <= ix < region.cells_x and 0 <= iy < region.cells_y
            and (ix, iy) not in region.excluded_cells)


def column_clear_reason(occupied: Set[Tuple[int, int]], ix: int, iy: int,
                        region: GridRegion) -> Optional[str]:
    """Why a FRONT grasp/placement at ``(ix, iy)`` is blocked, or ``None`` if
    clear.  ``occupied`` must EXCLUDE the object at ``(ix, iy)`` itself.

    Single source of the cell-occupancy blocking model — shared by the
    search (``pick_feasible`` / ``put_feasible``) and the feasibility
    quick-reject (``feasibility.prefilter_reject``) so they can't drift.
    """
    if not _reachable(region, ix, iy):
        return "cell-unreachable"
    for iy_prime in range(iy):                       # front-column occlusion
        if (ix, iy_prime) in occupied:
            return "front-blocked"
    # Adjacent-column blocking (confirmed against the FULL executor, see
    # examples/cs_rule_reconcile.py).  A cylinder in a directly-adjacent
    # column (ix±1) at the target's row OR anywhere in front of it (by<=iy)
    # blocks the FRONT reach: same row = the gripper jaw hits it; in front =
    # the arm sweeps past it on the way deep.  So a single cylinder at (0,0)
    # blocks ALL of the adjacent column 1 (the gripper/arm can't pass it),
    # while distance-2 columns like (3,0) stay clear.  A buffer parked at a
    # front cell is therefore allowed only while its neighbours are empty,
    # and it blocks the deep cells behind its neighbours — the search must
    # relocate it (backtrack) to reach them.
    for bx in (ix - 1, ix + 1):
        for by in range(iy + 1):
            if (bx, by) in occupied:
                return "lateral-blocked"
    return None


def column_clear(occupied: Set[Tuple[int, int]], ix: int, iy: int,
                 region: GridRegion) -> bool:
    return column_clear_reason(occupied, ix, iy, region) is None


def pick_feasible(arr: Arr, cyl: str, region: GridRegion) -> bool:
    """Can the FRONT gripper reach ``cyl`` at its current cell?"""
    ix, iy = arr[cyl]
    return column_clear(set(arr.values()) - {(ix, iy)}, ix, iy, region)


def put_feasible(arr: Arr, cyl: str, to: Tuple[int, int],
                 region: GridRegion) -> bool:
    """Can ``cyl`` (currently held, i.e. removed from ``arr``) be placed at
    ``to``?  ``to`` must be empty and front/laterally clear."""
    occ = set(arr.values()) - {arr.get(cyl, (-99, -99))}
    if to in occ:
        return False
    return column_clear(occ, to[0], to[1], region)


def move_feasible(arr: Arr, cyl: str, to: Tuple[int, int],
                  region: GridRegion) -> bool:
    return (pick_feasible(arr, cyl, region)
            and put_feasible(arr, cyl, to, region))


def _free_cells(arr: Arr, region: GridRegion) -> List[Tuple[int, int]]:
    occ = set(arr.values())
    return [(ix, iy) for ix in range(region.cells_x)
            for iy in range(region.cells_y)
            if (ix, iy) not in occ and _reachable(region, ix, iy)]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RearrangePlan:
    moves: List[Move]                 # (cyl, from_cell, to_cell) macro-moves
    monotone: bool
    n_relocations: int = 0            # moves beyond one-per-cylinder (buffers)
    nodes: int = 0

    def to_actions(self, region_name: str) -> List[Tuple]:
        """Flatten to PDDL ``("pick"|"put", cyl, cell_id)`` actions."""
        out: List[Tuple] = []
        for cyl, frm, to in self.moves:
            out.append(("pick", cyl, Cell(region_name, frm[0], frm[1]).id))
            out.append(("put", cyl, Cell(region_name, to[0], to[1]).id))
        return out


# ---------------------------------------------------------------------------
# Monotone solver — DFS, each cylinder moved at most once to its goal
# ---------------------------------------------------------------------------


def solve_monotone(init: Arr, goal: Arr, region: GridRegion,
                   *, node_budget: int = 50000) -> Optional[List[Move]]:
    seen: Set[frozenset] = set()
    nodes = [0]

    def dfs(arr: Arr) -> Optional[List[Move]]:
        nodes[0] += 1
        if nodes[0] > node_budget:
            return None
        if all(arr.get(c) == g for c, g in goal.items()):
            return []
        key = frozenset(arr.items())
        if key in seen:
            return None
        seen.add(key)
        occ = set(arr.values())
        for c, gcell in goal.items():
            if arr.get(c) == gcell:
                continue
            if gcell in occ:                      # goal cell still occupied
                continue
            if move_feasible(arr, c, gcell, region):
                arr2 = dict(arr)
                arr2[c] = gcell
                rest = dfs(arr2)
                if rest is not None:
                    return [(c, arr[c], gcell)] + rest
        return None

    plan = dfs(dict(init))
    return plan


# ---------------------------------------------------------------------------
# Buffer search — best-first, allows relocating blockers to buffer cells
# ---------------------------------------------------------------------------


def _blocking_candidates(arr: Arr, goal: Arr, region: GridRegion) -> List[str]:
    """Cylinders worth relocating: those not at goal, or sitting on another
    cylinder's goal cell."""
    goal_cells = set(goal.values())
    out = []
    for c in arr:
        if c not in goal:
            continue
        if arr[c] != goal[c]:
            out.append(c)
        elif arr[c] in goal_cells and any(
                gc == arr[c] and oc != c for oc, gc in goal.items()):
            out.append(c)
    return out


def solve_with_buffers(init: Arr, goal: Arr, region: GridRegion,
                       *, node_budget: int = 200000
                       ) -> Optional[List[Move]]:
    def h(arr: Arr) -> int:
        return sum(1 for c, g in goal.items() if arr.get(c) != g)

    start = dict(init)
    pq: List = [(h(start), 0, 0, start, [])]
    best_g: Dict[frozenset, int] = {frozenset(start.items()): 0}
    counter = 0
    nodes = 0

    while pq and nodes < node_budget:
        f, g, _, arr, plan = heapq.heappop(pq)
        nodes += 1
        if h(arr) == 0:
            for m in plan:
                pass
            return plan
        occ = set(arr.values())
        succ: List[Move] = []
        # 1) productive: move a misplaced cylinder straight to its goal.
        for c, gcell in goal.items():
            if arr.get(c) == gcell:
                continue
            if gcell not in occ and move_feasible(arr, c, gcell, region):
                succ.append((c, arr[c], gcell))
        # 2) buffer relocations for blocking cylinders.
        free = _free_cells(arr, region)
        for c in _blocking_candidates(arr, goal, region):
            for cell in free:
                if cell == goal.get(c):
                    continue                      # handled as productive move
                if move_feasible(arr, c, cell, region):
                    succ.append((c, arr[c], cell))
        for c, frm, to in succ:
            arr2 = dict(arr)
            arr2[c] = to
            key = frozenset(arr2.items())
            ng = g + 1
            if key in best_g and best_g[key] <= ng:
                continue
            best_g[key] = ng
            counter += 1
            heapq.heappush(pq, (ng + h(arr2), ng, counter, arr2, plan + [(c, frm, to)]))
    return None


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def solve(init: Arr, goal: Arr, region: GridRegion, *,
          mono_budget: int = 50000,
          buffer_budget: int = 200000) -> Optional[RearrangePlan]:
    """Return a verified rearrangement plan, or ``None`` if unsolvable
    within the search budgets.  Tries monotone first, then buffers."""
    n_misplaced = sum(1 for c, g in goal.items() if init.get(c) != g)
    mono = solve_monotone(init, goal, region, node_budget=mono_budget)
    if mono is not None:
        return RearrangePlan(moves=mono, monotone=True, n_relocations=0)
    buf = solve_with_buffers(init, goal, region, node_budget=buffer_budget)
    if buf is not None:
        return RearrangePlan(moves=buf, monotone=False,
                             n_relocations=len(buf) - n_misplaced)
    return None
