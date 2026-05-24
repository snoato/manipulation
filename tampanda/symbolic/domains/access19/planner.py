"""Plan generators for access-19.

Two entry points:

* :func:`oracle_plan` — greedy oracle that places blockers on the top
  deck (back-to-front, col-aligned) and ends with the OoI on the
  goal cell.  Fast (~1–3 s for L0–L4) but only supports goals of the
  form "OoI at top-deck goal cell; blockers wherever they ended up".

* :func:`astar_plan` — A* over symbolic states with the FAST feasibility
  check as the transition validator.  Supports arbitrary
  ``goal_layout`` (e.g., OoI on top AND every blocker back at its
  original interior cell).  Naturally handles multi-move OoI
  rearrangements: the heuristic allows the search to move the OoI to
  a temporary cell, return blockers, then move the OoI to its final
  cell.

Both emit pure PDDL action sequences — every action in the returned
plan succeeded under FAST feasibility (and FAST↔FULL agreement is
validated separately by ``examples/access19_feasibility_smoke.py``).

Strategy:

1. **Greedy with priorities.** At each step, try to commit a
   (pick blocker, put blocker on top deck) pair using the strongest
   heuristic order:

   * **Pick** front-to-back, cols 1 → 3 → 5, picking any blocker
     still in the cubicle.
   * **Put** column-aligned (same ix as pick), iy_top = 6 − iy_int
     (back-to-front on the top deck — proven safe in Phase 0).

2. **Per-action FAST check.** Every candidate (pick, put) pair is
   validated against the current state via :func:`check_action_sequence`.
   The state restore + sequence check is deterministic, so a positive
   result implies the FULL executor will accept the pair too (FAST↔
   FULL agreement validated by ``examples/access19_feasibility_smoke.py``).

3. **Fallback search over put cells.** If the column-aligned put is
   infeasible (e.g. that cell is already occupied or its put_deck
   traverse crosses a blocker), the planner tries other top-deck
   cells in priority order (further-back iy, then off-axis columns).

4. **Fallback search over picks.** If a chosen pick has no feasible
   put, try the next-priority pick.

5. **Backtracking on dead-end.** If no (pick, put) pair is feasible
   from the current state, undo the last committed pair and search
   alternatives at that level.  Bounded by ``max_backtrack``.

6. **Terminate.** Stop when an OoI ``[pick, put_at_goal]`` sequence
   is feasible — append it and return.

For small N (≤3 blockers) the greedy path almost always works
first-try; backtracking exists mainly to handle dense layouts where
the column-aligned dump conflicts with an existing top-deck cube.

Output::

    OracleResult(
        plan=[("pick", "blocker_0", "shelf_interior__1_0"),
              ("put",  "blocker_0", "shelf_top__1_6"),
              ...,
              ("pick", "ooi", "shelf_interior__3_6"),
              ("put",  "ooi", "shelf_top__3_3")],
        success=True,
        n_feasibility_checks=...,
        elapsed_s=...,
    )
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.access19.env_builder import Access19Config
from tampanda.symbolic.domains.access19.feasibility import (
    check_action_sequence,
)


@dataclass
class OracleResult:
    """Output of :func:`oracle_plan`."""
    plan: List[Tuple]
    success: bool
    n_feasibility_checks: int = 0
    elapsed_s: float = 0.0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------


def _pick_candidates(
    layout: Dict[str, str],
    object_names: List[str],
) -> List[Tuple[str, Cell]]:
    """List (obj, cell) for every object still in shelf_interior, in
    priority order: ascending iy (front-to-back), then ix (1, 3, 5)."""
    out: List[Tuple[str, Cell]] = []
    for obj in object_names:
        if obj == "ooi":
            continue
        cell_id = layout.get(obj)
        if cell_id is None:
            continue
        cell = Cell.parse(cell_id)
        if cell.region != "shelf_interior":
            continue
        out.append((obj, cell))
    out.sort(key=lambda oc: (oc[1].iy, oc[1].ix))
    return out


def _put_candidates(
    workspace: Workspace,
    pick_cell: Cell,
    layout: Dict[str, str],
    *,
    avoid_cells: Optional[List[Cell]] = None,
) -> List[Cell]:
    """List of top-deck cells in priority order for putting a blocker.

    Priority:
      1. Column-aligned, ``iy_top = 6 - pick_cell.iy`` (proven safe
         back-to-front order).
      2. Same column, other iy_top in descending order (deeper rows
         first).
      3. Off-axis cube columns in descending iy_top order.
      4. Empty (non-cube) columns last — these are the OoI-style cells.
    """
    avoid_cells = set(c.id for c in (avoid_cells or []))
    occupied_top = {layout[o] for o in layout
                          if layout[o].startswith("shelf_top__")}
    region = workspace["shelf_top"]
    cells_x = region.cells_x
    cells_y = region.cells_y

    def _is_free(ix: int, iy: int) -> bool:
        cell = Cell("shelf_top", ix, iy)
        if cell.id in occupied_top or cell.id in avoid_cells:
            return False
        return True

    candidates: List[Cell] = []
    cube_cols = (1, 3, 5)

    # Tier 1: col-aligned, B2F-canonical cell.
    primary_iy = 6 - pick_cell.iy
    if 0 <= primary_iy < cells_y and _is_free(pick_cell.ix, primary_iy):
        candidates.append(Cell("shelf_top", pick_cell.ix, primary_iy))

    # Tier 2: same column, other iy in descending order (deeper first).
    for iy in range(cells_y - 1, -1, -1):
        if iy == primary_iy:
            continue
        if _is_free(pick_cell.ix, iy):
            candidates.append(Cell("shelf_top", pick_cell.ix, iy))

    # Tier 3: other cube columns, descending iy.
    for ix in cube_cols:
        if ix == pick_cell.ix:
            continue
        for iy in range(cells_y - 1, -1, -1):
            if _is_free(ix, iy):
                candidates.append(Cell("shelf_top", ix, iy))

    # Tier 4: empty columns (0, 2, 4, 6).
    for ix in (0, 2, 4, 6):
        if ix >= cells_x:
            continue
        for iy in range(cells_y - 1, -1, -1):
            if _is_free(ix, iy):
                candidates.append(Cell("shelf_top", ix, iy))

    return candidates


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def oracle_plan(
    env,
    workspace: Workspace,
    config: Access19Config,
    initial_layout: Dict[str, str],
    goal_cell: Cell,
    object_names: List[str],
    pick_fn: Callable,
    put_fn: Callable,
    *,
    executor,
    home_qpos: np.ndarray,
    fast: bool = True,
    max_actions: int = 200,
    max_backtrack: int = 32,
    verbose: bool = False,
) -> OracleResult:
    """Greedy oracle with bounded backtracking.

    Args:
        initial_layout: ``{obj: cell_id}`` placement.
        goal_cell: where the OoI must end up.
        object_names: full roster.
        pick_fn / put_fn / executor / home_qpos: as wired by
            :func:`make_access19_pick_fn` / :func:`make_access19_put_fn`.
        fast: use FAST feasibility check (recommended).
        max_actions: hard cap on plan length.
        max_backtrack: total backtrack attempts allowed before failing.

    Returns:
        :class:`OracleResult` with plan + diagnostics.
    """
    t_start = time.perf_counter()
    plan: List[Tuple] = []
    notes: List[str] = []

    # Mutable layout used for state tracking; do not modify caller's dict.
    layout: Dict[str, str] = dict(initial_layout)
    # Stack of decision points for backtracking.
    # Each entry: (layout_before, plan_before, exhausted_pick_obj_ids).
    backtrack_stack: List[Tuple[Dict[str, str], List[Tuple], set]] = []
    n_checks = 0
    n_backtracks = 0

    def _check_seq(layout_snapshot, actions):
        nonlocal n_checks
        from tampanda.symbolic.domains.access19.parallel import _layout_to_state
        state = _layout_to_state(layout_snapshot, held=None)
        n_checks += 1
        return check_action_sequence(
            env, workspace, config, state, list(actions),
            object_names, pick_fn, put_fn,
            executor=executor, fast=fast, home_qpos=home_qpos,
        )

    def _try_terminate() -> bool:
        """If OoI is reachable from current layout, append the OoI
        [pick, put_at_goal] pair to the plan and return True."""
        ooi_cell = layout.get("ooi")
        if ooi_cell is None:
            return False
        seq = [
            ("pick", "ooi", ooi_cell),
            ("put", "ooi", goal_cell.id),
        ]
        res = _check_seq(layout, seq)
        if res["success"]:
            plan.extend(seq)
            layout["ooi"] = goal_cell.id
            return True
        return False

    while len(plan) < max_actions:
        # Termination check.
        if _try_terminate():
            return OracleResult(
                plan=plan, success=True,
                n_feasibility_checks=n_checks,
                elapsed_s=time.perf_counter() - t_start,
                notes=notes,
            )

        # Choose a (pick, put) pair.
        picks = _pick_candidates(layout, object_names)
        # Pull exhausted set from the most recent backtrack frame if any.
        exhausted = (backtrack_stack[-1][2] if backtrack_stack else set())
        # Reserve the OoI's terminal-put corridor: the put_deck Cartesian
        # traverse from staging to (goal.x, goal.y) at safe_z is a
        # straight line in xy; any blocker on the deck whose footprint
        # the held OoI crosses on the way will collide.  For the
        # canonical staging at col_3 and a same-column goal, this is
        # every same-column cell with iy_top <= goal.iy.  For different-
        # column goals, the corridor is wider but the conservative
        # approach is the same: never put a blocker between iy_top=0
        # and the goal in the goal's column.
        corridor: List[Cell] = [
            Cell("shelf_top", goal_cell.ix, iy)
            for iy in range(0, goal_cell.iy + 1)
        ]

        committed = False
        for obj, src_cell in picks:
            if obj in exhausted:
                continue
            puts = _put_candidates(workspace, src_cell, layout,
                                          avoid_cells=corridor)
            for dst_cell in puts:
                seq = [("pick", obj, src_cell.id),
                          ("put", obj, dst_cell.id)]
                res = _check_seq(layout, seq)
                if res["success"]:
                    plan.extend(seq)
                    layout[obj] = dst_cell.id
                    # Open a fresh backtrack frame after a commit.
                    backtrack_stack.append((dict(layout), list(plan), set()))
                    if verbose:
                        notes.append(f"commit: pick {obj}@{src_cell.id} → "
                                        f"put {obj}@{dst_cell.id} "
                                        f"({res['elapsed_s']*1000:.0f} ms)")
                    committed = True
                    break
            if committed:
                break
            # All puts for this pick failed → mark exhausted.
            exhausted.add(obj)
        if committed:
            continue

        # Dead-end: backtrack.
        if n_backtracks >= max_backtrack or not backtrack_stack:
            notes.append(f"DEAD-END: no feasible pick/put after "
                            f"{n_backtracks}/{max_backtrack} backtracks")
            return OracleResult(
                plan=plan, success=False,
                n_feasibility_checks=n_checks,
                elapsed_s=time.perf_counter() - t_start,
                notes=notes,
            )
        n_backtracks += 1
        prev_layout, prev_plan, _ = backtrack_stack.pop()
        layout = dict(prev_layout)
        plan = list(prev_plan)
        # Mark the just-committed pick as exhausted in the PARENT frame.
        if plan:
            last_pick = plan[-2]    # ("pick", obj, src_id)
            obj_last = last_pick[1]
            if backtrack_stack:
                backtrack_stack[-1][2].add(obj_last)
            # Undo the last commit's pick from the layout's exhausted view
            # by also removing the last pair.
            plan = plan[:-2]
            # Recover obj's previous cell by scanning earlier plan steps.
            recovered = initial_layout.get(obj_last)
            for act in plan:
                if act[0] == "put" and act[1] == obj_last:
                    recovered = act[2]
            layout[obj_last] = recovered
        if verbose:
            notes.append(f"backtrack #{n_backtracks}")

    notes.append(f"max_actions ({max_actions}) reached without goal")
    return OracleResult(
        plan=plan, success=False,
        n_feasibility_checks=n_checks,
        elapsed_s=time.perf_counter() - t_start,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# A* planner over symbolic states
# ---------------------------------------------------------------------------


def _state_key(layout: Dict[str, str], held: Optional[str]) -> Tuple:
    """Hashable state key: (sorted_layout_tuple, held)."""
    return (tuple(sorted(layout.items())), held)


def _h_misplaced(
    layout: Dict[str, str], held: Optional[str],
    goal_layout: Dict[str, str],
) -> int:
    """Admissible heuristic, tuned for access-19's geometry.

    Components:
      * ``2 × misplaced`` — every off-goal object needs at least
        one pick + one put.
      * ``+1 if held`` — one put remains for the held object.
      * ``+4 if blockers in OoI's path corridor`` — for every blocker
        currently parked in the goal-cell's column at iy_top > goal.iy
        whose goal is NOT that cell, the OoI must be temporarily
        moved off its goal cell and then back (cost = 2 pick+put = 4).
        Adds the overhead exactly once regardless of how many such
        blockers exist (one OoI shuffle covers them all).

    All terms are lower bounds on the true cost, so the heuristic
    remains admissible.
    """
    misplaced = sum(
        1 for obj, goal_cell in goal_layout.items()
        if layout.get(obj) != goal_cell
    )
    held_penalty = 1 if held is not None else 0

    ooi_goal_id = goal_layout.get("ooi")
    ooi_overhead = 0
    if ooi_goal_id is not None:
        goal_cell = Cell.parse(ooi_goal_id)
        if layout.get("ooi") == ooi_goal_id:
            # OoI sits at goal — any blocker that needs to return AND
            # is currently in the goal column at iy_top > goal.iy must
            # cross the OoI on its pick_deck traverse, forcing an
            # OoI shuffle (4 extra actions).
            for obj, current_cell in layout.items():
                if obj == "ooi" or current_cell is None:
                    continue
                obj_goal = goal_layout.get(obj)
                if obj_goal == current_cell:
                    continue
                cur = Cell.parse(current_cell)
                if (cur.region == "shelf_top"
                        and cur.ix == goal_cell.ix
                        and cur.iy > goal_cell.iy):
                    ooi_overhead = 4
                    break

    return 2 * misplaced + held_penalty + ooi_overhead


def _staging_cells(
    workspace: Workspace, held: str,
    initial_layout: Dict[str, str],
) -> List[str]:
    """Tiny set of staging cells for the held object — ≤ 4 per held
    state to keep A*'s branching manageable.

    OoI: 2 top-deck corners (out of the way of every cube col).
    Blocker: the column-aligned back-to-front dump cell, computed from
    the blocker's INITIAL interior cell.
    """
    if held == "ooi":
        return [
            Cell("shelf_top", 0, 6).id,
            Cell("shelf_top", 6, 6).id,
        ]
    init_cell_id = initial_layout.get(held)
    if init_cell_id is None:
        return []
    init_cell = Cell.parse(init_cell_id)
    if init_cell.region != "shelf_interior":
        return []
    # Canonical safe dump for a blocker picked from (ix, iy_int) is
    # (ix, 6 - iy_int) on the top deck (back-to-front order).
    canonical = Cell("shelf_top", init_cell.ix, 6 - init_cell.iy).id
    return [canonical]


def _generate_successor_actions(
    layout: Dict[str, str], held: Optional[str],
    workspace: Workspace, goal_layout: Dict[str, str],
    object_names: List[str],
    initial_layout: Dict[str, str],
) -> List[Tuple]:
    """Enumerate PDDL actions, ordered with goal-achieving actions first
    and aggressively pruned to access-19's structural priors.

    PICK order:
      1. Objects blocking some goal cell (occupant != intended goal-owner).
      2. Other misplaced objects (front-to-back if in interior).
      3. Correctly-placed objects (only relevant for "move out of the way"
         scenarios; OoI is last so it doesn't churn between cells).

    PUT order (for the held object):
      1. Object's goal cell (if free).
      2. Object's INITIAL cell (if different from goal and free) — useful
         when an object was temporarily relocated to make room.
      3. A small set of staging cells (top-deck cube cols for blockers,
         top-deck corners for OoI).
      4. Any other empty cell in the object's goal region.
    """
    if held is None:
        # PICK.
        occupied = set(layout.values())
        goal_owners = {cell: obj for obj, cell in goal_layout.items()}

        def _priority(obj: str) -> Tuple:
            current_cell = layout[obj]
            goal_cell = goal_layout.get(obj)
            is_blocking = (current_cell in goal_owners
                                and goal_owners[current_cell] != obj)
            misplaced = (goal_cell is not None and goal_cell != current_cell)
            iy_score = Cell.parse(current_cell).iy
            return (
                0 if is_blocking else 1,
                0 if misplaced else 1,
                0 if obj != "ooi" else 1,
                iy_score,
                obj,
            )

        ordered_objs = sorted(layout.keys(), key=_priority)
        return [("pick", obj, layout[obj]) for obj in ordered_objs]

    # PUT for held object.
    obj = held
    goal_cell_id = goal_layout.get(obj)
    initial_cell_id = initial_layout.get(obj)
    occupied = set(layout.values())
    seen: set = set()
    cells: List[str] = []

    def _try(cell_id: Optional[str]) -> None:
        if cell_id is None or cell_id in occupied or cell_id in seen:
            return
        seen.add(cell_id)
        cells.append(cell_id)

    _try(goal_cell_id)
    if initial_cell_id != goal_cell_id:
        _try(initial_cell_id)
    for staging in _staging_cells(workspace, obj, initial_layout):
        _try(staging)

    return [("put", obj, cid) for cid in cells]


def astar_plan(
    env,
    workspace: Workspace,
    config: Access19Config,
    initial_layout: Dict[str, str],
    goal_layout: Dict[str, str],
    object_names: List[str],
    pick_fn: Callable,
    put_fn: Callable,
    *,
    executor,
    home_qpos: np.ndarray,
    fast: bool = True,
    max_states: int = 20000,
    max_actions_per_state: int = 64,
    time_budget_s: float = 120.0,
    verbose: bool = False,
) -> OracleResult:
    """A* search over symbolic states.

    ``goal_layout`` specifies the target cell for every object that
    must end up somewhere specific.  Objects absent from
    ``goal_layout`` are unconstrained (the planner won't restrict
    their final cell, but they still need to be in some valid cell).

    The transition function uses FAST feasibility checks via
    :func:`check_action_sequence` on a length-1 action list.  Each
    expanded node restores the symbolic state in MuJoCo, validates
    the action, and stores the new state in the priority queue.

    Args:
        max_states: cap on expanded states.
        max_actions_per_state: per-node action ordering caps action
            list at this many candidates (defends against huge
            branching for held-state successors with empty cells).
        time_budget_s: wall-clock cap.

    Returns:
        :class:`OracleResult` with the action sequence and stats.
    """
    import heapq
    from tampanda.symbolic.domains.access19.parallel import _layout_to_state

    t_start = time.perf_counter()
    notes: List[str] = []
    n_checks = 0
    n_expanded = 0

    init_state = (dict(initial_layout), None)
    init_key = _state_key(*init_state)
    init_h = _h_misplaced(*init_state, goal_layout=goal_layout)
    # Frontier: (f, tiebreak_seq, state_key, layout, held, plan).
    seq = 0
    frontier: List = []
    heapq.heappush(frontier, (init_h, seq, init_key,
                                  init_state[0], init_state[1], []))
    closed: Dict[Tuple, int] = {init_key: 0}     # state → g

    def _is_goal(layout: Dict[str, str], held: Optional[str]) -> bool:
        if held is not None:
            return False
        for obj, goal_cell in goal_layout.items():
            if layout.get(obj) != goal_cell:
                return False
        return True

    while frontier:
        if time.perf_counter() - t_start > time_budget_s:
            notes.append(f"TIME BUDGET: {time_budget_s:.1f}s exhausted "
                            f"({n_expanded} expanded, {n_checks} checks)")
            return OracleResult(plan=[], success=False,
                                       n_feasibility_checks=n_checks,
                                       elapsed_s=time.perf_counter() - t_start,
                                       notes=notes)
        if n_expanded >= max_states:
            notes.append(f"MAX_STATES: {max_states} expanded "
                            f"({n_checks} checks)")
            return OracleResult(plan=[], success=False,
                                       n_feasibility_checks=n_checks,
                                       elapsed_s=time.perf_counter() - t_start,
                                       notes=notes)

        f, _, key, layout, held, plan = heapq.heappop(frontier)
        if closed.get(key, float("inf")) < len(plan):
            continue        # better path already found
        n_expanded += 1

        if _is_goal(layout, held):
            return OracleResult(
                plan=plan, success=True,
                n_feasibility_checks=n_checks,
                elapsed_s=time.perf_counter() - t_start,
                notes=notes,
            )

        actions = _generate_successor_actions(
            layout, held, workspace, goal_layout, object_names,
            initial_layout,
        )[:max_actions_per_state]

        for action in actions:
            # Build successor layout / held from action semantics.
            new_layout = dict(layout)
            new_held: Optional[str]
            if action[0] == "pick":
                _, obj, cell_id = action
                if new_layout.get(obj) != cell_id or held is not None:
                    continue
                new_layout.pop(obj)
                new_held = obj
            elif action[0] == "put":
                _, obj, cell_id = action
                if held != obj or cell_id in set(layout.values()):
                    continue
                new_layout[obj] = cell_id
                new_held = None
            else:
                continue

            new_key = _state_key(new_layout, new_held)
            new_g = len(plan) + 1
            if closed.get(new_key, float("inf")) <= new_g:
                continue

            # FAST feasibility check.
            state = _layout_to_state(layout, held=held)
            n_checks += 1
            res = check_action_sequence(
                env, workspace, config, state, [action],
                object_names, pick_fn, put_fn,
                executor=executor, fast=fast, home_qpos=home_qpos,
            )
            if not res["success"]:
                continue

            closed[new_key] = new_g
            new_h = _h_misplaced(new_layout, new_held, goal_layout)
            new_f = new_g + new_h
            seq += 1
            heapq.heappush(frontier, (new_f, seq, new_key,
                                              new_layout, new_held,
                                              plan + [action]))
            if verbose:
                notes.append(f"expand: g={new_g} h={new_h} action={action}")

    notes.append(f"FRONTIER EXHAUSTED: {n_expanded} expanded, "
                    f"{n_checks} checks, no plan found")
    return OracleResult(plan=[], success=False,
                               n_feasibility_checks=n_checks,
                               elapsed_s=time.perf_counter() - t_start,
                               notes=notes)


# ---------------------------------------------------------------------------
# Phased hybrid planner
# ---------------------------------------------------------------------------


def phased_plan(
    env,
    workspace: Workspace,
    config: Access19Config,
    initial_layout: Dict[str, str],
    goal_layout: Dict[str, str],
    object_names: List[str],
    pick_fn: Callable,
    put_fn: Callable,
    *,
    executor,
    home_qpos: np.ndarray,
    fast: bool = True,
    phase1_max_actions: int = 200,
    phase2_time_budget_s: float = 120.0,
    phase2_max_states: int = 10000,
    verbose: bool = False,
) -> OracleResult:
    """Two-phase hybrid planner for L4-scale instances.

    **Phase 1** uses :func:`oracle_plan` to greedily clear the cubicle
    and place the OoI at its goal cell, ignoring the blockers' final
    positions.  This runs in 1–2 s even for 18 blockers.

    **Phase 2** uses :func:`astar_plan` to solve the residual problem:
    starting from the post-Phase-1 state (blockers wherever the
    greedy dumped them, OoI at goal), with goal = the full
    ``goal_layout``.  The A* in Phase 2 only has to plan the return
    legs (and any OoI temporary moves needed to free a return
    path), which is a much smaller search than the full problem.

    Concatenates the two action sequences and returns a single
    :class:`OracleResult`.
    """
    t_start = time.perf_counter()

    ooi_goal_id = goal_layout.get("ooi")
    if ooi_goal_id is None:
        return OracleResult(
            plan=[], success=False,
            elapsed_s=time.perf_counter() - t_start,
            notes=["phased_plan requires goal_layout['ooi']"],
        )
    ooi_goal_cell = Cell.parse(ooi_goal_id)

    # Phase 1: oracle gets OoI to its goal cell (blockers anywhere).
    phase1 = oracle_plan(
        env, workspace, config, initial_layout, ooi_goal_cell,
        object_names, pick_fn, put_fn,
        executor=executor, home_qpos=home_qpos, fast=fast,
        max_actions=phase1_max_actions, max_backtrack=64,
        verbose=verbose,
    )
    if not phase1.success:
        return OracleResult(
            plan=phase1.plan, success=False,
            n_feasibility_checks=phase1.n_feasibility_checks,
            elapsed_s=time.perf_counter() - t_start,
            notes=["Phase 1 (oracle) failed"] + phase1.notes,
        )

    # Apply Phase 1 actions to derive the intermediate layout.
    mid_layout = dict(initial_layout)
    for action in phase1.plan:
        verb, obj, cell = action
        if verb == "pick":
            # The picked object's cell removal happens implicitly when
            # it's later put — track via put only (simpler invariant).
            pass
        elif verb == "put":
            mid_layout[obj] = cell

    # Check whether the goal is already satisfied (greedy lucky path).
    if all(mid_layout.get(o) == c for o, c in goal_layout.items()):
        return OracleResult(
            plan=phase1.plan, success=True,
            n_feasibility_checks=phase1.n_feasibility_checks,
            elapsed_s=time.perf_counter() - t_start,
            notes=["Phase 1 alone reached goal"],
        )

    # Phase 2: A* solves the residual return + final OoI placement.
    phase2 = astar_plan(
        env, workspace, config, mid_layout, goal_layout, object_names,
        pick_fn, put_fn,
        executor=executor, home_qpos=home_qpos, fast=fast,
        max_states=phase2_max_states,
        time_budget_s=phase2_time_budget_s,
        verbose=verbose,
    )
    combined_notes = (
        ["Phase 1 OK"] + phase1.notes
        + [f"Phase 2 {'OK' if phase2.success else 'FAIL'}"] + phase2.notes
    )

    return OracleResult(
        plan=list(phase1.plan) + list(phase2.plan),
        success=phase2.success,
        n_feasibility_checks=(phase1.n_feasibility_checks
                                       + phase2.n_feasibility_checks),
        elapsed_s=time.perf_counter() - t_start,
        notes=combined_notes,
    )
