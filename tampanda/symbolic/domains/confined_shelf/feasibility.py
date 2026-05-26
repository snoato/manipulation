"""Per-action feasibility checker for confined_shelf (Wang ICAPS-2022).

Two variants, sharing the same grasp-pose computation
(``executor.grasp_planner``) so FAST tracks FULL closely:

* :func:`check_full` — restore state, run the real RRT*-based
  ``PickPlaceExecutor.pick`` / ``.place`` with full physics, then a
  settle/stability check on a put.  Ground truth.  ~0.3–3 s/action.

* :func:`check_fast` — restore state, then a cheap geometric check:
  a **symbolic prefilter** (reachable cell ∧ front-column clear ∧
  lateral-neighbour clear — the empirically-measured FRONT-grasp
  blocking model) followed, if it passes, by a **LinearIK** two-stage
  joint-lerp (home→approach→grasp) with collision checks at a few
  substeps — **no RRT***.  ~0.01–0.1 s/action.

The blocking model (see examples/cs_probe_neighbors.py): a FRONT
(palm-+y) grasp of a vertical cylinder at cell ``(ix, iy)`` is blocked
by a cylinder in any front cell ``(ix, iy'<iy)`` of the same column, or
in either lateral neighbour ``(ix±1, iy)``; a cylinder behind or two
cells to the side does not block.

Action format::

    ("pick", cyl_name, cell_id)
    ("put",  cyl_name, cell_id)

Public API::

    check_action(env, ws, cfg, state, action, cyl_names, *, executor, lik, fast)
    check_action_sequence(...)
    check_fast(...) / check_full(...)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import (
    GRASP_CONTACT_OFFSET, LINK7_CAPSULE_RADIUS, quat_to_rotmat,
)
from tampanda.symbolic.workspace import Cell, GridRegion, Workspace

from tampanda.symbolic.domains.confined_shelf.env_builder import ConfinedShelfConfig
from tampanda.symbolic.domains.confined_shelf.planner import column_clear_reason
from tampanda.symbolic.domains.confined_shelf.state import (
    check_stability, restore_state,
)


# Canonical FRONT (palm-+y) grasp orientation — the single grasp this
# domain uses.  Passed explicitly to executor.place so a standalone put
# check doesn't depend on a prior pick having set _last_grasp_quat.
_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
# Hand -z axis in world for the FRONT grasp ( = +y, into the shelf).
_HAND_Z = quat_to_rotmat(_FRONT_QUAT)[:, 2]
_PUT_CLEARANCE = 0.008       # mirror of ConfinedShelfExecutor.place default
                             # (held bottle released ~5-7 mm above the shelf
                             # floor; a 3 mm release grazes shelf_floor mid-
                             # descent and every planning rung rejects it.
                             # The deepest cells tilt the held bottle ~2 mm
                             # at the IK config, so 8 mm is needed to clear
                             # the worst one — (7,3) — with margin)


# ---------------------------------------------------------------------------
# Symbolic prefilter — the empirically-grounded FRONT-grasp blocking model
# ---------------------------------------------------------------------------


def _occupied_set(state: Dict[Tuple, bool]) -> Set[Tuple[int, int]]:
    """Set of ``(ix, iy)`` of occupied cells from the ground-state dict."""
    out: Set[Tuple[int, int]] = set()
    for key, value in state.items():
        if not value:
            continue
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "occupied":
            c = Cell.parse(key[1])
            out.add((c.ix, c.iy))
    return out


def prefilter_reject(
    action: Tuple,
    occupied: Set[Tuple[int, int]],
    region: GridRegion,
) -> Tuple[bool, Optional[str]]:
    """Return ``(reject, reason)``.

    ``reject`` is True iff the action is provably infeasible from the
    occupancy alone — no IK / physics needed.  Applies to both ``pick``
    (source cell) and ``put`` (destination cell); the gripper geometry
    is the same either way.  The held cylinder (for a put) is not in
    ``occupied`` so it never blocks its own destination.
    """
    if not action or action[0] not in ("pick", "put"):
        return False, None
    cell = Cell.parse(action[2])
    # Exclude the action's own cell: a pick's target isn't a blocker of
    # itself; a put's destination is empty.  Same rule the search uses
    # (shared column_clear_reason) so search and FAST gate can't drift.
    reason = column_clear_reason(occupied - {(cell.ix, cell.iy)},
                                 cell.ix, cell.iy, region)
    return (reason is not None), reason


# ---------------------------------------------------------------------------
# FAST motion — discrete column-pose collision check (no RRT*, no stepping)
# ---------------------------------------------------------------------------
#
# Every grasp is FRONT into a single column, so the motion is a straight
# push-in.  Feasibility reduces to: can the arm (pick) or arm+held-cylinder
# (put) be at each EE key-pose along the column — the entrance (outside the
# open face) and each cell (ix, r) from the front up to the target —
# collision-free?  Each pose = one converge_ik + one is_collision_free (one
# mj_forward).  Strict subset of FULL: if every discrete pose is clear,
# FULL's straight joint-lerp push-in through them is clear, so FULL accepts
# too; FAST only extra-rejects cases that need a non-straight RRT* detour.


def _pose_reachable_cf(env, pos: np.ndarray, quat: np.ndarray,
                       home_q: np.ndarray) -> bool:
    """True iff FRONT EE pose ``(pos, quat)`` is IK-reachable AND
    collision-free.  IK seeds from the staging home (all poses are near the
    shelf front, so home is a good seed and checks stay independent).
    ``is_collision_free`` carries any held cylinder along
    (set_collision_held_body) and honours collision exceptions."""
    env.data.qpos[:7] = home_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(np.asarray(pos, float), np.asarray(quat, float))
    if not env.ik.converge_ik(0.005):
        return False
    q = env.ik.configuration.q[:7].copy()
    return env.is_collision_free(q)


def _front_ee_pos(cell_center: np.ndarray, table_z: float,
                  clearance: float) -> np.ndarray:
    """FRONT-grasp EE position for a cylinder centred at ``cell_center``.

    Mirrors ``GraspPlanner``/``PickPlaceExecutor.place``: shift by the
    grasp-contact offset along −hand-z, add ``clearance`` (0 for pick, the
    place clearance for put — so a held cylinder sits just ABOVE the floor
    rather than touching it, which would be a false robot↔floor contact),
    and clamp z above the link7 capsule for the near-horizontal FRONT
    grasp."""
    pos = np.asarray(cell_center, float) - _HAND_Z * GRASP_CONTACT_OFFSET
    pos[2] += clearance
    if abs(float(_HAND_Z[2])) < 0.7:                 # not top-down (FRONT)
        pos[2] = max(pos[2], table_z + LINK7_CAPSULE_RADIUS + 0.005)
    return pos


def _column_poses(workspace: Workspace, region_name: str, ix: int, iy: int,
                  table_z: float, clearance: float):
    """FRONT EE key-poses for a push-in to ``(ix, iy)``:
    ``(entrance, [(pos, quat) for r in 0..iy])``.  ``entrance`` is an 8 cm
    standoff in front of the front-most cell (outside the open face),
    matching the executor's approach distance."""
    rows: List[Tuple[np.ndarray, np.ndarray]] = []
    entrance: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for r in range(iy + 1):
        cc = np.asarray(workspace.pose_for(Cell(region_name, ix, r)))
        pos = _front_ee_pos(cc, table_z, clearance)
        rows.append((pos, _FRONT_QUAT))
        if r == 0:
            entrance = (pos - _HAND_Z * 0.08, _FRONT_QUAT)
    return entrance, rows


def _fast_geometric(env, workspace: Workspace, config: ConfinedShelfConfig,
                    action: Tuple, home_q: np.ndarray,
                    region_name: str) -> bool:
    """Discrete column-pose feasibility (see section header).  Assumes the
    scene is already restored (cylinders placed; for a put the held cylinder
    is attached upright as robot geometry)."""
    name, cyl, cell_id = action
    cell = Cell.parse(cell_id)
    ix, iy = cell.ix, cell.iy
    table_z = workspace[region_name].level_z - config.cylinder_half_height
    clearance = _PUT_CLEARANCE if name == "put" else 0.0
    entrance, rows = _column_poses(workspace, region_name, ix, iy,
                                   table_z, clearance)

    # PICK: the gripper closes around the target cylinder, so its contact
    # with the gripper is expected — except it out so the grasp pose isn't
    # a false collision.  PUT: the held cylinder is already robot geometry
    # via the attachment, so no exception is needed.
    if name == "pick":
        env.add_collision_exception(cyl)
    try:
        # Reject order: target pose first (quickest reject), then the
        # entrance, then the front rows from the target outward.
        order = [rows[iy], entrance] + [rows[r] for r in range(iy - 1, -1, -1)]
        for pos, quat in order:
            if not _pose_reachable_cf(env, pos, quat, home_q):
                return False
        return True
    finally:
        if name == "pick":
            env.remove_collision_exception(cyl)


# ---------------------------------------------------------------------------
# FULL motion — RRT* executor + physics settle
# ---------------------------------------------------------------------------


def _full_dispatch(env, workspace: Workspace, executor, action: Tuple,
                   config: ConfinedShelfConfig) -> bool:
    half = np.array([config.cylinder_radius, config.cylinder_radius,
                     config.cylinder_half_height])
    name, cyl, cell_id = action
    if name == "pick":
        pos, quat = env.get_object_pose(cyl)
        return bool(executor.pick(cyl, np.asarray(pos), half,
                                  np.asarray(quat)))
    if name == "put":
        target = workspace.cell(cell_id)
        tpos = np.asarray(workspace.pose_for(target))
        if not executor.place(cyl, tpos, ee_quat=_FRONT_QUAT):
            return False
        # The placed cylinder must rest stably (not topple onto a
        # neighbour) for the put to count as feasible.
        st = check_stability(env, settle_steps=200,
                             movement_threshold=0.01,
                             tracked_objects=[cyl])
        return bool(st["stable"])
    raise ValueError(f"confined_shelf actions are pick/put only, got {name!r}")


# ---------------------------------------------------------------------------
# Top-level checkers
# ---------------------------------------------------------------------------


def check_action(
    env,
    workspace: Workspace,
    config: ConfinedShelfConfig,
    state: Dict[Tuple, bool],
    action: Tuple,
    cylinder_names: List[str],
    *,
    executor,
    lik=None,
    fast: bool = True,
    home_qpos: Optional[np.ndarray] = None,
    region_name: str = "shelf_interior",
) -> Dict[str, Any]:
    """Check whether ``action`` is feasible from ``state``.

    Args:
        executor: ``PickPlaceExecutor``.  FULL calls its ``pick``/``place``
            (RRT* + physics + stability); FAST reuses its ``grasp_planner``
            to compute the FRONT column poses.
        lik: deprecated/unused (FAST no longer uses a LinearIK planner);
            kept for call-site compatibility.
        fast: FAST (µs occupancy quick-reject + discrete column-pose
            collision check) vs FULL (executor + physics settle).
        home_qpos: arm config applied before the action / IK seed.

    Returns ``{"success", "elapsed_s", "error", "fast",
    ["prefiltered", "reason"]}``.
    """
    t0 = time.perf_counter()
    region = workspace[region_name]

    if fast:
        # Stage 1 — µs symbolic quick-reject on cell occupancy.
        reject, reason = prefilter_reject(action, _occupied_set(state), region)
        if reject:
            return {"success": False, "elapsed_s": time.perf_counter() - t0,
                    "error": None, "fast": True, "prefiltered": True,
                    "reason": reason}

    restore_state(env, workspace, config, state, cylinder_names,
                  home_qpos=home_qpos, on_held="attach", region_name=region_name)
    home_q = (np.asarray(home_qpos, float)[:7] if home_qpos is not None
              else env.data.qpos[:7].copy())

    try:
        if fast:
            # Stage 2 — geometric column-pose collision check.
            ok = _fast_geometric(env, workspace, config, action,
                                 home_q, region_name)
        else:
            ok = _full_dispatch(env, workspace, executor, action, config)
        err = None
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    return {"success": bool(ok), "elapsed_s": time.perf_counter() - t0,
            "error": err, "fast": fast}


def check_action_sequence(
    env,
    workspace: Workspace,
    config: ConfinedShelfConfig,
    state: Dict[Tuple, bool],
    actions: List[Tuple],
    cylinder_names: List[str],
    *,
    executor,
    lik=None,
    fast: bool = True,
    home_qpos: Optional[np.ndarray] = None,
    short_circuit: bool = True,
    region_name: str = "shelf_interior",
) -> Dict[str, Any]:
    """Restore ``state`` once, then dispatch ``actions`` in order.

    Maintains a running symbolic layout: a ``pick`` removes the cylinder
    from its cell (held), a ``put`` re-adds it at the destination.  In
    FAST mode each action is prefiltered against the *current* layout, so
    front/lateral blocking is re-evaluated as the sequence evolves.  Used
    to validate reference plans during data generation.  (``lik`` is
    accepted for call-site compatibility but unused.)
    """
    t0 = time.perf_counter()

    # Thread (layout, held) through the sequence.  Each action is checked
    # by ``check_action`` from a freshly-restored pre-state: restore_state
    # attaches the held cylinder so a pick's cylinder is OFF its old cell
    # for the following put (it no longer front-blocks its own
    # destination).  Per-action restore keeps FAST and FULL identical in
    # how state evolves.
    layout: Dict[str, Tuple[int, int]] = {}
    held: Optional[str] = None
    for key, value in state.items():
        if not value or not isinstance(key, tuple):
            continue
        if len(key) == 3 and key[0] == "occupied":
            c = Cell.parse(key[1])
            layout[key[2]] = (c.ix, c.iy)
        elif len(key) == 2 and key[0] == "holding":
            held = key[1]

    per_action: List[Dict[str, Any]] = []
    overall = True
    for action in actions:
        pre_state: Dict[Tuple, bool] = {
            ("occupied", Cell(region_name, ix, iy).id, c): True
            for c, (ix, iy) in layout.items()
        }
        if held is not None:
            pre_state[("holding", held)] = True

        res = check_action(env, workspace, config, pre_state, action,
                           cylinder_names, executor=executor, lik=lik,
                           fast=fast, home_qpos=home_qpos,
                           region_name=region_name)
        ok = res["success"]
        entry = {"action": action, "success": ok,
                 "elapsed_s": res["elapsed_s"], "error": res.get("error")}
        if res.get("prefiltered"):
            entry["prefiltered"] = True
            entry["reason"] = res.get("reason")
        per_action.append(entry)

        if ok:
            name, cyl, cell_id = action
            if name == "pick":
                held = cyl
                layout.pop(cyl, None)
            else:
                c = Cell.parse(cell_id)
                held = None
                layout[cyl] = (c.ix, c.iy)
        else:
            overall = False
            if short_circuit:
                break

    return {"success": overall, "elapsed_s": time.perf_counter() - t0,
            "fast": fast, "per_action": per_action}


def check_fast(env, ws, cfg, state, action, cyl_names, *, executor, **kw):
    kw.pop("fast", None)
    return check_action(env, ws, cfg, state, action, cyl_names,
                        executor=executor, fast=True, **kw)


def check_full(env, ws, cfg, state, action, cyl_names, *, executor, **kw):
    kw.pop("fast", None)
    return check_action(env, ws, cfg, state, action, cyl_names,
                        executor=executor, fast=False, **kw)
