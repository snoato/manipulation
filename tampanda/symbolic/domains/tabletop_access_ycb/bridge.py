"""DomainBridge wiring for the dense-YCB tabletop-access fork.

Differences from the parent ``access`` bridge:

* **Multi-cell occupancy.** ``occupied(cell, obj)`` / ``empty(cell)`` are
  evaluated from each object's footprint at its current pose — an object
  covers all the cells of its ``dx × dy`` block, not one.
* **Per-footprint-size actions.** ``pick_<W>x<H>`` / ``put_<W>x<H>``
  executors (one per footprint size present) map the action's cell
  parameters to the object's SW anchor + centroid and dispatch to the
  chain ``pick_fn`` / ``put_fn``, then flip occupancy for *every* covered
  cell.

The static ``adjacent`` / ``fp_*`` predicates are NOT bridge predicates —
they're emitted in the problem ``:init`` by :mod:`pddl_gen` (enumerating
``adjacent`` over all cell pairs would blow up ``ground_state``).  The
fork plans with a custom feasibility-guided planner, not bridge/UP
planning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, GridRegion, Workspace
from tampanda.tamp import DomainBridge

from tampanda.symbolic.domains.tabletop_access_ycb.env_builder import (
    TabletopAccessYcbConfig,
)
from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    ObjectFootprint, anchor_of_cells, occupied_cells_from_pose,
)

_PDDL_DOMAIN = Path(__file__).parent / "pddl" / "domain.pddl"
_Z_TOLERANCE = 0.025   # body-z match window for region selection


def make_tabletop_access_ycb_bridge(
    env,
    workspace: Workspace,
    config: TabletopAccessYcbConfig,
    object_ids: Sequence[str],
    footprints: Dict[str, ObjectFootprint],
    executor=None,
    pick_fn=None,
    put_fn=None,
    domain_path: Path = _PDDL_DOMAIN,
) -> Tuple[DomainBridge, Dict[str, List[str]]]:
    """Wire the dense-YCB bridge.

    Returns ``(bridge, objects)`` where ``objects`` is the type→name dict
    (``cell`` + ``movable``; ``direction`` is a PDDL constant, ``fp_*`` a
    static emitted in the problem).
    """
    if any(name == "shelf" for name in object_ids):
        raise ValueError("'shelf' is reserved; cannot be a movable")

    bridge = DomainBridge(domain_path, env)

    cell_names: List[str] = []
    for region in workspace:
        cell_names.extend(c.id for c in region.cells())
    objects: Dict[str, List[str]] = {
        "cell": cell_names,
        "movable": list(object_ids),
    }

    regions = [r for r in workspace if isinstance(r, GridRegion)]

    def _obj_pos(obj: str) -> np.ndarray:
        pos, _ = env.get_object_pose(obj)
        return np.asarray(pos, dtype=float)

    def _active(obj: str) -> bool:
        return _obj_pos(obj)[0] < config.hide_far_x - 1.0

    def _region_for(obj: str, pos: np.ndarray) -> Optional[GridRegion]:
        """Region whose surface matches the object's resting body-z."""
        fp = footprints[obj]
        best = None
        best_err = _Z_TOLERANCE
        for r in regions:
            err = abs(pos[2] - (r.level_z - fp.bottom))
            if err < best_err:
                best, best_err = r, err
        return best

    _cover_memo: Dict[Tuple[str, bytes], List[Cell]] = {}

    def _covered_cells(obj: str) -> List[Cell]:
        if not _active(obj):
            return []
        pos = _obj_pos(obj)
        key = (obj, pos.tobytes())   # pose-keyed: stable within a ground_state
        hit = _cover_memo.get(key)
        if hit is not None:
            return hit
        region = _region_for(obj, pos)
        cells = (occupied_cells_from_pose(env, region, footprints[obj], pos)
                 if region is not None else [])
        _cover_memo[key] = cells
        return cells

    # ---- code predicates: multi-cell occupancy ----
    @bridge.predicate("occupied")
    def eval_occupied(env, fluents, cell_id: str, obj_name: str) -> bool:
        if obj_name not in footprints or "__" not in cell_id:
            return False
        return any(c.id == cell_id for c in _covered_cells(obj_name))

    @bridge.predicate("empty")
    def eval_empty(env, fluents, cell_id: str) -> bool:
        if "__" not in cell_id:
            return False
        for obj in object_ids:
            if any(c.id == cell_id for c in _covered_cells(obj)):
                return False
        return True

    bridge.fluent("holding", initial=None)
    bridge.fluent("gripper-empty", initial=True)

    # ---- per-size action executors ----
    if executor is not None or pick_fn is not None or put_fn is not None:
        sizes = sorted({(fp.dx, fp.dy) for fp in footprints.values()})
        for (W, H) in sizes:
            _register_size_actions(bridge, env, workspace, footprints,
                                   object_ids, W, H, pick_fn, put_fn, executor)

    return bridge, objects


def _register_size_actions(bridge, env, workspace, footprints, object_ids,
                           W, H, pick_fn, put_fn, executor):
    """Register pick_WxH / put_WxH executors (closures bind W, H)."""

    def _anchor_centroid(obj, cell_ids):
        anchor = anchor_of_cells(cell_ids)
        region = workspace[anchor.region]
        cx, cy = footprints[obj].centroid_world(region, anchor)
        return anchor, region, np.array([cx, cy, region.level_z])

    @bridge.action(f"pick_{W}x{H}")
    def exec_pick(env, fluents, obj, *cell_ids, _W=W, _H=H):
        _, _, centroid = _anchor_centroid(obj, cell_ids)
        if pick_fn is not None:
            ok = pick_fn(obj, cell_ids[0], centroid)
        else:
            ok = executor.pick(obj, _obj_world(env, obj),
                               footprints[obj], None)
        if not ok:
            return False, {}
        delta = {("holding", obj): True, ("gripper-empty",): False}
        for c in cell_ids:
            delta[("occupied", c, obj)] = False
            delta[("empty", c)] = True
        return True, delta

    @bridge.action(f"put_{W}x{H}")
    def exec_put(env, fluents, obj, *cell_ids, _W=W, _H=H):
        _, _, centroid = _anchor_centroid(obj, cell_ids)
        if put_fn is not None:
            ok = put_fn(obj, cell_ids[0], centroid)
        else:
            ok = executor.place(obj, centroid)
        if not ok:
            return False, {}
        delta = {("holding", obj): False, ("gripper-empty",): True}
        for c in cell_ids:
            delta[("occupied", c, obj)] = True
            delta[("empty", c)] = False
        return True, delta


def _obj_world(env, obj):
    pos, _ = env.get_object_pose(obj)
    return np.asarray(pos, dtype=float)
