"""Canonical multi-cell state restore for the dense-YCB tabletop-access fork.

Inverse of the bridge's ``ground_state``: drive the sim to match a
grounded PDDL state so every feasibility check starts from an identical
world regardless of history.  Multi-cell objects are placed at their
footprint centroid (via :meth:`ObjectFootprint.place_pose`, which also
applies the canonical grasp yaw and the body-origin/bottom-offset
correction); a ``holding`` object is attached to the EE at the canonical
grasp offset.

The object's anchor is recovered from its ``(occupied …)`` facts: the
set of occupied cells → south-west corner = anchor.

Public API::

    ground_to_object_anchors(state, object_ids)
    held_object_in_state(state)
    restore_state(env, ws, state, object_ids, footprints, *, executor=None,
                  home_qpos=None, on_held="attach")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    ObjectFootprint, anchor_of_cells,
)

_CUBE_GRASP_OFFSET = 0.010   # grasp this far below the object top
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_PARKED_XYZ: Tuple[float, float, float] = (100.0, 0.0, 0.10)


def ground_to_object_anchors(
    state: Dict[Tuple, bool], object_ids: List[str],
) -> Dict[str, Optional[Cell]]:
    """``{obj: anchor Cell or None}`` from the grounded ``(occupied …)`` facts."""
    cells_by_obj: Dict[str, List[str]] = {o: [] for o in object_ids}
    for key, val in state.items():
        if not val:
            continue
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "occupied":
            _, cell_id, obj = key
            if obj in cells_by_obj:
                cells_by_obj[obj].append(cell_id)
    return {o: (anchor_of_cells(cl) if cl else None)
            for o, cl in cells_by_obj.items()}


def held_object_in_state(state: Dict[Tuple, bool]) -> Optional[str]:
    for key, val in state.items():
        if val and isinstance(key, tuple) and len(key) == 2 and key[0] == "holding":
            return key[1]
    return None


def restore_state(
    env,
    workspace: Workspace,
    state: Dict[Tuple, bool],
    object_ids: List[str],
    footprints: Dict[str, ObjectFootprint],
    *,
    executor=None,
    parked_xyz: Tuple[float, float, float] = _PARKED_XYZ,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "attach",
) -> Dict[str, Any]:
    """Restore the MuJoCo state to match ``state`` (canonical, deterministic)."""
    if on_held not in ("attach", "park", "warn", "raise"):
        raise ValueError(f"on_held must be attach/park/warn/raise, got {on_held!r}")

    # 1. Hygiene.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    if env.controller is not None:
        env.controller.stop()
        env.controller.open_gripper()
    env.data.qpos[7:9] = 0.04
    if executor is not None:
        executor._held_grasp_dz = None

    # 2. Park everyone (identity).
    parked = np.asarray(parked_xyz, dtype=float)
    for name in object_ids:
        env.set_object_pose(name, parked, _IDENTITY_QUAT)

    # 3. Place objects at their footprint centroids.
    anchors = ground_to_object_anchors(state, object_ids)
    placed: List[Tuple[str, str]] = []
    for obj, anchor in anchors.items():
        if anchor is None:
            continue
        region = workspace[anchor.region]
        fp = footprints[obj]
        pos, quat = fp.place_pose(region, anchor)
        env.set_object_pose(obj, pos, quat)
        placed.append((obj, anchor.id))

    # 4. Arm to staging-home BEFORE held attach.
    if home_qpos is not None:
        env.data.qpos[: len(home_qpos)] = np.asarray(home_qpos, dtype=float)
    env.reset_velocities()
    env.forward()

    # 5. Held fluent.
    held = held_object_in_state(state)
    if held is not None:
        if on_held == "raise":
            raise NotImplementedError(f"held state {held!r}; use attach/park")
        if on_held in ("park", "warn"):
            if on_held == "warn":
                print(f"[restore_state] WARNING: parking held {held!r}")
            env.set_object_pose(held, parked, _IDENTITY_QUAT)
        else:
            half_z = float(env.get_object_half_size(held)[2])
            dz = -(half_z - _CUBE_GRASP_OFFSET)
            site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "attachment_site")
            ee = env.data.site_xpos[site].copy()
            env.set_object_pose(held, ee + np.array([0.0, GRASP_CONTACT_OFFSET, dz]),
                                np.array(footprints[held].quat, dtype=float))
            env.forward()
            env.attach_object_to_ee(held)
            env.data.qpos[7:9] = 0.020
            if env.controller is not None:
                env.data.ctrl[7] = -0.2
            env.forward()
            if executor is not None:
                executor._held_grasp_dz = dz

    parked_names = [n for n in object_ids
                    if n not in {p[0] for p in placed} and n != held]
    return {"placed": placed, "parked": parked_names, "held": held}
