"""State restore for the tabletop_access:access shelf.

Restore the MuJoCo sim to match a grounded PDDL state — the inverse of
the bridge's ground_state.  Used by the feasibility checker so every
check starts from an IDENTICAL canonical world regardless of history
(objects snapped to cell centres with identity orientation, velocities
zeroed, arm at staging-home, attachment cleared).  Without this, a
marginal FULL chain outcome depends on accumulated drift from prior
checks.

Held states (``holding`` set) attach the object to the EE at the
CANONICAL grasp offset and record it on the executor as
``_held_grasp_dz`` — so a restored held state places exactly like a
freshly-picked one (the put chain reads that offset).

Public API::

    ground_to_object_cells(state, object_names)
    held_object_in_state(state)
    restore_state(env, ws, state, object_names, *, executor=None,
                  home_qpos=None, on_held="attach")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.symbolic.workspace import Cell, Workspace

# Mirror chains.py / feasibility.py.
_ITEM_HALF_Z_REF = 0.05
_CUBE_GRASP_OFFSET = 0.010
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_PARKED_XYZ: Tuple[float, float, float] = (100.0, 0.0, 0.10)


def ground_to_object_cells(state: Dict[Tuple, bool],
                           object_names: List[str]) -> Dict[str, Optional[str]]:
    """Extract ``{object: cell_id or None}`` from the ground-state dict."""
    placement: Dict[str, Optional[str]] = {o: None for o in object_names}
    for key, value in state.items():
        if not value:
            continue
        if (isinstance(key, tuple) and len(key) == 3 and key[0] == "occupied"):
            _, cell_id, obj = key
            if obj in placement:
                placement[obj] = cell_id
    return placement


def held_object_in_state(state: Dict[Tuple, bool]) -> Optional[str]:
    for key, value in state.items():
        if value and isinstance(key, tuple) and len(key) == 2 and key[0] == "holding":
            return key[1]
    return None


def _rest_z(env, workspace: Workspace, cell: Cell, obj: str) -> float:
    """World-frame resting CENTRE z of ``obj`` on ``cell``'s surface."""
    region = workspace[cell.region]
    half_z = float(env.get_object_half_size(obj)[2])
    return (region.level_z - _ITEM_HALF_Z_REF) + half_z


def restore_state(
    env,
    workspace: Workspace,
    state: Dict[Tuple, bool],
    object_names: List[str],
    *,
    executor=None,
    parked_xyz: Tuple[float, float, float] = _PARKED_XYZ,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "attach",
) -> Dict[str, Any]:
    """Restore MuJoCo state to match ``state`` (canonical, deterministic)."""
    if on_held not in ("attach", "park", "warn", "raise"):
        raise ValueError(f"on_held must be attach/park/warn/raise, got {on_held!r}")

    # 1. Hygiene — detach, open gripper, clear exceptions, stop controller.
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

    # 2. Park everyone at identity orientation.
    parked = np.asarray(parked_xyz, dtype=float)
    for name in object_names:
        env.set_object_pose(name, parked, _IDENTITY_QUAT)

    # 3. Place objects per (occupied cell obj), snapped to surface + identity.
    placement = ground_to_object_cells(state, object_names)
    placed: List[Tuple[str, str]] = []
    for obj, cell_id in placement.items():
        if cell_id is None:
            continue
        cell = Cell.parse(cell_id)
        cx, cy, _ = workspace.pose_for(cell)
        env.set_object_pose(obj, np.array([cx, cy, _rest_z(env, workspace, cell, obj)]),
                            _IDENTITY_QUAT)
        placed.append((obj, cell_id))

    # 4. Arm to staging-home BEFORE held attach (the canonical world
    # offset is computed from the home EE pose).
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
        else:  # attach at the canonical grasp offset
            half_z = float(env.get_object_half_size(held)[2])
            dz = -(half_z - _CUBE_GRASP_OFFSET)   # object centre below EE
            site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "attachment_site")
            ee = env.data.site_xpos[site].copy()
            env.set_object_pose(held, ee + np.array([0.0, GRASP_CONTACT_OFFSET, dz]),
                                _IDENTITY_QUAT)
            env.forward()
            env.attach_object_to_ee(held)
            env.data.qpos[7:9] = 0.020
            if env.controller is not None:
                env.data.ctrl[7] = -0.2
            env.forward()
            if executor is not None:
                executor._held_grasp_dz = dz

    parked_names = [n for n in object_names
                    if n not in {p[0] for p in placed} and n != held]
    return {"placed": placed, "parked": parked_names, "held": held}
