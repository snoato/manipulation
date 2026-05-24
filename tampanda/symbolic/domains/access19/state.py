"""State restore + stability check for access-19.

Given a grounded PDDL state, restore the MuJoCo simulation to match —
the inverse of the bridge's ``ground_state``.  Used by the feasibility
checker and the oracle-with-backtracking plan generator to probe
transitions from arbitrary symbolic states.

Held-block states (``holding`` fluent set) are NOT restored as a true
kinematic attachment — the access-19 chain assumes the arm starts from
staging-home, so feasibility checks of held states should be driven
via a ``[pick, put]`` action sequence rather than restoring the
mid-grip state directly.  When a held state is passed, the held block
is parked and a warning is logged.

Public API::

    from tampanda.symbolic.domains.access19.state import (
        restore_state, ground_to_object_cells, check_stability,
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.access19.env_builder import Access19Config


_PARKED_XYZ: Tuple[float, float, float] = (100.0, 0.0, 0.05)


def ground_to_object_cells(
    state: Dict[Tuple, bool],
    object_names: List[str],
) -> Dict[str, Optional[str]]:
    """Extract ``{object: cell_id or None}`` from the ground-state dict.

    ``None`` means the object is not in any ``(occupied cell obj)``
    predicate (parked or held).  Access-19 objects occupy exactly one
    cell when placed.
    """
    placement: Dict[str, Optional[str]] = {o: None for o in object_names}
    for key, value in state.items():
        if not value:
            continue
        if not (isinstance(key, tuple) and len(key) == 3
                    and key[0] == "occupied"):
            continue
        _, cell_id, obj = key
        if obj in placement:
            placement[obj] = cell_id
    return placement


def held_object_in_state(state: Dict[Tuple, bool]) -> Optional[str]:
    """Return the held object name if any, else None."""
    for key, value in state.items():
        if not value:
            continue
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "holding":
            return key[1]
    return None


def restore_state(
    env,
    workspace: Workspace,
    config: Access19Config,
    state: Dict[Tuple, bool],
    object_names: List[str],
    *,
    parked_xyz: Tuple[float, float, float] = _PARKED_XYZ,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "attach",
) -> Dict[str, Any]:
    """Restore MuJoCo state to match the PDDL ``state``.

    Args:
        env: ``FrankaEnvironment``.
        workspace: ``Workspace``.
        config: ``Access19Config``.
        state: ground-state dict from ``bridge.ground_state(...)``.
        object_names: full roster (18 blockers + ooi).
        parked_xyz: world position for objects not in any
            ``(occupied ...)`` predicate.
        home_qpos: arm config to apply (None leaves arm untouched).
        on_held: ``"attach"`` (default) kinematically attaches the
            held object to the EE at the canonical grasp offset, so
            the chain's pick/put helpers see a valid held state
            (the access-19 chain has exactly one grasp pose — palm-+y
            with the cube held at ``+y * GRASP_CONTACT_OFFSET`` and
            ``-z * (cube_half_z - _CUBE_GRASP_OFFSET)`` from EE, so
            the canonical offset is deterministic).  ``"park"``
            silently parks the held object and clears the gripper
            fluent.  ``"warn"`` same + warning.  ``"raise"`` raises
            ``NotImplementedError``.

    Returns:
        ``{"placed": [(obj, cell)], "parked": [...], "held": obj | None}``.
    """
    if on_held not in ("attach", "park", "warn", "raise"):
        raise ValueError(
            f"on_held must be attach/park/warn/raise, got {on_held!r}"
        )

    # 1. Reset env hygiene — detach, open gripper, clear exceptions.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    if env.controller is not None:
        env.controller.open_gripper()
        env.data.ctrl[7] = 0.04

    # 2. Park everyone first.
    parked_arr = np.asarray(parked_xyz, dtype=float)
    for name in object_names:
        env.set_object_pose(name, parked_arr)

    # 3. Place objects per the (occupied cell obj) predicates.
    placement = ground_to_object_cells(state, object_names)
    placed: List[Tuple[str, str]] = []
    for obj, cell_id in placement.items():
        if cell_id is None:
            continue
        cell = Cell.parse(cell_id)
        x, y, z = workspace.pose_for(cell)
        # Surface-z correction: workspace.pose_for returns the cube
        # CENTRE z; ground level is centre - half.  Cubes are uniform
        # 4 cm half-extent in access-19.
        cube_half_z = float(env.get_object_half_size(obj)[2])
        env.set_object_pose(obj, np.array([x, y, z]))
        placed.append((obj, cell_id))

    # 4. Restore arm BEFORE handling held — the held attach needs the
    # EE site at the staging-home pose to compute the canonical world
    # offset.
    if home_qpos is not None:
        env.data.qpos[: len(home_qpos)] = np.asarray(home_qpos, dtype=float)
    env.reset_velocities()
    env.forward()

    # 5. Handle held fluent.
    held = held_object_in_state(state)
    if held is not None:
        if on_held == "raise":
            raise NotImplementedError(
                f"Restore of held state ({held!r}) requires an executor; "
                "use on_held='attach' or 'park'."
            )
        if on_held == "warn" or on_held == "park":
            if on_held == "warn":
                print(f"[restore_state] WARNING: parking held object "
                          f"{held!r}; attachment not restored.")
            env.set_object_pose(held, parked_arr)
        elif on_held == "attach":
            # Kinematic attach at the canonical grasp offset.  See
            # ``chains.py:_pick_interior`` for the source: palm-+y
            # holds the cube at EE + (0, +GRASP_OFFSET, 0) in world
            # frame, with the cube centre sitting below the EE by
            # (cube_half_z - _CUBE_GRASP_OFFSET) = 0.03 m in world z.
            # We avoid computing the EE-LOCAL canonical_rel_pos
            # (axis mapping depends on the EE quaternion in subtle
            # ways) — instead place the cube at the desired WORLD
            # pose first, then call attach without canonical_rel_pos
            # so it captures the relative pose from the current
            # configuration.
            from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
            cube_half_z = float(env.get_object_half_size(held)[2])
            _CUBE_GRASP_OFFSET = 0.010      # mirror of chains.py
            site_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )
            ee_pos = env.data.site_xpos[site_id].copy()
            cube_world = ee_pos + np.array([
                0.0,
                GRASP_CONTACT_OFFSET,
                -(cube_half_z - _CUBE_GRASP_OFFSET),
            ])
            env.set_object_pose(held, cube_world)
            env.forward()
            env.attach_object_to_ee(held)
            # Close gripper (kinematically) — the chains expect the
            # gripper closed when a block is held.
            env.data.qpos[7] = 0.020
            env.data.qpos[8] = 0.020
            if env.controller is not None:
                env.data.ctrl[7] = -0.2
            env.forward()

    parked = [n for n in object_names
                  if n not in {p[0] for p in placed}
                  and (held is None or n != held)]
    return {"placed": placed, "parked": parked, "held": held}


def check_stability(
    env,
    *,
    settle_steps: int = 300,
    movement_threshold: float = 0.005,
    tracked_objects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Step physics with no controller input; report max object displacement.

    Used after a placement to detect topple — if any tracked object
    moves by more than ``movement_threshold`` (m), the placement is
    unstable.
    """
    if tracked_objects is None:
        names: List[str] = []
        for bid in range(env.model.nbody):
            n = env.model.body(bid).name
            if n.startswith("blocker_") or n == "ooi":
                names.append(n)
        tracked_objects = names

    start_pos: Dict[str, np.ndarray] = {}
    for name in tracked_objects:
        try:
            pos, _ = env.get_object_pose(name)
            start_pos[name] = np.asarray(pos, dtype=float).copy()
        except Exception:
            continue

    for _ in range(settle_steps):
        mujoco.mj_step(env.model, env.data)

    per_object: Dict[str, float] = {}
    for name, p0 in start_pos.items():
        try:
            pos, _ = env.get_object_pose(name)
            displacement = float(np.linalg.norm(np.asarray(pos) - p0))
        except Exception:
            displacement = float("nan")
        per_object[name] = displacement

    max_disp = max(per_object.values()) if per_object else 0.0
    return {
        "max_displacement": max_disp,
        "per_object": per_object,
        "stable": max_disp < movement_threshold,
    }
