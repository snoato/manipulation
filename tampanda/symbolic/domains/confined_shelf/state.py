"""State restore + stability check for confined_shelf (Wang ICAPS-2022).

Given a grounded PDDL state, restore the MuJoCo simulation to match — the
inverse of the bridge's ``ground_state``.  Used by the feasibility checker
and the rearrangement search to probe transitions from arbitrary symbolic
arrangements.

The primary feasibility unit is a ``[pick, put]`` pair evaluated from a
fully-grounded arrangement (no ``holding`` fluent), so held-state restore
is secondary.  When a ``holding`` fluent IS present, ``on_held="attach"``
kinematically attaches the held cylinder to the EE at the canonical FRONT
grasp offset (``_HELD_REL_POS``, EE-frame, measured by
``examples/cs_state_validate.py``).

Public API::

    from tampanda.symbolic.domains.confined_shelf.state import (
        restore_state, ground_to_object_cells, held_object_in_state,
        check_stability,
    )
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.confined_shelf.env_builder import ConfinedShelfConfig


# Canonical held-cylinder pose in the EE frame for a palm-+y FRONT grasp,
# measured empirically (see examples/cs_state_validate.py): the gripper
# grabs the bottle around its body, so the cylinder centre sits a little
# in front of and below the attachment site.  Updated from measurement.
_HELD_REL_POS: np.ndarray = np.array([-0.0012, -0.0001, 0.0144])


def _parked_xyz(config: ConfinedShelfConfig) -> Tuple[float, float, float]:
    return (config.hide_far_x, 0.0, config.cylinder_half_height)


def ground_to_object_cells(
    state: Dict[Tuple, bool],
    cylinder_names: List[str],
) -> Dict[str, Optional[str]]:
    """Extract ``{cylinder: cell_id or None}`` from the ground-state dict.

    ``None`` means the cylinder is not in any ``(occupied cell cyl)``
    predicate (parked or held).  Each cylinder occupies exactly one cell
    when placed.
    """
    placement: Dict[str, Optional[str]] = {c: None for c in cylinder_names}
    for key, value in state.items():
        if not value:
            continue
        if not (isinstance(key, tuple) and len(key) == 3
                and key[0] == "occupied"):
            continue
        _, cell_id, cyl = key
        if cyl in placement:
            placement[cyl] = cell_id
    return placement


def held_object_in_state(state: Dict[Tuple, bool]) -> Optional[str]:
    """Return the held cylinder name if any, else ``None``."""
    for key, value in state.items():
        if not value:
            continue
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "holding":
            return key[1]
    return None


def restore_state(
    env,
    workspace: Workspace,
    config: ConfinedShelfConfig,
    state: Dict[Tuple, bool],
    cylinder_names: List[str],
    *,
    parked_xyz: Optional[Tuple[float, float, float]] = None,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "attach",
    region_name: str = "shelf_interior",
) -> Dict[str, Any]:
    """Restore MuJoCo state to match the PDDL ``state``.

    Args:
        env: ``FrankaEnvironment``.
        workspace: ``Workspace`` with the ``shelf_interior`` region.
        config: ``ConfinedShelfConfig``.
        state: ground-state dict from ``bridge.ground_state(...)``.
        cylinder_names: full roster ``["cyl_0", ...]``.
        parked_xyz: world position for cylinders not in any
            ``(occupied ...)`` predicate.  Defaults to the sentinel
            ``(hide_far_x, 0, half_height)``.
        home_qpos: arm config to apply (``None`` leaves the arm
            untouched).
        on_held: ``"attach"`` (default) kinematically attaches the held
            cylinder to the EE at the canonical FRONT grasp offset;
            ``"park"`` parks it and clears the gripper fluent; ``"warn"``
            same + warning; ``"raise"`` raises ``NotImplementedError``.

    Returns:
        ``{"placed": [(cyl, cell)], "parked": [...], "held": cyl | None}``.
    """
    if on_held not in ("attach", "park", "warn", "raise"):
        raise ValueError(
            f"on_held must be attach/park/warn/raise, got {on_held!r}")

    parked = np.asarray(parked_xyz if parked_xyz is not None
                        else _parked_xyz(config), dtype=float)

    # 1. Hygiene — detach, open gripper, clear collision exceptions.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    if env.controller is not None:
        env.controller.open_gripper()
        env.data.ctrl[7] = 0.04

    # 2. Park everyone first.
    for name in cylinder_names:
        env.set_object_pose(name, parked)

    # 3. Place cylinders per the (occupied cell cyl) predicates.
    placement = ground_to_object_cells(state, cylinder_names)
    placed: List[Tuple[str, str]] = []
    for cyl, cell_id in placement.items():
        if cell_id is None:
            continue
        cell = Cell.parse(cell_id)
        x, y, z = workspace.pose_for(cell)
        env.set_object_pose(cyl, np.array([x, y, z]))
        placed.append((cyl, cell_id))

    # 4. Restore arm BEFORE handling held (the attach reads the EE site
    #    at the staging-home pose).
    if home_qpos is not None:
        env.data.qpos[: len(home_qpos)] = np.asarray(home_qpos, dtype=float)
    env.reset_velocities()
    env.forward()

    # 5. Handle the held fluent.
    held = held_object_in_state(state)
    if held is not None:
        if on_held == "raise":
            raise NotImplementedError(
                f"Restore of held state ({held!r}) needs an executor; use "
                "on_held='attach' or 'park'.")
        if on_held in ("park", "warn"):
            if on_held == "warn":
                print(f"[restore_state] WARNING: parking held cylinder "
                      f"{held!r}; attachment not restored.")
            env.set_object_pose(held, parked)
        else:  # attach: hold the cylinder UPRIGHT at the grasp offset.
            site_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            ee_pos = env.data.site_xpos[site_id].copy()
            ee_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
            cyl_world = ee_pos + ee_mat @ _HELD_REL_POS
            # Identity quat = cylinder axis along world +z (upright), as it
            # rests in a cell.  Then attach WITHOUT canonical_rel_pos so the
            # UPRIGHT relative orientation is captured.  Passing
            # canonical_rel_pos resets orientation to the palm-+y EE frame,
            # which holds the bottle ON ITS SIDE — it then "topples" the
            # instant it's released (it was never upright).
            env.set_object_pose(held, cyl_world,
                                np.array([1.0, 0.0, 0.0, 0.0]))
            env.forward()
            env.attach_object_to_ee(held)
            # Close gripper kinematically — the chains expect the gripper
            # closed when a cylinder is held.
            env.data.qpos[7] = 0.02
            env.data.qpos[8] = 0.02
            if env.controller is not None:
                env.data.ctrl[7] = -0.2
            env.forward()

    parked_names = [n for n in cylinder_names
                    if n not in {p[0] for p in placed}
                    and (held is None or n != held)]
    return {"placed": placed, "parked": parked_names, "held": held}


def check_stability(
    env,
    *,
    settle_steps: int = 300,
    movement_threshold: float = 0.005,
    tracked_objects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Step physics with no controller input; report max displacement.

    Used after a placement to detect a toppled / shifted cylinder.
    """
    if tracked_objects is None:
        tracked_objects = [
            env.model.body(bid).name
            for bid in range(env.model.nbody)
            if env.model.body(bid).name.startswith("cyl_")
        ]

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
            per_object[name] = float(np.linalg.norm(np.asarray(pos) - p0))
        except Exception:
            per_object[name] = float("nan")

    max_disp = max(per_object.values()) if per_object else 0.0
    return {
        "max_displacement": max_disp,
        "per_object": per_object,
        "stable": max_disp < movement_threshold,
    }
