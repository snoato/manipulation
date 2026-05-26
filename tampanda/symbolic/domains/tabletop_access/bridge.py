"""DomainBridge wiring for the tabletop-access (HAL 2025) domain.

Supports two PDDL modes:

* ``"filter"`` — actions are ``(pick ?obj ?cel)`` and ``(put ?obj ?cel)``.
  Faces are a refinement detail handled by the feasibility checker, not
  visible to the planner.
* ``"face"``  — actions are ``(pick ?obj ?face ?cel)`` and
  ``(put ?obj ?face ?cel)``.  The planner reasons about faces; the
  bridge evaluates ``(face-grasp-clear ?obj ?face)`` from the live
  MuJoCo state.

Both modes share the same predicate evaluators for ``occupied`` /
``empty`` / ``holding`` / ``gripper-empty`` and the same set of named
objects across all the workspace grids.  Faces are PDDL constants so
they don't appear in the ``objects`` dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, GridRegion, Workspace
from tampanda.tamp import DomainBridge

from tampanda.symbolic.domains.tabletop_access.env_builder import TabletopAccessConfig


_PDDL_FILTER = Path(__file__).parent / "pddl" / "domain.pddl"
_PDDL_FACE = Path(__file__).parent / "pddl" / "domain_face.pddl"

# Valid face values in face-mode (must match the constants in the PDDL).
_FACES: Tuple[str, ...] = ("top", "bottom", "front", "back", "left", "right")

# Per-region, the set of access_modes to face names (used by face-mode for
# face-grasp-clear evaluation).
_ACCESS_TO_FACE: Dict[str, str] = {
    "top_down": "top",
    "front": "front",
    "back": "back",
    "left": "left",
    "right": "right",
}

# Per-face, the (dx, dy, dz) direction toward which the gripper enters.
# (dx, dy) are cell-grid offsets used for the "adjacent cell empty" check
# in face-grasp-clear; dz is purely informational here.
_FACE_DXDY: Dict[str, Tuple[int, int]] = {
    "top": (0, 0),       # adjacent cell check is N/A for vertical faces
    "bottom": (0, 0),
    "front": (1, 0),     # +x direction (paper-camera-facing)
    "back": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Z-tolerance: an object is "in" a cell only if its centre is within this
# of the cell's resting plane.  Must accommodate small physics jitter.
_Z_TOLERANCE = 0.020


def make_tabletop_access_bridge(
    env,
    workspace: Workspace,
    config: TabletopAccessConfig,
    object_ids: Sequence[str],
    mode: str = "filter",
    executor=None,
    pick_fn=None,
    put_fn=None,
) -> Tuple[DomainBridge, Dict[str, List[str]]]:
    """Wire up the tabletop-access bridge.

    Args:
        env: A built FrankaEnvironment.
        workspace: The :class:`Workspace` returned alongside the env_builder.
        config: The :class:`TabletopAccessConfig` used at build time.
        object_ids: Names of all movable objects in the scene (target +
            blockers).  These match the body names registered with the
            SceneBuilder.
        mode: ``"filter"`` (default) or ``"face"``.
        executor: Optional :class:`PickPlaceExecutor` for action execution.
        pick_fn / put_fn: Optional
            ``(obj_name, cell_id, world_pos) -> bool`` callbacks that
            override ``executor.pick`` / ``executor.place`` for the
            corresponding bridge action.  Use these for access-19,
            where the default executor paths can't thread the linear-
            IK chain through the closed-top cubicle or around the deck
            top wall — see
            :func:`tampanda.symbolic.domains.tabletop_access.chains.make_access19_pick_fn`
            and :func:`...make_access19_put_fn`.  When ``None``
            (default), the legacy executor paths are used so callers
            that don't need chain-based actions (e.g., the open
            ``access`` variant) keep their existing behaviour.

    Returns:
        ``(bridge, objects)`` — ``objects`` is the type→name dict.
    """
    if mode not in ("filter", "face"):
        raise ValueError(f"mode must be 'filter' or 'face', got {mode!r}")
    if any(name == "shelf" for name in object_ids):
        raise ValueError("'shelf' is a reserved scene-body name; cannot be a movable")

    pddl_path = _PDDL_FILTER if mode == "filter" else _PDDL_FACE
    bridge = DomainBridge(pddl_path, env)

    cell_names: List[str] = []
    for region in workspace:
        cell_names.extend(c.id for c in region.cells())

    objects: Dict[str, List[str]] = {
        "cell": cell_names,
        "movable": list(object_ids),
    }
    if mode == "face":
        # Face values are PDDL constants — not added to ``objects`` here;
        # they're already declared in the domain.  But the bridge's
        # code-evaluated ``face-grasp-clear`` needs the face list to
        # iterate over, so we expose it via the ``face`` type for
        # grounding purposes.
        objects["face"] = list(_FACES)

    # ---- Helpers -----------------------------------------------------

    def _obj_world_pos(obj_name: str) -> np.ndarray:
        pos, _ = env.get_object_pose(obj_name)
        return np.asarray(pos)

    def _is_active(obj_name: str) -> bool:
        return _obj_world_pos(obj_name)[0] < config.hide_far_x - 1.0

    def _cell_for(obj_name: str) -> Optional[Cell]:
        if not _is_active(obj_name):
            return None
        p = _obj_world_pos(obj_name)
        # Probe each region in declaration order; first one whose level_z
        # matches within tolerance wins.  Two regions can have identical
        # XY footprints (interior + top), so we must filter by z.
        for region in workspace:
            if not isinstance(region, GridRegion):
                continue
            if abs(p[2] - region.level_z) > _Z_TOLERANCE:
                continue
            cell = region.cell_for(float(p[0]), float(p[1]), float(p[2]))
            if cell is not None:
                return cell
        return None

    def _occupant(cell: Cell) -> Optional[str]:
        """First object whose current cell is ``cell``."""
        for name in object_ids:
            if _cell_for(name) == cell:
                return name
        return None

    # ---- Code-evaluated predicates ----------------------------------

    @bridge.predicate("occupied")
    def eval_occupied(env, fluents, cell_id: str, obj_name: str) -> bool:
        if "__" not in cell_id:
            return False
        try:
            target = workspace.cell(cell_id)
        except (KeyError, ValueError):
            return False
        return _cell_for(obj_name) == target

    @bridge.predicate("empty")
    def eval_empty(env, fluents, cell_id: str) -> bool:
        if "__" not in cell_id:
            return False
        try:
            target = workspace.cell(cell_id)
        except (KeyError, ValueError):
            return False
        return _occupant(target) is None

    if mode == "face":

        @bridge.predicate("face-grasp-clear")
        def eval_face_clear(env, fluents, obj_name: str, face_name: str) -> bool:
            if face_name not in _FACES:
                return False
            cell = _cell_for(obj_name)
            if cell is None:
                # Object isn't in any cell (parked or held) — no face is
                # actionable in that state.
                return False
            region = workspace.region_of(cell)
            allowed_faces = {
                _ACCESS_TO_FACE.get(mode_name)
                for mode_name in region.access_modes
            }
            allowed_faces.discard(None)
            if face_name not in allowed_faces:
                return False
            # Adjacent-cell check for horizontal faces only.
            dx, dy = _FACE_DXDY[face_name]
            if dx == 0 and dy == 0:
                # Vertical face — the access_modes check above already
                # passed, so allow.
                return True
            adj_ix = cell.ix + dx
            adj_iy = cell.iy + dy
            if not isinstance(region, GridRegion):
                return False
            if not region.in_bounds(adj_ix, adj_iy):
                # Off the grid edge in that direction = open space, allowed.
                return True
            adj_cell = Cell(region.name, adj_ix, adj_iy)
            return _occupant(adj_cell) is None

    # ---- Fluent predicates ------------------------------------------

    bridge.fluent("holding", initial=None)
    bridge.fluent("gripper-empty", initial=True)

    # ---- Action executors -------------------------------------------

    if executor is not None:

        if mode == "filter":

            @bridge.action("pick")
            def exec_pick(env, fluents, obj_name: str, cell_id: str):
                pos = _obj_world_pos(obj_name)
                if pick_fn is not None:
                    ok = pick_fn(obj_name, cell_id, pos)
                else:
                    # Legacy executor.pick path — used by callers that
                    # haven't wired a chain-based pick_fn.
                    quat = env.get_object_orientation(obj_name)
                    ok = executor.pick(obj_name, pos,
                                          _half_size_estimate(env, obj_name),
                                          quat)
                if not ok:
                    return False, {}
                return True, {
                    ("holding", obj_name): True,
                    ("gripper-empty",): False,
                    ("occupied", cell_id, obj_name): False,
                    ("empty", cell_id): True,
                }

            @bridge.action("put")
            def exec_put(env, fluents, obj_name: str, cell_id: str):
                target = workspace.cell(cell_id)
                cx, cy, cz = workspace.pose_for(target)
                target_pos = np.array([cx, cy, cz])
                if put_fn is not None:
                    # Chain-based put (e.g., access-19's lift-then-
                    # traverse for shelf_top, column-align + row-step
                    # + retreat for shelf_interior).  put_fn handles
                    # detach + gripper open + retreat internally.
                    ok = put_fn(obj_name, cell_id, target_pos)
                else:
                    # Legacy fall-through for callers that haven't
                    # wired a chain-based put_fn.  Region-aware quat
                    # tweaks for the open ``access`` variant.
                    if target.region == "shelf_top":
                        place_quat = np.array([0.0, 1.0, 0.0, 0.0])
                        place_kwargs = dict(approach_height=0.03,
                                             retreat_lift=0.05)
                    else:
                        place_quat = None
                        place_kwargs = {}
                    ok = executor.place(obj_name, target_pos,
                                         ee_quat=place_quat,
                                         **place_kwargs)
                if not ok:
                    return False, {}
                return True, {
                    ("holding", obj_name): False,
                    ("gripper-empty",): True,
                    ("occupied", cell_id, obj_name): True,
                    ("empty", cell_id): False,
                }

        else:  # mode == "face"

            @bridge.action("pick")
            def exec_pick_face(env, fluents, obj_name: str,
                                 face_name: str, cell_id: str):
                pos = _obj_world_pos(obj_name)
                if pick_fn is not None:
                    ok = pick_fn(obj_name, cell_id, pos)
                else:
                    quat = env.get_object_orientation(obj_name)
                    ok = executor.pick(obj_name, pos,
                                          _half_size_estimate(env, obj_name),
                                          quat)
                if not ok:
                    return False, {}
                return True, {
                    ("holding", obj_name): True,
                    ("gripper-empty",): False,
                    ("occupied", cell_id, obj_name): False,
                    ("empty", cell_id): True,
                }

            @bridge.action("put")
            def exec_put_face(env, fluents, obj_name: str,
                                face_name: str, cell_id: str):
                target = workspace.cell(cell_id)
                cx, cy, cz = workspace.pose_for(target)
                target_pos = np.array([cx, cy, cz])
                if put_fn is not None:
                    ok = put_fn(obj_name, cell_id, target_pos)
                else:
                    if target.region == "shelf_top":
                        place_quat = np.array([0.0, 1.0, 0.0, 0.0])
                        place_kwargs = dict(approach_height=0.03,
                                             retreat_lift=0.05)
                    else:
                        place_quat = None
                        place_kwargs = {}
                    ok = executor.place(obj_name, target_pos,
                                         ee_quat=place_quat,
                                         **place_kwargs)
                if not ok:
                    return False, {}
                return True, {
                    ("holding", obj_name): False,
                    ("gripper-empty",): True,
                    ("occupied", cell_id, obj_name): True,
                    ("empty", cell_id): False,
                }

    return bridge, objects


def _half_size_estimate(env, obj_name: str) -> np.ndarray:
    """Best-effort bounding-box half-extents for an object body.

    Reads the geom size attached to the body — works for box and cylinder
    geoms shipped via the AssetSet templates.  Falls back to a 2.5 cm cube
    if unrecognised.
    """
    import mujoco
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if body_id < 0:
        return np.array([0.025, 0.025, 0.025])
    # First geom of the body
    for i in range(env.model.ngeom):
        if env.model.geom_bodyid[i] == body_id:
            return env.model.geom_size[i].copy()
    return np.array([0.025, 0.025, 0.025])


_QUAT_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
# 90° rotation around world-z (MuJoCo WXYZ).
_QUAT_ROT_Z_90 = np.array([0.7071067811865476, 0.0, 0.0,
                              0.7071067811865476])


def _graspable_quat(env, name: str) -> np.ndarray:
    """Choose the spawn orientation that keeps the world-x extent of
    the item under the Franka gripper's max opening (0.08 m).  For
    a FRONT_X grasp the finger axis is world-x, so the item's
    half_x must be the SMALLER of (half_x, half_y).  When the
    default orientation has half_x > half_y, rotate the block 90°
    around z so its short side ends up along world-x.
    """
    half = env.get_object_half_size(name)
    if float(half[0]) > float(half[1]):
        return _QUAT_ROT_Z_90
    return _QUAT_IDENTITY


def set_objects_at_cells(
    env,
    workspace: Workspace,
    config: TabletopAccessConfig,
    placements: Dict[str, Cell],
    all_object_ids: Sequence[str],
) -> None:
    """Park all then teleport listed objects to cell centres.

    Each item's z is computed PER-ITEM as ``surface_z + half_z`` so
    items of different heights rest correctly.  Each item's
    orientation is chosen so the world-x extent stays under the
    gripper opening (rotated 90° around z when half_x > half_y).
    """
    parked_xyz = np.array([config.hide_far_x, 0.0, 0.05])
    for name in all_object_ids:
        env.set_object_pose(name, parked_xyz)
    for name, cell in placements.items():
        region = workspace.region_of(cell)
        x, y, _ = workspace.pose_for(cell)
        if config.scene_variant == "access19":
            surface_z = region.level_z - 0.040  # uniform cube
        else:
            surface_z = region.level_z - 0.05    # access builder's _ITEM_HALF_Z
        half_z = float(env.get_object_half_size(name)[2])
        z = surface_z + half_z + 0.001  # 1 mm clearance, physics settles
        quat = _graspable_quat(env, name)
        env.set_object_pose(name, np.array([x, y, z]), quat)
    env.reset_velocities()
    env.forward()
