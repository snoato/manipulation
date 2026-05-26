"""DomainBridge wiring for the confined-shelf (Wang ICAPS-2022) domain.

Exposes :func:`make_confined_shelf_bridge` which:

* Builds a :class:`~tampanda.tamp.DomainBridge` from
  ``pddl/domain.pddl``.
* Registers code-evaluated predicates ``occupied`` and ``empty`` against
  the live MuJoCo state of the cylinders relative to the workspace
  :class:`~tampanda.symbolic.workspace.Workspace`.
* Registers fluent predicates ``holding`` and ``gripper-empty`` whose
  truth is updated by action executors.
* Registers the static predicate ``color-of`` from the colour assignment
  passed in.
* Optionally registers ``pick`` and ``put`` action executors when an
  :class:`~tampanda.planners.pick_place.PickPlaceExecutor` is provided.
  When omitted, the bridge supports grounding and planning but cannot
  actually move the robot — useful for the reliability test suite.

Cylinder world position is the source of truth: a cylinder is "occupied"
in cell *c* iff its world XY centre is inside cell *c*'s footprint AND
its world Z is within ``±height_tolerance`` of the resting plane.
Cylinders parked at the sentinel x position (``cfg.hide_far_x``) are
treated as not-yet-active and don't appear in any cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, GridRegion, Workspace
from tampanda.tamp import DomainBridge

from tampanda.symbolic.domains.confined_shelf.env_builder import ConfinedShelfConfig


_PDDL_PATH = Path(__file__).parent / "pddl" / "domain.pddl"

# A cylinder is considered to be "in" a cell only if its centre is within
# this z-tolerance of the resting plane.  Tighter than the legacy tabletop
# (which used 1 cm) because the shelf interior is more constrained.
_Z_TOLERANCE = 0.015


def make_confined_shelf_bridge(
    env,
    workspace: Workspace,
    config: ConfinedShelfConfig,
    cylinder_colors: Sequence[str],
    region_name: str = "shelf_interior",
    executor=None,
) -> Tuple[DomainBridge, Dict[str, List[str]]]:
    """Wire up the confined-shelf domain bridge.

    Args:
        env: A built :class:`~tampanda.environments.franka_env.FrankaEnvironment`
            (typically returned from
            :func:`make_confined_shelf_builder`'s builder).
        workspace: The :class:`Workspace` returned alongside that builder.
        config: The :class:`ConfinedShelfConfig` used at build time —
            needed for sentinel ``hide_far_x`` and per-cylinder z-extent.
        cylinder_colors: Per-cylinder colour group name, indexed by
            cylinder index.  Length must equal ``config.n_cylinders``.
            Common groups: ``"red"``, ``"green"``, ...; the strings are
            passed through verbatim as PDDL ``color`` constants.
        region_name: Region key in the workspace that the cylinders
            occupy.  Default ``"shelf_interior"``.
        executor: Optional :class:`~tampanda.planners.pick_place.PickPlaceExecutor`.
            When given, ``pick`` and ``put`` are registered as action
            executors that drive real robot motion.  Omit to use the
            bridge for grounding-only / planning-only.

    Returns:
        ``(bridge, objects)`` where *objects* is the
        ``{type_name: [names]}`` dict ready to pass to ``ground_state``,
        ``plan``, ``execute_action``.
    """
    if region_name not in workspace.regions:
        raise KeyError(f"region {region_name!r} not in workspace")
    region = workspace[region_name]
    if not isinstance(region, GridRegion):
        raise TypeError(
            f"region {region_name!r} must be a GridRegion, got {type(region).__name__}"
        )

    if len(cylinder_colors) != config.n_cylinders:
        raise ValueError(
            f"cylinder_colors length {len(cylinder_colors)} != "
            f"n_cylinders {config.n_cylinders}"
        )

    cylinder_names = [f"cyl_{i}" for i in range(config.n_cylinders)]
    cell_names = [c.id for c in region.cells()]
    color_names = sorted(set(cylinder_colors))
    objects: Dict[str, List[str]] = {
        "cylinder": cylinder_names,
        "cell": cell_names,
        "color": color_names,
    }

    bridge = DomainBridge(_PDDL_PATH, env)

    # ---- Cylinder lookup helpers ----------------------------------------

    def _cyl_world_pos(cyl_name: str) -> np.ndarray:
        pos, _ = env.get_object_pose(cyl_name)
        return np.asarray(pos)

    def _is_active(cyl_name: str) -> bool:
        """A cylinder is active iff it is not parked at the sentinel x."""
        return _cyl_world_pos(cyl_name)[0] < config.hide_far_x - 1.0

    def _resting_z() -> float:
        return region.level_z

    def _cell_for_cyl(cyl_name: str) -> Optional[Cell]:
        """Return the cell occupied by ``cyl_name``, or ``None``.

        A cylinder is considered to be in a cell iff:
          * It is active (not parked).
          * Its world XY lies inside the region.
          * Its world Z is within :data:`_Z_TOLERANCE` of the resting plane
            (i.e., it isn't being held mid-air).
        """
        if not _is_active(cyl_name):
            return None
        p = _cyl_world_pos(cyl_name)
        if abs(p[2] - _resting_z()) > _Z_TOLERANCE:
            return None
        return region.cell_for(float(p[0]), float(p[1]), float(p[2]))

    # ---- Code-evaluated predicates --------------------------------------

    @bridge.predicate("occupied")
    def eval_occupied(env, fluents, cell_id: str, cyl_name: str) -> bool:
        target = workspace.cell(cell_id) if "__" in cell_id else None
        if target is None:
            return False
        cur = _cell_for_cyl(cyl_name)
        return cur == target

    @bridge.predicate("empty")
    def eval_empty(env, fluents, cell_id: str) -> bool:
        target = workspace.cell(cell_id) if "__" in cell_id else None
        if target is None:
            return False
        for cyl in cylinder_names:
            if _cell_for_cyl(cyl) == target:
                return False
        return True

    @bridge.predicate("color-of")
    def eval_color(env, fluents, cyl_name: str, color_name: str) -> bool:
        idx = cylinder_names.index(cyl_name)
        return cylinder_colors[idx] == color_name

    # ---- Fluent predicates ---------------------------------------------

    bridge.fluent("holding", initial=None)
    bridge.fluent("gripper-empty", initial=True)

    # ---- Action executors ----------------------------------------------

    if executor is not None:

        @bridge.action("pick")
        def exec_pick(env, fluents, cyl_name: str, cell_id: str):
            pos = _cyl_world_pos(cyl_name)
            half = np.array([config.cylinder_radius, config.cylinder_radius,
                             config.cylinder_half_height])
            quat = env.get_object_orientation(cyl_name)
            ok = executor.pick(cyl_name, pos, half, quat)
            if not ok:
                return False, {}
            return True, {
                ("holding", cyl_name): True,
                ("gripper-empty",): False,
                ("occupied", cell_id, cyl_name): False,
                ("empty", cell_id): True,
            }

        @bridge.action("put")
        def exec_put(env, fluents, cyl_name: str, cell_id: str):
            target = workspace.cell(cell_id)
            cx, cy, cz = workspace.pose_for(target)
            place_pos = np.array([cx, cy, cz])
            # Reuse the pick's grasp orientation — closed-top cubicles
            # use FRONT (palm-+y), and forcing a palm-down place quat
            # makes the approach pose unreachable.
            ok = executor.place(cyl_name, place_pos, ee_quat=None)
            if not ok:
                return False, {}
            return True, {
                ("holding", cyl_name): False,
                ("gripper-empty",): True,
                ("occupied", cell_id, cyl_name): True,
                ("empty", cell_id): False,
            }

    return bridge, objects


def set_cylinders_at_cells(
    env,
    workspace: Workspace,
    config: ConfinedShelfConfig,
    placements: Dict[str, Cell],
    region_name: str = "shelf_interior",
) -> None:
    """Teleport listed cylinders to their target cell centres in the sim.

    Cylinders whose names appear as keys in ``placements`` are placed at the
    corresponding cell's world XYZ.  Cylinders not listed are parked at the
    hide sentinel ``(cfg.hide_far_x, 0, half_height)``.

    This is the inverse operation of :func:`_cell_for_cyl` in the bridge —
    it lets a problem-construction script stage an arbitrary initial state
    in the simulation, ready to be grounded by ``bridge.ground_state``.
    """
    region = workspace[region_name]
    if not isinstance(region, GridRegion):
        raise TypeError(f"region {region_name!r} must be a GridRegion")

    parked_xyz = np.array([config.hide_far_x, 0.0, config.cylinder_half_height])

    # First park everyone, then place the listed ones.
    for i in range(config.n_cylinders):
        env.set_object_pose(f"cyl_{i}", parked_xyz)

    for cyl_name, cell in placements.items():
        x, y, z = workspace.pose_for(cell)
        env.set_object_pose(cyl_name, np.array([x, y, z]))

    env.reset_velocities()
    env.forward()
