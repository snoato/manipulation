"""State restore + stability check for multilevel_blocks.

Provides the inverse of :func:`bridge._block_cells` — given a grounded
PDDL state, restore the MuJoCo simulation to match.  Used by the
feasibility checker and BFS-based data-gen pipeline so transitions can
be probed from arbitrary symbolic states.

Public API::

    from tampanda.symbolic.domains.multilevel_blocks.state import (
        restore_state, ground_to_block_layout, check_stability,
    )

The restore handles every ``(in block cell)`` predicate.  Held-block
fluents (``held-cube`` etc.) require an executor to set up the arm +
attachment; without one they're treated as "block parked, gripper
empty" with a warning.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    oblong_block_name,
)


# Default world position for parked / off-grid blocks.  Matches
# ``MultilevelBlocksConfig.hide_far_x = 100.0`` so the bridge's
# ``_block_cells`` correctly returns empty for parked bodies.
_PARKED_XYZ: Tuple[float, float, float] = (100.0, 0.0, 0.05)

# Quaternions per orientation (Hamilton wxyz).  Matches the conventions
# used by the bridge's ``_orientation_from_quat`` inverse.
_HALF_PI_COS = math.cos(math.pi / 4)
_HALF_PI_SIN = math.sin(math.pi / 4)

_QUAT_FLAT_X = np.array([1.0, 0.0, 0.0, 0.0])
_QUAT_FLAT_Y = np.array([_HALF_PI_COS, 0.0, 0.0, _HALF_PI_SIN])
_QUAT_UPRIGHT = np.array([_HALF_PI_COS, 0.0, _HALF_PI_SIN, 0.0])

# Held-block fluent names (matches bridge.fluent declarations).
_HELD_FLUENTS = ("held-cube", "held-flat-x", "held-flat-y", "held-upright")


# ---------------------------------------------------------------------------
# Predicate extraction helpers
# ---------------------------------------------------------------------------


def ground_to_block_layout(
    state: Dict[Tuple, bool],
    block_names: List[str],
) -> Dict[str, List[str]]:
    """Extract ``{block: [cell_ids]}`` from the ground-state dict.

    Only ``("in", block, cell)`` entries with True value are included.
    Held blocks (with any of the ``held-*`` fluents true) are omitted —
    callers should check for them separately.
    """
    layout: Dict[str, List[str]] = {b: [] for b in block_names}
    for key, value in state.items():
        if not value:
            continue
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == "in"):
            continue
        _, block, cell = key
        if block in layout:
            layout[block].append(cell)
    return layout


def held_block_in_state(state: Dict[Tuple, bool]) -> Optional[Tuple[str, str]]:
    """Return ``(fluent_name, block_name)`` if any held-* fluent is True.

    Returns ``None`` if the gripper is empty.  At most one block may be
    held at a time; the first true fluent wins.
    """
    for key, value in state.items():
        if not value:
            continue
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        fluent, block = key
        if fluent in _HELD_FLUENTS:
            return fluent, block
    return None


# ---------------------------------------------------------------------------
# Shape / orientation inference
# ---------------------------------------------------------------------------


def _block_shape(name: str, cfg: MultilevelBlocksConfig) -> str:
    """Return ``"cube" | "oblong" | "long"`` based on block-name index sets."""
    cube_set = {cube_block_name(i) for i in range(cfg.n_cubes)}
    long_set = {long_block_name(i) for i in range(cfg.n_long)}
    if name in cube_set:
        return "cube"
    if name in long_set:
        return "long"
    return "oblong"


def _classify_cells(
    cells: List[Cell],
) -> Tuple[str, List[Cell]]:
    """Identify orientation from a list of occupied cells.

    Returns ``(orientation, sorted_cells)`` where ``orientation`` is one
    of:
      * ``"single"`` — 1 cell (cube).
      * ``"flat-x"`` / ``"flat-y"`` — 2 cells differing in ix/iy at the
        same region.
      * ``"upright-2"`` — 2 cells stacked across adjacent regions
        (oblong upright).
      * ``"long-x"`` / ``"long-y"`` — 3 cells in a line along ix/iy.
      * ``"long-upright"`` — 3 cells stacked across 3 adjacent regions.
      * ``"unknown"`` — any other combination.

    ``sorted_cells`` is the input list in ascending (ix, iy, region)
    order so the centroid can be computed deterministically.
    """
    if len(cells) == 1:
        return "single", list(cells)
    regions = sorted({c.region for c in cells})
    if len(cells) == 2:
        if len(regions) == 2:
            return "upright-2", sorted(cells, key=lambda c: c.region)
        ixs = sorted({c.ix for c in cells})
        iys = sorted({c.iy for c in cells})
        if len(ixs) == 2 and len(iys) == 1:
            return "flat-x", sorted(cells, key=lambda c: c.ix)
        if len(iys) == 2 and len(ixs) == 1:
            return "flat-y", sorted(cells, key=lambda c: c.iy)
        return "unknown", list(cells)
    if len(cells) == 3:
        if len(regions) == 3:
            return "long-upright", sorted(cells, key=lambda c: c.region)
        ixs = sorted({c.ix for c in cells})
        iys = sorted({c.iy for c in cells})
        if len(ixs) == 3 and len(iys) == 1:
            return "long-x", sorted(cells, key=lambda c: c.ix)
        if len(iys) == 3 and len(ixs) == 1:
            return "long-y", sorted(cells, key=lambda c: c.iy)
        return "unknown", list(cells)
    return "unknown", list(cells)


def _centroid_and_quat(
    workspace: Workspace,
    orientation: str,
    sorted_cells: List[Cell],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute world centroid + quaternion for a cell set + orientation.

    Centroid is the average of the cell world positions.  Quaternion
    matches the canonical orientation used by the bridge inverse.
    """
    positions = np.array([workspace.pose_for(c) for c in sorted_cells])
    centroid = positions.mean(axis=0)
    if orientation == "single":
        quat = _QUAT_FLAT_X
    elif orientation in ("flat-x", "long-x"):
        quat = _QUAT_FLAT_X
    elif orientation in ("flat-y", "long-y"):
        quat = _QUAT_FLAT_Y
    elif orientation in ("upright-2", "long-upright"):
        quat = _QUAT_UPRIGHT
    else:
        quat = _QUAT_FLAT_X
    return centroid, quat


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def restore_state(
    env,
    workspace: Workspace,
    config: MultilevelBlocksConfig,
    state: Dict[Tuple, bool],
    *,
    parked_xyz: Tuple[float, float, float] = _PARKED_XYZ,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "park",
) -> Dict[str, Any]:
    """Restore MuJoCo state to match the given PDDL ``state``.

    Args:
        env: ``FrankaEnvironment``.
        workspace: ``Workspace``.
        config: ``MultilevelBlocksConfig``.
        state: ground-state dict from ``bridge.ground_state(...)``.
        parked_xyz: world position for blocks not in any ``(in ...)``
            predicate.
        home_qpos: optional 7-DoF arm config.  ``None`` leaves the arm
            untouched.
        on_held: how to handle held-block predicates.  Choices:
            ``"park"`` (default) — park the held block, clear the
            gripper fluent; ``"warn"`` — same but logs a warning;
            ``"raise"`` — raise NotImplementedError.

    Returns:
        Summary dict ``{"placed": [(name, cells)], "parked": [...],
        "held": (fluent, block) or None, "unknown": [...]}``.
    """
    if on_held not in ("park", "warn", "raise"):
        raise ValueError(f"on_held must be park/warn/raise, got {on_held!r}")

    # 1. Reset env hygiene — drop attachments, open gripper, clear
    # collision exceptions.  Don't run physics; the caller's set_object_pose
    # calls will re-establish a consistent state.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    if env.controller is not None:
        env.controller.open_gripper()
        # Set ctrl[7] to the actual qpos target so the actuator settles
        # at full-open without applying a tendon force on first forward.
        env.data.ctrl[7] = 0.04

    # 2. Build the block roster.
    block_names: List[str] = []
    block_names += [cube_block_name(i) for i in range(config.n_cubes)]
    block_names += [oblong_block_name(i) for i in range(config.n_oblong)]
    block_names += [long_block_name(i) for i in range(config.n_long)]

    # 3. Park everyone first.  Subsequent set_object_pose calls override
    # for placed blocks.
    parked_arr = np.asarray(parked_xyz, dtype=float)
    for name in block_names:
        env.set_object_pose(name, parked_arr)

    # 4. Place blocks per the (in ...) predicates.
    layout = ground_to_block_layout(state, block_names)
    placed: List[Tuple[str, List[str]]] = []
    unknown: List[Tuple[str, List[str]]] = []
    for block, cell_ids in layout.items():
        if not cell_ids:
            continue
        cells = [Cell.parse(cid) for cid in cell_ids]
        orientation, sorted_cells = _classify_cells(cells)
        if orientation == "unknown":
            unknown.append((block, cell_ids))
            continue
        centroid, quat = _centroid_and_quat(workspace, orientation,
                                                  sorted_cells)
        # Apply per-shape z correction for upright blocks: the cell-centre
        # average assumes the block centroid sits at the average region z,
        # which is correct for stacked cells (cube_size pitch matches
        # level_z pitch).  No correction needed.
        env.set_object_pose(block, centroid, quat)
        placed.append((block, [c.id for c in sorted_cells]))

    # 5. Handle held-block fluents.
    held = held_block_in_state(state)
    if held is not None:
        fluent, block = held
        if on_held == "raise":
            raise NotImplementedError(
                f"Restore of held-block state ({fluent} {block}) requires "
                "an executor; pass on_held='park' to fall back to parking."
            )
        if on_held == "warn":
            print(f"[restore_state] WARNING: parking held block "
                      f"{block!r} (fluent {fluent!r}); attachment not "
                      "restored.")
        env.set_object_pose(block, parked_arr)

    # 6. Restore arm + run forward to update derived MuJoCo state.
    if home_qpos is not None:
        env.data.qpos[: len(home_qpos)] = np.asarray(home_qpos, dtype=float)
    env.reset_velocities()
    env.forward()

    return {
        "placed": placed,
        "parked": [b for b in block_names
                       if b not in {p[0] for p in placed}
                       and (held is None or b != held[1])],
        "held": held,
        "unknown": unknown,
    }


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------


def check_stability(
    env,
    *,
    settle_steps: int = 300,
    movement_threshold: float = 0.005,
    tracked_blocks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Step physics with no controller input; report per-block displacement.

    Used after a placement to detect topple — if any tracked block moves
    by more than ``movement_threshold`` (m) during the settle, the stack
    is unstable.

    Args:
        env: ``FrankaEnvironment``.
        settle_steps: physics steps to run.
        movement_threshold: per-block max displacement that counts as
            "moved" (default 5 mm).
        tracked_blocks: list of block names to track; ``None`` tracks
            every body that ``env.get_object_pose`` accepts.

    Returns:
        ``{"max_displacement", "per_block", "stable"}``.
    """
    if tracked_blocks is None:
        # Try to discover from MuJoCo body names that look like blocks.
        # The bridge's block roster isn't exposed here, so fall back to
        # any body matching cube_/oblong_/long_ prefix.
        names: List[str] = []
        for bid in range(env.model.nbody):
            n = env.model.body(bid).name
            if n.startswith(("cube_", "oblong_", "long_")):
                names.append(n)
        tracked_blocks = names

    start_pos: Dict[str, np.ndarray] = {}
    for name in tracked_blocks:
        try:
            pos, _ = env.get_object_pose(name)
            start_pos[name] = np.asarray(pos, dtype=float).copy()
        except Exception:
            continue

    # Settle.  Use mj_step directly (no controller, no rate.sleep) so
    # the headless path is as fast as possible.
    for _ in range(settle_steps):
        mujoco.mj_step(env.model, env.data)

    per_block: Dict[str, float] = {}
    for name, p0 in start_pos.items():
        try:
            pos, _ = env.get_object_pose(name)
            displacement = float(np.linalg.norm(np.asarray(pos) - p0))
        except Exception:
            displacement = float("nan")
        per_block[name] = displacement

    max_disp = max(per_block.values()) if per_block else 0.0
    stable = max_disp < movement_threshold

    return {
        "max_displacement": max_disp,
        "per_block": per_block,
        "stable": stable,
    }
