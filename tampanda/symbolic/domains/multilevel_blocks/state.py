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

# Z-offset from EE attachment site to the centroid of a top-down grasped
# block.  Mirrors executor._EE_TO_BLOCK_CENTRE_Z — duplicated here to
# avoid importing from executor.py (which would create a cycle via
# bridge.register_executor).
_EE_TO_BLOCK_CENTRE_Z = 0.014


def _set_block_collision(env, body_name: str, enabled: bool) -> None:
    """Toggle contype + conaffinity on every geom of a block body.

    Parked blocks (those NOT in any ``(in ...)`` predicate) live at
    ``_PARKED_XYZ`` — the SAME world point for every block.  With default
    ``contype = conaffinity = 1``, MuJoCo's broadphase finds every pair
    of parked blocks AABB-overlapping (~ 38 × 37 / 2 ≈ 700 mutual pairs
    for the rgnet 38-block config) and runs narrow-phase contact
    computation on all of them.  This was the dominant cost in
    ``mj_forward`` (~5 ms / call out of ~6 ms total on the rgnet config).

    Disabling collision on parked blocks makes the broadphase skip them
    entirely; per-call ``mj_forward`` drops from ~5 ms to <0.2 ms.

    Re-enable when the block is placed at a real cell or attached to
    the EE (held-body collision tracking expects active contype).
    """
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    flag = 1 if enabled else 0
    for gid in range(env.model.ngeom):
        if env.model.geom_bodyid[gid] == body_id:
            env.model.geom_contype[gid] = flag
            env.model.geom_conaffinity[gid] = flag


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


def _attach_held_block(
    env,
    config: MultilevelBlocksConfig,
    fluent: str,
    block_name: str,
    executor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set up kinematic attachment for a held block, mirroring the world
    state the real pick chain would leave behind.

    Teleports the arm to the executor's precomputed handoff matching
    ``fluent`` (top-down for held-cube / held-flat-x / held-flat-y; front
    for held-upright), places the block at the canonical world-frame
    offset from the EE for that grasp family, snaps fingers, and calls
    :meth:`env.attach_object_to_ee`.  Also writes ``executor._held_block``
    and ``executor._held_offset`` so subsequent put / transform methods
    read the same state as if a real pick had just run.

    Args:
        env: ``FrankaEnvironment`` — must have an attached
            ``mocap`` target body (the franka base XML provides this).
        config: ``MultilevelBlocksConfig`` — needed for ``cube_size``
            and shape inference.
        fluent: one of the four held-* fluent names.
        block_name: e.g. ``"cube_3"``, ``"oblong_1"``, ``"long_0"``.
        executor: ``MultilevelBlocksExecutor`` (or subclass) whose
            ``_handoff_qs`` mapping is already populated.  ``_held_block``
            and ``_held_offset`` are written.

    Returns:
        ``(ee_pos, block_world_pos)`` — the EE world position and the
        chosen block world position; useful for callers who want to
        verify the attachment geometry.

    Raises:
        ValueError: ``fluent`` is not a known held-* fluent.
        RuntimeError: executor is missing the required precomputed
            handoff key.
    """
    if fluent not in _HELD_FLUENTS:
        raise ValueError(
            f"_attach_held_block: unknown fluent {fluent!r}; expected one "
            f"of {_HELD_FLUENTS}"
        )

    # Pick the canonical handoff for this fluent family.  Use the STACK
    # handoff (matches where post-pick chains terminate by default; the
    # block's world XY doesn't actually matter for the kinematic attach
    # — only the relative pose to the EE does, and at any handoff that
    # rel pose is the same).
    if fluent == "held-upright":
        handoff_key = ("stack", "front")
    else:
        handoff_key = ("stack", "top_down")

    if not hasattr(executor, "_handoff_qs"):
        raise RuntimeError(
            "executor has no _handoff_qs; pass an instance of "
            "MultilevelBlocksExecutor (or FastFeasibilityExecutor)."
        )
    if handoff_key not in executor._handoff_qs:
        raise RuntimeError(
            f"executor missing precomputed handoff {handoff_key!r}; "
            f"available={list(executor._handoff_qs)}"
        )

    # Detach anything previously attached and clear the held-collision
    # body — keeps idempotent in case restore is called twice in a row.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()

    # Teleport the arm to the handoff.
    handoff_q = executor._handoff_qs[handoff_key]
    env.data.qpos[:7] = handoff_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    # Read the EE world pose after teleport.
    ee_pos = env.data.site_xpos[executor.ee_site_id].copy()

    # World-frame block centroid relative to the EE site.
    #
    # For top-down grasps (held-cube / held-flat-x / held-flat-y): the
    # gripper points -world_z, the EE site sits 14mm above the block
    # centroid (matches executor._EE_TO_BLOCK_CENTRE_Z).
    #
    # For front grasps (held-upright): the gripper points +world_y; the
    # real pick anchors the EE at the UPPER cell of the upright block.
    # The block centroid is 0.5*cube_size below the EE in world_z for a
    # 2-cell oblong, or 1.0*cube_size below for a 3-cell long block
    # (pick_long_upright also grasps at the top cell).
    cube_size = config.cube_size
    shape = _block_shape(block_name, config)

    if fluent == "held-upright":
        if shape == "long":
            world_offset = np.array([0.0, 0.0, -cube_size])
        else:
            world_offset = np.array([0.0, 0.0, -cube_size / 2.0])
        block_quat = _QUAT_UPRIGHT
    elif fluent == "held-flat-y":
        world_offset = np.array([0.0, 0.0, -_EE_TO_BLOCK_CENTRE_Z])
        block_quat = _QUAT_FLAT_Y
    else:
        # held-cube or held-flat-x
        world_offset = np.array([0.0, 0.0, -_EE_TO_BLOCK_CENTRE_Z])
        block_quat = _QUAT_FLAT_X

    block_world_pos = ee_pos + world_offset
    env.set_object_pose(block_name, block_world_pos, block_quat)
    # Re-enable collision on the held block (was disabled in the
    # restore_state parking pass).  Held-body collision machinery
    # requires active contype/conaffinity to register contacts.
    _set_block_collision(env, block_name, enabled=True)

    # Snap fingers to closed-on-block — mirrors FastFeasibilityExecutor._close_attach.
    cube_half = config.cube_half_extent
    env.data.qpos[7] = cube_half
    env.data.qpos[8] = cube_half
    mujoco.mj_forward(env.model, env.data)

    # Capture relative pose by calling attach_object_to_ee without
    # canonical_rel_pos — the block is already at the right offset.
    env.attach_object_to_ee(block_name)

    # Mirror executor._close_attach bookkeeping so subsequent put /
    # transform methods read the correct held state.
    executor._held_block = block_name
    executor._held_offset = block_world_pos - ee_pos

    return ee_pos, block_world_pos


def restore_state(
    env,
    workspace: Workspace,
    config: MultilevelBlocksConfig,
    state: Dict[Tuple, bool],
    *,
    parked_xyz: Tuple[float, float, float] = _PARKED_XYZ,
    home_qpos: Optional[np.ndarray] = None,
    on_held: str = "park",
    executor=None,
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
            ``"raise"`` — raise NotImplementedError; ``"attach"`` —
            kinematically attach the block to the EE via
            :func:`_attach_held_block` (requires ``executor``).
        executor: ``MultilevelBlocksExecutor`` instance, required when
            ``on_held="attach"``.  Ignored otherwise.

    Returns:
        Summary dict ``{"placed": [(name, cells)], "parked": [...],
        "held": (fluent, block) or None, "unknown": [...]}``.
    """
    valid_on_held = ("park", "warn", "raise", "attach")
    if on_held not in valid_on_held:
        raise ValueError(
            f"on_held must be one of {valid_on_held}, got {on_held!r}"
        )
    if on_held == "attach" and executor is None:
        raise ValueError(
            "restore_state(on_held='attach') requires executor=; pass "
            "the MultilevelBlocksExecutor (or FastFeasibilityExecutor) "
            "instance.  Without the executor's precomputed handoff "
            "configs the attach cannot place the arm correctly."
        )

    # 1. Reset env hygiene — drop attachments, open gripper, clear
    # collision exceptions.  Don't run physics; the caller's set_object_pose
    # calls will re-establish a consistent state.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    if hasattr(env, "clear_collision_exceptions"):
        env.clear_collision_exceptions()
    # Reset executor's held-state bookkeeping too, so callers that reuse
    # a single executor across many restore_state invocations (e.g. the
    # plan-validation harness or the rgnet feasibility checker) start
    # each problem with a clean held state.  If the new state has a
    # held-* fluent and on_held="attach", _attach_held_block (below)
    # re-sets these.
    if executor is not None:
        executor._held_block = None
        executor._held_offset = np.zeros(3)
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

    # 3. Park everyone first AND disable collision on every block — this
    # is critical for mj_forward speed because parked blocks all share
    # the same world position, and with default contype/conaffinity=1
    # they pile up as O(N^2) mutual contact candidates.  Re-enabling
    # collision happens below (step 4 for placed, step 5 for held).
    parked_arr = np.asarray(parked_xyz, dtype=float)
    for name in block_names:
        env.set_object_pose(name, parked_arr)
        _set_block_collision(env, name, enabled=False)

    # 4. Place blocks per the (in ...) predicates.  Re-enable collision
    # on placed blocks so the arm correctly sees them.
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
        _set_block_collision(env, block, enabled=True)
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
        if on_held == "attach":
            # NB: this overwrites the parked pose set above and the arm
            # qpos that the caller may pass via home_qpos — by design.
            # The attach needs the arm at a handoff config so the held
            # block lands at the right pose; we apply home_qpos below
            # *only* when no held block needs to be attached.
            _attach_held_block(env, config, fluent, block, executor)
        else:
            # park / warn: drop the block at the parked sentinel.
            env.set_object_pose(block, parked_arr)

    # 6. Restore arm + run forward to update derived MuJoCo state.
    # Skip the home_qpos write when we just attached a held block: the
    # attach step put the arm at a handoff config, and overwriting it
    # would tear the block off the EE.
    if home_qpos is not None and not (held is not None and on_held == "attach"):
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
