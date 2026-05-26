"""MP-based action executor for the redesigned multilevel_blocks domain.

Each of the 14 PDDL actions maps to a Python method that:

* Generates the grasp/place quaternion candidates per the symmetry of the
  block shape (4 yaws for cube and upright oblong, 2 yaws for flat oblong).
* Attempts IK convergence at each candidate; the first one that succeeds
  drives the linear-IK chain.
* Executes the chain end-to-end (approach → descend → grasp → lift, or the
  reverse for puts).

The executor is registered into a :class:`DomainBridge` via the bridge's
``@action`` decorator so the planner-emitted plan can be dispatched directly.

In-hand transforms (``make-upright-from-x`` etc.) rotate the gripper in free
air above the workspace; the bridge MP performs a short lift → rotate →
descend sequence, keeping the held block attached.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
    cube_block_name,
    oblong_block_name,
)


# ---------------------------------------------------------------------------
# Grasp quaternion helpers
# ---------------------------------------------------------------------------


def _yaw_quat(yaw: float) -> np.ndarray:
    """Quaternion for a rotation of ``yaw`` radians around world-z."""
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication ``q1 * q2`` (Hamilton, wxyz convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _gripper_z_world(quat: np.ndarray) -> np.ndarray:
    """World-frame +z axis of the gripper (palm direction) for a given quat.

    The gripper's body z-axis is the palm direction — where the parallel-
    jaw fingers point.  For a grasp approach, the gripper comes from the
    OPPOSITE direction, so ``approach_pos = anchor - gripper_z_world *
    approach_height`` places the pre-grasp staging on the correct side
    of the block regardless of which yaw IK selected.

    Quat is Hamilton wxyz; result is the third column of the rotation
    matrix.
    """
    w, x, y, z = quat
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


# Top-down base (gripper z-axis pointing -world-z, gripper x-axis aligned with
# world-x).  Adapted from GraspPlanner._QUAT_TOP_DOWN_X.
_QUAT_TOP_DOWN_X = np.array([0.0, 1.0/math.sqrt(2.0), 1.0/math.sqrt(2.0), 0.0])
# Top-down with fingers along world-y (gripper rotated 90 deg around its z).
_QUAT_TOP_DOWN_Y = np.array([0.0, 1.0, 0.0, 0.0])

# Front-facing base (gripper z-axis along +world-y).  Adapted from access19's
# FRONT_QUAT used for grasping the front of a cubicle.
_QUAT_FRONT_Y = np.array([-0.5, 0.5, 0.5, 0.5])
# Front-facing rotated 90 deg around the gripper's own z (180 deg of yaw
# around the approach axis).
_QUAT_FRONT_Y_FLIP = np.array([-0.5, 0.5, -0.5, -0.5])


def _cube_grasp_quats() -> List[np.ndarray]:
    """Four top-down grasp orientations for a 1×1 cube.

    Symmetric under 90 deg rotation about world-z; parallel-jaw symmetry
    collapses 0°↔180° and 90°↔270° so only 2 distinct grasps remain, but
    trying 4 IK seeds increases convergence robustness.
    """
    base = _QUAT_TOP_DOWN_X
    return [_quat_mul(_yaw_quat(yaw), base)
            for yaw in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)]


def _upright_grasp_quats() -> List[np.ndarray]:
    """Eight front-facing grasp orientations for an upright 2×1 oblong.

    The block has TWO independent rotational symmetries:
      * 4-fold around world-z (square 30×30 mm cross-section, 4 lateral
        faces interchangeable) → 4 yaws.
      * 2-fold around the gripper's palm axis / block's long axis
        (block is symmetric end-to-end along z, so a 180° roll
        produces the same final placement) → 2 rolls per yaw.

    4 yaws × 2 rolls = 8 candidates that all yield the SAME block
    placement but with different wrist configurations.  We probe all 8
    in the filter because IK basins differ between them — at
    corner / dense-neighbour cells only specific (yaw, roll)
    combinations IK-converge.  Parallel-jaw symmetry further halves the
    distinct-grasp count, but the redundant probes are cheap relative
    to the cost of an upright failure mid-build.
    """
    base = _QUAT_FRONT_Y
    # 180° rotation around body z (palm axis).
    roll_180 = np.array([0.0, 0.0, 0.0, 1.0])
    yaw_quats = [_quat_mul(_yaw_quat(yaw), base)
                  for yaw in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    rolled = [_quat_mul(q, roll_180) for q in yaw_quats]
    return yaw_quats + rolled


def _upright_grasp_quats_sorted(centroid_xyz: np.ndarray) -> List[np.ndarray]:
    """Heuristically-sorted upright grasp quats.

    Sorts the 8 quats by alignment of their palm direction (gripper
    body z-axis in world frame) with the vector from the robot base
    origin to ``centroid_xyz`` (block centre).  The quat whose palm
    points most directly toward the block is tried first — that's the
    "natural" front approach for the block's xy location, and on the
    full reliability matrix it's the quat that IK-converges the
    fastest.  Probing it first cuts upright filter cost by ~4x on
    average when only 1-2 of the 8 quats actually succeed.

    Tie-breaking: original-order index (so the un-rolled yaw beats the
    rolled variant when both have the same palm dot, since the un-rolled
    has the more natural wrist config).
    """
    all_quats = _upright_grasp_quats()
    base_to_block = np.asarray(centroid_xyz[:2], dtype=float)
    norm = np.linalg.norm(base_to_block)
    if norm < 1e-6:
        return all_quats
    base_to_block = base_to_block / norm
    target3 = np.array([base_to_block[0], base_to_block[1], 0.0])

    def score(idx_q):
        idx, q = idx_q
        gz = _gripper_z_world(q)
        gz_xy = np.array([gz[0], gz[1], 0.0])
        n = np.linalg.norm(gz_xy)
        if n < 1e-6:
            return (-1.0, idx)
        gz_xy = gz_xy / n
        # Higher dot ↔ palm aligned with target direction (= approach
        # from base side INTO the block).  Sort descending by dot, then
        # ascending by original index for tie-breaking.
        return (float(np.dot(gz_xy, target3)), -idx)

    ordered = sorted(enumerate(all_quats), key=score, reverse=True)
    return [q for _, q in ordered]


def _flat_x_grasp_quats() -> List[np.ndarray]:
    """Two top-down grasp orientations for a flat-x oblong.

    The block long axis is along world-x; gripper fingers MUST close
    perpendicular (along ±y).  Two yaws give the two 180 deg-flipped grasps,
    which are equivariant under gripper symmetry.
    """
    base = _QUAT_TOP_DOWN_Y  # fingers close along y
    return [_quat_mul(_yaw_quat(yaw), base) for yaw in (0.0, math.pi)]


def _flat_y_grasp_quats() -> List[np.ndarray]:
    """Two top-down grasp orientations for a flat-y oblong.

    Long axis along world-y; fingers must close along ±x.
    """
    base = _QUAT_TOP_DOWN_X
    return [_quat_mul(_yaw_quat(yaw), base) for yaw in (0.0, math.pi)]


# ---------------------------------------------------------------------------
# Home / staging seeds (one per workspace direction)
# ---------------------------------------------------------------------------


_HOME_STACK = np.array([np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
_HOME_PARTS = np.array([-np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def _home_for_cell(cell: Cell) -> np.ndarray:
    """Pick the canonical IK seed for the cell's region."""
    return _HOME_PARTS if cell.region == "parts" else _HOME_STACK


# ---------------------------------------------------------------------------
# Geometric constants
# ---------------------------------------------------------------------------


_APPROACH_HEIGHT = 0.10   # m above the cell's level_z for staging
_LIFT_HEIGHT = 0.08       # m above the cell's level_z after grasp
_PLACE_CLEARANCE = 0.002  # m above target before release (avoids contact during settle)
_HIGH_ABOVE_HEIGHT = 0.15 # m above the cell's level_z for the high-above staging
                          # pose at the start of put_top_down — lower than
                          # the handoff z so corner/edge cells stay within reach
# z-offset from EE attachment site to the centre of the grasped block.
# Franka panda hand: attachment_site is at hand-z=0.09; fingertip pads
# extend to hand-z=0.10.  So the fingertips are only ~0.014 m past the
# EE site along the gripper z-axis.  For a top-down grasp at a cube's
# centroid, EE.z = cube_centre.z + 0.014.  Larger offsets push the
# fingertips below the cube bottom and the jaws close on air.
# Matches ``GRASP_CONTACT_OFFSET = 0.0137`` from the access-19 chain.
_EE_TO_BLOCK_CENTRE_Z = 0.014

# Hand-off poses: one per (workspace, orientation) combination.  These
# are EE positions ABOVE each table at a moderate height — every pick
# ends at its workspace's top-down hand-off, and every put starts by
# transitioning to its workspace's hand-off in the correct orientation.
# In-hand transforms (held-flat-x ↔ held-upright, etc.) are performed
# at the stack hand-off by lerping between the two orientation poses.
#
# Workspace transition (parts → stack or vice versa) is a single lerp
# between two hand-off poses at the same z, with a base-joint swing of
# ~π.  Above the tables so no collision risk.
# Hand-off positions are derived from the current ``cfg.{parts,stack}_table_pos``
# by ``_compute_handoff_pos``, so they automatically track a runtime
# table-move.  Constants:
_HANDOFF_STACK_FRONT_Y_OFFSET = 0.20   # how far in front of stack table
_HANDOFF_HEIGHT_ABOVE_TOP = 0.28        # height of hand-off above the
                                          # table's TOP face (not its body
                                          # origin).  Original geometry
                                          # had handoff.z = 0.55 = 0.27
                                          # (table top) + 0.28 (this).
_TABLE_TOP_LOCAL_Z = 0.27               # body-local z of the table top
                                          # geom (matches env_builder).

# IK seeds (q[0] is the base joint — ±π/2 to face -y / +y respectively).
_HOME_PARTS_HANDOFF = np.array(
    [-np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
_HOME_STACK_HANDOFF = np.array(
    [+np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Neutral HOME pose — base centred (q[0] = 0, between the parts and
# stack workspaces), arm folded high so the EE sits well above both
# tables.  The executor lerps to this pose at the END of every pick
# and at the START of every put, so cross-workspace transitions go
# pick-handoff → HOME → put-handoff in TWO smaller joint-space
# segments rather than one big swing of the base joint by π.
_HOME_NEUTRAL_Q = np.array(
    [0.0, -0.8, 0.0, -2.2, 0.0, 1.5, -0.7853])

# Step sizes per execution phase, copied from the access-19 chain values.
# Travel phases are 0.01 rad; precision phases (grasp/place/lift) are
# 0.003 rad so the controller actually tracks each waypoint instead of
# blasting through, which is what was making the block fly out of the
# gripper.
_STEP_APPROACH = 0.01
_STEP_GRASP    = 0.003
_STEP_LIFT     = 0.003
_STEP_PLACE    = 0.003
_STEP_RETREAT  = 0.01

# Controller "advance delta" override during precision phases.  Default
# is 0.1 rad (~5.7°), which is way larger than the 3 mm grasp step, so
# the controller advances past waypoints without ever reaching them.
# Tightening to 0.01 rad (~0.6°) makes the controller wait for each
# waypoint before advancing.
_ADV_DELTA_PRECISION = 0.01
_ADV_DELTA_TRAVEL    = 0.1

# Settle steps after every motion to let physics damp out.
_SETTLE_STEPS = 8

# Gripper velocity thresholds for two-phase close detection.
_GRIPPER_START_VEL = 1e-3
_GRIPPER_STOP_VEL  = 5e-4

# put_upright filter: keep all 8 quats.  Tried K=4 first but the
# reliability_l0 upright matrix dropped from 256/256 to 216/256
# (40 false negatives at deep cells where only lower-priority yaws
# work).  The 8-quat sweep is the slow path on INFEAS put-upright
# (~40 s), but the cheap-IK gate (Tier-0 in _probe) catches most of
# that cost up-front — IK-unreachable quats now reject in ~50 ms
# instead of running the full Cartesian probe.  Net: INFEAS time
# drops to ~5-10 s while keeping all valid quats.  Reintroduce K
# limit only if profiling shows the cheap-IK gate isn't enough.
_PUT_UPRIGHT_QUAT_PROBE_TOPK = 8

# Pre-grasp finger width for the descent into a pick.  The default open
# spread (qpos≈0.04 each = 80 mm finger-face gap) puts the finger-pad
# OUTER edges ±48 mm from the EE, wider than the 46.5 mm hand-body
# half-extent.  At dense neighbour spacings (≤2 cells in the close-axis
# direction) those open fingertips clip the adjacent block during
# descent (verified empirically: -5.88 mm interpenetration of right
# finger pad vs neighbour at 60 mm centre-to-centre).  Pre-closing to
# 40 mm gap pulls the outer edges in to ±28 mm, restoring 17 mm
# clearance at 1-cell-gap parts spacing.  The Franka gripper actuator
# takes target finger-position in metres (ctrlrange [0, 0.04]):
# 0.02 → 20 mm per finger → 40 mm gap.
_PREGRASP_CTRL = 0.02


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MultilevelBlocksExecutor:
    """MP-based dispatcher for the 14 PDDL actions.

    Owns the linear-IK planner and the gripper attachment state; the bridge
    invokes its methods directly (each returns ``True`` on success) and
    relies on its fluent_delta wiring to update gripper state.
    """

    def __init__(self, env, workspace: Workspace,
                  config: MultilevelBlocksConfig,
                  motion_planner=None,
                  ik_pos_threshold: float = 0.005,
                  ik_ori_threshold: float = 5e-3,
                  verbose: bool = False) -> None:
        self.env = env
        self.workspace = workspace
        self.config = config
        self.verbose = verbose
        env.ik.pos_threshold = ik_pos_threshold
        env.ik.ori_threshold = ik_ori_threshold
        self.lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
        self.planner = motion_planner   # RRTStar for longer transit; None ok
        self.ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site",
        )

        # Pre-compute hand-off joint configs.  Three named hand-offs:
        # parts top-down, stack top-down, stack front-facing.  Each is
        # the IK solution for (handoff_xyz, target_quat) from the
        # canonical workspace seed.  Used at action boundaries to
        # establish a known good config in the correct IK basin.
        self._handoff_qs: Dict[Tuple[str, str], np.ndarray] = {}
        self._precompute_handoffs()

        # Stash for the world-frame (block - EE) offset of the currently
        # held block.  Refreshed at attach time and after each in-hand
        # rotation; used by put actions to compute the EE pose that
        # places the block centroid at the intended cell.
        self._held_block: Optional[str] = None
        self._held_offset: np.ndarray = np.zeros(3)

        # Per-action trace of (phase, ee_pos, block_pos) — for the grasp
        # accuracy benchmark.  Cleared at the start of every pick.
        self._trace: List[Tuple[str, np.ndarray, np.ndarray]] = []

        # IK seed LUT (loaded once per executor).  When the .npz lives
        # alongside this module, ``_seed_arm_for_cell`` uses it to warm-
        # start mink before slow IK phases; otherwise calls become
        # no-ops and the chain runs cold-start IK as before.  Built by
        # ``examples/precompute_ik_seeds.py`` — typically on workstation.
        from tampanda.symbolic.domains.multilevel_blocks.ik_seed_lut import (
            load_default as _load_ik_seed_lut,
        )
        try:
            self._ik_seed_lut = _load_ik_seed_lut(strict=False)
        except Exception:
            self._ik_seed_lut = None

        # Phase 4: optional parallel yaw pool.  ``_make_executor`` in
        # feasibility.py sets this when constructed with a pool; chain
        # phases that loop over K grasp-yaw candidates dispatch to the
        # pool when present.  None -> serial yaw probing (default).
        self._yaw_pool = None

    def _trace_phase(self, label: str) -> None:
        """Append a trace entry if a block is currently held."""
        if self._held_block is None:
            return
        ee = self._ee_pos()
        blk = self._block_pos(self._held_block)
        self._trace.append((label, ee.copy(), blk.copy()))

    def _compute_handoff_pos(self, region: str, orient: str) -> np.ndarray:
        """Derive a hand-off pose for ``(region, orient)`` from the current
        ``self.config.{parts,stack}_table_pos``.  Called every time
        ``_precompute_handoffs`` runs, so moving a table at runtime + a
        re-precompute updates the hand-offs automatically.

        The hand-off z is ``pos.z + 0.27 + 0.28`` — i.e. the TABLE TOP
        plus a 28 cm clearance.  Adding the offset to ``pos.z`` directly
        would put the hand-off 1 cm above the table (top is 27 cm above
        body origin) and the gripper fingers would penetrate the slab.
        """
        if region == "parts":
            px, py, pz = self.config.parts_table_pos
            return np.array([px, py,
                                pz + _TABLE_TOP_LOCAL_Z
                                    + _HANDOFF_HEIGHT_ABOVE_TOP])
        # stack
        sx, sy, sz = self.config.stack_table_pos
        if orient == "front":
            # Pull the FRONT hand-off back toward the robot so the arm
            # can extend forward.  The stack table's near edge is at
            # ``sy - half_extent``; the hand-off sits another 20cm in
            # front of that edge, at a height clear of any block.
            half_y = (self.config.stack_grid_cells[1]
                          * self.config.cube_size / 2
                          + self.config.table_margin)
            front_y = sy - half_y - _HANDOFF_STACK_FRONT_Y_OFFSET
            return np.array([sx, front_y,
                                sz + _TABLE_TOP_LOCAL_Z
                                    + _HANDOFF_HEIGHT_ABOVE_TOP])
        # top-down
        return np.array([sx, sy,
                            sz + _TABLE_TOP_LOCAL_Z
                                + _HANDOFF_HEIGHT_ABOVE_TOP])

    def _precompute_handoffs(self) -> None:
        specs = [
            ("parts", "top_down",
                 self._compute_handoff_pos("parts", "top_down"),
                 _QUAT_TOP_DOWN_Y, _HOME_PARTS_HANDOFF),
            ("stack", "top_down",
                 self._compute_handoff_pos("stack", "top_down"),
                 _QUAT_TOP_DOWN_Y, _HOME_STACK_HANDOFF),
            ("stack", "front",
                 self._compute_handoff_pos("stack", "front"),
                 _QUAT_FRONT_Y,   _HOME_STACK_HANDOFF),
        ]
        # Save current state so the env isn't perturbed by IK probing.
        save_q = self.env.data.qpos.copy()
        save_v = self.env.data.qvel.copy()
        try:
            for region, orient, xyz, quat, seed in specs:
                self.env.data.qpos[:7] = seed
                self.env.data.qvel[:] = 0.0
                mujoco.mj_forward(self.env.model, self.env.data)
                self.env.ik.update_configuration(self.env.data.qpos)
                self.env.ik.set_target_position(xyz, quat)
                # Tight threshold first; the IK can land within ~5 mm of
                # the target but miss the 5 mm threshold by a hair.  If
                # the result is geometrically close to the target, keep
                # it instead of falling back to the seed (which puts the
                # EE at the SEED's FK pose, not the handoff target —
                # observed bug: stack-front fallback put EE at y=0.554
                # instead of y=0.13).
                ok = self.env.ik.converge_ik(0.005)
                self.env.data.qpos[:7] = self.env.ik.configuration.q[:7]
                mujoco.mj_forward(self.env.model, self.env.data)
                ee_actual = self.env.data.site_xpos[self.ee_site_id]
                ee_err = float(np.linalg.norm(ee_actual - xyz))
                if ok or ee_err < 0.02:
                    self._handoff_qs[(region, orient)] = \
                        self.env.ik.configuration.q[:7].copy()
                else:
                    self._log(f"WARN: handoff ({region}, {orient}) IK "
                                  f"did not converge (err={ee_err*1000:.1f} "
                                  "mm) — using seed as fallback")
                    self._handoff_qs[(region, orient)] = seed.copy()
        finally:
            self.env.data.qpos[:] = save_q
            self.env.data.qvel[:] = save_v
            mujoco.mj_forward(self.env.model, self.env.data)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[multilevel_blocks executor] {msg}")

    def _err(self, msg: str) -> None:
        """Always-printed error message (failure reason).  Use this when
        an action returns False so the user knows which step gave up."""
        print(f"[multilevel_blocks executor]  ERROR: {msg}")

    def _cell_pose(self, cell_id: str) -> np.ndarray:
        return np.asarray(self.workspace.pose_for(self._parse_cell(cell_id)))

    def _parse_cell(self, cell_id: str) -> Cell:
        """Parse ``cell_id`` case-insensitively against the workspace's
        actual region names.  Fast Downward lowercases identifiers in its
        plan output so e.g. ``stack_l0__0_0`` must round-trip to
        ``stack_L0__0_0``."""
        cell = Cell.parse(cell_id)
        if cell.region in self.workspace.regions:
            return cell
        wanted = cell.region.lower()
        for region_name in self.workspace.regions:
            if region_name.lower() == wanted:
                return Cell(region_name, cell.ix, cell.iy)
        raise KeyError(f"unknown region {cell.region!r}")

    def _set_home(self, cell: Cell) -> None:
        """Reset arm to the canonical seed for the cell's workspace."""
        q = _home_for_cell(cell)
        self.env.data.qpos[:7] = q
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def _try_plan_to_pose(self, target: np.ndarray, quats: Sequence[np.ndarray],
                            n_substeps: int = 12) -> Optional[Tuple[List, np.ndarray]]:
        """Try each ``quat`` in order; return the first ``(path, quat)`` that
        plans successfully.  ``plan_joint_lerp`` first, then ``plan_to_pose``."""
        for q in quats:
            path = self.lik.plan_joint_lerp(target, q, n_substeps=n_substeps)
            if path is not None:
                return path, q
        for q in quats:
            path = self.lik.plan_to_pose(target, q, n_substeps=n_substeps,
                                            slerp_orientation=False)
            if path is not None:
                return path, q
        return None

    def _ee_pos(self) -> np.ndarray:
        return self.env.data.site_xpos[self.ee_site_id].copy()

    def _block_pos(self, block_name: str) -> np.ndarray:
        pos, _ = self.env.get_object_pose(block_name)
        return np.asarray(pos).copy()

    def _log_phase(self, phase: str, intended_xyz: np.ndarray,
                       block_name: Optional[str] = None) -> None:
        """Print intent-vs-actual EE pose at a phase boundary, optionally
        with the held block's world position too.  Verbose-mode only."""
        if not self.verbose:
            return
        ee = self._ee_pos()
        err = np.linalg.norm(ee - intended_xyz) * 1000
        msg = (f"[{phase}] EE intended={intended_xyz} actual={ee} "
                f"err={err:.1f}mm")
        if block_name is not None:
            try:
                blk = self._block_pos(block_name)
                msg += f"  block@{blk}"
            except Exception:
                pass
        print(msg)

    def _execute(self, path: List[np.ndarray],
                    step_size: float = _STEP_APPROACH,
                    precision: bool = False) -> None:
        """Run a path through the controller.

        Args:
            path: list of qpos waypoints.
            step_size: joint-space step size for the controller.
            precision: when True, set ``_advance_delta_override`` to
                ``_ADV_DELTA_PRECISION`` so the controller actually
                reaches each waypoint before advancing.  Restored after
                the path completes.
        """
        if precision:
            self.env.controller._advance_delta_override = _ADV_DELTA_PRECISION
        try:
            self.env.execute_path(path, self.planner, step_size=step_size)
            self.env.wait_idle(settle_steps=_SETTLE_STEPS)
        finally:
            if precision:
                self.env.controller._advance_delta_override = _ADV_DELTA_TRAVEL

    def _wait_gripper_closed(self, max_steps: int = 1000) -> None:
        """Two-phase wait: fingers start moving, then stop (contact or limit).

        Copy of ``PickPlaceExecutor._wait_gripper_closed`` — guarantees the
        gripper has fully closed before we move on.
        """
        started = False
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            vel = abs(self.env.data.qvel[7])
            if not started:
                if vel > _GRIPPER_START_VEL:
                    started = True
            else:
                if vel < _GRIPPER_STOP_VEL:
                    return

    def _wait_gripper_open(self, steps: int = 300) -> None:
        """Run physics for a fixed number of steps after opening the
        gripper, to let fingers fully retract."""
        for _ in range(steps):
            self.env.controller.step()
            self.env.step()

    def _preclose_for_descent(self, max_steps: int = 400) -> None:
        """Narrow gripper to PREGRASP gap before descending into a grasp.

        Without this, the descent-IK collision check sees the fingers at
        their fully-open spread, which can interpenetrate a neighbour
        block at dense parts-table spacing.  See ``_PREGRASP_CTRL`` for
        the geometry.
        """
        self.env.data.ctrl[7] = _PREGRASP_CTRL
        started = False
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            vel = abs(self.env.data.qvel[7])
            if not started:
                if vel > _GRIPPER_START_VEL:
                    started = True
            else:
                if vel < _GRIPPER_STOP_VEL:
                    return

    def _close_attach(self, block_name: str) -> None:
        """Close gripper deliberately + wait for full closure + attach.

        Also captures the world-frame (block - EE) offset at the moment
        of attach so subsequent put actions can reproduce it.
        """
        self.env.controller.close_gripper()
        self._wait_gripper_closed()
        self.env.attach_object_to_ee(block_name)
        self._held_block = block_name
        self._held_offset = (self._block_pos(block_name)
                                - self._ee_pos())

    def _detach_open(self) -> None:
        """Detach attachment, open gripper deliberately, wait for full
        retraction.  Clears the held-block offset stash."""
        if getattr(self.env, "_attached", None) is not None:
            self.env.detach_object()
        self.env.controller.open_gripper()
        self._wait_gripper_open()
        self._held_block = None
        self._held_offset = np.zeros(3)

    def _refresh_held_offset(self) -> None:
        """Re-measure the world-frame (block - EE) offset.  Call after any
        in-hand rotation so the recorded offset reflects the post-rotation
        pose."""
        if self._held_block is None:
            return
        self._held_offset = (self._block_pos(self._held_block)
                                - self._ee_pos())

    def _handoff_for(self, region: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(handoff_xyz, ik_seed_q)`` for the workspace ``region``
        (``"parts"`` or ``"stack_L<k>"``).  XYZ is derived from the current
        ``cfg.{parts,stack}_table_pos`` so this stays correct after a
        runtime table move + ``_precompute_handoffs`` refresh."""
        if region == "parts":
            return (self._compute_handoff_pos("parts", "top_down"),
                       _HOME_PARTS_HANDOFF)
        return (self._compute_handoff_pos("stack", "top_down"),
                   _HOME_STACK_HANDOFF)

    def _orient_for_quat(self, quat: np.ndarray) -> str:
        """Classify ``quat`` as 'top_down' or 'front' for handoff lookup."""
        # Top-down quats have gripper-z pointing -world-z, which means
        # quat's rotated [0,0,1] has world-z component near -1.
        # Front quats have gripper-z in the world XY plane.
        w, x, y, z = quat
        # local-z rotated to world:
        wz_z = 1 - 2 * (x * x + y * y)
        return "top_down" if wz_z < -0.5 else "front"

    def _to_neutral_home(self) -> bool:
        """Lerp from current q to the neutral HOME pose.

        Called at the end of every pick and the start of every put so
        cross-workspace transitions go via a centred, safe config instead
        of trying to swing the base joint by π in one lerp.  Returns
        True if the lerp segment is collision-free.
        """
        start_q = self.env.data.qpos[:7].copy()
        goal_q = _HOME_NEUTRAL_Q
        n = 20
        if self._segment_collision_free(start_q, goal_q, n):
            path = self._build_lerp_path(start_q, goal_q, n)
            self._execute(path, step_size=_STEP_APPROACH)
            return True
        # Fallback: lerp via Cartesian-substep IK from current EE pose
        # to HOME's EE pose.
        save = self.env.data.qpos.copy()
        try:
            self.env.data.qpos[:7] = goal_q
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            ee_target = self.env.data.site_xpos[self.ee_site_id].copy()
            # Use current orientation as a soft target — we just want
            # the EE in roughly the right xyz, orientation is best-effort.
            mat = self.env.data.site_xmat[self.ee_site_id].copy()
        finally:
            self.env.data.qpos[:] = save
            mujoco.mj_forward(self.env.model, self.env.data)
        # Try plan_to_pose with various neutral-friendly quats.
        for q in (_QUAT_TOP_DOWN_Y, _QUAT_TOP_DOWN_X):
            res = self._try_plan_to_pose(ee_target, [q], n_substeps=24)
            if res is not None:
                self._execute(res[0], step_size=_STEP_APPROACH)
                return True
        self._err("to_neutral_home: all fallbacks failed")
        return False

    def _to_handoff(self, region: str, gripper_quat: np.ndarray) -> bool:
        """Move to ``(region, orientation)`` hand-off via its cached q-config.

        Uses ``plan_joint_lerp`` from the current q to the pre-computed
        hand-off q.  Sub-segments collision-check; if the lerp passes,
        the path is executed.  Falls back to ``plan_to_pose`` if the lerp
        fails (e.g. across a base-joint swing of π).
        """
        orient = self._orient_for_quat(gripper_quat)
        key = (region, orient)
        if key not in self._handoff_qs:
            self._err(f"to_handoff: no cached q for {key}")
            return False
        goal_q = self._handoff_qs[key]
        start_q = self.env.data.qpos[:7].copy()

        # Try a direct joint-space lerp first.
        n_substeps = 20
        if self._segment_collision_free(start_q, goal_q, n_substeps):
            path = self._build_lerp_path(start_q, goal_q, n_substeps)
            self._execute(path, step_size=_STEP_APPROACH)
            return True

        # Fallback: Cartesian path with slerp_orientation=True.
        target_xyz, _ = self._handoff_for(region)
        self.env.ik.update_configuration(self.env.data.qpos)
        res = self._try_plan_to_pose(target_xyz, [gripper_quat],
                                          n_substeps=24)
        if res is not None:
            self._execute(res[0], step_size=_STEP_APPROACH)
            return True
        self._err(f"to_handoff({region}, {orient}): all fallbacks failed")
        return False

    def _segment_collision_free(self, q_a: np.ndarray, q_b: np.ndarray,
                                    n: int) -> bool:
        # E2: is_path_collision_free does 1 mj_forward per step + 1
        # restore (vs 3 mj_forward per step for is_collision_free).  ~2.4x
        # fewer mj_forward calls for the same coverage; extra check at
        # alpha=0 (q_a) is free.  Used by _to_neutral_home and
        # _to_handoff segment-feasibility checks.
        return self.env.is_path_collision_free(q_a, q_b, steps=n)

    def _build_lerp_path(self, q_a: np.ndarray, q_b: np.ndarray,
                            n: int) -> List[np.ndarray]:
        path = [q_a.copy()]
        for k in range(1, n + 1):
            alpha = k / n
            path.append((1 - alpha) * q_a + alpha * q_b)
        return path

    def _seed_arm_for_cell(self, cell_id: str, family: str,
                                 quat: np.ndarray,
                                 fallback_quats: Optional[
                                     Sequence[np.ndarray]] = None,
                                 ) -> bool:
        """Seed env.data.qpos[:7] with a cached IK config for the
        (cell, family) target, then mj_forward + update mink.

        Lookup order: first try the exact (cell, family, quat) triple;
        on miss, try any quat in ``fallback_quats`` (if given) in order;
        on full miss, return False (caller proceeds with current arm
        config).

        The fallback exists because the precompute filters out IK
        targets that didn't converge — so SOME (cell, quat) pairs miss
        but OTHER quats at the same cell hit.  Any cached arm config
        near the cell helps mink converge subsequent IK probes far
        more than the handoff seed does.
        """
        if self._ik_seed_lut is None:
            return False
        arm_q = self._ik_seed_lut.lookup(cell_id, family, quat)
        if arm_q is None and fallback_quats is not None:
            for q in fallback_quats:
                arm_q = self._ik_seed_lut.lookup(cell_id, family, q)
                if arm_q is not None:
                    break
        if arm_q is None:
            return False
        self.env.data.qpos[:7] = arm_q
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.ik.update_configuration(self.env.data.qpos)
        return True

    def _filter_quats_by_anchor_ik(self,
                                          anchor_pos: np.ndarray,
                                          grasp_quats: List[np.ndarray],
                                          ) -> List[np.ndarray]:
        """Cheap IK pre-filter at the action's grasp / place anchor.

        For each candidate quat, try ``converge_ik`` at ``anchor_pos``
        with that orientation.  Returns the subset that IK-converges.
        Conservative: never rejects a quat that could work — IK
        convergence is a necessary condition for the full chain.

        Mirrors the pre-filter already in ``pick_upright`` (line ~931).
        max_iters is temporarily capped to 100 (vs the default 1000) so
        unreachable quats fail fast (~10 ms each instead of ~100 ms).
        Reachable quats converge in 10-30 iterations regardless of cap,
        so the cap only bounds the INFEAS worst case.
        """
        if not grasp_quats:
            return []
        # Snapshot env state — converge_ik mutates mink's configuration
        # and the mocap target; restore afterwards.
        save_q = self.env.data.qpos.copy()
        save_v = self.env.data.qvel.copy()
        # Cap iters so unreachable probes fail fast.  Reachable ones
        # converge well below 100 iters anyway.
        ik = self.env.ik
        saved_max_iters = ik.max_iters
        ik.max_iters = 100
        try:
            ik.update_configuration(self.env.data.qpos)
            viable: List[np.ndarray] = []
            for q in grasp_quats:
                ik.set_target_position(anchor_pos, q)
                if ik.converge_ik(0.005):
                    viable.append(q)
                # Re-seed mink from the saved arm pose for the next quat
                # so each candidate is judged independently.
                ik.update_configuration(save_q)
            return viable
        finally:
            ik.max_iters = saved_max_iters
            self.env.data.qpos[:] = save_q
            self.env.data.qvel[:] = save_v
            mujoco.mj_forward(self.env.model, self.env.data)

    def _seeded_lerp_check(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        n_substeps: int,
        max_iters: int,
        seed_arm_q: Optional[np.ndarray] = None,
        lut_cell_id: Optional[str] = None,
        lut_family: Optional[str] = None,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """LUT-aware joint-lerp probe used by pick_upright phases.

        Mirrors :meth:`tampanda.planners.linear_ik.LinearIKPlanner.plan_joint_lerp`
        but pre-seeds mink with either an explicit ``seed_arm_q`` or the LUT
        entry for ``(lut_cell_id, lut_family, target_quat)``, then runs
        ``converge_ik`` with a tight ``max_iters`` cap.  Builds the joint-
        lerp path from the CURRENT arm config (not the seed) so executing
        the returned path stays consistent with the simulator state.

        LUT policy: when the LUT is loaded and a ``lut_cell_id`` is given,
        a missing entry is treated as infeasible (returns None).  The
        precompute had a 500-iter budget from HOME; if it didn't converge
        there, runtime cold-IK won't converge either.

        Returns ``(path, goal_q)`` on success, ``None`` on any failure
        (IK non-convergence, goal collision, lerp collision).
        """
        # Choose the seed.
        seed = seed_arm_q
        if seed is None and lut_cell_id is not None:
            if self._ik_seed_lut is None:
                seed = None
            elif not self._ik_seed_lut.has(lut_cell_id, lut_family,
                                                 target_quat):
                return None
            else:
                seed = self._ik_seed_lut.lookup(
                    lut_cell_id, lut_family, target_quat,
                )

        qpos_save = self.env.data.qpos.copy()
        qvel_save = self.env.data.qvel.copy()
        saved_iters = self.env.ik.max_iters
        try:
            if seed is not None:
                self.env.data.qpos[:7] = seed
                self.env.data.qvel[:] = 0.0
                mujoco.mj_forward(self.env.model, self.env.data)
            self.env.ik.update_configuration(self.env.data.qpos)
            self.env.ik.set_target_position(target_pos, target_quat)
            self.env.ik.max_iters = max_iters
            if not self.env.ik.converge_ik(0.005):
                return None
            goal_q = self.env.ik.configuration.q[:7].copy()
        finally:
            self.env.ik.max_iters = saved_iters
            self.env.data.qpos[:] = qpos_save
            self.env.data.qvel[:] = qvel_save
            mujoco.mj_forward(self.env.model, self.env.data)

        # Path starts from the REAL current arm config; goal_q is the
        # seeded-IK solution.
        start_q = self.env.data.qpos[:7].copy()
        if not self.env.is_collision_free(goal_q):
            return None
        for j in range(1, n_substeps + 1):
            alpha = j / n_substeps
            q_mid = (1 - alpha) * start_q + alpha * goal_q
            if not self.env.is_collision_free(q_mid):
                return None

        path: List[np.ndarray] = [start_q.copy()]
        for k in range(1, n_substeps + 1):
            alpha = k / n_substeps
            path.append((1 - alpha) * start_q + alpha * goal_q)
        return path, goal_q

    def _return_after_pickput(self, region: str,
                                  used_quat: np.ndarray) -> None:
        """Post-action return: handoff in the given region's orientation,
        then NEUTRAL_HOME so the next action starts cleanly.

        Overridable: ``FastFeasibilityExecutor`` no-ops this because for
        feasibility checks we only need to know if the pick / put itself
        succeeded — the return path doesn't change the verdict and costs
        ~2 plan_joint_lerps (~25-30% of per-check wall-clock).
        """
        self._to_handoff(region, used_quat)
        self._to_neutral_home()

    def _fast_column_align_substeps_halved(self) -> bool:
        """In fast mode we halve column-align Cartesian substeps from
        20 -> 10 (no execution arc, just IK + collision coverage).
        Overridden by FastFeasibilityExecutor to return True."""
        return False

    def _validate_put_upright_return(self) -> bool:
        """Whether to validate post-detach return phases (lift-back,
        retract, handoff) in :meth:`put_upright`.

        Once ``_detach_open`` succeeds the block placement is locked in
        — for feasibility the only remaining question is "can the arm
        get out of the way?", which is bounded by phases the executor
        will redo at the start of the next action anyway.  Real
        execution must validate it; fast feasibility can skip the ~5-10
        IK calls (often the slowest in the chain — they probe Cartesian
        substep IK at multiple z heights in the tight upright column).
        """
        return True

    # -----------------------------------------------------------------
    # Pick actions
    # -----------------------------------------------------------------

    def _pick_top_down(self, block_name: str, anchor_pos: np.ndarray,
                          grasp_quats: List[np.ndarray],
                          home_seed: np.ndarray,
                          source_region: str = "stack") -> bool:
        """Common pick chain for top-down grasps.

        anchor_pos: world (x,y,z) of the EE *attachment site* at grasp time.
        grasp_quats: ordered list of grasp orientations to try (4 for cube/
            upright, 2 for flat).
        home_seed: q-seed for IK (corresponds to parts vs stack workspace).
        source_region: ``"parts"`` or ``"stack"`` — used to pick the
            workspace hand-off to return to after the pick.
        """
        self.env.add_collision_exception(block_name)
        try:
            # PRE-FILTER (B1): cheap IK reachability at the grasp anchor
            # for each candidate quat.  If no quat IK-converges the cell
            # is unreachable in any orientation — abort BEFORE the
            # expensive transit chain.  Conservative: never rejects a
            # quat that could pass the full chain.
            filtered_quats = self._filter_quats_by_anchor_ik(
                anchor_pos, grasp_quats,
            )
            if not filtered_quats:
                self._err(
                    f"pick {block_name}: no quat IK-converges at grasp anchor "
                    f"{anchor_pos.round(3).tolist()} "
                    f"(probed {len(grasp_quats)} candidates)"
                )
                return False
            grasp_quats = filtered_quats

            # Start with neutral HOME so this pick begins from a clean
            # config regardless of where the arm was after the prior
            # action (mirrors what put does at its start).
            self._to_neutral_home()
            # Then lerp to the source workspace's hand-off.
            self._to_handoff(source_region, grasp_quats[0])

            self.env.ik.update_configuration(self.env.data.qpos)

            # Phase 1: approach (lift above anchor by _APPROACH_HEIGHT).
            # Fast travel step size — no precision needed at this height.
            approach = anchor_pos + np.array([0.0, 0.0, _APPROACH_HEIGHT])
            res = self._try_plan_to_pose(approach, grasp_quats, n_substeps=20)
            if res is None:
                self._err(f"pick {block_name}: approach IK failed")
                return False
            path, used_quat = res
            self._execute(path, step_size=_STEP_APPROACH)

            # Pre-close gripper to PREGRASP width so the descent-IK
            # collision check sees a narrow gripper profile (≥17 mm
            # clearance from neighbours at 1-cell-gap parts spacing).
            self._preclose_for_descent()
            self.env.ik.update_configuration(self.env.data.qpos)

            # Phase 2: descend to grasp anchor — PRECISION phase.  Tight
            # advance delta + slow step size so the controller tracks each
            # waypoint instead of blasting past.
            res = self._try_plan_to_pose(anchor_pos, [used_quat], n_substeps=10)
            if res is None:
                self._err(f"pick {block_name}: descend IK failed")
                return False
            path, _ = res
            self._execute(path, step_size=_STEP_GRASP, precision=True)

            # Phase 3: grasp + attach — deliberate close-and-wait.
            self._close_attach(block_name)
            self._trace_phase("pick:post-attach")

            # Phase 4: lift — slow lift step so the block doesn't slip.
            lift = anchor_pos + np.array([0.0, 0.0, _LIFT_HEIGHT])
            res = self._try_plan_to_pose(lift, [used_quat], n_substeps=10)
            if res is not None:
                self._execute(res[0], step_size=_STEP_LIFT, precision=True)
            self._trace_phase("pick:post-lift")
            # Phase 5: return to source workspace's hand-off, then to the
            # neutral HOME pose.  The subsequent put starts from HOME, not
            # from source-handoff, so cross-workspace transit happens in
            # TWO short segments (handoff→HOME, then HOME→target-handoff
            # inside put).
            self._return_after_pickput(source_region, used_quat)
            self._trace_phase("pick:at-home")
            return True
        finally:
            self.env.clear_collision_exceptions()

    def pick_cube(self, block_name: str, cell_id: str) -> bool:
        """Top-down pick of a 1×1 cube at ``cell_id``.

        Tries 4 grasp yaws (90 deg increments)."""
        cell = self._parse_cell(cell_id)
        cube_centre = self._cell_pose(cell_id)
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        region = "parts" if cell.region == "parts" else "stack"
        return self._pick_top_down(block_name, anchor,
                                       _cube_grasp_quats(),
                                       _home_for_cell(cell),
                                       source_region=region)

    def pick_flat_x(self, block_name: str, c1_id: str, c2_id: str) -> bool:
        """Top-down pick of a flat-x oblong spanning c1 (west) and c2 (east).

        Block centroid is midway between the two cells.  Tries 2 grasp yaws."""
        p1 = self._cell_pose(c1_id)
        p2 = self._cell_pose(c2_id)
        cube_centre = (p1 + p2) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._pick_top_down(block_name, anchor,
                                       _flat_x_grasp_quats(),
                                       _home_for_cell(cell),
                                       source_region=region)

    def pick_flat_y(self, block_name: str, c1_id: str, c2_id: str) -> bool:
        """Top-down pick of a flat-y oblong spanning c1 (south) and c2 (north)."""
        p1 = self._cell_pose(c1_id)
        p2 = self._cell_pose(c2_id)
        cube_centre = (p1 + p2) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._pick_top_down(block_name, anchor,
                                       _flat_y_grasp_quats(),
                                       _home_for_cell(cell),
                                       source_region=region)

    def pick_long_x(self, block_name: str,
                       c1_id: str, c2_id: str, c3_id: str) -> bool:
        """Top-down pick of a 3×1 long block spanning c1..c3 along world-x.

        Centroid is the midpoint of c1 (west) and c3 (east).  Same grasp
        family as flat-x oblong (fingers close along y).  c2 is the
        middle cell and is not used for the anchor computation.
        """
        p1 = self._cell_pose(c1_id)
        p3 = self._cell_pose(c3_id)
        cube_centre = (p1 + p3) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._pick_top_down(block_name, anchor,
                                       _flat_x_grasp_quats(),
                                       _home_for_cell(cell),
                                       source_region=region)

    def pick_long_y(self, block_name: str,
                       c1_id: str, c2_id: str, c3_id: str) -> bool:
        """Top-down pick of a 3×1 long block spanning c1..c3 along world-y."""
        p1 = self._cell_pose(c1_id)
        p3 = self._cell_pose(c3_id)
        cube_centre = (p1 + p3) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._pick_top_down(block_name, anchor,
                                       _flat_y_grasp_quats(),
                                       _home_for_cell(cell),
                                       source_region=region)

    def pick_upright(self, block_name: str, c_low_id: str,
                       c_high_id: str) -> bool:
        """Front-facing pick of an upright oblong spanning c_low and c_high.

        Grasps the upper half of the block; EE attachment site at the
        midpoint of c_high.  Tries 4 grasp yaws (the upright block's
        4-fold lateral symmetry).

        Phase 3.7: each phase consults the IK seed LUT keyed by
        (c_high_id, "upright", q) — the anchor target IS the cell pose
        of c_high, so LUT-seeded IK converges in 0-3 iters.  Same
        skip-uncached / tier-skip / cached-goal_q pattern as put_upright.
        """
        p_low = self._cell_pose(c_low_id)
        p_high = self._cell_pose(c_high_id)
        cube_centre_high = p_high
        # Front-facing: EE approaches horizontally; EE position is to the
        # SIDE of the block.  For simplicity, anchor the EE at the block's
        # upper-half centroid (the block sits inside the parallel-jaw gap
        # when the gripper is centred there).
        anchor = cube_centre_high
        cell = self._parse_cell(c_low_id)
        self.env.add_collision_exception(block_name)
        try:
            self.env.data.qpos[:7] = _home_for_cell(cell)
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.ik.update_configuration(self.env.data.qpos)

            candidate_quats = _upright_grasp_quats_sorted(anchor)

            # ---- Phase A: anchor filter ----
            # The anchor IS ws.pose_for(c_high) — an exact LUT target.
            # LUT path: cached probes converge in 0-3 iters; uncached
            # are rejected (precompute had 500-iter HOME budget — runtime
            # can't do better with a cold seed).  No-LUT path: full
            # cold-IK plan_joint_lerp (original behaviour).
            lut_loaded = self._ik_seed_lut is not None
            filtered_quats: List[np.ndarray] = []
            anchor_goal_q: Dict[Tuple, np.ndarray] = {}
            for q in candidate_quats:
                if lut_loaded:
                    res = self._seeded_lerp_check(
                        anchor, q, n_substeps=10, max_iters=15,
                        lut_cell_id=c_high_id, lut_family="upright",
                    )
                    goal_q = res[1] if res is not None else None
                else:
                    path0 = self.lik.plan_joint_lerp(
                        anchor, q, n_substeps=10,
                    )
                    goal_q = path0[-1][:7].copy() if path0 is not None else None
                if goal_q is not None:
                    filtered_quats.append(q)
                    anchor_goal_q[tuple(np.round(q, 6).tolist())] = goal_q
            if not filtered_quats:
                self._err(
                    f"pick_upright {block_name}: no quat reaches grasp anchor  "
                    f"target=({anchor[0]:.3f}, {anchor[1]:.3f}, "
                    f"{anchor[2]:.3f}) (probed {len(candidate_quats)} yaws)"
                )
                return False

            # ---- Phase B: approach per yaw (seeded from anchor_goal_q) ----
            # Approach is ~10 cm offset along the gripper z-axis at the
            # same yaw — anchor_goal_q is an excellent seed.  Cap at 50
            # iters; cold-IK from HOME takes 200+ here, mink with the
            # cell-local seed converges in 10-30.
            used_quat = None
            approach = None
            path = None
            for q in filtered_quats:
                gz_world = _gripper_z_world(q)
                cand_approach = anchor - gz_world * _APPROACH_HEIGHT
                qkey = tuple(np.round(q, 6).tolist())
                p_res = self._seeded_lerp_check(
                    cand_approach, q, n_substeps=20, max_iters=50,
                    seed_arm_q=anchor_goal_q[qkey],
                )
                if p_res is not None:
                    path, used_quat, approach = p_res[0], q, cand_approach
                    break
            if path is None:
                self._err(
                    f"pick_upright {block_name}: approach IK failed for all "
                    f"{len(filtered_quats)} filtered yaws"
                )
                return False
            self._execute(path, step_size=_STEP_APPROACH)

            # Pre-close gripper before descending to the grasp anchor;
            # narrows the finger footprint for the dense-neighbour case.
            self._preclose_for_descent()
            self.env.ik.update_configuration(self.env.data.qpos)

            # ---- Phase C: descent ----
            # We already have anchor_goal_q from Phase A.  In fast mode,
            # build the joint-lerp directly from the post-approach config
            # to the cached goal_q (skips plan_to_pose's Cartesian
            # substep IK chain).  Real mode keeps the full path planner
            # for execution fidelity.
            used_qkey = tuple(np.round(used_quat, 6).tolist())
            cached_goal = anchor_goal_q[used_qkey]
            if self._fast_column_align_substeps_halved():
                res = self._seeded_lerp_check(
                    anchor, used_quat, n_substeps=10, max_iters=10,
                    seed_arm_q=cached_goal,
                )
            else:
                planned = self._try_plan_to_pose(anchor, [used_quat],
                                                       n_substeps=10)
                res = planned if planned is not None else None
            if res is None:
                self._err(f"pick_upright {block_name}: descend IK failed")
                return False
            self._execute(res[0], step_size=_STEP_GRASP, precision=True)

            self._close_attach(block_name)

            # ---- Phase D: lift (seeded from descent goal_q) ----
            lift = anchor + np.array([0.0, 0.0, _LIFT_HEIGHT])
            if self._fast_column_align_substeps_halved():
                res = self._seeded_lerp_check(
                    lift, used_quat, n_substeps=10, max_iters=50,
                    seed_arm_q=cached_goal,
                )
            else:
                res = self._try_plan_to_pose(lift, [used_quat], n_substeps=10)
            if res is not None:
                self._execute(res[0], step_size=_STEP_LIFT, precision=True)
            return True
        finally:
            self.env.clear_collision_exceptions()

    # -----------------------------------------------------------------
    # Put actions
    # -----------------------------------------------------------------

    def _put_top_down(self, block_name: str, anchor_pos: np.ndarray,
                        grasp_quats: List[np.ndarray],
                        home_seed: np.ndarray,
                        target_region: str = "stack") -> bool:
        """Common put chain for top-down placement.

        Transitions to the target workspace's hand-off first (ensures the
        IK basin is right for the upcoming approach), then plans the local
        approach → descend → release → lift → handoff sequence.
        """
        # Add the held-block to the collision exception set so brief
        # finger-block contacts during transit (the parallel-jaw
        # gripper pads sit right against the block at TOP_DOWN grasp)
        # don't fail the collision check.  Pick adds the same
        # exception; put didn't before this change.
        self.env.add_collision_exception(block_name)

        # PRE-FILTER (B1): cheap IK reachability at the place anchor.
        # If no quat IK-converges the place column is unreachable —
        # abort BEFORE the expensive transit chain.
        filtered_quats = self._filter_quats_by_anchor_ik(
            anchor_pos, grasp_quats,
        )
        if not filtered_quats:
            self._err(
                f"put {block_name}: no quat IK-converges at place anchor "
                f"{anchor_pos.round(3).tolist()} "
                f"(probed {len(grasp_quats)} candidates)"
            )
            self.env.clear_collision_exceptions()
            return False
        grasp_quats = filtered_quats

        # Step 0a: go to neutral HOME first.  This breaks any prior
        # cross-workspace transit into two short joint-space lerps.
        self._to_neutral_home()
        # Step 0b: from HOME, lerp to the target workspace's hand-off.
        if not self._to_handoff(target_region, grasp_quats[0]):
            self._err(f"put {block_name}: handoff transition failed")
            self.env.clear_collision_exceptions()
            return False
        self._trace_phase("put:at-target-handoff")
        self.env.ik.update_configuration(self.env.data.qpos)

        # Step 1: high pose directly above target xy.  z is anchor +
        # _HIGH_ABOVE_HEIGHT (lower than the handoff z) so corner/edge
        # cells stay within Franka's reach with TOP_DOWN orientation.
        high_above = anchor_pos + np.array([0.0, 0.0, _HIGH_ABOVE_HEIGHT])
        res = self._try_plan_to_pose(high_above, grasp_quats, n_substeps=24)
        if res is None:
            self._err(
                f"put {block_name}: high-above IK failed  "
                f"target=({high_above[0]:.3f}, {high_above[1]:.3f}, "
                f"{high_above[2]:.3f}) quats={[q.tolist() for q in grasp_quats]}"
            )
            return False
        path, used_quat = res
        self._execute(path, step_size=_STEP_APPROACH)

        approach = anchor_pos + np.array([0.0, 0.0, _APPROACH_HEIGHT])
        res = self._try_plan_to_pose(approach, [used_quat], n_substeps=14)
        if res is None:
            self._err(
                f"put {block_name}: approach IK failed  "
                f"target=({approach[0]:.3f}, {approach[1]:.3f}, "
                f"{approach[2]:.3f}) quat={used_quat.tolist()}"
            )
            return False
        path, _ = res
        # Slow + precision from here on so the block doesn't shift on
        # the way down.
        self._execute(path, step_size=_STEP_PLACE, precision=True)

        descend = anchor_pos + np.array([0.0, 0.0, _PLACE_CLEARANCE])
        res = self._try_plan_to_pose(descend, [used_quat], n_substeps=10)
        if res is None:
            self._err(
                f"put {block_name}: descend IK failed  "
                f"target=({descend[0]:.3f}, {descend[1]:.3f}, "
                f"{descend[2]:.3f}) quat={used_quat.tolist()}"
            )
            return False
        self._execute(res[0], step_size=_STEP_PLACE, precision=True)
        self._log_phase(f"put-top-down/{block_name}/pre-detach", descend,
                         block_name=block_name)
        self._trace_phase("put:pre-detach")

        self._detach_open()
        self._log_phase(f"put-top-down/{block_name}/post-detach", descend,
                         block_name=block_name)

        lift = anchor_pos + np.array([0.0, 0.0, _LIFT_HEIGHT])
        res = self._try_plan_to_pose(lift, [used_quat], n_substeps=10)
        if res is not None:
            self._execute(res[0], step_size=_STEP_LIFT, precision=True)
        self._log_phase(f"put-top-down/{block_name}/post-lift", lift,
                         block_name=block_name)
        # Return to the target workspace's hand-off, then to neutral HOME
        # so the next action starts from a known clean state.  Skipped
        # in the fast executor.
        self._return_after_pickput(target_region, used_quat)
        self.env.clear_collision_exceptions()
        return True

    def put_cube(self, block_name: str, cell_id: str) -> bool:
        cell = self._parse_cell(cell_id)
        cube_centre = self._cell_pose(cell_id)
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        region = "parts" if cell.region == "parts" else "stack"
        return self._put_top_down(block_name, anchor,
                                       _cube_grasp_quats(),
                                       _home_for_cell(cell),
                                       target_region=region)

    def put_flat_x(self, block_name: str, c1_id: str, c2_id: str) -> bool:
        p1 = self._cell_pose(c1_id)
        p2 = self._cell_pose(c2_id)
        cube_centre = (p1 + p2) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._put_top_down(block_name, anchor,
                                       _flat_x_grasp_quats(),
                                       _home_for_cell(cell),
                                       target_region=region)

    def put_flat_y(self, block_name: str, c1_id: str, c2_id: str) -> bool:
        p1 = self._cell_pose(c1_id)
        p2 = self._cell_pose(c2_id)
        cube_centre = (p1 + p2) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._put_top_down(block_name, anchor,
                                       _flat_y_grasp_quats(),
                                       _home_for_cell(cell),
                                       target_region=region)

    def put_long_x(self, block_name: str,
                      c1_id: str, c2_id: str, c3_id: str) -> bool:
        """Top-down place of a 3×1 long block spanning c1..c3 along x."""
        p1 = self._cell_pose(c1_id)
        p3 = self._cell_pose(c3_id)
        cube_centre = (p1 + p3) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._put_top_down(block_name, anchor,
                                       _flat_x_grasp_quats(),
                                       _home_for_cell(cell),
                                       target_region=region)

    def put_long_y(self, block_name: str,
                      c1_id: str, c2_id: str, c3_id: str) -> bool:
        """Top-down place of a 3×1 long block spanning c1..c3 along y."""
        p1 = self._cell_pose(c1_id)
        p3 = self._cell_pose(c3_id)
        cube_centre = (p1 + p3) / 2
        anchor = cube_centre + np.array([0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z])
        cell = self._parse_cell(c1_id)
        region = "parts" if cell.region == "parts" else "stack"
        return self._put_top_down(block_name, anchor,
                                       _flat_y_grasp_quats(),
                                       _home_for_cell(cell),
                                       target_region=region)

    def put_upright(self, block_name: str, c_low_id: str,
                      c_high_id: str) -> bool:
        """Front-facing place of an upright block at c_low / c_high.

        Traverse-at-altitude chain (access-19 pattern adapted): the gripper
        moves to ABOVE the target column at a safe ``traverse_z`` (above
        the maximum possible stack), then descends vertically at the
        target.  Earlier versions did the long y-translation at
        ``ee_z = anchor.z + 0.04`` (just above the table top), which
        swept the gripper hand body over every cell in between.  The
        traverse-at-altitude pattern confines the table-top-proximity
        motion to the final vertical descent at the target column.

        Chain:
          1. Stack FRONT handoff      (cached handoff_q)
          2. Column-align at altitude (anchor.xy - held_off.xy, traverse_z)
          3. Final descent            (..., ee_z)
          4. Detach
          5. Lift back to traverse_z  (..., traverse_z)
          6. Retract to handoff y     (anchor.x, front_handoff_y, traverse_z)
          7. Return to handoff
        """
        p_low = self._cell_pose(c_low_id)
        p_high = self._cell_pose(c_high_id)
        anchor = (p_low + p_high) / 2     # block centroid at placement

        # Use the runtime-measured block-to-EE offset (captured at the
        # most recent close-attach + refreshed by every in-hand
        # transform).
        held_off = self._held_offset.copy()

        # Step 0a: neutral HOME so the workspace transit is in 2 segments.
        self._to_neutral_home()
        # 1. Stack-FRONT handoff.
        if not self._to_handoff("stack", _QUAT_FRONT_Y):
            self._err(f"put_upright {block_name}: handoff failed")
            return False

        # NOTE (Phase 3.5 step 4): tried seeding the arm with the
        # cached (cell, upright, FRONT_Y) config here.  Made things
        # WORSE — the chain's first IK target after handoff is the
        # settle-to-traverse pose at (handoff_xy, traverse_z), which
        # is far from the cell_centre seed.  Mink then can't converge
        # in 200 iters and the chain bails at settle.  The cached LUT
        # would be useful for column-align / descent phases (after the
        # arm is over the target column), but applying it there is a
        # bigger refactor.  Leaving the LUT loaded for future use.
        self._trace_phase("put-upright:at-stack-handoff")
        self.env.ik.update_configuration(self.env.data.qpos)

        # ee_z: link7_capsule has 5 cm safety radius.  Empirically the
        # smallest feasible EE clearance is 4 cm above anchor.z.  Block
        # drops 4 cm on release — unavoidable for FRONT_Y at table-top.
        ee_z = anchor[2] + 0.04

        # traverse_z: safe altitude above the max possible stack so the
        # gripper hand body clears every cell when translating in y.
        # = stack_table_top + max_stack_height + 8 cm clearance for the
        # gripper body itself.  Mirrors access-19's
        # ``_SAFE_Z_ABOVE_SHELF_TOP``.
        stack_table_top_z = (self.config.stack_table_pos[2]
                                  + _TABLE_TOP_LOCAL_Z)
        max_stack_height = (self.config.stack_grid_cells[2]
                                  * self.config.cube_size)
        traverse_z = stack_table_top_z + max_stack_height + 0.08

        # Front handoff y kept for the retract stage.
        front_handoff_y = self._compute_handoff_pos("stack", "front")[1]

        # 1b. Settle vertically to traverse_z at handoff_xy BEFORE any
        # horizontal motion.  Splits the chain into pure-vertical +
        # pure-horizontal phases so the EE traces a rectangular path
        # (up → over → down) instead of a curved diagonal arc.
        handoff_ee = self.env.data.site_xpos[self.ee_site_id].copy()
        settle_pose = np.array([handoff_ee[0], handoff_ee[1], traverse_z])
        res = self._try_plan_to_pose(settle_pose, [_QUAT_FRONT_Y],
                                              n_substeps=10)
        if res is None:
            self._err(
                f"put_upright {block_name}: settle-to-traverse IK failed  "
                f"target=({settle_pose[0]:.3f}, {settle_pose[1]:.3f}, "
                f"{settle_pose[2]:.3f})"
            )
            return False
        self._execute(res[0], step_size=_STEP_APPROACH)
        self.env.ik.update_configuration(self.env.data.qpos)

        column_align_preview = np.array([anchor[0] - held_off[0],
                                              anchor[1] - held_off[1],
                                              traverse_z])
        final_descent_preview = np.array([anchor[0] - held_off[0],
                                                anchor[1] - held_off[1],
                                                ee_z])

        # Pre-filter quats: keep only those whose path planner can reach
        # BOTH the column-align (at altitude) AND the final descent
        # (closest to the table).  Mirror the execution-time fallback:
        # try plan_joint_lerp first (fast, joint-space), then
        # plan_to_pose (Cartesian-substep) if joint-lerp fails.  Without
        # the Cartesian probe, the filter rejects quats that would
        # actually succeed at execution time — observed in dense scenes
        # like upright_bridges where joint-lerp paths clip already-
        # placed neighbours but Cartesian-substep paths stay clear.
        # _upright_grasp_quats() returns (0, π/2, π, 3π/2) — order is
        # preserved by the list comprehension, so yaw=0 (front approach)
        # is used whenever globally feasible; side/back yaws are only a
        # fallback.
        # Cache goal_q for each (yaw, label) so the column-align
        # execution below can skip its own plan_to_pose IK and just
        # teleport (in fast mode) — the IK was already done by Tier 0.
        probe_goal_q: Dict[Tuple[int, str], np.ndarray] = {}

        def _probe(pos: np.ndarray, q: np.ndarray,
                       label: str, yaw_idx: int) -> bool:
            # Tier 0: cheap IK gate.  If bare IK can't even converge at
            # the target pose, the expensive plan_to_pose Cartesian
            # substeps WILL fail (every substep also runs converge_ik
            # at an intermediate pose, and a wholly-unreachable target
            # tends to fail every substep too).  Skipping the Cartesian
            # tier on IK-failed quats is the biggest wall-clock win on
            # the INFEAS put_upright slow path (~5x per failed quat).
            #
            # Phase 3.6: when the IK seed LUT has a cached arm_q for
            # (c_low_id, "upright", q), seed mink with it and converge
            # with a tight 15-iter cap.  Empirically (see
            # examples/multilevel_blocks_lut_seed_test.py) cached probes
            # converge in 3-5 iters at column_align_preview and 1-2 at
            # final_descent_preview, so 15 is well above the worst case
            # while rejecting genuinely-infeasible (cell, yaw, z) combos
            # at the same rate as the cold-seeded 100-iter probe.
            qpos_save = self.env.data.qpos.copy()
            qvel_save = self.env.data.qvel.copy()
            saved_iters = self.env.ik.max_iters
            tier0_goal_q = None    # captured when Tier 0 succeeds — lets
                                       # Tier 1 below skip its own IK call.
            try:
                lut = self._ik_seed_lut
                if lut is not None:
                    # Pick the LUT family for this phase.  Phase 3.8 added
                    # "upright_descent" (cached at the EE descent z) so the
                    # descent IK has a near-target seed.  For backwards
                    # compatibility with LUTs that pre-date this family,
                    # fall back to "upright" when "upright_descent" is
                    # absent — that matches the pre-3.8 behaviour exactly.
                    if label == "descend" and lut.has(
                            c_low_id, "upright_descent", q):
                        family = "upright_descent"
                    else:
                        family = "upright"
                    if lut.has(c_low_id, family, q):
                        # LUT hit: seed + 15-iter converge.
                        arm_q = lut.lookup(c_low_id, family, q)
                        self.env.data.qpos[:7] = arm_q
                        self.env.data.qvel[:] = 0.0
                        mujoco.mj_forward(self.env.model, self.env.data)
                        self.env.ik.update_configuration(self.env.data.qpos)
                        self.env.ik.set_target_position(pos, q)
                        self.env.ik.max_iters = 15
                        ik_ok = self.env.ik.converge_ik(0.005)
                    else:
                        # LUT exists but this (cell, yaw) isn't cached.
                        # Precompute ran with max_iters=500 from a HOME
                        # seed and rejected this combination — treat it
                        # as infeasible without re-probing.  Saves the
                        # cold-IK cap-hit cost (~90 ms × 2 missing yaws
                        # per put_upright at typical cells).
                        ik_ok = False
                else:
                    # No LUT available: standard Tier 0 with a tight
                    # 30-iter cap.  Median cold-IK converges in 11 iters
                    # so cap=30 covers ~94% of feasible cases.
                    self.env.ik.update_configuration(self.env.data.qpos)
                    self.env.ik.set_target_position(pos, q)
                    self.env.ik.max_iters = 30
                    ik_ok = self.env.ik.converge_ik(0.005)
                if ik_ok:
                    tier0_goal_q = self.env.ik.configuration.q[:7].copy()
                    qkey = tuple(np.round(q, 6).tolist())
                    probe_goal_q[(qkey, label)] = tier0_goal_q
            finally:
                self.env.ik.max_iters = saved_iters
                self.env.data.qpos[:] = qpos_save
                self.env.data.qvel[:] = qvel_save
                mujoco.mj_forward(self.env.model, self.env.data)
            if not ik_ok:
                if self.verbose:
                    print(f"    [filter] yaw{yaw_idx} {label}: "
                              f"IK gate rejected (skipping Cartesian probe)")
                return False
            # Tier 1: joint-lerp with the goal_q we ALREADY have from
            # Tier 0 — skip plan_joint_lerp's internal IK call.  Just
            # collision-check the goal and the joint-space midpoints.
            # Saves ~50 ms per yaw probe across the K-yaw filter.
            start_q = self.env.data.qpos[:7].copy()
            if self.env.is_collision_free(tier0_goal_q):
                jl_clean = True
                for j in range(1, 11):
                    alpha = j / 10.0
                    q_mid = (1 - alpha) * start_q + alpha * tier0_goal_q
                    if not self.env.is_collision_free(q_mid):
                        jl_clean = False
                        break
                if jl_clean:
                    return True
            # Tier 2: Cartesian-substep (also full check).  Only reached
            # if Tier 0 passed (IK converges) and Tier 1 failed (lerp
            # path has a collision).
            cs = self.lik.plan_to_pose(pos, q, slerp_orientation=False,
                                              n_substeps=10)
            if cs is not None:
                return True
            # Tier 3: bare-IK acceptance — the goal pose is reachable
            # (Tier 0 confirmed) but both planned paths hit collisions.
            # Accept and let execution decide; downstream code has
            # always done this (was needed for held-block weld edge
            # cases at densely-packed cells).
            if self.verbose:
                print(f"    [filter] yaw{yaw_idx} {label}: "
                          f"IK-only fallback accepted (lerp+Cartesian failed)")
            return True

        candidate_quats = _upright_grasp_quats_sorted(
            final_descent_preview)[:_PUT_UPRIGHT_QUAT_PROBE_TOPK]
        filtered_quats = []
        for yaw_idx, q in enumerate(candidate_quats):
            ok_col = _probe(column_align_preview, q, "col-align", yaw_idx)
            ok_dsc = _probe(final_descent_preview, q, "descend", yaw_idx)
            if ok_col and ok_dsc:
                filtered_quats.append(q)
            elif self.verbose:
                print(f"    [filter] yaw{yaw_idx} rejected "
                          f"(col_ok={ok_col}, dsc_ok={ok_dsc})")
        if not filtered_quats:
            self._err(
                f"put_upright {block_name}: no quat passes column-align+descend  "
                f"col=({column_align_preview[0]:.3f}, "
                f"{column_align_preview[1]:.3f}, "
                f"{column_align_preview[2]:.3f}) "
                f"descend=({final_descent_preview[0]:.3f}, "
                f"{final_descent_preview[1]:.3f}, "
                f"{final_descent_preview[2]:.3f}) "
                f"(probed {len(candidate_quats)} yaws)"
            )
            return False

        # 2. Column-align at traverse_z — Cartesian-straight from
        # (handoff_xy, traverse_z) to (target_xy, traverse_z).  We prefer
        # plan_to_pose (Cartesian-substep IK) over plan_joint_lerp here
        # because the latter produces a STRAIGHT LINE in joint space,
        # which corresponds to a CURVED ARC in world coordinates — and
        # for a long handoff→back-row move the arc can swing wide over
        # the table.  Cartesian-straight keeps the EE on a world-frame
        # line.  Falls back to joint-space lerp if Cartesian-substep
        # IK can't converge (rare at this altitude + constant
        # orientation).
        # Per-yaw substep count.  Fast mode halves it (10 vs 20) — no
        # execution arc to validate, just IK convergence + per-substep
        # collision.  The column-align move is at traverse_z (above the
        # max stack) so the path passes through open space; 10 substeps
        # is plenty for collision coverage there.
        ca_substeps = 10 if self._fast_column_align_substeps_halved() else 20

        # Phase 4: when a yaw pool is attached (fast mode + rgnet),
        # dispatch the K-yaw column-align in parallel.  Each worker
        # runs plan_to_pose for ONE quat; the first success wins.
        # Wall-clock drops from K × per_probe to ~per_probe.
        # The path comes back from the worker but the MAIN env's arm
        # hasn't moved; downstream self._execute(path, ...) below
        # applies it.
        path = None
        used_quat = None
        if self._yaw_pool is not None and len(filtered_quats) > 1:
            arm_qpos = self.env.data.qpos[:7].copy()
            res = self._yaw_pool.first_success_plan(
                arm_qpos, column_align_preview,
                filtered_quats, n_substeps=ca_substeps,
                slerp_orientation=False,
            )
            if res is not None:
                path, used_quat = res
        if path is None:
            for q in filtered_quats:
                # Phase 3.7: in fast mode, reuse the goal_q we already
                # computed in _probe instead of running plan_to_pose's
                # Cartesian-substep IK (10 calls × ~33 ms = ~330 ms
                # eliminated per column-align).  Tier 1's collision
                # check already validated the joint-lerp to this goal_q,
                # so teleporting straight there is sound.
                qkey = tuple(np.round(q, 6).tolist())
                cached = probe_goal_q.get((qkey, "col-align"))
                if (cached is not None
                        and self._fast_column_align_substeps_halved()):
                    path = [self.env.data.qpos[:7].copy(), cached]
                    used_quat = q
                    break
                path = self.lik.plan_to_pose(column_align_preview, q,
                                                  slerp_orientation=False,
                                                  n_substeps=ca_substeps)
                if path is not None:
                    used_quat = q
                    break
        if path is None:
            res = self._try_plan_to_pose(column_align_preview, filtered_quats,
                                              n_substeps=ca_substeps)
            if res is None:
                self._err(
                    f"put_upright {block_name}: column-align IK failed  "
                    f"target=({column_align_preview[0]:.3f}, "
                    f"{column_align_preview[1]:.3f}, "
                    f"{column_align_preview[2]:.3f}) "
                    f"({len(filtered_quats)}/{len(candidate_quats)} yaws after filter)"
                )
                return False
            path, used_quat = res
        self._execute(path, step_size=_STEP_APPROACH)

        # Refresh held_off now that IK has chosen its yaw — gripper
        # rotation during column-align changes the world-frame offset.
        self._refresh_held_offset()
        held_off = self._held_offset.copy()

        # 3. Final descent — short vertical drop at the target column.
        # Fast mode halves substeps (14 -> 7); the descent is pure -z
        # along the target column, no contact until detach, so coarser
        # substepping is collision-safe.
        fd_substeps = 7 if self._fast_column_align_substeps_halved() else 14
        place_pose = np.array([anchor[0] - held_off[0],
                                  anchor[1] - held_off[1],
                                  ee_z])
        # Phase 3.8 (fast-mode only): reuse the converged goal_q the probe
        # already computed for this exact (cell, yaw, descend) target —
        # same teleport pattern as column-align above.  The probe seeded
        # mink with the LUT "upright_descent" entry (a config AT the
        # descent z, with the right elbow posture), so its converged
        # goal_q is the right place_pose arm config.  Without this reuse,
        # plan_to_pose re-runs Cartesian-substep IK from the column-align
        # config (16 cm above) — mink with max_iters=100 can't bridge
        # that at boundary cells (ix=6+) and the chain fails despite the
        # geometry being feasible.
        res = None
        if self._fast_column_align_substeps_halved():
            qkey = tuple(np.round(used_quat, 6).tolist())
            cached_descent = probe_goal_q.get((qkey, "descend"))
            if cached_descent is not None:
                start_q = self.env.data.qpos[:7].copy()
                # Goal-pose collision check — only need this in fast mode
                # because no physics runs between waypoints; the joint-
                # lerp itself can clip transiently without corrupting
                # state.  If the GOAL clips a neighbour (happens at ix=6
                # boundary cells where the LUT config's elbow posture was
                # computed in an empty env), the cached config is unusable
                # for THIS scene — fall through to the Cartesian-substep
                # fallback below, which lets mink pick a scene-aware
                # elbow posture per substep.
                if self.env.is_collision_free(cached_descent):
                    res = ([start_q, cached_descent.copy()], used_quat)
        if res is None:
            # Cached teleport not available OR rejected by collision —
            # fall back to Cartesian-substep IK with the FULL executor's
            # settings (14 substeps, 1000 iters) so the fallback at
            # boundary cells (ix=6+) actually succeeds.  We know from
            # the FULL executor's prior runs that those cells ARE
            # reachable with this budget.  Costs ~500 ms extra per
            # affected put-upright; only fires on cache miss / collision.
            saved_iters = self.env.ik.max_iters
            try:
                self.env.ik.max_iters = max(saved_iters, 1000)
                res = self._try_plan_to_pose(place_pose, [used_quat],
                                                      n_substeps=14)
            finally:
                self.env.ik.max_iters = saved_iters
        if res is None:
            ee_now = self.env.data.site_xpos[self.ee_site_id].copy()
            self._err(
                f"put_upright {block_name}: final descent IK failed  "
                f"target=({place_pose[0]:.3f}, {place_pose[1]:.3f}, "
                f"{place_pose[2]:.3f}) from=({ee_now[0]:.3f}, "
                f"{ee_now[1]:.3f}, {ee_now[2]:.3f}) "
                f"quat={used_quat.tolist()}"
            )
            return False
        self._execute(res[0], step_size=_STEP_PLACE, precision=True)

        # 3b. Re-measure post-descent offset + nudge if drifted.  In
        # fast mode skip the nudge entirely — the held offset on a
        # kinematic attach is deterministic by construction (no physics
        # drift), so the corrected pose equals place_pose to ~1e-6.
        self._refresh_held_offset()
        held_off = self._held_offset.copy()
        corrected = np.array([anchor[0] - held_off[0],
                                 anchor[1] - held_off[1],
                                 ee_z])
        if (not self._fast_column_align_substeps_halved()
                  and np.linalg.norm(corrected - place_pose) > 0.003):
            res = self._try_plan_to_pose(corrected, [used_quat],
                                              n_substeps=8)
            if res is not None:
                self._execute(res[0], step_size=_STEP_PLACE, precision=True)
        self._log_phase(f"put-upright/{block_name}/pre-detach", place_pose,
                          block_name=block_name)
        self._trace_phase("put-upright:pre-detach")

        # 4. Detach.  Block drops the 4 cm of clearance onto the cell.
        self._detach_open()
        self._log_phase(f"put-upright/{block_name}/post-detach", place_pose,
                          block_name=block_name)

        # A1 extension: short-circuit the return-trip in fast mode.
        # Once detach is done the put is locked in; phases 5-7 only
        # validate arm-recovery.  Instead of skipping entirely (which
        # leaves the arm above the placed block — breaks the NEXT
        # check_action's start-of-chain transit), teleport to
        # NEUTRAL_HOME so the arm is in a canonical pose at exit.
        if not self._validate_put_upright_return():
            self.env.data.qpos[:7] = _HOME_NEUTRAL_Q
            self.env.data.qvel[:] = 0.0
            if getattr(self.env, "_attached", None) is not None:
                self.env._apply_attachment()
            mujoco.mj_forward(self.env.model, self.env.data)
            if self.env.controller is not None:
                self.env.controller.stop()
            self._trace_phase("put-upright:fast-teleport-home")
            return True

        # 5. Lift back to traverse_z — pure +z motion at the target column.
        lift_back = np.array([place_pose[0], place_pose[1], traverse_z])
        res = self._try_plan_to_pose(lift_back, [used_quat], n_substeps=10)
        if res is None:
            self._err(
                f"put_upright {block_name}: lift-back IK failed  "
                f"target=({lift_back[0]:.3f}, {lift_back[1]:.3f}, "
                f"{lift_back[2]:.3f})"
            )
            return False
        self._execute(res[0], step_size=_STEP_LIFT, precision=True)

        # 6. Retract at traverse_z to the front handoff y.  Cascading
        # fallbacks because the IK basin from lift_back to a -y retract
        # pose is fragile at +x corner cells:
        #   primary:  (anchor.x, front_handoff_y, traverse_z) — pure -y
        #             from lift_back, cleanest path when feasible.
        #   x=0 fb:   (0, front_handoff_y, traverse_z) — diagonal -y + -x
        #             helps when holding +x with FRONT_Y at low y is bad.
        #   handoff:  skip the explicit retract and rely on _to_handoff
        #             (which uses the cached handoff joint config, not
        #             ad-hoc IK) — bypasses the IK basin issue entirely.
        # The block is already detached + lifted to traverse_z, so the
        # placement is locked in; only the arm-transit needs to succeed.
        retract_primary = np.array([anchor[0], front_handoff_y, traverse_z])
        retract_fallback = np.array([0.0, front_handoff_y, traverse_z])
        res = self._try_plan_to_pose(retract_primary, [used_quat],
                                              n_substeps=12)
        if res is not None:
            self._execute(res[0], step_size=_STEP_APPROACH)
        else:
            res = self._try_plan_to_pose(retract_fallback, [used_quat],
                                                  n_substeps=12)
            if res is not None:
                self._execute(res[0], step_size=_STEP_APPROACH)
            else:
                # Both ad-hoc retracts failed.  Skip the explicit retract
                # and go directly to handoff via the cached joint config.
                if not self._to_handoff("stack", used_quat):
                    self._err(
                        f"put_upright {block_name}: -y retract + handoff "
                        f"fallback both failed  "
                        f"primary=({retract_primary[0]:.3f}, "
                        f"{retract_primary[1]:.3f}, "
                        f"{retract_primary[2]:.3f}) "
                        f"x0_fb=({retract_fallback[0]:.3f}, "
                        f"{retract_fallback[1]:.3f}, "
                        f"{retract_fallback[2]:.3f})"
                    )
                    return False
                # _to_handoff succeeded — arm is at handoff, done.
                return True

        # 7. Return to handoff (after a successful explicit retract).
        self._to_handoff("stack", used_quat)
        return True

    # -----------------------------------------------------------------
    # In-hand transforms
    # -----------------------------------------------------------------

    def _do_transform(self, region: str, target_quat: np.ndarray) -> bool:
        """Run an in-hand transform: move to hand-off in the new quat, then
        re-measure the (block - EE) offset since the rotation around the
        EE site swings the block centroid relative to the EE."""
        ok = self._to_handoff(region, target_quat)
        if ok:
            self._refresh_held_offset()
            self._log_phase(f"transform/{region}/{target_quat.tolist()[:2]}",
                              self._ee_pos(), block_name=self._held_block)
            self._trace_phase(f"transform:post")
        return ok

    def make_upright_from_x(self, block_name: str) -> bool:
        return self._do_transform("stack", _QUAT_FRONT_Y)

    def make_upright_from_y(self, block_name: str) -> bool:
        target = _quat_mul(_yaw_quat(math.pi / 2), _QUAT_FRONT_Y)
        return self._do_transform("stack", target)

    def make_flat_x_from_upright(self, block_name: str) -> bool:
        return self._do_transform("stack", _QUAT_TOP_DOWN_Y)

    def make_flat_y_from_upright(self, block_name: str) -> bool:
        return self._do_transform("stack", _QUAT_TOP_DOWN_X)

    def turn_x_to_y(self, block_name: str) -> bool:
        region = self._current_workspace()
        return self._do_transform(region, _QUAT_TOP_DOWN_X)

    def turn_y_to_x(self, block_name: str) -> bool:
        region = self._current_workspace()
        return self._do_transform(region, _QUAT_TOP_DOWN_Y)

    # -----------------------------------------------------------------
    # 3×1 long block: upright + transforms.  Same geometry as 2×1
    # (anchor at c_high for pick, midpoint of c_low/c_high for put);
    # the middle cell c_mid is part of the API for PDDL symmetry but
    # isn't used for grasping geometry.  The held block's actual
    # length is captured at runtime via _refresh_held_offset, so the
    # 2×1 and 3×1 chains work transparently with different block
    # heights.
    # -----------------------------------------------------------------

    def pick_long_upright(self, block_name: str,
                              c_low_id: str, c_mid_id: str,
                              c_high_id: str) -> bool:
        """Front-facing pick of a 3×1 upright spanning c_low..c_high."""
        return self.pick_upright(block_name, c_low_id, c_high_id)

    def put_long_upright(self, block_name: str,
                             c_low_id: str, c_mid_id: str,
                             c_high_id: str) -> bool:
        """Front-facing place of a 3×1 upright spanning c_low..c_high."""
        return self.put_upright(block_name, c_low_id, c_high_id)

    def make_long_upright_from_x(self, block_name: str) -> bool:
        return self.make_upright_from_x(block_name)

    def make_long_upright_from_y(self, block_name: str) -> bool:
        return self.make_upright_from_y(block_name)

    def make_long_flat_x_from_upright(self, block_name: str) -> bool:
        return self.make_flat_x_from_upright(block_name)

    def make_long_flat_y_from_upright(self, block_name: str) -> bool:
        return self.make_flat_y_from_upright(block_name)

    def turn_long_x_to_y(self, block_name: str) -> bool:
        return self.turn_x_to_y(block_name)

    def turn_long_y_to_x(self, block_name: str) -> bool:
        return self.turn_y_to_x(block_name)

    def _current_workspace(self) -> str:
        """Infer the current workspace from the sign of the base joint."""
        return "stack" if self.env.data.qpos[0] > 0 else "parts"


# ---------------------------------------------------------------------------
# Bridge wiring
# ---------------------------------------------------------------------------


def register_executor(bridge, executor: MultilevelBlocksExecutor) -> None:
    """Wire each PDDL action to a method on ``executor`` via
    ``bridge.action``.

    The fluent_delta returned by each handler is the minimal update
    consistent with the PDDL action's effects on gripper-orientation
    fluents.  Cell-occupancy fluents (``in``, ``empty``) are evaluated
    from sim by the bridge's predicate framework — no manual update.
    """

    @bridge.action("pick-cube")
    def _pc(env, fluents, b, c):
        ok = executor.pick_cube(b, c)
        delta = ({("held-cube", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("pick-flat-x")
    def _pfx(env, fluents, b, c1, c2):
        ok = executor.pick_flat_x(b, c1, c2)
        delta = ({("held-flat-x", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("pick-flat-y")
    def _pfy(env, fluents, b, c1, c2):
        ok = executor.pick_flat_y(b, c1, c2)
        delta = ({("held-flat-y", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("pick-upright")
    def _pup(env, fluents, b, c_low, c_high):
        ok = executor.pick_upright(b, c_low, c_high)
        delta = ({("held-upright", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("put-cube")
    def _puc(env, fluents, b, c):
        ok = executor.put_cube(b, c)
        delta = ({("held-cube", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("put-flat-x")
    def _pufx(env, fluents, b, c1, c2):
        ok = executor.put_flat_x(b, c1, c2)
        delta = ({("held-flat-x", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("put-flat-y")
    def _pufy(env, fluents, b, c1, c2):
        ok = executor.put_flat_y(b, c1, c2)
        delta = ({("held-flat-y", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("put-upright")
    def _puup(env, fluents, b, c_low, c_high):
        ok = executor.put_upright(b, c_low, c_high)
        delta = ({("held-upright", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-upright-from-x")
    def _muix(env, fluents, b):
        ok = executor.make_upright_from_x(b)
        delta = ({("held-flat-x", b): False, ("held-upright", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-upright-from-y")
    def _muiy(env, fluents, b):
        ok = executor.make_upright_from_y(b)
        delta = ({("held-flat-y", b): False, ("held-upright", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-flat-x-from-upright")
    def _mfxu(env, fluents, b):
        ok = executor.make_flat_x_from_upright(b)
        delta = ({("held-upright", b): False, ("held-flat-x", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-flat-y-from-upright")
    def _mfyu(env, fluents, b):
        ok = executor.make_flat_y_from_upright(b)
        delta = ({("held-upright", b): False, ("held-flat-y", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("turn-x-to-y")
    def _txy(env, fluents, b):
        ok = executor.turn_x_to_y(b)
        delta = ({("held-flat-x", b): False, ("held-flat-y", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("turn-y-to-x")
    def _tyx(env, fluents, b):
        ok = executor.turn_y_to_x(b)
        delta = ({("held-flat-y", b): False, ("held-flat-x", b): True}
                  if ok else {})
        return ok, delta

    # ------------------------------------------------------------------
    # 3×1 long block actions.  Reuse the held-flat-{x,y} / held-upright
    # fluents — the PDDL (long ?b) static predicate is what differentiates
    # the long variants from the oblong ones.
    # ------------------------------------------------------------------

    @bridge.action("pick-long-x")
    def _plx(env, fluents, b, c1, c2, c3):
        ok = executor.pick_long_x(b, c1, c2, c3)
        delta = ({("held-flat-x", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("pick-long-y")
    def _ply(env, fluents, b, c1, c2, c3):
        ok = executor.pick_long_y(b, c1, c2, c3)
        delta = ({("held-flat-y", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("pick-long-upright")
    def _plup(env, fluents, b, c_low, c_mid, c_high):
        ok = executor.pick_long_upright(b, c_low, c_mid, c_high)
        delta = ({("held-upright", b): True, ("gripper-empty",): False}
                  if ok else {})
        return ok, delta

    @bridge.action("put-long-x")
    def _pulx(env, fluents, b, c1, c2, c3):
        ok = executor.put_long_x(b, c1, c2, c3)
        delta = ({("held-flat-x", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("put-long-y")
    def _puly(env, fluents, b, c1, c2, c3):
        ok = executor.put_long_y(b, c1, c2, c3)
        delta = ({("held-flat-y", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("put-long-upright")
    def _pulup(env, fluents, b, c_low, c_mid, c_high):
        ok = executor.put_long_upright(b, c_low, c_mid, c_high)
        delta = ({("held-upright", b): False, ("gripper-empty",): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-long-upright-from-x")
    def _mluix(env, fluents, b):
        ok = executor.make_long_upright_from_x(b)
        delta = ({("held-flat-x", b): False, ("held-upright", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-long-upright-from-y")
    def _mluiy(env, fluents, b):
        ok = executor.make_long_upright_from_y(b)
        delta = ({("held-flat-y", b): False, ("held-upright", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-long-flat-x-from-upright")
    def _mlfxu(env, fluents, b):
        ok = executor.make_long_flat_x_from_upright(b)
        delta = ({("held-upright", b): False, ("held-flat-x", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("make-long-flat-y-from-upright")
    def _mlfyu(env, fluents, b):
        ok = executor.make_long_flat_y_from_upright(b)
        delta = ({("held-upright", b): False, ("held-flat-y", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("turn-long-x-to-y")
    def _tlxy(env, fluents, b):
        ok = executor.turn_long_x_to_y(b)
        delta = ({("held-flat-x", b): False, ("held-flat-y", b): True}
                  if ok else {})
        return ok, delta

    @bridge.action("turn-long-y-to-x")
    def _tlyx(env, fluents, b):
        ok = executor.turn_long_y_to_x(b)
        delta = ({("held-flat-y", b): False, ("held-flat-x", b): True}
                  if ok else {})
        return ok, delta
