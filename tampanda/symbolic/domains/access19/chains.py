"""Linear-IK chain pick + put trajectories for the access-19 domain.

The default ``PickPlaceExecutor.pick`` / ``.place`` each plan a single
trajectory which can't thread the gripper through access-19's
closed-top cubicle (for ``shelf_interior`` cells) or around the deck
top wall (for ``shelf_top`` cells).  This module provides chain-based
overrides for both — exposed via ``make_access19_pick_fn`` /
``make_access19_put_fn``, which the bridge consumes through its
``pick_fn`` / ``put_fn`` constructor parameters.

Validated by ``examples/analyze_palmy_reachability.py`` — 21/21
interior cells + 9/9 deck cells under the ``front_x`` quat at the
default ``table_pos=(0.35, 0.40, 0.0)``.

Two trajectory shapes, mirrored for pick and put (gripper close/open
and attach/detach are the only structural difference):

* ``shelf_interior``: column-align (front face) → row-by-row descent
  in y → final descent.  Pick: close + attach.  Put: detach + open.
  Then 4 cm lift → reverse row-step → exit to the column-approach
  pose.

* ``shelf_top``: withdraw to a clean pose in front of the cubicle if
  we're inside it → vertical lift to ``shelf_top_outer_z + 0.08``
  (clears link7 + 2 cm safety) → horizontal traverse over the target
  cell → descend.  Pick: close + attach.  Put: detach + open.  Then
  lift back to the safe altitude.  Cartesian-substep IK
  (``plan_to_pose`` with ``slerp_orientation=False``) so the EE
  follows a straight world-frame line — joint-space lerps from a low
  staging pose to a high deck-above pose sweep the wrist through the
  shelf top wall.

Both shapes try the FRONT quat and its 180°-around-hand-z twin at every
step (parallel-jaw invariance).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.symbolic.workspace import Cell, Workspace
from tampanda.symbolic.domains.access19.env_builder import (
    Access19Config,
)


_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])
_QUATS = (_FRONT_QUAT, _FRONT_QUAT_FLIPPED)

# Wrist clearance above the cubicle top wall during the deck traverse.
# link7 capsule hangs ~5.5 cm below the EE on palm-+y, so the safe
# altitude must be at least that + a 2 cm margin.
_SAFE_Z_ABOVE_SHELF_TOP = 0.10

# Controller-convergence tuning for ``_execute`` (Phase 5 fix).
# After each ``execute_path`` + ``wait_idle`` we check the L2 norm of
# (actual_q - target_q) over the 7 arm joints; if above
# ``_CONVERGE_TOL`` we re-issue the last waypoint up to
# ``_MAX_RETRIES`` extra cycles.  0.02 rad ≈ ~1 degree per joint on
# average — tight enough that the next chain IK probe seeds from a
# basin close to the planned one.
_CONVERGE_TOL = 0.02
_MAX_RETRIES = 3
# Grasp 1 cm below the cube top so link7 sits above the cubicle floor.
# Mirror of ``_build_access19_pick_fn``'s ``target_grasp_z`` formula.
_CUBE_GRASP_OFFSET = 0.010
# At attach time the held block is captured ``cube_half_z -
# _CUBE_GRASP_OFFSET`` BELOW the EE in world frame (the EE site is the
# grasp point at the top of the cube; the cube centre is below it).
# When PUTting, the cube must hover this much above its final resting
# centre PLUS a release clearance so physics can drop the block onto
# the surface cleanly instead of having the cube already in contact
# with it (which gets flagged as a held-vs-shelf collision and the
# descent plan fails).  Matches the ``place_clearance`` parameter of
# ``executor.place`` (default 3 mm).
_PLACE_CLEARANCE = 0.003


PutFn = Callable[[str, str, np.ndarray], bool]


def make_access19_put_fn(env, executor, workspace: Workspace,
                            config: Access19Config,
                            cube_half_z: Optional[float] = None,
                            lik: Optional[LinearIKPlanner] = None,
                            ) -> PutFn:
    """Build the put_fn the bridge dispatches to for access-19.

    Returns a closure ``put_fn(obj_name, cell_id, target_pos) -> bool``
    that the bridge's ``exec_put`` action invokes when the user has
    wired this domain's chain-based puts.  When omitted from the
    bridge construction, the legacy ``executor.place`` path is used.

    Args:
        env: The built FrankaEnvironment.
        executor: The same PickPlaceExecutor passed to the bridge.  Used
            for its ``planner`` (RRT*) so ``env.execute_path`` can
            smooth + execute the linear-IK paths, and for its
            ``settle_steps`` / step-size constants for consistency with
            the executor's own pace.
        workspace: The Workspace returned alongside the env builder.
        config: Access19Config (used for cube half-size if not
            explicitly passed).
        cube_half_z: Half-height of the held cube in metres.  Defaults
            to 0.040 (access-19's standard cube).
        lik: Optional pre-built LinearIKPlanner; defaults to the one
            already attached to ``executor``.
    """
    if lik is None:
        lik = executor.linear_ik_planner
    if cube_half_z is None:
        cube_half_z = 0.040

    region_i = workspace["shelf_interior"]
    region_t = workspace["shelf_top"]
    front_face_y = region_i.origin[1] - 0.02
    shelf_top_outer_z = region_t.level_z - cube_half_z

    ee_site_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site",
    )

    # --- helpers --------------------------------------------------------

    def _try_lerp(target: np.ndarray, n_substeps: int):
        for q in _QUATS:
            p = lik.plan_joint_lerp(target, q, n_substeps=n_substeps)
            if p is not None:
                return p
        return None

    def _try_cartesian(target: np.ndarray, n_substeps: int):
        for q in _QUATS:
            p = lik.plan_to_pose(target, q, n_substeps=n_substeps,
                                    slerp_orientation=False)
            if p is not None:
                return p
        return None

    def _try_lerp_then_cartesian(target: np.ndarray,
                                    lerp_substeps: int,
                                    cart_substeps: int):
        """Joint-lerp first (basin-stable for short moves), fall back to
        Cartesian-substep IK (Cartesian path preserved, useful around
        obstacles).  Use this for the short same-orientation moves
        (vertical lift / descent / lift-back) where the post-pick joint
        config can be in a basin where pure Cartesian-substep IK can't
        navigate.
        """
        p = _try_lerp(target, n_substeps=lerp_substeps)
        if p is not None:
            return p
        return _try_cartesian(target, n_substeps=cart_substeps)

    def _execute(path: List[np.ndarray], step_size: float) -> None:
        """Execute ``path`` then verify the controller actually converged
        on ``path[-1]``.

        At L4 return-all scale we observed FAST↔FULL divergence rooted
        in the controller leaving the arm slightly off the planned
        endpoint after each ``execute_path``.  Subsequent chain IK
        probes then seeded from the off-pose and rejected
        feasibility.  This wrapper re-issues the last waypoint until
        the arm is within ``_CONVERGE_TOL`` of ``path[-1]``, or gives
        up after ``_MAX_RETRIES`` extra cycles (in which case the
        chain's next probe still sees the off-pose and may fail —
        making the failure visible to the caller rather than silent).
        """
        env.execute_path(path, executor.planner, step_size=step_size)
        env.wait_idle(settle_steps=executor.settle_steps)
        target_q = np.asarray(path[-1], dtype=float)[:7]
        for _ in range(_MAX_RETRIES):
            actual_q = np.asarray(env.data.qpos[:7], dtype=float)
            err = float(np.linalg.norm(actual_q - target_q))
            if err < _CONVERGE_TOL:
                return
            env.execute_path([actual_q.copy(), target_q.copy()],
                                  executor.planner, step_size=step_size)
            env.wait_idle(settle_steps=executor.settle_steps)

    def _detach_and_open(obj_name: str) -> None:
        if executor.use_attachment and env._attached is not None:
            env.detach_object()
        env.controller.open_gripper()
        executor._wait_gripper_open()

    def _withdraw_to_cubicle_front() -> bool:
        """Row-step out of the cubicle to a clear staging-like pose.

        Always-safe Phase 0 for both interior and deck puts.  After a
        pick, the EE is inside the cubicle (xy footprint = shelf body)
        at low z; a vertical lift from there sweeps the held block
        through the cubicle's top wall.  This routine re-establishes a
        clean pose at ``(cur_x, front_face_y - 0.06, cur_z + 0.04)``
        — same column as the pick, slightly above the grasp z, just
        outside the cubicle's open face.  Idempotent: if the EE is
        already in front of the cubicle, returns immediately.

        Uses ``plan_joint_lerp`` (same gripper-orientation, short
        moves through the cubicle's open face) — Cartesian-substep IK
        is unnecessary here because there's no wall to thread around,
        just the existing row-step corridor.
        """
        cur_ee = env.data.site_xpos[ee_site_id].copy()
        # Y-threshold: if we're already at least 5 cm in front of the
        # cubicle's open face, no withdraw needed.
        if cur_ee[1] < front_face_y - 0.05:
            return True
        retreat_z = float(cur_ee[2]) + 0.04
        # Lift 4 cm in place first to clear any cube top in the cell.
        p = _try_lerp(np.array([cur_ee[0], cur_ee[1], retreat_z]),
                        n_substeps=6)
        if p is None:
            print("[access19 put withdraw] in-place lift failed")
            return False
        _execute(p, executor.retreat_step_size)

        # Reverse row-step at retreat_z back to the cubicle front row.
        front_row_y = region_i.origin[1] + 0.5 * region_i.cell_size
        cur_y = float(cur_ee[1])
        step = region_i.cell_size
        while cur_y > front_row_y + 1e-6:
            step_y = max(cur_y - step, front_row_y)
            p = _try_lerp(np.array([cur_ee[0], step_y, retreat_z]),
                            n_substeps=8)
            if p is None:
                print(f"[access19 put withdraw] reverse-step at y="
                      f"{step_y:.3f} failed")
                return False
            _execute(p, executor.retreat_step_size)
            cur_y = step_y

        # Exit to the column-approach anchor (outside the cubicle).
        p = _try_lerp(
            np.array([cur_ee[0], front_face_y - 0.06, retreat_z]),
            n_substeps=8,
        )
        if p is None:
            print("[access19 put withdraw] exit-to-anchor failed")
            return False
        _execute(p, executor.retreat_step_size)
        return True

    # --- shelf_interior put ---------------------------------------------

    def _put_interior(obj_name: str, target_pos: np.ndarray) -> bool:
        # 0. Withdraw to a clean pose in front of the cubicle.  If
        # we just picked, the EE is inside the cubicle and a direct
        # column-align lerp can sweep the held block through walls.
        if not _withdraw_to_cubicle_front():
            return False

        col_x = float(target_pos[0])
        # Palm-+y holds the block at ``EE + (0, GRASP_CONTACT_OFFSET, 0)``;
        # to place the block CENTRE at ``target_pos[1]``, the EE must
        # target ``target_y - GRASP_CONTACT_OFFSET``.
        target_y = float(target_pos[1]) - GRASP_CONTACT_OFFSET
        # +_PLACE_CLEARANCE so the held block hovers 3 mm above its
        # resting centre during descent — keeps the block clear of the
        # cubicle floor so held-vs-other-cube intersections at the
        # target cell are still detected (rather than the held block
        # touching the floor on every put).
        grasp_z = (float(target_pos[2]) + cube_half_z
                    - _CUBE_GRASP_OFFSET + _PLACE_CLEARANCE)
        approach = np.array(
            [col_x, front_face_y - 0.06, grasp_z + 0.02])

        # 1. Column-align approach (outside the open face).
        path = _try_lerp(approach, n_substeps=20)
        if path is None:
            print("[access19 put_interior] approach plan failed")
            return False
        _execute(path, executor.approach_step_size)

        # Tighten controller delta for the in-cubicle precision phase.
        env.controller._advance_delta_override = 0.01

        # 2. Row-by-row descent in y until aligned with the target cell.
        front_row_y = region_i.origin[1] + 0.5 * region_i.cell_size
        cur_y = front_row_y
        step = region_i.cell_size
        while cur_y < target_y - 1e-6:
            step_y = min(cur_y + step, target_y)
            p = _try_lerp(np.array([col_x, step_y, grasp_z]),
                            n_substeps=8)
            if p is None:
                env.controller._advance_delta_override = 0.1
                print(f"[access19 put_interior] row-step at y={step_y:.3f}"
                      f" plan failed")
                return False
            _execute(p, executor.place_step_size)
            cur_y = step_y

        # 3. Final descent to the placement pose.
        p = _try_lerp(np.array([col_x, target_y, grasp_z]),
                        n_substeps=6)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 put_interior] final descent plan failed")
            return False
        _execute(p, executor.place_step_size)

        # 4. Place: detach attachment + open gripper.  Past this point
        # the block is on the surface and we don't roll back — any
        # retreat-plan failure leaves a partially-retracted arm, which
        # the next operation can plan from.
        _detach_and_open(obj_name)

        # 5. Lift 4 cm.
        retreat_z = grasp_z + 0.04
        p = _try_lerp(np.array([col_x, target_y, retreat_z]),
                        n_substeps=6)
        if p is not None:
            _execute(p, executor.retreat_step_size)

        # 6. Reverse row-step at retreat z.
        cur_y = target_y
        while cur_y > front_row_y + 1e-6:
            step_y = max(cur_y - step, front_row_y)
            p = _try_lerp(np.array([col_x, step_y, retreat_z]),
                            n_substeps=8)
            if p is None:
                env.controller._advance_delta_override = 0.1
                return True  # block placed; partial retreat acceptable
            _execute(p, executor.retreat_step_size)
            cur_y = step_y

        # 7. Exit to the column-approach pose outside the front face.
        p = _try_lerp(
            np.array([col_x, front_face_y - 0.06, retreat_z]),
            n_substeps=8,
        )
        if p is not None:
            _execute(p, executor.retreat_step_size)

        env.controller._advance_delta_override = 0.1
        return True

    # --- shelf_top put --------------------------------------------------

    def _put_deck(obj_name: str, target_pos: np.ndarray) -> bool:
        # 0. Withdraw to a clean pose in front of the cubicle.  The
        # vertical lift in phase 1 must start from a pose whose xy is
        # OUTSIDE the cubicle's footprint — otherwise the held block
        # sweeps through the cubicle's top wall on the way up.
        if not _withdraw_to_cubicle_front():
            return False

        col_x = float(target_pos[0])
        # Same hand-z offset compensation as the interior put — palm-+y
        # holds the block ahead of the EE in +y, so target the EE
        # behind the desired block centre.
        target_y = float(target_pos[1]) - GRASP_CONTACT_OFFSET
        # +_PLACE_CLEARANCE keeps the held block 3 mm above the deck
        # surface at the descent endpoint — without it the cube
        # bottom touches the deck on every descent and the collision
        # check fails before we ever reach the release point.
        deck_z = (float(target_pos[2]) + cube_half_z
                   - _CUBE_GRASP_OFFSET + _PLACE_CLEARANCE)
        safe_z = shelf_top_outer_z + _SAFE_Z_ABOVE_SHELF_TOP
        cur_ee = env.data.site_xpos[ee_site_id].copy()

        # 1. Vertical lift to safe_z.  Same-orientation move; lerp
        # first for basin stability, Cartesian-substep IK as fallback.
        p = _try_lerp_then_cartesian(
            np.array([cur_ee[0], cur_ee[1], safe_z]),
            lerp_substeps=16, cart_substeps=14)
        if p is None:
            print("[access19 put_deck] vertical lift plan failed")
            return False
        _execute(p, executor.approach_step_size)

        env.controller._advance_delta_override = 0.01

        # 2. Horizontal traverse at safe_z.  Cartesian-only — the
        # Cartesian straight line must stay above the shelf top wall;
        # joint-lerp can swing the wrist through it.
        p = _try_cartesian(np.array([col_x, target_y, safe_z]),
                              n_substeps=16)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 put_deck] traverse plan failed")
            return False
        _execute(p, executor.approach_step_size)

        # 3. Descend to deck.
        p = _try_lerp_then_cartesian(
            np.array([col_x, target_y, deck_z]),
            lerp_substeps=10, cart_substeps=12)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 put_deck] descent plan failed")
            return False
        _execute(p, executor.place_step_size)

        # 4. Place: detach + open gripper.
        _detach_and_open(obj_name)

        # 5. Lift back to safe_z.
        p = _try_lerp_then_cartesian(
            np.array([col_x, target_y, safe_z]),
            lerp_substeps=10, cart_substeps=12)
        if p is not None:
            _execute(p, executor.retreat_step_size)

        env.controller._advance_delta_override = 0.1
        return True

    # --- dispatch -------------------------------------------------------

    def put_fn(obj_name: str, cell_id: str,
                 target_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
        # IMPORTANT: do NOT add the held block to the collision
        # exception list.  ``attach_object_to_ee`` already added it to
        # ``_collision_body_ids`` so its contacts with the gripper
        # count as self-collisions and are skipped.  Adding it to
        # ``_collision_exception_ids`` on top would skip ALL its
        # contacts — including held-vs-other-cube contacts during the
        # descent, letting the chain pass straight through whatever
        # is already at the target cell and tip it off the deck on
        # release.  Mirrors the comment in ``executor.place``
        # (``pick_place.py:386-393``).
        if cell.region == "shelf_interior":
            return _put_interior(obj_name, target_pos)
        if cell.region == "shelf_top":
            return _put_deck(obj_name, target_pos)
        raise ValueError(
            f"access19 put_fn: unknown region {cell.region!r}")

    return put_fn


PickFn = Callable[[str, str, np.ndarray], bool]


def make_access19_pick_fn(env, executor, workspace: Workspace,
                             config: Access19Config,
                             cube_half_z: Optional[float] = None,
                             lik: Optional[LinearIKPlanner] = None,
                             ) -> PickFn:
    """Mirror of :func:`make_access19_put_fn` for picks.

    Same withdraw / column-align / row-step / Cartesian-traverse
    primitives, but at the bottom of each chain the gripper *closes*
    and the block is kinematically attached to the EE — rather than
    detached + opened.  Returned closure signature:
    ``pick_fn(obj_name, cell_id, source_pos) -> bool`` where
    ``source_pos`` is the world position of the block to pick (a
    palm-+y EE pose at ``source_pos - hand_z * GRASP_CONTACT_OFFSET``
    is solved for via IK).

    Bridge plumbing: the bridge's ``exec_pick`` looks up the block's
    current world pose (``env.get_object_pose(...)``) and passes it as
    ``source_pos``.  Same wiring pattern as ``put_fn``.
    """
    if lik is None:
        lik = executor.linear_ik_planner
    if cube_half_z is None:
        cube_half_z = 0.040

    region_i = workspace["shelf_interior"]
    region_t = workspace["shelf_top"]
    front_face_y = region_i.origin[1] - 0.02
    shelf_top_outer_z = region_t.level_z - cube_half_z

    ee_site_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site",
    )

    def _try_lerp(target: np.ndarray, n_substeps: int):
        for q in _QUATS:
            p = lik.plan_joint_lerp(target, q, n_substeps=n_substeps)
            if p is not None:
                return p
        return None

    def _try_cartesian(target: np.ndarray, n_substeps: int):
        for q in _QUATS:
            p = lik.plan_to_pose(target, q, n_substeps=n_substeps,
                                    slerp_orientation=False)
            if p is not None:
                return p
        return None

    def _try_lerp_then_cartesian(target: np.ndarray,
                                    lerp_substeps: int,
                                    cart_substeps: int):
        p = _try_lerp(target, n_substeps=lerp_substeps)
        if p is not None:
            return p
        return _try_cartesian(target, n_substeps=cart_substeps)

    def _execute(path: List[np.ndarray], step_size: float) -> None:
        """Execute ``path`` then verify the controller actually converged
        on ``path[-1]``.

        At L4 return-all scale we observed FAST↔FULL divergence rooted
        in the controller leaving the arm slightly off the planned
        endpoint after each ``execute_path``.  Subsequent chain IK
        probes then seeded from the off-pose and rejected
        feasibility.  This wrapper re-issues the last waypoint until
        the arm is within ``_CONVERGE_TOL`` of ``path[-1]``, or gives
        up after ``_MAX_RETRIES`` extra cycles (in which case the
        chain's next probe still sees the off-pose and may fail —
        making the failure visible to the caller rather than silent).
        """
        env.execute_path(path, executor.planner, step_size=step_size)
        env.wait_idle(settle_steps=executor.settle_steps)
        target_q = np.asarray(path[-1], dtype=float)[:7]
        for _ in range(_MAX_RETRIES):
            actual_q = np.asarray(env.data.qpos[:7], dtype=float)
            err = float(np.linalg.norm(actual_q - target_q))
            if err < _CONVERGE_TOL:
                return
            env.execute_path([actual_q.copy(), target_q.copy()],
                                  executor.planner, step_size=step_size)
            env.wait_idle(settle_steps=executor.settle_steps)

    def _close_and_attach(obj_name: str) -> None:
        env.controller.close_gripper()
        executor._wait_gripper_closed()
        if executor.use_attachment:
            env.attach_object_to_ee(obj_name)

    def _withdraw_to_cubicle_front() -> bool:
        """Same Phase 0 as the put_fn — get the EE to a clean staging-
        like pose in front of the cubicle before the chain begins."""
        cur_ee = env.data.site_xpos[ee_site_id].copy()
        if cur_ee[1] < front_face_y - 0.05:
            return True
        retreat_z = float(cur_ee[2]) + 0.04
        p = _try_lerp(np.array([cur_ee[0], cur_ee[1], retreat_z]),
                        n_substeps=6)
        if p is None:
            print("[access19 pick withdraw] in-place lift failed")
            return False
        _execute(p, executor.retreat_step_size)

        front_row_y = region_i.origin[1] + 0.5 * region_i.cell_size
        cur_y = float(cur_ee[1])
        step = region_i.cell_size
        while cur_y > front_row_y + 1e-6:
            step_y = max(cur_y - step, front_row_y)
            p = _try_lerp(np.array([cur_ee[0], step_y, retreat_z]),
                            n_substeps=8)
            if p is None:
                print(f"[access19 pick withdraw] reverse-step at y="
                      f"{step_y:.3f} failed")
                return False
            _execute(p, executor.retreat_step_size)
            cur_y = step_y

        p = _try_lerp(
            np.array([cur_ee[0], front_face_y - 0.06, retreat_z]),
            n_substeps=8,
        )
        if p is None:
            print("[access19 pick withdraw] exit-to-anchor failed")
            return False
        _execute(p, executor.retreat_step_size)
        return True

    # --- shelf_interior pick --------------------------------------------

    def _pick_interior(obj_name: str, source_pos: np.ndarray) -> bool:
        if not _withdraw_to_cubicle_front():
            return False

        col_x = float(source_pos[0])
        # Palm-+y holds the block at ``EE + (0, GRASP_CONTACT_OFFSET, 0)``;
        # to grasp the block whose centre is at ``source_pos``, the EE
        # must target ``source_y - GRASP_CONTACT_OFFSET``.
        target_y = float(source_pos[1]) - GRASP_CONTACT_OFFSET
        grasp_z = float(source_pos[2]) + cube_half_z - _CUBE_GRASP_OFFSET
        approach = np.array(
            [col_x, front_face_y - 0.06, grasp_z + 0.02])

        path = _try_lerp(approach, n_substeps=20)
        if path is None:
            print("[access19 pick_interior] approach plan failed")
            return False
        _execute(path, executor.approach_step_size)

        env.controller._advance_delta_override = 0.01

        # Row-step descent — block at this column already counts as
        # part of the gripper-collision exception list, so the fingers
        # can close around it.
        front_row_y = region_i.origin[1] + 0.5 * region_i.cell_size
        cur_y = front_row_y
        step = region_i.cell_size
        while cur_y < target_y - 1e-6:
            step_y = min(cur_y + step, target_y)
            p = _try_lerp(np.array([col_x, step_y, grasp_z]),
                            n_substeps=8)
            if p is None:
                env.controller._advance_delta_override = 0.1
                print(f"[access19 pick_interior] row-step at y="
                      f"{step_y:.3f} plan failed")
                return False
            _execute(p, executor.grasp_step_size)
            cur_y = step_y

        p = _try_lerp(np.array([col_x, target_y, grasp_z]),
                        n_substeps=6)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 pick_interior] final descent plan failed")
            return False
        _execute(p, executor.grasp_step_size)

        # Grasp: close gripper + attach.  Mirror of put's
        # detach-and-open; past this point we treat the pick as
        # successful and any retreat-plan failure just leaves the
        # arm partially retracted with the block in hand.
        _close_and_attach(obj_name)

        # Lift 4 cm.
        retreat_z = grasp_z + 0.04
        p = _try_lerp(np.array([col_x, target_y, retreat_z]),
                        n_substeps=6)
        if p is not None:
            _execute(p, executor.lift_step_size)

        # Reverse row-step.
        cur_y = target_y
        while cur_y > front_row_y + 1e-6:
            step_y = max(cur_y - step, front_row_y)
            p = _try_lerp(np.array([col_x, step_y, retreat_z]),
                            n_substeps=8)
            if p is None:
                env.controller._advance_delta_override = 0.1
                return True  # block in hand; partial retreat OK
            _execute(p, executor.retreat_step_size)
            cur_y = step_y

        # Exit to the column-approach anchor.
        p = _try_lerp(
            np.array([col_x, front_face_y - 0.06, retreat_z]),
            n_substeps=8,
        )
        if p is not None:
            _execute(p, executor.retreat_step_size)

        env.controller._advance_delta_override = 0.1
        return True

    # --- shelf_top pick -------------------------------------------------

    def _pick_deck(obj_name: str, source_pos: np.ndarray) -> bool:
        if not _withdraw_to_cubicle_front():
            return False

        # Close gripper kinematically before the transit lift + traverse.
        # An OPEN gripper's finger spread (~8 cm in x) overlaps the
        # x-extents of cubes one column away from the target column
        # at safe_z — IK then fails to find a Cartesian or lerp path.
        # Symmetric ``put_deck`` doesn't hit this because the held
        # block keeps the gripper closed throughout.  Direct qpos
        # write (no physics) avoids the gravity-drift that
        # ``_wait_gripper_closed`` introduced when called right after
        # ``_set_home``.
        env.data.qpos[7:9] = 0.0
        env.data.ctrl[7] = -0.2          # match controller's "closed" target
        mujoco.mj_forward(env.model, env.data)

        col_x = float(source_pos[0])
        target_y = float(source_pos[1]) - GRASP_CONTACT_OFFSET
        grasp_z = float(source_pos[2]) + cube_half_z - _CUBE_GRASP_OFFSET
        safe_z = shelf_top_outer_z + _SAFE_Z_ABOVE_SHELF_TOP
        cur_ee = env.data.site_xpos[ee_site_id].copy()

        # Vertical lift to safe_z.  Lerp-first like the put — keeps
        # the IK basin stable from a low post-staging pose.
        p = _try_lerp_then_cartesian(
            np.array([cur_ee[0], cur_ee[1], safe_z]),
            lerp_substeps=16, cart_substeps=14)
        if p is None:
            print("[access19 pick_deck] vertical lift plan failed")
            return False
        _execute(p, executor.approach_step_size)

        env.controller._advance_delta_override = 0.01

        # Horizontal traverse — Cartesian first (preserves orientation
        # for repeatable behaviour); lerp fallback is safe here because
        # the hand is EMPTY on the outbound pick (no held block to
        # sweep through the top wall — only the gripper itself, which
        # the env's collision check will reject anyway).  Cartesian-
        # only failed at certain edge-column targets (col_5) when the
        # IK basin from staging diverged from the basin reached during
        # symmetric put_deck calls.
        p = _try_cartesian(np.array([col_x, target_y, safe_z]),
                              n_substeps=16)
        if p is None:
            p = _try_lerp(np.array([col_x, target_y, safe_z]),
                              n_substeps=16)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 pick_deck] traverse plan failed")
            return False
        _execute(p, executor.approach_step_size)

        # Re-open gripper just before descent so it can close around
        # the target block at grasp z.
        env.controller.open_gripper()
        executor._wait_gripper_open()

        # Descend to grasp z.
        p = _try_lerp_then_cartesian(
            np.array([col_x, target_y, grasp_z]),
            lerp_substeps=10, cart_substeps=12)
        if p is None:
            env.controller._advance_delta_override = 0.1
            print("[access19 pick_deck] descent plan failed")
            return False
        _execute(p, executor.grasp_step_size)

        # Grasp: close + attach.
        _close_and_attach(obj_name)

        # Lift back to safe_z.
        p = _try_lerp_then_cartesian(
            np.array([col_x, target_y, safe_z]),
            lerp_substeps=10, cart_substeps=12)
        if p is not None:
            _execute(p, executor.lift_step_size)

        env.controller._advance_delta_override = 0.1
        return True

    def _cell_occupied(cell_id_to_check: str,
                              exclude_obj: Optional[str] = None) -> bool:
        """Return True if any non-excluded object body is centred at
        ``cell_id_to_check``.  Used as a chain-level precondition check
        for pick_deck (the hand body extends ~10 cm in +y from the EE,
        so the +y-neighbouring deck cell must be empty)."""
        try:
            target_cell = Cell.parse(cell_id_to_check)
        except (KeyError, ValueError):
            return False
        try:
            target_pos = np.asarray(workspace.pose_for(target_cell))
        except (KeyError, ValueError):
            return False
        region = workspace[target_cell.region]
        tol_xy = float(region.cell_size) * 0.5
        tol_z = 0.05
        for bid in range(env.model.nbody):
            name = env.model.body(bid).name
            if not (name.startswith("blocker_") or name == "ooi"):
                continue
            if name == exclude_obj:
                continue
            try:
                pos, _ = env.get_object_pose(name)
            except Exception:
                continue
            p = np.asarray(pos, dtype=float)
            if (abs(p[0] - target_pos[0]) < tol_xy
                    and abs(p[1] - target_pos[1]) < tol_xy
                    and abs(p[2] - target_pos[2]) < tol_z):
                return True
        return False

    def pick_fn(obj_name: str, cell_id: str,
                  source_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
        # NOTE: a "+y neighbour empty" precondition for pick_deck was
        # tried here (state-audit at action 39 of an L4 return-all plan
        # showed the Franka hand_c body extending ~10 cm in +y from
        # the EE site knocks the +y-neighbour cube by ~42 mm during
        # pick / close).  Enforcing the precondition correctly
        # rejects those picks, but on the dense L4 top deck (18 cubes
        # + OoI) it forces every Phase-2 pick to start at iy_top=6,
        # which is past the Franka's reach in the chain's palm-+y
        # configuration — so all picks dead-end and A* can't unwind
        # the return.  Need a chain redesign (palm-down for top-deck
        # picks) before the precondition can be re-enabled.  See
        # DIVERGENCES.md for details.
        # Collision exception so the gripper can be in contact with the
        # block during the final descent + close phase without the
        # planner flagging it.  Mirror of the put_fn pattern.
        env.add_collision_exception(obj_name)
        try:
            if cell.region == "shelf_interior":
                return _pick_interior(obj_name, source_pos)
            if cell.region == "shelf_top":
                return _pick_deck(obj_name, source_pos)
            raise ValueError(
                f"access19 pick_fn: unknown region {cell.region!r}")
        finally:
            env.remove_collision_exception(obj_name)

    return pick_fn
