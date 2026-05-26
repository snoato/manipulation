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
from tampanda.symbolic.domains.tabletop_access.env_builder import (
    TabletopAccessConfig,
)


_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])
_QUATS = (_FRONT_QUAT, _FRONT_QUAT_FLIPPED)

# Wrist clearance above the cubicle top wall during the deck traverse.
# link7 capsule hangs ~5.5 cm below the EE on palm-+y, so the safe
# altitude must be at least that + a 2 cm margin.
_SAFE_Z_ABOVE_SHELF_TOP = 0.08
# Controller-convergence tuning for the access chains' ``_execute``:
# after each execute_path + wait_idle, re-issue the last waypoint up to
# ``_MAX_RETRIES`` times until the arm is within ``_CONVERGE_TOL`` rad of
# the planned endpoint (so the next chain IK probe seeds from the right
# basin).  Mirror of the access-19 chains.
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
                            config: TabletopAccessConfig,
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
        config: TabletopAccessConfig (used for cube half-size if not
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
        env.execute_path(path, executor.planner, step_size=step_size)
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
                             config: TabletopAccessConfig,
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
        env.execute_path(path, executor.planner, step_size=step_size)
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

    def pick_fn(obj_name: str, cell_id: str,
                  source_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
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


# ======================================================================
# access (3-tier free-standing shelf) chains
# ----------------------------------------------------------------------
# Unlike access-19's closed-top cubicle, the ``access`` shelf is open-
# front and free-standing: every region (floor_left / floor_right /
# middle_deck / top_deck) is reached from the open -y face, so a SINGLE
# front-approach chain shape covers all four — no Cartesian traverse
# over a top wall.  The only per-region difference is the vertical
# window the gripper may occupy: bounded below by link7 clearance over
# the region's own surface and above by the deck slab overhead (no
# ceiling for top_deck).  Reuses every reliability mechanism from the
# access-19 chains above (``_execute`` convergence retry, the always-
# safe withdraw-to-front Phase 0, held-block-not-collision-exempted,
# GRASP_CONTACT_OFFSET + _PLACE_CLEARANCE, FAST mode).
# ======================================================================

_ACCESS_REGIONS = ("floor_left", "floor_right", "middle_deck", "top_deck")
# link7 capsule hangs ~6 cm below the EE on palm-+y → floor of the
# per-region z-window.  HAND_TOP keeps the wrist clear of the deck slab
# overhead.  ITEM_HALF_Z_REF recovers the true surface z from a
# GridRegion's ``level_z`` (which bakes in a reference item half-height
# in the access workspace).  Values mirror ``reachability.py``.
_ACCESS_LINK7_SAFETY = 0.060
_ACCESS_HAND_TOP_SAFETY = 0.080
_ACCESS_ITEM_HALF_Z_REF = 0.05


def _access_geometry(env, workspace: Workspace):
    """Return ``(regions, front_face_y, z_windows)`` for the access shelf.

    ``front_face_y`` is the world y of the shelf's open -y face, walked
    off the ``shelf`` body's geoms (the region origins are inset by
    ``hand_clearance`` so they can't be used directly).  ``z_windows``
    maps each region name to ``(z_lo, z_hi)`` the EE may occupy there.
    """
    regions = {n: workspace[n] for n in _ACCESS_REGIONS}
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "shelf")
    fallback = min(r.origin[1] for r in regions.values()) - 0.03
    if body_id < 0:
        front_face_y = fallback
    else:
        body_y = float(env.model.body(body_id).pos[1])
        ys = [body_y + float(env.model.geom_pos[g][1])
              - float(env.model.geom_size[g][1])
              for g in range(env.model.ngeom)
              if env.model.geom_bodyid[g] == body_id]
        front_face_y = min(ys) if ys else fallback

    surface_z = {n: r.level_z - _ACCESS_ITEM_HALF_Z_REF
                 for n, r in regions.items()}
    sorted_surf = sorted(set(surface_z.values()))

    def _ceil(s: float) -> float:
        for nxt in sorted_surf:
            if nxt > s + 1e-6:
                return nxt - _ACCESS_HAND_TOP_SAFETY
        return float("inf")  # top_deck — open above

    z_windows = {n: (surface_z[n] + _ACCESS_LINK7_SAFETY, _ceil(surface_z[n]))
                 for n in regions}
    return regions, front_face_y, z_windows


def make_access_chains(env, executor, workspace: Workspace,
                       lik: Optional[LinearIKPlanner] = None):
    """Build ``(pick_fn, put_fn)`` for the 3-tier ``access`` shelf.

    Both closures take ``(obj_name, cell_id, pos)`` — ``pos`` is the
    object's current world centre (pick) or its intended resting world
    centre (put).  Object half-height is read live from the env, so the
    heterogeneous YCB-proxy items are handled per-object.
    """
    if lik is None:
        lik = executor.linear_ik_planner

    regions, front_face_y, z_windows = _access_geometry(env, workspace)
    ee_site_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    # --- per-level hand-off poses --------------------------------------
    # Each level (floor_left / floor_right / middle_deck / top_deck) gets
    # its own staging config: a palm-+y pose just outside the open front
    # face at that level's height, IK-solved once.  The chain teleports
    # the arm to the target level's hand-off before operating there, so
    # every front-approach starts from a reliable per-level basin instead
    # of lerping across a big vertical change (cross-level lerps swept the
    # wrist / held block through a deck slab).  A held object rides along
    # via the attachment hook.  Teleporting between hand-off points is an
    # accepted simplification for now — the collision-checked motion that
    # matters for feasibility happens WITHIN a level.
    def _region_center_x(r) -> float:
        return r.origin[0] + 0.5 * r.cells_x * r.cell_size

    # Per-level seed configs to try for the hand-off IK — the first that
    # converges within tolerance wins.  The side floor compartments sit
    # at the x-extremes and converge to a better basin from a shoulder-
    # rotated seed, so we offer a couple of alternates ("multiple IK
    # seeds").
    _HANDOFF_SEEDS = (
        np.array([np.pi / 2, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7854]),
        np.array([np.pi / 2, 0.30, 0.0, -2.0, 0.0, 1.9, -0.7854]),
        np.array([np.pi / 2, -0.30, 0.0, -2.0, 0.0, 1.9, -0.7854]),
    )

    def _solve_handoff(target_z: float, center_x: float) -> np.ndarray:
        saved_q = env.data.qpos.copy()
        target = np.array([center_x, front_face_y - 0.08, target_z])
        sp, so = env.ik.pos_threshold, env.ik.ori_threshold
        env.ik.pos_threshold, env.ik.ori_threshold = 0.005, 5e-3
        best_q, best_err = None, np.inf
        for seed in _HANDOFF_SEEDS:
            env.data.qpos[:7] = seed
            mujoco.mj_forward(env.model, env.data)
            env.ik.update_configuration(env.data.qpos)
            env.ik.set_target_position(target, _FRONT_QUAT)
            env.ik.converge_ik(0.005)
            q7 = env.ik.configuration.q[:7].copy()
            env.data.qpos[:7] = q7
            mujoco.mj_forward(env.model, env.data)
            err = float(np.linalg.norm(env.data.site_xpos[ee_site_id] - target))
            if err < best_err:
                best_q, best_err = q7, err
            if err < 0.01:
                break
        env.ik.pos_threshold, env.ik.ori_threshold = sp, so
        env.data.qpos[:] = saved_q
        mujoco.mj_forward(env.model, env.data)
        return np.concatenate([best_q, [0.04, 0.04]])

    def _handoff_z(region_name: str) -> float:
        # Stage at the region's grasp height, EXCEPT the floor
        # compartments: their level_z sits at the world floor, so a held
        # object would hang below it (start-config collision) and the
        # approach can't plan.  Stage high in the floor opening instead
        # so a held object clears the floor; the approach then descends.
        z_lo, z_hi = z_windows[region_name]
        if region_name in ("floor_left", "floor_right"):
            return float(min(z_hi - 0.02, z_lo + 0.08))
        return regions[region_name].level_z

    hand_off = {n: _solve_handoff(_handoff_z(n), _region_center_x(regions[n]))
                for n in _ACCESS_REGIONS}

    def _goto_handoff(region_name: str) -> None:
        if env.controller is not None:
            env.controller.stop()
        q = hand_off[region_name]
        env.data.qpos[: len(q)] = q
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        if getattr(env, "_attached", None) is not None:
            env._apply_attachment()  # held object rides the teleport

    # --- motion helpers (mirror the access-19 chains) -------------------

    def _try_lerp(target: np.ndarray, n_substeps: int):
        for q in _QUATS:
            p = lik.plan_joint_lerp(target, q, n_substeps=n_substeps)
            if p is not None:
                return p
        return None

    def _row_step_substeps(default: int) -> int:
        return 1 if getattr(env, "_fast_mode", False) else default

    def _execute(path: List[np.ndarray], step_size: float) -> None:
        env.execute_path(path, executor.planner, step_size=step_size)
        env.wait_idle(settle_steps=executor.settle_steps)
        target_q = np.asarray(path[-1], dtype=float)[:7]
        for _ in range(_MAX_RETRIES):
            actual_q = np.asarray(env.data.qpos[:7], dtype=float)
            if float(np.linalg.norm(actual_q - target_q)) < _CONVERGE_TOL:
                return
            env.execute_path([actual_q.copy(), target_q.copy()],
                             executor.planner, step_size=step_size)
            env.wait_idle(settle_steps=executor.settle_steps)

    def _close_and_attach(obj_name: str) -> None:
        env.controller.close_gripper()
        executor._wait_gripper_closed()
        if executor.use_attachment:
            env.attach_object_to_ee(obj_name)
        # Capture the world-frame z-offset between the held object's
        # centre and the EE site.  The attachment keeps this constant
        # (same front-quat orientation throughout), so the subsequent
        # put can place the EE such that the object lands exactly on the
        # target surface — region-independent, so cross-region transfers
        # land correctly.  When a state with ``holding`` is RESTORED
        # (no pick happened), the restore path must set this same value
        # (canonical grasp-high) so puts stay correct; the put falls
        # back to that canonical offset if it's unset.
        obj_c = np.asarray(env.get_object_pose(obj_name)[0], dtype=float)
        ee_z = float(env.data.site_xpos[ee_site_id][2])
        executor._held_grasp_dz = float(obj_c[2] - ee_z)

    def _detach_and_open(obj_name: str) -> None:
        if executor.use_attachment and env._attached is not None:
            env.detach_object()
        env.controller.open_gripper()
        executor._wait_gripper_open()
        executor._held_grasp_dz = None

    # --- shared front-approach chain ------------------------------------

    def _front_chain(obj_name: str, region_name: str, col_x: float,
                     target_y: float, work_z: float, bottom_fn) -> bool:
        if region_name not in regions:
            raise ValueError(f"access chain: unknown region {region_name!r}")
        region = regions[region_name]
        z_lo, z_hi = z_windows[region_name]
        work_z = float(min(max(work_z, z_lo), z_hi))

        # Teleport to this level's hand-off basin (held object rides along).
        _goto_handoff(region_name)

        # 1. Column-align approach, outside the open front face.  The
        # tight floor / middle compartments have a narrow IK basin —
        # try several approach heights (mirrors reachability.py) so a
        # single basin can be found across deep cells.
        approach_zs: List[float] = []
        for dz in (0.05, 0.02, -0.02, 0.10):
            z = float(min(max(work_z + dz, z_lo), z_hi))
            if z not in approach_zs:
                approach_zs.append(z)
        p = None
        for az in approach_zs:
            p = _try_lerp(np.array([col_x, front_face_y - 0.06, az]),
                          n_substeps=20)
            if p is not None:
                break
        if p is None:
            print(f"[access chain {region_name}] approach plan failed")
            return False
        _execute(p, executor.approach_step_size)

        env.controller._advance_delta_override = 0.01
        try:
            # 2. Row-by-row descent in y to the target cell.
            front_row_y = region.origin[1] + 0.5 * region.cell_size
            cur_y = front_row_y
            step = region.cell_size
            while cur_y < target_y - 1e-6:
                step_y = min(cur_y + step, target_y)
                p = _try_lerp(np.array([col_x, step_y, work_z]),
                              n_substeps=_row_step_substeps(8))
                if p is None:
                    print(f"[access chain {region_name}] row-step at "
                          f"y={step_y:.3f} failed")
                    return False
                _execute(p, executor.place_step_size)
                cur_y = step_y

            # 3. Final descent / advance to the work pose.
            p = _try_lerp(np.array([col_x, target_y, work_z]),
                          n_substeps=_row_step_substeps(6))
            if p is None:
                print(f"[access chain {region_name}] final approach failed")
                return False
            _execute(p, executor.place_step_size)

            # 4. Bottom action: grasp (close+attach) or place (detach+open).
            bottom_fn(obj_name)

            # 5. Lift within the z-window.
            retreat_z = float(min(work_z + 0.04, z_hi))
            p = _try_lerp(np.array([col_x, target_y, retreat_z]),
                          n_substeps=_row_step_substeps(6))
            if p is not None:
                _execute(p, executor.retreat_step_size)

            # 6. Reverse row-step back to the front row at retreat z.
            cur_y = target_y
            while cur_y > front_row_y + 1e-6:
                step_y = max(cur_y - step, front_row_y)
                p = _try_lerp(np.array([col_x, step_y, retreat_z]),
                              n_substeps=_row_step_substeps(8))
                if p is None:
                    return True  # action done; partial retreat acceptable
                _execute(p, executor.retreat_step_size)
                cur_y = step_y

            # 7. Exit to the column-approach anchor outside the front face.
            p = _try_lerp(np.array([col_x, front_face_y - 0.06, retreat_z]),
                          n_substeps=8)
            if p is not None:
                _execute(p, executor.retreat_step_size)
            return True
        finally:
            env.controller._advance_delta_override = 0.1

    # --- public closures ------------------------------------------------

    def pick_fn(obj_name: str, cell_id: str, source_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
        z_lo, z_hi = z_windows[cell.region]
        col_x = float(source_pos[0])
        target_y = float(source_pos[1]) - GRASP_CONTACT_OFFSET
        work_z = float(min(max(float(source_pos[2]), z_lo), z_hi))  # centre grasp
        env.add_collision_exception(obj_name)
        try:
            return _front_chain(obj_name, cell.region, col_x, target_y,
                                work_z, _close_and_attach)
        finally:
            env.remove_collision_exception(obj_name)

    def put_fn(obj_name: str, cell_id: str, target_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
        region = regions[cell.region]
        z_lo, z_hi = z_windows[cell.region]
        col_x = float(target_pos[0])
        target_y = float(target_pos[1]) - GRASP_CONTACT_OFFSET
        # Place the object's CENTRE so its bottom rests on the deck
        # surface, hovering _PLACE_CLEARANCE above it (physics settles
        # it the last mm; starting in contact would trip the collision
        # check).  The surface is the region's ``level_z`` minus the
        # item-half reference baked into it — using only target_pos's
        # xy makes this robust to the caller's z convention.
        surface_z = region.level_z - _ACCESS_ITEM_HALF_Z_REF
        half_z = float(env.get_object_half_size(obj_name)[2])
        target_center_z = surface_z + half_z
        # The held object sits ``held_dz`` (in z) from the EE — captured
        # at grasp, or the canonical grasp-high offset when a holding
        # state was restored without a pick.  Solve for the EE z that
        # lands the object centre at the target (hovering _PLACE_CLEARANCE
        # so physics settles it the last mm).
        held_dz = getattr(executor, "_held_grasp_dz", None)
        if held_dz is None:
            held_dz = -(half_z - _CUBE_GRASP_OFFSET)
        work_z = float(min(max(target_center_z - held_dz + _PLACE_CLEARANCE,
                               z_lo), z_hi))
        return _front_chain(obj_name, cell.region, col_x, target_y,
                            work_z, _detach_and_open)

    return pick_fn, put_fn


def make_access_pick_fn(env, executor, workspace: Workspace,
                        lik: Optional[LinearIKPlanner] = None) -> PickFn:
    """Front-approach pick_fn for the 3-tier access shelf (wrapper around
    :func:`make_access_chains`)."""
    return make_access_chains(env, executor, workspace, lik)[0]


def make_access_put_fn(env, executor, workspace: Workspace,
                       lik: Optional[LinearIKPlanner] = None) -> PutFn:
    """Front-approach put_fn for the 3-tier access shelf (wrapper around
    :func:`make_access_chains`)."""
    return make_access_chains(env, executor, workspace, lik)[1]
