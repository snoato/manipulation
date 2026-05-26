"""Per-cell executability specs for the tabletop_access domain (both
``access`` and ``access-19`` variants).

See :mod:`tampanda.symbolic.domains._reachability` for the contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from tampanda.symbolic.domains._reachability import (
    DomainSetup, ReachabilitySpec,
)
from tampanda.symbolic.domains.tabletop_access import (
    access_default_goal,
    access_default_initial_layout,
    apply_runtime_tweaks,
    make_access_builder,
    make_access19_builder,
    make_tabletop_access_bridge,
    set_objects_at_cells,
)
from tampanda.symbolic.workspace import Cell


def _build_executor(env, table_z, allowed_types, rrt_iters=3000):
    from tampanda.planners.rrt_star import RRTStar
    from tampanda.planners.grasp_planner import GraspPlanner
    from tampanda.planners.pick_place import PickPlaceExecutor
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3
    rrt = RRTStar(env, max_iterations=rrt_iters)
    grasp = GraspPlanner(table_z=table_z, allowed_types=allowed_types)
    return PickPlaceExecutor(env, rrt, grasp, use_attachment=True,
                             max_plan_iters=rrt_iters)


# ---------------------------------------------------------------------------
# access-19 (deck-style closed cubicle)
# ---------------------------------------------------------------------------

# FRONT-grasp quaternions and 180°-rotated counterpart.  The Franka
# parallel-jaw gripper is invariant under this rotation (the same
# pose with fingers swapped) — but it lies in a different IK basin,
# so trying both unlocks cells where the primary basin is in a
# wrist singularity.
_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])


def _build_access19_pick_fn(env, ws, lik, object_half, set_arm_home_q,
                              cube_half_z: float, front_face_y: float):
    """Returns a ``pick_fn(name, pos, half, quat) -> bool`` that uses
    column-aligned approach + row-by-row-style descent + gripper-
    invariance, validated by
    ``examples/measure_access19_arm_extent.py`` to give 21/21
    reachable cells.

    Strategy per cell:

    1. Identify the cube's column (its x-position determines this).
    2. Build a column-aligned approach pose just outside the front
       face, at slightly elevated z.
    3. Joint-lerp from staging-home to the column approach (try both
       gripper-invariant quats).
    4. Joint-lerp from column approach to the cube's grasp pose
       (likewise try both quats).
    5. Lift 4 cm.

    Each phase tolerates either gripper rotation as success.
    """
    region = ws["shelf_interior"]

    def _try_lerp(target_pos, n_substeps=12):
        for q in (_FRONT_QUAT, _FRONT_QUAT_FLIPPED):
            path = lik.plan_joint_lerp(target_pos, q, dt=0.005,
                                         n_substeps=n_substeps)
            if path is not None:
                return path, q
        return None, None

    def _pick(name, pos, half, quat):
        # Reset to shelf-home seed.
        set_arm_home_q()
        import mujoco as _m

        # Column x is the cube's world x.  Approach pose is in
        # front of the cubicle face, aligned with this column.
        # Grasp HIGH on the cube — the link7 capsule dips ~5.5 cm
        # below the EE on palm-+y, so grasping at cube-centre puts
        # the wrist below the cubicle floor.  Grasping near cube-top
        # (cube_pos.z + cube_half_z - 1 cm) keeps the wrist clear.
        col_x = float(pos[0])
        target_grasp_z = float(pos[2]) + cube_half_z - 0.010
        approach_pos = np.array([col_x, front_face_y - 0.06,
                                  target_grasp_z + 0.02])

        # Phase 1: staging → column approach
        path1, _ = _try_lerp(approach_pos, n_substeps=20)
        if path1 is None:
            return False
        env.data.qpos[:7] = path1[-1]
        env.data.qvel[:] = 0.0
        _m.mj_forward(env.model, env.data)

        # Phase 2: row-by-row in-cubicle descent.  A direct lerp
        # from approach to a deep cell can dip the wrist below the
        # cubicle floor (link7 hangs ~5.5 cm below the EE on
        # palm-+y); stepping in cell-size increments keeps the
        # wrist trajectory inside the cubicle interior.
        env.add_collision_exception(name)
        try:
            target_y = float(pos[1])
            # Estimate which row the target is at, then walk from
            # the front row to it.  Use small y-steps (0.06 m).
            # The Y of the front row is region_origin_y + 0.5*cell.
            front_row_y = region.origin[1] + 0.5 * region.cell_size
            cur_y = front_row_y
            step = region.cell_size
            while cur_y < target_y - 1e-6:
                step_y = min(cur_y + step, target_y)
                step_pos = np.array([col_x, step_y, target_grasp_z])
                p, _ = _try_lerp(step_pos, n_substeps=8)
                if p is None:
                    return False
                env.data.qpos[:7] = p[-1]
                _m.mj_forward(env.model, env.data)
                cur_y = step_y

            # Final descent to the cube's exact grasp pose.
            grasp_pos = np.array([col_x, target_y, target_grasp_z])
            path_final, _ = _try_lerp(grasp_pos, n_substeps=6)
            if path_final is None:
                return False
            env.data.qpos[:7] = path_final[-1]
            _m.mj_forward(env.model, env.data)

            # Phase 3: lift 4 cm
            lift_pos = grasp_pos + np.array([0.0, 0.0, 0.04])
            path_lift, _ = _try_lerp(lift_pos, n_substeps=6)
            if path_lift is None:
                return False
            return True
        finally:
            env.clear_collision_exceptions()

    return _pick


def make_setup_access19(scratch_dir: Path, motion: bool = True) -> DomainSetup:
    from tampanda.planners.grasp_planner import GraspType
    from tampanda.planners.linear_ik import LinearIKPlanner

    builder, ws, cfg = make_access19_builder(scratch_dir=scratch_dir)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    object_ids = [f"blocker_{i}" for i in range(18)] + ["ooi"]
    layout: Dict[str, Cell] = {}
    col_ix = [1, 3, 5]
    for ci, ix in enumerate(col_ix):
        for iy in range(6):
            layout[f"blocker_{ci * 6 + iy}"] = Cell("shelf_interior", ix, iy)
    layout["ooi"] = Cell("shelf_interior", 3, 6)
    set_objects_at_cells(env, ws, cfg, layout, object_ids)

    cube_half = float(env.get_object_half_size("ooi")[2])
    table_z = ws["shelf_interior"].level_z - cube_half
    executor = (_build_executor(env, table_z=table_z,
                                allowed_types=[GraspType.FRONT])
                if motion else None)

    bridge, objects = make_tabletop_access_bridge(
        env, ws, cfg, object_ids, mode="filter", executor=executor,
    )
    goal = [("occupied",
             Cell("shelf_top",
                  ws["shelf_top"].cells_x // 2,
                  ws["shelf_top"].cells_y // 2).id,
             "ooi")]

    def _half(name):
        return np.asarray(env.get_object_half_size(name))

    def _place(env_, ws_, name, cell_id):
        cell = ws_.cell(cell_id)
        x, y, z = ws_.pose_for(cell)
        env_.set_object_pose(name, np.array([x, y, z]))

    # Shelf-home: palm-+y staging IK-solved at the centre column of
    # the cubicle, just outside the front face.  Locked-in basin
    # for the column-aligned approach.
    shelf_home = _solve_access19_staging(env, ws, cfg)

    # Custom pick function — column-aligned approach + row-style
    # descent + gripper invariance.
    region = ws["shelf_interior"]
    interior_y = region.cells_y * region.cell_size + 0.06  # + margins
    front_face_y = cfg.shelf_pos[1] - interior_y / 2

    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)

    def _set_home():
        env.data.qpos[: len(shelf_home)] = shelf_home
        env.data.qvel[:] = 0.0
        import mujoco as _m
        _m.mj_forward(env.model, env.data)

    pick_fn = _build_access19_pick_fn(
        env, ws, lik, _half, _set_home,
        cube_half_z=cube_half, front_face_y=front_face_y,
    ) if motion else None

    return DomainSetup(
        name="tabletop_access:access-19",
        env=env, workspace=ws, object_ids=object_ids,
        initial_layout=layout, goal=goal,
        executor=executor,
        place_at_cell=_place,
        object_half_extents=_half,
        parked_xyz=(cfg.hide_far_x, 0.0, 0.05),
        home_qpos=shelf_home,
        pick_fn=pick_fn,
    )


def _solve_access19_staging(env, ws, cfg) -> np.ndarray:
    """Solve once for a palm-+y staging pose just outside the
    cubicle's front face, aligned with shelf centre.  Used as the
    home seed for every cell test."""
    import mujoco as _m
    region = ws["shelf_interior"]
    interior_y = region.cells_y * region.cell_size + 0.06
    front_face_y = cfg.shelf_pos[1] - interior_y / 2
    # Park objects far away first so they don't interfere with IK
    # collision check.
    parked = np.array([cfg.hide_far_x, 0.0, 0.05])
    saved_poses = {}
    object_ids = [f"blocker_{i}" for i in range(18)] + ["ooi"]
    for n in object_ids:
        saved_poses[n] = env.get_object_pose(n)
        env.set_object_pose(n, parked)

    target_pos = np.array([cfg.shelf_pos[0],
                            front_face_y - 0.06,
                            region.level_z + 0.05])
    target_quat = _FRONT_QUAT
    seed = np.array([np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    env.data.qpos[:7] = seed
    _m.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(target_pos, target_quat)
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3
    env.ik.converge_ik(0.005)
    q7 = env.ik.configuration.q[:7].copy()

    # Restore objects to wherever they were.
    for n, (pos, quat) in saved_poses.items():
        env.set_object_pose(n, np.asarray(pos), np.asarray(quat))
    env.reset_velocities()
    env.forward()

    return np.concatenate([q7, [0.04, 0.04]])


def reachability_spec_access19() -> ReachabilitySpec:
    # Layout cells (19) + the goal cell (top-deck centre).
    extra = (Cell("shelf_top", 3, 3),)
    return ReachabilitySpec(
        domain_name="tabletop_access:access-19",
        full_regions=("shelf_interior", "shelf_top"),
        layout_proxy="ooi",
        full_proxy="ooi",
        extra_goal_cells=extra,
    )


# ---------------------------------------------------------------------------
# access (free-standing 3-tier shelf, YCB-proxy items)
# ---------------------------------------------------------------------------

def _build_access_put_fn(env, ws, lik, object_half_fn, set_arm_home_q,
                            shelf_front_y: float, executor,
                            real_execution: bool = True):
    """Mirror of ``_build_access_pick_fn`` for puts.

    Reuses the column-align + row-step chain shape; at the placement
    pose, detaches + opens the gripper instead of closing + attaching.
    Then row-steps back out of the compartment.

    Uses the executor's stashed ``_last_grasp_quat`` so the put runs
    in the SAME IK basin as the prior pick — no basin search needed.
    Returns ``put_fn(obj_name, cell_id, target_pos)``.
    """
    import mujoco as _m
    region_names = ("floor_left", "floor_right", "middle_deck", "top_deck")
    regions = {n: ws[n] for n in region_names}
    LINK7_SAFETY = 0.060
    HAND_TOP_SAFETY = 0.080
    sorted_levels = sorted({r.level_z for r in regions.values()})

    def _ceil_for(level_z):
        for nxt in sorted_levels:
            if nxt > level_z + 1e-6:
                return nxt - HAND_TOP_SAFETY
        return float("inf")

    z_windows = {n: (r.level_z + LINK7_SAFETY, _ceil_for(r.level_z))
                  for n, r in regions.items()}

    def _advance(path, step_size):
        if not real_execution:
            env.data.qpos[:7] = path[-1]
            _m.mj_forward(env.model, env.data)
        else:
            env.execute_path(path, executor.planner, step_size=step_size)
            env.wait_idle(settle_steps=executor.settle_steps)

    def _detach_open():
        if not real_execution:
            return
        if executor.use_attachment and env._attached is not None:
            env.detach_object()
        env.controller.open_gripper()
        executor._wait_gripper_open()

    def _plan(target_pos, quat, n_substeps):
        """Joint-lerp first, Cartesian-substep IK fallback.

        Tries both FRONT-quat basins (the grasp's and its 180°-around-
        hand-z twin) — parallel-jaw symmetry, one basin's IK often
        converges where the other doesn't.

        Temporarily relaxes ``env.ik.pos_threshold`` to 5 mm during
        the chain — env_inspector defaults to 2 mm for free-space
        precision, but for shelf-approach poses near the edge of
        reach, the IK iterations don't quite converge to 2 mm even
        when a valid basin exists at 5 mm.  The executability test
        uses the 5 mm threshold and finds all cells reachable; we
        match it here so the chain doesn't degrade in real-sim.
        """
        saved_pos = env.ik.pos_threshold
        saved_ori = env.ik.ori_threshold
        env.ik.pos_threshold = 0.005
        env.ik.ori_threshold = 5e-3
        try:
            quats = [np.asarray(quat, dtype=float)]
            hand_z_180 = np.array([0.0, 0.0, 0.0, 1.0])
            flipped = np.zeros(4)
            mujoco.mju_mulQuat(flipped, quats[0], hand_z_180)
            quats.append(flipped)
            for q in quats:
                p = lik.plan_joint_lerp(target_pos, q, dt=0.005,
                                          n_substeps=n_substeps)
                if p is not None:
                    return p
            for q in quats:
                p = lik.plan_to_pose(target_pos, q, dt=0.005,
                                        n_substeps=max(n_substeps, 12),
                                        slerp_orientation=False)
                if p is not None:
                    return p
            return None
        finally:
            env.ik.pos_threshold = saved_pos
            env.ik.ori_threshold = saved_ori

    def put_fn(obj_name: str, cell_id: str,
                 target_pos: np.ndarray) -> bool:
        cell = Cell.parse(cell_id)
        region_name = cell.region
        if region_name not in regions:
            return False
        region = regions[region_name]
        quat = getattr(executor, "_last_grasp_quat", None)
        if quat is None:
            return False  # no prior pick — nothing to place
        quat = np.asarray(quat, dtype=float)

        cube_half_z = float(object_half_fn(obj_name)[2])
        # +0.005 release clearance so the block lands ABOVE the
        # surface and physics settles it down — without this the
        # block geom touches the surface during descent and the
        # collision check rejects the plan.
        place_clearance = 0.005
        col_x = float(target_pos[0])
        target_y = float(target_pos[1])
        z_lo, z_hi = z_windows[region_name]
        place_z = max(z_lo, min(z_hi, float(target_pos[2]) + cube_half_z
                                          - 0.010 + place_clearance))

        # 1. Approach (outside front face, z just above placement).
        # Pick chain returns the arm to the converged staging pose
        # (see ``_return_to_home`` in the pick builder), so we start
        # from a known basin.  Some target regions still can't be
        # approached from this basin with a held block — see the
        # methodology doc's "access cross-region put basin coupling"
        # note for the limitation.
        approach_z = max(z_lo, min(z_hi, place_z + 0.05))
        approach = np.array([col_x, shelf_front_y - 0.06, approach_z])
        p1 = _plan(approach, quat, n_substeps=20)
        if p1 is None:
            print("[access put] approach plan failed")
            return False
        _advance(p1, executor.approach_step_size)

        env.add_collision_exception(obj_name)
        # Tighten controller tolerance for the in-shelf phases so the
        # held block actually reaches the placement pose before the
        # gripper opens (default 0.1 rad delta lets the controller
        # call waypoints reached with cm-scale joint slop; the block
        # then gets released several cm short of the target cell).
        if real_execution:
            env.controller._advance_delta_override = 0.01
        try:
            # 2. Row-step descent into the compartment.
            front_row_y = region.origin[1] + 0.5 * region.cell_size
            cur_y = front_row_y
            step = region.cell_size
            while cur_y < target_y - 1e-6:
                step_y = min(cur_y + step, target_y)
                p = _plan(np.array([col_x, step_y, place_z]),
                            quat, n_substeps=8)
                if p is None:
                    print(f"[access put] row-step at y={step_y:.3f} failed")
                    return False
                _advance(p, executor.place_step_size)
                cur_y = step_y

            # 3. Final descent to the placement pose.
            place_pose = np.array([col_x, target_y, place_z])
            p2 = _plan(place_pose, quat, n_substeps=6)
            if p2 is None:
                print("[access put] final descent plan failed")
                return False
            _advance(p2, executor.place_step_size)

            # 4. Detach + open gripper.
            _detach_open()

            # 5. Retreat: reverse row-step at the same z, then exit.
            cur_y = target_y
            while cur_y > front_row_y + 1e-6:
                step_y = max(cur_y - step, front_row_y)
                p = _plan(np.array([col_x, step_y, place_z]),
                            quat, n_substeps=8)
                if p is None:
                    return True  # block placed; partial retreat OK
                _advance(p, executor.retreat_step_size)
                cur_y = step_y
            p = _plan(approach, quat, n_substeps=8)
            if p is not None:
                _advance(p, executor.retreat_step_size)
            return True
        finally:
            env.clear_collision_exceptions()
            if real_execution:
                env.controller._advance_delta_override = 0.1

    return put_fn


def _build_access_pick_fn(env, ws, lik, object_half_fn, set_arm_home_q,
                            shelf_front_y: float, executor=None,
                            real_execution: bool = False,
                            home_qpos=None):
    """Region-aware pick_fn for the 3-tier ``access`` shelf.

    Routes by region:

    * ``top_deck`` — TOP_DOWN_Y from above (no obstruction).
    * ``floor_left`` / ``floor_right`` / ``middle_deck`` — FRONT
      approach, palm-+y, with the **grasp-high** trick (clamp near
      the cube's top so link7 clears the deck below) and gripper-
      rotation invariance.

    Returns ``(name, pos, half, quat) -> bool``.
    """
    import mujoco as _m
    regions = {n: ws[n] for n in
                ("floor_left", "floor_right", "middle_deck", "top_deck")}

    # Per-region z-window for the *front-grasp approach pose* — the
    # gripper has to enter the open front face between two horizontal
    # decks (or above the top deck).  Compute each region's window
    # from the workspace itself: the floor of the window is
    # ``level_z + link7_safety`` (link7 dips ~5.5 cm below EE on
    # palm-+y), the ceiling is the next deck above minus a hand-top
    # safety margin (top deck has no ceiling).
    LINK7_SAFETY = 0.060
    # 8 cm clearance between the EE and the deck *above*.  With
    # palm-+y and the IK basin chosen by row-by-row joint-lerp from
    # the approach pose, the link7 capsule extends ~5 cm ABOVE the
    # EE and the hand_capsule another ~4 cm — without this margin
    # the wrist clips the deck above when grasping deeper rows.
    HAND_TOP_SAFETY = 0.080
    # ``level_z`` in the access workspace is ``surface_z +
    # _ITEM_HALF_Z`` (the resting Z of an item with the reference
    # half-height).  For z-window computations we need the actual
    # SURFACE z (the deck top, where link7 would clip).  Recover it
    # by subtracting the access-builder's reference half.
    _ITEM_HALF_Z_REF = 0.05
    surface_z_of = {n: r.level_z - _ITEM_HALF_Z_REF
                       for n, r in regions.items()}
    sorted_surfaces = sorted({s for s in surface_z_of.values()})
    def _ceil_for(surface_z):
        for nxt in sorted_surfaces:
            if nxt > surface_z + 1e-6:
                return nxt - HAND_TOP_SAFETY
        return float("inf")  # top deck
    z_windows = {n: (surface_z_of[n] + LINK7_SAFETY,
                       _ceil_for(surface_z_of[n]))
                  for n in regions}

    def _which_region(pos: np.ndarray) -> str:
        """Pick the region by both x-membership AND closest level_z.
        Floor compartments share the same level_z and disambiguate
        only by x; if pos.x is outside every region, fall back to the
        closest level_z across all regions."""
        z_target = float(pos[2])
        x_target = float(pos[0])
        candidates = []
        for n, r in regions.items():
            x0, y0 = r.origin
            ex, _ = r.extent
            if x0 - 0.02 <= x_target <= x0 + ex + 0.02:
                candidates.append((n, r))
        pool = candidates if candidates else list(regions.items())
        return min(pool, key=lambda nr: abs(z_target - nr[1].level_z))[0]

    def _try_lerp(target_pos, quats, n_substeps=12):
        """Try joint-lerp (basin-stable) first, fall back to Cartesian-
        substep IK (preserves Cartesian path) — same fallback pattern
        as the access19 chains.  Returns ``(path, used_quat)`` so
        downstream chain steps can commit to the same basin.
        """
        for q in quats:
            path = lik.plan_joint_lerp(target_pos, q, dt=0.005,
                                         n_substeps=n_substeps)
            if path is not None:
                return path, q
        for q in quats:
            path = lik.plan_to_pose(target_pos, q, dt=0.005,
                                       n_substeps=max(n_substeps, 12),
                                       slerp_orientation=False)
            if path is not None:
                return path, q
        return None, None

    # _advance_path: switch between executability-test teleport
    # (``real_execution=False``) and bridge-driven real execution
    # (``real_execution=True``, executor required).  Teleport keeps the
    # executability run fast; real execution drives the controller so
    # the bridge's ``exec_pick`` actually moves the simulator.
    def _advance_path(path, step_size):
        if not real_execution:
            env.data.qpos[:7] = path[-1]
            _m.mj_forward(env.model, env.data)
        else:
            env.execute_path(path, executor.planner, step_size=step_size)
            env.wait_idle(settle_steps=executor.settle_steps)

    def _set_advance_delta(delta):
        """Tighten/loosen the controller's IDLE-check tolerance.
        Default 0.1 rad lets the controller call waypoints reached
        with ~5 cm of joint slop — fine for free-space approach but
        too loose for the final grasp descent (gripper closes before
        the EE has actually reached the cube).  Set 0.01 during
        precision phases."""
        if real_execution:
            env.controller._advance_delta_override = float(delta)

    def _return_to_home() -> bool:
        """Plan + execute a joint-space lerp from the current arm
        config to the converged staging pose.  Called at the end of
        every pick (real-execution only) so every put starts from
        a known-good IK basin — without this, the post-pick basin
        depends on which region the pick happened from, and the
        put_fn's "retract first" lerp fails for some target regions.

        The held block rides along via ``_collision_held_body`` (set
        by ``attach_object_to_ee``); the collision check at every
        substep catches any held-block-vs-shelf interference.

        Returns True on success.  On failure (collision or no
        home_qpos provided) returns True silently so the pick still
        counts as successful — the put_fn just inherits an awkward
        basin like before.
        """
        if home_qpos is None or not real_execution:
            return True
        cur_q = env.data.qpos[:7].copy()
        target_q = np.asarray(home_qpos, dtype=float)[:7]
        n_steps = 24
        path = []
        for k in range(n_steps + 1):
            alpha = k / n_steps
            q_k = (1 - alpha) * cur_q + alpha * target_q
            if not env.is_collision_free(q_k):
                # Joint lerp not collision-free — skip the return,
                # leave the arm where it is.  The put_fn can still
                # try; it might or might not succeed.
                return True
            path.append(q_k)
        env.execute_path(path, executor.planner,
                            step_size=executor.approach_step_size)
        env.wait_idle(settle_steps=executor.settle_steps)
        return True

    def _close_and_attach(obj_name, quat=None):
        if not real_execution:
            return  # teleport mode — no gripper actuation
        env.controller.close_gripper()
        executor._wait_gripper_closed()
        if executor.use_attachment:
            env.attach_object_to_ee(obj_name)
        # Stash the grasp quat on the executor so the bridge's
        # subsequent ``executor.place`` (or chain-based put) has the
        # orientation to release the block in.
        if quat is not None:
            executor._last_grasp_quat = np.asarray(quat, dtype=float)

    def _pick(name, pos, half, quat):
        set_arm_home_q()
        region_name = _which_region(pos)
        cube_half_z = float(half[2])

        if region_name == "top_deck":
            # Top deck — z is at the upper edge of the Franka's
            # reach.  Palm-down (TOP_DOWN_Y) is physically not
            # achievable at the deck's centre column, so use a FRONT
            # grasp from the open front face.  Staging is the palm-
            # +y home, so just lerp to the deck level.
            col_x = float(pos[0])
            # Centre-grasp (see comment in the front/floor branch
            # below): grasp at block COM so the lift doesn't tilt
            # tall items.  Top deck has no plate above, so the only
            # constraint is the LINK7_SAFETY floor on the top deck
            # surface.
            target_grasp_z = max(float(pos[2]),
                                  surface_z_of[region_name] + LINK7_SAFETY)
            front_face_y = shelf_front_y
            approach_pos = np.array([col_x, front_face_y - 0.06,
                                       target_grasp_z + 0.02])
            path1, used_q = _try_lerp(approach_pos, _FRONT_QUATS_LIST,
                                        n_substeps=20)
            if path1 is None:
                return False
            _advance_path(path1,
                          executor.approach_step_size
                            if executor else 0.01)
            chain_q = [used_q]
            env.add_collision_exception(name)
            try:
                grasp = np.array([col_x, float(pos[1]), target_grasp_z])
                p2, _ = _try_lerp(grasp, chain_q, n_substeps=14)
                if p2 is None:
                    return False
                # Tighten controller tolerance before the final
                # descent so the EE actually reaches the grasp pose
                # before the gripper closes.
                _set_advance_delta(0.01)
                _advance_path(p2,
                              executor.grasp_step_size
                                if executor else 0.003)
                _close_and_attach(name, quat=chain_q[0])
                lift = grasp + np.array([0.0, 0.0, 0.04])
                p3, _ = _try_lerp(lift, chain_q, n_substeps=6)
                if p3 is None:
                    return False
                _advance_path(p3,
                              executor.lift_step_size
                                if executor else 0.003)
                _set_advance_delta(0.1)  # restore loose tolerance
                _return_to_home()
                return True
            finally:
                env.clear_collision_exceptions()
                _set_advance_delta(0.1)

        # FRONT-style approach for floor / middle deck cells.  Gripper
        # enters between deck levels — palm-+y, grasp at the block
        # CENTRE (its centre of mass) where possible so the held
        # block doesn't tilt during the lift.  For items short enough
        # that the COM falls below the link7-clearance floor, clamp
        # UP to that floor (the grip ends up near the cube top, but
        # link7 doesn't dip into the deck).  Capped by the ceiling
        # clearance so the wrist doesn't punch the deck above.
        #
        # Earlier formula ``pos[2] + cube_half_z - 0.010`` grasped at
        # ``cube_top - 0.010``: fine for the 8 cm uniform cubes in
        # access19 but for tall YCB items (cracker_box, 12 cm)
        # the grip is 5 cm above the COM and the block tilts on
        # lift.  Centre-grasping fixes that.
        col_x = float(pos[0])
        region = regions[region_name]
        front_face_y = shelf_front_y
        z_lo, z_hi = z_windows[region_name]
        target_grasp_z = max(float(pos[2]), z_lo)
        target_grasp_z = min(target_grasp_z, z_hi)

        # Approach pose is column-aligned, just outside the open
        # front face, with z chosen inside the region's access
        # window.  Try a few candidate z's (preferring ``+0.05``
        # above the grasp where the window allows it) so a single
        # IK basin can be found across cells.
        path1 = None
        approach_zs = []
        for dz in (0.05, 0.02, -0.02, 0.10):
            z = max(z_lo, min(z_hi, target_grasp_z + dz))
            if z not in approach_zs:
                approach_zs.append(z)
        # Try each IK basin as an INDEPENDENT chain.  Mixing basins
        # between approach and later steps lets each individual IK
        # succeed but the joint-lerp between them traverses a singular
        # region and fails the segment-collision check.  Floor_left
        # cells fail with basin 1 at the approach (basin 0 IK doesn't
        # converge from staging) but require basin 0 at the grasp
        # (basin 1 IK doesn't converge from the basin-1-approach
        # config).  Trying basin 0 end-to-end may succeed at the
        # approach with a different approach-z choice that basin 1
        # couldn't reach.
        target_y = float(pos[1])
        front_row_y = region.origin[1] + 0.5 * region.cell_size

        def _try_chain_with_basin(chain_q_only):
            # Re-seed back to the converged home before each attempt.
            # In real-execution mode this physically moves the arm back
            # to staging; in teleport mode it just sets qpos.
            if executor is None:
                set_arm_home_q()
            else:
                # Execute a plan back to staging if not already there.
                home_q = set_arm_home_q.__defaults__  # not always available
                # Simpler: lerp to staging via plan_joint_lerp from current
                # qpos.  set_arm_home_q in the executor-mode caller is
                # expected to teleport to the home pose; but to physically
                # return we need an explicit trajectory.  For now we just
                # set qpos (the bridge calls pick_fn from a known home
                # state, so this is OK for the first attempt; a second
                # attempt with different basin would need a real retreat).
                set_arm_home_q()
            path1 = None
            for z in approach_zs:
                approach_pos = np.array([col_x, front_face_y - 0.06, z])
                p, _ = _try_lerp(approach_pos, chain_q_only, n_substeps=20)
                if p is not None:
                    path1 = p
                    break
            if path1 is None:
                return False
            _advance_path(path1,
                          executor.approach_step_size
                            if executor else 0.01)

            env.add_collision_exception(name)
            try:
                # Tighten controller tolerance for the in-shelf
                # row-step + grasp + lift sequence.  Loose
                # tolerance lets the controller call waypoints
                # reached with ~5 cm of joint slop; the gripper
                # then closes before the EE actually reaches the
                # cube and the chain "succeeds" with the block
                # still untouched.
                _set_advance_delta(0.01)
                cur_y = front_row_y
                step = region.cell_size
                while cur_y < target_y - 1e-6:
                    step_y = min(cur_y + step, target_y)
                    step_pos = np.array([col_x, step_y, target_grasp_z])
                    p, _ = _try_lerp(step_pos, chain_q_only, n_substeps=8)
                    if p is None:
                        return False
                    _advance_path(p,
                                  executor.grasp_step_size
                                    if executor else 0.003)
                    cur_y = step_y

                grasp_pos = np.array([col_x, target_y, target_grasp_z])
                path2, _ = _try_lerp(grasp_pos, chain_q_only, n_substeps=6)
                if path2 is None:
                    return False
                _advance_path(path2,
                              executor.grasp_step_size
                                if executor else 0.003)
                _close_and_attach(name, quat=chain_q_only[0])
                lift_pos = grasp_pos + np.array([0.0, -0.04, 0.0])
                path3, _ = _try_lerp(lift_pos, chain_q_only, n_substeps=6)
                if path3 is None:
                    return False
                _advance_path(path3,
                              executor.lift_step_size
                                if executor else 0.003)
                _set_advance_delta(0.1)
                _return_to_home()
                return True
            finally:
                env.clear_collision_exceptions()
                _set_advance_delta(0.1)

        for q in _FRONT_QUATS_LIST:
            if _try_chain_with_basin([q]):
                return True
        return False

    return _pick


_FRONT_QUATS_LIST = [_FRONT_QUAT, _FRONT_QUAT_FLIPPED]


def _solve_access_staging(env, ws, cfg, object_ids) -> np.ndarray:
    """IK-solve a palm-+y staging pose just outside the open front of
    the 3-tier shelf, centred in x.  Returns a 9-vector qpos (7 arm +
    2 fingers) used as the home seed for every cell test.
    """
    import mujoco as _m
    # Front face of the open cubicle: floor_left region origin is at
    # the inside-of-front-leg y; pull back another 5 cm so we're truly
    # in front of the body.
    front_face_y = ws["floor_left"].origin[1] - 0.05

    # Park objects far away while we IK so they don't interfere.
    parked = np.array([cfg.hide_far_x, 0.0, 0.10])
    saved_poses = {}
    for n in object_ids:
        saved_poses[n] = env.get_object_pose(n)
        env.set_object_pose(n, parked)

    # Staging z sits at the MIDDLE of the shelf body — equidistant from
    # the floor compartments and the top deck — so the joint-space
    # lerp to either region traverses a small angular change and stays
    # in a stable IK basin.  Was previously ``shelf_pos.z + 0.10``
    # which assumed the legacy ``base_height=0.45`` pedestal; with
    # ``base_height=0`` the +0.10 staging at z=0.10 is too close to
    # the world floor and lerps to the middle deck (approach z ≈ 0.45)
    # fail.
    mid_deck_world_z = ws["middle_deck"].level_z
    target_pos = np.array([cfg.shelf_pos[0],
                            front_face_y - 0.08,
                            mid_deck_world_z])
    target_quat = _FRONT_QUAT
    seed = np.array([np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    env.data.qpos[:7] = seed
    _m.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(target_pos, target_quat)
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3
    env.ik.converge_ik(0.005)
    q7 = env.ik.configuration.q[:7].copy()

    # Restore objects.
    for n, (pos, quat) in saved_poses.items():
        env.set_object_pose(n, np.asarray(pos), np.asarray(quat))
    env.reset_velocities()
    env.forward()
    return np.concatenate([q7, [0.04, 0.04]])


def make_setup_access(scratch_dir: Path, motion: bool = True) -> DomainSetup:
    from tampanda.planners.grasp_planner import GraspType
    from tampanda.planners.linear_ik import LinearIKPlanner

    builder, ws, cfg = make_access_builder(scratch_dir=scratch_dir)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    ycb_items = ["meat_can", "tomato_soup_can", "tuna_can",
                 "gelatin_box", "pudding_box"]
    object_ids = ["ooi" if x == "meat_can" else x for x in ycb_items]
    layout = access_default_initial_layout(ws, object_ids, target_id="ooi")

    def _half(name):
        return np.asarray(env.get_object_half_size(name))

    def _place(env_, ws_, name, cell_id):
        # Place body so its BOTTOM rests on the deck surface.
        cell = ws_.cell(cell_id)
        cx, cy, _level = ws_.pose_for(cell)
        # ``level_z`` is the deck surface; place the body half-z above it.
        own_half_z = float(_half(name)[2])
        deck_z = ws_.region_of(cell).level_z
        env_.set_object_pose(name, np.array([cx, cy, deck_z + own_half_z]))

    # Re-place objects so they sit on top of decks (not centred at level_z).
    for n, cell in layout.items():
        _place(env, ws, n, cell.id)
    env.reset_velocities()
    env.forward()

    table_z = ws["floor_left"].level_z
    executor = (_build_executor(env, table_z=table_z,
                                allowed_types=[GraspType.FRONT,
                                               GraspType.TOP_DOWN_X,
                                               GraspType.TOP_DOWN_Y])
                if motion else None)

    bridge, objects = make_tabletop_access_bridge(
        env, ws, cfg, object_ids, mode="filter", executor=executor,
    )
    goal = access_default_goal(ws, target_id="ooi")

    # Solve once for a palm-+y staging pose at the front of the shelf;
    # this is a stable home seed in the right IK basin.
    shelf_home = _solve_access_staging(env, ws, cfg, object_ids)

    def _set_home():
        env.data.qpos[: len(shelf_home)] = shelf_home
        env.data.qvel[:] = 0.0
        import mujoco as _m
        _m.mj_forward(env.model, env.data)

    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
    # Front face of the shelf body (open -y face).  The workspace's
    # floor_left origin has been inset by ``hand_clearance``, so we
    # can't infer the body front from there — recover it from the
    # shelf body extent measured directly off the geometry.
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "shelf")
    if body_id < 0:
        shelf_front_y = cfg.shelf_pos[1] - 0.225  # fallback
    else:
        # Walk the shelf body's geoms; min over y of (body_y + geom_y - geom_size_y).
        ys = []
        for gid in range(env.model.ngeom):
            if env.model.geom_bodyid[gid] == body_id:
                gpy = env.model.geom_pos[gid][1]
                gsy = env.model.geom_size[gid][1]
                ys.append(cfg.shelf_pos[1] + gpy - gsy)
        shelf_front_y = min(ys) if ys else cfg.shelf_pos[1] - 0.225

    pick_fn = _build_access_pick_fn(
        env, ws, lik, _half, _set_home,
        shelf_front_y=shelf_front_y, executor=executor,
    ) if motion else None

    return DomainSetup(
        name="tabletop_access:access",
        env=env, workspace=ws, object_ids=object_ids,
        initial_layout=layout, goal=goal,
        executor=executor,
        place_at_cell=_place,
        object_half_extents=_half,
        parked_xyz=(cfg.hide_far_x, 0.0, 0.10),
        home_qpos=shelf_home,
        pick_fn=pick_fn,
    )


def reachability_spec_access() -> ReachabilitySpec:
    # Heterogeneous YCB-proxy items — layout-mode tests each body at
    # its own cell (no layout_proxy → per-body testing).  Full-mode
    # uses the OoI (small meat-can-sized) as a representative.  The
    # extra goal cell is the top-deck centre, populated dynamically
    # in check_executability since it depends on the workspace shape.
    return ReachabilitySpec(
        domain_name="tabletop_access:access",
        full_regions=("floor_left", "floor_right", "middle_deck", "top_deck"),
        full_proxy="ooi",
    )
