"""Per-cell executability spec for the access-19 domain.

Forked from ``tabletop_access.reachability``; keeps only the
access-19-specific entry points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import mujoco
import numpy as np

from tampanda.symbolic.domains._reachability import (
    DomainSetup, ReachabilitySpec,
)
from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks,
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


# FRONT-grasp quaternions and 180°-rotated counterpart.  Gripper
# invariance — trying both unlocks cells where the primary basin
# hits a wrist singularity.
_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])


def _build_access19_pick_fn(env, ws, lik, object_half, set_arm_home_q,
                              cube_half_z: float, front_face_y: float):
    """Column-aligned approach + row-by-row descent + gripper-invariance
    pick.  Returns ``pick_fn(name, pos, half, quat) -> bool``.

    Strategy per cell:
      1. Identify the cube's column (x-position).
      2. Approach just outside the front face at slightly elevated z.
      3. Joint-lerp from staging-home to column approach (both quats).
      4. Row-step inward to the target cell.
      5. Lift 4 cm.
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
        set_arm_home_q()
        import mujoco as _m

        col_x = float(pos[0])
        # Grasp near cube-top — link7 capsule dips ~5.5 cm below EE on
        # palm-+y, so grasping at cube-centre puts wrist below floor.
        target_grasp_z = float(pos[2]) + cube_half_z - 0.010
        approach_pos = np.array([col_x, front_face_y - 0.06,
                                  target_grasp_z + 0.02])

        path1, _ = _try_lerp(approach_pos, n_substeps=20)
        if path1 is None:
            return False
        env.data.qpos[:7] = path1[-1]
        env.data.qvel[:] = 0.0
        _m.mj_forward(env.model, env.data)

        env.add_collision_exception(name)
        try:
            target_y = float(pos[1])
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

            grasp_pos = np.array([col_x, target_y, target_grasp_z])
            path_final, _ = _try_lerp(grasp_pos, n_substeps=6)
            if path_final is None:
                return False
            env.data.qpos[:7] = path_final[-1]
            _m.mj_forward(env.model, env.data)

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

    shelf_home = _solve_access19_staging(env, ws, cfg)

    region = ws["shelf_interior"]
    interior_y = region.cells_y * region.cell_size + 0.06
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
        name="access19",
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
    """Solve once for a palm-+y staging pose just outside the cubicle's
    front face, aligned with shelf centre.  Home seed for every test."""
    import mujoco as _m
    region = ws["shelf_interior"]
    interior_y = region.cells_y * region.cell_size + 0.06
    front_face_y = cfg.shelf_pos[1] - interior_y / 2
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

    for n, (pos, quat) in saved_poses.items():
        env.set_object_pose(n, np.asarray(pos), np.asarray(quat))
    env.reset_velocities()
    env.forward()

    return np.concatenate([q7, [0.04, 0.04]])


def reachability_spec_access19() -> ReachabilitySpec:
    extra = (Cell("shelf_top", 3, 3),)
    return ReachabilitySpec(
        domain_name="access19",
        full_regions=("shelf_interior", "shelf_top"),
        layout_proxy="ooi",
        full_proxy="ooi",
        extra_goal_cells=extra,
    )
