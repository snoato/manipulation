"""Linear-IK FRONT-grasp chains for the dense-YCB tabletop-access fork.

Forked from ``tabletop_access.chains.make_access_chains`` (the 3-tier
``access`` shelf), with two adaptations for this domain:

* **Two regions only** — ``middle_deck`` (pick + relocate) and
  ``top_deck`` (OoI goal); the floor compartments are dropped.
* **``level_z`` is the deck SURFACE** here (the parent baked a 0.05 m
  item-half reference into ``level_z``); the surface-recovery subtraction
  is removed throughout.

Everything else is reused verbatim because it already handles what this
domain needs: object half-height is read live (heterogeneous YCB items),
the grasp is at the object CENTRE (tall items don't tilt on lift), the
held grasp z-offset is captured at pick so cross-region puts land
correctly, and FAST mode collapses the row-step lerps to 1 substep.

The chains are footprint-agnostic: they grasp / place at the world pose
the caller passes (the footprint centroid), so multi-cell objects need no
special handling here — the bridge / state layer owns occupancy.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.symbolic.workspace import Cell, Workspace

_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])
_QUATS = (_FRONT_QUAT, _FRONT_QUAT_FLIPPED)

_CONVERGE_TOL = 0.02
_MAX_RETRIES = 3
_CUBE_GRASP_OFFSET = 0.010
_PLACE_CLEARANCE = 0.003

_ACCESS_REGIONS = ("middle_deck", "top_deck")
_LINK7_SAFETY = 0.060
_HAND_TOP_SAFETY = 0.080

PickFn = Callable[[str, str, np.ndarray], bool]
PutFn = Callable[[str, str, np.ndarray], bool]


def _access_geometry(env, workspace: Workspace):
    """``(regions, front_face_y, z_windows)`` for the 2-region shelf.

    ``front_face_y`` is the shelf body's open -y face (walked off geoms,
    since region origins are inset).  ``z_windows[name] = (z_lo, z_hi)``
    is the EE band: floor ``surface + link7``; ceiling = next deck up
    minus hand-top safety (top deck is open above).
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

    surface_z = {n: r.level_z for n, r in regions.items()}   # level_z == surface
    sorted_surf = sorted(set(surface_z.values()))

    def _ceil(s: float) -> float:
        for nxt in sorted_surf:
            if nxt > s + 1e-6:
                return nxt - _HAND_TOP_SAFETY
        return float("inf")  # top_deck — open above

    z_windows = {n: (surface_z[n] + _LINK7_SAFETY, _ceil(surface_z[n]))
                 for n in regions}
    return regions, front_face_y, z_windows


def make_ycb_access_chains(env, executor, workspace: Workspace,
                           lik: Optional[LinearIKPlanner] = None):
    """Build ``(pick_fn, put_fn)`` for the dense-YCB shelf.

    Both closures take ``(obj_name, cell_id, pos)`` where ``pos`` is the
    object's footprint-centroid world pose (centre z for pick, surface z
    for put — the put recomputes its grasp z from the live half-height).
    """
    if lik is None:
        lik = executor.linear_ik_planner

    regions, front_face_y, z_windows = _access_geometry(env, workspace)
    ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                   "attachment_site")

    def _region_center_x(r) -> float:
        return r.origin[0] + 0.5 * r.cells_x * r.cell_size

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

    hand_off = {n: _solve_handoff(regions[n].level_z, _region_center_x(regions[n]))
                for n in _ACCESS_REGIONS}

    def _goto_handoff(region_name: str) -> None:
        if env.controller is not None:
            env.controller.stop()
        q = hand_off[region_name]
        env.data.qpos[: len(q)] = q
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        if getattr(env, "_attached", None) is not None:
            env._apply_attachment()

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
        obj_c = np.asarray(env.get_object_pose(obj_name)[0], dtype=float)
        ee_z = float(env.data.site_xpos[ee_site_id][2])
        executor._held_grasp_dz = float(obj_c[2] - ee_z)

    def _detach_and_open(obj_name: str) -> None:
        if executor.use_attachment and env._attached is not None:
            env.detach_object()
        env.controller.open_gripper()
        executor._wait_gripper_open()
        executor._held_grasp_dz = None

    def _front_chain(obj_name: str, region_name: str, col_x: float,
                     target_y: float, work_z: float, bottom_fn) -> bool:
        if region_name not in regions:
            raise ValueError(f"access chain: unknown region {region_name!r}")
        region = regions[region_name]
        z_lo, z_hi = z_windows[region_name]
        work_z = float(min(max(work_z, z_lo), z_hi))

        _goto_handoff(region_name)

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
            print(f"[ycb chain {region_name}] approach plan failed")
            return False
        _execute(p, executor.approach_step_size)

        env.controller._advance_delta_override = 0.01
        try:
            front_row_y = region.origin[1] + 0.5 * region.cell_size
            cur_y = front_row_y
            step = region.cell_size
            while cur_y < target_y - 1e-6:
                step_y = min(cur_y + step, target_y)
                p = _try_lerp(np.array([col_x, step_y, work_z]),
                              n_substeps=_row_step_substeps(8))
                if p is None:
                    print(f"[ycb chain {region_name}] row-step y={step_y:.3f} failed")
                    return False
                _execute(p, executor.place_step_size)
                cur_y = step_y

            p = _try_lerp(np.array([col_x, target_y, work_z]),
                          n_substeps=_row_step_substeps(6))
            if p is None:
                print(f"[ycb chain {region_name}] final approach failed")
                return False
            _execute(p, executor.place_step_size)

            bottom_fn(obj_name)

            retreat_z = float(min(work_z + 0.04, z_hi))
            p = _try_lerp(np.array([col_x, target_y, retreat_z]),
                          n_substeps=_row_step_substeps(6))
            if p is not None:
                _execute(p, executor.retreat_step_size)

            cur_y = target_y
            while cur_y > front_row_y + 1e-6:
                step_y = max(cur_y - step, front_row_y)
                p = _try_lerp(np.array([col_x, step_y, retreat_z]),
                              n_substeps=_row_step_substeps(8))
                if p is None:
                    return True
                _execute(p, executor.retreat_step_size)
                cur_y = step_y

            p = _try_lerp(np.array([col_x, front_face_y - 0.06, retreat_z]),
                          n_substeps=8)
            if p is not None:
                _execute(p, executor.retreat_step_size)
            return True
        finally:
            env.controller._advance_delta_override = 0.1

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
        surface_z = region.level_z                       # level_z == surface
        half_z = float(env.get_object_half_size(obj_name)[2])
        target_center_z = surface_z + half_z
        held_dz = getattr(executor, "_held_grasp_dz", None)
        if held_dz is None:
            held_dz = -(half_z - _CUBE_GRASP_OFFSET)
        work_z = float(min(max(target_center_z - held_dz + _PLACE_CLEARANCE,
                               z_lo), z_hi))
        return _front_chain(obj_name, cell.region, col_x, target_y,
                            work_z, _detach_and_open)

    return pick_fn, put_fn
