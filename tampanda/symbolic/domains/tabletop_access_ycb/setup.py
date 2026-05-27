"""One-call setup for the dense-YCB tabletop-access fork.

Builds the env, executor, FRONT-grasp chains, staging home, and the
per-object footprints — everything the feasibility checker, executability
sweep, planner, and data-gen need.  Mirrors ``make_setup_access`` but for
the 2-region / multi-cell fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import mujoco

from tampanda.symbolic.workspace import Workspace

from tampanda.symbolic.domains.tabletop_access_ycb.env_builder import (
    TabletopAccessYcbConfig, make_tabletop_access_ycb_builder, apply_runtime_tweaks,
)
from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    ObjectFootprint, compute_all_footprints,
)
from tampanda.symbolic.domains.tabletop_access_ycb.chains import (
    make_ycb_access_chains, _access_geometry, _FRONT_QUAT,
)


@dataclass
class YcbAccessSetup:
    env: object
    workspace: Workspace
    config: TabletopAccessYcbConfig
    footprints: Dict[str, ObjectFootprint]
    executor: object
    pick_fn: object
    put_fn: object
    home_qpos: np.ndarray


def _build_executor(env, table_z, rrt_iters=3000):
    from tampanda.planners.rrt_star import RRTStar
    from tampanda.planners.grasp_planner import GraspPlanner, GraspType
    from tampanda.planners.pick_place import PickPlaceExecutor
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3
    rrt = RRTStar(env, max_iterations=rrt_iters)
    grasp = GraspPlanner(table_z=table_z, allowed_types=[GraspType.FRONT])
    return PickPlaceExecutor(env, rrt, grasp, use_attachment=True,
                             max_plan_iters=rrt_iters)


def _solve_staging(env, workspace, cfg, object_ids) -> np.ndarray:
    """Palm-+y staging pose just outside the shelf front at middle-deck z."""
    regions, front_face_y, _ = _access_geometry(env, workspace)
    parked = np.array([cfg.hide_far_x, 0.0, 0.10])
    saved = {n: env.get_object_pose(n) for n in object_ids}
    for n in object_ids:
        env.set_object_pose(n, parked)
    target = np.array([cfg.shelf_pos[0], front_face_y - 0.08,
                       workspace["middle_deck"].level_z])
    seed = np.array([np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    env.data.qpos[:7] = seed
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(target, _FRONT_QUAT)
    env.ik.pos_threshold, env.ik.ori_threshold = 0.005, 5e-3
    env.ik.converge_ik(0.005)
    q7 = env.ik.configuration.q[:7].copy()
    for n, (p, q) in saved.items():
        env.set_object_pose(n, np.asarray(p), np.asarray(q))
    env.reset_velocities()
    env.forward()
    return np.concatenate([q7, [0.04, 0.04]])


def build_setup(scratch_dir: Path, roster=None, motion: bool = True) -> YcbAccessSetup:
    from tampanda.planners.linear_ik import LinearIKPlanner

    builder, ws, cfg = make_tabletop_access_ycb_builder(scratch_dir, roster=roster)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    footprints = compute_all_footprints(env, cfg.object_ids, cfg.cell_size)

    executor = pick_fn = put_fn = None
    home = None
    if motion:
        executor = _build_executor(env, table_z=ws["middle_deck"].level_z)
        lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
        pick_fn, put_fn = make_ycb_access_chains(env, executor, ws, lik)
        home = _solve_staging(env, ws, cfg, list(cfg.object_ids))

    return YcbAccessSetup(env=env, workspace=ws, config=cfg, footprints=footprints,
                          executor=executor, pick_fn=pick_fn, put_fn=put_fn,
                          home_qpos=home)
