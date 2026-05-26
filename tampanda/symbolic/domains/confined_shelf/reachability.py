"""Per-cell executability spec for the confined_shelf (Wang) domain.

See :mod:`tampanda.symbolic.domains._reachability` for the contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tampanda.symbolic.domains._reachability import (
    DomainSetup, ReachabilitySpec,
)
from tampanda.symbolic.domains.confined_shelf import (
    ConfinedShelfConfig,
    apply_runtime_tweaks,
    default_color_sort_goal,
    default_initial_layout,
    make_confined_shelf_bridge,
    make_confined_shelf_builder,
    set_cylinders_at_cells,
)
from tampanda.symbolic.domains.confined_shelf.env_builder import (
    STAGING_HOME_QPOS,
)


def _build_executor(env, table_z):
    # Domain-local executor: FRONT-only grasps + the LadderRRT planner so
    # place's approach/descent/retreat use the same joint-lerp ladder as
    # pick (FULL put ~1.6s -> ~0.3s), with RRT* as the fallback.  See
    # confined_shelf/executor.py.
    from tampanda.symbolic.domains.confined_shelf.executor import (
        build_confined_shelf_executor,
    )
    return build_confined_shelf_executor(env, table_z)


def make_setup(scratch_dir: Path, motion: bool = True) -> DomainSetup:
    cfg = ConfinedShelfConfig()
    builder, ws, cfg = make_confined_shelf_builder(scratch_dir=scratch_dir,
                                                    config=cfg)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)

    color_groups = [i % cfg.n_color_groups for i in range(cfg.n_cylinders)]
    color_names = ["red", "green", "yellow", "blue",
                   "purple", "orange"][: max(color_groups) + 1]
    cylinder_colors = [color_names[g] for g in color_groups]

    initial = default_initial_layout(cfg)
    set_cylinders_at_cells(env, ws, cfg, initial)

    object_ids = [f"cyl_{i}" for i in range(cfg.n_cylinders)]
    table_z = ws["shelf_interior"].level_z - cfg.cylinder_half_height
    executor = _build_executor(env, table_z) if motion else None

    bridge, objects = make_confined_shelf_bridge(
        env, ws, cfg, cylinder_colors, executor=executor,
    )
    goal = default_color_sort_goal(cfg, color_groups)

    half = np.array([cfg.cylinder_radius, cfg.cylinder_radius,
                     cfg.cylinder_half_height])

    def _half(name):
        return half

    def _place(env_, ws_, name, cell_id):
        cell = ws_.cell(cell_id)
        x, y, z = ws_.pose_for(cell)
        env_.set_object_pose(name, np.array([x, y, z]))

    # Palm-+y staging home (single source of truth in env_builder).
    shelf_home = STAGING_HOME_QPOS.copy()
    return DomainSetup(
        name="confined_shelf",
        env=env, workspace=ws, object_ids=object_ids,
        initial_layout=initial, goal=goal,
        executor=executor,
        place_at_cell=_place,
        object_half_extents=_half,
        parked_xyz=(cfg.hide_far_x, 0.0, cfg.cylinder_half_height),
        home_qpos=shelf_home,
    )


def reachability_spec() -> ReachabilitySpec:
    return ReachabilitySpec(
        domain_name="confined_shelf",
        full_regions=("shelf_interior",),
        layout_proxy="cyl_0",
        full_proxy="cyl_0",
    )
