"""Per-cell executability spec for the redesigned multilevel_blocks domain.

Two physical tables (parts behind, stack in front).  Stack grid is a 3D
:class:`GridRegion` stack — five 2D regions named ``stack_L0`` … ``stack_L4``.

================================================================
What this checker tests vs what the reliability tests cover
================================================================
This module builds the setup used by ``examples/check_executability.py``,
which runs a SINGLE-PICK IK reachability probe per cell:

    * Executor used:   ``PickPlaceExecutor`` (generic top-down + GraspPlanner)
    * Test:            place ONE block at the cell, call ``executor.pick()``,
                       report success/failure.  No put.  No multi-block.
    * Grasp logic:     GraspPlanner generates TOP_DOWN_X / TOP_DOWN_Y
                       candidates around the block centroid; filters on
                       table-clearance and finger-tip-below geometry.
    * Scope:           proves the cell xy is kinematically REACHABLE by a
                       Franka with a top-down approach.

It is NOT a test of the symbolic-action MP chain.  The symbolic actions
(pick_cube, pick_flat_x, pick_upright, put_*, in-hand transforms) live in
:class:`MultilevelBlocksExecutor` and have their own grasp constants
(``_EE_TO_BLOCK_CENTRE_Z = 0.014``, no table-clearance margin — the chain
relies on the high-above → approach → descend stages to manage clearance
indirectly).  Reliability of THOSE actions is covered by:

    * ``examples/reliability_l0.py``   — full pick→put matrix on L0 for
                                        every shape (cube + flat + upright
                                        + 3×1 long).  Verifies block xy
                                        landed within tolerance.
    * ``examples/stacking_test.py``    — multi-block tower builds; checks
                                        placement drift + stack stability.
    * ``examples/test_multilevel_setup.py`` — interactive REPL.

If ``check_executability`` passes a cell that ``reliability_l0`` fails,
the failure is in the symbolic-action chain (e.g., retract IK basin),
not in pure kinematic reach.  Inversely, if check_executability fails a
cell, the symbolic-action chain CANNOT succeed there either — the
underlying Franka geometry can't reach.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from tampanda.symbolic.domains._reachability import (
    DomainSetup, ReachabilitySpec,
)
from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    cube_block_name,
    make_multilevel_blocks_builder,
    oblong_block_name,
    stack_region_name,
)
from tampanda.symbolic.workspace import Cell


def _build_pick_executor(env, cfg: MultilevelBlocksConfig):
    """Generic top-down PickPlaceExecutor for IK reachability probes.

    Uses the LOWEST table top z (the parts table) as ``table_z`` so the
    grasp-z clamp is conservative: picks of blocks resting on the parts
    table won't be raised; picks of blocks at higher levels (stack_L0+)
    are unaffected because their natural grasp z is already above the
    clamp.  GraspType defaults to TOP_DOWN — fine for flat blocks on
    either table; the symbolic-action FRONT chain is exercised by
    ``examples/reliability_l0.py``, not by ``check_executability``.
    """
    from tampanda.planners.rrt_star import RRTStar
    from tampanda.planners.grasp_planner import GraspPlanner
    from tampanda.planners.pick_place import PickPlaceExecutor

    parts_top_z = cfg.parts_table_pos[2] + 0.27   # _TABLE_TOP_LOCAL_Z
    rrt = RRTStar(env, max_iterations=6000)
    # Default GraspPlanner table_clearance is 25 mm, calibrated for
    # larger blocks (the cylinder/box benchmarks).  For multilevel_blocks'
    # 30 mm cubes that margin pushes the raised grasp pose ABOVE the
    # cube top (finger_tip_below=22.6 mm + 25 mm clearance + 13.7 mm
    # contact-offset hits cube_top within 5 mm — the planner returns
    # "block too short").  Tighten to 5 mm so small cubes pass.
    grasp = GraspPlanner(
        table_z=parts_top_z,
        table_clearance=0.005,
        approach_dist=0.10,
        lift_height=0.08,
    )
    return PickPlaceExecutor(env, rrt, grasp, use_attachment=True,
                                 max_plan_iters=6000)


def make_setup(scratch_dir: Path, motion: bool = True) -> DomainSetup:
    """Build the canonical reachability-test setup.

    Layout: every block parked off-workspace; the executability test will
    place the proxy block (``cube_0``) at each cell of every full region
    and attempt pick + put.
    """
    cfg = MultilevelBlocksConfig()
    builder, ws, cfg = make_multilevel_blocks_builder(
        scratch_dir=scratch_dir, config=cfg,
    )
    env = builder.build_env(rate=10000.0)

    object_ids: list = [cube_block_name(i) for i in range(cfg.n_cubes)]
    object_ids += [oblong_block_name(i) for i in range(cfg.n_oblong)]

    # Trivial initial layout — proxy block at the parts cell (0, 0); all
    # others parked.  The executability test will iterate cells.
    initial: Dict[str, Cell] = {cube_block_name(0): Cell("parts", 0, 0)}

    # Generic top-down PickPlaceExecutor for the IK reachability check.
    # The full symbolic-action MP (pick_cube / pick_flat_x / pick_upright
    # / etc.) lives in MultilevelBlocksExecutor; this lighter executor is
    # only what ``check_executability`` needs (a single ``.pick(name, pos,
    # half, quat)`` call probing IK + collision-free RRT* approach).
    executor = _build_pick_executor(env, cfg) if motion else None

    half = np.array([cfg.cube_half_extent] * 3)

    def _half(name: str) -> np.ndarray:
        return half

    def _place(env_, ws_, name: str, cell_id: str) -> None:
        cell = Cell.parse(cell_id)
        x, y, z = ws_.pose_for(cell)
        env_.set_object_pose(name, np.array([x, y, z]))

    # Goal: cube_0 at the centre cell of the top stack level (stack_L4).
    cells_x, cells_y, cells_z = cfg.stack_grid_cells
    top_centre = Cell(stack_region_name(cells_z - 1),
                       cells_x // 2, cells_y // 2)
    goal = [("in", cube_block_name(0), top_centre.id)]

    # Parts-side home seed (J1=-π/2, arm pointing -y) so the IK has a
    # good basin for reaching the parts table (default parts_table_pos
    # is at world -y).  The full-grid sweep across stack levels needs
    # a different seed and is handled separately by the executor's
    # per-region home selection; the layout-mode test only probes
    # parts__0_0, so the parts-side seed is sufficient here.
    parts_home = np.array([
        -np.pi / 2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853,
        0.04, 0.04,  # gripper open
    ])

    return DomainSetup(
        name="multilevel_blocks",
        env=env, workspace=ws, object_ids=object_ids,
        initial_layout=initial, goal=goal,
        executor=executor,
        place_at_cell=_place,
        object_half_extents=_half,
        parked_xyz=(cfg.hide_far_x, 0.0, cfg.cube_half_extent + 0.005),
        home_qpos=parts_home,
    )


def reachability_spec() -> ReachabilitySpec:
    """Reachability spec covers every cell of every region.

    Phase 0 probe showed all parts cells and all stack cells (at every
    level) are IK-reachable; the spec tests them all so any regression in
    geometry would surface immediately.
    """
    cfg = MultilevelBlocksConfig()
    cells_z = cfg.stack_grid_cells[2]
    full_regions = ("parts",) + tuple(stack_region_name(k)
                                         for k in range(cells_z))
    return ReachabilitySpec(
        domain_name="multilevel_blocks",
        full_regions=full_regions,
        layout_proxy=cube_block_name(0),
        full_proxy=cube_block_name(0),
    )
