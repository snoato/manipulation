"""Reliability tests for the redesigned multi-level blocks (Kulshrestha) domain.

Pure-Python — no MuJoCo simulation.  Block poses come from a FakeEnv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    cube_block_name,
    make_multilevel_blocks_bridge,
    make_multilevel_blocks_builder,
    oblong_block_name,
    stack_region_name,
)
from tampanda.symbolic.workspace import Cell


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal env stub: just per-object pose lookup."""

    def __init__(self, poses: Dict[str, tuple]) -> None:
        # poses: {name: (pos_xyz, quat_wxyz)}
        self._poses = dict(poses)

    def get_object_pose(self, name: str):
        pos, quat = self._poses[name]
        return np.asarray(pos), np.asarray(quat)


_HIDE = (np.array([100.0, 0.0, 0.05]),
         np.array([1.0, 0.0, 0.0, 0.0]))
_QUAT_FLAT_X = np.array([1.0, 0.0, 0.0, 0.0])
_QUAT_FLAT_Y = np.array([0.7071068, 0.0, 0.0, 0.7071068])
_QUAT_UPRIGHT = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def _small_cfg(**overrides) -> MultilevelBlocksConfig:
    base = dict(n_cubes=2, n_oblong=2,
                stack_grid_cells=(3, 3, 3),
                parts_grid_cells=(3, 3))
    base.update(overrides)
    return MultilevelBlocksConfig(**base)


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults_match_phase0_geometry(self):
        cfg = MultilevelBlocksConfig()
        assert cfg.cube_half_extent == pytest.approx(0.015)
        assert cfg.cube_size == pytest.approx(0.030)
        assert cfg.stack_table_pos == (0.00, 0.50, 0.00)
        assert cfg.parts_table_pos == (0.00, -0.45, 0.00)
        assert cfg.stack_grid_cells == (10, 10, 5)
        assert cfg.parts_grid_cells == (15, 15)

    def test_rejects_negative_block_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            MultilevelBlocksConfig(n_cubes=-1)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one block"):
            MultilevelBlocksConfig(n_cubes=0, n_oblong=0)

    def test_n_blocks_sum(self):
        cfg = MultilevelBlocksConfig(n_cubes=3, n_oblong=2)
        assert cfg.n_blocks == 5


# ===========================================================================
# Builder & workspace
# ===========================================================================


class TestBuilder:
    @pytest.fixture
    def built(self, tmp_path: Path):
        return make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=_small_cfg(),
        )

    def test_workspace_has_parts_plus_stack_levels(self, built):
        _, ws, cfg = built
        regions = list(ws.regions)
        assert "parts" in regions
        for level in range(cfg.stack_levels):
            assert stack_region_name(level) in regions
        assert len(regions) == 1 + cfg.stack_levels

    def test_stack_levels_z_increase(self, built):
        _, ws, cfg = built
        zs = [ws[stack_region_name(level)].level_z
              for level in range(cfg.stack_levels)]
        for a, b in zip(zs, zs[1:]):
            assert b - a == pytest.approx(cfg.cube_size)

    def test_parts_grid_dimensions(self, built):
        _, ws, cfg = built
        parts = ws["parts"]
        assert (parts.cells_x, parts.cells_y) == cfg.parts_grid_cells

    def test_stack_grid_dimensions(self, built):
        _, ws, cfg = built
        sx, sy, _ = cfg.stack_grid_cells
        for level in range(cfg.stack_levels):
            r = ws[stack_region_name(level)]
            assert (r.cells_x, r.cells_y) == (sx, sy)

    def test_parts_table_behind_robot(self, built):
        _, _, cfg = built
        assert cfg.parts_table_pos[1] < 0  # behind robot (-y)

    def test_stack_table_in_front(self, built):
        _, _, cfg = built
        assert cfg.stack_table_pos[1] > 0  # in front of robot (+y)


# ===========================================================================
# Bridge: object set and static predicates
# ===========================================================================


class TestBridgeObjects:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        cfg = _small_cfg()
        _, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=cfg,
        )
        env = _FakeEnv({
            cube_block_name(i): _HIDE for i in range(cfg.n_cubes)
        } | {
            oblong_block_name(i): _HIDE for i in range(cfg.n_oblong)
        })
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        return bridge, objects, ws, cfg

    def test_object_set_contains_all_blocks(self, setup):
        _, objects, _, cfg = setup
        for i in range(cfg.n_cubes):
            assert cube_block_name(i) in objects["block"]
        for i in range(cfg.n_oblong):
            assert oblong_block_name(i) in objects["block"]
        assert len(objects["block"]) == cfg.n_blocks

    def test_object_set_contains_all_cells(self, setup):
        _, objects, ws, cfg = setup
        # 3*3 parts + 3*(3*3) stack = 9 + 27 = 36 cells
        expected_count = (cfg.parts_grid_cells[0] * cfg.parts_grid_cells[1]
                          + cfg.stack_grid_cells[0] * cfg.stack_grid_cells[1]
                              * cfg.stack_grid_cells[2])
        assert len(objects["cell"]) == expected_count

    def test_shape_predicates_disjoint(self, setup):
        bridge, objects, _, cfg = setup
        state = bridge.ground_state(objects)
        for i in range(cfg.n_cubes):
            assert state[("cube", cube_block_name(i))] is True
            assert state[("oblong", cube_block_name(i))] is False
        for i in range(cfg.n_oblong):
            assert state[("cube", oblong_block_name(i))] is False
            assert state[("oblong", oblong_block_name(i))] is True

    def test_region_markers(self, setup):
        bridge, objects, _, cfg = setup
        state = bridge.ground_state(objects)
        # spot checks
        assert state[("in-parts", "parts__0_0")] is True
        assert state[("in-stack", "parts__0_0")] is False
        assert state[("in-stack", "stack_L0__0_0")] is True
        assert state[("in-parts", "stack_L0__0_0")] is False


# ===========================================================================
# Bridge: directional adjacency
# ===========================================================================


class TestAdjacency:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        cfg = _small_cfg()
        _, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=cfg,
        )
        env = _FakeEnv({
            cube_block_name(i): _HIDE for i in range(cfg.n_cubes)
        } | {
            oblong_block_name(i): _HIDE for i in range(cfg.n_oblong)
        })
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        return bridge, objects

    def test_east_of_is_directional(self, setup):
        bridge, objects = setup
        state = bridge.ground_state(objects)
        assert state[("east-of", "parts__0_0", "parts__1_0")] is True
        # reverse direction must be false (asymmetric)
        assert state[("east-of", "parts__1_0", "parts__0_0")] is False

    def test_north_of_is_directional(self, setup):
        bridge, objects = setup
        state = bridge.ground_state(objects)
        assert state[("north-of", "parts__0_0", "parts__0_1")] is True
        assert state[("north-of", "parts__0_1", "parts__0_0")] is False

    def test_above_only_within_stack(self, setup):
        bridge, objects = setup
        state = bridge.ground_state(objects)
        assert state[("above", "stack_L0__0_0", "stack_L1__0_0")] is True
        assert state[("above", "stack_L1__0_0", "stack_L0__0_0")] is False
        # parts cells have no above relation
        assert state[("above", "parts__0_0", "parts__0_1")] is False

    def test_above_capped_at_top_level(self, setup):
        bridge, objects = setup
        state = bridge.ground_state(objects)
        # 3-level test config, so stack_L2 is top.  Nothing should be above it.
        for ix in range(3):
            for iy in range(3):
                top = f"stack_L2__{ix}_{iy}"
                for c in objects["cell"]:
                    assert state[("above", top, c)] is False, \
                        f"unexpected above({top}, {c})"


# ===========================================================================
# Bridge: pose-to-cell grounding for cubes
# ===========================================================================


class TestCubeGrounding:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        cfg = _small_cfg()
        _, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=cfg,
        )
        # All blocks hidden initially.
        env = _FakeEnv({
            cube_block_name(i): _HIDE for i in range(cfg.n_cubes)
        } | {
            oblong_block_name(i): _HIDE for i in range(cfg.n_oblong)
        })
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        return bridge, objects, ws, cfg, env

    def test_hidden_block_in_no_cell(self, setup):
        bridge, objects, _, cfg, _ = setup
        state = bridge.ground_state(objects)
        for i in range(cfg.n_cubes):
            for c in objects["cell"]:
                assert state[("in", cube_block_name(i), c)] is False

    def test_cube_placed_on_stack_grounds_to_cell(self, setup):
        bridge, objects, ws, _, env = setup
        target = Cell("stack_L0", 1, 1)
        x, y, z = ws.pose_for(target)
        env._poses[cube_block_name(0)] = (np.array([x, y, z]), _QUAT_FLAT_X)
        state = bridge.ground_state(objects)
        assert state[("in", cube_block_name(0), target.id)] is True
        assert state[("empty", target.id)] is False
        # other cells stay empty
        other = Cell("stack_L0", 0, 0)
        assert state[("empty", other.id)] is True

    def test_held_cube_does_not_occupy_any_cell(self, setup):
        bridge, objects, ws, _, env = setup
        target = Cell("stack_L0", 1, 1)
        x, y, z = ws.pose_for(target)
        # Block physically at the cell pose, but the gripper "thinks" it's holding it.
        env._poses[cube_block_name(0)] = (np.array([x, y, z]), _QUAT_FLAT_X)
        bridge._fluent_state[("held-cube", cube_block_name(0))] = True
        bridge._fluent_state[("gripper-empty",)] = False
        state = bridge.ground_state(objects)
        # The held block must not register as in any cell.
        assert state[("in", cube_block_name(0), target.id)] is False
        # The cell stays empty (no other block is there).
        assert state[("empty", target.id)] is True


# ===========================================================================
# Bridge: pose-to-cell grounding for oblong blocks
# ===========================================================================


class TestOblongGrounding:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        cfg = _small_cfg(n_cubes=0, n_oblong=2)
        _, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=cfg,
        )
        env = _FakeEnv({
            oblong_block_name(i): _HIDE for i in range(cfg.n_oblong)
        })
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        return bridge, objects, ws, cfg, env

    def test_flat_x_oblong_spans_two_east_west_cells(self, setup):
        bridge, objects, ws, cfg, env = setup
        c1 = Cell("stack_L0", 0, 1)
        c2 = Cell("stack_L0", 1, 1)
        # block centroid is the midpoint of c1 and c2
        x1, y1, z1 = ws.pose_for(c1)
        x2, _, _ = ws.pose_for(c2)
        env._poses[oblong_block_name(0)] = (
            np.array([(x1 + x2) / 2, y1, z1]), _QUAT_FLAT_X,
        )
        state = bridge.ground_state(objects)
        assert state[("in", oblong_block_name(0), c1.id)] is True
        assert state[("in", oblong_block_name(0), c2.id)] is True
        assert state[("empty", c1.id)] is False
        assert state[("empty", c2.id)] is False

    def test_flat_y_oblong_spans_two_north_south_cells(self, setup):
        bridge, objects, ws, cfg, env = setup
        c1 = Cell("stack_L0", 1, 0)
        c2 = Cell("stack_L0", 1, 1)
        x1, y1, z1 = ws.pose_for(c1)
        _, y2, _ = ws.pose_for(c2)
        env._poses[oblong_block_name(0)] = (
            np.array([x1, (y1 + y2) / 2, z1]), _QUAT_FLAT_Y,
        )
        state = bridge.ground_state(objects)
        assert state[("in", oblong_block_name(0), c1.id)] is True
        assert state[("in", oblong_block_name(0), c2.id)] is True

    def test_upright_oblong_spans_two_levels(self, setup):
        bridge, objects, ws, cfg, env = setup
        c_low = Cell("stack_L0", 1, 1)
        c_high = Cell("stack_L1", 1, 1)
        xl, yl, zl = ws.pose_for(c_low)
        _, _, zh = ws.pose_for(c_high)
        env._poses[oblong_block_name(0)] = (
            np.array([xl, yl, (zl + zh) / 2]), _QUAT_UPRIGHT,
        )
        state = bridge.ground_state(objects)
        assert state[("in", oblong_block_name(0), c_low.id)] is True
        assert state[("in", oblong_block_name(0), c_high.id)] is True


# ===========================================================================
# Planning end-to-end: 3-block tower puzzle
# ===========================================================================


class TestPlanning:
    """The smoke test that the user requested: 2 oblong + 1 cube on parts;
    goal = cube at top of a stack column."""

    def test_three_block_tower_plan(self, tmp_path: Path):
        # 2x3 parts grid + 1x1x5 stack column.  Just enough cells for the
        # initial scatter + tower construction.
        cfg = MultilevelBlocksConfig(
            n_cubes=1, n_oblong=2,
            parts_grid_cells=(2, 3),
            stack_grid_cells=(1, 1, 5),
        )
        _, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=tmp_path, config=cfg,
        )

        # Initial placements
        env = _FakeEnv({
            oblong_block_name(0): (
                np.array(ws.pose_for(Cell("parts", 0, 0)))
                + np.array([cfg.cube_size / 2, 0, 0]),
                _QUAT_FLAT_X,
            ),
            oblong_block_name(1): (
                np.array(ws.pose_for(Cell("parts", 0, 1)))
                + np.array([cfg.cube_size / 2, 0, 0]),
                _QUAT_FLAT_X,
            ),
            cube_block_name(0): (
                np.array(ws.pose_for(Cell("parts", 0, 2))),
                _QUAT_FLAT_X,
            ),
        })

        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        goals = [("in", cube_block_name(0), "stack_L4__0_0")]
        plan = bridge.plan(objects, goals=goals)
        assert plan is not None, "Fast Downward returned no plan"
        assert len(plan) == 8
        # Last action must place the cube at the top.  Fast Downward
        # lowercases identifiers, so compare case-insensitively.
        last_action, last_params = plan[-1]
        assert last_action == "put-cube"
        assert last_params[0].lower() == cube_block_name(0).lower()
        assert last_params[1].lower() == "stack_l4__0_0"
