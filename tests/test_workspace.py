"""Unit tests for the multi-grid workspace abstractions.

These are pure-Python tests — no MuJoCo, no simulation.  They cover:

1. Cell id round-trip parsing.
2. GridRegion geometry (cells enumeration, point-to-cell, cell-to-pose,
   neighbours, footprint validation).
3. Workspace lookup across multiple regions in declaration order.
4. OccupancyGrid placement, multi-cell footprint marking, conflict detection,
   and unplacement.
"""

from __future__ import annotations

import pytest

from tampanda.symbolic.workspace import (
    Cell,
    Footprint,
    GridRegion,
    OccupancyGrid,
    Workspace,
)


# ----------------------------------------------------------------------
# Cell id round-trip
# ----------------------------------------------------------------------

class TestCell:
    def test_id_format_uses_double_underscore(self):
        cell = Cell("shelf_interior", 3, 7)
        assert cell.id == "shelf_interior__3_7"

    def test_parse_roundtrips_simple_region_name(self):
        cell = Cell("table", 0, 0)
        assert Cell.parse(cell.id) == cell

    def test_parse_roundtrips_region_with_underscores(self):
        cell = Cell("shelf_top", 12, 5)
        assert Cell.parse(cell.id) == cell

    def test_parse_rejects_missing_separator(self):
        with pytest.raises(ValueError, match="must contain '__'"):
            Cell.parse("cell_0_0")

    def test_parse_rejects_missing_indices(self):
        with pytest.raises(ValueError, match="missing indices"):
            Cell.parse("region__")

    def test_str_and_id_agree(self):
        cell = Cell("r", 2, 4)
        assert str(cell) == cell.id

    def test_cell_is_hashable(self):
        cells = {Cell("r", 0, 0), Cell("r", 0, 0), Cell("r", 1, 0)}
        assert len(cells) == 2


# ----------------------------------------------------------------------
# Footprint
# ----------------------------------------------------------------------

class TestFootprint:
    def test_single_cell_footprint(self):
        fp = Footprint(1, 1)
        cells = fp.cells_at(Cell("r", 5, 5))
        assert cells == [Cell("r", 5, 5)]

    def test_multi_cell_footprint_2x1(self):
        fp = Footprint(2, 1)
        cells = fp.cells_at(Cell("r", 0, 0))
        assert cells == [Cell("r", 0, 0), Cell("r", 1, 0)]

    def test_multi_cell_footprint_2x2(self):
        fp = Footprint(2, 2)
        cells = fp.cells_at(Cell("r", 0, 0))
        assert set(cells) == {
            Cell("r", 0, 0),
            Cell("r", 0, 1),
            Cell("r", 1, 0),
            Cell("r", 1, 1),
        }

    def test_zero_extent_rejected(self):
        with pytest.raises(ValueError):
            Footprint(0, 1)
        with pytest.raises(ValueError):
            Footprint(1, 0)


# ----------------------------------------------------------------------
# GridRegion
# ----------------------------------------------------------------------

class TestGridRegion:
    @pytest.fixture
    def region(self) -> GridRegion:
        # 4×3 grid, 5 cm cells, anchored at world (0.2, 0.4), z=0.27
        return GridRegion(
            name="shelf",
            origin=(0.2, 0.4),
            extent=(0.2, 0.15),
            cell_size=0.05,
            level_z=0.27,
            access_modes=("front",),
        )

    def test_cell_count(self, region: GridRegion):
        assert region.cells_x == 4
        assert region.cells_y == 3
        assert sum(1 for _ in region.cells()) == 12

    def test_pose_for_cell_centre(self, region: GridRegion):
        # cell (0, 0) centre is at origin + (0.5, 0.5) * cell_size
        x, y, z = region.pose_for(Cell("shelf", 0, 0))
        assert pytest.approx(x) == 0.225
        assert pytest.approx(y) == 0.425
        assert pytest.approx(z) == 0.27

    def test_pose_for_far_cell(self, region: GridRegion):
        x, y, z = region.pose_for(Cell("shelf", 3, 2))
        assert pytest.approx(x) == 0.375
        assert pytest.approx(y) == 0.525

    def test_cell_for_world_position(self, region: GridRegion):
        cell = region.cell_for(0.225, 0.425, 0.0)
        assert cell == Cell("shelf", 0, 0)

    def test_cell_for_outside_returns_none(self, region: GridRegion):
        assert region.cell_for(0.0, 0.0, 0.0) is None
        assert region.cell_for(0.5, 0.5, 0.0) is None

    def test_cell_for_clamps_upper_edge(self, region: GridRegion):
        # exact max coord — must land in the last cell, not off-grid
        x_max = 0.2 + 0.2
        y_max = 0.4 + 0.15
        cell = region.cell_for(x_max, y_max, 0.0)
        assert cell == Cell("shelf", 3, 2)

    def test_cell_pose_roundtrip(self, region: GridRegion):
        for cell in region.cells():
            x, y, z = region.pose_for(cell)
            recovered = region.cell_for(x, y, z)
            assert recovered == cell

    def test_neighbours_corner(self, region: GridRegion):
        nbrs = dict(region.neighbours(Cell("shelf", 0, 0)))
        assert set(nbrs) == {"north", "east"}
        assert nbrs["north"] == Cell("shelf", 0, 1)
        assert nbrs["east"] == Cell("shelf", 1, 0)

    def test_neighbours_interior(self, region: GridRegion):
        nbrs = dict(region.neighbours(Cell("shelf", 1, 1)))
        assert set(nbrs) == {"north", "south", "east", "west"}

    def test_pose_for_rejects_foreign_region(self, region: GridRegion):
        with pytest.raises(ValueError, match="not in region"):
            region.pose_for(Cell("table", 0, 0))

    def test_footprint_at_within_grid(self, region: GridRegion):
        cells = region.footprint_at(Cell("shelf", 0, 0), Footprint(2, 1))
        assert cells == [Cell("shelf", 0, 0), Cell("shelf", 1, 0)]

    def test_footprint_runs_off_grid(self, region: GridRegion):
        # 4x3 grid; anchor at (3,0) with 2x1 footprint runs off east edge
        assert region.footprint_at(Cell("shelf", 3, 0), Footprint(2, 1)) is None

    def test_constructor_rejects_double_underscore_in_name(self):
        with pytest.raises(ValueError, match="must not contain '__'"):
            GridRegion("bad__name", origin=(0, 0), extent=(0.1, 0.1), cell_size=0.05)

    def test_constructor_rejects_unknown_access_mode(self):
        with pytest.raises(ValueError, match="unknown access mode"):
            GridRegion(
                "r", origin=(0, 0), extent=(0.1, 0.1), cell_size=0.05,
                access_modes=("flying",),
            )

    def test_constructor_rejects_zero_cells(self):
        with pytest.raises(ValueError, match="zero cells"):
            GridRegion("r", origin=(0, 0), extent=(0.001, 0.05), cell_size=0.05)


# ----------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------

class TestWorkspace:
    @pytest.fixture
    def ws(self) -> Workspace:
        # Two disjoint regions stacked vertically: shelf interior (z=0.27)
        # and shelf top (z=0.55).  Both have the same XY footprint.
        interior = GridRegion(
            name="shelf_interior",
            origin=(0.2, 0.4),
            extent=(0.2, 0.15),
            cell_size=0.05,
            level_z=0.27,
            access_modes=("front",),
        )
        top = GridRegion(
            name="shelf_top",
            origin=(0.2, 0.4),
            extent=(0.2, 0.15),
            cell_size=0.05,
            level_z=0.55,
            access_modes=("top_down",),
        )
        return Workspace([interior, top])

    def test_iter_regions(self, ws: Workspace):
        names = [r.name for r in ws]
        assert names == ["shelf_interior", "shelf_top"]

    def test_lookup_by_name(self, ws: Workspace):
        assert ws["shelf_top"].level_z == pytest.approx(0.55)

    def test_total_cells(self, ws: Workspace):
        assert sum(1 for _ in ws.cells()) == 24  # 12 × 2

    def test_cell_lookup_parses_global_id(self, ws: Workspace):
        cell = ws.cell("shelf_top__1_2")
        assert cell == Cell("shelf_top", 1, 2)

    def test_cell_lookup_rejects_unknown_region(self, ws: Workspace):
        with pytest.raises(KeyError, match="unknown region"):
            ws.cell("garage__0_0")

    def test_pose_for_dispatches_by_region(self, ws: Workspace):
        _, _, z_int = ws.pose_for(Cell("shelf_interior", 0, 0))
        _, _, z_top = ws.pose_for(Cell("shelf_top", 0, 0))
        assert z_int == pytest.approx(0.27)
        assert z_top == pytest.approx(0.55)

    def test_duplicate_region_name_rejected(self):
        a = GridRegion("r", origin=(0, 0), extent=(0.1, 0.1), cell_size=0.05)
        b = GridRegion("r", origin=(1, 1), extent=(0.1, 0.1), cell_size=0.05)
        with pytest.raises(ValueError, match="duplicate region name"):
            Workspace([a, b])


# ----------------------------------------------------------------------
# OccupancyGrid
# ----------------------------------------------------------------------

class TestOccupancyGrid:
    @pytest.fixture
    def ws(self) -> Workspace:
        return Workspace([
            GridRegion(
                name="shelf",
                origin=(0.0, 0.0),
                extent=(0.2, 0.15),
                cell_size=0.05,
            ),
        ])

    def test_starts_empty(self, ws: Workspace):
        assert len(ws.occupancy) == 0
        for cell in ws.cells():
            assert ws.occupancy.is_empty(cell)

    def test_place_single_cell(self, ws: Workspace):
        anchor = Cell("shelf", 1, 1)
        cells = ws.occupancy.place("box_a", anchor)
        assert cells == [anchor]
        assert ws.occupancy.is_occupied(anchor)
        assert ws.occupancy.occupant(anchor) == "box_a"

    def test_place_multi_cell_marks_all(self, ws: Workspace):
        anchor = Cell("shelf", 0, 0)
        cells = ws.occupancy.place("big_box", anchor, Footprint(2, 2))
        assert set(cells) == {
            Cell("shelf", 0, 0),
            Cell("shelf", 1, 0),
            Cell("shelf", 0, 1),
            Cell("shelf", 1, 1),
        }
        for c in cells:
            assert ws.occupancy.occupant(c) == "big_box"

    def test_place_conflict_with_other_object(self, ws: Workspace):
        ws.occupancy.place("a", Cell("shelf", 1, 1))
        with pytest.raises(ValueError, match="already occupied"):
            ws.occupancy.place("b", Cell("shelf", 0, 1), Footprint(2, 1))

    def test_place_off_grid_rejected(self, ws: Workspace):
        # 4x3 grid -> Footprint(2,1) at (3,0) runs off
        with pytest.raises(ValueError, match="runs off region"):
            ws.occupancy.place("a", Cell("shelf", 3, 0), Footprint(2, 1))

    def test_double_place_same_object_rejected(self, ws: Workspace):
        ws.occupancy.place("a", Cell("shelf", 0, 0))
        with pytest.raises(ValueError, match="already placed"):
            ws.occupancy.place("a", Cell("shelf", 1, 1))

    def test_unplace_releases_all_cells(self, ws: Workspace):
        ws.occupancy.place("big", Cell("shelf", 0, 0), Footprint(2, 2))
        released = ws.occupancy.unplace("big")
        assert len(released) == 4
        for cell in released:
            assert ws.occupancy.is_empty(cell)
        assert "big" not in ws.occupancy

    def test_unplace_unknown_object_is_noop(self, ws: Workspace):
        assert ws.occupancy.unplace("ghost") == []

    def test_pick_then_replace_reusing_freed_cells(self, ws: Workspace):
        ws.occupancy.place("a", Cell("shelf", 0, 0), Footprint(2, 1))
        ws.occupancy.unplace("a")
        # cells freed up — another object can claim them, possibly partially overlapping
        ws.occupancy.place("b", Cell("shelf", 0, 0), Footprint(1, 1))
        assert ws.occupancy.occupant(Cell("shelf", 0, 0)) == "b"
        assert ws.occupancy.is_empty(Cell("shelf", 1, 0))

    def test_anchor_and_cells_of(self, ws: Workspace):
        ws.occupancy.place("c", Cell("shelf", 2, 1), Footprint(1, 2))
        assert ws.occupancy.anchor_of("c") == Cell("shelf", 2, 1)
        assert set(ws.occupancy.cells_of("c")) == {
            Cell("shelf", 2, 1),
            Cell("shelf", 2, 2),
        }
