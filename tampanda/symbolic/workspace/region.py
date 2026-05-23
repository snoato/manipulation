"""Region abstractions for multi-grid workspaces.

A :class:`Region` is a bounded sub-volume where the robot picks and places.
Every pickable/placeable surface in a domain is exactly one region; multiple
regions form a :class:`~tampanda.symbolic.workspace.workspace.Workspace`.

Cells are identified globally by ``"<region>__<ix>_<iy>"`` (double underscore
between region and indices) so PDDL never has to qualify cells by region.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Iterator, List, Optional, Tuple


# Module-level alias instead of typing.Literal so this stays Python 3.7-friendly.
ACCESS_MODES = ("top_down", "front", "back", "left", "right", "side", "any")


@dataclass(frozen=True)
class Cell:
    """A discrete location inside a :class:`Region`.

    Cells are immutable and hashable; they appear directly as PDDL objects
    via :attr:`id`.
    """

    region: str
    ix: int
    iy: int

    @property
    def id(self) -> str:
        return f"{self.region}__{self.ix}_{self.iy}"

    @classmethod
    def parse(cls, cell_id: str) -> "Cell":
        if "__" not in cell_id:
            raise ValueError(f"cell id must contain '__': {cell_id!r}")
        region, idx_part = cell_id.rsplit("__", 1)
        ix_str, _, iy_str = idx_part.partition("_")
        if not ix_str or not iy_str:
            raise ValueError(f"cell id missing indices: {cell_id!r}")
        return cls(region=region, ix=int(ix_str), iy=int(iy_str))

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True)
class Footprint:
    """Cells covered by an object placed at a given anchor cell.

    A footprint is described by an integer ``(dx, dy)`` extent measured in
    cells; the anchor cell sits at the south-west corner of the footprint.
    Single-cell objects use ``Footprint(1, 1)``.

    Yaw is *not* modelled — objects align to grid axes.  Domains that need
    rotated footprints either pre-pad the cell ranges or use a finer grid.
    """

    dx: int
    dy: int

    def __post_init__(self) -> None:
        if self.dx < 1 or self.dy < 1:
            raise ValueError(f"Footprint extents must be >=1: {self}")

    def cells_at(self, anchor: Cell) -> List[Cell]:
        """Cells covered when the object's anchor sits at ``anchor``."""
        return [
            Cell(anchor.region, anchor.ix + ddx, anchor.iy + ddy)
            for ddx in range(self.dx)
            for ddy in range(self.dy)
        ]


SINGLE_CELL = Footprint(1, 1)


class Region(ABC):
    """Abstract pickable/placeable region.

    Subclasses must implement geometry queries (:meth:`cell_for`,
    :meth:`pose_for`) and enumeration (:meth:`cells`, :meth:`__iter__`).
    """

    name: str
    access_modes: Tuple[str, ...]

    @abstractmethod
    def cells(self) -> Iterator[Cell]:
        """Iterate every cell in the region in ``(ix, iy)`` row-major order."""

    @abstractmethod
    def in_region(self, x: float, y: float, z: float) -> bool:
        """True iff ``(x, y, z)`` lies inside the region's bounding volume."""

    @abstractmethod
    def cell_for(self, x: float, y: float, z: float) -> Optional[Cell]:
        """Return the cell containing ``(x, y, z)``, or ``None`` if outside."""

    @abstractmethod
    def pose_for(self, cell: Cell) -> Tuple[float, float, float]:
        """World position of the cell centre at the region's resting plane."""

    @abstractmethod
    def neighbours(self, cell: Cell) -> List[Tuple[str, Cell]]:
        """Return ``(direction, neighbour_cell)`` pairs inside the region."""

    def __iter__(self) -> Iterator[Cell]:
        return self.cells()


class GridRegion(Region):
    """Static axis-aligned 2D grid region.

    Cell ``(0, 0)`` sits at ``origin``; cell ``(cells_x-1, cells_y-1)`` at
    ``origin + (extent_x, extent_y)``.  All cells share a single resting
    z-height ``level_z``.

    Args:
        name:         Region identifier — must be a valid PDDL atom prefix
                      (letters, digits, underscore; no double underscore).
        origin:       World-frame ``(x, y)`` coordinate of the south-west
                      corner of the grid.
        extent:       ``(size_x, size_y)`` in metres.
        cell_size:    Edge length of a square cell in metres.
        level_z:      Resting plane z in metres (usually shelf-floor or
                      table top + half-object-height).
        access_modes: Allowed grasp approach directions.  Used by feasibility
                      checkers to filter candidate end-effector poses.
    """

    def __init__(
        self,
        name: str,
        origin: Tuple[float, float],
        extent: Tuple[float, float],
        cell_size: float,
        level_z: float = 0.0,
        access_modes: Tuple[str, ...] = ("top_down",),
        excluded_cells: Iterable[Tuple[int, int]] = (),
    ) -> None:
        if "__" in name:
            raise ValueError(f"region name must not contain '__': {name!r}")
        for mode in access_modes:
            if mode not in ACCESS_MODES:
                raise ValueError(
                    f"unknown access mode {mode!r}; valid: {ACCESS_MODES}"
                )
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive: {cell_size}")

        self.name = name
        self.origin = (float(origin[0]), float(origin[1]))
        self.extent = (float(extent[0]), float(extent[1]))
        self.cell_size = float(cell_size)
        self.level_z = float(level_z)
        self.access_modes = tuple(access_modes)

        self.cells_x = int(round(self.extent[0] / self.cell_size))
        self.cells_y = int(round(self.extent[1] / self.cell_size))
        if self.cells_x < 1 or self.cells_y < 1:
            raise ValueError(
                f"region {name!r} has zero cells "
                f"(extent {self.extent} / cell_size {self.cell_size})"
            )

        # ``excluded_cells`` flags grid positions the geometry permits
        # but the kinematic stack cannot reach (e.g. column-1 of a
        # narrow shelf where the gripper would clip the separator
        # post; back-corner cells past the Franka's xy reach).  Such
        # cells are hidden from :meth:`cells`, :meth:`cell_for` and
        # :meth:`neighbours` so the PDDL bridge — which builds its
        # cell-object set via ``region.cells()`` — never sees them.
        # ``pose_for`` and ``cell(ix, iy)`` still work, since callers
        # of those expect a geometric query irrespective of
        # reachability.
        self.excluded_cells: FrozenSet[Tuple[int, int]] = frozenset(
            (int(ix), int(iy)) for ix, iy in excluded_cells
        )
        for ix, iy in self.excluded_cells:
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise ValueError(
                    f"region {name!r}: excluded cell ({ix}, {iy}) is outside "
                    f"the {self.cells_x}x{self.cells_y} grid"
                )

    # ------------------------------------------------------------------
    # Region API
    # ------------------------------------------------------------------

    def cells(self) -> Iterator[Cell]:
        """Iterate every *reachable* cell.  Cells listed in
        ``excluded_cells`` are skipped so the PDDL bridge never grounds
        them as objects.
        """
        for ix in range(self.cells_x):
            for iy in range(self.cells_y):
                if (ix, iy) in self.excluded_cells:
                    continue
                yield Cell(self.name, ix, iy)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.cells_x and 0 <= iy < self.cells_y

    def is_excluded(self, ix: int, iy: int) -> bool:
        return (ix, iy) in self.excluded_cells

    def cell(self, ix: int, iy: int) -> Cell:
        if not self.in_bounds(ix, iy):
            raise IndexError(f"cell ({ix}, {iy}) outside {self.name}")
        return Cell(self.name, ix, iy)

    def in_region(self, x: float, y: float, z: float) -> bool:
        x0, y0 = self.origin
        x1 = x0 + self.extent[0]
        y1 = y0 + self.extent[1]
        return x0 <= x <= x1 and y0 <= y <= y1

    def cell_for(self, x: float, y: float, z: float) -> Optional[Cell]:
        if not self.in_region(x, y, z):
            return None
        ix = int((x - self.origin[0]) / self.cell_size)
        iy = int((y - self.origin[1]) / self.cell_size)
        # Clamp the upper edge — exact-max coordinates round outside the
        # last cell otherwise.
        ix = min(max(ix, 0), self.cells_x - 1)
        iy = min(max(iy, 0), self.cells_y - 1)
        if (ix, iy) in self.excluded_cells:
            return None
        return Cell(self.name, ix, iy)

    def pose_for(self, cell: Cell) -> Tuple[float, float, float]:
        if cell.region.lower() != self.name.lower():
            raise ValueError(
                f"cell {cell.id!r} not in region {self.name!r}"
            )
        if not self.in_bounds(cell.ix, cell.iy):
            raise IndexError(f"{cell.id} outside grid extent")
        cx = self.origin[0] + (cell.ix + 0.5) * self.cell_size
        cy = self.origin[1] + (cell.iy + 0.5) * self.cell_size
        return (cx, cy, self.level_z)

    def neighbours(self, cell: Cell) -> List[Tuple[str, Cell]]:
        if cell.region.lower() != self.name.lower():
            raise ValueError(
                f"cell {cell.id!r} not in region {self.name!r}"
            )
        out: List[Tuple[str, Cell]] = []
        for direction, dx, dy in (
            ("north", 0, 1),
            ("south", 0, -1),
            ("east", 1, 0),
            ("west", -1, 0),
        ):
            ix2, iy2 = cell.ix + dx, cell.iy + dy
            if self.in_bounds(ix2, iy2) and (ix2, iy2) not in self.excluded_cells:
                out.append((direction, Cell(self.name, ix2, iy2)))
        return out

    # ------------------------------------------------------------------
    # Footprint helpers
    # ------------------------------------------------------------------

    def footprint_at(self, anchor: Cell, footprint: Footprint) -> Optional[List[Cell]]:
        """Cells covered by an object whose anchor is at ``anchor``.

        Returns ``None`` if the footprint runs off the grid OR includes
        any excluded cell (so callers can treat that as an invalid
        placement).
        """
        cells = footprint.cells_at(anchor)
        for c in cells:
            if not self.in_bounds(c.ix, c.iy):
                return None
            if (c.ix, c.iy) in self.excluded_cells:
                return None
        return cells

    def __repr__(self) -> str:
        return (
            f"GridRegion(name={self.name!r}, cells={self.cells_x}x{self.cells_y}, "
            f"cell_size={self.cell_size:.4f}m, level_z={self.level_z:.3f}m, "
            f"access={self.access_modes})"
        )
