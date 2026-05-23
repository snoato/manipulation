"""Workspace: a named collection of disjoint :class:`Region` s.

Domains stand up a workspace at scene-builder time and pass it to their
:class:`~tampanda.tamp.DomainBridge` factory and feasibility checker.  The
workspace owns:

* The set of regions (``shelf_interior``, ``shelf_top``, ``table``, …).
* A live :class:`OccupancyGrid` mapping every cell to the object currently
  placed there (or ``None``).

Picking marks the cells covered by an object's footprint as empty; placing
marks them as occupied by that object.  Multi-cell footprints are handled by
:meth:`OccupancyGrid.place` / :meth:`unplace`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from tampanda.symbolic.workspace.region import Cell, Footprint, GridRegion, Region


class Workspace:
    """An ordered collection of regions, plus the occupancy bookkeeping."""

    def __init__(self, regions: List[Region]) -> None:
        self._regions: "OrderedDict[str, Region]" = OrderedDict()
        # Case-insensitive index: lowercased-name -> canonical key.  Lets
        # callers that come from a case-insensitive source (e.g. PDDL
        # parsed by pymimir, which lowercases identifiers on parse) find
        # regions stored with mixed-case canonical names like ``stack_L0``.
        self._regions_ci: Dict[str, str] = {}
        for r in regions:
            if r.name in self._regions or r.name.lower() in self._regions_ci:
                raise ValueError(
                    f"duplicate region name {r.name!r} (case-insensitive)"
                )
            self._regions[r.name] = r
            self._regions_ci[r.name.lower()] = r.name
        self.occupancy = OccupancyGrid(self)

    # ------------------------------------------------------------------
    # Region access
    # ------------------------------------------------------------------

    @property
    def regions(self) -> "OrderedDict[str, Region]":
        return self._regions

    def _resolve_name(self, name: str) -> Optional[str]:
        """Resolve ``name`` to a canonical region key, trying exact match
        first then case-insensitive fallback.  Returns ``None`` if unknown."""
        if name in self._regions:
            return name
        return self._regions_ci.get(name.lower())

    def __getitem__(self, name: str) -> Region:
        canonical = self._resolve_name(name)
        if canonical is None:
            raise KeyError(name)
        return self._regions[canonical]

    def __contains__(self, name: str) -> bool:
        return self._resolve_name(name) is not None

    def __iter__(self) -> Iterator[Region]:
        return iter(self._regions.values())

    def cells(self) -> Iterator[Cell]:
        """Every cell across all regions, in region-then-row-major order."""
        for r in self._regions.values():
            yield from r.cells()

    def cell(self, cell_id: str) -> Cell:
        """Parse a global cell id and validate that the region exists."""
        cell = Cell.parse(cell_id)
        if self._resolve_name(cell.region) is None:
            raise KeyError(f"unknown region {cell.region!r} in cell id {cell_id!r}")
        return cell

    def region_of(self, cell: Cell) -> Region:
        canonical = self._resolve_name(cell.region)
        if canonical is None:
            raise KeyError(f"unknown region {cell.region!r}")
        return self._regions[canonical]

    def pose_for(self, cell: Cell) -> Tuple[float, float, float]:
        return self.region_of(cell).pose_for(cell)

    def cell_for(self, x: float, y: float, z: float) -> Optional[Cell]:
        """First region in declaration order that contains ``(x, y, z)``.

        Region order matters for overlapping projections — e.g. ``shelf_top``
        is declared before ``shelf_interior`` so a position on the shelf top
        is grounded to the top region rather than the interior.
        """
        for r in self._regions.values():
            cell = r.cell_for(x, y, z)
            if cell is not None:
                return cell
        return None


@dataclass
class _Occupant:
    """Internal: record of which cells an object covers and its anchor."""

    name: str
    anchor: Cell
    footprint: Footprint


class OccupancyGrid:
    """Workspace-wide ``cell_id -> object_name`` mapping.

    All cells start empty.  ``place(obj, anchor, footprint)`` claims every cell
    covered by the footprint; ``unplace(obj)`` releases them.

    Domains use this for two things:

    1. Emitting ``(occupied …) (empty …)`` PDDL fluents in the bridge.
    2. Rejecting placements that collide with already-placed objects.

    The grid is a pure-Python data structure — it does not read from MuJoCo;
    the bridge keeps it consistent with the simulation by calling place/unplace
    in action effects.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        # cell_id (str) -> object name placed there
        self._cells: Dict[str, str] = {}
        # object name -> Occupant record
        self._occupants: Dict[str, _Occupant] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def place(self, obj: str, anchor: Cell, footprint: Footprint = Footprint(1, 1)) -> List[Cell]:
        """Claim cells under ``footprint`` for ``obj``.

        Raises ``ValueError`` if any covered cell is already occupied (by a
        different object) or runs off the grid.
        """
        if obj in self._occupants:
            raise ValueError(
                f"{obj!r} already placed at {self._occupants[obj].anchor.id}; "
                f"unplace it before placing again"
            )
        region = self._workspace.region_of(anchor)
        if not isinstance(region, GridRegion):
            raise TypeError(
                f"footprint placement only supported on GridRegion, got {type(region).__name__}"
            )
        cells = region.footprint_at(anchor, footprint)
        if cells is None:
            raise ValueError(
                f"footprint {footprint} at {anchor.id} runs off region {region.name!r}"
            )
        for c in cells:
            if c.id in self._cells and self._cells[c.id] != obj:
                raise ValueError(
                    f"cell {c.id} already occupied by {self._cells[c.id]!r}"
                )
        for c in cells:
            self._cells[c.id] = obj
        self._occupants[obj] = _Occupant(name=obj, anchor=anchor, footprint=footprint)
        return cells

    def unplace(self, obj: str) -> List[Cell]:
        """Release every cell currently held by ``obj``.  Returns those cells."""
        if obj not in self._occupants:
            return []
        rec = self._occupants.pop(obj)
        cells = rec.footprint.cells_at(rec.anchor)
        for c in cells:
            if self._cells.get(c.id) == obj:
                del self._cells[c.id]
        return cells

    def clear(self) -> None:
        self._cells.clear()
        self._occupants.clear()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def is_occupied(self, cell: Cell) -> bool:
        return cell.id in self._cells

    def is_empty(self, cell: Cell) -> bool:
        return cell.id not in self._cells

    def occupant(self, cell: Cell) -> Optional[str]:
        return self._cells.get(cell.id)

    def cells_of(self, obj: str) -> List[Cell]:
        rec = self._occupants.get(obj)
        if rec is None:
            return []
        return rec.footprint.cells_at(rec.anchor)

    def anchor_of(self, obj: str) -> Optional[Cell]:
        rec = self._occupants.get(obj)
        return rec.anchor if rec is not None else None

    def __contains__(self, obj: str) -> bool:
        return obj in self._occupants

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return f"OccupancyGrid({len(self._occupants)} objects, {len(self._cells)} cells)"
