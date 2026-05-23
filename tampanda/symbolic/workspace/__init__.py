"""Multi-grid workspace abstractions used by the new manipulation domains.

A :class:`Workspace` is a named collection of disjoint :class:`Region`
instances; every place the robot picks or places from is exactly one region.
Domains compose a workspace from one or more regions and pass it to their
:class:`~tampanda.tamp.DomainBridge` factory and feasibility checker.

This package is *additive* — the existing tabletop ``GridDomain`` and blocks
``BlocksDomain`` are untouched.  New domains under
``tampanda/symbolic/domains/{confined_shelf,tabletop_access,confined_pickonly,
multilevel_blocks}/`` build on top of this package.

See :mod:`tampanda.symbolic.workspace.region` for the region types and
:mod:`tampanda.symbolic.workspace.workspace` for the workspace + occupancy
bookkeeping.
"""

from tampanda.symbolic.workspace.region import (
    ACCESS_MODES,
    Cell,
    Footprint,
    GridRegion,
    Region,
    SINGLE_CELL,
)
from tampanda.symbolic.workspace.workspace import OccupancyGrid, Workspace

__all__ = [
    "ACCESS_MODES",
    "Cell",
    "Footprint",
    "GridRegion",
    "OccupancyGrid",
    "Region",
    "SINGLE_CELL",
    "Workspace",
]
