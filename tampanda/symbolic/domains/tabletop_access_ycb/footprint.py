"""Overlap-derived multi-cell footprints for the dense-YCB domain.

An object's footprint is the set of grid cells its **collision
cross-section actually overlaps** (projected to the resting plane at the
object's canonical grasp yaw, inflated by a place-clearance margin).  A
round object (can, fruit) yields a round/plus pattern with its bounding-
box corners left free; a box yields the full rectangle.  This realises
the invariant requested for this domain: *a cell left ``empty`` is
genuinely clear of the object* — so neighbours may pack into the corner
cells round objects don't fill.

The canonical pose places the object so its cross-section centre lands at
the geometric centre of its ``dx × dy`` bounding-cell block and its base
rests on the region surface.  Mesh body-origins are not AABB-centred, so
:meth:`ObjectFootprint.place_pose` derives the body-origin position from
the measured cross-section centre and bottom-z (the Phase-1 finding).

Footprints are computed once per object against the built env (the mesh
verts are only available after compilation) and are fixed thereafter.
Objects sharing the same occupied-offset pattern share a PDDL action
schema (keyed by :attr:`ObjectFootprint.key`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import mujoco
from matplotlib.path import Path as _MplPath

from tampanda.symbolic.workspace import Cell, GridRegion

_DEFAULT_CLEAR = 0.006     # place-clearance margin (m) — keeps "free" cells free
_FILL_STEP = 0.004         # hull-fill lattice spacing (m)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _collision_points_bodyframe(env, body_id: int) -> np.ndarray:
    """Return (N,3) collision-geom vertices in the body frame.

    Mesh geoms contribute their vertices; primitive geoms contribute their
    8 box-corner / bounding-box points.  Visual-only geoms (contype=0 and
    conaffinity=0) are skipped — mirrors ``get_object_half_size``.
    """
    m = env.model
    out: List[np.ndarray] = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != body_id:
            continue
        if m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0:
            continue
        gtype = m.geom_type[gid]
        size = m.geom_size[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gid]
            s = m.mesh_vertadr[mid]
            n = m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[s:s + n], dtype=float)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            hx, hy, hz = size[:3]
            v = np.array([[sx * hx, sy * hy, sz * hz]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        else:  # cylinder / capsule / sphere -> bounding box
            r = float(size[0])
            hz = float(size[1]) if gtype != mujoco.mjtGeom.mjGEOM_SPHERE else r
            v = np.array([[sx * r, sy * r, sz * hz]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        # apply the geom's local pose within the body
        gpos = np.array(m.geom_pos[gid], dtype=float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array(m.geom_quat[gid], dtype=float))
        v = v @ R.reshape(3, 3).T + gpos
        out.append(v)
    if not out:
        raise ValueError(f"body id {body_id} has no collision geom")
    return np.vstack(out)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain).  Returns CCW hull verts."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _fill_hull(hull: np.ndarray, step: float) -> np.ndarray:
    """Dense lattice of points inside the hull (plus the hull verts).

    Needed because raw mesh verts cluster on the surface — a large box
    would leave its interior cells with no nearby vert and be wrongly
    classified free.  Filling the hull makes occupancy conservative.
    """
    if len(hull) < 3:
        return hull.reshape(-1, 2)
    path = _MplPath(hull)
    lo = hull.min(0)
    hi = hull.max(0)
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = path.contains_points(grid)
    pts = grid[inside]
    return np.vstack([pts, hull]) if len(pts) else hull


# ----------------------------------------------------------------------
# Footprint
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectFootprint:
    """Fixed multi-cell footprint of one object on the grid.

    Attributes:
        body_id:  Scene/PDDL body id.
        dx, dy:   Bounding-cell extents.
        offsets:  Occupied cell offsets ``(ddx, ddy)`` relative to the
                  south-west anchor (a subset of the ``dx × dy`` block for
                  round objects; the full block for boxes).
        yaw:      Canonical resting yaw (min horizontal extent along world-x,
                  the FRONT-grasp finger axis).
        quat:     Body orientation [w,x,y,z] at the canonical yaw.
        cx, cy:   Cross-section AABB centre in the yawed body frame.
        bottom:   Min collision-vertex z in the body frame (resting offset).
        key:      Schema key shared by objects with the same offset pattern.
    """

    body_id: str
    dx: int
    dy: int
    offsets: Tuple[Tuple[int, int], ...]
    yaw: float
    quat: Tuple[float, float, float, float]
    cx: float
    cy: float
    bottom: float
    key: str

    @property
    def n_cells(self) -> int:
        return len(self.offsets)

    def cells_at(self, anchor: Cell) -> List[Cell]:
        """Occupied cells when the SW anchor sits at ``anchor``."""
        return [Cell(anchor.region, anchor.ix + ddx, anchor.iy + ddy)
                for ddx, ddy in self.offsets]

    def bounding_cells_at(self, anchor: Cell) -> List[Cell]:
        """All ``dx × dy`` bounding cells (occupied + free corners)."""
        return [Cell(anchor.region, anchor.ix + ix, anchor.iy + iy)
                for ix in range(self.dx) for iy in range(self.dy)]

    def centroid_world(self, region: GridRegion, anchor: Cell) -> Tuple[float, float]:
        ax, ay, _ = region.pose_for(anchor)
        cs = region.cell_size
        return (ax + (self.dx - 1) / 2 * cs, ay + (self.dy - 1) / 2 * cs)

    def place_pose(self, region: GridRegion, anchor: Cell):
        """``(pos, quat)`` body pose to place the object at ``anchor``.

        Cross-section centre lands at the bounding-block centroid; base
        rests on ``region.level_z`` (the deck surface).
        """
        wx, wy = self.centroid_world(region, anchor)
        pos = np.array([wx - self.cx, wy - self.cy, region.level_z - self.bottom])
        return pos, np.array(self.quat, dtype=float)


def compute_footprint(env, body_id: str, cell_size: float,
                      clear: float = _DEFAULT_CLEAR,
                      fill_step: float = _FILL_STEP) -> ObjectFootprint:
    """Measure ``body_id``'s overlap footprint on a ``cell_size`` grid."""
    bid = env.get_object_id(body_id)
    pts = _collision_points_bodyframe(env, bid)
    xy = pts[:, :2]
    bottom = float(pts[:, 2].min())

    # canonical yaw: align the narrower horizontal extent with world-x
    ext_x = float(np.ptp(xy[:, 0]))
    ext_y = float(np.ptp(xy[:, 1]))
    yaw = 0.0 if ext_x <= ext_y else np.pi / 2
    c, s = np.cos(yaw), np.sin(yaw)
    pxy = xy @ np.array([[c, -s], [s, c]]).T

    lo = pxy.min(0)
    hi = pxy.max(0)
    cx, cy = (lo + hi) / 2
    wx, wy = hi - lo

    dx = max(1, int(np.ceil((wx + 2 * clear) / cell_size)))
    dy = max(1, int(np.ceil((wy + 2 * clear) / cell_size)))

    centered = pxy - [cx, cy]
    solid = _fill_hull(_convex_hull(centered), fill_step)

    h = cell_size / 2
    offsets: List[Tuple[int, int]] = []
    for jx in range(dx):
        ccx = (jx - (dx - 1) / 2) * cell_size
        for jy in range(dy):
            ccy = (jy - (dy - 1) / 2) * cell_size
            # distance from each solid point to this cell square (0 inside)
            ox = np.clip(np.abs(solid[:, 0] - ccx) - h, 0, None)
            oy = np.clip(np.abs(solid[:, 1] - ccy) - h, 0, None)
            if (np.hypot(ox, oy) <= clear).any():
                offsets.append((jx, jy))

    key = f"p{dx}x{dy}c{len(offsets)}"
    quat = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))
    return ObjectFootprint(body_id, dx, dy, tuple(offsets), float(yaw),
                           quat, float(cx), float(cy), bottom, key)


def compute_all_footprints(env, object_ids: Sequence[str], cell_size: float,
                           **kw) -> Dict[str, ObjectFootprint]:
    """Footprints for every object, with schema keys disambiguated.

    Two objects whose offset patterns differ but collide on the
    ``pNxMcK`` key get a ``_<n>`` suffix so each distinct pattern maps to
    exactly one schema and each schema to one pattern.
    """
    fps: Dict[str, ObjectFootprint] = {}
    key_to_pattern: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for oid in object_ids:
        fp = compute_footprint(env, oid, cell_size, **kw)
        base = fp.key
        suffix = 0
        while base in key_to_pattern and key_to_pattern[base] != fp.offsets:
            suffix += 1
            base = f"{fp.key}_{suffix}"
        if base != fp.key:
            fp = ObjectFootprint(fp.body_id, fp.dx, fp.dy, fp.offsets, fp.yaw,
                                 fp.quat, fp.cx, fp.cy, fp.bottom, base)
        key_to_pattern[base] = fp.offsets
        fps[oid] = fp
    return fps


def anchor_of_cells(cell_ids: Sequence[str]) -> Cell:
    """South-west anchor (min ix, min iy) of a set of cell ids.

    All cells must share one region.  Used to recover an object's anchor
    from its ``(occupied …)`` facts.
    """
    if not cell_ids:
        raise ValueError("no cells")
    cells = [Cell.parse(c) for c in cell_ids]
    region = cells[0].region
    if any(c.region != region for c in cells):
        raise ValueError(f"cells span multiple regions: {cell_ids}")
    return Cell(region, min(c.ix for c in cells), min(c.iy for c in cells))


def occupied_cells_from_pose(env, region: GridRegion, fp: ObjectFootprint,
                             obj_pos) -> List[Cell]:
    """Cells covered by ``fp`` given the object's current body position.

    The cross-section centre is ``body_xy + (cx, cy)`` (at the canonical
    yaw the offset is world-aligned); the SW anchor sits half a block
    south-west of it.  Returns ``[]`` if the anchor is off-grid / excluded.
    """
    import numpy as np
    cs = region.cell_size
    cxw = float(obj_pos[0]) + fp.cx
    cyw = float(obj_pos[1]) + fp.cy
    anchor_x = cxw - (fp.dx - 1) / 2 * cs
    anchor_y = cyw - (fp.dy - 1) / 2 * cs
    anchor = region.cell_for(anchor_x, anchor_y, region.level_z)
    if anchor is None:
        return []
    return fp.cells_at(anchor)


def ascii_pattern(fp: ObjectFootprint) -> str:
    """Small ASCII art of the occupied (#) vs free (.) bounding cells."""
    occ = set(fp.offsets)
    rows = []
    for jy in range(fp.dy - 1, -1, -1):   # north (high iy) on top
        rows.append("".join("#" if (jx, jy) in occ else "." for jx in range(fp.dx)))
    return "\n".join(rows)
