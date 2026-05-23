"""DomainBridge wiring for the redesigned multi-level wooden-blocks domain.

Cells have FIXED world positions (computed from the workspace grid origins
and `level_z`).  A block's PDDL-cell assignment is derived from its MuJoCo
world pose by:

* Computing every cell the block's bounding footprint covers, given its
  shape (cube or oblong) and current orientation (inferred from quat).
* For 2×1 oblong blocks, returning the two cells the block spans.

Held blocks are tracked via the held-cube / held-flat-x / held-flat-y /
held-upright fluents (action-maintained); when held, the block isn't at any
cell.

Action executors (MP side) are deferred to a follow-up commit; this module
provides the symbolic plumbing — predicate evaluation, fluent tracking, PDDL
problem assembly — so the planner can be exercised end-to-end against a
running env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tampanda.symbolic.workspace import Cell, GridRegion, Workspace
from tampanda.tamp import DomainBridge

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    oblong_block_name,
    stack_region_name,
)


_PDDL_PATH = Path(__file__).parent / "pddl" / "domain.pddl"

# Tolerance (m) for pose-to-cell matching.  A block sits "in" a cell if its
# centroid lies within ``cube_half`` of the cell's world centre along each
# axis.
_XY_TOL_FACTOR = 0.9   # use 0.9 * cube_half (= 90% of the half-edge) as max
                         # allowable offset; loose enough for sim settling drift
                         # but tight enough to disambiguate neighbouring cells
_Z_TOL_FACTOR = 0.9


# ---------------------------------------------------------------------------
# Pose <-> cell geometry helpers
# ---------------------------------------------------------------------------


def _cell_world_centre(workspace: Workspace, cell: Cell
                          ) -> Tuple[float, float, float]:
    """World-frame centre of ``cell`` (the same as ``workspace.pose_for``)."""
    return workspace.pose_for(cell)


def _orientation_from_quat(quat: np.ndarray) -> str:
    """Identify oblong orientation from its world quaternion.

    The default oblong asset has its long axis along world-x at quat
    ``(1, 0, 0, 0)``.  Yaw 90 deg around z → long axis along y (flat-y).
    Pitch 90 deg around y → long axis along z (upright).

    Returns one of ``"flat-x"``, ``"flat-y"``, ``"upright"``, or
    ``"unknown"`` if no canonical orientation matches.
    """
    q = np.asarray(quat, dtype=float)
    # Rotate the block's local x-axis (its long axis) by the quaternion.
    # Standard quat-to-rotmat row 0 (rotation applied to [1, 0, 0]):
    w, x, y, z = q
    rx = 1 - 2 * (y * y + z * z)
    ry = 2 * (x * y + z * w)
    rz = 2 * (x * z - y * w)
    long_axis = np.array([rx, ry, rz])
    abs_axis = np.abs(long_axis)
    primary = int(np.argmax(abs_axis))
    if abs_axis[primary] < 0.85:
        return "unknown"
    return {0: "flat-x", 1: "flat-y", 2: "upright"}[primary]


def _block_cells(env, workspace: Workspace, name: str,
                   shape: str, cube_half: float) -> List[Cell]:
    """Return the cells block ``name`` currently occupies.

    Held blocks (parked off-workspace) return an empty list.  Blocks
    outside any region return an empty list.

    For oblong blocks, the orientation is derived from the world quat.
    """
    pos, quat = env.get_object_pose(name)
    pos = np.asarray(pos, dtype=float)
    if pos[0] > 50.0:   # sentinel x; matches MultilevelBlocksConfig.hide_far_x
        return []

    if shape == "cube":
        c = _cell_at(workspace, pos, cube_half)
        return [c] if c is not None else []

    # Oblong / long: orientation from quat
    orientation = _orientation_from_quat(quat)
    if orientation == "unknown":
        return []

    if shape == "long":
        # 3×1 long: cell pitch = cube_size; cells at centroid ± cube_size
        # along the long axis (the middle cell is at centroid).
        step = 2 * cube_half  # = cube_size
        if orientation == "flat-x":
            offsets = [np.array([-step, 0, 0]),
                          np.array([0, 0, 0]),
                          np.array([+step, 0, 0])]
        elif orientation == "flat-y":
            offsets = [np.array([0, -step, 0]),
                          np.array([0, 0, 0]),
                          np.array([0, +step, 0])]
        elif orientation == "upright":
            offsets = [np.array([0, 0, -step]),
                          np.array([0, 0, 0]),
                          np.array([0, 0, +step])]
        else:
            return []
        cells = [_cell_at(workspace, pos + o, cube_half) for o in offsets]
        return [c for c in cells if c is not None]

    # Oblong (2×1): two cells at centroid ± half_long along the long axis.
    long_dim = 2 * cube_half  # cell pitch == cube_size
    half_long = long_dim / 2
    if orientation == "flat-x":
        c1 = _cell_at(workspace, pos + np.array([-half_long, 0, 0]), cube_half)
        c2 = _cell_at(workspace, pos + np.array([+half_long, 0, 0]), cube_half)
    elif orientation == "flat-y":
        c1 = _cell_at(workspace, pos + np.array([0, -half_long, 0]), cube_half)
        c2 = _cell_at(workspace, pos + np.array([0, +half_long, 0]), cube_half)
    elif orientation == "upright":
        c1 = _cell_at(workspace, pos + np.array([0, 0, -half_long]), cube_half)
        c2 = _cell_at(workspace, pos + np.array([0, 0, +half_long]), cube_half)
    else:
        return []
    return [c for c in (c1, c2) if c is not None]


def _cell_at(workspace: Workspace, pos: np.ndarray,
               cube_half: float) -> Optional[Cell]:
    """Locate the cell whose centre is within tolerance of ``pos``.

    Returns ``None`` if no region's level_z and (ix, iy) extent match.
    """
    xy_tol = cube_half * _XY_TOL_FACTOR
    z_tol = cube_half * _Z_TOL_FACTOR
    for region_name in workspace.regions:
        region = workspace[region_name]
        if abs(pos[2] - region.level_z) > z_tol:
            continue
        ox, oy = region.origin
        ex, ey = region.extent
        if pos[0] < ox - xy_tol or pos[0] > ox + ex + xy_tol:
            continue
        if pos[1] < oy - xy_tol or pos[1] > oy + ey + xy_tol:
            continue
        # Index into the grid
        ix = int(round((pos[0] - ox - region.cell_size / 2) / region.cell_size))
        iy = int(round((pos[1] - oy - region.cell_size / 2) / region.cell_size))
        if not region.in_bounds(ix, iy):
            continue
        cell = Cell(region_name, ix, iy)
        cx, cy, cz = workspace.pose_for(cell)
        if (abs(cx - pos[0]) < xy_tol
                and abs(cy - pos[1]) < xy_tol
                and abs(cz - pos[2]) < z_tol):
            return cell
    return None


# ---------------------------------------------------------------------------
# Static adjacency helpers
# ---------------------------------------------------------------------------


def _parse_cell_id(cell_id: str) -> Optional[Cell]:
    try:
        return Cell.parse(cell_id)
    except Exception:
        return None


def _cells_above(cell: Cell, workspace: Workspace) -> Optional[Cell]:
    """The cell directly above ``cell``, or None if ``cell`` is at the top
    of its column."""
    if not cell.region.startswith("stack_L"):
        return None
    level = int(cell.region.split("_L")[1])
    if level + 1 >= 5:
        return None
    higher = Cell(stack_region_name(level + 1), cell.ix, cell.iy)
    if higher.region in workspace.regions:
        return higher
    return None


def _cell_east(cell: Cell, workspace: Workspace) -> Optional[Cell]:
    region = workspace[cell.region]
    if region.in_bounds(cell.ix + 1, cell.iy):
        return Cell(cell.region, cell.ix + 1, cell.iy)
    return None


def _cell_north(cell: Cell, workspace: Workspace) -> Optional[Cell]:
    region = workspace[cell.region]
    if region.in_bounds(cell.ix, cell.iy + 1):
        return Cell(cell.region, cell.ix, cell.iy + 1)
    return None


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------


def make_multilevel_blocks_bridge(
    env,
    workspace: Workspace,
    config: MultilevelBlocksConfig,
    executor=None,
) -> Tuple[DomainBridge, Dict[str, List[str]]]:
    """Wire up the multilevel-blocks bridge.

    Returns ``(bridge, objects)`` where ``objects`` is the
    ``{type_name: [object_name, ...]}`` dict expected by
    ``bridge.ground_state`` / ``bridge.plan``.  ``objects`` contains every
    block and every reachable cell of the workspace.

    Action executors are NOT registered in this commit — the bridge is
    plan-ready but execution is deferred.
    """
    bridge = DomainBridge(_PDDL_PATH, env, strict_preconditions=False)

    # If an executor is provided, wire its action methods into the bridge.
    # The executor handles the MP side of pick/put/transform; bridge fluents
    # are updated via the (success, fluent_delta) tuple each handler returns.
    if executor is not None:
        from tampanda.symbolic.domains.multilevel_blocks.executor import (
            register_executor,
        )
        register_executor(bridge, executor)

    # ------------------------------------------------------------------
    # Object set
    # ------------------------------------------------------------------
    block_names: List[str] = []
    for i in range(config.n_cubes):
        block_names.append(cube_block_name(i))
    for i in range(config.n_oblong):
        block_names.append(oblong_block_name(i))
    for i in range(config.n_long):
        block_names.append(long_block_name(i))

    cell_names: List[str] = []
    for region_name in workspace.regions:
        region = workspace[region_name]
        for cell in region.cells():
            cell_names.append(cell.id)

    objects = {"block": block_names, "cell": cell_names}

    # ------------------------------------------------------------------
    # Static block-shape predicates
    # ------------------------------------------------------------------
    cube_set = {cube_block_name(i) for i in range(config.n_cubes)}
    oblong_set = {oblong_block_name(i) for i in range(config.n_oblong)}
    long_set = {long_block_name(i) for i in range(config.n_long)}

    @bridge.predicate("cube")
    def _is_cube(env, fluents, b):
        return b in cube_set

    @bridge.predicate("oblong")
    def _is_oblong(env, fluents, b):
        return b in oblong_set

    @bridge.predicate("long")
    def _is_long(env, fluents, b):
        return b in long_set

    # ------------------------------------------------------------------
    # Static cell-region predicates
    # ------------------------------------------------------------------
    @bridge.predicate("in-parts")
    def _in_parts(env, fluents, c):
        return c.startswith("parts__")

    @bridge.predicate("in-stack")
    def _in_stack(env, fluents, c):
        return c.startswith("stack_L")

    # ------------------------------------------------------------------
    # Static directional adjacency
    # ------------------------------------------------------------------
    @bridge.predicate("above")
    def _above(env, fluents, c_low, c_up):
        cl = _parse_cell_id(c_low)
        cu = _parse_cell_id(c_up)
        if cl is None or cu is None:
            return False
        expected = _cells_above(cl, workspace)
        return expected is not None and expected.id == c_up

    @bridge.predicate("east-of")
    def _east_of(env, fluents, c1, c2):
        a = _parse_cell_id(c1)
        b = _parse_cell_id(c2)
        if a is None or b is None:
            return False
        expected = _cell_east(a, workspace)
        return expected is not None and expected.id == c2

    @bridge.predicate("north-of")
    def _north_of(env, fluents, c1, c2):
        a = _parse_cell_id(c1)
        b = _parse_cell_id(c2)
        if a is None or b is None:
            return False
        expected = _cell_north(a, workspace)
        return expected is not None and expected.id == c2

    # ------------------------------------------------------------------
    # Dynamic predicates: in, empty
    # ------------------------------------------------------------------
    cube_half = config.cube_half_extent

    def _is_held(name: str, fluents: Dict) -> bool:
        return any(
            fluents.get((p, name), False)
            for p in ("held-cube", "held-flat-x",
                       "held-flat-y", "held-upright")
        )

    def _block_shape(name: str) -> str:
        if name in cube_set:
            return "cube"
        if name in long_set:
            return "long"
        return "oblong"

    @bridge.predicate("in")
    def _eval_in(env, fluents, b, c):
        if _is_held(b, fluents):
            return False
        cells = _block_cells(env, workspace, b, _block_shape(b), cube_half)
        return any(cell.id == c for cell in cells)

    @bridge.predicate("empty")
    def _eval_empty(env, fluents, c):
        # A cell is empty iff no block occupies it.
        for name in block_names:
            if _is_held(name, fluents):
                continue
            cells = _block_cells(env, workspace, name,
                                    _block_shape(name), cube_half)
            if any(cell.id == c for cell in cells):
                return False
        return True

    # ------------------------------------------------------------------
    # Fluents (action-tracked)
    # ------------------------------------------------------------------
    bridge.fluent("gripper-empty", initial=True)
    bridge.fluent("held-cube")
    bridge.fluent("held-flat-x")
    bridge.fluent("held-flat-y")
    bridge.fluent("held-upright")

    return bridge, objects
