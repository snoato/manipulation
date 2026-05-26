"""Buildable structure templates for the multilevel_blocks data-gen pipeline.

Each template returns a :class:`Template` describing:

* ``goal_placements`` — ``(block_name, [cell_ids], target_orientation)``
  triples saying where each block ends up on the stack table.
* ``source_placements`` — ``(block_name, [cell_ids], source_orientation)``
  triples saying where each block STARTS on the parts table.
* ``build_order`` — list of block names in dependency-respecting order
  (lower-level supports first, then bridges).

Orientation strings: ``"flat-x"``, ``"flat-y"``, ``"upright"`` (matches
the bridge's :func:`_orientation_from_quat` output and the executor's
pick/put action names).

Templates available:

* :func:`cube_tower(h, ix, iy)` — vertical stack of ``h`` cubes at
  ``(ix, iy)`` spanning L0..L(h-1).
* :func:`oblong_tower(h, ix, iy)` — vertical stack of ``h`` flat
  oblongs at the (ix, ix+1)×iy footprint.
* :func:`upright_bridges(ix, iy)` — 4 uprights at the 3×3 corners +
  2 long-y bridges at the top.  Mirrors the stacking_test template.
* :func:`long_pyramid(ix, iy)` — long-x at L0, flat oblong at L1,
  cube at L2.
* :func:`random_mix(rng, ...)` — random selection of the above with
  random parameters.

Sources for templates are laid out in a compact 1-cell-gap parts kit
(per the dense-parts spacing now allowed by the gripper pre-close fix).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
    cube_block_name,
    long_block_name,
    oblong_block_name,
    stack_region_name,
)


_Placement = Tuple[str, List[str], str]   # (block, [cell_ids], orientation)


@dataclass(frozen=True)
class Template:
    """Buildable structure spec.

    ``goal_placements`` and ``source_placements`` are aligned on
    ``block_name`` (one entry per block in each list).  ``build_order``
    is a list of block names — the planner walks them in this order,
    issuing pick → [transform] → put for each.
    """
    name: str
    goal_placements: List[_Placement]
    source_placements: List[_Placement]
    build_order: List[str]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parts-kit allocator
# ---------------------------------------------------------------------------


class _PartsAllocator:
    """Doles out non-overlapping parts cells for source blocks.

    Layout strategy: place blocks in rows of 2-cell gap (= 1 empty cell)
    in iy.  Within a row, advance in ix by (footprint_x + 1) cells.  The
    gripper pre-close fix supports 1-cell-gap density.  Oblongs and
    longs are placed FLAT-X by default (long axis along world-x).
    """

    def __init__(self, cfg: MultilevelBlocksConfig):
        self.cells_x, self.cells_y = cfg.parts_grid_cells
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._row_height: int = 1   # one cell tall per row

    def _advance_row(self, new_height: int) -> None:
        # row+1 empty buffer then continue
        self._cursor_y += self._row_height + 1
        self._cursor_x = 0
        self._row_height = new_height
        if self._cursor_y + new_height > self.cells_y:
            raise RuntimeError(
                "parts grid exhausted; reduce block count or enlarge "
                "parts_grid_cells"
            )

    def alloc(self, footprint_x: int, footprint_y: int = 1) -> List[Tuple[int, int]]:
        """Allocate a footprint_x × footprint_y region; return cell coords."""
        if footprint_y != 1 and self._cursor_x == 0:
            # promote row height when first block of row is taller
            self._row_height = footprint_y
        if self._cursor_x + footprint_x > self.cells_x:
            self._advance_row(footprint_y)
        elif self._row_height < footprint_y:
            self._advance_row(footprint_y)
        # claim region
        cells = []
        for dy in range(footprint_y):
            for dx in range(footprint_x):
                cells.append((self._cursor_x + dx, self._cursor_y + dy))
        self._cursor_x += footprint_x + 1   # 1-cell gap to next
        return cells


# ---------------------------------------------------------------------------
# Stack-cell helpers
# ---------------------------------------------------------------------------


def _stack(level: int, ix: int, iy: int) -> str:
    return f"{stack_region_name(level)}__{ix}_{iy}"


def _parts(ix: int, iy: int) -> str:
    return f"parts__{ix}_{iy}"


def _cells_xy_to_ids(cells: List[Tuple[int, int]], region: str = "parts"
                       ) -> List[str]:
    return [f"{region}__{ix}_{iy}" for ix, iy in cells]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def cube_tower(h: int, ix: int, iy: int, *, cfg: Optional[MultilevelBlocksConfig] = None,
                  start_idx: int = 0) -> Template:
    """Vertical stack of ``h`` cubes at ``(ix, iy)`` spanning L0..L(h-1)."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if h > cfg.stack_grid_cells[2]:
        raise ValueError(f"h={h} exceeds stack height {cfg.stack_grid_cells[2]}")

    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []
    for k in range(h):
        block = cube_block_name(start_idx + k)
        order.append(block)
        # goal cell on stack
        goal.append((block, [_stack(k, ix, iy)], "flat-x"))
        # source cell on parts
        s_cells = alloc.alloc(1, 1)
        src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))
    return Template(
        name=f"cube_tower_h{h}",
        goal_placements=goal,
        source_placements=src,
        build_order=order,
        metadata={"h": h, "anchor": (ix, iy)},
    )


def oblong_tower(h: int, ix: int, iy: int, *,
                    cfg: Optional[MultilevelBlocksConfig] = None,
                    flat_orientation: str = "flat-x",
                    start_idx: int = 0) -> Template:
    """``h`` flat oblongs stacked at footprint (ix, ix+1)x(iy) (flat-x)
    or (ix)x(iy, iy+1) (flat-y).  Each oblong spans 2 cells at one level."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    if h < 1 or h > cfg.stack_grid_cells[2]:
        raise ValueError(f"invalid h={h}")
    if flat_orientation not in ("flat-x", "flat-y"):
        raise ValueError(f"flat_orientation must be flat-x or flat-y")

    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []
    for k in range(h):
        block = oblong_block_name(start_idx + k)
        order.append(block)
        if flat_orientation == "flat-x":
            goal_cells = [_stack(k, ix, iy), _stack(k, ix + 1, iy)]
        else:  # flat-y
            goal_cells = [_stack(k, ix, iy), _stack(k, ix, iy + 1)]
        goal.append((block, goal_cells, flat_orientation))
        # Source: flat-x on parts (default orientation).  If target is
        # flat-y, plan will insert a turn-x-to-y in-hand transform.
        s_cells = alloc.alloc(2, 1)
        src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))
    return Template(
        name=f"oblong_tower_h{h}_{flat_orientation}",
        goal_placements=goal,
        source_placements=src,
        build_order=order,
        metadata={"h": h, "anchor": (ix, iy),
                       "flat_orientation": flat_orientation},
    )


def upright_bridges(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       oblong_start: int = 0,
                       long_start: int = 0) -> Template:
    """4 upright oblongs at L0+L1 corners + 2 long-y bridges at L2.

    Footprint: 3×3 at (ix..ix+2, iy..iy+2).  Uprights at the 4 corners;
    bridges span y at columns ix and ix+2.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    if cfg.stack_grid_cells[2] < 3:
        raise ValueError("upright_bridges needs >=3 stack levels")

    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []

    # Uprights: BACK row first (iy+2 in stack), then front (iy).  This is
    # the order stacking_test confirmed works.
    upright_positions = [
        (ix, iy + 2),       # back-left
        (ix + 2, iy + 2),   # back-right
        (ix, iy),           # front-left
        (ix + 2, iy),       # front-right
    ]
    for k, (sx, sy) in enumerate(upright_positions):
        block = oblong_block_name(oblong_start + k)
        order.append(block)
        goal.append((block, [_stack(0, sx, sy), _stack(1, sx, sy)],
                          "upright"))
        # Source flat-x on parts.
        s_cells = alloc.alloc(2, 1)
        src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))

    # Long-y bridges at L2 over the upright columns.  Two bridges along
    # x = ix and x = ix+2, spanning y in [iy..iy+2].
    bridge_xs = [ix, ix + 2]
    for k, bx in enumerate(bridge_xs):
        block = long_block_name(long_start + k)
        order.append(block)
        goal_cells = [_stack(2, bx, iy), _stack(2, bx, iy + 1),
                          _stack(2, bx, iy + 2)]
        goal.append((block, goal_cells, "flat-y"))
        # Source flat-x on parts (3 cells along x).
        s_cells = alloc.alloc(3, 1)
        src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))

    return Template(
        name=f"upright_bridges_anchor({ix},{iy})",
        goal_placements=goal,
        source_placements=src,
        build_order=order,
        metadata={"anchor": (ix, iy)},
    )


def long_pyramid(ix: int, iy: int, *,
                   cfg: Optional[MultilevelBlocksConfig] = None,
                   long_start: int = 0,
                   oblong_start: int = 0,
                   cube_start: int = 0) -> Template:
    """3-2-1 pyramid: long-x at L0 (3 cells), flat-x oblong at L1
    (2 cells, left-aligned), cube at L2 (1 cell, left-aligned).

    Plan length: 3 picks + 3 puts = 6 actions (no transforms).
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    if cfg.stack_grid_cells[2] < 3:
        raise ValueError("long_pyramid needs >=3 stack levels")
    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []

    # L0: long-x at (ix..ix+2, iy)
    long_b = long_block_name(long_start)
    order.append(long_b)
    goal.append((long_b,
                       [_stack(0, ix, iy), _stack(0, ix + 1, iy), _stack(0, ix + 2, iy)],
                       "flat-x"))
    s_cells = alloc.alloc(3, 1)
    src.append((long_b, _cells_xy_to_ids(s_cells), "flat-x"))

    # L1: oblong flat-x at (ix..ix+1, iy)
    obl_b = oblong_block_name(oblong_start)
    order.append(obl_b)
    goal.append((obl_b, [_stack(1, ix, iy), _stack(1, ix + 1, iy)], "flat-x"))
    s_cells = alloc.alloc(2, 1)
    src.append((obl_b, _cells_xy_to_ids(s_cells), "flat-x"))

    # L2: cube at (ix, iy)
    cube_b = cube_block_name(cube_start)
    order.append(cube_b)
    goal.append((cube_b, [_stack(2, ix, iy)], "flat-x"))
    s_cells = alloc.alloc(1, 1)
    src.append((cube_b, _cells_xy_to_ids(s_cells), "flat-x"))

    return Template(
        name=f"long_pyramid_anchor({ix},{iy})",
        goal_placements=goal,
        source_placements=src,
        build_order=order,
        metadata={"anchor": (ix, iy)},
    )


def random_mix(rng: np.random.Generator,
                  cfg: Optional[MultilevelBlocksConfig] = None,
                  *, ix_range: Tuple[int, int] = (2, 5),
                  iy_range: Tuple[int, int] = (2, 5)) -> Template:
    """Random selection of cube_tower / oblong_tower / upright_bridges /
    long_pyramid with random parameters.

    Anchor cell is sampled inside the per-template safe range; height
    is sampled within the template's valid range.  Returns a single
    template instance with metadata describing what was sampled.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    cells_x = cfg.stack_grid_cells[0]
    cells_y = cfg.stack_grid_cells[1]
    cells_z = cfg.stack_grid_cells[2]

    choice = rng.choice(["cube_tower", "oblong_tower", "upright_bridges",
                              "long_pyramid"])
    ix = int(rng.integers(ix_range[0], min(ix_range[1], cells_x - 4) + 1))
    iy = int(rng.integers(iy_range[0], min(iy_range[1], cells_y - 4) + 1))

    if choice == "cube_tower":
        h = int(rng.integers(2, min(cells_z, 4) + 1))
        return cube_tower(h, ix, iy, cfg=cfg)
    if choice == "oblong_tower":
        h = int(rng.integers(2, min(cells_z, 3) + 1))
        flat = rng.choice(["flat-x", "flat-y"])
        return oblong_tower(h, ix, iy, cfg=cfg, flat_orientation=str(flat))
    if choice == "upright_bridges":
        return upright_bridges(ix, iy, cfg=cfg)
    if choice == "long_pyramid":
        return long_pyramid(ix, iy, cfg=cfg)
    raise RuntimeError(f"unreachable: {choice}")


# ---------------------------------------------------------------------------
# Curriculum-extension templates (L0, L4, L5)
# ---------------------------------------------------------------------------


def cube_pick_put(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       start_idx: int = 0) -> Template:
    """L0: a single cube placed at L0 (ix, iy).  Plan = 2 actions."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = cube_block_name(start_idx)
    s_cells = alloc.alloc(1, 1)
    return Template(
        name=f"cube_pick_put_anchor({ix},{iy})",
        goal_placements=[(block, [_stack(0, ix, iy)], "flat-x")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 0, "anchor": (ix, iy)},
    )


# ---------------------------------------------------------------------------
# Primitive-exposure templates — single-block "lessons" that teach a model
# the L4 motion primitives (put-upright, make-upright-from-x, put-long-x,
# put-long-y, turn-long-x-to-y) in isolation, in 2-3 action problems.
# Added at L0 / L1 so the training distribution covers these primitives
# at low difficulty before they appear inside L4's denser compositions.
# ---------------------------------------------------------------------------


def simple_long_x(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       start_idx: int = 0) -> Template:
    """L0: a single long block placed flat-x at (ix..ix+2, iy).

    Plan = 2 actions: pick-long-x, put-long-x.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(start_idx)
    s_cells = alloc.alloc(3, 1)
    return Template(
        name=f"simple_long_x_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix,   iy),
                                  _stack(0, ix+1, iy),
                                  _stack(0, ix+2, iy)],
                                 "flat-x")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 0, "anchor": (ix, iy)},
    )


def simple_upright_pickput(ix: int, iy: int, *,
                                 cfg: Optional[MultilevelBlocksConfig] = None,
                                 oblong_start: int = 0) -> Template:
    """L1: a single oblong picked flat-x, rotated upright, placed at
    (ix, iy) spanning L0 + L1.

    Plan = 3 actions: pick-flat-x, make-upright-from-x, put-upright.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = oblong_block_name(oblong_start)
    s_cells = alloc.alloc(2, 1)
    return Template(
        name=f"simple_upright_pickput_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy), _stack(1, ix, iy)],
                                 "upright")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_long_rotate(ix: int, iy: int, *,
                            cfg: Optional[MultilevelBlocksConfig] = None,
                            long_start: int = 0) -> Template:
    """L1: a single long block picked flat-x, rotated to flat-y, placed
    along y at (ix, iy..iy+2).

    Plan = 3 actions: pick-long-x, turn-long-x-to-y, put-long-y.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(long_start)
    s_cells = alloc.alloc(3, 1)
    return Template(
        name=f"simple_long_rotate_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy),
                                  _stack(0, ix, iy+1),
                                  _stack(0, ix, iy+2)],
                                 "flat-y")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


# ===========================================================================
# Extended primitive-exposure templates (L0/L1/L2)
# ===========================================================================
# Cover the 17 actions not exposed by the first 3 simple_* templates:
#   * Group A — y-axis symmetric variants (pick-flat-y, pick-long-y,
#     turn-y-to-x, turn-long-y-to-x, make-upright-from-y,
#     make-long-upright-from-{x,y})
#   * Group B — basic put-flat-{x,y} as 2-action problems
#   * Group C — pick-from-stack reverse primitives (pick-upright,
#     pick-long-upright, make-flat-{x,y}-from-upright,
#     make-long-flat-{x,y}-from-upright, put-long-upright)
#
# All single-block 2-3 action templates.  Most use ``alloc(1, N)`` for
# flat-y sources; "pick-from-stack" templates skip parts allocator
# entirely (source cells are stack cells with multi-cell orientation).
# ===========================================================================


def simple_oblong_flat_x(ix: int, iy: int, *,
                              cfg: Optional[MultilevelBlocksConfig] = None,
                              oblong_start: int = 0) -> Template:
    """L0: oblong picked flat-x, placed flat-x at (ix..ix+1, iy).
    Plan = 2 actions: pick-flat-x, put-flat-x."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = oblong_block_name(oblong_start)
    s_cells = alloc.alloc(2, 1)
    return Template(
        name=f"simple_oblong_flat_x_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy), _stack(0, ix+1, iy)],
                                 "flat-x")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 0, "anchor": (ix, iy)},
    )


def simple_oblong_flat_y(ix: int, iy: int, *,
                              cfg: Optional[MultilevelBlocksConfig] = None,
                              oblong_start: int = 0) -> Template:
    """L0: oblong picked flat-y (source cells along y), placed flat-y
    at (ix, iy..iy+1).  Plan = 2 actions: pick-flat-y, put-flat-y."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = oblong_block_name(oblong_start)
    s_cells = alloc.alloc(1, 2)   # 1 wide × 2 tall (y direction)
    return Template(
        name=f"simple_oblong_flat_y_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy), _stack(0, ix, iy+1)],
                                 "flat-y")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 0, "anchor": (ix, iy)},
    )


def simple_long_y(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       long_start: int = 0) -> Template:
    """L0: long picked flat-y, placed flat-y at (ix, iy..iy+2).
    Plan = 2 actions: pick-long-y, put-long-y."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(long_start)
    s_cells = alloc.alloc(1, 3)
    return Template(
        name=f"simple_long_y_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy),
                                  _stack(0, ix, iy+1),
                                  _stack(0, ix, iy+2)],
                                 "flat-y")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 0, "anchor": (ix, iy)},
    )


def simple_pick_upright_then_put_upright(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        oblong_start: int = 0) -> Template:
    """L0: oblong starts upright at stack (ix, iy)+(L0,L1); picked,
    placed upright at stack (ix2, iy2)+(L0,L1).  Source ≠ Goal cells.
    Plan = 2 actions: pick-upright, put-upright."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = oblong_block_name(oblong_start)
    return Template(
        name=f"simple_pick_upright_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2), _stack(1, ix2, iy2)],
                                 "upright")],
        source_placements=[(block,
                                  [_stack(0, ix, iy), _stack(1, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 0,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def simple_pick_long_upright_then_put_long_upright(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        long_start: int = 0) -> Template:
    """L0: long starts upright at stack (ix,iy)+(L0,L1,L2); picked,
    placed upright at (ix2,iy2)+(L0,L1,L2).  Plan = 2 actions:
    pick-long-upright, put-long-upright."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = long_block_name(long_start)
    return Template(
        name=f"simple_pick_long_upright_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2),
                                  _stack(1, ix2, iy2),
                                  _stack(2, ix2, iy2)],
                                 "upright")],
        source_placements=[(block,
                                  [_stack(0, ix, iy),
                                   _stack(1, ix, iy),
                                   _stack(2, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 0,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def simple_oblong_rotate_y_to_x(ix: int, iy: int, *,
                                       cfg: Optional[MultilevelBlocksConfig] = None,
                                       oblong_start: int = 0) -> Template:
    """L1: oblong picked flat-y, rotated to flat-x, placed flat-x at
    (ix..ix+1, iy).  Plan = 3 actions:
    pick-flat-y, turn-y-to-x, put-flat-x."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = oblong_block_name(oblong_start)
    s_cells = alloc.alloc(1, 2)
    return Template(
        name=f"simple_oblong_rotate_y_to_x_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy), _stack(0, ix+1, iy)],
                                 "flat-x")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_long_rotate_y_to_x(ix: int, iy: int, *,
                                     cfg: Optional[MultilevelBlocksConfig] = None,
                                     long_start: int = 0) -> Template:
    """L1: long picked flat-y, rotated to flat-x, placed flat-x along
    (ix..ix+2, iy).  Plan = 3 actions: pick-long-y, turn-long-y-to-x,
    put-long-x."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(long_start)
    s_cells = alloc.alloc(1, 3)
    return Template(
        name=f"simple_long_rotate_y_to_x_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy),
                                  _stack(0, ix+1, iy),
                                  _stack(0, ix+2, iy)],
                                 "flat-x")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_make_upright_from_y(ix: int, iy: int, *,
                                      cfg: Optional[MultilevelBlocksConfig] = None,
                                      oblong_start: int = 0) -> Template:
    """L1: oblong picked flat-y, rotated upright via the y-axis variant,
    placed upright at stack (ix,iy)+(L0,L1).  Plan = 3 actions:
    pick-flat-y, make-upright-from-y, put-upright."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = oblong_block_name(oblong_start)
    s_cells = alloc.alloc(1, 2)
    return Template(
        name=f"simple_make_upright_from_y_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy), _stack(1, ix, iy)],
                                 "upright")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_make_long_upright_from_x(ix: int, iy: int, *,
                                            cfg: Optional[MultilevelBlocksConfig] = None,
                                            long_start: int = 0) -> Template:
    """L1: long picked flat-x, rotated upright (long-upright via x),
    placed upright at stack (ix,iy)+(L0,L1,L2).  Plan = 3 actions:
    pick-long-x, make-long-upright-from-x, put-long-upright."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(long_start)
    s_cells = alloc.alloc(3, 1)
    return Template(
        name=f"simple_make_long_upright_from_x_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy),
                                  _stack(1, ix, iy),
                                  _stack(2, ix, iy)],
                                 "upright")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-x")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_make_long_upright_from_y(ix: int, iy: int, *,
                                            cfg: Optional[MultilevelBlocksConfig] = None,
                                            long_start: int = 0) -> Template:
    """L1: long picked flat-y, rotated upright (long-upright via y),
    placed upright at stack (ix,iy)+(L0,L1,L2).  Plan = 3 actions:
    pick-long-y, make-long-upright-from-y, put-long-upright."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    alloc = _PartsAllocator(cfg)
    block = long_block_name(long_start)
    s_cells = alloc.alloc(1, 3)
    return Template(
        name=f"simple_make_long_upright_from_y_anchor({ix},{iy})",
        goal_placements=[(block,
                                 [_stack(0, ix, iy),
                                  _stack(1, ix, iy),
                                  _stack(2, ix, iy)],
                                 "upright")],
        source_placements=[(block,
                                  _cells_xy_to_ids(s_cells), "flat-y")],
        build_order=[block],
        metadata={"difficulty": 1, "anchor": (ix, iy)},
    )


def simple_unstack_oblong_to_flat_x(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        oblong_start: int = 0) -> Template:
    """L2: oblong starts upright at stack (ix,iy)+(L0,L1); picked,
    rotated to flat-x in hand, placed flat-x at (ix2..ix2+1, iy2).
    Plan = 3 actions: pick-upright, make-flat-x-from-upright,
    put-flat-x."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = oblong_block_name(oblong_start)
    return Template(
        name=f"simple_unstack_oblong_to_flat_x_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2),
                                  _stack(0, ix2+1, iy2)],
                                 "flat-x")],
        source_placements=[(block,
                                  [_stack(0, ix, iy), _stack(1, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 2,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def simple_unstack_oblong_to_flat_y(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        oblong_start: int = 0) -> Template:
    """L2: oblong starts upright at stack (ix,iy)+(L0,L1); picked,
    rotated to flat-y, placed flat-y at (ix2, iy2..iy2+1).
    Plan = 3 actions: pick-upright, make-flat-y-from-upright,
    put-flat-y."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = oblong_block_name(oblong_start)
    return Template(
        name=f"simple_unstack_oblong_to_flat_y_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2),
                                  _stack(0, ix2, iy2+1)],
                                 "flat-y")],
        source_placements=[(block,
                                  [_stack(0, ix, iy), _stack(1, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 2,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def simple_unstack_long_to_flat_x(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        long_start: int = 0) -> Template:
    """L2: long starts upright at stack (ix,iy)+(L0,L1,L2); picked,
    rotated to flat-x, placed flat-x at (ix2..ix2+2, iy2).
    Plan = 3 actions: pick-long-upright, make-long-flat-x-from-upright,
    put-long-x."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = long_block_name(long_start)
    return Template(
        name=f"simple_unstack_long_to_flat_x_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2),
                                  _stack(0, ix2+1, iy2),
                                  _stack(0, ix2+2, iy2)],
                                 "flat-x")],
        source_placements=[(block,
                                  [_stack(0, ix, iy),
                                   _stack(1, ix, iy),
                                   _stack(2, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 2,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def simple_unstack_long_to_flat_y(
        ix: int, iy: int, ix2: int, iy2: int, *,
        cfg: Optional[MultilevelBlocksConfig] = None,
        long_start: int = 0) -> Template:
    """L2: long starts upright at stack (ix,iy)+(L0,L1,L2); picked,
    rotated to flat-y, placed flat-y at (ix2, iy2..iy2+2).
    Plan = 3 actions: pick-long-upright, make-long-flat-y-from-upright,
    put-long-y."""
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    block = long_block_name(long_start)
    return Template(
        name=f"simple_unstack_long_to_flat_y_anchor({ix},{iy})_to_({ix2},{iy2})",
        goal_placements=[(block,
                                 [_stack(0, ix2, iy2),
                                  _stack(0, ix2, iy2+1),
                                  _stack(0, ix2, iy2+2)],
                                 "flat-y")],
        source_placements=[(block,
                                  [_stack(0, ix, iy),
                                   _stack(1, ix, iy),
                                   _stack(2, ix, iy)],
                                  "upright")],
        build_order=[block],
        metadata={"difficulty": 2,
                       "anchor": (ix, iy), "goal_anchor": (ix2, iy2)},
    )


def multi_tower(rng: np.random.Generator, *,
                   cfg: Optional[MultilevelBlocksConfig] = None,
                   n_towers: int = 3, height_range: Tuple[int, int] = (2, 4),
                   cube_start: int = 0) -> Template:
    """L5: N independent cube towers at non-overlapping anchors.

    Total cube count = sum of heights.  Anchors are chosen ≥3 cells
    apart in both x and y so the put_cube descents have clearance from
    neighbour towers.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    cells_x, cells_y, cells_z = cfg.stack_grid_cells

    # Sample n_towers anchors on a 3-spaced grid.  Coarse positions
    # available in a 10x10 stack: ix in {0, 3, 6, 9} × iy in {0, 3, 6, 9}.
    # Drop the corner-most ones if put_upright reach-limit applies (we
    # only use cubes here so no upright issue).
    candidates: List[Tuple[int, int]] = []
    for ix in range(1, cells_x - 1, 3):
        for iy in range(1, cells_y - 1, 3):
            candidates.append((ix, iy))
    if len(candidates) < n_towers:
        n_towers = len(candidates)
    chosen = list(rng.choice(len(candidates), size=n_towers, replace=False))
    anchors = [candidates[i] for i in chosen]

    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []
    cube_idx = cube_start
    heights: List[int] = []
    for (ix, iy) in anchors:
        h = int(rng.integers(height_range[0],
                                 min(height_range[1], cells_z) + 1))
        heights.append(h)
        for k in range(h):
            block = cube_block_name(cube_idx)
            cube_idx += 1
            order.append(block)
            goal.append((block, [_stack(k, ix, iy)], "flat-x"))
            s_cells = alloc.alloc(1, 1)
            src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))
    return Template(
        name=f"multi_tower_n{n_towers}_h{'_'.join(str(h) for h in heights)}",
        goal_placements=goal, source_placements=src, build_order=order,
        metadata={"difficulty": 5, "anchors": anchors, "heights": heights},
    )


def staircase(ix: int, iy: int, *,
                 cfg: Optional[MultilevelBlocksConfig] = None,
                 n: int = 4, cube_start: int = 0) -> Template:
    """L5: descending heights at columns ix, ix+3, ix+6, ix+9, … in iy.

    Column k has (n - k) cubes stacked.  Total cubes = n*(n+1)/2.
    For n=4: 4+3+2+1 = 10 cubes.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    cells_x, _, cells_z = cfg.stack_grid_cells
    if n < 1 or n > cells_z:
        raise ValueError(f"staircase n must be in [1, {cells_z}], got {n}")
    if ix + 3 * (n - 1) >= cells_x:
        raise ValueError(
            f"staircase n={n} needs ix < {cells_x - 3*(n-1)}, got {ix}")

    alloc = _PartsAllocator(cfg)
    goal: List[_Placement] = []
    src: List[_Placement] = []
    order: List[str] = []
    cube_idx = cube_start
    for col, height in enumerate(range(n, 0, -1)):
        col_ix = ix + col * 3
        for k in range(height):
            block = cube_block_name(cube_idx)
            cube_idx += 1
            order.append(block)
            goal.append((block, [_stack(k, col_ix, iy)], "flat-x"))
            s_cells = alloc.alloc(1, 1)
            src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))
    return Template(
        name=f"staircase_n{n}_anchor({ix},{iy})",
        goal_placements=goal, source_placements=src, build_order=order,
        metadata={"difficulty": 5, "anchor": (ix, iy), "n": n},
    )


def double_bridges(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       oblong_start: int = 0,
                       long_start: int = 0) -> Template:
    """L5: two upright_bridges side-by-side along x.

    Bridges A at (ix, iy) and B at (ix+4, iy).  Each bridge is a 3×3
    footprint; 4-cell anchor spacing means B starts 1 cell after A's
    rightmost column.  Total: 8 uprights + 4 longs = 12 blocks, ~36
    actions.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    cells_x = cfg.stack_grid_cells[0]
    if ix + 6 >= cells_x:
        raise ValueError(
            f"double_bridges needs ix < {cells_x - 6}, got {ix}")

    # Build the two bridges independently with merged kit allocation.
    a = upright_bridges(ix, iy, cfg=cfg,
                              oblong_start=oblong_start,
                              long_start=long_start)
    # Bridge B uses the next oblong/long indices.
    b_oblong_start = oblong_start + 4
    b_long_start = long_start + 2
    b = upright_bridges(ix + 4, iy, cfg=cfg,
                              oblong_start=b_oblong_start,
                              long_start=b_long_start)

    # Concatenate but re-allocate parts cells so the two kits don't
    # collide on the parts table.  Easier: re-derive sources via the
    # shared allocator over all blocks in build_order.
    alloc = _PartsAllocator(cfg)
    new_src: List[_Placement] = []
    src_by_block_a = {block: (cells, orient)
                          for block, cells, orient in a.source_placements}
    src_by_block_b = {block: (cells, orient)
                          for block, cells, orient in b.source_placements}
    for block in a.build_order:
        _, orient = src_by_block_a[block]
        footprint = 3 if block.startswith("long_") else \
                          (2 if block.startswith("oblong_") else 1)
        s_cells = alloc.alloc(footprint, 1)
        new_src.append((block, _cells_xy_to_ids(s_cells), orient))
    for block in b.build_order:
        _, orient = src_by_block_b[block]
        footprint = 3 if block.startswith("long_") else \
                          (2 if block.startswith("oblong_") else 1)
        s_cells = alloc.alloc(footprint, 1)
        new_src.append((block, _cells_xy_to_ids(s_cells), orient))

    return Template(
        name=f"double_bridges_anchor({ix},{iy})",
        goal_placements=a.goal_placements + b.goal_placements,
        source_placements=new_src,
        build_order=a.build_order + b.build_order,
        metadata={"difficulty": 5, "anchor": (ix, iy)},
    )


def tower_on_bridge(ix: int, iy: int, *,
                       cfg: Optional[MultilevelBlocksConfig] = None,
                       tower_h: int = 2,
                       oblong_start: int = 0,
                       long_start: int = 0,
                       cube_start: int = 0) -> Template:
    """L4: upright_bridges + a cube tower on top of the L2 bridge.

    Tower is at (ix, iy+1) — the centre of the L2 long-y bridge along
    column ix — and rises to L2+tower_h levels.  Total: 4 uprights + 2
    longs + tower_h cubes.  For tower_h=2: 8 blocks, ~22 actions.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    if cfg.stack_grid_cells[2] < 3 + tower_h:
        raise ValueError(
            f"tower_on_bridge needs >={3 + tower_h} stack levels for "
            f"tower_h={tower_h}")
    base = upright_bridges(ix, iy, cfg=cfg,
                                 oblong_start=oblong_start,
                                 long_start=long_start)
    # Re-allocate parts kit using shared allocator.
    alloc = _PartsAllocator(cfg)
    new_src: List[_Placement] = []
    src_by_block = {block: (cells, orient)
                       for block, cells, orient in base.source_placements}
    for block in base.build_order:
        _, orient = src_by_block[block]
        footprint = 3 if block.startswith("long_") else \
                          (2 if block.startswith("oblong_") else 1)
        s_cells = alloc.alloc(footprint, 1)
        new_src.append((block, _cells_xy_to_ids(s_cells), orient))

    # Add cubes at L3, L4, ... at the centre of bridge A (column ix,
    # iy+1 — the middle cell of the long-y bridge).
    goal = list(base.goal_placements)
    order = list(base.build_order)
    cube_idx = cube_start
    for k in range(tower_h):
        block = cube_block_name(cube_idx)
        cube_idx += 1
        order.append(block)
        goal.append((block, [_stack(3 + k, ix, iy + 1)], "flat-x"))
        s_cells = alloc.alloc(1, 1)
        new_src.append((block, _cells_xy_to_ids(s_cells), "flat-x"))

    return Template(
        name=f"tower_on_bridge_h{tower_h}_anchor({ix},{iy})",
        goal_placements=goal, source_placements=new_src,
        build_order=order,
        metadata={"difficulty": 4, "anchor": (ix, iy), "tower_h": tower_h},
    )


def compound(rng: np.random.Generator, *,
                cfg: Optional[MultilevelBlocksConfig] = None,
                n_substructures: int = 2) -> Template:
    """L5: combine 2-3 substructures at non-overlapping anchors.

    Picks random substructures from a curated list and stitches them
    together.  Anchor regions are partitioned so the substructures
    don't overlap on the stack grid.  Parts kit is shared via the
    global allocator.
    """
    if cfg is None:
        cfg = MultilevelBlocksConfig()
    cells_x, cells_y, _ = cfg.stack_grid_cells

    # Partition the stack grid into halves (x or y) — each substructure
    # gets one half.  Keeps anchors well-separated and avoids footprint
    # overlap.  For n_substructures=2: split x in half.  For 3: split
    # into 3 columns.
    n_sub = min(max(n_substructures, 2), 3)
    sub_region_x = cells_x // n_sub
    sub_anchors: List[Tuple[int, int]] = []
    for k in range(n_sub):
        # Anchor in middle-ish of the k-th column.
        ix = k * sub_region_x + sub_region_x // 2 - 1
        iy = int(rng.integers(2, cells_y - 4))
        sub_anchors.append((max(0, ix), iy))

    # Curated substructure builders that fit in ~3x3 footprint.
    # Upright substructures (upright_bridges, tower_on_bridge) were
    # excluded historically due to put_upright failures in the FAST
    # executor.  Those failures are mostly gone after Phase 3.7/3.8,
    # except at boundary stack cells (ix >= 5) where the LUT's cached
    # arm posture can clip neighbour substructures.  Bias placement so
    # upright variants land at LEFT-half anchors (ix < 5) where the
    # fast IK is reliable; right-half anchors get cube/pyramid variants
    # most of the time, occasionally upright (10%) — let the
    # generator's reject-and-retry handle the failures so L5 still has
    # some right-half upright samples in the dataset.
    _UPRIGHT_CHOICES = ["tower_on_bridge", "upright_bridges"]
    _CUBE_CHOICES = ["cube_tower", "long_pyramid"]
    _RIGHT_HALF_UPRIGHT_PROB = 0.10  # weak chance per right-half anchor

    # Aggregate.
    alloc = _PartsAllocator(cfg)
    all_goal: List[_Placement] = []
    all_src: List[_Placement] = []
    all_order: List[str] = []
    cube_idx = 0
    oblong_idx = 0
    long_idx = 0
    subs: List[str] = []
    for (sa_ix, sa_iy) in sub_anchors:
        if sa_ix >= 5:
            # Right-half anchor: strongly bias toward non-upright variants
            # to avoid the LUT-cached-posture clipping issue.
            if rng.random() < _RIGHT_HALF_UPRIGHT_PROB:
                choice = str(rng.choice(_UPRIGHT_CHOICES))
            else:
                choice = str(rng.choice(_CUBE_CHOICES))
        else:
            # Left-half anchor: uniform across all 4 variants.
            choice = str(rng.choice(_UPRIGHT_CHOICES + _CUBE_CHOICES))
        if choice == "cube_tower":
            h = int(rng.integers(2, min(cfg.stack_grid_cells[2], 4) + 1))
            sub = cube_tower(h, sa_ix, sa_iy, cfg=cfg,
                                  start_idx=cube_idx)
            cube_idx += h
        elif choice == "long_pyramid":
            sub = long_pyramid(sa_ix, sa_iy, cfg=cfg,
                                     long_start=long_idx,
                                     oblong_start=oblong_idx,
                                     cube_start=cube_idx)
            long_idx += 1
            oblong_idx += 1
            cube_idx += 1
        elif choice == "tower_on_bridge":
            if sa_ix + 2 >= cells_x or sa_iy + 2 >= cells_y:
                # Fall back to cube_tower for small region.
                h = int(rng.integers(2, 4))
                sub = cube_tower(h, sa_ix, sa_iy, cfg=cfg,
                                       start_idx=cube_idx)
                cube_idx += h
                choice = "cube_tower"
            else:
                sub = tower_on_bridge(sa_ix, sa_iy, cfg=cfg,
                                              oblong_start=oblong_idx,
                                              long_start=long_idx,
                                              cube_start=cube_idx)
                oblong_idx += 4
                long_idx += 2
                cube_idx += 2
        elif choice == "upright_bridges":
            if sa_ix + 2 >= cells_x or sa_iy + 2 >= cells_y:
                # Same fallback as tower_on_bridge — 3x3 footprint
                # doesn't fit at this anchor.
                h = int(rng.integers(2, 4))
                sub = cube_tower(h, sa_ix, sa_iy, cfg=cfg,
                                       start_idx=cube_idx)
                cube_idx += h
                choice = "cube_tower"
            else:
                sub = upright_bridges(sa_ix, sa_iy, cfg=cfg,
                                              oblong_start=oblong_idx,
                                              long_start=long_idx)
                oblong_idx += 4
                long_idx += 2
        subs.append(choice)
        # Merge goal placements (build orders) and re-allocate sources.
        all_goal.extend(sub.goal_placements)
        all_order.extend(sub.build_order)
        # Re-source each block via the shared allocator.
        src_by_block = {block: (cells, orient)
                              for block, cells, orient in sub.source_placements}
        for block in sub.build_order:
            _, orient = src_by_block[block]
            footprint = 3 if block.startswith("long_") else \
                              (2 if block.startswith("oblong_") else 1)
            s_cells = alloc.alloc(footprint, 1)
            all_src.append((block,
                                  _cells_xy_to_ids(s_cells), orient))

    return Template(
        name=f"compound_{'_'.join(subs)}",
        goal_placements=all_goal,
        source_placements=all_src,
        build_order=all_order,
        metadata={"difficulty": 5, "subs": subs,
                       "anchors": sub_anchors},
    )
