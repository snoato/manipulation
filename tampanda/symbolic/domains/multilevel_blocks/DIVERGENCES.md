# multilevel_blocks — divergences from the source paper

Source: Kulshrestha, Qureshi, *Structural Concept Learning via Graph Attention for Multi-Level Rearrangement Planning*, CoRL 2023.

## Object types

| | Paper | This implementation |
| --- | --- | --- |
| Block shapes | 8 primitives — cuboids, cylinders, pyramids, prisms — at varied sizes | Single shape: 2.5 cm wooden cubes (uniform) |
| Block count | 8–25 (training), 8–25 (test, with generalisation regimes 10–15, 15–20, 20–25) | Default 6; configurable via `n_blocks` |
| Stable orientation | Cylinders cannot stand vertically; pyramids placed on a side, not upright | All cubes — no orientation constraint |

## Workspace abstraction

| | Paper | This implementation |
| --- | --- | --- |
| Goal representation | Point-cloud + per-object segmentation describing the target structure | Symbolic PDDL goal listing concrete `(occupied <cell> <block>)` literals |
| Block geometry | Continuous SE(3) poses with plane-fitting on top surfaces (allows non-axis-aligned stacking) | Discrete grid cells; cubes axis-aligned to the table grid |
| Surface model | Continuous top surface of every object | One 1×1 :class:`GridRegion` per block (`block_X_top`) — the **coarse** surface variant from the planning conversation |
| Multi-supporter stacking | Allowed: an upper block may rest on TWO lower blocks (plane-fit on the union of two top faces) | **Not supported.**  1×1 surface ⇒ tower-style stacking only.  Documented divergence; the framework supports SurfaceGrid sizes >1×1 in future work. |

## PDDL

| | This implementation |
| --- | --- |
| Types | `cell`, `block` |
| Predicates | `occupied`, `empty`, `holding`, `gripper-empty`, `surface-of` (static) |
| Actions | `pick(?obj, ?cel, ?surf)`, `put(?obj, ?cel)` |

Pick takes an extra `?surf` parameter so the action can require the
block being picked is *clear*: the static `(surface-of ?obj ?surf)`
constraint pins `?surf` to the block's surface cell, and `(empty
?surf)` enforces clearness.  No conditional effects, no derived
predicates — fully pyperplan-compatible.

The `gripper-empty` precondition on pick (combined with the fact that
a held block has no defined cell) implicitly prevents the planner from
placing things on a held block's surface.

## Geometric pose resolution

The 1×1 surface region of each block tracks the block's *current*
position dynamically — when block X moves, its surface cell `block_X_top__0_0`
moves with it.  Static `GridRegion.pose_for(cell)` returns a stale
placeholder; the bridge's `_block_to_cell` and the
`set_blocks_at_cells` helper resolve cell IDs to live world poses
recursively (a block on `block_2_top` lookups `block_2`'s current
world position, then offsets by one cube size in z).

## Goals

`tower_goal([2, 1, 0], base_cell)` builds the goal literals for a
3-block tower with `block_2` at `base_cell`, `block_1` on top of
`block_2`, and `block_0` on top of `block_1`.  Free-form structures
can be expressed by hand-writing the equivalent `(occupied …)` list.

Constraints inherent to the 1×1 surface model:

* No two blocks can share a surface cell.
* No multi-supporter configurations (T-shape, brick-pattern, …).
* Towers can be arbitrarily tall but always single-column.

## What we made easier

* **Cubes only** — no orientation reasoning, no stable-orientation rules.
* **Single-supporter (tower) stacking only** — paper allows multi-supporter.
* **Goal as PDDL literals** — paper uses learned point-cloud
  embeddings; we use explicit symbolic goals.
* **Discrete table grid** — paper uses continuous poses on the table.

## What we did NOT make easier

* **Multi-level structures** — the symbolic abstraction supports
  stacks of arbitrary depth.
* **Per-block surface tracking** — each block has its own surface cell
  that becomes available iff that block is placed and clear.

## v1 TODOs

* Coarser surface grids (e.g. 2×2) for two-block-wide upper layers.
* Multi-supporter stacking (paper's structural-dependency feature) —
  needs derived predicates or per-supporter-pair PDDL actions.
* Mixed primitives (cuboids of different sizes, cylinders) — the
  AssetSet abstraction supports it; the bridge's stacked-detection
  logic would need per-block size info.
* `feasibility.py` — wraps `RRTStar` for actual block manipulation.
