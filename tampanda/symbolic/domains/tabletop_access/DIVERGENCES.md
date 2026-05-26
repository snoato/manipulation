# tabletop_access — divergences from the source paper

Source: Ait Bouhsain, Merlau, Alami, Simeon, *Long-Horizon Task and Motion Planning with Learning-Based Geometric Reasoning*, HAL 2025.

This domain implements the paper's `access` and `access-19` problems.  Both share
one PDDL family (filter-mode `domain.pddl` and face-mode `domain_face.pddl`) and
differ only in scene config: shelf shape, asset palette, and grid count.

## Scene structure — `access`

| | Paper | This implementation |
| --- | --- | --- |
| Shelf shape | free-standing 2-deck shelf with 4 corner legs | identical (:class:`MultiTierShelf`) |
| Decks | 2 horizontal slabs supported by 4 legs | 2 slabs at z = 0.40 m and 0.62 m by default |
| Items | YCB meshes (meat can, mustard bottle, soup can, cracker box, gelatin box, pudding box) | YCB-proxy axis-aligned **boxes** sized to the YCB bounding boxes, swappable later for real meshes |
| Placement regions | floor, middle deck, top deck | three :class:`GridRegion`s: `floor`, `middle_deck`, `top_deck` |
| Grasps | 5-face grasp DB per object, ~200 grasps for box-shaped items | 6-face symbolic abstraction (top, bottom, front, back, left, right); face geometry not modelled per-asset |

## Scene structure — `access-19`

| | Paper | This implementation |
| --- | --- | --- |
| Shelf shape | open-tunnel deck shelf | :class:`Shelf` with `open_faces=("+x", "-x")` |
| Cubes | 18 generic small red cubes + 1 distinguished cube (OoI) | 18 reds (`blocker_0..17`) + 1 blue (`ooi`); 4×4×6 cm half-extents (40×40×60 mm) |
| Layout | 3 columns × 6 rows + OoI at the back of the middle column | identical |
| Placement regions | shelf interior + shelf top | two :class:`GridRegion`s: `shelf_interior`, `shelf_top` |

## Symbolic representation

The paper does **not** use PDDL — it uses a TAMP search where actions decompose
to `Pick + Place` over 5 grasp types per object.  We use PDDL with two
parameterisations:

### Filter mode (`domain.pddl`)
Same predicates as `confined_shelf` (`occupied`, `empty`, `holding`,
`gripper-empty`).  Action signatures are `(pick ?obj ?cel)` and
`(put ?obj ?cel)` — face is a refinement detail handled by the feasibility
checker.  The planner doesn't see faces; it just asks "is the action feasible
at all?", and the feasibility checker tries every face internally.

### Face mode (`domain_face.pddl`)
Adds a `face` type (six values: `top`, `bottom`, `front`, `back`, `left`,
`right`).  Action signatures are `(pick ?obj ?face ?cel)` and
`(put ?obj ?face ?cel)` with `(face-grasp-clear ?obj ?face)` as a
precondition.  The planner can reason about exposing/covering specific faces.

`(face-grasp-clear ?obj ?face)` is **code-evaluated** by the bridge from the
live MuJoCo state, with two checks:

1. **Region-level** — the region containing `?obj` must list the face's
   approach direction in its `access_modes`.  E.g., shelf-interior cells
   with `access_modes=("front", "back")` block `top`, `bottom`, `left`, `right`.
2. **Adjacent-cell** — for horizontal faces, the cell adjacent in the face
   direction must be empty.  Off-grid edges count as open space.

This is a v1 approximation — the paper's geometric reasoning is more
sophisticated.  We don't model:

* Per-object grasp cone (some YCB shapes have non-axis-aligned faces).
* Gripper-finger clearance beyond the immediate neighbouring cell.
* Reachability of vertical faces from above (we always say `top` is allowed
  iff the region permits it).

## Goals

The paper's goal is "target object reaches a specified pose; blockers return
to their home poses".  We encode this as concrete `(occupied ?cel ?obj)`
literals in PDDL — each blocker has a `(occupied home_cel_i blocker_i)` goal
and the OoI has `(occupied goal_cel ooi)`.  This matches the paper's
evaluation but is expressed as a STRIPS goal so any STRIPS-compatible UP
backend (pyperplan, fast-downward) can plan over it.

## What we made easier

* **Generic boxes for `access-19`**, axis-aligned YCB-proxy boxes for
  `access` — no real meshes in v1.  The :class:`AssetSet` swap mechanism
  lets a future ticket replace these with mesh templates without touching
  domain code.
* **Five-grasp DB → six face symbols**.  The paper's grasps are continuous
  poses sampled from each face; we discretise to one grasp per face.
* **Discrete goal cells**.  The paper specifies SE(3) goal poses; we
  discretise to the closest cell on the appropriate placement region.
* **Single-arm Franka**.  The paper uses the same.

## What we did NOT make easier

* **Multi-grid workspace** preserved — three regions for `access`, two for
  `access-19` — so the planner can move objects across grids the way the
  paper does (e.g., OoI ends up on the top deck, not the middle deck).
* **Multi-cell footprints** are supported by the workspace abstraction;
  larger YCB-proxy items will occupy >1 cell once the bridge is updated to
  use :class:`Footprint` per object (this is a v1 TODO; current bridge
  treats every object as 1-cell).
* **Both modes coexist** — filter and face — as separate problem families
  per the planning conversation, so rgnet can train one policy per family.

## Assumptions

* OoI body name is `ooi` (renamed from the paper's "target" to avoid
  collision with the `FrankaEnvironment`'s built-in mocap target body).
* PDDL type is `movable` (renamed from `object` to avoid recursion in
  `unified_planning`'s PDDL type-resolution — `object` is a built-in type
  and re-declaring it loops).
* `access` shelf body sits at world `(0.55, 0, 0)`; `access-19` shelf at
  world `(0.5, 0, 0)` on top of a `simple_table` body.

## v1 TODOs

* `feasibility.py` — wraps `ConfinedMotionPlanner` for both scene variants;
  filter-mode iterates faces internally, face-mode delegates to the
  per-face check.  Not blocking the symbolic pipeline.
* Multi-cell footprint per asset — currently every object occupies one
  cell regardless of its half-extents.  Important for cracker-box-sized
  YCB items in `access`.
