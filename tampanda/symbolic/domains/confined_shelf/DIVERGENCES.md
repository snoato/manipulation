# confined_shelf — divergences from the source paper

Source: Wang, Gao, Yu, Bekris, *Lazy Rearrangement Planning in Confined Spaces*, ICAPS 2022.

## Geometry

| | Paper | This implementation |
| --- | --- | --- |
| Robot | Motoman SDA10F (dual-arm, right arm used) | Franka Emika Panda (single-arm) |
| Shelf interior | not given numerically; cuboid sized for SDA10F reach | 0.22 × 0.42 × 0.20 m by default; tunable via `ConfinedShelfConfig.interior_size` |
| Open faces | one — robot-facing | one — robot-facing (default `"-x"`) |
| Cell size | implicit lattice from object diameter | explicit `cell_size = 7 cm` — 1 cm of slack over the 6 cm cylinder diameter so the gripper fingers have somewhere to land |
| Grid | implicit lattice from interior + cell_size | **9 × 3 = 27 cells**; back-right corner (8, 2) excluded as kinematically unreachable |
| Cylinder radius / height | not given numerically | `cylinder_radius = 0.030 m` × half-height 10 cm (6 cm diameter × 20 cm tall — soft-drink-bottle proxy) |
| Cylinder count | 8–16 (monotone) and 12–16 (non-monotone) regimes | tunable via `n_cylinders`; default 5 placed on the even-x sublattice of row 0 (every other cell, leaving gripper buffers between every pair of neighbouring cylinders) |

## Workspace abstraction

* The paper uses a discretised lattice of placement candidates `PT` over the shelf floor with cell adjacency (row/column neighbours).
* We use a uniform :class:`GridRegion` with the same row/column adjacency available via `region.neighbours(cell)` — same semantics, different implementation path.

## Symbolic representation

* The paper does **not** use PDDL — it builds an arrangement-graph `G = (V, E)` where vertices are full configurations and edges are pick-and-place actions, with reachability constraints encoded as a constraint set `C(α)`.
* We use PDDL with **the same predicate vocabulary as the existing tabletop domain** — `(occupied …) (empty …) (holding …) (gripper-empty)` — so rgnet's existing predicate handling carries over without changes. Predicates the paper does not need but we add:
  * `(color-of ?cyl ?c)` — static; lets us encode colour-group membership as a first-class fact rather than via cylinder naming.

## Goals

* The paper's goal is "colours grouped by column" (semantic) — internally it's encoded as a target arrangement.
* We encode goals as concrete `(occupied <cell-id> <cylinder-id>)` literals in the PDDL problem, with the goal-state generator picking a per-colour column assignment and emitting one literal per cylinder. Same structure as the paper's evaluation, expressed as a STRIPS goal so any STRIPS-compatible UP backend can plan over it.

## Action set

| | Paper | This implementation |
| --- | --- | --- |
| `pick(obj, cell)` | implicit in pick-and-place edges | explicit |
| `put(obj, cell)` | implicit in pick-and-place edges | explicit (matches the tabletop `spatial_put` variant) |
| `drop` | not used | not exposed (Wang's setting has no discard) |

## Feasibility

* The paper's monotone solver lazily checks reachability: it expands the search tree without verifying the arm path, and only collision-checks when a candidate solution is found.
* Our feasibility hook is the upcoming `confined_shelf/feasibility.py` (next deliverable) — it eagerly evaluates each candidate `(pick …)`/`(put …)` against an RRT\* with the plan2policy-style fallback ladder before reporting it as feasible. Eager checking is more conservative; the rgnet AV-Star training loop benefits from up-front filtering.

## What we made easier

* Identical-cylinder pool: every cylinder has the same shape and colour-group palette — we drop the paper's per-object size variation (which it treats as a parameter sweep, not a core problem feature).
* Cell-aligned placements only: cylinders rest exactly at cell centres. The paper allows continuous placement within a cell.
* Single-arm: we drop the dual-arm option from the paper. All grasps come from the Franka.
* Static buffer rule: we enforce the "no two cylinders in adjacent cells" rule statically in the layout helper.  The paper checks it at runtime as part of its lazy reachability test; for our PDDL grounding we encode it in the spawn logic instead.  Data-generation pipelines should respect the same rule when sampling random instances.

## Gripper model tweak

The Franka panda's default `hand_capsule` collision proxy is a 4 cm radius capsule centred on the wrist — designed for self-collision avoidance, not slim grasps.  During a palm-+y FRONT grasp of a vertical cylinder, this capsule sits perpendicular to the cylinder's axis and physically overlaps the cylinder's body.  In an enclosed cubicle with no top-down access, that overlap pushes the cylinder forward several centimetres during the descent (visible in the simulator as the cylinder being dragged or tipped before the gripper closes).

To preserve clean palm-+y picks at the dense cylinder lattice this domain uses, `apply_runtime_tweaks(env, cfg)` shrinks the capsule to 2 cm radius at build time (`cfg.hand_capsule_radius_override`).  This is a domain-specific override — other domains still get the full 4 cm wrist guard.

## What we did NOT make easier

* The shelf has just one open face — the geometric corridor that makes RRT\* fail is preserved.
* Reachability constraints between objects' starts and goals are emergent: the feasibility checker discovers them from the live MuJoCo state, just like the paper's monotone solver does.

## Assumptions

* Robot base at world origin; table at world `(0.5, 0, 0)`; shelf placed `relative_to="simple_table"` so the shelf body sits flush above the table top.
* The robot's "front" is `+x` from the base, so `open_face="-x"` puts the cubicle's open face on the side closest to the robot.
* Cylinders parked at the sentinel x position (default `100.0`) are treated by the bridge as inactive (not in any cell).  Calling
  `set_cylinders_at_cells` brings them into the workspace.
