# multilevel_blocks — design notes (2026-05 redesign)

We rebuilt the `multilevel_blocks` domain around two physical tables and a 3D
stacking grid, replacing the prior single-table dynamic-surface formulation.
The redesign decouples flat layout from vertical structure and introduces a
second block shape that supports orientation-change actions.

## Workspace

Two free-standing tables share the same Franka workspace.

* **Parts table** — behind the robot (centred at `(0, -0.45, 0)`, 65 × 65 cm).
  Holds a 2D grid of **15 × 15** cells where blocks rest flat at table-top
  height.  No stacking on this table.
* **Stack table** — directly in front of the robot (centred at `(0, +0.50, 0)`,
  50 × 50 cm).  Holds a 3D grid of **10 × 10 × 5** cells modelled as five
  stacked 2D `GridRegion`s (`stack_L0` … `stack_L4`), spaced one cube-edge
  apart in z.

Both grids have cell pitch equal to the block edge (30 mm); blocks pack tight
with no inter-cell gap.  The arm rotates its base joint by ±π/2 to alternate
between the two workspaces.

Geometry was locked in by a dual-table IK probe: every cell on both grids is
fully reachable at all stack levels.

## Block shapes

* **1 × 1 cubes** (30 × 30 × 30 mm), occupying one cell.
* **2 × 1 oblong blocks** (60 × 30 × 30 mm), occupying two cells.  Three valid
  placement orientations:
    - **flat-x**: two cells along +x at the same level (long axis along world-x).
    - **flat-y**: two cells along +y at the same level (long axis along world-y).
    - **upright**: two cells at the same (ix, iy) one level apart (long axis
      along world-z).  Only valid on the stack table.

## Grasp model

All grasps are kinematically constrained — the gripper approaches the long
sides of each block face-centred:

* **Cube** — top-down, centred on the cube.
* **Oblong, flat-x or flat-y** — top-down, fingers close perpendicular to the
  block's long axis.
* **Oblong, upright** — front-facing horizontal approach, fingers close on the
  two longest vertical faces.

## PDDL

Domain `multilevel-blocks`, planned with **Fast Downward** (via the
`unified-planning` interface).  The vocabulary uses static directional
adjacency (`above`, `east-of`, `north-of`; asymmetric, so reverse directions
follow from swapped arguments) plus shape and region markers.  Fluents track
per-block cell occupancy (`in ?b ?c`, multi-valued for 2×1 blocks),
emptiness, gripper state, and held orientation (`held-cube`, `held-flat-x`,
`held-flat-y`, `held-upright`).

### Predicates

| kind     | predicate                          | semantics                                      |
|----------|------------------------------------|------------------------------------------------|
| static   | `(cube ?b)` / `(oblong ?b)`        | block shape                                    |
| static   | `(in-parts ?c)` / `(in-stack ?c)`  | region marker                                  |
| static   | `(above ?c-low ?c-up)`             | `c-up` is directly above `c-low` (+z, same xy) |
| static   | `(east-of ?c1 ?c2)`                | `c2` is directly east of `c1` (+x)             |
| static   | `(north-of ?c1 ?c2)`               | `c2` is directly north of `c1` (+y)            |
| fluent   | `(in ?b ?c)`                       | block occupies cell (multi-valued for 2×1)     |
| fluent   | `(empty ?c)`                       | cell is unoccupied                             |
| fluent   | `(gripper-empty)`                  | gripper holds nothing                          |
| fluent   | `(held-cube ?b)`                   | gripper holds a cube                           |
| fluent   | `(held-flat-x ?b)`                 | gripper holds an oblong, long axis along x     |
| fluent   | `(held-flat-y ?b)`                 | gripper holds an oblong, long axis along y     |
| fluent   | `(held-upright ?b)`                | gripper holds an oblong, long axis along z     |

### Actions

The 14 actions decompose into three groups:

* **4 picks** — `pick-cube`, `pick-flat-x`, `pick-flat-y`, `pick-upright`.
* **4 puts** — `put-cube`, `put-flat-x`, `put-flat-y`, `put-upright`.
* **6 in-hand transforms** — `make-upright-from-x`, `make-upright-from-y`,
  `make-flat-x-from-upright`, `make-flat-y-from-upright`, `turn-x-to-y`,
  `turn-y-to-x`.  Mutate the held-orientation predicate without changing world
  state, modelling free-air gripper rotation.

### Invariants

* **No-overhang**: pick preconditions require that any directly-above cell be
  empty —
  `(not (exists ?c-above (and (above ?c ?c-above) (not (empty ?c-above)))))`.
  Put preconditions require the target cell be empty *and* either at
  table-level (`(not (exists ?c-below (above ?c-below ?c)))`) or above an
  occupied cell.  Together these make overhangs unreachable: any block in a
  stack column must be supported by its direct neighbour below or by the
  table.
* **Gripper-finger clearance** against neighbour cells is **not** encoded in
  PDDL — enforced by the motion-planning bridge's feasibility check at
  execution time, matching the convention used in the `access-19` shelf
  domain.

## Difficulty / OOD axes

Difficulty is encoded in the curriculum, not in the workspace geometry: every
grid cell is fully IK-reachable (100 % on both grids in our probe).  The
intended OOD axes match SCL's training/test split:

* **Object count** — 6–10 train → up to 15 test.
* **Tower height** — ≤ 3 levels train → up to 5 test.
* **Stacks per scene** — 1–3 train → up to 4 test.

## Planner requirement

The PDDL uses negated existentials in preconditions; pyperplan rejects these
(strict-STRIPS CNF).  The domain therefore requires Fast Downward via
`up-fast-downward`.  Other domains in this repo continue to use pyperplan
unchanged.
