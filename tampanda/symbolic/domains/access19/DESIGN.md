# access19 — design notes

`access19` is the closed-top cubicle variant of the Bouhsain HAL 2025
tabletop-access task: 18 red blocker cubes packed front-to-back in a
3-column shelf, plus one blue object-of-interest (OoI) hidden at the
back.  The robot must extract the OoI to the open top deck above the
shelf.  Forked from `tabletop_access` so the deck-style geometry
evolves independently of the 3-tier `access` variant.

## Workspace

One free-standing shelf body (no separate table).  The shelf consists
of a pedestal, an enclosed lower cubicle (the "interior"), and an open
deck on top of the cubicle's roof.

* **Shelf interior** — closed cubicle, robot-facing -y face open.
  Interior cavity is `7 × 7` cells (interior_x × interior_y) at pitch
  `cell_size = 0.06 m`.  Cubes sit on the floor of the cubicle (cube
  centre z = `0.452 m`).  Three cube columns at `ix ∈ {1, 3, 5}`,
  separated by empty "gripper channel" columns at `ix ∈ {0, 2, 4, 6}`
  that give the gripper width clearance to slide between columns.
* **Shelf top** — open `7 × 7` grid sitting on the cubicle's outer roof
  at `z = 0.704 m`.  Open above; no walls.  Used as scratch space for
  blockers that have to be temporarily relocated during OoI extraction.

Geometry was locked in by:
* `examples/measure_access19_arm_extent.py` — reachability sweep that
  found `table_pos = (0.35, 0.40)` gives full coverage of all 49
  interior cells via column-aligned approach + row-by-row descent.
* `interior_height_z = 0.24 m` — bumped from the original 0.20 m after
  visualisation showed the forearm clipping the cubicle's top wall
  during pick.  0.24 keeps the elbow clear without raising the deck
  out of the Franka's reach envelope (0.28 m broke col-5 deck reach).

## Cube geometry

All 19 movables are uniform `4 × 4 × 8 cm` prisms (`cube_half_extents
= (0.020, 0.020, 0.040)`).  The 8 cm height was chosen so that
grasping near the cube top keeps the wrist's link7 capsule above the
cubicle floor (`grasp_z = cube_z + cube_half_z - 0.010`).  This is
asymmetric — pinch contact is offset 3 cm above the cube's centre of
mass — but it's necessary inside the closed-top cubicle.

## Cell packing

Cubes pack tight in y: `cell_size = 0.06 m`, `cube_d = 0.04 m`, gap
= 2 cm.  Front-to-back rows touch (paper-faithful — only front-row
cubes are accessible until front rows are removed).  In x the gripper
channels at `ix ∈ {0, 2, 4, 6}` provide a full 6 cm of clear space
between cube columns.

## Action vocabulary

PDDL filter mode (`pddl/domain.pddl`):

* `(pick ?obj - movable ?cel - cell)` — preconditions `(gripper-empty)`
  and `(occupied cel obj)`.
* `(put ?obj - movable ?cel - cell)` — preconditions `(holding obj)`
  and `(empty cel)`.

Grasp face is a refinement detail handled by the feasibility checker;
the PDDL only sees pick/put.  All grasps use the palm-+y orientation
(`FRONT_QUAT = [-0.5, 0.5, 0.5, 0.5]`) — the gripper enters the
cubicle through the open -y face.

## Spatial structure (`adjacent`)

Mirrors the tabletop spatial-put domain: `(adjacent ?dir ?c1 ?c2)`
static predicate with `direction` type and `north`/`east` constants.

* `north` = `+iy` (depth, away from robot).
* `east`  = `+ix` (column index, robot's right).

Emitted per grid in the problem `:init`:

* `shelf_interior` 7×7 with corners `(0,0)` and `(6,6)` excluded →
  80 edges (84 minus 4 incident to the two excluded corners).
* `shelf_top` 7×7 → 84 edges.
* No cross-grid adjacency — the two grids are disjoint by design
  (lower cubicle and upper deck are at different `level_z` and only
  the chain executor knows how to traverse between them).

The actions don't use adjacency as a precondition.  The predicate
exists purely as a structural signal for the GNN: cell nodes get
linked via the `adjacent` edges so message-passing propagates
embeddings along the grid structure, letting the value function
learn spatial semantics (front-to-back row order, column adjacency)
from the edge topology rather than needing them encoded as
per-cell positional features.

## Chain executor (`chains.py`)

Mirrors `tabletop_access` but specialised to the cubicle's tight
packing:

* **`make_access19_pick_fn`** → dispatches to `_pick_interior`
  (cubicle picks) or `_pick_deck` (top-deck picks).
* **`make_access19_put_fn`** → dispatches to `_put_interior` or
  `_put_deck`.
* **Row-step pattern** — both `_pick_interior` and `_put_interior`
  enter the cubicle column-aligned, then step one cell at a time
  forward / backward in y.  This avoids the wrist swinging through
  the cubicle's side walls during a long lerp.
* **Traverse-at-altitude pattern** for `_put_deck` / `_pick_deck` —
  lift to `safe_z`, Cartesian traverse to the column above the
  target, descend to grasp / place, lift back.  `safe_z = top_deck_z
  + 0.10 m` clears the top of any cube already on the deck plus
  the gripper's finger collision capsule (a 1-cm offset below the
  EE site).
* **Controller-convergence retry** in `_execute` — after every
  `execute_path + wait_idle`, the wrapper checks that the arm
  actually reached `path[-1]` within `_CONVERGE_TOL = 0.02 rad`.
  Re-issues the last waypoint up to 3 times if not.

## Curriculum

Five levels (see `generate_data.py:_LEVEL_TEMPLATE_NAMES`):

| Level | Layout | n_blockers | Goal | Plan length |
|--|--|--:|--|--:|
| L0 | `front_row_subset` n=0..1 | 0–1 | OoI to top | 2–4 |
| L1 | `front_row_subset` n=2..3 | 2–3 | OoI to top | 6 |
| L2 | `dense_front` / `scattered_subset` | 3–6 | OoI to top + return blockers | 8–12 |
| L3 | as L2 + `gotcha_corridor_jam` | 3–10 | OoI to top + return blockers | 8–12 |
| L4 | `canonical_18` | 18 | OoI to top + return blockers (full) | 74 |

Half of each level's instances get `mirror_x` applied for variant
multiplication.  Templates are described in `templates.py`.

## Planning

Three planners in `planner.py`:

* **`oracle_plan`** — greedy + bounded backtracking.  Fast (~1–3 s)
  for L0–L4 with OoI-only goals.  Uses heuristic priorities
  (front-to-back picks, column-aligned back-to-front dumps).
* **`astar_plan`** — A* over symbolic states with the FAST
  feasibility check as the transition validator.  Supports
  arbitrary `goal_layout` including return-all.  Uses a tight
  action-pruning heuristic to make the L2-L3 case tractable.
* **`phased_plan`** — hybrid that runs `oracle_plan` for phase 1
  (clear cubicle + OoI to top) then `astar_plan` for phase 2
  (return blockers).  Required for L4 return-all because the full
  joint state-space is too large for plain A*.

Per-level planner choice is in `generate_data.py:_LEVEL_PLANNER`.

## Feasibility checking

`feasibility.py` provides:

* **`check_action`** / **`check_action_sequence`** — runs the chain
  pick/put with optional FAST mode (`_fast_env` context manager
  patches `env.execute_path` / `wait_idle` / gripper waits to skip
  physics).  Same chain logic; only the path-execution step is
  short-circuited.
* **Snap-between-actions** — `check_action_sequence` snaps every
  placed cube to its symbolic cell centre (with identity quat) before
  every action.  Kills cumulative drift in FULL replay; no-op in
  FAST.

FAST↔FULL agreement validated by `examples/access19_full_validation.py`:
* L4 OoI-only (38 actions): 38/38 ✓
* L4 return-all (74 actions): 74/74 ✓ (with `safe_z = 0.10`)

## Parallel feasibility pool

`parallel.py:ParallelFeasibilityChecker` — persistent multiprocessing
pool.  Each worker builds env + chains once at startup.  Compact
`(layout, held, action)` payloads sent per check.  Used by the
data-gen pipeline's `--num-workers` flag.

## Datasets

Two dataset layouts are shipped:

### v1 — `data/access19/` (original, kept for backward compatibility)

The original 180-instance dataset.  Train uses the full curriculum
with canonical_18 at L4 (18-blocker return-all).  Layout:

```
data/access19/
├── domain.pddl
├── train/L0..L4/  120 instances
├── val/L0..L4/    30 instances
└── test/L0..L4/   30 instances
```

### v2 — `data/access19_v2/` (Option-B + generalisation evals)

Designed for "train on max-12 blockers, test generalisation to 18".
Four datasets in one bundle, all output dirs **flat** (no `L<level>/`
subdirs — level is recorded in plan-file metadata only).

```
data/access19_v2/
├── domain.pddl
├── train/         300 instances, max 12 blockers (Option-B)
├── val/            50 instances, same distribution as train
├── eval_pre_b/     30 instances, original (canonical_18 at L4)
└── eval_full/      30 instances, all canonical_18 + return-all
```

The four sets in detail:

| Subset | Count | L4 template | Purpose |
|--|--:|--|--|
| `train/` | 300 (40+60+80+80+40) | `canonical_12` | training |
| `val/` | 50 (10/level) | `canonical_12` | mid-training validation |
| `eval_pre_b/` | 30 (6/level) | `canonical_18` | "does Option-B model handle 18-blocker mixed-difficulty?" |
| `eval_full/` | 30 | `canonical_18` only | "does Option-B model solve the canonical Bouhsain HAL 2025 problem?" |

`canonical_12` = `dense_front(4)`: 12 blockers in front 4 rows × 3
columns + OoI at back.  Plan length: ~26-30 actions.

Generate the v2 bundle in one shot:
```bash
python -m tampanda.symbolic.domains.access19.generate_data \
    --bundle-v2 --num-workers 4 --time-budget 300 \
    --output-dir data/access19_v2
```

## Data generation (per-domain mechanics)

`generate_data.py` mirrors `multilevel_blocks/generate_data.py`:

* `Template` dataclass with `source_placements`, `goal_placements`,
  metadata.
* `build_plan(template, ...)` — dispatches to the right planner.
* `write_pddl_problem` / `write_plan_file` — PDDL writers.
* `_generate_one` → `_generate_split_parallel` → `_run_curriculum`
  → `main`.

CLI:
```
# Full curriculum
python -m tampanda.symbolic.domains.access19.generate_data \
    --curriculum-spec train_120 --num-workers 4 \
    --output-dir data/access19

# Single level (shell-driver dispatch)
python -m tampanda.symbolic.domains.access19.generate_data \
    --level 4 --num 16 --split train \
    --output-dir data/access19
```

Curriculum presets in `_CURRICULA`:
* `train_120` — 16+24+32+32+16 across L0–L4.
* `val_per_level` / `test_per_level` — 6 per level.

## Bugs we hit and fixed

See `DIVERGENCES.md` for detailed diagnostics.  Headline issues:

* **`interior_height_z = 0.20` clipped the forearm** — bumped to 0.24.
* **`_SAFE_Z_ABOVE_SHELF_TOP = 0.08` clipped cube tops during
  traverse** — bumped to 0.10.  The original value put the EE
  exactly at cube-top level, and the gripper's collision capsule
  (extending ~2 cm below the EE site) was scraping cubes during
  every top-deck traverse.  Caught by the per-substep EE probe in
  `examples/access19_action39_probe.py` after the state audit
  pinpointed action 39 of an L4 return-all plan as the corruption
  point (held cube tilted 29°, neighbour cube knocked 42 mm).
* **Controller-tracking convergence** — `chains._execute` re-issues
  the last waypoint if the arm doesn't reach within 2 rad.

## Debug harness

* **`examples/access19_solvability.py`** — 38-action OoI-only run.
  `--viz` opens the MuJoCo viewer.
* **`examples/access19_full_validation.py`** — L4 plan + FAST replay
  + FULL replay; per-action disagreement count.
* **`examples/access19_state_audit.py`** — captures per-action
  metrics (EE tracking err, held cube tilt, unplanned cube
  movements, attach/detach events).  Dumps JSON timeline.
* **`examples/access19_action39_probe.py`** — focused per-substep EE
  + gripper trace of the action that broke L4 return-all.
* **`examples/access19_return_all_viz.py`** — plans the canonical L4
  return-all (caches to `/tmp/access19_return_all_plan.json`), then
  replays in a fresh env at viewer rate.  Use `mjpython`.
* **`examples/access19_replay_from_disk.py`** — round-trip validation
  of stored PDDL plans through FULL replay.
* **`examples/access19_l4_stress_test.py`** — 5 L4 variants (OoI at
  cols 1/3/5, with/without `mirror_x`) for reliability check.
