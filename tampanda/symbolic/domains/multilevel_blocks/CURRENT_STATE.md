# multilevel_blocks — current-state baseline (2026-05-25)

Reference snapshot of the domain BEFORE the Phase 1-5 work begins
(domain shrink, geometric pre-filters, executor quick wins, parallel
yaws). Future changes are compared against the numbers here.

The code-path state captured below already includes the two
correctness fixes landed earlier today:

* **held-state attach** in `state.py::restore_state` — held-* fluents
  now kinematically attach the block to the EE via
  `_attach_held_block` instead of silently parking. Wired through
  `check_action` (defaults to `on_held="attach"` when an executor is
  passed).
* **`hand_capsule` disabled** by default in `env_builder.py` via
  `apply_runtime_tweaks`. The dataset was generated against an env
  with the capsule disabled; the wrap in `make_multilevel_blocks_builder`
  ensures every downstream consumer (rgnet, demos, validator) sees the
  same env without having to remember to call the tweak. Opt-out via
  `make_multilevel_blocks_builder(..., disable_hand_capsule=False)`.

Item 3 (the `<option>` block) was tried and REVERTED — it broke
previously-validated L4 plans by perturbing physics. See
`examples/multilevel_blocks_l4_regression_probe.py` for the bisection.

---

## Scene + grid configuration

Set by `MultilevelBlocksConfig` defaults in `env_builder.py`:

| Field | Value | Notes |
|---|---|---|
| `n_cubes` | 20 | dataset-wide constant |
| `n_oblong` | 12 | dataset-wide constant (2×1 blocks) |
| `n_long` | 6 | dataset-wide constant (3×1 blocks) |
| `cube_half_extent` | 0.015 m | 30 mm cubes |
| `parts_grid_cells` | (15, 15) | 225 cells; only ~26 ever used in plans |
| `stack_grid_cells` | (10, 10, 5) | 5 levels × 10×10 = 500 cells |
| `parts_table_pos` | (0, -0.45, 0) | behind the robot |
| `stack_table_pos` | (0, 0.50, 0) | in front of the robot |

**MuJoCo env counts** at the dataset config (20+12+6 blocks):

| Quantity | Value |
|---|---|
| `nbody` | 53 |
| `ngeom` | 131 |
| `njnt` | 49 |
| `nq` | 289 |

**MuJoCo opt** (post hand_capsule fix, no `<option>` block injected):
defaults — `solver=Newton (2)`, `iterations=100`, `ls_iterations=50`,
`integrator=Euler (0)`, `timestep=0.002`.

**PDDL object count per problem** (the load on rgnet's GNN):

| Type | Count | Notes |
|---|---|---|
| `block` | 38 | 20 cube + 12 oblong + 6 long |
| `cell` (parts) | 225 | full 15×15 declared |
| `cell` (stack) | 500 | 10×10×5 = 5 levels of 10×10 declared |
| **Total objects** | **763** | every problem |

The cell count is the biggest object-list contributor; only 26 of 225
parts cells are ever referenced in plan actions, but all 225 are
declared in every problem's `(:objects ...)`. This is the load Phase 1
(dynamic per-problem grid) targets.

---

## Action families + chain map

Every PDDL action dispatches to one of five chains in
`executor.py`. PJL = `plan_joint_lerp` (1 IK + 17 collision checks);
PTP = `plan_to_pose` (heavier fallback, rarely fires).

### `_pick_top_down` (5 actions: pick-cube / flat-x / flat-y / long-x / long-y)

| Phase | Op | Quats tried | Substeps |
|---|---|---|---|
| 0 | `add_collision_exception(block)` | — | — |
| 1 | `_to_neutral_home` | (n/a) | 20 |
| 2 | `_to_handoff(src_region)` | quats[0] | 20 |
| 3 | approach `_try_plan_to_pose(anchor + 0.10z)` | **all K** | 20 |
| 4 | `_preclose_for_descent` | — | — |
| 5 | descend `_try_plan_to_pose(anchor)` | **chosen quat only** | 10 |
| 6 | `_close_attach` | — | — |
| 7 | lift `_try_plan_to_pose(anchor + 0.08z)` | chosen quat | 10 |
| 8 | `_to_handoff(src_region)` | chosen quat | 20 |
| 9 | `_to_neutral_home` | (n/a) | 20 |

Phases 0-7 = "core feasibility". Phases 8-9 = post-action cleanup
(Phase 3.1 / "A1" target — skippable in fast mode).

### `pick_upright` (2 actions: pick-upright, pick-long-upright)

| Phase | Op |
|---|---|
| 0 | `_set_home(cell)` (direct qpos write) |
| 1 | `add_collision_exception(block)` |
| 2 | **filter quats** by IK at grasp anchor (1 PJL per candidate) — already a cheap reachability gate |
| 3 | approach (PJL per filtered quat, first success wins) |
| 4 | `_preclose_for_descent` |
| 5 | descend `_try_plan_to_pose(anchor)` with chosen quat |
| 6 | `_close_attach` |
| 7 | lift |

No explicit return-to-handoff at the end — chain ends after lift.
The IK-filter pattern at phase 2 is the cheap-pre-filter shape we
want to port to `_pick_top_down` / `_put_top_down` in Phase 3.3 / "B1".

### `_put_top_down` (5 actions: put-cube / flat-x / flat-y / long-x / long-y)

| Phase | Op | Quats tried | Substeps |
|---|---|---|---|
| 0 | `add_collision_exception(block)` | — | — |
| 1 | `_to_neutral_home` | (n/a) | 20 |
| 2 | `_to_handoff(tgt_region)` | quats[0] | 20 |
| 3 | high-above `_try_plan_to_pose(anchor + 0.15z)` | **all K** | 24 |
| 4 | approach `_try_plan_to_pose(anchor + 0.10z)` | **chosen quat only** | 14 |
| 5 | descend `_try_plan_to_pose(anchor + 0.002z)` | chosen quat | 10 |
| 6 | `_detach_open` | — | — |
| 7 | lift `_try_plan_to_pose(anchor + 0.08z)` | chosen quat | 10 |
| 8 | `_to_handoff(tgt_region)` | chosen quat | 20 |
| 9 | `_to_neutral_home` | (n/a) | 20 |

Yaw is committed at phase 3 and never reconsidered — the root of the
descend failures (see "known failure modes" below). Same A1 / B1
optimisation targets as `_pick_top_down`.

### `put_upright` (2 actions: put-upright, put-long-upright)

| Phase | Op |
|---|---|
| 0 | `_to_neutral_home` |
| 1 | `_to_handoff("stack", FRONT_Y)` |
| 2 | settle to `traverse_z` (safe altitude above max stack) |
| 3 | translate over target xy at `traverse_z` |
| 4 | `_refresh_held_offset` + column-align descent to `ee_z = anchor.z + 0.04` |
| 5 | **final descent** to `place_pose` — failure point in capsule-on baseline |
| 6 | correction nudge if drifted > 3 mm |
| 7 | `_detach_open` |
| 8 | lift to `traverse_z` |
| 9 | `_to_neutral_home` |

The longest chain. Phase 5 is where the L4 failures landed before
the capsule fix.

### `_do_transform` (6 actions: make-upright-from-x/y, make-flat-x/y-from-upright, turn-x↔y, plus the `-long-` variants that delegate to the same)

| Phase | Op |
|---|---|
| 0 | `_to_handoff(region, target_quat)` |
| 1 | `_refresh_held_offset` |

Already minimal.

---

## mj_forward call count per check_action (FAST mode)

Measured via `examples/multilevel_blocks_count_calls.py`:

| Config | nbody | per-call mj_forward | total mj_forward / check | elapsed / check |
|---|---|---|---|---|
| bench dev (2c + 2o + 2l = 6 blocks) | 21 | 0.06 ms | 321 | **366 ms** |
| rgnet (20c + 12o + 6l = 38 blocks) | 53 | 4.39 ms | 321 | **5 095 ms** |

Same algorithmic structure on both scenes; only per-call cost scales
with broadphase work. mj_forward total time is ~28 % of elapsed in
fast mode; mink IK + Python overhead make up the rest.

---

## Validator pass rates (post-hand_capsule fix, 30 problems, seed 0)

**Pre-fix baseline** (capsule active, what rgnet was hitting):

| Level | Pass | Note |
|---|---|---|
| L0 | 5/5 | all good |
| L1 | 5/5 | all good |
| L2 | 5/5 | all good |
| L3 | 5/5 | all good |
| L4 | **0/5** | every put-upright fails — capsule clips neighbor towers |
| L5 | 3/5 | put-cube into dense tower fails when capsule clips adjacent column |
| **Overall** | **23/30 (76.7 %)** | |

**Post-fix baseline** (capsule disabled, same 30 problems, seed 0,
`/tmp/baseline_validator.json`):

| Level | Pass | Avg per-problem elapsed |
|---|---|---|
| L0 | 5 / 5 | ~4.8 s (2-step pick+put) |
| L1 | 5 / 5 | ~9.7 s (4-6 step) |
| L2 | 5 / 5 | ~8.6 s (4-6 step) |
| L3 | 5 / 5 | ~12.0 s (6-9 step) |
| L4 | 5 / 5 | ~105 s (18-22 step, includes put-upright) |
| L5 | 5 / 5 | ~28.6 s (10-22 step, mixed) |
| **Overall** | **30 / 30 (100.0 %)** | |

Total wall-clock for the 30-problem panel: ~15 min on Mac.

**Δ vs pre-fix (the hand_capsule disable's impact)**:

| Level | Pre-fix | Post-fix | Δ |
|---|---|---|---|
| L0 | 5/5 | 5/5 | — |
| L1 | 5/5 | 5/5 | — |
| L2 | 5/5 | 5/5 | — |
| L3 | 5/5 | 5/5 | — |
| L4 | **0/5** | **5/5** | **+5** (100 % recovery) |
| L5 | 3/5 | 5/5 | +2 |

The two correctness fixes together (held-state attach + capsule disable)
move the dataset from 76.7 % to 100 % executable on the test panel.
**This is the reference number for everything that follows.** Any
Phase 1-4 change that drops pass rate below 100/100 on this panel is
a regression and must be either fixed or reverted before merging.

---

## Reachability sweep (252 isolation picks)

Source: `examples/multilevel_blocks_parts_reachability_sweep.py` on
iy ∈ {0, 2} (the rows used by the dataset).

| Shape | Cells × Trials | Success |
|---|---|---|
| cube | 30 × 3 | 90 / 90 |
| flat-x | 28 × 3 | 84 / 84 |
| long-x | 26 × 3 | 78 / 78 |
| **Total** | **252** | **252 / 252 (100 %)** |

Every cell in the dataset-used rows is geometrically reachable in
isolation. Any failure in the dataset must therefore be neighbor-
density-driven, not reach-driven.

---

## Deferred TODOs (not blocking)

* **E2 — `is_path_collision_free` swap in `LinearIKPlanner`.** Replacing
  the per-step `env.is_collision_free` loop in
  `_segment_collision_free` / `_segment_collision_free_n` with one
  `env.is_path_collision_free` call cuts ~2.4× mj_forward calls per
  segment check.  Modest gain (mj_forward share is now only ~8 % of
  per-check after the parked-collision-elision in
  ``state.py::_set_block_collision``) — worth picking up later.
  **Must implement as a domain-local override** (subclass
  ``LinearIKPlanner`` in this domain's package, or wrap it) — do NOT
  edit ``tampanda/planners/linear_ik.py`` since it's shared with
  access19, tabletop, confined_*, and concurrent sessions edit those
  freely.  Path:
  ``tampanda/symbolic/domains/multilevel_blocks/linear_ik_override.py``
  with a ``MultilevelBlocksLinearIK(LinearIKPlanner)`` subclass; wire
  in via the executor's ``self.lik = MultilevelBlocksLinearIK(env, …)``.

## Known failure modes (post-fix code)

1. **Yaw committed too early.** `_pick_top_down` / `_put_top_down`
   select a quat at the approach / high-above phase and lock it for
   descend. When descend would only work with a different yaw (e.g.,
   90 ° rotation to put the wrist body parallel to a neighbor instead
   of perpendicular), the chain dies. Currently masked by the
   `hand_capsule` disable (Phase 4 will re-enable via parallel yaws).
2. **Pre-existing yaw filter only in `pick_upright`.** Other chains
   lack the cheap IK reachability gate at the action anchor. Phase 3.3
   ports the pattern.
3. **Post-action cleanup (`_to_handoff` + `_to_neutral_home`) runs
   even in fast mode.** ~25-30 % of per-check time spent verifying a
   return path that no downstream consumer cares about. Phase 3.1
   removes it.
4. **Per-segment `is_collision_free` does 3 mj_forward each.**
   `is_path_collision_free` (already on the env) does 1 forward per
   step + 1 restore. Phase 3.2 swaps it in.
5. **Dataset declares 763 PDDL objects per problem** (38 blocks +
   725 cells) even though typical plans touch < 30 cells. Bloats the
   GNN inference cost ~4 × vs other domains. Phase 1 targets this.

---

## Auxiliary files

* `examples/multilevel_blocks_solver_probe.py` — mj_forward timing
  across scenes / option configs.
* `examples/multilevel_blocks_count_calls.py` — per-check mj_forward
  call count.
* `examples/multilevel_blocks_check_timing.py` — single-check elapsed
  + share breakdown.
* `examples/multilevel_blocks_attach_smoke.py` — restore-state attach
  unit check (5 fluent families × 1 trial each).
* `examples/multilevel_blocks_state_restore_demo.py` — full state
  restore visual demo (renders before/after-exec/after-restore for
  each plan step).
* `examples/multilevel_blocks_parts_reachability_sweep.py` — per-cell
  per-shape isolation pick sweep.
* `examples/multilevel_blocks_plan_validator.py` — N-per-level full
  plan execution validator.
* `examples/multilevel_blocks_failure_viz.py` — re-renders the seven
  failure cases from the pre-fix validator with annotated targets.
* `examples/multilevel_blocks_l4_regression_probe.py` — bisection of
  the reverted item-3 options block.
* `examples/multilevel_blocks_parts_remap.py` — dataset cell-remap
  script (built but not currently applied; available if a future
  sweep reveals problem cells).
