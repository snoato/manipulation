# tabletop_access:access — design notes

The HAL `access` problem (Ait Bouhsain et al.): a single Panda must
retrieve an object-of-interest (OoI) from a **free-standing 3-tier
shelf** where other objects block access, and deliver it to a target on
the top deck.  Trained on few blockers, generalizes to more.  Forked
intent from `access-19` (the closed cubicle); shares the
`tabletop_access` package + PDDL family.

## Workspace

Free-standing shelf (no table), four `GridRegion`s, no cross-grid
adjacency (levels are traversed by the chain, not the symbolic layer):

* `floor_left` / `floor_right` — the two halves of the bottom
  compartment (split by a separator), reached from the open front.
  Each effectively a single reachable column × 5 rows (the inner
  column is `excluded_cells`).
* `middle_deck` — main shelf, 6 × 5, front + top access.
* `top_deck` — open upper shelf, 6 × 5 (two back corners excluded);
  the OoI goal lives here (centre cell).

**Reachability is fully proven** (`examples/access_reach_map.py`, FAST):
middle 30/30, top 28/28, floor 5/5 each — every reachable cell does both
pick and put.  No cells beyond the pre-existing exclusions.

## Object roster

`make_access_builder(n_uniform_blockers=N)` builds OoI + N uniform
generic blocker boxes (short + graspable, like access19's cubes) — the
dataset uses N=10 (supports 3 occluders + up to 7 clutter).  The default
(no `n_uniform_blockers`) builds the YCB-proxy mix for visual/demo use.

## Execution chain (`chains.py:make_access_chains`)

One front-approach shape for all four regions (open-front shelf — no
closed-top traverse).  Ported from the access19 template; key mechanisms:

* **Per-level hand-off + teleport.** Each level has its own IK-solved
  staging pose (3 seeds, best residual; per-region centre-x; the floor
  hand-off is staged high in the compartment so a held object clears the
  world floor).  The chain teleports to the target level's hand-off (held
  object rides via the attachment hook) then does the in-level approach —
  this fixed floor/top reach.  Teleporting between hand-off points is an
  accepted simplification; the collision-checked motion that matters for
  feasibility happens within a level.
* **Capture grasp offset.** `executor._held_grasp_dz` (object-centre − EE
  at grasp) is captured at pick and used to place correctly across
  regions; a restored `holding` state sets the same canonical offset.
* Convergence retry in `_execute`; held block NOT collision-exempt.

## Feasibility (`feasibility.py`)

Feasibility is **derived from the full chain** (run it; feasible ⇔ it
succeeds).  `check_action` / `check_action_sequence`:

* `fast=False` (FULL) — real physics, ground truth.
* `fast=True` — `fast_mode` teleports execution and **caps IK iters to
  80** (an unreachable gripper-flip quat otherwise runs MinkIK's full
  1000-iter cap, ~290 ms); a reachable solve converges in <10 iters.
  FAST is ~8–29 ms/check vs ~0.4–1.1 s FULL (16–84×).

**FAST == FULL** is validated (12 scenarios + 48 random checks) **only
under canonical `restore_state`** — without it, marginal cells flip with
history.  So `solve()`/data-gen restore state before every check.

### Blocking rules (empirical; `access_collision_probe.py`)

Every blocked-pick collision is the gripper hitting the blocker, i.e. the
palm-+y swept volume:

* `structurally_blocks(b, t)` (PICK): same-column-front, or adjacent-
  column front/same-row (hand/link7 corridor + finger span).
* `blocks_put(b, t)` (PUT): adds same-column-**behind** (the wrist
  extends back over the target).  Used to keep relocation scratch cells
  clear of the OoI goal put.

## State restore (`state.py:restore_state`)

Canonical, history-independent: objects snapped to cell centres + identity
quat, velocities zeroed, arm at staging-home, attachment cleared; a held
fluent attaches the object at the canonical grasp offset (sets
`_held_grasp_dz`).

## Templates + planner

* `templates.py` — `make_instance` builds OoI + occluders (OoI column
  front; capped at 3 — deeper stacks need too many reliable top scratch
  cells) + non-blocking clutter + optional **goal-clutter** (a blocker at
  the OoI's target).  `sample` / `sample_by_counts` / `random_layout` for
  structured + random instances.
* `planner.py:solve` — feasibility-guided (no search blow-up): clears the
  transitive set of blockers in the way of the OoI pick **or** the goal
  put, relocating each to a feasibility-verified scratch cell across **all
  regions** (the top deck alone is too cramped for wide boxes), then
  delivers the OoI.  Returns `None` if stuck (caller resamples).

## Dataset (`generate_data.py`)

Structural generalization: train on few blockers (active occluders),
generalize to denser scenes (the paper's object-count-scaling axis).

| split | occluders | clutter | goal-clutter | count |
|---|---|---|---|---|
| `train/` | 1–3 | 0–2 | rare | 300 (~30% random) |
| `val/` | 1–3 | 0–2 | rare | 50 |
| `eval_ood/` | 1–3 | 3–7 (→ up to ~10 total) | ~50% | 50 |

* `solve()`'s per-action FAST checks (from canonical restore) are the
  soundness gate; a **10% FULL spot-check** guards against FAST leniency.
  Unsolvable samples are reject+resampled.
* PDDL problems emit only the **present** movables + all reachable cells +
  within-grid `(adjacent …)` edges (the GNN's spatial signal — pick/put
  don't use them).  `domain.pddl` (single file — goals are concrete
  `(occupied …)`, no derived-predicate variant needed) is bundled.

Generate:
```
python -m tampanda.symbolic.domains.tabletop_access.generate_data \
    --output-dir data/access --train 300 --val 50 --eval 50
```

## Diagnostic / viz scripts (`examples/`, not committed)

`access_exec_probe`, `access_reach_map`, `access_feasibility_check`,
`access_fast_full_agreement`, `access_collision_probe`,
`access_feas_timing`, `access_plan_validate`, `access_examples_viz`
(initial+final grid), `access_exec_viz` / `access_gather_viz` (mp4).
