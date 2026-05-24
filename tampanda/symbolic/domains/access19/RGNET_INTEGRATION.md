# access-19 → rgnet integration handoff

You (the new Claude Code session) are picking up where the access-19
domain ships into the rgnet repo.  The domain is **complete on the
tampanda side**: code, dataset, design notes, debug harness, pre-rgnet
QA all green.  Your job is the rgnet-side wiring (mirroring how
`multilevel_blocks` was added) plus an end-to-end smoke test.

## What access-19 is — 30 second version

* PDDL domain `tabletop-access` in **filter mode** (`pick`, `put`).
* 19 movable objects: 18 red blocker cubes + 1 blue object-of-interest
  (`ooi`).  All uniform 4×4×8 cm.
* Closed-top cubicle scene (Bouhsain HAL 2025, the access-19 variant).
* Cells form two grid regions: `shelf_interior` (7×7 inside the
  cubicle) and `shelf_top` (7×7 open deck above).
* `(pick obj cell)` / `(put obj cell)` actions with a single canonical
  grasp pose per object (palm-+y, `FRONT_QUAT`).
* Five curriculum levels L0–L4; L4 = full 18-blocker return-all (74
  actions).
* 180 instances pre-generated: `train_120`, `val_30`, `test_30`.

Full background: `tampanda/symbolic/domains/access19/DESIGN.md`.

## What's already done on the tampanda side

| Capability | Where | Verified |
|--|--|--|
| Scene builder | `access19/env_builder.py:make_access19_builder` | yes |
| PDDL domain | `access19/pddl/domain.pddl` | yes |
| Bridge | `access19/bridge.py:make_tabletop_access_bridge` | yes |
| State restore (incl. held) | `access19/state.py:restore_state` | `access19_rgnet_qa.py` |
| Chain executors | `access19/chains.py:make_access19_pick_fn,_put_fn` | L4 stress test 5/5 |
| FAST feasibility | `access19/feasibility.py:check_action(_sequence)` | 366/366 agree w/ FULL |
| Parallel pool | `access19/parallel.py:ParallelFeasibilityChecker` | 9.5 calls/sec @ 4 workers |
| Planners | `access19/planner.py:{oracle,astar,phased}_plan` | 5 L4 variants |
| Data writer | `access19/generate_data.py` | `data/access19/` |
| Round-trip replay | — | `access19_replay_from_disk.py` 15/15 |

If anything below seems wrong, re-run `examples/access19_rgnet_qa.py`
locally first — it covers state restore, held restore, and
single/parallel throughput in one shot.

## Public API (use these — don't import internals)

```python
from tampanda.symbolic.domains.access19 import (
    Access19Config,
    make_access19_builder,
    apply_runtime_tweaks,
    make_tabletop_access_bridge,
    set_objects_at_cells,
    make_access19_pick_fn,
    make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    check_action, check_action_sequence,
)
from tampanda.symbolic.domains.access19.parallel import (
    ParallelFeasibilityChecker, _layout_to_state,
)
from tampanda.symbolic.domains.access19.state import (
    restore_state, ground_to_object_cells, held_object_in_state,
)
```

The 19-object roster:
```python
_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]
```

## Reference: multilevel_blocks (what to mirror)

The rgnet plangolin wiring for `multilevel_blocks` is the closest
working example.  Read **all** of these on the rgnet side before you
start writing access19 files:

* `plangolin/domains/multilevel_blocks.py` — domain class
* The dense-reward kit registration call site
* Any rgnet config (`configs/`) that names `multilevel_blocks`
* The smoke / training-launch script for it

Then make a copy → rename → adjust.  Diffs you'll need:

| What | multilevel_blocks | access19 |
|--|--|--|
| Cell namespaces | `parts__r_c`, `stack_LN__r_c` | `shelf_interior__r_c`, `shelf_top__r_c` |
| Movable shapes | 1×1 cube, 2×1 oblong, 3×1 long | uniform 4×4×8 cm (one shape) |
| Held fluents | `held-cube`, `held-flat-x`, `held-flat-y`, `held-upright` | `holding` (single fluent, one arg) |
| Action vocab | `pick`, `put_flat`, `put_upright`, `transform`, etc. | `pick`, `put` (2 actions, that's it) |
| Total movables | varies (templates) | exactly 19, fixed roster |
| Grid pose lookup | `MultilevelBlocksConfig` cell math | `Workspace.pose_for(Cell.parse(cell_id))` |

The single-shape, single-grasp-pose, two-action vocabulary makes
access-19 **considerably simpler** than multilevel_blocks.  If a
plangolin abstraction makes sense for mlb but seems unnecessary here,
trust that instinct — skip it.

## Dataset location

* In the tampanda repo: `data/access19/`
  * `domain.pddl` — symlink target / canonical domain file
  * `{train,val,test}/L{0,1,2,3,4}/config_NNN.pddl`
  * `{train,val,test}/L{0,1,2,3,4}/config_NNN.pddl.plan` — solver-style
    plan (one `(action arg1 arg2 ...)` per line)
* 120 train + 30 val + 30 test = 180 instances total.
* Mix: 16/24/32/32/16 across L0–L4 in train.

Plans were generated using the `phased_plan` planner for L4 and
`astar_plan` for the rest.  Plan lengths: L0 ≈ 2–4, L1 ≈ 6, L2–L3 ≈
8–12, L4 = 70–74.

The dataset is **not** committed to git — regenerate on the target
machine:

```bash
cd <repo>
python -m tampanda.symbolic.domains.access19.generate_data \
    --curriculum-spec train_120 --num-workers 4 \
    --output-dir data/access19
```

This takes ~20 min on a 4-worker pool.

## Concrete steps

### Step 1 — get the code on the target machine

```bash
cd <repo on remote>
git fetch
git checkout feat/multigrid-domains
git pull
conda activate rgnet_fresh
export MUJOCO_GL=egl
```

### Step 2 — sanity-check the tampanda side

Before touching rgnet, prove the domain is wired correctly on this
machine.

```bash
python examples/access19_rgnet_qa.py
```

Expect: 4/4 PASS.  If anything fails here, **stop and ask** — do not
try to fix tampanda-side issues from the rgnet handoff; it means the
checkout / env is broken.

### Step 3 — regenerate the dataset

```bash
python -m tampanda.symbolic.domains.access19.generate_data \
    --curriculum-spec train_120 --num-workers 4 \
    --output-dir data/access19
```

Verify: `find data/access19 -name "*.pddl" | wc -l` → 181 (1 domain
file + 180 problems).

### Step 4 — round-trip a few plans

```bash
python examples/access19_replay_from_disk.py --instances-per-level 1
```

Expect: all 15 (3 splits × 5 levels) PASS.

### Step 5 — write `plangolin/domains/access19.py`

Mirror `plangolin/domains/multilevel_blocks.py`.  Anchor points the
domain class must provide:

1. **Action grounder** — given a PDDL action token from xmimir, return
   the tampanda action tuple `("pick", obj, cell_id)` or `("put", obj,
   cell_id)`.  Cell id format: `shelf_interior__r_c` or
   `shelf_top__r_c` (note the **double underscore**).
2. **State grounder** — given a `(layout, held)` pair, build the dict
   that `restore_state` consumes.  Use `_layout_to_state` from
   `access19.parallel`.
3. **Feasibility hook** — call `check_action` or `check_action_sequence`
   with `fast=True` for training-time pruning.  Pass the cached
   `pick_fn` / `put_fn` / `executor` / `shelf_home` from the env
   factory; do **not** rebuild them per call.
4. **Bridge factory closure** — fresh `make_tabletop_access_bridge`
   each `reset()` (matches the spawn-safe pattern from the
   `DomainBridge` docstring).

A minimal env factory looks like:

```python
def _build_access19():
    builder, ws, cfg = make_access19_builder()
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    cube_half = float(env.get_object_half_size("ooi")[2])
    table_z = ws["shelf_interior"].level_z - cube_half
    executor = _build_executor(env, table_z=table_z,
                               allowed_types=[GraspType.FRONT])
    shelf_home = _solve_access19_staging(env, ws, cfg)
    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
    pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                    cube_half_z=cube_half, lik=lik)
    put_fn = make_access19_put_fn(env, executor, ws, cfg,
                                  cube_half_z=cube_half, lik=lik)
    return env, ws, cfg, executor, pick_fn, put_fn, shelf_home
```

This is also what `examples/access19_replay_from_disk.py:_setup` does
— start by reading that file.

### Step 6 — register the domain in rgnet

Find where `multilevel_blocks` is registered (likely a config block, a
`__init__` import, and/or a `DOMAINS` registry dict).  Add access19
parallel to it.

### Step 7 — smoke test one episode end-to-end

In rgnet, write a one-shot script that:
1. Builds the env via the new domain class.
2. Loads one L1 plan from `data/access19/val/L1/config_*.pddl`.
3. Steps the rgnet env through each action.
4. Verifies the final symbolic state matches the goal.

If this works, you're done — the training loop will work because all
the pieces it depends on (state restore, feasibility, action grounder)
are the same pieces this smoke test exercises.

### Step 8 — launch a tiny training run

Use the same launch script flow as `multilevel_blocks`, but with
the access-19 domain.  Run 10–100 episodes only — confirm rewards
shape sensibly and there are no NaNs / crashes.  **Do not** kick off
a full training run from this handoff; report back first.

## Throughput / sizing notes

From the pre-rgnet QA (`access19_rgnet_qa.py`):

* Single env, fast feasibility: ~360 ms mean, ~470 ms median per call.
* 4-worker parallel pool: ~105 ms wall per call, ~9.5 calls/sec
  aggregate.  Sub-linear scaling (Python-side IK + chain logic is the
  bottleneck, not pure compute).

For an episode replay with N actions, expect roughly
`N × 350 ms` of feasibility-check wall time on a single worker.

## Known constraints / gotchas

* **macOS sim viz requires `mjpython`** — but rgnet on remote uses
  EGL, no viewer needed.
* **Always set `MUJOCO_GL=egl`** on headless remotes.
* **State restore now supports held states** (`on_held="attach"`,
  default).  Do not work around this — use the canonical attach path.
* **PDDL domain is filter mode**, not face mode.  If you see action
  tokens with grasp face arguments (e.g. `(pick obj cell FRONT)`),
  something is wrong upstream.
* **`safe_z = top_deck_z + 0.10`** in `chains.py`.  Do not lower —
  this was the root cause of the L4 return-all 13/74 disagreement bug
  (`DIVERGENCES.md`).
* **One canonical grasp per object** — palm-+y, `FRONT_QUAT =
  [-0.5, 0.5, 0.5, 0.5]`.  No grasp-selection branching in the
  rgnet-side action grounder.

## When you're done

1. Push the rgnet branch.
2. Report back:
   * Step 4 (round-trip replay) result.
   * Step 7 (smoke test) result, with the exact action sequence
     length and wall time.
   * Step 8 (tiny training run) result, with reward values for the
     first ~10 episodes.
3. Mark task #99 (`rgnet: add plangolin/domains/access19.py`) done in
   the tampanda repo's `TaskList`.

If you hit a wall, the fastest unstick is to re-read
`tampanda/symbolic/domains/access19/DESIGN.md` and to grep for how
`multilevel_blocks` solves the same problem.  90% of the integration
is "do what mlb does, but simpler".
