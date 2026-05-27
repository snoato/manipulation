# rgnet integration handoff — `tabletop_access_ycb`

Quick handoff for the rgnet session.  This domain integrates the **same
way as `access-19`, `confined_shelf`, `tabletop`, and the parent
`tabletop_access:access`** — read
`tampanda/symbolic/domains/access19/RGNET_INTEGRATION.md` and the parent
`tabletop_access/RGNET_HANDOFF.md` as the closest templates.

## TL;DR

- New PDDL domain **`tabletop-access-ycb`**: the HAL `access` retrieval
  task (OoI from a cluttered middle shelf → top deck), but with **real
  GSO/YCB meshes, a 3 cm grid, multi-cell footprints, and per-footprint-
  size `pick_<W>x<H>` / `put_<W>x<H>` actions**.
- Dataset is generated locally + **gitignored** — regenerate on the target
  machine, then point the loader at it.
- STRIPS only (no derived predicates) → **one `domain.pddl`**, no dual-file
  variant.  Goals are concrete `(occupied …)`.

## PDDL schema (what the loader / encoder sees)

```
domain tabletop-access-ycb   (:requirements :strips :typing)
types:      cell  movable  direction
constants:  north east - direction
predicates: (adjacent ?dir ?c1 ?c2) (occupied ?c ?o) (empty ?c)
            (holding ?o) (gripper-empty) (fp_2x2 ?o) (fp_2x3 ?o) … (fp_4x4 ?o)
actions:    pick_<W>x<H> / put_<W>x<H>   (one pair per footprint size)
goal:       (and (occupied <goal_cell_i> ooi) …)   ; the OoI's footprint cells
```

- A `put_<W>x<H>` has **W·H cell parameters** pinned into a rigid block by
  `(adjacent east/north …)` static preconditions; it marks all W·H cells
  occupied.  **The encoder must ingest the `adjacent` edges** (the GNN
  spatial signal) — same as `tabletop`/`access-19`.
- `(fp_<W>x<H> ?o)` is a static per-object marker selecting its schema.
- **Plans are grounded per-size actions**: a plan line is
  `(put_3x3 ooi <9 cell ids in domain param order>)`.  The cell order is
  i-outer, j-inner; the **SW anchor = first cell** (`pddl_gen.grounded_plan_action`
  is the writer / reference for the order).
- Object count varies per problem (present movables only) + a **dynamic
  per-problem cell roster** (plan-touched cells + 1-ring margin, ≈100, not
  the full 240).  Keep the encoder size-agnostic.

## Dataset

Regenerate (~run on a workstation/SLURM; FULL spot-check ≈ 6 s/action):
```
python -m tampanda.symbolic.domains.tabletop_access_ycb.generate_data \
    --output-dir data/access_ycb --train 300 --val 50 --eval 50
```
Layout: `domain.pddl` + flat per-split dirs `train/ val/ eval_ood/`, each
instance a `<split>_<NNNN>.pddl` + `.pddl.plan` (grounded actions + a
`;`-metadata trailer: `n_occ`, `n_clutter`, `n_objects`, `plan_len`).
`eval_ood` is the denser-clutter generalization probe.

Plans are FAST-feasible by construction, 10% FULL-spot-checked, and
symbolically valid (UP sequential simulator: applicable + goal reached).

## If rgnet calls back into tampanda feasibility

```python
from tampanda.symbolic.domains.tabletop_access_ycb import (
    make_tabletop_access_ycb_builder, apply_runtime_tweaks,
    make_tabletop_access_ycb_bridge, make_ycb_access_chains,
    restore_state, build_setup, compute_all_footprints,
)
from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import (
    check_action, check_action_sequence,
)
```

- `build_setup(scratch_dir)` returns everything (env, workspace, config,
  **footprints**, executor, pick_fn, put_fn, home_qpos) — the one-call
  entry point.  `make_fast_oracle(setup)` gives `feasible(layout, held,
  action)`.
- **Footprints are required** by the bridge / state / feasibility (they own
  the cell↔centroid mapping).  Compute once via `compute_all_footprints`
  (needs the built env — mesh verts).
- `restore_state(env, ws, state, object_ids, footprints, …)` is canonical /
  history-independent — call before each check; FAST==FULL only holds under
  it.
- Action grounding: a PDDL action token `put_3x3 ooi c0 … c8` → tampanda
  tuple `("put", "ooi", <SW anchor = c0>)`.  The bridge's per-size
  executors already do this internally.
- **Present-only objects.** The MuJoCo scene compiles all 11 roster bodies
  (absent ones parked off-screen).  The written problem files declare only
  the PRESENT subset (correct — verified).  If you call
  `bridge.ground_state(objects)` at runtime, pass **only the present
  movables** in `objects["movable"]` — parked objects yield no `occupied`
  facts but would otherwise show up as isolated GNN nodes.

## Gotchas (all previously bit other domains)

- **pymimir lowercases all symbols** — cell ids (`middle_deck__3_2`),
  object/predicate/action names (`put_3x3`, `fp_3x3`) — parse
  case-insensitively.
- **Variable object + cell counts** per problem; no fixed maximum.
- The action generator emits type-correct but **geometrically impossible**
  candidates — the tampanda feasibility prefilters classify them cleanly;
  don't expect PDDL statics to have ruled them out.
- **Grid is fixed** for this dataset (3 cm); any geometry change invalidates
  the cells/adjacency/footprint-sizes in existing PDDLs → regenerate, and
  re-run `examples/ta_ycb_validate.py`.
- **Mass override is runtime** (`apply_runtime_tweaks`) — call it after
  `build_env`, before any interaction (raw mesh masses are 0.09–5.7 kg).
- Mesh body-origins are not AABB-centred — use `ObjectFootprint.place_pose`
  for placement, never `surface + half_z` on the body origin directly.

## Status

tampanda side complete + validated (Phases 1–5).  See `DESIGN.md` for the
full design and `examples/ta_ycb_validate.py` for the contract harness
(scene/mass, occupancy round-trip, FAST==FULL, reachmap).  Remaining:
run data-gen at scale, rsync to the shared FS, and write the rgnet-side
domain class.
