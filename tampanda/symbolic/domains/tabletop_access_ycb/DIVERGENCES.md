# tabletop_access_ycb — divergences

From the source paper (Ait Bouhsain et al., *Long-Horizon TAMP with
Learning-Based Geometric Reasoning*, HAL 2025) and from the parent
`tabletop_access:access` domain it forks.

## vs the parent `tabletop_access:access`

| | parent `access` | this fork |
| --- | --- | --- |
| Objects | uniform placeholder boxes (or YCB-proxy *boxes*) | **real GSO/YCB meshes** (validated roster) |
| Grid | `cell_size` 6 cm, single-cell objects | **3 cm, multi-cell rectangular footprints** |
| Regions | 4 (floor_left/right, middle_deck, top_deck) | **2 (middle_deck, top_deck)** |
| PDDL actions | `(pick ?o ?c)` / `(put ?o ?c)` | **per-size `pick_<W>x<H>` / `put_<W>x<H>`** (rigid-block cell params) |
| `level_z` | deck surface + 0.05 m item-half ref | **deck surface** |
| Mass | template-set (0.05 kg) | mesh-derived → **overridden to 0.05 kg at runtime** |
| Cell roster | all reachable cells (~68) | **dynamic per-problem** (plan-touched + margin) |

The shelf body, FRONT-grasp chain mechanics, FAST/FULL feasibility design,
and canonical-restore pattern are reused (forked, not edited).

## vs the paper

Inherits the parent's paper-divergences (single-arm Franka; discrete cells;
STRIPS goal instead of learned point-cloud goal; one canonical grasp per
object rather than a sampled grasp DB) and adds:

* **Object set** is the validated graspable YCB subset, not the paper's
  full YCB palette — items with protruding handles (mug, pitcher), rollers
  (peach, pear), oversize (wood blocks), and tall/tapered items (mustard,
  bleach) are excluded because the single FRONT palm-+y grasp + closed-z
  shelf window can't handle them reliably.
* **Footprints are axis-aligned rectangles** at the chosen 3 cm grid (round
  approximation degenerates to rectangles below ~5–7 cells across; see
  DESIGN.md).  The paper reasons over continuous SE(3) placement.
* **No return-all / blocker-home goals** — the task is OoI-to-top only, so
  plans are short (2–8 actions).  Relocated blockers stay where the planner
  parks them.

## What we made easier

* Two regions instead of three (dropped the fragile floor compartments —
  see DESIGN.md "Why two regions").
* Identity-orientation, cell-aligned, single-canonical-yaw placement
  (no in-plan reorientation).
* Mass/inertia homogenised to the placeholder values for gentle contacts.

## What we did NOT make easier

* Real mesh collision geometry (CoACD convex pieces) — the grasp + the
  multi-cell footprint overlap are computed from the actual collision mesh.
* Tight packing on a fine grid with heterogeneous footprints — the whole
  point of the fork.
* The symbolic↔geometric contract: every dataset action is FAST-feasible by
  construction (canonical restore + chain check), with a 10% FULL spot-check.

## FAST soundness boundary (known limitation)

FAST is a fast, **near-sound** feasibility filter, not a perfect one.
Dense-scene stress (`examples/ta_ycb_stress.py`, 30 layouts / 120 probes):
**97.5% FAST==FULL agreement**, with ~1.7% **FAST-accept / FULL-reject**
on picks.  Diagnosed cause (`examples/ta_ycb_diagnose_optimism.py`): the
Franka hand body sweeps ~10 cm in +y (palm-+y) behind the grasp point;
the hand_capsule collision proxy is shrunk to 2 cm to fit the shelf, so
the collision *check* under-represents that sweep — FAST (check-only,
teleport) misses a behind-neighbour that FULL *physics* (real hand
geometry) knocks.  This is the same hand-sweep limitation the parent
`access19` documented and did not fully close; gripper-clearance
inflation can't fix it (the reach mismatch is ~10 cm, not a radius graze).

**Decision (deliberate):** keep FAST as the fast near-sound filter — the
**dataset is made sound by the 100% FULL gate** in `generate_data` (every
plan re-validated under FULL, reject+resample).  The residual FAST
optimism only affects rgnet's *training-time* pruning (≈1.7% false-accept
on dense picks — label noise for a learned heuristic), not the written
data.  Closing it fully would need either a structural behind-sweep
pick-precondition (adds conservatism / limits tightness) or a
micro-physics-settle in FAST (removes its physics-free speed); neither was
judged worth it.

## Validated facts (this machine, M-series Mac)

* FAST==FULL: 11/11 roster objects, single pick-middle + put-top; 97.5% on
  120 dense-scene probes (boundary above).
* Reachability (FAST): middle_deck pick 80/80 anchors, top_deck put 79/80
  (extreme back-right anchor feasibility-rejected).
* Every written plan is FULL-executable (100% FULL gate, reject+resample).
* Generated plans are symbolically valid (UP sequential simulator:
  applicable + goal reached).
* FAST per-check ≈ 70–165 ms vs FULL ≈ 1.5–3.4 s (20–30×).
