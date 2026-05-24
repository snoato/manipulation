# access19 — Design Divergences

Notes on where access-19 diverges from sister domains (tabletop_access,
multilevel_blocks) and from the FAST↔FULL contract.

## Hard fork from tabletop_access (Phase 1)

* `Access19Config` renamed from `TabletopAccessConfig` (the 3-tier
  `access` variant retains its own copy in `tabletop_access/`).
* `interior_height_z = 0.24` (was 0.20 in tabletop_access).  Visual
  validation in `examples/access19_phase0b_failure_viz.py` showed the
  forearm clipping the closed-top cubicle's top wall at 0.20.  0.28
  fully clears the elbow but raises the top deck enough that the
  Franka can't reach the +x edge columns at safe_z.  0.24 is the
  geometric compromise.
* `_SAFE_Z_ABOVE_SHELF_TOP = 0.08` (chains.py).  Reverted from a
  short-lived 0.13 bump; the bump fixed col-5 pick_deck IK in
  isolation but pushed put_deck reach over the limit.

## Cubicle pick mechanics

The chain row-steps through the cubicle (front-clearance ordered)
because the closed-top design forbids top-down grasps.  Therefore
**any blocker at (col, iy_int) requires (col, 0..iy_int-1) to be
empty** before it can be picked.  Templates obeying the
"subset-of-18" rule (cube cols 1/3/5, rows 0..5) preserve this
invariant; the planner clears front rows first.

## OoI corridor reservation (Phase 3)

`planner.py:_h_misplaced` adds a +4 admissible penalty whenever a
blocker sits in the OoI goal cell's column at iy_top > goal.iy.
Reason: returning that blocker requires the held block's pick_deck
traverse to cross the OoI cube on the deck, triggering the
`hand_c`-vs-cube collision visualised in
`examples/access19_phase0b_failure_viz.py` (the
`/tmp/access19_phase0b_failure.mp4` artefact shows substep 14/16
interpenetration).  The heuristic teaches A* to budget for the
required OoI shuffle, making L4 return-all tractable from the search
side.

## FIXED: FAST↔FULL divergence at L4 return-all (Phase 5)

**Root cause** (found via `state_audit.py` + per-substep EE probe in
`examples/access19_action39_probe.py`): the chain's `_SAFE_Z_ABOVE_
SHELF_TOP = 0.08` placed the EE at exactly cube-top level during the
top-deck traverse.  The gripper finger collision geometry extends
~2 cm below the EE site, so the fingertips scraped the top of every
deck cube the gripper passed over.  At action 39 of an L4 return-all
plan, the traverse to col 1 dragged blocker_13 forward by 5 cm into
blocker_12, knocked blocker_12 +2.8 cm in x and y, and left
blocker_13 tilted 28° in the gripper.  Downstream chain IK probes
then rejected the corrupted state and the plan failed by action 42.

**Fix**: bumped `_SAFE_Z_ABOVE_SHELF_TOP` to 0.10 so the gripper
clears cube tops by 2 cm during traverse.

Validation: L4 return-all FULL replay now passes **0/74 disagreements**
(was 13/74 before the fix).

The state-audit protocol (`state_audit.py` + `access19_state_audit.py`)
and the per-action probe (`access19_action39_probe.py`) are kept in
the tree as the diagnostic harness for future chain-level issues.

## Earlier (incorrect) diagnoses, kept for context

The validator in `examples/access19_full_validation.py` confirms
FAST↔FULL agreement on:

* **L0–L3 with full-return goals** (≤16 actions): per-action agreement
  100%.
* **L4 OoI-only** (38 actions): 38/38 agree, FULL replay PASS.

For **L4 with full-return** (74 actions, includes multi-OoI staging),
the per-action agreement is partial — roughly **13/74 actions** that
FAST accepts are rejected by FULL.  The disagreements cluster around:

* `put_interior` row-step actions that return a blocker to its
  original cell mid-plan.
* `pick_deck` traverse actions in the Phase 2 return leg, after many
  prior actions have populated the top deck.

What's contributing:

1. **Cube geometry is physically unstable.** Each blocker is a
   4×4×8 cm prism — tall, narrow base.  Each FULL `put_deck` release
   imparts a tiny rotational torque (gripper-open + detach), and the
   cube can wobble or visibly tilt while subsequent arm motion
   continues nearby.  By action 40 the cubes on the deck are
   noticeably off-axis (image artefact in this folder's git history
   shows ~5–10° tilts).
2. **Snap-between-actions only mitigates the symbolic state.**
   `check_action_sequence` now snaps every placed cube to its
   symbolic cell centre + identity quat before every action — but
   this fires AT the start of action N, not during action N-1's
   execution.  Action 40's row-step plan sees a clean state at
   start, so why does it still fail?  See point 3.
3. **The chain's `_execute` under FULL drives the arm via real
   controller + physics.**  The controller doesn't perfectly track
   the planned path; per-substep deviations accumulate inside a
   single chain call (lift → traverse → descend → grasp).  By the
   time the chain reaches its next IK probe, the arm config differs
   from the planning-time prediction, so the next plan starts from
   an off-basin pose and rejects.

This is a chain-level limitation (`chains.py`'s row-step / lift /
traverse / descend sequence is brittle to controller imperfection
inside a single chain call), not a flaw in the FAST checker.
FAST's idealised teleport bypasses controller tracking entirely.

## Attempted fix: controller-convergence retry (Phase 5)

`chains.py:_execute` now verifies the arm reached `path[-1]` within
`_CONVERGE_TOL = 0.02 rad` after every `execute_path + wait_idle`,
re-issuing the last waypoint up to `_MAX_RETRIES = 3` extra cycles
if not.  This addresses controller tracking error specifically.

It did NOT change the L4 return-all disagreement count.  Diagnostic
walk-through:

* The action that fails under FULL (e.g., #40: `put blocker_13 to
  shelf_interior__3_4`) fails in chain PLANNING (`row-step plan
  failed`), not in execution.
* The IK probe for the next row-step pose runs after the chain's
  prior `_execute` call has driven the controller + physics.  Even
  though the snap-between-actions ensures cubes are at symbolic
  positions at the START of the action, the chain's own internal
  `execute_path` (approach lerp) runs physics that nudges cubes
  mid-action.
* The next IK probe then sees the nudged cubes and rejects the
  plan as colliding — but FAST never ran the physics, so its
  internal probes always saw clean state.

So the residual divergence is **mid-chain cube physics**, not arm
tracking error.  The convergence retry helps the arm side but
doesn't address this.

## State-audit findings (Phase 5 deep dive)

`tampanda/symbolic/domains/access19/state_audit.py` instruments every
chain transition; `examples/access19_state_audit.py` runs an L4
return-all plan via FULL with full telemetry.  The audit JSON shows
that the first FULL-only failure is **execution quality**, not drift:

| Action | Description | Held tilt | Max placed-cube tilt | Unplanned cube movements |
|--------|--|----------:|---------------------:|--|
| 37 | pick blocker_16 from top(5, 0) | 5.16° | 0.19° | blocker_15: 9.68 mm |
| 38 | put blocker_16 to int(3, 5)    | 0° | 0.19° | — |
| **39** | **pick blocker_13 from top(1, 0)** | **29.51°** | **19.03°** | **blocker_10: 18.34 mm; blocker_12: 42.33 mm** |
| 40 | put blocker_13 to int(3, 4) | (chain rejects: row-step plan failed) | — | — |

So when the chain's `pick_deck` grasps `blocker_13` at top `(1, 0)`:

1. The Franka hand_c body extends ~10 cm in +y from the EE site under
   FRONT_QUAT (palm-+y).
2. `blocker_12` at `(1, 1)` — the +y-neighbour cell — is right where
   the hand body sweeps during descent / grasp / lift.
3. The hand body physically knocks `blocker_12` by **42 mm** and
   tilts the picked `blocker_13` by **30°** as the gripper closes
   off-axis.
4. `blocker_10` at `(1, 2)` gets a downstream tap of 18 mm.

The chain's IK collision check has too little margin to detect the
hand-vs-neighbour proximity (~mm scale clearance), so it accepts the
plan; FULL physics manifests the contact.  Snap-between-actions
doesn't help because the corruption happens *during* the action, not
between.

## Attempted precondition (NOT kept enabled)

Added a "+y neighbour empty" precondition to `pick_deck`:
`pick_deck(col, iy_top)` requires `(col, iy_top + 1)` empty.  This
correctly rejects action 39's pick of `(1, 0)` while `(1, 1)` is
occupied.

**Outcome**: A* then can't find a feasible Phase-2 ordering for L4.
The precondition forces all top-deck picks to start at iy_top=6 (no
+y neighbour), but at the chain's palm-+y geometry, iy_top=6 targets
are at the Franka's reach limit — chain reports `traverse plan
failed`.  All picks dead-end, A* exhausts the frontier.

The fix has been reverted to keep L4 OoI-only working.  See
`tampanda/symbolic/domains/access19/chains.py:pick_fn` for the note.

## Possible engineering fixes (NOT implemented)

* **Palm-down `pick_deck` (recommended for L4 return-all).**  The
  top deck is open above — pick the cube from straight above with
  the gripper rotated palm-down (fingers pointing −z).  In that
  orientation the hand body extends in −z (above the EE) not in
  +y, so adjacent deck cells aren't intruded upon.  Requires a new
  approach chain (palm-down lift, traverse-at-altitude, descent),
  parallel to but separate from the existing palm-+y pick_interior /
  pick_deck.  Combined with the +y precondition re-enabled, this
  should make L4 return-all feasible end-to-end.
* **Cubic blocks (4×4×4 cm).**  Stable geometry; no tilt wobble.  But
  the current 8 cm cube height was chosen so that grasping at
  cube-top keeps the wrist's link7 capsule above the cubicle floor.
  Cubic blocks would require redesigning the grasp height
  clearance.
* **Kinematic-snap-at-detach in chains.py.**  After
  `_detach_and_open` the chain currently leaves the block at the
  attached pose and lets physics settle.  Adding
  `env.set_object_pose(obj, target_cell_centre)` immediately after
  detach would eliminate the post-place wobble.  Sim-only fix —
  doesn't transfer to a real robot.
* **Tighter IK collision margin.**  The IK check rejects only if
  `contact.dist < 0.001` (1 mm penetration).  Bumping to e.g. 5 mm
  would catch the hand-vs-neighbour proximity at FAST planning
  time and force the planner to find alternatives.  Risk: rejects
  some currently-feasible plans.

## Implications for the dataset

* **L0–L3 instances are FULL-validated** — safe for real-robot use.
* **L4 OoI-only instances are FULL-validated** — also safe.
* **L4 return-all instances generated by FAST should not be assumed
  FULL-executable.**  The data-gen pipeline writes them with the
  FAST agreement guarantee; consumers that need FULL-replayable plans
  must either restrict to L0–L3 / L4-OoI-only, or pay the
  ~700 ms/action cost of generating with FULL throughout.

The validation artefact lives at
`examples/access19_full_validation.py`; the per-instance log on
disagreement shows exactly which actions diverge.

## Future fixes (not done)

* Make the chain's `_execute` more robust to controller tracking
  errors — e.g., a closed-loop "did we reach the target?" check
  with a re-plan if drift exceeds a threshold.
* Increase the IK margin so small post-execution drifts don't push
  configs out of the basin.
* Investigate whether the controller's `_advance_delta_override`
  override (`= 0.01` during traverse) is too tight and lets the
  controller move past targets.
