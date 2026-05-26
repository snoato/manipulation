"""Action feasibility for the tabletop_access:access shelf.

Feasibility is DERIVED FROM the full chain execution (the single source
of truth in ``chains.py``): an action is feasible iff running the chain
succeeds.  Two modes share the identical chain logic — only the path
execution differs:

* ``fast=False`` (FULL) — real physics.  Ground truth.
* ``fast=True``  (FAST) — ``env.execute_path`` / ``wait_idle`` /
  ``_wait_gripper_*`` are monkey-patched to a kinematic teleport, so
  every IK probe + collision-checked plan still runs but physics is
  skipped.  An optimistic filter — validate generated plans with FULL.

Action format (filter-mode PDDL)::

    ("pick", obj_name, cell_id)
    ("put",  obj_name, cell_id)

A ``put`` assumes the object is already held (a preceding ``pick``);
callers running a single ``put`` in isolation must set up the held
state first.
"""
from __future__ import annotations

import contextlib
from typing import Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

# Surface offset baked into a GridRegion's level_z in the access
# workspace (mirror of chains.py / reachability.py).
_ITEM_HALF_Z_REF = 0.05

# IK iteration cap during FAST checks.  MinkIK.max_iters defaults to
# 1000; an UNREACHABLE target (e.g. the flipped gripper quat at an
# occluded cell) iterates the full cap (~290 ms) before giving up,
# which dominates the cost of an infeasible pick.  A reachable target
# converges in <10 iters, so capping low makes a failing solve bail
# fast without affecting feasible results.  FULL keeps the full cap
# (it drives real motion).  Validate FAST==FULL when changing this.
_FAST_IK_MAX_ITERS = 80


@contextlib.contextmanager
def fast_mode(env, executor):
    """Swap physics-advancing calls for kinematic teleports.  Restores
    originals on exit.  Mirror of access19's ``_fast_env``."""
    orig_execute_path = env.execute_path
    orig_wait_idle = env.wait_idle
    orig_closed = getattr(executor, "_wait_gripper_closed", None)
    orig_open = getattr(executor, "_wait_gripper_open", None)
    orig_max_iters = getattr(env.ik, "max_iters", None)

    def _fast_execute_path(path, planner, step_size=None, **kwargs):
        if not path:
            return
        last = np.asarray(path[-1], dtype=float)
        env.data.qpos[: len(last)] = last
        env.data.qvel[: len(last)] = 0.0
        if env.controller is not None:
            env.controller.stop()
        mujoco.mj_forward(env.model, env.data)
        if getattr(env, "_attached", None) is not None:
            env._apply_attachment()
            mujoco.mj_forward(env.model, env.data)

    def _noop(*a, **k):
        return

    def _fast_closed(*a, **k):
        env.data.qpos[7] = 0.02
        env.data.qpos[8] = 0.02
        mujoco.mj_forward(env.model, env.data)

    def _fast_open(*a, **k):
        env.data.qpos[7] = 0.04
        env.data.qpos[8] = 0.04
        mujoco.mj_forward(env.model, env.data)

    env.execute_path = _fast_execute_path
    env.wait_idle = _noop
    if orig_closed is not None:
        executor._wait_gripper_closed = _fast_closed
    if orig_open is not None:
        executor._wait_gripper_open = _fast_open
    if orig_max_iters is not None:
        env.ik.max_iters = _FAST_IK_MAX_ITERS
    env._fast_mode = True
    try:
        yield
    finally:
        env.execute_path = orig_execute_path
        env.wait_idle = orig_wait_idle
        if orig_closed is not None:
            executor._wait_gripper_closed = orig_closed
        if orig_open is not None:
            executor._wait_gripper_open = orig_open
        if orig_max_iters is not None:
            env.ik.max_iters = orig_max_iters
        env._fast_mode = False


def structurally_blocks(blocker_cell: str, target_cell: str) -> bool:
    """Cheap structural test: does an object at ``blocker_cell`` occlude a
    front pick/put of ``target_cell``?  Derived empirically (FULL=FAST,
    `access_collision_probe.py`) — it IS the palm-+y gripper's swept
    volume: the hand/link7 sweeping in from the open -y face (same column,
    in front) plus the finger span sideways (adjacent column, front or
    same row).  Same region only (levels don't occlude each other).

    Used CONSTRUCTIVELY to place occluding blockers and to order plan
    clearing; the constructed plan is still feasibility-validated, so a
    minor rule imperfection is absorbed by reject+resample.
    """
    b, t = Cell.parse(blocker_cell), Cell.parse(target_cell)
    if b.region != t.region:
        return False
    dx = abs(b.ix - t.ix)
    if dx == 0:
        return b.iy < t.iy                  # same column, in front
    if dx == 1:
        return b.iy <= t.iy                 # adjacent column, front or same row
    return False


def blocks_put(blocker_cell: str, target_cell: str) -> bool:
    """Like ``structurally_blocks`` but for a PUT target — stricter: a
    same-column object BEHIND the target also blocks it (the palm-+y wrist
    extends back over the target during the place), per the feasibility
    sweep (`access_feasibility_check.py`).  Used to keep relocation scratch
    cells clear of the OoI's goal put."""
    b, t = Cell.parse(blocker_cell), Cell.parse(target_cell)
    if b.region != t.region:
        return False
    dx = abs(b.ix - t.ix)
    if dx == 0:
        return b.iy != t.iy                 # same column, front OR behind
    if dx == 1:
        return b.iy <= t.iy                 # adjacent, front or same row
    return False


def put_target(env, workspace: Workspace, obj_name: str, cell_id: str) -> np.ndarray:
    """World-frame resting centre for ``obj_name`` placed at ``cell_id``."""
    cell = Cell.parse(cell_id)
    region = workspace[cell.region]
    cx, cy, _ = workspace.pose_for(cell)
    half_z = float(env.get_object_half_size(obj_name)[2])
    return np.array([cx, cy, (region.level_z - _ITEM_HALF_Z_REF) + half_z])


def check_action(env, workspace, executor, pick_fn, put_fn,
                 action: Tuple[str, str, str], *, fast: bool) -> bool:
    """Return whether ``action`` is feasible by running the chain.

    ``action`` = ``("pick"|"put", obj_name, cell_id)``.  The scene
    (object placements + gripper/held state) must already reflect the
    pre-action world; this just runs the chain and reports success.
    """
    kind, obj_name, cell_id = action

    def _run() -> bool:
        if kind == "pick":
            pos = np.asarray(env.get_object_pose(obj_name)[0])
            return bool(pick_fn(obj_name, cell_id, pos))
        if kind == "put":
            return bool(put_fn(obj_name, cell_id,
                               put_target(env, workspace, obj_name, cell_id)))
        raise ValueError(f"unknown action kind {kind!r}")

    if fast:
        with fast_mode(env, executor):
            return _run()
    return _run()


def check_action_sequence(env, workspace, executor, pick_fn, put_fn,
                          source_layout: Dict[str, str], actions, object_names,
                          *, fast: bool, home_qpos=None):
    """Validate a whole plan action-by-action.

    Before each action the world is restored to the CURRENT symbolic
    state (objects snapped to cells, held attached canonically) so every
    check is history-independent, then the symbolic state is evolved by
    the action.  Returns ``(ok, failed_action_or_None)``.
    """
    from tampanda.symbolic.domains.tabletop_access.state import restore_state
    layout = dict(source_layout)
    held = None
    for action in actions:
        state = {("occupied", c, o): True for o, c in layout.items()}
        if held is not None:
            state[("holding", held)] = True
        restore_state(env, workspace, state, object_names,
                      executor=executor, home_qpos=home_qpos, on_held="attach")
        if not check_action(env, workspace, executor, pick_fn, put_fn,
                            action, fast=fast):
            return False, action
        kind, obj, cell = action
        if kind == "pick":
            held = obj
            layout.pop(obj, None)
        else:
            layout[obj] = cell
            held = None
    return True, None
