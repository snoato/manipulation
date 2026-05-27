"""Action feasibility for the dense-YCB tabletop-access fork.

Forked from ``tabletop_access.feasibility``; same FAST/FULL design
(feasibility = does the chain succeed), adapted for:

* the fork's **multi-cell** ``restore_state`` (footprints threaded
  through), and
* the **footprint-centroid** pick/put pose (the action's ``cell_id`` is
  the object's SW anchor; the grasp/place pose is the footprint centroid,
  not a single cell centre).

``fast=True`` monkey-patches physics-advancing calls to a kinematic
teleport and caps IK iters, so every IK probe + collision-checked plan
still runs but physics is skipped.  Validate generated plans with FULL.

Action format::  ``("pick"|"put", obj_name, anchor_cell_id)``
"""

from __future__ import annotations

import contextlib
from typing import Dict, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.tabletop_access_ycb.footprint import ObjectFootprint
from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state

_FAST_IK_MAX_ITERS = 80


@contextlib.contextmanager
def fast_mode(env, executor):
    """Swap physics-advancing calls for kinematic teleports (restores on exit)."""
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


def footprint_overlap(a_anchor: Cell, fa: ObjectFootprint,
                      b_anchor: Cell, fb: ObjectFootprint) -> bool:
    """True iff two placed footprints share any cell (cheap structural test)."""
    if a_anchor.region != b_anchor.region:
        return False
    acells = {(c.ix, c.iy) for c in fa.cells_at(a_anchor)}
    bcells = {(c.ix, c.iy) for c in fb.cells_at(b_anchor)}
    return not acells.isdisjoint(bcells)


# --- single-cell front-occlusion rules (copied from the parent access;
#     the palm-+y gripper's swept volume) -------------------------------

def _cell_blocks_pick(b: Cell, t: Cell) -> bool:
    if b.region != t.region:
        return False
    dx = abs(b.ix - t.ix)
    if dx == 0:
        return b.iy < t.iy                 # same column, in front
    if dx == 1:
        return b.iy <= t.iy                # adjacent column, front or same row
    return False


def _cell_blocks_put(b: Cell, t: Cell) -> bool:
    if b.region != t.region:
        return False
    dx = abs(b.ix - t.ix)
    if dx == 0:
        return b.iy != t.iy                # same column, front OR behind
    if dx == 1:
        return b.iy <= t.iy
    return False


def footprint_blocks_pick(b_anchor: Cell, fb: ObjectFootprint,
                          t_anchor: Cell, ft: ObjectFootprint) -> bool:
    """Does footprint B occlude a FRONT pick of footprint T?  Any B-cell in
    the swept corridor of any T-cell (heuristic; the feasibility oracle is
    the real gate)."""
    bcells = fb.cells_at(b_anchor)
    tcells = ft.cells_at(t_anchor)
    return any(_cell_blocks_pick(bc, tc) for bc in bcells for tc in tcells)


def footprint_blocks_put(b_anchor: Cell, fb: ObjectFootprint,
                         t_anchor: Cell, ft: ObjectFootprint) -> bool:
    """Like :func:`footprint_blocks_pick` but for a PUT target (also blocks
    from behind)."""
    bcells = fb.cells_at(b_anchor)
    tcells = ft.cells_at(t_anchor)
    return any(_cell_blocks_put(bc, tc) for bc in bcells for tc in tcells)


def _centroid_pose(workspace, obj, anchor_cell_id, footprints, *, centre_z):
    """Footprint-centroid world pose for the object at the given anchor.

    ``centre_z=True`` returns the object's CENTRE z (surface + half_z) for
    a centre-grasp pick; ``False`` returns the surface z for a put (the
    chain recomputes its grasp z from the live half-height).
    """
    anchor = Cell.parse(anchor_cell_id)
    region = workspace[anchor.region]
    cx, cy = footprints[obj].centroid_world(region, anchor)
    return np.array([cx, cy, region.level_z])  # z handled by caller/chain


def check_action(env, workspace, executor, pick_fn, put_fn,
                 footprints: Dict[str, ObjectFootprint],
                 action: Tuple[str, str, str], *, fast: bool) -> bool:
    """Return whether ``action`` is feasible by running the chain.

    The scene (placements + gripper/held state) must already reflect the
    pre-action world (call ``restore_state`` first, e.g. via
    :func:`check_action_sequence`).
    """
    kind, obj_name, cell_id = action
    anchor = Cell.parse(cell_id)
    region = workspace[anchor.region]
    cx, cy = footprints[obj_name].centroid_world(region, anchor)

    def _run() -> bool:
        if kind == "pick":
            half_z = float(env.get_object_half_size(obj_name)[2])
            pos = np.array([cx, cy, region.level_z + half_z])  # centre grasp
            return bool(pick_fn(obj_name, cell_id, pos))
        if kind == "put":
            pos = np.array([cx, cy, region.level_z])
            return bool(put_fn(obj_name, cell_id, pos))
        raise ValueError(f"unknown action kind {kind!r}")

    if fast:
        with fast_mode(env, executor):
            return _run()
    return _run()


def check_action_sequence(env, workspace, executor, pick_fn, put_fn,
                          footprints: Dict[str, ObjectFootprint],
                          source_layout: Dict[str, str], actions, object_names,
                          *, fast: bool, home_qpos=None):
    """Validate a whole plan action-by-action under canonical restore.

    ``source_layout`` maps obj -> anchor cell id.  Before each action the
    world is restored to the current symbolic state (footprint occupancy),
    so every check is history-independent.  Returns ``(ok, failed_or_None)``.
    """
    layout = dict(source_layout)
    held = None
    for action in actions:
        state: Dict[Tuple, bool] = {}
        for obj, anchor_id in layout.items():
            anchor = Cell.parse(anchor_id)
            for c in footprints[obj].cells_at(anchor):
                state[("occupied", c.id, obj)] = True
        if held is not None:
            state[("holding", held)] = True
        restore_state(env, workspace, state, object_names, footprints,
                      executor=executor, home_qpos=home_qpos, on_held="attach")
        if not check_action(env, workspace, executor, pick_fn, put_fn,
                            footprints, action, fast=fast):
            return False, action
        kind, obj, cell = action
        if kind == "pick":
            held = obj
            layout.pop(obj, None)
        else:
            layout[obj] = cell
            held = None
    return True, None
