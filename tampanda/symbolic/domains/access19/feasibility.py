"""Per-action feasibility checker for access-19.

Two variants:

* :func:`check_full` — restore state, run the real chain-based
  pick/put with full physics, return the chain's success bool.
  Runtime: ~0.5–3 s per action (limited by IK + step physics).
  Ground truth.

* :func:`check_fast` — restore state, then run the SAME chain with
  ``env.execute_path`` / ``env.wait_idle`` / ``_wait_gripper_*``
  monkey-patched out so physics is skipped.  Every IK probe + plan
  validation runs identically; only the path execution step is
  short-circuited to a kinematic teleport.
  Runtime: ~0.05–0.5 s per action.

Action format (filter-mode PDDL)::

    ("pick", obj_name, cell_id)
    ("put",  obj_name, cell_id)

Sequence checks restore state once and then dispatch in order, so the
gripper state evolves naturally between actions (a ``pick`` leaves
the gripper holding, a subsequent ``put`` consumes that held state).

Public API::

    check_action(env, ws, cfg, state, action, pick_fn, put_fn, *, fast)
    check_action_sequence(env, ws, cfg, state, actions, pick_fn, put_fn, *, fast)
    check_fast(...) / check_full(...)
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace

from tampanda.symbolic.domains.access19.env_builder import Access19Config
from tampanda.symbolic.domains.access19.state import restore_state


# ---------------------------------------------------------------------------
# Fast-mode env patches
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fast_env(env, executor):
    """Context manager that swaps ``env.execute_path``, ``env.wait_idle``
    and the executor's ``_wait_gripper_*`` methods for no-physics
    equivalents.  Restores originals on exit.

    Kinematic semantics:
      * ``execute_path(path, ...)`` → set ``qpos[:7] = path[-1]``,
        ``mj_forward``, ``_apply_attachment`` if held.
      * ``wait_idle(...)`` → no-op.
      * ``_wait_gripper_closed/open`` → no-op (qpos already set by
        the chain's ``close_gripper`` direct write, or by the snap
        below).
    """
    orig_execute_path = env.execute_path
    orig_wait_idle = env.wait_idle
    orig_w_closed = getattr(executor, "_wait_gripper_closed", None)
    orig_w_open = getattr(executor, "_wait_gripper_open", None)

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

    def _fast_wait_idle(*args, **kwargs):
        return

    def _fast_w_closed(*args, **kwargs):
        # Snap gripper qpos to closed-on-cube (cube half_x = 0.02 m).
        env.data.qpos[7] = 0.020
        env.data.qpos[8] = 0.020
        mujoco.mj_forward(env.model, env.data)

    def _fast_w_open(*args, **kwargs):
        env.data.qpos[7] = 0.040
        env.data.qpos[8] = 0.040
        mujoco.mj_forward(env.model, env.data)

    env.execute_path = _fast_execute_path
    env.wait_idle = _fast_wait_idle
    if orig_w_closed is not None:
        executor._wait_gripper_closed = _fast_w_closed
    if orig_w_open is not None:
        executor._wait_gripper_open = _fast_w_open
    # Flag read by ``chains.py`` row-step substep helper to drop
    # intermediate collision checks for inside-cubicle short hops.
    env._fast_mode = True
    try:
        yield
    finally:
        env.execute_path = orig_execute_path
        env.wait_idle = orig_wait_idle
        if orig_w_closed is not None:
            executor._wait_gripper_closed = orig_w_closed
        if orig_w_open is not None:
            executor._wait_gripper_open = orig_w_open
        env._fast_mode = False


# ---------------------------------------------------------------------------
# Symbolic pre-filter (fast mode only)
# ---------------------------------------------------------------------------


_INTERIOR_PREFIX = "shelf_interior__"


def _occupied_cells(state: Dict[Tuple, bool]) -> set:
    """Extract the set of currently-occupied cell ids from ``state``."""
    return {
        key[1]
        for key, value in state.items()
        if value and isinstance(key, tuple) and len(key) == 3
            and key[0] == "occupied"
    }


def _column_front_blocked(occupied: set, cell_id: str) -> bool:
    """True iff the chain's interior approach to ``cell_id`` is blocked
    by a cube at the same column with a smaller (closer-to-front)
    ``iy``.

    Access-19 interior chains enter at the open ``-y`` face and
    row-step inward in ``iy``.  A cube at ``(ix, iy')`` with
    ``iy' < iy`` occludes the chain — the row-step IK fails at the
    first blocker.  This check is O(rows) and runs no IK or physics.

    Returns ``False`` for non-interior cells (``shelf_top__`` is open
    from above) or malformed cell ids — keeps the check safe to apply
    blanket-style.
    """
    if not cell_id.startswith(_INTERIOR_PREFIX):
        return False
    try:
        suffix = cell_id[len(_INTERIOR_PREFIX):]
        ix_str, iy_str = suffix.split("_")
        ix, iy = int(ix_str), int(iy_str)
    except ValueError:
        return False
    for iy_prime in range(iy):
        if f"{_INTERIOR_PREFIX}{ix}_{iy_prime}" in occupied:
            return True
    return False


def _prefilter_reject(
    action: Tuple, occupied: set,
) -> bool:
    """Fast-mode symbolic pre-filter — currently just the column-front
    occlusion check for interior picks and puts.  Returns True iff the
    action is provably infeasible from ``occupied`` and can be
    short-circuited without restoring state or running the chain.
    """
    if not action or action[0] not in ("pick", "put"):
        return False
    _, _, cell_id = action
    return _column_front_blocked(occupied, cell_id)


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    env,
    workspace: Workspace,
    pick_fn: Callable,
    put_fn: Callable,
    action: Tuple,
) -> bool:
    """Call the chain pick/put for ``action`` and return its success."""
    action_name, *args = action
    if action_name == "pick":
        obj_name, cell_id = args
        pos, _ = env.get_object_pose(obj_name)
        return bool(pick_fn(obj_name, cell_id, np.asarray(pos)))
    if action_name == "put":
        obj_name, cell_id = args
        target = workspace.cell(cell_id)
        target_pos = np.asarray(workspace.pose_for(target))
        return bool(put_fn(obj_name, cell_id, target_pos))
    raise ValueError(f"access-19 actions are 'pick' / 'put' only, "
                          f"got {action_name!r}")


# ---------------------------------------------------------------------------
# Top-level checkers
# ---------------------------------------------------------------------------


def check_action(
    env,
    workspace: Workspace,
    config: Access19Config,
    state: Dict[Tuple, bool],
    action: Tuple,
    object_names: List[str],
    pick_fn: Callable,
    put_fn: Callable,
    *,
    executor,
    fast: bool = True,
    home_qpos: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Check whether ``action`` is feasible from the given ``state``.

    Args:
        env, workspace, config: scene + config.
        state: PDDL ground-state dict (gripper-empty assumed; held
            states should be reached via an action sequence).
        action: ``("pick" | "put", obj_name, cell_id)``.
        object_names: full object roster.
        pick_fn / put_fn: chain functions from
            :func:`make_access19_pick_fn` / :func:`make_access19_put_fn`.
        executor: the ``PickPlaceExecutor`` used by the chains.
            Required so ``_fast_env`` can monkey-patch its
            ``_wait_gripper_*`` methods.
        fast: if True, run the action with physics short-circuited.
        home_qpos: arm config to apply before the action.

    Returns:
        ``{"success": bool, "elapsed_s": float, "error": str | None,
        "fast": bool}``.
    """
    t_start = time.perf_counter()

    # Fast-mode pre-filter: reject column-front-occluded interior
    # actions without restoring state or running the chain.  Exact
    # under chain semantics (no false negatives) — see
    # ``_column_front_blocked``.  FULL mode keeps full ground-truth
    # semantics and skips the filter.
    if fast and _prefilter_reject(action, _occupied_cells(state)):
        return {
            "success": False,
            "elapsed_s": time.perf_counter() - t_start,
            "error": None,
            "fast": fast,
            "prefiltered": True,
        }

    restore_state(env, workspace, config, state, object_names,
                       home_qpos=home_qpos)

    try:
        with (_fast_env(env, executor) if fast else contextlib.nullcontext()):
            ok = _dispatch(env, workspace, pick_fn, put_fn, action)
        err = None
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    return {
        "success": bool(ok),
        "elapsed_s": time.perf_counter() - t_start,
        "error": err,
        "fast": fast,
    }


def check_action_sequence(
    env,
    workspace: Workspace,
    config: Access19Config,
    state: Dict[Tuple, bool],
    actions: List[Tuple],
    object_names: List[str],
    pick_fn: Callable,
    put_fn: Callable,
    *,
    executor,
    fast: bool = True,
    home_qpos: Optional[np.ndarray] = None,
    short_circuit: bool = True,
) -> Dict[str, Any]:
    """Restore ``state`` once, then dispatch ``actions`` in order.

    The natural way to check held-state-dependent actions: a leading
    ``pick`` action sets up the held state before subsequent ``put``s
    run.
    """
    t_start = time.perf_counter()
    restore_state(env, workspace, config, state, object_names,
                       home_qpos=home_qpos)

    # Build a running symbolic layout from the input state so we can
    # snap all placed cubes back to their symbolic cell centres (with
    # identity quat) before every action.  Why: the 4×4×8 cube is a
    # TALL, narrow-base prism — under FULL physics, each put_deck
    # release imparts tiny rotational momentum from gripper-open +
    # detach, and over many actions the cube orientations accumulate
    # visible tilt (~degrees off vertical).  Snapping pose+quat
    # between actions kills both the position drift AND the tilt;
    # FAST never drifts so the snap is a no-op there.  Held objects
    # are skipped (tracked separately via the attachment hook).
    running_layout: Dict[str, str] = {}
    for key, value in state.items():
        if not value:
            continue
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "occupied":
            _, cell_id, obj = key
            running_layout[obj] = cell_id

    per_action: List[Dict[str, Any]] = []
    overall_ok = True
    ctx_factory = (_fast_env if fast
                       else lambda env_, ex_: contextlib.nullcontext())
    with ctx_factory(env, executor):
        for action in actions:
            # Reset arm to home + snap placed objects to symbolic cells.
            # The chains assume the arm starts at staging-home and the
            # cubes are at their cell centres in canonical orientation.
            # Held attachments persist across qpos writes via the hook.
            if home_qpos is not None:
                env.data.qpos[: len(home_qpos)] = np.asarray(home_qpos,
                                                                       dtype=float)
                env.data.qvel[:] = 0.0
                mujoco.mj_forward(env.model, env.data)
            for _obj, _cell_id in running_layout.items():
                _cell = workspace.cell(_cell_id)
                _x, _y, _z = workspace.pose_for(_cell)
                # quat=None defaults to identity in set_object_pose.
                env.set_object_pose(_obj, np.array([_x, _y, _z]))
            mujoco.mj_forward(env.model, env.data)
            if getattr(env, "_attached", None) is not None:
                env._apply_attachment()
                mujoco.mj_forward(env.model, env.data)

            t_a = time.perf_counter()
            # Fast-mode pre-filter (same semantics as ``check_action``).
            # Build ``occupied`` from the running layout, not from the
            # initial ``state`` — keeps the check honest as the
            # sequence evolves.
            prefiltered = False
            if fast and _prefilter_reject(
                action, set(running_layout.values())
            ):
                ok = False
                err = None
                prefiltered = True
            else:
                try:
                    ok = _dispatch(env, workspace, pick_fn, put_fn, action)
                    err = None
                    if ok:
                        if action[0] == "pick":
                            _, _obj, _ = action
                            running_layout.pop(_obj, None)
                        elif action[0] == "put":
                            _, _obj, _cell_id = action
                            running_layout[_obj] = _cell_id
                except Exception as exc:
                    ok = False
                    err = f"{type(exc).__name__}: {exc}"
            entry = {
                "action": action,
                "success": bool(ok),
                "elapsed_s": time.perf_counter() - t_a,
                "error": err,
            }
            if prefiltered:
                entry["prefiltered"] = True
            per_action.append(entry)
            if not ok:
                overall_ok = False
                if short_circuit:
                    break

    return {
        "success": overall_ok,
        "elapsed_s": time.perf_counter() - t_start,
        "fast": fast,
        "per_action": per_action,
    }


def check_fast(env, ws, cfg, state, action, object_names,
                  pick_fn, put_fn, *, executor, **kwargs):
    """Fast feasibility check — physics short-circuited."""
    kwargs.pop("fast", None)
    return check_action(env, ws, cfg, state, action, object_names,
                              pick_fn, put_fn, executor=executor,
                              fast=True, **kwargs)


def check_full(env, ws, cfg, state, action, object_names,
                  pick_fn, put_fn, *, executor, **kwargs):
    """Full feasibility check — real physics."""
    kwargs.pop("fast", None)
    return check_action(env, ws, cfg, state, action, object_names,
                              pick_fn, put_fn, executor=executor,
                              fast=False, **kwargs)
