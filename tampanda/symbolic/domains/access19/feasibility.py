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


def _occupant_at(state: Dict[Tuple, bool], cell_id: str) -> Optional[str]:
    """Return the object name occupying ``cell_id``, or ``None`` if empty."""
    for key, value in state.items():
        if not value or not isinstance(key, tuple) or len(key) != 3:
            continue
        if key[0] == "occupied" and key[1] == cell_id:
            return str(key[2])
    return None


def _held_object(state: Dict[Tuple, bool]) -> Optional[str]:
    """Return the currently-held object name, or ``None`` if empty."""
    for key, value in state.items():
        if not value or not isinstance(key, tuple) or len(key) != 2:
            continue
        if key[0] == "holding":
            return str(key[1])
    return None


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


def _lateral_front_blocked(occupied: set, cell_id: str) -> bool:
    """True iff reaching interior cell ``(ix, iy)`` is blocked by a cube
    in an *adjacent cube column* that sits *in front of* the target.

    Empirically measured (palm-+y FRONT grasp, 4×4×8 cube grid): while
    the chain row-steps from the open -y face down column ``ix`` to the
    target row, the Franka hand over-reaches laterally into the
    neighbouring cube columns (``ix ± 2`` — cube columns are at grid
    ix 1/3/5 with empty channels between).  A cube in a neighbour
    column at a row *strictly in front* of the target (``iy' < iy``,
    i.e. one the hand sweeps past on the way in) is clipped by ~4-7 mm.
    Same-row neighbours (the hand descends centred there) and
    behind-row neighbours (never approached) do NOT clip — so the band
    is rows ``0 .. iy-1`` only.

    The collision proxy (hand_capsule) is ~4 mm thinner than the real
    hand mesh and the mesh is collision-disabled, so MuJoCo's
    ``check_collisions`` misses this clip — hence this symbolic
    fast-reject.

    ASYMMETRIC: the hand over-reaches toward LOWER ix only.  Measured
    clips reaching column 3:
      * -x neighbour (col 1): -4 to -7 mm  → real clip, reject
      * +x neighbour (col 5): -0.6 to +0.2 mm → within the ~2 mm
        tolerance the system already allows → do NOT reject
    A symmetric rule produces false negatives (it rejects the +x
    graze) and breaks valid reference plans that leave a +x-neighbour
    cube in place.  So only the ``ix - 2`` column is checked.

    O(rows); no IK / physics.  ``False`` for non-interior / malformed.
    """
    if not cell_id.startswith(_INTERIOR_PREFIX):
        return False
    try:
        suffix = cell_id[len(_INTERIOR_PREFIX):]
        ix_str, iy_str = suffix.split("_")
        ix, iy = int(ix_str), int(iy_str)
    except ValueError:
        return False
    ix_adj = ix - 2                        # -x neighbour cube column only
    if ix_adj not in (1, 3, 5):
        return False                       # no cube column on the -x side
    for iy_prime in range(iy):             # rows 0 .. iy-1 (in front)
        if f"{_INTERIOR_PREFIX}{ix_adj}_{iy_prime}" in occupied:
            return True
    return False


def _prefilter_reject(
    action: Tuple, state: Dict[Tuple, bool],
) -> bool:
    """Fast-mode symbolic pre-filter.  Returns True iff the action is
    provably infeasible from ``state`` and can be short-circuited
    without restoring state or running the chain.

    Catches:
      * Column-front occlusion (interior cells) — chain would fail in
        the first row-step lerp.
      * Held-state mismatch — pick while holding anything, put while
        not holding the target object.  Chain would fail at attach /
        detach.
      * Target-cell already occupied (put) — chain would fail when
        the descent collides with the resident cube (or pass through
        and place on top of it, which the chain rejects via the
        cube-cube collision check).
      * Source-cell not occupied by target obj (pick) — chain would
        descend onto empty air and grasp nothing.
      * Lateral front clip (interior) — a cube in an adjacent cube
        column in front of the target clips the Franka hand (a ~4-7mm
        collision the MuJoCo proxy misses); see
        ``_lateral_front_blocked``.
    """
    if not action or action[0] not in ("pick", "put"):
        return False
    occupied = _occupied_cells(state)
    held = _held_object(state)
    verb, obj, cell_id = action[0], action[1], action[2]

    # Column-front occlusion — covers both pick and put for interior.
    if _column_front_blocked(occupied, cell_id):
        return True
    # Lateral front clip from adjacent cube columns.  For a put, the
    # held obj isn't in ``occupied`` so no self-exclusion needed.
    if _lateral_front_blocked(occupied, cell_id):
        return True

    if verb == "pick":
        # Must be holding nothing.
        if held is not None:
            return True
        # The target cell must be occupied by ``obj``.
        if _occupant_at(state, cell_id) != obj:
            return True
        return False

    # verb == "put"
    # Must be holding the target object.
    if held != obj:
        return True
    # The target cell must be empty.
    if _occupant_at(state, cell_id) is not None:
        return True
    return False


# ---------------------------------------------------------------------------
# Combined pick-place pre-filter + dispatch (domain_combined.pddl)
# ---------------------------------------------------------------------------


def _prefilter_reject_pick_place(
    action: Tuple, state: Dict[Tuple, bool],
) -> bool:
    """Fast-mode symbolic pre-filter for the combined
    ``(pick-place ?obj ?cf ?ct)`` action.

    Rejects if the source cell isn't occupied by the target obj OR
    the target cell is already occupied OR either cell is column-
    front-blocked OR laterally front-clipped.  Same one-way
    implication soundness as the base ``_prefilter_reject``.

    Lateral check on the target cell uses ``occupied - {cf}``: by the
    time the put sub-chain reaches ``ct`` the obj has been picked, so
    it no longer occupies ``cf`` (which could otherwise be a front
    neighbour of ``ct``).
    """
    if not action or action[0] != "pick-place" or len(action) < 4:
        return False
    _, obj, cf, ct = action[0], action[1], action[2], action[3]
    occupied = _occupied_cells(state)
    if _column_front_blocked(occupied, cf):
        return True
    if _occupant_at(state, cf) != obj:
        return True
    if _occupant_at(state, ct) is not None:
        return True
    # Lateral clip — source reach uses current occupancy; target reach
    # uses occupancy after the pick (obj removed from cf).
    if _lateral_front_blocked(occupied, cf):
        return True
    occupied_after_pick = occupied - {cf}
    if _column_front_blocked(occupied_after_pick, ct):
        return True
    if _lateral_front_blocked(occupied_after_pick, ct):
        return True
    return False


def _dispatch_pick_place(
    env,
    workspace: Workspace,
    pick_fn: Callable,
    put_fn: Callable,
    action: Tuple,
) -> bool:
    """Run the combined ``pick-place`` chain for ``action``.

    Source position comes from the env's current pose of ``obj``
    (set by ``restore_state`` from the symbolic ``(occupied cf obj)``
    fact).  Target position is the canonical cell centre of ``ct``.
    """
    if action[0] != "pick-place" or len(action) < 4:
        raise ValueError(
            f"_dispatch_pick_place expects (pick-place obj cf ct), got "
            f"{action!r}"
        )
    _, obj_name, cf, ct = action[0], action[1], action[2], action[3]
    source_pos, _ = env.get_object_pose(obj_name)
    target = workspace.cell(ct)
    target_pos = np.asarray(workspace.pose_for(target))
    if not pick_fn(obj_name, cf, np.asarray(source_pos)):
        return False
    return bool(put_fn(obj_name, ct, target_pos))


def check_pick_place_action(
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
    """Feasibility checker for the combined ``pick-place`` action set
    (``domain_combined.pddl``).

    Same shape as :func:`check_action` — pre-filter + state restore +
    dispatch + timing.  Only FAST mode is fully supported (the chain
    sequence relies on the pick chain's post-grasp short-circuit to
    teleport the arm back to ``home_qpos`` before the put chain
    starts).  FULL mode falls through to the same code path; callers
    needing ground-truth physics should sequence ``check_action``
    calls instead.
    """
    t_start = time.perf_counter()
    if fast and _prefilter_reject_pick_place(action, state):
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
            ok = _dispatch_pick_place(env, workspace, pick_fn, put_fn,
                                                 action)
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
    if fast and _prefilter_reject(action, state):
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
    running_held: Optional[str] = None
    for key, value in state.items():
        if not value:
            continue
        if isinstance(key, tuple) and len(key) == 3 and key[0] == "occupied":
            _, cell_id, obj = key
            running_layout[obj] = cell_id
        elif isinstance(key, tuple) and len(key) == 2 and key[0] == "holding":
            running_held = str(key[1])

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
            # Build a synthetic state dict from the running layout +
            # held so the pre-filter sees the same shape as the
            # top-level ``check_action`` path.
            _seq_state: Dict[Tuple, bool] = {
                ("occupied", _cid, _obj): True
                for _obj, _cid in running_layout.items()
            }
            if running_held is not None:
                _seq_state[("holding", running_held)] = True
            if fast and _prefilter_reject(action, _seq_state):
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
                            running_held = _obj
                        elif action[0] == "put":
                            _, _obj, _cell_id = action
                            running_layout[_obj] = _cell_id
                            running_held = None
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
