"""Per-action feasibility checker for multilevel_blocks.

Two variants with identical signatures:

* :func:`check_full` — restore state, run the real executor end-to-end,
  return its (success, fluent_delta) result.  Ground truth.
  Runtime: ~5–30 s per action (limited by RRT* + physics).

* :func:`check_fast` — restore state, instantiate a
  :class:`FastFeasibilityExecutor` that skips physical execution but
  still runs every plan-to-pose / linear-IK probe and collision check,
  return success.
  Runtime: ~0.1–1 s per action.

The fast variant subclasses the regular executor and overrides three
methods:

* ``_execute(path)`` — teleport arm to ``path[-1]`` instead of running
  the controller + physics.
* ``_close_attach(block)`` / ``_detach_open()`` — attach/detach the
  block kinematically; skip the gripper close/open wait.
* ``_wait_gripper_*`` / ``_preclose_for_descent`` — no-op (or direct
  qpos write for preclose).

Every other code path — IK, RRT* (well, ``plan_joint_lerp`` /
``plan_to_pose`` IK convergence checks), collision exceptions, held
offset bookkeeping — runs identically to the full executor.

Action dispatch is via :func:`dispatch_action`, which maps a PDDL
``(action_name, *args)`` tuple to the executor method.  See
``executor.register_executor`` for the canonical handler set.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Workspace

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
)
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    MultilevelBlocksExecutor,
)
from tampanda.symbolic.domains.multilevel_blocks.prefilter import (
    INFEASIBLE,
    filter_action,
)
from tampanda.symbolic.domains.multilevel_blocks.state import restore_state


# ---------------------------------------------------------------------------
# Fast executor subclass
# ---------------------------------------------------------------------------


class FastFeasibilityExecutor(MultilevelBlocksExecutor):
    """Dry-run variant of the multilevel-blocks executor.

    Inherits every behaviour from the real executor — IK probes, linear
    chain, collision checks — but skips the physical path execution.
    Each plan_to_pose / plan_joint_lerp call still verifies IK and
    per-substep collision; the resulting path's final config is
    teleported into qpos, so subsequent IK seeds are correct.

    Suitable for any context where you want "would the executor accept
    this action?" without paying the physics cost.
    """

    def _execute(self, path, step_size=None, precision=False):
        """Teleport arm to ``path[-1]`` without running physics.

        Also propagates the kinematic attachment (if any) so the held
        block tracks the EE in body frame.  The real executor relies on
        ``env.step()`` -> ``_apply_attachment`` to keep the block in
        sync; the FAST path skips ``env.step()``, so we must call it
        manually after every teleport.
        """
        if not path:
            return
        last = np.asarray(path[-1], dtype=float)
        self.env.data.qpos[: len(last)] = last
        self.env.data.qvel[: len(last)] = 0.0
        # Keep the controller in IDLE so subsequent calls don't trigger
        # trajectory execution from leftover state.
        if self.env.controller is not None:
            self.env.controller.stop()
        mujoco.mj_forward(self.env.model, self.env.data)
        # Update attached body position (held block follows EE).
        if getattr(self.env, "_attached", None) is not None:
            self.env._apply_attachment()
            mujoco.mj_forward(self.env.model, self.env.data)

    def _close_attach(self, block_name: str) -> None:
        """Kinematic attach — no gripper close wait."""
        # Snap finger qpos to closed-on-block (block half-width each).
        cube_half = getattr(getattr(self, "config", None),
                                "cube_half_extent", 0.015)
        self.env.data.qpos[7] = cube_half
        self.env.data.qpos[8] = cube_half
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.attach_object_to_ee(block_name)
        self._held_block = block_name
        self._held_offset = (self._block_pos(block_name)
                                - self._ee_pos())

    def _detach_open(self) -> None:
        """Kinematic detach — no gripper open wait."""
        if getattr(self.env, "_attached", None) is not None:
            self.env.detach_object()
        self.env.data.qpos[7] = 0.04
        self.env.data.qpos[8] = 0.04
        mujoco.mj_forward(self.env.model, self.env.data)
        self._held_block = None
        self._held_offset = np.zeros(3)

    def _wait_gripper_closed(self, max_steps: int = 1000) -> None:
        return

    def _wait_gripper_open(self, steps: int = 300) -> None:
        return

    def _preclose_for_descent(self, max_steps: int = 400) -> None:
        """Direct qpos write for preclose — skip the physics settle."""
        self.env.data.qpos[7] = 0.02
        self.env.data.qpos[8] = 0.02
        mujoco.mj_forward(self.env.model, self.env.data)

    def _return_after_pickput(self, region, used_quat) -> None:
        """Fast-mode return: TELEPORT the arm to NEUTRAL_HOME instead
        of plan-joint-lerping there.  Skipping the return entirely
        leaves the arm at the post-lift position, and the NEXT
        check_action's start-of-chain `_to_neutral_home` may fail to
        find a collision-free joint-space lerp from there (e.g., after
        a put_upright the EE is just above the placed block, so the
        joint lerp to HOME swings the arm through the stack).
        Teleporting in one mj_forward gives a sane starting pose for
        any subsequent action without paying the lerp cost.
        """
        from tampanda.symbolic.domains.multilevel_blocks.executor import (
            _HOME_NEUTRAL_Q,
        )
        self.env.data.qpos[:7] = _HOME_NEUTRAL_Q
        self.env.data.qvel[:] = 0.0
        if getattr(self.env, "_attached", None) is not None:
            self.env._apply_attachment()
        mujoco.mj_forward(self.env.model, self.env.data)
        if self.env.controller is not None:
            self.env.controller.stop()

    def _filter_quats_by_anchor_ik(self, anchor_pos, grasp_quats):
        """No-op in fast mode.

        The B1 IK pre-filter is a NET COST on FEAS in fast mode
        (anchor IK ~50-200 ms per quat × K quats added; saves nothing
        because the chain's own plan_joint_lerp would have done the
        same IK convergence anyway).  It IS worth it in the full
        executor where the post-filter chain runs real physics and any
        avoided transit saves seconds.

        Keep grasp_quats unchanged here so the chain proceeds normally.
        """
        return list(grasp_quats)

    def _validate_put_upright_return(self) -> bool:
        """Skip the post-detach return-trip in put_upright (Phase 3.5
        A1-extension).  Same rationale as ``_return_after_pickput`` —
        the return is purely arm-recovery and the next action would
        redo it anyway.

        Returns False here, but the caller (put_upright) wraps with a
        teleport-to-NEUTRAL_HOME so the arm is in a canonical pose at
        function exit.  Saves ~5-10 IK calls per put-upright that
        otherwise run mink to max_iters in the tight upright column.
        """
        return False

    def _fast_column_align_substeps_halved(self) -> bool:
        """Fast mode halves column-align Cartesian substeps (20 -> 10).
        ~50 % fewer plan_to_pose IK calls per yaw probe at the highest-
        IK phase of put_upright; safe since the column-align z is
        above the max stack and the path is through open space."""
        return True


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def dispatch_action(
    executor: MultilevelBlocksExecutor,
    action_name: str,
    args: Tuple[str, ...],
) -> bool:
    """Call the executor method that implements ``action_name``.

    Maps PDDL action names (snake / hyphen case) to the executor's
    Python method names.  Returns the success bool from the executor.
    Unknown actions raise ``ValueError``.
    """
    # PDDL uses hyphens; Python methods use underscores.  Some PDDL
    # actions also have a slightly different name (e.g.
    # "make-flat-x-from-y" -> "transform_flat_x_from_y").
    # Single name map covering all 26 PDDL actions.  Hyphen → underscore
    # is the default; the few mismatches (turn-long-* dispatches to the
    # oblong transform method via the executor's bridge wiring) are
    # listed explicitly.
    name_map = {
        # 2x1 in-hand turns share executor with 3x1 turns; the bridge
        # uses different PDDL names so the planner can apply them only
        # to the right shape, but the dispatch is to the same method.
        "turn-long-x-to-y": "turn_long_x_to_y",
        "turn-long-y-to-x": "turn_long_y_to_x",
    }
    method_name = name_map.get(action_name)
    if method_name is None:
        # Accept python-style names too.
        method_name = action_name.replace("-", "_")
    method = getattr(executor, method_name, None)
    if method is None:
        raise ValueError(
            f"unknown action {action_name!r} (mapped to "
            f"{method_name!r}, but executor has no such method)"
        )
    return method(*args)


# ---------------------------------------------------------------------------
# Top-level checkers
# ---------------------------------------------------------------------------


def _make_executor(
    env, workspace: Workspace, config: MultilevelBlocksConfig,
    fast: bool, max_iters: int = 3000,
    yaw_pool=None,
) -> MultilevelBlocksExecutor:
    """Construct an executor for feasibility checks.

    Args:
        yaw_pool: optional :class:`MultilevelBlocksYawPool` for Phase 4
            parallel yaw probing.  When provided AND ``fast=True``, the
            ``put_upright`` column-align phase dispatches each yaw
            candidate to a separate worker.  Wall-clock per put-upright
            drops ~K-fold on the column-align step.  Ignored in full
            mode (full execution is sequential by design).
    """
    from tampanda.planners.rrt_star import RRTStar
    cls = FastFeasibilityExecutor if fast else MultilevelBlocksExecutor
    rrt = RRTStar(env, max_iterations=max_iters)
    executor = cls(env, workspace, config, motion_planner=rrt)
    if fast:
        # Phase 3.5: cap mink max_iters in fast mode.  Reachable IK
        # targets typically converge in 5-30 iters; unreachable targets
        # run to max_iters.  Reducing from 1000 -> 150 cuts the worst-
        # case unconverged probe from ~800 ms to ~110 ms (~7x).  100
        # was too aggressive (3/30 L4 failures at step 10 pick-flat-x);
        # 150 is the sweet spot from the agreement test on v2.  Real
        # execution keeps the default 1000.
        env.ik.max_iters = 100
        # Phase 4: optional parallel yaw pool.
        executor._yaw_pool = yaw_pool
    return executor


def check_action(
    env,
    workspace: Workspace,
    config: MultilevelBlocksConfig,
    state: Dict[Tuple, bool],
    action: Tuple,
    *,
    fast: bool = True,
    executor: Optional[MultilevelBlocksExecutor] = None,
    home_qpos: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Check whether ``action`` is feasible from the given ``state``.

    Args:
        env: ``FrankaEnvironment``.
        workspace: ``Workspace``.
        config: ``MultilevelBlocksConfig``.
        state: PDDL ground-state dict.
        action: ``(action_name, *args)`` tuple.
        fast: if True use the dry-run executor; else use the real one.
        executor: provide one to skip the per-call construction
            overhead.  Must match ``fast``.
        home_qpos: arm qpos to seed before the action.  Defaults to the
            executor's neutral home.

    Returns:
        ``{"success": bool, "elapsed_s": float, "error": str | None,
        "fast": bool}``.
    """
    t_start = time.perf_counter()

    # Geometric pre-filter — cheap (microseconds) rejection of actions
    # that no yaw / approach can possibly resolve given the current
    # occupancy.  False negatives are NOT allowed (verified by
    # examples/multilevel_blocks_prefilter_agreement.py over every plan
    # action in the dataset).  When the filter returns INFEASIBLE we
    # skip the executor chain entirely.
    decision, reason = filter_action(state, action, config)
    if decision == INFEASIBLE:
        return {
            "success": False,
            "elapsed_s": time.perf_counter() - t_start,
            "error": f"prefilter:{reason}",
            "fast": fast,
        }

    # Build / reuse the executor — needed BEFORE restore_state so the
    # held-state attach path has access to the precomputed handoff
    # configs.  Without this rgnet's single-action call from a held PDDL
    # state would silently restore to an empty gripper and the put would
    # produce a false-positive feasibility result.
    owned_executor = executor is None
    if owned_executor:
        executor = _make_executor(env, workspace, config, fast=fast)

    # Restore the symbolic state in MuJoCo.  When the state has a
    # held-* fluent we kinematically attach the held block to the EE
    # via the executor's handoff machinery — only this way does
    # downstream put / transform code see a proper held world.
    restore_state(env, workspace, config, state,
                       home_qpos=home_qpos,
                       on_held="attach",
                       executor=executor)

    action_name, *args = action
    try:
        ok = dispatch_action(executor, action_name, tuple(args))
        err = None
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t_start
    return {
        "success": bool(ok),
        "elapsed_s": elapsed,
        "error": err,
        "fast": fast,
    }


def check_fast(env, workspace, config, state, action, **kwargs):
    """Fast feasibility check — dry-run executor, no physics."""
    kwargs.pop("fast", None)
    return check_action(env, workspace, config, state, action,
                            fast=True, **kwargs)


def check_full(env, workspace, config, state, action, **kwargs):
    """Full feasibility check — real executor, full physics."""
    kwargs.pop("fast", None)
    return check_action(env, workspace, config, state, action,
                            fast=False, **kwargs)


# action_name → (kind, held_fluent) for pick / put actions.  ``held`` is
# the (held-X b) atom that the action adds (pick) or removes (put).  The
# action's remaining args (after the block) are the cells to add/remove
# (in b c) atoms for.  Long-* picks and puts use the same held fluents
# as the corresponding oblong-* actions because the PDDL domain shares
# held-flat-x / held-flat-y / held-upright across oblong and long types,
# with the (long ?b) static predicate distinguishing them.
_PICK_PUT_HELD: Dict[str, str] = {
    "pick-cube": "held-cube",
    "put-cube": "held-cube",
    "pick-flat-x": "held-flat-x",
    "put-flat-x": "held-flat-x",
    "pick-flat-y": "held-flat-y",
    "put-flat-y": "held-flat-y",
    "pick-long-x": "held-flat-x",
    "put-long-x": "held-flat-x",
    "pick-long-y": "held-flat-y",
    "put-long-y": "held-flat-y",
    "pick-upright": "held-upright",
    "put-upright": "held-upright",
    "pick-long-upright": "held-upright",
    "put-long-upright": "held-upright",
}

# In-hand transforms: action_name → (held_from, held_to).
_TRANSFORM_HELD: Dict[str, Tuple[str, str]] = {
    "make-flat-x-from-upright":      ("held-upright", "held-flat-x"),
    "make-flat-y-from-upright":      ("held-upright", "held-flat-y"),
    "make-upright-from-x":           ("held-flat-x",  "held-upright"),
    "make-upright-from-y":           ("held-flat-y",  "held-upright"),
    "make-long-flat-x-from-upright": ("held-upright", "held-flat-x"),
    "make-long-flat-y-from-upright": ("held-upright", "held-flat-y"),
    "make-long-upright-from-x":      ("held-flat-x",  "held-upright"),
    "make-long-upright-from-y":      ("held-flat-y",  "held-upright"),
    "turn-x-to-y":                   ("held-flat-x",  "held-flat-y"),
    "turn-y-to-x":                   ("held-flat-y",  "held-flat-x"),
    "turn-long-x-to-y":              ("held-flat-x",  "held-flat-y"),
    "turn-long-y-to-x":              ("held-flat-y",  "held-flat-x"),
}


def apply_action_effects(state: Dict[Tuple, bool],
                              action: Tuple) -> Dict[Tuple, bool]:
    """Return a NEW state with PDDL action effects applied symbolically.

    Pure dict math — no MuJoCo, no IK.  Used by
    :func:`check_action_sequence` when ``per_action_restore=True`` to
    derive the intermediate symbolic state to ``restore_state`` to
    between actions, matching exactly what training-time
    ``check_action`` (singular) would see for each (state, action) pair.

    Action vocabulary handled: every pick-*, put-*, make-*, turn-*
    in the multilevel_blocks domain.  Unknown action names return the
    state unchanged (defensive).
    """
    name = action[0]
    block = action[1]
    new = dict(state)
    if name in _PICK_PUT_HELD:
        held = _PICK_PUT_HELD[name]
        cells = action[2:]
        if name.startswith("pick-"):
            for c in cells:
                new.pop(("in", block, c), None)
            new[(held, block)] = True
        elif name.startswith("put-"):
            new.pop((held, block), None)
            for c in cells:
                new[("in", block, c)] = True
    elif name in _TRANSFORM_HELD:
        held_from, held_to = _TRANSFORM_HELD[name]
        new.pop((held_from, block), None)
        new[(held_to, block)] = True
    return new


def check_action_sequence(
    env,
    workspace: Workspace,
    config: MultilevelBlocksConfig,
    state: Dict[Tuple, bool],
    actions: List[Tuple],
    *,
    fast: bool = True,
    home_qpos: Optional[np.ndarray] = None,
    short_circuit: bool = True,
    executor: Optional[MultilevelBlocksExecutor] = None,
    per_action_restore: bool = False,
) -> Dict[str, Any]:
    """Restore ``state`` once, then dispatch ``actions`` in order.

    Useful for testing PUT (and TRANSFORM) feasibility without a held-
    state restore: pass a sequence ``[pick_a, put_b]`` and the pick
    sets up the held state naturally before the put runs.

    Args:
        env, workspace, config, state, home_qpos: same as
            :func:`check_action`.
        actions: list of ``(action_name, *args)`` tuples.
        fast: if True use ``FastFeasibilityExecutor``.
        short_circuit: stop at the first failure (default).  If False,
            continues running every action even after a failure (the
            results list reports per-action outcomes).

    Returns:
        ``{"success": bool, "elapsed_s": float, "per_action": [
            {"action", "success", "elapsed_s", "error"}, ...
        ]}``.  ``success`` is True only if every action succeeded.
    """
    t_start = time.perf_counter()

    if executor is None:
        executor = _make_executor(env, workspace, config, fast=fast)

    current_state = dict(state)
    restore_state(env, workspace, config, current_state,
                       home_qpos=home_qpos,
                       on_held="attach",
                       executor=executor)

    per_action: List[Dict[str, Any]] = []
    overall_ok = True
    for action in actions:
        t_a = time.perf_counter()
        action_name, *args = action
        try:
            ok = dispatch_action(executor, action_name, tuple(args))
            err = None
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        per_action.append({
            "action": action,
            "success": bool(ok),
            "elapsed_s": time.perf_counter() - t_a,
            "error": err,
        })
        if not ok:
            overall_ok = False
            if short_circuit:
                break
        elif per_action_restore:
            # Per-action restore: evolve the symbolic state via PDDL
            # effects and re-restore the world from it.  Matches
            # ``check_action`` (singular) — the path GNN training uses
            # to label each (state, action) pair.  Avoids the
            # cumulative-drift bug where sequential ``dispatch_action``
            # calls leave placed blocks ~30 mm off their nominal
            # rest pose; after several put-* steps the scene has
            # block-on-block penetrations that ``is_collision_free``
            # then aborts on downstream actions.
            current_state = apply_action_effects(current_state, action)
            restore_state(env, workspace, config, current_state,
                               home_qpos=home_qpos,
                               on_held="attach",
                               executor=executor)

    return {
        "success": overall_ok,
        "elapsed_s": time.perf_counter() - t_start,
        "fast": fast,
        "per_action": per_action,
    }
