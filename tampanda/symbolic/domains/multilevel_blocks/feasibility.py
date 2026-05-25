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
        """Skip the post-action return-trip — feasibility is fully
        determined by the pick / put core itself.  Saves ~2
        plan_joint_lerps per check (~25-30% wall-clock on typical
        cube and flat actions)."""
        return

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
        """Skip the post-detach return-trip in fast mode (Phase 3.5
        A1-extension).  The block is placed at this point; phases 5-7
        of put_upright only validate that the arm can recover, which
        the next action's start-of-chain transit redoes anyway.

        Saves ~5-10 IK calls per put-upright, most of which run mink to
        max_iters in the tight upright column.  Expected drop:
        ~45 s → ~few seconds per put-upright check.
        """
        return False


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
) -> MultilevelBlocksExecutor:
    """Construct an executor for feasibility checks."""
    from tampanda.planners.rrt_star import RRTStar
    cls = FastFeasibilityExecutor if fast else MultilevelBlocksExecutor
    rrt = RRTStar(env, max_iterations=max_iters)
    executor = cls(env, workspace, config, motion_planner=rrt)
    if fast:
        # Phase 3.5: cap mink max_iters in fast mode.  Reachable IK
        # targets typically converge in 5-30 iters; unreachable targets
        # run to max_iters.  Reducing from 1000 -> 200 cuts the worst-
        # case unconverged probe from ~800 ms to ~150 ms (~5x).  Real
        # execution keeps the default 1000.
        env.ik.max_iters = 200
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

    restore_state(env, workspace, config, state,
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

    return {
        "success": overall_ok,
        "elapsed_s": time.perf_counter() - t_start,
        "fast": fast,
        "per_action": per_action,
    }
