"""State-audit instrumentation for access-19 chain execution.

Captures every transition during a chain action — entry / exit of
``env.execute_path``, attach / detach events, gripper commands — so we
can debug the FAST↔FULL divergence at the executor level.

Usage::

    auditor = StateAuditor(env, workspace, object_names)
    with auditor.instrument():
        # run actions through check_action_sequence or directly
        ...
    auditor.summarise()  # prints per-action metrics
    auditor.dump_json("/tmp/access19_audit.json")

The instrument() context patches env methods.  Each patched call
records a :class:`Checkpoint`; an outer :meth:`begin_action` /
:meth:`end_action` API brackets each PDDL action so per-action
metrics can be derived.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.workspace import Cell, Workspace


_CANONICAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0])    # identity in MuJoCo convention


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """One env-state snapshot — pure values, no MuJoCo references."""
    t: float                                    # sim_time
    arm_qpos: List[float]                       # 7 arm joints
    gripper_qpos: List[float]                   # 2 finger joints
    gripper_ctrl: float
    ee_xyz: List[float]
    ee_rotmat: List[float]                      # flattened 3x3
    cube_poses: Dict[str, Tuple[List[float], List[float]]]   # name -> (pos, quat)
    attached_body: Optional[str]
    attached_rel_pos: Optional[List[float]]     # block pos in EE frame
    attached_rel_quat: Optional[List[float]]    # block quat in EE frame


@dataclass
class Checkpoint:
    """One audit event."""
    kind: str          # "begin_action", "execute_pre", "execute_post",
                          # "attach", "detach", "gripper_close",
                          # "gripper_open", "end_action"
    action_idx: int    # which action this belongs to (0 if pre-action)
    phase: str         # free-text human-readable phase tag
    snapshot: Snapshot
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionMetrics:
    """Per-action summary derived from the checkpoint stream."""
    action_idx: int
    action: Tuple
    success: bool
    chain_prints: List[str]                   # captured chain failure prints
    duration_s: float
    n_execute_calls: int
    max_ee_tracking_err_m: float              # max ||planned - achieved|| EE xyz
    max_arm_tracking_err_rad: float           # max joint-space tracking err
    held_obj_at_end: Optional[str]
    held_offset_drift_mm: float               # drift of held block in EE frame
    held_tilt_deg: float                      # deviation of held block from canonical quat
    unplanned_cube_movements_mm: Dict[str, float]   # cubes-not-acted-on, max displacement
    max_unplanned_cube_movement_mm: float
    max_cube_tilt_deg: float                  # max tilt across all placed cubes
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# StateAuditor
# ---------------------------------------------------------------------------


def _quat_tilt_deg(quat: np.ndarray) -> float:
    """Angle (degrees) by which ``quat`` deviates from identity."""
    q = np.asarray(quat, dtype=float)
    if np.allclose(q, 0):
        return 0.0
    q = q / max(np.linalg.norm(q), 1e-12)
    # angle = 2 * acos(|w|)
    w = abs(float(q[0]))
    w = min(1.0, max(-1.0, w))
    return math.degrees(2.0 * math.acos(w))


class StateAuditor:
    """Instruments env to capture chain-execution telemetry."""

    def __init__(
        self,
        env,
        workspace: Workspace,
        object_names: List[str],
    ) -> None:
        self.env = env
        self.workspace = workspace
        self.object_names = list(object_names)
        self.checkpoints: List[Checkpoint] = []
        self._current_action_idx: int = 0
        self._current_action: Optional[Tuple] = None
        self._action_start_t: Optional[float] = None
        self._action_chain_prints: List[str] = []
        self._action_metrics: List[ActionMetrics] = []
        # Per-action: tracks the LAST execute_pre's intended target
        # (used to compute tracking error on the matching execute_post).
        self._pending_execute_target: Optional[np.ndarray] = None
        self._pending_execute_xyz: Optional[np.ndarray] = None
        # Original env method references (set during instrument()).
        self._orig: Dict[str, Any] = {}

    # ----- snapshot ----------------------------------------------------

    def _ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        pos = np.asarray(self.env.data.site_xpos[site_id], dtype=float).copy()
        rotmat = np.asarray(
            self.env.data.site_xmat[site_id], dtype=float
        ).copy()
        return pos, rotmat

    def _attached_rel(self) -> Tuple[Optional[str], Optional[np.ndarray],
                                            Optional[np.ndarray]]:
        att = getattr(self.env, "_attached", None)
        if att is None:
            return None, None, None
        name = att.get("body_name")
        rel_pos = att.get("rel_pos")
        rel_mat = att.get("rel_mat")
        rel_quat = None
        if rel_mat is not None:
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, np.asarray(rel_mat).flatten())
            rel_quat = q / max(np.linalg.norm(q), 1e-12)
        return (name,
                np.asarray(rel_pos, dtype=float) if rel_pos is not None else None,
                rel_quat)

    def snapshot(self) -> Snapshot:
        ee_pos, ee_rot = self._ee_pose()
        cube_poses: Dict[str, Tuple[List[float], List[float]]] = {}
        for name in self.object_names:
            try:
                pos, quat = self.env.get_object_pose(name)
                cube_poses[name] = (
                    np.asarray(pos, dtype=float).tolist(),
                    np.asarray(quat, dtype=float).tolist(),
                )
            except Exception:
                cube_poses[name] = ([float("nan")] * 3, [1.0, 0.0, 0.0, 0.0])
        att_name, rel_pos, rel_quat = self._attached_rel()
        return Snapshot(
            t=float(self.env.data.time),
            arm_qpos=np.asarray(self.env.data.qpos[:7], dtype=float).tolist(),
            gripper_qpos=np.asarray(
                self.env.data.qpos[7:9], dtype=float
            ).tolist(),
            gripper_ctrl=float(self.env.data.ctrl[7]),
            ee_xyz=ee_pos.tolist(),
            ee_rotmat=ee_rot.tolist(),
            cube_poses=cube_poses,
            attached_body=att_name,
            attached_rel_pos=(rel_pos.tolist() if rel_pos is not None
                                       else None),
            attached_rel_quat=(rel_quat.tolist() if rel_quat is not None
                                        else None),
        )

    # ----- action bracketing ------------------------------------------

    def begin_action(self, action_idx: int, action: Tuple) -> None:
        self._current_action_idx = action_idx
        self._current_action = action
        self._action_start_t = time.perf_counter()
        self._action_chain_prints = []
        self._record("begin_action", f"action {action_idx}: {action}")

    def end_action(self, success: bool) -> None:
        if self._current_action is None:
            return
        self._record("end_action",
                          f"action {self._current_action_idx}: "
                          f"{'OK' if success else 'FAIL'}")
        # Derive per-action metrics from the checkpoints since begin_action.
        metrics = self._derive_metrics(success)
        self._action_metrics.append(metrics)
        self._current_action = None
        self._action_start_t = None

    # ----- record events ----------------------------------------------

    def _record(self, kind: str, phase: str, **extras) -> None:
        self.checkpoints.append(Checkpoint(
            kind=kind, action_idx=self._current_action_idx,
            phase=phase, snapshot=self.snapshot(),
            extras=extras,
        ))

    # ----- env patches -------------------------------------------------

    @contextlib.contextmanager
    def instrument(self):
        env = self.env
        self._orig["execute_path"] = env.execute_path
        self._orig["attach"] = env.attach_object_to_ee
        self._orig["detach"] = env.detach_object
        if env.controller is not None:
            self._orig["close"] = env.controller.close_gripper
            self._orig["open"] = env.controller.open_gripper

        def _wrapped_execute_path(path, planner, *args, **kwargs):
            target_qpos = (np.asarray(path[-1], dtype=float)
                              if path else None)
            self._pending_execute_target = target_qpos
            target_xyz = self._ee_pose_for_q(target_qpos) if target_qpos is not None else None
            self._pending_execute_xyz = target_xyz
            self._record("execute_pre",
                              phase=f"execute_path n={len(path) if path else 0}",
                              target_qpos=target_qpos.tolist()
                                              if target_qpos is not None else None,
                              target_xyz=target_xyz.tolist()
                                              if target_xyz is not None else None)
            ret = self._orig["execute_path"](path, planner, *args, **kwargs)
            # post-exec snapshot
            post_arm = np.asarray(env.data.qpos[:7], dtype=float)
            ee_xyz_now = self._ee_pose()[0]
            track_qpos = (
                float(np.linalg.norm(post_arm - target_qpos[:7]))
                if target_qpos is not None else 0.0
            )
            track_xyz = (
                float(np.linalg.norm(ee_xyz_now - target_xyz))
                if target_xyz is not None else 0.0
            )
            self._record("execute_post",
                              phase=f"execute_path complete",
                              track_qpos=track_qpos,
                              track_xyz=track_xyz)
            return ret

        def _wrapped_attach(*args, **kwargs):
            ret = self._orig["attach"](*args, **kwargs)
            self._record("attach", phase=f"attach {args[0] if args else '?'}")
            return ret

        def _wrapped_detach(*args, **kwargs):
            self._record("detach_pre", phase="detach pre")
            ret = self._orig["detach"](*args, **kwargs)
            self._record("detach", phase="detach")
            return ret

        def _wrapped_close(*args, **kwargs):
            ret = self._orig["close"](*args, **kwargs)
            self._record("gripper_close", phase="close_gripper")
            return ret

        def _wrapped_open(*args, **kwargs):
            ret = self._orig["open"](*args, **kwargs)
            self._record("gripper_open", phase="open_gripper")
            return ret

        env.execute_path = _wrapped_execute_path
        env.attach_object_to_ee = _wrapped_attach
        env.detach_object = _wrapped_detach
        if env.controller is not None:
            env.controller.close_gripper = _wrapped_close
            env.controller.open_gripper = _wrapped_open
        try:
            yield self
        finally:
            env.execute_path = self._orig["execute_path"]
            env.attach_object_to_ee = self._orig["attach"]
            env.detach_object = self._orig["detach"]
            if env.controller is not None:
                env.controller.close_gripper = self._orig["close"]
                env.controller.open_gripper = self._orig["open"]

    def _ee_pose_for_q(self, q: np.ndarray) -> np.ndarray:
        """Compute EE xyz that would be reached by ``q`` (forward kinematics)."""
        env = self.env
        save = env.data.qpos[:7].copy()
        try:
            env.data.qpos[:7] = q[:7]
            mujoco.mj_forward(env.model, env.data)
            return self._ee_pose()[0]
        finally:
            env.data.qpos[:7] = save
            mujoco.mj_forward(env.model, env.data)

    # ----- per-action metrics -----------------------------------------

    def _derive_metrics(self, success: bool) -> ActionMetrics:
        # Look at checkpoints for the current action only.
        evs = [c for c in self.checkpoints
                   if c.action_idx == self._current_action_idx]
        if not evs:
            return ActionMetrics(
                action_idx=self._current_action_idx,
                action=self._current_action or ("?",),
                success=success, chain_prints=[],
                duration_s=0.0, n_execute_calls=0,
                max_ee_tracking_err_m=0.0,
                max_arm_tracking_err_rad=0.0,
                held_obj_at_end=None,
                held_offset_drift_mm=0.0,
                held_tilt_deg=0.0,
                unplanned_cube_movements_mm={},
                max_unplanned_cube_movement_mm=0.0,
                max_cube_tilt_deg=0.0,
            )
        begin = evs[0].snapshot
        end = evs[-1].snapshot

        ee_errs = [c.extras.get("track_xyz", 0.0)
                       for c in evs if c.kind == "execute_post"]
        q_errs = [c.extras.get("track_qpos", 0.0)
                      for c in evs if c.kind == "execute_post"]
        n_exec = sum(1 for c in evs if c.kind == "execute_pre")

        # Identify which object (if any) this action manipulates.
        acted_on = self._current_action[1] if (
            self._current_action and len(self._current_action) > 1
        ) else None

        # Unplanned cube movement: every cube NOT acted on shouldn't have
        # moved between begin and end.
        unplanned: Dict[str, float] = {}
        for name in self.object_names:
            if name == acted_on:
                continue
            p0 = np.asarray(begin.cube_poses.get(name, [(0, 0, 0)])[0])
            p1 = np.asarray(end.cube_poses.get(name, [(0, 0, 0)])[0])
            d = float(np.linalg.norm(p1 - p0))
            if d > 0.001:        # >1 mm threshold
                unplanned[name] = round(d * 1000.0, 2)

        # Held cube offset drift: the attachment rel_pos at first attach
        # event vs the actual EE-to-block transform at end.
        held_drift_mm = 0.0
        held_at_end = end.attached_body
        if held_at_end is not None:
            # initial rel offset
            init_rel = None
            for c in evs:
                if c.kind == "attach":
                    init_rel = c.snapshot.attached_rel_pos
                    break
            if init_rel is not None:
                # current actual offset
                p_end, q_end = end.cube_poses.get(held_at_end, ([0, 0, 0],
                                                                          [1, 0, 0, 0]))
                p_end = np.asarray(p_end)
                ee_p = np.asarray(end.ee_xyz)
                ee_R = np.asarray(end.ee_rotmat).reshape(3, 3)
                actual_rel = ee_R.T @ (p_end - ee_p)
                held_drift_mm = float(
                    np.linalg.norm(actual_rel - np.asarray(init_rel))
                ) * 1000.0

        # Held tilt: deviation of held block's quat from identity (or
        # from the attach-time quat).
        held_tilt = 0.0
        if held_at_end is not None:
            _, q_end = end.cube_poses.get(held_at_end, ([0, 0, 0],
                                                                [1, 0, 0, 0]))
            held_tilt = _quat_tilt_deg(np.asarray(q_end))

        # Max placed-cube tilt across the scene.
        max_tilt = 0.0
        for name in self.object_names:
            if name == held_at_end:
                continue
            _, q = end.cube_poses.get(name, ([0, 0, 0], [1, 0, 0, 0]))
            t = _quat_tilt_deg(np.asarray(q))
            if t > max_tilt:
                max_tilt = t

        return ActionMetrics(
            action_idx=self._current_action_idx,
            action=self._current_action or ("?",),
            success=success,
            chain_prints=list(self._action_chain_prints),
            duration_s=(time.perf_counter() - self._action_start_t
                              if self._action_start_t else 0.0),
            n_execute_calls=n_exec,
            max_ee_tracking_err_m=max(ee_errs) if ee_errs else 0.0,
            max_arm_tracking_err_rad=max(q_errs) if q_errs else 0.0,
            held_obj_at_end=held_at_end,
            held_offset_drift_mm=round(held_drift_mm, 2),
            held_tilt_deg=round(held_tilt, 2),
            unplanned_cube_movements_mm=unplanned,
            max_unplanned_cube_movement_mm=(
                max(unplanned.values()) if unplanned else 0.0
            ),
            max_cube_tilt_deg=round(max_tilt, 2),
        )

    # ----- chain print capture ----------------------------------------

    @contextlib.contextmanager
    def capture_chain_prints(self):
        """Redirect stdout temporarily; collect chain failure prints."""
        import io, sys
        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            yield
        finally:
            sys.stdout = orig
            for line in buf.getvalue().splitlines():
                if "[access19" in line:
                    self._action_chain_prints.append(line.strip())
            # echo to real stdout so the run still shows progress
            orig.write(buf.getvalue())

    # ----- output -----------------------------------------------------

    def metrics(self) -> List[ActionMetrics]:
        return list(self._action_metrics)

    def dump_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "metrics": [asdict(m) for m in self._action_metrics],
                "n_checkpoints": len(self.checkpoints),
            }, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    def summarise(self, top_k: int = 10) -> None:
        if not self._action_metrics:
            print("[StateAuditor] no metrics recorded")
            return
        n = len(self._action_metrics)
        n_fail = sum(1 for m in self._action_metrics if not m.success)
        print(f"\n=== State audit summary ({n} actions; {n_fail} failures) ===")
        # First failure.
        for m in self._action_metrics:
            if not m.success:
                print(f"  FIRST FAILURE: action {m.action_idx+1} "
                          f"{m.action}")
                if m.chain_prints:
                    for line in m.chain_prints[:5]:
                        print(f"    chain: {line}")
                break
        # Top actions by metric.
        def _topk(key: Callable[[ActionMetrics], float], label: str,
                       unit: str) -> None:
            ranked = sorted(self._action_metrics, key=key, reverse=True)[:top_k]
            print(f"\n  Top {top_k} actions by {label} ({unit}):")
            for m in ranked:
                if key(m) <= 0:
                    continue
                print(f"    a{m.action_idx+1:>3} {m.action[0]:<4} "
                          f"{m.action[1]:<10} → {m.action[2] if len(m.action) > 2 else '':<30}  "
                          f"{label}={key(m):.2f} {unit}  "
                          f"{'OK' if m.success else 'FAIL'}")

        _topk(lambda m: m.max_ee_tracking_err_m * 1000, "EE tracking err", "mm")
        _topk(lambda m: m.max_unplanned_cube_movement_mm,
                  "unplanned cube movement", "mm")
        _topk(lambda m: m.held_tilt_deg, "held cube tilt", "°")
        _topk(lambda m: m.max_cube_tilt_deg, "max placed-cube tilt", "°")
        _topk(lambda m: m.held_offset_drift_mm, "held cube offset drift", "mm")
