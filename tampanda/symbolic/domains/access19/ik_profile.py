"""Phase-0 instrumentation for access-19 feasibility checks.

Monkey-patches ``LinearIKPlanner.plan_to_pose`` and
``LinearIKPlanner.plan_joint_lerp`` from inside access19 so per-call
wall time + caller-frame info can be recorded without touching shared
planner infra.

Enable at the start of a profiling run, disable + dump at the end.
The original methods are restored on disable.

Caller-frame walk: ``plan_*`` is called from a closure (``_try_lerp`` /
``_try_cartesian``) inside ``chains.py``; the closure is itself called
from a chain phase function (``_put_interior``, ``_put_deck``,
``_pick_interior``, ``_pick_deck``).  We record both frames so the
phase + specific call site (line number in ``chains.py``) are
identifiable.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tampanda.planners.linear_ik import LinearIKPlanner


_orig_plan_to_pose = None
_orig_plan_joint_lerp = None
_TIMINGS: List[Dict[str, Any]] = []
_ENABLED = False


def _frames():
    """Walk up the stack to find the chains.py phase function.

    Returns ``(closure_name, phase_name, phase_line)`` — the
    ``_try_*`` closure that called plan_*, and the chain phase
    function that called the closure.  Defaults to ``("?", "?", -1)``
    if the expected frames are not found.
    """
    f = sys._getframe(2)  # skip ourselves + the wrapper
    closure = f.f_code.co_name
    phase = "?"
    phase_line = -1
    g = f.f_back
    if g is not None:
        phase = g.f_code.co_name
        phase_line = g.f_lineno
    return closure, phase, phase_line


def _plan_to_pose_timed(self, target_pos, target_quat, dt=0.005,
                          n_substeps=None, slerp_orientation=True):
    t0 = time.perf_counter()
    result = _orig_plan_to_pose(self, target_pos, target_quat, dt=dt,
                                n_substeps=n_substeps,
                                slerp_orientation=slerp_orientation)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    closure, phase, phase_line = _frames()
    _TIMINGS.append({
        "method": "plan_to_pose",
        "closure": closure,
        "phase": phase,
        "phase_line": phase_line,
        "ms": dt_ms,
        "ok": result is not None,
        "n_substeps": n_substeps if n_substeps is not None
                       else self.n_substeps,
        "target": [float(x) for x in np.asarray(target_pos).ravel()[:3]],
    })
    return result


def _plan_joint_lerp_timed(self, target_pos, target_quat, dt=0.005,
                             n_substeps=16):
    t0 = time.perf_counter()
    result = _orig_plan_joint_lerp(self, target_pos, target_quat, dt=dt,
                                     n_substeps=n_substeps)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    closure, phase, phase_line = _frames()
    _TIMINGS.append({
        "method": "plan_joint_lerp",
        "closure": closure,
        "phase": phase,
        "phase_line": phase_line,
        "ms": dt_ms,
        "ok": result is not None,
        "n_substeps": n_substeps,
        "target": [float(x) for x in np.asarray(target_pos).ravel()[:3]],
    })
    return result


def enable() -> None:
    """Install timing wrappers around the planner methods.  Idempotent."""
    global _orig_plan_to_pose, _orig_plan_joint_lerp, _ENABLED
    if _ENABLED:
        return
    _orig_plan_to_pose = LinearIKPlanner.plan_to_pose
    _orig_plan_joint_lerp = LinearIKPlanner.plan_joint_lerp
    LinearIKPlanner.plan_to_pose = _plan_to_pose_timed
    LinearIKPlanner.plan_joint_lerp = _plan_joint_lerp_timed
    _ENABLED = True


def disable() -> None:
    """Restore original planner methods.  Idempotent."""
    global _ENABLED
    if not _ENABLED:
        return
    LinearIKPlanner.plan_to_pose = _orig_plan_to_pose
    LinearIKPlanner.plan_joint_lerp = _orig_plan_joint_lerp
    _ENABLED = False


def clear() -> None:
    _TIMINGS.clear()


def timings() -> List[Dict[str, Any]]:
    return list(_TIMINGS)


def dump(path: Path) -> None:
    Path(path).write_text(json.dumps(_TIMINGS, indent=2))


def summary() -> str:
    """One-shot stdout summary: per phase aggregate."""
    if not _TIMINGS:
        return "(no timings recorded)"
    from collections import defaultdict
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    ok_by: Dict[tuple, int] = defaultdict(int)
    for t in _TIMINGS:
        key = (t["method"], t["phase"], t["phase_line"], t["n_substeps"])
        buckets[key].append(t["ms"])
        if t["ok"]:
            ok_by[key] += 1
    lines = [
        f"{'method':<16} {'phase':<18} {'line':>4} {'nsub':>5} "
        f"{'count':>6} {'ok%':>5} {'mean':>7} {'p50':>7} {'p95':>7} "
        f"{'max':>7}",
    ]
    for key in sorted(buckets, key=lambda k: -sum(buckets[k])):
        method, phase, line, nsub = key
        arr = np.asarray(buckets[key])
        ok_frac = 100.0 * ok_by[key] / len(arr)
        lines.append(
            f"{method:<16} {phase:<18} {line:>4} {nsub:>5} "
            f"{len(arr):>6} {ok_frac:>4.0f}% "
            f"{arr.mean():>6.1f} {np.median(arr):>6.1f} "
            f"{np.percentile(arr, 95):>6.1f} {arr.max():>6.1f}"
        )
    return "\n".join(lines)
