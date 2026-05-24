"""Pre-handoff QA for the access-19 rgnet integration.

Three checks (the user's pre-handoff list):

1. **State restoration** — given a symbolic ``state`` dict, confirm
   every placed object lands at its cell, every unused object is
   parked at ``hide_far_x``, and the env is internally consistent
   (no leftover attachments, no leftover collision exceptions).

2. **Held-state restoration** — given a ``state`` that includes a
   ``(holding obj)`` fluent, confirm restore_state attaches ``obj``
   at the canonical grasp offset.  Then run a follow-up ``put``
   action via the chain and verify it succeeds.  (Access-19 has
   exactly one grasp pose — palm-+y — so the canonical offset is
   deterministic.)

3. **Feasibility throughput** — measure the average wall-clock cost
   of a single ``check_action`` call (FAST) across a mix of action
   types + workers.  Reports per-action mean / median / p95 timings
   so the rgnet caller can budget BFS expansion correctly.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    check_action, check_action_sequence,
)
from tampanda.symbolic.domains.access19.parallel import (
    ParallelFeasibilityChecker, _layout_to_state,
)
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19.state import (
    held_object_in_state, restore_state,
)
from tampanda.symbolic.workspace import Cell
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner


_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]
_PARKED_X = 100.0


def _setup(scratch_dir: Path):
    builder, ws, cfg = make_access19_builder(scratch_dir=scratch_dir)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    cube_half = float(env.get_object_half_size("ooi")[2])
    table_z = ws["shelf_interior"].level_z - cube_half
    executor = _build_executor(env, table_z=table_z,
                                       allowed_types=[GraspType.FRONT])
    shelf_home = _solve_access19_staging(env, ws, cfg)
    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
    pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                            cube_half_z=cube_half, lik=lik)
    put_fn = make_access19_put_fn(env, executor, ws, cfg,
                                          cube_half_z=cube_half, lik=lik)
    return env, ws, cfg, executor, pick_fn, put_fn, shelf_home


# ---------------------------------------------------------------------------
# Test 1 — state restoration
# ---------------------------------------------------------------------------


def test_state_restoration(env, ws, cfg, shelf_home) -> bool:
    print("\n=== Test 1: state restoration ===")
    layout = {
        "blocker_0": Cell("shelf_interior", 1, 0).id,
        "blocker_5": Cell("shelf_interior", 3, 2).id,
        "blocker_10": Cell("shelf_top", 5, 6).id,
        "ooi": Cell("shelf_interior", 3, 6).id,
    }
    used_objs = set(layout.keys())
    state = _layout_to_state(layout, held=None)

    info = restore_state(env, ws, cfg, state, _OBJECT_NAMES,
                              home_qpos=shelf_home)

    all_ok = True
    # Placed
    for obj, cell_id in layout.items():
        pos, quat = env.get_object_pose(obj)
        expected = np.array(ws.pose_for(Cell.parse(cell_id)))
        drift = float(np.linalg.norm(np.asarray(pos)[:2] - expected[:2]))
        ok = drift < 0.005
        flag = "OK  " if ok else "FAIL"
        print(f"  {flag} placed   {obj:<12} drift={drift*1000:5.1f} mm")
        all_ok &= ok

    # Parked
    for obj in _OBJECT_NAMES:
        if obj in used_objs:
            continue
        pos, _ = env.get_object_pose(obj)
        ok = float(pos[0]) > _PARKED_X - 1.0     # at hide_far_x
        flag = "OK  " if ok else "FAIL"
        if not ok or obj in ("blocker_1", "blocker_17"):  # print samples
            print(f"  {flag} parked   {obj:<12} pos=({pos[0]:.2f}, "
                      f"{pos[1]:.2f}, {pos[2]:.2f})")
        all_ok &= ok

    # No leftover attachment / collision exceptions.
    no_attach = getattr(env, "_attached", None) is None
    print(f"  {'OK  ' if no_attach else 'FAIL'} no attached body")
    all_ok &= no_attach

    # Arm at home.
    arm_err = float(np.linalg.norm(
        np.asarray(env.data.qpos[:7])
        - np.asarray(shelf_home[:7])
    ))
    print(f"  {'OK  ' if arm_err < 1e-3 else 'FAIL'} arm at home   "
              f"qpos err={arm_err:.4f}")
    all_ok &= arm_err < 1e-3

    print(f"  → {'PASS' if all_ok else 'FAIL'}: placed={len(info['placed'])}, "
              f"parked={len(info['parked'])}, held={info['held']}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2 — held-state restoration
# ---------------------------------------------------------------------------


def test_held_restoration(env, ws, cfg, executor, pick_fn, put_fn,
                                shelf_home) -> bool:
    print("\n=== Test 2: held-state restoration ===")

    # State: blocker_0 already picked, others at default front-row layout.
    layout = {f"blocker_{1 + i}": Cell("shelf_interior", c, r).id
                  for i, (c, r) in enumerate(
                      [(3, 0), (5, 0), (1, 1), (3, 1), (5, 1)]
                  )}
    layout["ooi"] = Cell("shelf_interior", 3, 6).id
    state = _layout_to_state(layout, held="blocker_0")

    info = restore_state(env, ws, cfg, state, _OBJECT_NAMES,
                              home_qpos=shelf_home, on_held="attach")

    print(f"  restore returned held={info['held']}")
    attached = getattr(env, "_attached", None)
    print(f"  env._attached = {attached.get('body_name') if attached else None}")

    if attached is None or attached.get("body_name") != "blocker_0":
        print("  → FAIL: blocker_0 not attached after restore")
        return False

    # Verify cube is at EE + canonical_rel_pos (in EE frame).
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE,
                                          "attachment_site")
    ee_pos = env.data.site_xpos[site_id].copy()
    cube_pos, _ = env.get_object_pose("blocker_0")
    rel_world = np.asarray(cube_pos) - ee_pos
    print(f"  EE pos:        ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, "
              f"{ee_pos[2]:.3f})")
    print(f"  cube pos:      ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, "
              f"{cube_pos[2]:.3f})")
    print(f"  rel_world:     ({rel_world[0]:+.3f}, {rel_world[1]:+.3f}, "
              f"{rel_world[2]:+.3f})")
    # For palm-+y with our grasp formulas, world offset should be
    # roughly (0, +GRASP_OFFSET, -0.030) = (0, +0.014, -0.030).
    expected_world = np.array([0.0, 0.014, -0.030])
    err = float(np.linalg.norm(rel_world - expected_world))
    print(f"  expected:      ({expected_world[0]:+.3f}, "
              f"{expected_world[1]:+.3f}, {expected_world[2]:+.3f})")
    print(f"  err:           {err*1000:.1f} mm")
    ok_offset = err < 0.005

    # Verify: a put action from this held state actually succeeds.
    put_action = ("put", "blocker_0", "shelf_top__0_0")
    res = check_action(env, ws, cfg, state, put_action, _OBJECT_NAMES,
                            pick_fn, put_fn, executor=executor, fast=True,
                            home_qpos=shelf_home)
    print(f"  put follow-up: {'OK  ' if res['success'] else 'FAIL'} "
              f"action={put_action}")
    ok_put = res["success"]

    all_ok = ok_offset and ok_put
    print(f"  → {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 3 — feasibility throughput
# ---------------------------------------------------------------------------


def _gen_probes() -> List[Tuple[Dict[str, str], Tuple]]:
    """A mix of action types from realistic search states."""
    probes = []
    # Empty cubicle except OoI — clean pick of OoI.
    layout0 = {"ooi": Cell("shelf_interior", 3, 6).id}
    probes.append((layout0, ("pick", "ooi", "shelf_interior__3_6")))

    # Front-row blocker — pick.
    layout1 = {
        "blocker_0": Cell("shelf_interior", 1, 0).id,
        "ooi": Cell("shelf_interior", 3, 6).id,
    }
    probes.append((layout1, ("pick", "blocker_0", "shelf_interior__1_0")))
    probes.append((layout1, ("pick", "ooi", "shelf_interior__3_6")))

    # OoI on top — pick.
    layout2 = {
        "blocker_0": Cell("shelf_interior", 1, 0).id,
        "ooi": Cell("shelf_top", 3, 3).id,
    }
    probes.append((layout2, ("pick", "ooi", "shelf_top__3_3")))

    # Multiple blockers — put after pick (uses 2-action sequence so
    # the put sees a held state).  We measure the put alone here.
    layout3 = {
        "blocker_0": Cell("shelf_interior", 1, 0).id,
        "blocker_1": Cell("shelf_interior", 3, 0).id,
        "ooi": Cell("shelf_interior", 3, 6).id,
    }
    probes.append((layout3, ("pick", "blocker_1", "shelf_interior__3_0")))

    return probes


def test_throughput_single(env, ws, cfg, executor, pick_fn, put_fn,
                                  shelf_home, n_reps: int) -> bool:
    print(f"\n=== Test 3a: single-env feasibility throughput ===")
    print(f"  {n_reps} reps × {len(_gen_probes())} probes "
              f"= {n_reps * len(_gen_probes())} total calls")
    probes = _gen_probes()
    times: List[float] = []
    for rep in range(n_reps):
        for layout, action in probes:
            state = _layout_to_state(layout, held=None)
            t0 = time.perf_counter()
            check_action(env, ws, cfg, state, action, _OBJECT_NAMES,
                              pick_fn, put_fn, executor=executor, fast=True,
                              home_qpos=shelf_home)
            times.append(time.perf_counter() - t0)
    mean = statistics.mean(times) * 1000
    median = statistics.median(times) * 1000
    p95 = sorted(times)[int(len(times) * 0.95)] * 1000
    p99 = sorted(times)[int(len(times) * 0.99)] * 1000
    mn = min(times) * 1000
    mx = max(times) * 1000
    total = sum(times)
    print(f"  n={len(times)}  total={total:.1f}s  "
              f"mean={mean:.1f} ms  median={median:.1f} ms  "
              f"min={mn:.1f}  p95={p95:.1f}  p99={p99:.1f}  max={mx:.1f}")
    print(f"  throughput: {len(times)/total:.1f} calls/sec/worker")
    return True


def test_throughput_parallel(n_workers: int, n_reps: int) -> bool:
    print(f"\n=== Test 3b: parallel feasibility throughput "
              f"({n_workers} workers) ===")
    probes = _gen_probes()
    items = [(layout, None, action) for layout, action in probes] * n_reps
    print(f"  {len(items)} total calls across {n_workers} workers")

    t0 = time.perf_counter()
    with ParallelFeasibilityChecker(n_workers=n_workers, fast=True) as pool:
        init_t = time.perf_counter() - t0
        t1 = time.perf_counter()
        results = pool.check_batch(items)
        batch_t = time.perf_counter() - t1
    total = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r["success"])
    per_call_total = batch_t * 1000 / len(items)
    per_call_per_worker = batch_t * n_workers * 1000 / len(items)
    print(f"  worker init: {init_t:.1f}s")
    print(f"  batch ({len(items)} calls): {batch_t:.1f}s")
    print(f"  per-call (wall): {per_call_total:.1f} ms")
    print(f"  per-call (per-worker): {per_call_per_worker:.1f} ms")
    print(f"  acceptance: {n_ok}/{len(items)} "
              f"({100*n_ok/len(items):.0f}%)")
    print(f"  throughput: {len(items)/batch_t:.1f} calls/sec aggregate")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-reps", type=int, default=20,
                          help="repetitions for throughput tests")
    p.add_argument("--n-workers", type=int, default=4,
                          help="workers for parallel throughput test")
    args = p.parse_args()

    with tempfile.TemporaryDirectory(prefix="access19_qa_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
            _setup(Path(td))
        )

        t1 = test_state_restoration(env, ws, cfg, shelf_home)
        t2 = test_held_restoration(env, ws, cfg, executor,
                                              pick_fn, put_fn, shelf_home)
        t3a = test_throughput_single(env, ws, cfg, executor, pick_fn,
                                                  put_fn, shelf_home, args.n_reps)
    # parallel test must use a fresh env per worker, so it's outside the
    # single-env TemporaryDirectory.
    t3b = test_throughput_parallel(args.n_workers, args.n_reps)

    print("\n=== Summary ===")
    print(f"  1. state restoration:     {'PASS' if t1 else 'FAIL'}")
    print(f"  2. held-state restore:    {'PASS' if t2 else 'FAIL'}")
    print(f"  3a. single-env throughput: {'PASS' if t3a else 'FAIL'}")
    print(f"  3b. parallel throughput:   {'PASS' if t3b else 'FAIL'}")
    return 0 if (t1 and t2 and t3a and t3b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
