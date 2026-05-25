"""Profile every IK call made during a single put-upright check.

Patches ``mink.Configuration.integrate_inplace`` and tracks the
calling frame (file:line + function) so we can attribute each IK
convergence step to its phase in put_upright.  Reports:

  * total IK calls
  * per-(file, line) call count + cumulative time
  * per-function (caller) call count + cumulative time
  * for each IK call: convergence iterations
"""
from __future__ import annotations

import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    _make_executor,
    check_action,
)


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    with tempfile.TemporaryDirectory(prefix="ik_profile_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        bridge, _ = make_multilevel_blocks_bridge(env, ws, cfg)
        executor = _make_executor(env, ws, cfg, fast=True)

        # Patch env.ik.converge_ik to record callsite + iteration count.
        # mink's converge_ik internally iterates until error <= threshold
        # OR max_iters.  We patch to wrap the call and record the iter
        # count via instrumenting the internal loop.

        orig = env.ik.converge_ik
        records = []  # list of (caller_file, caller_line, caller_func, elapsed, iters_used)

        def patched(dt):
            # Find the calling frame (skip the patcher itself).
            f = sys._getframe(1)
            while ("ik.py" in f.f_code.co_filename
                       or "feasibility.py" in f.f_code.co_filename and f.f_code.co_name == "_filter_quats_by_anchor_ik"):
                f = f.f_back
                if f is None:
                    break
            if f is None:
                site = ("<unknown>", 0, "<unknown>")
            else:
                site = (Path(f.f_code.co_filename).name,
                            f.f_lineno, f.f_code.co_name)
            # Wrap the inner loop by running mink manually with iter counter.
            t0 = time.perf_counter()
            # Replicate mink's converge logic with a counter.
            import mink
            ik = env.ik
            T_wt = mink.SE3.from_mocap_name(ik.model, ik.data, ik.target_name)
            ik.ee_task.set_target(T_wt)
            ik.posture_task.set_target_from_configuration(ik.configuration)
            iters_used = 0
            ok = False
            for i in range(ik.max_iters):
                vel = mink.solve_ik(ik.configuration, ik.tasks(), dt,
                                              ik.solver, 1e-3)
                ik.configuration.integrate_inplace(vel, dt)
                err = ik.ee_task.compute_error(ik.configuration)
                pos_ok = np.linalg.norm(err[:3]) <= ik.pos_threshold
                ori_ok = np.linalg.norm(err[3:]) <= ik.ori_threshold
                iters_used = i + 1
                if pos_ok and ori_ok:
                    ok = True
                    break
            elapsed = time.perf_counter() - t0
            records.append((*site, elapsed, iters_used, ok))
            return ok

        env.ik.converge_ik = patched

        # ---- Run one put-upright ----
        state = {("held-upright", "oblong_0"): True}
        action = ("put-upright", "oblong_0",
                       "stack_L0__7_3", "stack_L1__7_3")
        # warm-up (don't count)
        records.clear()
        t0 = time.perf_counter()
        res = check_action(env, ws, cfg, state, action,
                                fast=True, executor=executor)
        wall = time.perf_counter() - t0

        # ---- Report ----
        n = len(records)
        total_ik_time = sum(r[3] for r in records)
        print(f"put-upright on stack_L0__7_3/stack_L1__7_3 (held)")
        print(f"  total wall-clock: {wall*1000:.0f} ms  success={res['success']}")
        print(f"  IK calls: {n}  total IK time: {total_ik_time*1000:.0f} ms"
                  f"  ({total_ik_time/wall*100:.0f}% of elapsed)")
        print()
        # Per-(file:line) breakdown
        per_site = defaultdict(lambda: [0, 0.0, 0])  # [count, time, iters]
        for fname, lineno, func, elapsed, iters, ok in records:
            key = f"{fname}:{lineno} ({func})"
            per_site[key][0] += 1
            per_site[key][1] += elapsed
            per_site[key][2] += iters
        print("Per-callsite (caller of converge_ik):")
        print(f"  {'count':>5s} {'total ms':>9s} {'avg iters':>10s} {'avg ms':>7s} site")
        for key, (cnt, t, iters) in sorted(per_site.items(),
                                                      key=lambda kv: -kv[1][1]):
            print(f"  {cnt:>5d} {t*1000:>9.1f} {iters/cnt:>10.0f} "
                      f"{t/cnt*1000:>7.1f}  {key}")
        print()
        # Per-iteration histogram
        iters_used = [r[4] for r in records]
        print(f"Iter histogram (max_iters={env.ik.max_iters}):")
        bins = [(0, 10), (10, 30), (30, 50), (50, 75), (75, 100), (100, 200)]
        for lo, hi in bins:
            n_in = sum(1 for it in iters_used if lo <= it < hi)
            print(f"  {lo:>3d}-{hi:<3d} iters: {n_in:>3d}")
        # Show cap-hitting (iters == max_iters means unconverged)
        n_cap = sum(1 for r in records if r[4] == env.ik.max_iters)
        n_ok = sum(1 for r in records if r[5])
        print(f"  iters == cap ({env.ik.max_iters}): {n_cap} (unconverged)")
        print(f"  converged: {n_ok} / {n}  fail rate: {(n-n_ok)/n*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
