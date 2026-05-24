"""Visualize the L4 return-all 74-action plan in the MuJoCo viewer.

Generates the canonical 18-blocker + full-return plan via FAST
(``phased_plan``), then replays it through FULL execution with the
MuJoCo passive viewer attached at real-time rate.  Resets the arm
to staging-home between actions (matching what ``check_action_
sequence`` does) so each pick/put is clearly separable.

Requires mjpython on macOS::

    mjpython examples/access19_return_all_viz.py

Headless variant (no viewer, just runs the FULL replay)::

    python examples/access19_return_all_viz.py --no-viz
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import mujoco

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    _dispatch, check_action_sequence,
)
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.planner import phased_plan
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19.state import restore_state
from tampanda.symbolic.domains.access19.templates import (
    canonical_18,
    goal_layout as template_goal_layout,
    source_layout as template_source_layout,
)
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner


_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]


def _setup(scratch_dir: Path, rate: float):
    builder, ws, cfg = make_access19_builder(scratch_dir=scratch_dir)
    env = builder.build_env(rate=rate)
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


_CACHE_PATH = Path("/tmp/access19_return_all_plan.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-viz", action="store_true",
                          help="run headless (no MuJoCo viewer)")
    p.add_argument("--rate", type=float, default=240.0,
                          help="sim rate (Hz); 240 ~ real-time for viewer")
    p.add_argument("--force-replan", action="store_true",
                          help="ignore the cached plan and re-plan from scratch")
    p.add_argument("--time-budget", type=float, default=600.0,
                          help="phase-2 planner budget (s); bump if CPU "
                                   "contended.  Default 600 s (10 min).")
    args = p.parse_args()

    use_viz = not args.no_viz
    print("=== access-19 L4 return-all visualization ===")
    template = canonical_18(return_blockers=True)
    init = template_source_layout(template)
    goal = template_goal_layout(template)

    # Try the cache first — the canonical L4 return-all plan is
    # deterministic, so once we've solved it, every subsequent viz
    # reuse it (no waiting on the 3-5 min planner under CPU
    # contention).  Cache stores the action sequence and a hash
    # tying it to the current template; if the template changes the
    # cache is invalidated.
    plan: list = []
    cache_key = (("canonical_18_return_all",) + tuple(sorted(init.items()))
                       + tuple(sorted(goal.items())))
    cache_hash = str(hash(cache_key))
    if _CACHE_PATH.exists() and not args.force_replan:
        try:
            data = json.loads(_CACHE_PATH.read_text())
            if data.get("hash") == cache_hash:
                plan = [tuple(a) for a in data["plan"]]
                print(f"  loaded cached plan from {_CACHE_PATH} "
                          f"({len(plan)} actions).  Use --force-replan to "
                          f"regenerate.")
        except Exception as exc:
            print(f"  cache read failed ({exc}); will replan.")
            plan = []

    if not plan:
        # Planning needs a fast (rate=10000) env — chain IK probes
        # call wait_idle which rate-limits, so doing it at viewer
        # rate would take ~30 min instead of ~3.
        with tempfile.TemporaryDirectory(prefix="access19_viz_plan_") as td:
            env_p, ws_p, cfg_p, executor_p, pick_p, put_p, home_p = (
                _setup(Path(td), rate=10000.0)
            )
            print(f"  planning (phased, budget={args.time_budget:.0f}s) "
                      f"...", flush=True)
            t0 = time.perf_counter()
            res = phased_plan(env_p, ws_p, cfg_p, init, goal, _OBJECT_NAMES,
                                  pick_p, put_p,
                                  executor=executor_p, home_qpos=home_p,
                                  fast=True,
                                  phase2_time_budget_s=args.time_budget,
                                  phase2_max_states=30000)
            if not res.success:
                print("  planning failed; aborting.", file=sys.stderr)
                for note in res.notes[-5:]:
                    print(f"    {note}", file=sys.stderr)
                return 1
            plan = list(res.plan)
            print(f"  planned in {time.perf_counter()-t0:.0f}s "
                      f"({len(plan)} actions, "
                      f"{res.n_feasibility_checks} checks)")
        # Cache it.
        try:
            _CACHE_PATH.write_text(json.dumps({
                "hash": cache_hash,
                "plan": [list(a) for a in plan],
            }))
            print(f"  cached plan → {_CACHE_PATH}")
        except Exception as exc:
            print(f"  cache write failed ({exc})")

    # Rebuild a fresh env for replay at the viewer rate.  We use
    # check_action_sequence(fast=False) for the replay because it
    # already handles snap-between-actions (cube positions reset to
    # symbolic between actions, attachment kept).  This is the same
    # path the validator uses for L4 return-all, which is verified
    # 74/74.
    with tempfile.TemporaryDirectory(prefix="access19_viz_") as td:
        rate = args.rate if use_viz else 10000.0
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
            _setup(Path(td), rate=rate)
        )
        if use_viz:
            viewer = env.launch_viewer()
            if viewer is None:
                print("  WARNING: viewer didn't launch.  On macOS you "
                          "must run with 'mjpython', not 'python'.",
                          file=sys.stderr)

        init_state = _layout_to_state(init, held=None)
        restore_state(env, ws, cfg, init_state, _OBJECT_NAMES,
                          home_qpos=shelf_home)
        if use_viz and env.viewer is not None:
            env.viewer.sync()
            time.sleep(1.0)        # give the user a beat to see start state

        print(f"\n  replaying {len(plan)} actions via FULL ...")
        t_start = time.perf_counter()
        res2 = check_action_sequence(
            env, ws, cfg, init_state, plan, _OBJECT_NAMES,
            pick_fn, put_fn, executor=executor, fast=False,
            home_qpos=shelf_home, short_circuit=False,
        )
        elapsed = time.perf_counter() - t_start
        n_ok = sum(1 for pa in res2["per_action"] if pa["success"])
        print(f"  done: {n_ok}/{len(plan)} actions OK in {elapsed:.1f}s")
        if not res2["success"]:
            for i, pa in enumerate(res2["per_action"]):
                if not pa["success"]:
                    print(f"    FIRST FAILURE @ action {i+1}: "
                              f"{pa['action']}  err={pa.get('error')}")
                    break
        # Keep the viewer alive so the user can inspect the final state.
        if use_viz and env.viewer is not None:
            print("  viewer alive — close the window to exit.")
            try:
                while env.viewer.is_running():
                    env.viewer.sync()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
    return 0 if res2["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
