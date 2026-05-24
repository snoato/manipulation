"""Phase 5 — 18-blocker held-out FULL-executor validation.

Generates the canonical 18-blocker access-19 instance via the FAST-
based ``phased_plan``, then re-runs every action of the resulting
plan through ``check_action_sequence(fast=False)`` so real physics
runs.  Reports FAST↔FULL agreement + per-action timing.

If any action that FAST accepted is rejected by FULL, the data-gen
pipeline can't be trusted for the L4 scenario — the script prints
the disagreement and exits non-zero.

Run::

  python examples/access19_full_validation.py
  python examples/access19_full_validation.py --return-all
  python examples/access19_full_validation.py --n-instances 3 --seed 42
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    check_action_sequence,
)
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.planner import (
    oracle_plan, phased_plan,
)
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19.templates import (
    canonical_18,
    goal_layout as template_goal_layout,
    source_layout as template_source_layout,
)
from tampanda.symbolic.workspace import Cell
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner


_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]


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


def _validate_one(
    instance_idx: int,
    return_all: bool,
    args,
) -> Tuple[bool, Dict[str, float], List[str]]:
    """Run one validation cycle: FAST plan → FAST replay → FULL replay.

    Returns ``(agreement_ok, timings, notes)``.
    """
    notes: List[str] = []
    timings: Dict[str, float] = {}

    with tempfile.TemporaryDirectory(prefix="access19_eval_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = _setup(Path(td))
        template = canonical_18(return_blockers=return_all)
        init = template_source_layout(template)
        goal = template_goal_layout(template)

        # Plan via FAST.
        t0 = time.perf_counter()
        if return_all:
            res = phased_plan(
                env, ws, cfg, init, goal, _OBJECT_NAMES,
                pick_fn, put_fn, executor=executor, home_qpos=shelf_home,
                fast=True, phase2_time_budget_s=args.time_budget,
                phase2_max_states=30000,
            )
        else:
            goal_cell = Cell.parse(goal["ooi"])
            res = oracle_plan(
                env, ws, cfg, init, goal_cell, _OBJECT_NAMES,
                pick_fn, put_fn, executor=executor, home_qpos=shelf_home,
                fast=True, max_actions=200, max_backtrack=64,
            )
        timings["plan_s"] = time.perf_counter() - t0
        if not res.success:
            notes.append("FAST planning failed")
            return False, timings, notes
        plan = list(res.plan)
        notes.append(f"plan_len={len(plan)} "
                          f"({res.n_feasibility_checks} feasibility checks)")

        # FAST replay sanity check.
        init_state = _layout_to_state(init, held=None)
        t0 = time.perf_counter()
        fast_replay = check_action_sequence(
            env, ws, cfg, init_state, plan, _OBJECT_NAMES,
            pick_fn, put_fn, executor=executor, fast=True,
            home_qpos=shelf_home, short_circuit=False,
        )
        timings["fast_replay_s"] = time.perf_counter() - t0
        if not fast_replay["success"]:
            notes.append("FAST replay failed (oracle vs replay disagreement)")
            return False, timings, notes

        # FULL replay — real physics, no monkey patches.
        t0 = time.perf_counter()
        full_replay = check_action_sequence(
            env, ws, cfg, init_state, plan, _OBJECT_NAMES,
            pick_fn, put_fn, executor=executor, fast=False,
            home_qpos=shelf_home, short_circuit=False,
        )
        timings["full_replay_s"] = time.perf_counter() - t0

        # Per-action agreement.
        n_disagree = 0
        for i, (fast_a, full_a) in enumerate(zip(
                fast_replay["per_action"], full_replay["per_action"]
        )):
            if fast_a["success"] != full_a["success"]:
                n_disagree += 1
                notes.append(
                    f"action {i+1}/{len(plan)} DISAGREE: "
                    f"fast={fast_a['success']} full={full_a['success']} "
                    f"action={fast_a['action']}"
                )
        timings["per_action_full_ms"] = (
            1000 * timings["full_replay_s"] / len(plan)
        )
        timings["per_action_fast_ms"] = (
            1000 * timings["fast_replay_s"] / len(plan)
        )
        notes.append(
            f"FULL replay overall: "
            f"{'success' if full_replay['success'] else 'FAIL'}; "
            f"per-action disagree: {n_disagree}/{len(plan)}"
        )
        return (
            (full_replay["success"] and n_disagree == 0),
            timings,
            notes,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-instances", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget", type=float, default=300.0)
    p.add_argument("--return-all", action="store_true",
                          help="full-return goal (every blocker back at "
                                   "original cell).  Default: OoI-only goal.")
    args = p.parse_args()

    print(f"=== Phase 5 — 18-blocker FULL validation "
              f"(n={args.n_instances}, return_all={args.return_all}) ===\n")
    any_fail = False
    summaries: List[str] = []
    for i in range(args.n_instances):
        print(f"--- instance {i+1}/{args.n_instances} ---")
        t0 = time.perf_counter()
        ok, timings, notes = _validate_one(i, args.return_all, args)
        wall = time.perf_counter() - t0
        for note in notes:
            print(f"  {note}")
        summary = (
            f"instance {i+1}: {'PASS' if ok else 'FAIL'} "
            f"plan={timings.get('plan_s', 0):.1f}s "
            f"fast_replay={timings.get('fast_replay_s', 0):.1f}s "
            f"full_replay={timings.get('full_replay_s', 0):.1f}s "
            f"(fast/full per-action: "
            f"{timings.get('per_action_fast_ms', 0):.0f} / "
            f"{timings.get('per_action_full_ms', 0):.0f} ms) "
            f"total={wall:.1f}s"
        )
        print(f"  {summary}\n")
        summaries.append(summary)
        any_fail |= not ok

    print("=== summary ===")
    for s in summaries:
        print(f"  {s}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
