"""Stress-test the L4 return-all pipeline across varied 18-blocker layouts.

The canonical L4 layout (`canonical_18` with default OoI at (3, 6)) is
deterministic — one plan, one replay outcome.  This test varies:

  * OoI starting cell ∈ {(1, 6), (3, 6), (5, 6)} (the three back-row
    cube-column cells).
  * mirror_x on / off.

Generates plan + FULL replay for each variant, reports per-instance
agreement.  Goal: confirm the safe_z=0.10 fix isn't seed-specific
and the pipeline is reliable across the L4 variants the curriculum
will sample.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    check_action_sequence,
)
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.planner import phased_plan
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19.templates import (
    canonical_18,
    goal_layout as template_goal_layout,
    mirror_x,
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


def _build_variants() -> List[Tuple[str, callable]]:
    """Each variant returns a Template when called."""
    def _canon_at(ooi_ix: int, mirror: bool):
        ooi_cell = Cell("shelf_interior", ooi_ix, 6).id
        def _make():
            tpl = canonical_18(return_blockers=True, ooi_cell=ooi_cell)
            if mirror:
                tpl = mirror_x(tpl)
            return tpl
        return _make

    return [
        ("OoI@col_3, no-mirror",  _canon_at(3, False)),
        ("OoI@col_3, mirror_x",   _canon_at(3, True)),
        ("OoI@col_1, no-mirror",  _canon_at(1, False)),
        ("OoI@col_5, no-mirror",  _canon_at(5, False)),
        ("OoI@col_1, mirror_x",   _canon_at(1, True)),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--time-budget", type=float, default=300.0)
    args = p.parse_args()

    variants = _build_variants()
    print(f"=== L4 return-all stress test ({len(variants)} variants) ===\n")
    print(f"  {'variant':<26} {'plan':<10} {'plan_t':<8} "
              f"{'full_replay_t':<14} {'agree':<10}")

    any_fail = False
    for label, make_template in variants:
        with tempfile.TemporaryDirectory(prefix="access19_stress_") as td:
            env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
                _setup(Path(td))
            )
            template = make_template()
            init = template_source_layout(template)
            goal = template_goal_layout(template)

            t0 = time.perf_counter()
            res = phased_plan(env, ws, cfg, init, goal, _OBJECT_NAMES,
                                   pick_fn, put_fn,
                                   executor=executor, home_qpos=shelf_home,
                                   fast=True,
                                   phase2_time_budget_s=args.time_budget,
                                   phase2_max_states=30000)
            plan_t = time.perf_counter() - t0
            if not res.success:
                print(f"  {label:<26} PLAN FAIL ({plan_t:.0f}s)")
                any_fail = True
                continue

            init_state = _layout_to_state(init, held=None)
            t0 = time.perf_counter()
            full = check_action_sequence(
                env, ws, cfg, init_state, list(res.plan), _OBJECT_NAMES,
                pick_fn, put_fn, executor=executor, fast=False,
                home_qpos=shelf_home, short_circuit=False,
            )
            full_t = time.perf_counter() - t0
            n_act = len(res.plan)
            n_disagree = sum(1 for pa in full["per_action"]
                                  if not pa["success"])
            agree = f"{n_act - n_disagree}/{n_act}"
            ok = (n_disagree == 0)
            if not ok:
                any_fail = True
            print(f"  {label:<26} {n_act:<10} {plan_t:5.0f}s   "
                      f"{full_t:5.0f}s         {agree:<10} "
                      f"{'PASS' if ok else 'FAIL'}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
