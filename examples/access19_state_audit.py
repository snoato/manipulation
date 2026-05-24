"""Run the L4 return-all plan via FULL with state-audit instrumentation.

For every action, captures:
  - arm tracking err per execute_path (planned EE xyz vs reached)
  - unplanned cube displacement (cubes NOT being acted on)
  - held cube tilt at end of action
  - held cube offset drift from initial attach
  - chain failure prints

Prints a summary report at the end and writes a JSON timeline.

Usage::

  python examples/access19_state_audit.py
  python examples/access19_state_audit.py --no-return        # OoI-only
  python examples/access19_state_audit.py --actions-limit 45 # short run
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    _dispatch, _fast_env,
)
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.planner import (
    oracle_plan, phased_plan,
)
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19.state import restore_state
from tampanda.symbolic.domains.access19.state_audit import StateAuditor
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-return", action="store_true",
                          help="OoI-only goal (default: return-all)")
    p.add_argument("--actions-limit", type=int, default=None,
                          help="stop after N actions (useful for short runs)")
    p.add_argument("--out", type=Path,
                          default=Path("/tmp/access19_state_audit.json"))
    args = p.parse_args()

    return_all = not args.no_return
    with tempfile.TemporaryDirectory(prefix="access19_audit_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = _setup(Path(td))
        template = canonical_18(return_blockers=return_all)
        init = template_source_layout(template)
        goal = template_goal_layout(template)

        print(f"=== access-19 state-audit "
                  f"({'return-all' if return_all else 'OoI-only'}) ===")
        print("  planning via FAST ...", flush=True)
        t0 = time.perf_counter()
        if return_all:
            res = phased_plan(env, ws, cfg, init, goal, _OBJECT_NAMES,
                                   pick_fn, put_fn,
                                   executor=executor, home_qpos=shelf_home,
                                   fast=True, phase2_time_budget_s=300.0,
                                   phase2_max_states=30000)
        else:
            goal_cell = Cell.parse(goal["ooi"])
            res = oracle_plan(env, ws, cfg, init, goal_cell, _OBJECT_NAMES,
                                   pick_fn, put_fn,
                                   executor=executor, home_qpos=shelf_home,
                                   fast=True, max_actions=200,
                                   max_backtrack=64)
        if not res.success:
            print("  planning failed; aborting.", file=sys.stderr)
            return 1
        plan = list(res.plan)
        if args.actions_limit:
            plan = plan[: args.actions_limit]
        print(f"  planned in {time.perf_counter()-t0:.0f}s "
                  f"({len(plan)} actions)")

        # Restore env to the source layout.
        init_state = _layout_to_state(init, held=None)
        restore_state(env, ws, cfg, init_state, _OBJECT_NAMES,
                          home_qpos=shelf_home)

        # Run plan via FULL with audit instrumentation.
        auditor = StateAuditor(env, ws, _OBJECT_NAMES)
        print(f"  running FULL replay with audit ({len(plan)} actions) ...",
                  flush=True)
        with auditor.instrument():
            for i, action in enumerate(plan):
                # Same per-action prelude as check_action_sequence.
                env.data.qpos[: len(shelf_home)] = shelf_home
                env.data.qvel[:] = 0.0
                mujoco.mj_forward(env.model, env.data)
                if getattr(env, "_attached", None) is not None:
                    env._apply_attachment()
                    mujoco.mj_forward(env.model, env.data)

                auditor.begin_action(i, action)
                with auditor.capture_chain_prints():
                    try:
                        ok = _dispatch(env, ws, pick_fn, put_fn, action)
                    except Exception as exc:
                        print(f"  action {i+1} raised "
                                  f"{type(exc).__name__}: {exc}")
                        ok = False
                auditor.end_action(ok)

                if not ok:
                    print(f"  STOPPING — action {i+1} ({action}) failed.")
                    break

        auditor.summarise(top_k=8)
        auditor.dump_json(str(args.out))
        print(f"\n  audit JSON: {args.out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
