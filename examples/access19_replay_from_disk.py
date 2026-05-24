"""Round-trip validation: load stored PDDL plans from disk and replay
through ``check_action_sequence(fast=False)``.

Picks one instance per (split, level) tuple, parses both the
``.pddl`` problem file (for the initial layout + goal) and the
``.pddl.plan`` file (for the action sequence), reconstructs the
runtime state, and replays in real physics.

Confirms the on-disk format actually round-trips — catches PDDL
writer bugs, plan-file parsing bugs, or any silent corruption
during the train_120 generation pass.
"""
from __future__ import annotations

import argparse
import re
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
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
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


def _parse_problem(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Extract initial (occupied cell obj) + goal layouts from a PDDL file."""
    text = path.read_text()
    init_layout: Dict[str, str] = {}
    for m in re.finditer(r"\(occupied\s+(\S+)\s+(\S+)\)", text):
        cell, obj = m.group(1), m.group(2)
        # We must distinguish :init from :goal occupied predicates.  The
        # :goal sits inside (:goal (and ...)) so use a state machine.
    # Simpler: split sections.
    init_section = text.split("(:init")[1].split("(:goal")[0]
    goal_section = text.split("(:goal")[1].split("(:goal")[0]
    init_layout = {}
    for m in re.finditer(r"\(occupied\s+(\S+)\s+(\S+)\)", init_section):
        init_layout[m.group(2)] = m.group(1)
    goal_layout: Dict[str, str] = {}
    for m in re.finditer(r"\(occupied\s+(\S+)\s+(\S+)\)", goal_section):
        goal_layout[m.group(2)] = m.group(1)
    return init_layout, goal_layout


def _parse_plan(path: Path) -> List[Tuple]:
    """Read a .pddl.plan file into a list of (action_name, *args) tuples."""
    actions: List[Tuple] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # Expect (action arg1 arg2 ...)
        m = re.match(r"\((\S+)\s+(.+)\)", line)
        if not m:
            continue
        actions.append((m.group(1), *m.group(2).split()))
    return actions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/access19"))
    p.add_argument("--instances-per-level", type=int, default=1)
    args = p.parse_args()

    print("=== access-19 round-trip replay validation ===\n")
    print(f"  data dir: {args.data_dir}")
    print(f"  sampling {args.instances_per_level} instance(s) per "
              f"(split, level)\n")
    print(f"  {'split':<8} {'level':<5} {'file':<20} {'len':<5} "
              f"{'replay_t':<10} {'result':<10}")

    any_fail = False
    with tempfile.TemporaryDirectory(prefix="access19_replay_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
            _setup(Path(td))
        )

        for split in ("train", "val", "test"):
            for level in (0, 1, 2, 3, 4):
                level_dir = args.data_dir / split / f"L{level}"
                if not level_dir.exists():
                    continue
                pddl_files = sorted(level_dir.glob("config_*.pddl"))
                # Exclude .pddl.plan from this glob.
                pddl_files = [f for f in pddl_files if f.suffix == ".pddl"]
                pddl_files = pddl_files[: args.instances_per_level]

                for pddl_path in pddl_files:
                    plan_path = pddl_path.with_suffix(".pddl.plan")
                    if not plan_path.exists():
                        print(f"  {split:<8} L{level:<4} "
                                  f"{pddl_path.name:<20} MISSING plan")
                        any_fail = True
                        continue
                    init_layout, _ = _parse_problem(pddl_path)
                    plan = _parse_plan(plan_path)

                    init_state = _layout_to_state(init_layout, held=None)
                    t0 = time.perf_counter()
                    res = check_action_sequence(
                        env, ws, cfg, init_state, plan, _OBJECT_NAMES,
                        pick_fn, put_fn, executor=executor, fast=False,
                        home_qpos=shelf_home, short_circuit=False,
                    )
                    dt = time.perf_counter() - t0
                    ok = res["success"]
                    print(f"  {split:<8} L{level:<4} {pddl_path.name:<20} "
                              f"{len(plan):<5} {dt:5.1f}s     "
                              f"{'PASS' if ok else 'FAIL':<10}")
                    if not ok:
                        any_fail = True
                        for i, pa in enumerate(res["per_action"]):
                            if not pa["success"]:
                                print(f"      first fail @ action {i+1}: "
                                          f"{pa['action']}")
                                break
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
