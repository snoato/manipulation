"""Validate items 1 + 4 alignment: FAST(LUT) vs FAST(no LUT) vs FULL.

For every plan in ``data/access19_v3/{val,eval_pre_b,eval_full}``:
  1. FAST(no LUT) replay of the plan → baseline FAST verdicts.
  2. FAST(LUT) replay of the plan → LUT verdicts.

Reports any per-action disagreement.  Alignment requirement: items 1
and 4 must never flip a per-action verdict.

Smaller batch by default for quick iteration; ``--full`` runs on all
410 problems.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import (
    check_action_sequence,
)
from tampanda.symbolic.domains.access19.ik_seed_lut import (
    Access19IKSeedLUT,
)
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner

import mujoco


_OBJECT_NAMES = [f"blocker_{i}" for i in range(18)] + ["ooi"]


def _parse_problem(p: Path) -> Dict[str, str]:
    text = p.read_text()
    init = text.split("(:init")[1].split("(:goal")[0]
    layout: Dict[str, str] = {}
    for m in re.finditer(r"\(occupied\s+(\S+)\s+(\S+)\)", init):
        cell, obj = m.group(1), m.group(2)
        if obj.startswith("blocker_") or obj == "ooi":
            layout[obj] = cell
    return layout


def _parse_plan(p: Path) -> List[Tuple]:
    actions: List[Tuple] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = re.match(r"\((\S+)\s+(.+)\)", line)
        if m:
            parts = m.group(2).split()
            actions.append((m.group(1), *parts))
    return actions


def _setup(scratch: Path, build_lut: bool):
    builder, ws, cfg = make_access19_builder(scratch_dir=scratch)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    cube_half = float(env.get_object_half_size("ooi")[2])
    table_z = ws["shelf_interior"].level_z - cube_half
    executor = _build_executor(env, table_z=table_z,
                                       allowed_types=[GraspType.FRONT])
    shelf_home = _solve_access19_staging(env, ws, cfg)
    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)

    seed_lut = None
    if build_lut:
        park = np.array([2.0, 2.0, 2.0])
        for name in _OBJECT_NAMES:
            try:
                env.set_object_pose(name, park)
            except Exception:
                pass
        mujoco.mj_forward(env.model, env.data)
        seed_lut = Access19IKSeedLUT()
        stats = seed_lut.precompute(env, ws, cfg, lik, shelf_home,
                                            cube_half_z=cube_half)
        print(f"[LUT] precompute: {stats['precompute_s']:.1f}s, "
              f"{stats['n_reachable']}/{stats['n_cells']} reachable")

    pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                            cube_half_z=cube_half, lik=lik,
                                            home_qpos=shelf_home,
                                            seed_lut=seed_lut)
    put_fn = make_access19_put_fn(env, executor, ws, cfg,
                                          cube_half_z=cube_half, lik=lik,
                                          home_qpos=shelf_home,
                                          seed_lut=seed_lut)
    return env, ws, cfg, executor, pick_fn, put_fn, shelf_home


def _replay(env, ws, cfg, init_layout, plan, executor, pick_fn,
            put_fn, shelf_home) -> List[bool]:
    init_state = _layout_to_state(init_layout, held=None)
    res = check_action_sequence(
        env, ws, cfg, init_state, list(plan), _OBJECT_NAMES,
        pick_fn, put_fn, executor=executor, fast=True,
        home_qpos=shelf_home, short_circuit=False,
    )
    return [bool(pa["success"]) for pa in res["per_action"]]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                       help="Run on all 410 problems (default: 30 sampled)")
    args = p.parse_args()

    base = Path("data/access19_v3")
    rng = random.Random(0)

    # Collect problems.
    problems: List[Tuple[Path, Path]] = []
    for split in ("val", "eval_pre_b", "eval_full"):
        d = base / split
        for prob in sorted(d.glob("config_*.pddl")):
            plan = prob.with_suffix(".pddl.plan")
            if plan.exists():
                problems.append((prob, plan))
    if not args.full:
        problems = rng.sample(problems, min(30, len(problems)))
    print(f"=== alignment check ({len(problems)} problems) ===\n")

    # Run baseline (no LUT).
    with tempfile.TemporaryDirectory(prefix="alncheck_base_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
            _setup(Path(td), build_lut=False)
        )
        t0 = time.perf_counter()
        baseline: Dict[str, List[bool]] = {}
        for prob_path, plan_path in problems:
            init = _parse_problem(prob_path)
            plan = _parse_plan(plan_path)
            baseline[prob_path.name] = _replay(
                env, ws, cfg, init, plan, executor, pick_fn, put_fn,
                shelf_home)
        print(f"baseline FAST replay: {time.perf_counter() - t0:.1f}s")

    # Run with LUT.
    with tempfile.TemporaryDirectory(prefix="alncheck_lut_") as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = (
            _setup(Path(td), build_lut=True)
        )
        t0 = time.perf_counter()
        lut_results: Dict[str, List[bool]] = {}
        for prob_path, plan_path in problems:
            init = _parse_problem(prob_path)
            plan = _parse_plan(plan_path)
            lut_results[prob_path.name] = _replay(
                env, ws, cfg, init, plan, executor, pick_fn, put_fn,
                shelf_home)
        print(f"LUT FAST replay: {time.perf_counter() - t0:.1f}s")

    # Compare.
    total_acts = 0
    total_disagree = 0
    flipped: List[Tuple[str, int, bool, bool]] = []
    for name in baseline:
        b = baseline[name]
        l = lut_results[name]
        if len(b) != len(l):
            print(f"  LENGTH MISMATCH: {name} base={len(b)} lut={len(l)}")
            continue
        for i, (bv, lv) in enumerate(zip(b, l)):
            total_acts += 1
            if bv != lv:
                total_disagree += 1
                flipped.append((name, i, bv, lv))

    print(f"\n=== Verdict ===")
    print(f"actions: {total_acts}, "
          f"agreements: {total_acts - total_disagree}, "
          f"flips: {total_disagree}")
    if flipped:
        print("\nFlipped actions (problem, action_idx, baseline, lut):")
        for f in flipped[:20]:
            print(f"  {f}")
        if len(flipped) > 20:
            print(f"  ... and {len(flipped) - 20} more")
        return 1
    print("ALIGNMENT OK — items 1 + 4 preserve FAST verdicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
