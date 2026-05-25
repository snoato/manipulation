"""Sample problems per level and full-execute their plans.

For each problem:

  1. Build env once (reused across all problems).
  2. ``restore_state(initial)`` with on_held=attach.
  3. Run each plan action through ``bridge.execute_action``.
  4. Record per-step success, total elapsed, first-failure metadata.

Output: JSON list of records + per-level summary table + the
first-failure detail for every problem that didn't run to completion.

Purpose: catch dataset infeasibility under the FULL physics executor.
The reachability sweep already proved every cell picks fine *in
isolation*; this catches the "neighbor density" failure mode.

Usage::

  python examples/multilevel_blocks_plan_validator.py \\
      --samples-per-level 5 --max-actions 8 \\
      --output /tmp/plan_validation.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    MultilevelBlocksExecutor,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
    restore_state,
)
from tampanda.planners.rrt_star import RRTStar


_RX_INIT = re.compile(r"\(:init\s+(.*?)\)\s*\(:goal", re.DOTALL)
_RX_INPRED = re.compile(r"\(in\s+(\w+)\s+(\w+)\)")
_RX_HELD = re.compile(r"\((held-(?:cube|flat-x|flat-y|upright))\s+(\w+)\)")
_RX_PLAN = re.compile(r"\(([\w-]+)((?:\s+\w+)+)\)")


def parse_initial_state(pddl_path: Path) -> Dict[Tuple, bool]:
    text = pddl_path.read_text()
    m = _RX_INIT.search(text)
    if m is None:
        raise RuntimeError(f"no (:init ...) block in {pddl_path}")
    init = m.group(1)
    state: Dict[Tuple, bool] = {}
    for block, cell in _RX_INPRED.findall(init):
        if block.startswith(("cube_", "oblong_", "long_")):
            state[("in", block, cell)] = True
    for fluent, block in _RX_HELD.findall(init):
        state[(fluent, block)] = True
    return state


def parse_plan(plan_path: Path) -> List[Tuple]:
    actions: List[Tuple] = []
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = _RX_PLAN.match(line)
        if m is None:
            continue
        actions.append((m.group(1), *m.group(2).split()))
    return actions


def execute_plan(
    env, ws, cfg, executor, bridge,
    init_state: Dict[Tuple, bool],
    plan: List[Tuple],
    max_actions: Optional[int],
    quiet_executor: bool,
) -> Dict:
    """Run the plan end-to-end; return a record dict."""
    if max_actions is not None:
        plan = plan[:max_actions]

    t_total = time.perf_counter()
    restore_state(env, ws, cfg, init_state,
                       on_held="attach", executor=executor)

    per_step: List[Dict] = []
    first_failure: Optional[Dict] = None

    for step_idx, action in enumerate(plan, start=1):
        t_step = time.perf_counter()
        buf = StringIO()
        try:
            if quiet_executor:
                with redirect_stdout(buf), redirect_stderr(buf):
                    success, delta = bridge.execute_action(action[0], *action[1:])
            else:
                success, delta = bridge.execute_action(action[0], *action[1:])
            err = None
        except Exception as exc:
            success = False
            delta = {}
            err = f"{type(exc).__name__}: {exc}"
        dt_step = time.perf_counter() - t_step

        # Pull a useful error excerpt from the captured log (last ERROR line).
        if not success and err is None:
            lines = [ln for ln in buf.getvalue().splitlines()
                          if "ERROR" in ln or "fail" in ln.lower()]
            err = lines[-1].strip() if lines else "executor returned False"

        per_step.append({
            "step": step_idx,
            "action": list(action),
            "success": success,
            "elapsed_s": dt_step,
            "error": err if not success else None,
        })

        if not success:
            first_failure = per_step[-1]
            break

    return {
        "total_steps": len(plan),
        "steps_completed": sum(1 for s in per_step if s["success"]),
        "first_failure": first_failure,
        "per_step_elapsed": [s["elapsed_s"] for s in per_step],
        "total_elapsed_s": time.perf_counter() - t_total,
    }


def sample_problems(in_dir: Path, levels: List[int], samples_per_level: int,
                          seed: int) -> Dict[int, List[Path]]:
    rng = random.Random(seed)
    chosen: Dict[int, List[Path]] = {}
    for level in levels:
        level_dir = in_dir / "test" / f"L{level}"
        if not level_dir.exists():
            print(f"WARNING: {level_dir} does not exist; skipping L{level}",
                      file=sys.stderr)
            chosen[level] = []
            continue
        pddls = sorted(level_dir.glob("config_*.pddl"))
        if len(pddls) > samples_per_level:
            pddls = rng.sample(pddls, samples_per_level)
        chosen[level] = pddls
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path,
                            default=Path("data/multilevel_blocks"))
    parser.add_argument("--levels", type=int, nargs="+",
                            default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--samples-per-level", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=None,
                            help="cap per-plan; default = run full plan")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                            default=Path("plan_validation.json"))
    parser.add_argument("--quiet-executor", action="store_true", default=True)
    parser.add_argument("--verbose-executor", dest="quiet_executor",
                            action="store_false")
    args = parser.parse_args()

    sampled = sample_problems(args.in_dir, args.levels,
                                       args.samples_per_level, args.seed)
    total = sum(len(p) for p in sampled.values())
    print(f"Validating {total} problems across levels {args.levels}")
    for level, problems in sampled.items():
        print(f"  L{level}: {len(problems)} problems")
        for p in problems:
            print(f"    {p.name}")
    print()

    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    with tempfile.TemporaryDirectory(prefix="plan_val_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        rrt = RRTStar(env, max_iterations=3000)
        executor = MultilevelBlocksExecutor(env, ws, cfg, motion_planner=rrt)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg, executor=executor)

        results: List[Dict] = []
        t_run_start = time.perf_counter()
        n_done = 0
        for level, problems in sampled.items():
            for pddl_path in problems:
                plan_path = pddl_path.with_suffix(".pddl.plan")
                if not plan_path.exists():
                    plan_path = Path(str(pddl_path) + ".plan")
                if not plan_path.exists():
                    print(f"WARNING: no plan for {pddl_path.name}; skipping")
                    continue
                init_state = parse_initial_state(pddl_path)
                plan = parse_plan(plan_path)
                rec = execute_plan(env, ws, cfg, executor, bridge,
                                       init_state, plan, args.max_actions,
                                       args.quiet_executor)
                rec.update({
                    "level": level,
                    "problem": str(pddl_path.relative_to(args.in_dir)),
                })
                results.append(rec)
                n_done += 1
                eta_s = ((time.perf_counter() - t_run_start) / max(1, n_done)
                              ) * (total - n_done)
                status = ("OK" if rec["first_failure"] is None else
                              f"FAIL@{rec['first_failure']['step']}")
                print(f"  [{n_done:>3d}/{total}]  L{level} {pddl_path.name:>20s}  "
                          f"{rec['steps_completed']}/{rec['total_steps']} steps  "
                          f"{status:>7s}  "
                          f"{rec['total_elapsed_s']:5.1f}s  "
                          f"(eta {eta_s/60:.1f} min)")
                if rec["first_failure"] is not None:
                    ff = rec["first_failure"]
                    print(f"      first fail: step {ff['step']} {ff['action']}")
                    print(f"      err: {(ff['error'] or 'unknown')[:120]}")

        # ---- Summary ----
        print()
        print("==================== summary ====================")
        per_level_pass = {l: 0 for l in args.levels}
        per_level_total = {l: 0 for l in args.levels}
        for r in results:
            per_level_total[r["level"]] += 1
            if r["first_failure"] is None:
                per_level_pass[r["level"]] += 1
        print(f"{'level':>5s}  {'pass / total':>14s}  {'pass rate':>10s}")
        overall_pass = 0
        overall_total = 0
        for l in args.levels:
            t = per_level_total[l]
            p = per_level_pass[l]
            rate = (p / t * 100) if t else float("nan")
            print(f"  L{l:>2d}   {p:>5d} / {t:<5d}    {rate:>7.1f} %")
            overall_pass += p
            overall_total += t
        rate = (overall_pass / overall_total * 100) if overall_total else 0
        print(f"  OVERALL   {overall_pass} / {overall_total}    {rate:.1f} %")

        # Failing problems detail.
        failures = [r for r in results if r["first_failure"] is not None]
        if failures:
            print(f"\n{len(failures)} failing problems:")
            for r in failures:
                ff = r["first_failure"]
                print(f"  L{r['level']} {r['problem']}: "
                          f"step {ff['step']} ({ff['action'][0]} ...) — {(ff['error'] or '')[:120]}")
        else:
            print("\nAll sampled plans ran to completion.")

        args.output.write_text(json.dumps({
            "samples_per_level": args.samples_per_level,
            "max_actions": args.max_actions,
            "seed": args.seed,
            "results": results,
        }, indent=2))
        print(f"\nWrote {args.output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
