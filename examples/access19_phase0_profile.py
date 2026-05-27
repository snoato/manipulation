"""Phase-0 instrumentation run for access-19 feasibility checks.

Mirrors rgnet's call pattern: one process, one env, serial
``check_action`` calls with ``fast=True``.  Per-call IK wall times are
recorded by ``access19.ik_profile`` (LinearIKPlanner monkey-patch);
emit a summary to stdout and dump raw timings to JSON for later
analysis.

Run from repo root::

    python examples/access19_phase0_profile.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tampanda.symbolic.domains.access19 import (
    apply_runtime_tweaks, make_access19_builder,
    make_access19_pick_fn, make_access19_put_fn,
)
from tampanda.symbolic.domains.access19.feasibility import check_action
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.reachability import (
    _build_executor, _solve_access19_staging,
)
from tampanda.symbolic.domains.access19 import ik_profile
from tampanda.symbolic.domains.access19.ik_seed_lut import (
    Access19IKSeedLUT,
)
from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner


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


def _apply_action(layout: Dict[str, str], held: str,
                  action: Tuple) -> Tuple[Dict[str, str], str]:
    """Symbolic step: update layout/held to mirror ``action``."""
    name = action[0]
    if name == "pick":
        obj = action[1]
        layout = {k: v for k, v in layout.items() if k != obj}
        return layout, obj
    if name == "put":
        obj, cell = action[1], action[2]
        layout = dict(layout)
        layout[obj] = cell
        return layout, ""
    raise ValueError(f"unknown action {action!r}")


def _classify(action: Tuple) -> str:
    cell = action[2]
    if cell.startswith("shelf_interior"):
        loc = "interior"
    elif cell.startswith("shelf_top"):
        loc = "deck"
    else:
        loc = "?"
    return f"{action[0]}_{loc}"


def _setup(scratch: Path, use_lut: bool = False):
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
    if use_lut:
        # Park all 19 physical bodies far away so the empty-env
        # precompute sees a clean shelf.
        park = np.array([2.0, 2.0, 2.0])
        for name in _OBJECT_NAMES:
            try:
                env.set_object_pose(name, park)
            except Exception:
                pass
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        seed_lut = Access19IKSeedLUT()
        stats = seed_lut.precompute(env, ws, cfg, lik, shelf_home,
                                            cube_half_z=cube_half)
        print(f"[LUT] precompute: {stats}")

    pick_fn = make_access19_pick_fn(env, executor, ws, cfg,
                                            cube_half_z=cube_half, lik=lik,
                                            home_qpos=shelf_home,
                                            seed_lut=seed_lut)
    put_fn = make_access19_put_fn(env, executor, ws, cfg,
                                          cube_half_z=cube_half, lik=lik,
                                          home_qpos=shelf_home,
                                          seed_lut=seed_lut)
    return env, ws, cfg, executor, pick_fn, put_fn, shelf_home


def _sample_problems(base: Path, n_per_split: int, rng: random.Random):
    """Pick a small batch covering different curriculum levels.

    Returns list of (problem_path, plan_path) tuples.  Mix:
      * eval_pre_b: in-distribution-like (≤12 blockers)
      * eval_full:  canonical_18 + return-all (longest plans, lots of
                    interior + deck picks/puts)
    """
    picks: List[Tuple[Path, Path]] = []
    for split in ("eval_pre_b", "eval_full"):
        d = base / split
        problems = sorted(d.glob("config_*.pddl"))
        problems = [p for p in problems if p.suffix == ".pddl"]
        for prob in rng.sample(problems, min(n_per_split, len(problems))):
            plan = prob.with_suffix(".pddl.plan")
            if plan.exists():
                picks.append((prob, plan))
    return picks


def main() -> int:
    rng = random.Random(0)
    base = Path("data/access19_v3")
    out_dir = Path("/tmp/access19_phase0")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _sample_problems(base, n_per_split=5, rng=rng)
    if not samples:
        print(f"FAIL: no problems found under {base}", file=sys.stderr)
        return 2

    print(f"=== access19 Phase 0 profile ({len(samples)} problems) ===\n")

    with tempfile.TemporaryDirectory(prefix="a19phase0_") as td:
        use_lut = "--lut" in sys.argv
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = _setup(
            Path(td), use_lut=use_lut)

        # Warm-up: dispatch one easy action so any first-call init costs
        # don't pollute the IK timings.
        warm_layout = {"ooi": "shelf_interior__3_0"}
        warm_state = _layout_to_state(warm_layout, held=None)
        check_action(
            env, ws, cfg, warm_state,
            ("pick", "ooi", "shelf_interior__3_0"),
            _OBJECT_NAMES, pick_fn, put_fn,
            executor=executor, fast=True, home_qpos=shelf_home,
        )

        # Reset state recorders.
        ik_profile.clear()
        ik_profile.enable()

        # Per-action wall + classification, in parallel with the IK
        # planner-level timings recorded by ik_profile.
        action_records: List[Dict] = []
        for prob_path, plan_path in samples:
            layout = _parse_problem(prob_path)
            plan = _parse_plan(plan_path)
            held = ""
            for action in plan:
                state = _layout_to_state(
                    layout, held=(held if held else None))
                t0 = time.perf_counter()
                res = check_action(
                    env, ws, cfg, state, action,
                    _OBJECT_NAMES, pick_fn, put_fn,
                    executor=executor, fast=True, home_qpos=shelf_home,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                action_records.append({
                    "problem": prob_path.name,
                    "action": list(action),
                    "kind": _classify(action),
                    "ok": bool(res["success"]),
                    "wall_ms": dt_ms,
                    "prefiltered": res.get("prefiltered", False),
                })
                # Symbolic step regardless of success — we want to
                # cover the next action's intended pre-state, not
                # whatever state the failed chain may have left.
                layout, held = _apply_action(layout, held, action)

        # ----- Synthetic infeasible-candidate sweep -----
        # eval/rgnet calls check_action on actions the GNN proposes,
        # most of which are infeasible.  When mink thrashes to iter
        # cap on every substep without converging, per-call wall blows
        # up.  Mirror that workload: for one mid-plan state per
        # problem, enumerate candidate (pick/put, ooi, deck_cell)
        # actions and check each.
        deck_cells = [f"shelf_top__{ix}_{iy}"
                       for ix in range(7) for iy in range(7)]
        rng2 = random.Random(0)
        synth_records: List[Dict] = []
        for prob_path, plan_path in samples[:3]:  # 3 problems × 49 cells × 2
            layout = _parse_problem(prob_path)
            plan = _parse_plan(plan_path)
            # Apply the first ~5 plan actions to get a non-trivial state.
            held = ""
            for action in plan[:5]:
                layout, held = _apply_action(layout, held, action)
            state = _layout_to_state(layout, held=(held if held else None))
            # Skip if mid-state has gripper holding — for cleaner pick
            # candidate semantics.
            if held:
                continue
            for cell in deck_cells:
                for verb in ("pick", "put"):
                    action = (verb, "ooi", cell)
                    t0 = time.perf_counter()
                    try:
                        res = check_action(
                            env, ws, cfg, state, action,
                            _OBJECT_NAMES, pick_fn, put_fn,
                            executor=executor, fast=True,
                            home_qpos=shelf_home,
                        )
                        ok = bool(res["success"])
                        prefilt = res.get("prefiltered", False)
                    except Exception:
                        ok = False
                        prefilt = False
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    synth_records.append({
                        "kind": f"synth_{verb}_deck",
                        "ok": ok,
                        "prefiltered": prefilt,
                        "wall_ms": dt_ms,
                    })

        ik_profile.disable()

        # Reports
        ik_summary = ik_profile.summary()
        timings = ik_profile.timings()

        # Per-kind action wall summary
        kind_buckets: Dict[str, List[float]] = defaultdict(list)
        for r in action_records:
            if r["prefiltered"]:
                continue
            kind_buckets[r["kind"]].append(r["wall_ms"])

        print("--- IK planner per-call (chains.py phase x line) ---")
        print(ik_summary)
        print("\n--- Per-action check_action wall (excl. prefilter hits) ---")
        print(f"{'kind':<18} {'count':>5} {'ok%':>5} {'mean':>7} {'p50':>7} "
              f"{'p95':>7} {'max':>8}")
        for kind in sorted(kind_buckets):
            arr = np.asarray(kind_buckets[kind])
            print(
                f"{kind:<18} {len(arr):>5} {' ':>5} {arr.mean():>6.1f} "
                f"{np.median(arr):>6.1f} "
                f"{np.percentile(arr, 95):>6.1f} {arr.max():>7.1f}"
            )

        # Synthetic-candidate buckets, split feasible vs infeasible.
        synth_kind: Dict[str, List[float]] = defaultdict(list)
        synth_ok: Dict[str, int] = defaultdict(int)
        for r in synth_records:
            if r["prefiltered"]:
                continue
            tag = r["kind"] + ("_OK" if r["ok"] else "_INFEAS")
            synth_kind[tag].append(r["wall_ms"])
            if r["ok"]:
                synth_ok[tag] += 1
        for tag in sorted(synth_kind):
            arr = np.asarray(synth_kind[tag])
            print(
                f"{tag:<18} {len(arr):>5} {' ':>5} {arr.mean():>6.1f} "
                f"{np.median(arr):>6.1f} "
                f"{np.percentile(arr, 95):>6.1f} {arr.max():>7.1f}"
            )

        # Dump raw timings + per-action records for downstream analysis.
        ik_profile.dump(out_dir / "ik_timings.json")
        (out_dir / "action_records.json").write_text(
            json.dumps(action_records, indent=2))
        print(f"\n=> raw → {out_dir}/ik_timings.json + action_records.json"
              f"  ({len(timings)} IK calls, "
              f"{len(action_records)} action calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
