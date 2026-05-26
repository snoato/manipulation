"""Dataset generator for the tabletop_access:access task.

Each instance: an OoI somewhere on the shelf with blockers occluding it
(and optionally one sitting at its goal), solved by the feasibility-guided
planner.  Every action is FAST-feasible by construction; a fraction get a
FULL-physics spot-check.  Writes one PDDL problem + plan per instance and
bundles the (adjacency-carrying) domain.pddl.

Splits (blockers-only count; OoI is extra):

  train     occluders 1-3, clutter 0-2, rare goal-clutter, ~30% random
  val       same distribution (in-distribution)
  eval_ood  occluders 1-3, clutter 3-7 (-> up to ~10 total), goal-clutter

Run:
    python -m tampanda.symbolic.domains.tabletop_access.generate_data \
        --output-dir data/access [--train 300 --val 50 --eval 50]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tampanda.planners.grasp_planner import GraspType
from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.symbolic.workspace import Cell
from tampanda.symbolic.domains.tabletop_access.env_builder import (
    make_access_builder, apply_runtime_tweaks,
)
from tampanda.symbolic.domains.tabletop_access import reachability as R
from tampanda.symbolic.domains.tabletop_access.chains import make_access_chains
from tampanda.symbolic.domains.tabletop_access import templates as T
from tampanda.symbolic.domains.tabletop_access.planner import solve
from tampanda.symbolic.domains.tabletop_access.feasibility import (
    check_action, check_action_sequence,
)
from tampanda.symbolic.domains.tabletop_access.state import restore_state

_N_BLOCKERS = 10
_BLOCKER_NAMES = [f"blocker_{i}" for i in range(_N_BLOCKERS)]
_OBJECT_IDS = ["ooi"] + _BLOCKER_NAMES
_DOMAIN_PDDL = Path(__file__).resolve().parent / "pddl" / "domain.pddl"
_FULL_CHECK_FRAC = 0.10
_RANDOM_FRAC = 0.30


# --------------------------------------------------------------------------
# Setup + PDDL writers
# --------------------------------------------------------------------------

def _setup(scratch_dir: Path):
    builder, ws, cfg = make_access_builder(scratch_dir=scratch_dir,
                                           n_uniform_blockers=_N_BLOCKERS)
    env = builder.build_env(rate=10000.0)
    apply_runtime_tweaks(env, cfg)
    ex = R._build_executor(env, table_z=ws["floor_left"].level_z,
                           allowed_types=[GraspType.FRONT, GraspType.TOP_DOWN_X,
                                          GraspType.TOP_DOWN_Y])
    home = R._solve_access_staging(env, ws, cfg, _OBJECT_IDS)
    lik = LinearIKPlanner(env, n_substeps=12, joint_check_steps=8)
    pick_fn, put_fn = make_access_chains(env, ex, ws, lik=lik)

    def feasible(layout, held, action):
        state = {("occupied", c, o): True for o, c in layout.items()}
        if held is not None:
            state[("holding", held)] = True
        restore_state(env, ws, state, _OBJECT_IDS, executor=ex,
                      home_qpos=home, on_held="attach")
        return check_action(env, ws, ex, pick_fn, put_fn, action, fast=True)

    return env, ws, cfg, ex, pick_fn, put_fn, home, feasible


def _cells_adjacency(ws) -> Tuple[List[str], List[str]]:
    """All reachable cells + within-grid (adjacent dir c1 c2) edges
    (north=+iy depth, east=+ix column); no cross-grid adjacency."""
    cells: List[str] = []
    adj: List[str] = []
    for rname in ws.regions:
        region = ws[rname]
        valid = {(c.ix, c.iy) for c in region.cells()}
        for ix, iy in sorted(valid):
            here = f"{rname}__{ix}_{iy}"
            cells.append(here)
            if (ix, iy + 1) in valid:
                adj.append(f"(adjacent north {here} {rname}__{ix}_{iy + 1})")
            if (ix + 1, iy) in valid:
                adj.append(f"(adjacent east {here} {rname}__{ix + 1}_{iy})")
    return cells, adj


def write_pddl_problem(path: Path, name: str, source_layout: Dict[str, str],
                       goal: Dict[str, str], cells: List[str],
                       adjacency: List[str]) -> None:
    movables = sorted(source_layout.keys())
    occupied = set(source_layout.values())
    init = list(adjacency)
    for obj, cid in source_layout.items():
        init.append(f"(occupied {cid} {obj})")
    init += [f"(empty {c})" for c in cells if c not in occupied]
    init.append("(gripper-empty)")

    lines = [f"(define (problem {name})", "  (:domain tabletop-access)",
             "  (:objects",
             f"    {' '.join(movables)} - movable",
             f"    {' '.join(cells)} - cell", "  )", "  (:init"]
    lines += [f"    {p}" for p in init]
    lines += ["  )", "  (:goal (and"]
    lines += [f"    (occupied {cid} {obj})" for obj, cid in goal.items()]
    lines += ["  ))", ")"]
    path.write_text("\n".join(lines) + "\n")


def write_plan_file(path: Path, plan: List[Tuple], meta: Dict) -> None:
    lines = ["(" + " ".join(map(str, a)) + ")" for a in plan]
    lines.append(f"; cost = {len(plan)} (unit cost)")
    lines += [f"; {k}: {v}" for k, v in meta.items()]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Instance sampling
# --------------------------------------------------------------------------

def _sample_instance(ws, split: str, rng: np.random.Generator):
    """Return ``(source_layout, goal, meta)`` per the split distribution."""
    is_random = rng.random() < _RANDOM_FRAC
    n_occ = int(rng.integers(1, 4))                       # 1..3
    if split == "eval_ood":
        n_clutter_total = int(rng.integers(3, 8))         # 3..7
        goal_clutter = rng.random() < 0.5
    else:
        n_clutter_total = int(rng.integers(0, 3))         # 0..2
        goal_clutter = rng.random() < 0.10
    goal_clutter = bool(goal_clutter and n_clutter_total >= 1)

    if is_random:
        n_blk = n_occ + n_clutter_total
        layout, goal = T.random_layout(ws, n_blk, rng, blocker_names=_BLOCKER_NAMES)
        meta = {"mode": "random", "n_blockers": n_blk}
        return layout, goal, meta

    n_clutter = n_clutter_total - (1 if goal_clutter else 0)
    t = T.sample_by_counts(ws, rng, n_occ=n_occ, n_clutter=n_clutter,
                           blocker_names=_BLOCKER_NAMES, goal_clutter=goal_clutter)
    meta = {"mode": "structured", "n_occ": n_occ, "n_clutter": n_clutter_total,
            "goal_clutter": goal_clutter, "ooi_cell": t.metadata["ooi_cell"]}
    return T.source_layout(t), T.goal_layout(t), meta


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_split(split: str, count: int, outdir: Path, ctx, rng):
    env, ws, cfg, ex, pick_fn, put_fn, home, feasible = ctx
    cells, adjacency = _cells_adjacency(ws)
    outdir.mkdir(parents=True, exist_ok=True)
    made = attempts = full_checked = 0
    t0 = time.time()
    while made < count:
        attempts += 1
        if attempts > count * 30 + 200:
            raise RuntimeError(f"{split}: too many resamples ({attempts})")
        try:
            layout, goal, meta = _sample_instance(ws, split, rng)
        except (ValueError, RuntimeError):
            continue
        plan = solve(ws, layout, goal, feasible)
        if not plan:
            continue
        if rng.random() < _FULL_CHECK_FRAC:                # FULL spot-check
            ok, _ = check_action_sequence(env, ws, ex, pick_fn, put_fn, layout,
                                          plan, _OBJECT_IDS, fast=False,
                                          home_qpos=home)
            full_checked += 1
            if not ok:
                continue
        name = f"{split}_{made:04d}"
        meta.update({"split": split, "plan_len": len(plan),
                     "k_blockers": len(layout) - 1})
        write_pddl_problem(outdir / f"{name}.pddl", name, layout, goal,
                           cells, adjacency)
        write_plan_file(outdir / f"{name}.pddl.plan", plan, meta)
        made += 1
        if made % 25 == 0:
            print(f"  {split}: {made}/{count}  (attempts={attempts}, "
                  f"full-checked={full_checked}, {time.time()-t0:.0f}s)")
    print(f"{split}: {made} instances, {attempts} attempts, "
          f"{full_checked} FULL-checked, {time.time()-t0:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("data/access"))
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tempfile
    ctx = _setup(Path(tempfile.mkdtemp(prefix="access_gen_")))
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_DOMAIN_PDDL, args.output_dir / "domain.pddl")
    for split, n in (("train", args.train), ("val", args.val),
                     ("eval_ood", args.eval)):
        if n > 0:
            generate_split(split, n, args.output_dir / split, ctx, rng)
    print("DONE:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
