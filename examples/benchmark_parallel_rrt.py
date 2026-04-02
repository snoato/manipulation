"""Benchmark: sequential vs parallel RRT planners.

Compares each parallel planner against its sequential baseline on the same
set of random start→goal queries in the tabletop scene.

Usage
-----
    python examples/benchmark_parallel_rrt.py

Adjust the constants below to change difficulty / worker count.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tampanda.environments.franka_env import FrankaEnvironment
from tampanda.planners import (
    RRTStar,
    FeasibilityRRT,
    CollisionWorkerPool,
    ParallelEdgeRRTStar,
    SpeculativeFeasibilityRRT,
)
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder
from tampanda.symbolic import GridDomain, StateManager

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_QUERIES       = 25
N_WORKERS       = 4
BATCH_SIZE      = 4
MAX_ITER        = 2000
STEP_SIZE       = 0.1
GOAL_THRESHOLD  = 0.05
COLLISION_STEPS = 5
N_CYLINDERS     = 8
RANDOM_SEED     = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_scene() -> tuple[str, FrankaEnvironment]:
    builder = make_symbolic_builder()
    xml = builder.build_xml()
    fd, xml_path = tempfile.mkstemp(suffix=".xml", dir=str(builder._base.parent))
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return xml_path, FrankaEnvironment(xml_path)


def setup_scene(env: FrankaEnvironment, rng: np.random.Generator) -> None:
    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.4),
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
    )
    StateManager(grid, env).sample_random_state(n_cylinders=N_CYLINDERS)


def sample_configs(env: FrankaEnvironment, n: int, rng: np.random.Generator):
    lo, hi = env.model.jnt_range[:7, 0], env.model.jnt_range[:7, 1]
    configs, attempts = [], 0
    while len(configs) < n:
        q = rng.uniform(lo, hi)
        if env.is_collision_free(q):
            configs.append(q)
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("Too few collision-free configs — scene too cluttered?")
    return configs


def run_planner(label, plan_fn, queries):
    times, successes = [], 0
    for start, goal in queries:
        t0 = time.perf_counter()
        path = plan_fn(start, goal)
        times.append(time.perf_counter() - t0)
        if path is not None:
            successes += 1
    arr = np.array(times)
    print(f"\n  {label}")
    print(f"    Success  {successes:2d}/{len(queries)}")
    print(f"    Mean {arr.mean():.3f}s  Median {np.median(arr):.3f}s  Std {arr.std():.3f}s")
    print(f"    Total {arr.sum():.2f}s")
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Parallel RRT Benchmark — tabletop scene")
    print("=" * 60)

    print("\n[1/4] Building scene …")
    xml_path, env = build_scene()
    try:
        setup_scene(env, rng)
        print(f"      {N_CYLINDERS} cylinders placed on table")

        print(f"\n[2/4] Sampling {N_QUERIES * 2} collision-free configs …")
        configs = sample_configs(env, N_QUERIES * 2, rng)
        queries = [(configs[i], configs[i + N_QUERIES]) for i in range(N_QUERIES)]
        print(f"      {N_QUERIES} start→goal pairs ready")

        print(f"\n[3/4] Starting worker pool ({N_WORKERS} workers) …")
        t0 = time.perf_counter()
        pool = CollisionWorkerPool(xml_path, n_workers=N_WORKERS,
                                   collision_check_steps=COLLISION_STEPS)
        pool.set_scene(env)
        pool.check_edges_parallel([(configs[0], configs[1])])  # warm up
        print(f"      Pool ready in {time.perf_counter() - t0:.2f}s")

        rrt_kwargs = dict(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                          search_radius=0.5, goal_threshold=GOAL_THRESHOLD)
        feas_kwargs = dict(max_iterations=MAX_ITER, step_size=STEP_SIZE,
                           goal_threshold=GOAL_THRESHOLD,
                           collision_check_steps=COLLISION_STEPS)

        rrt_base  = RRTStar(env, **rrt_kwargs)
        rrt_base.collision_check_steps = COLLISION_STEPS
        rrt_par   = ParallelEdgeRRTStar(env, pool, **rrt_kwargs)
        rrt_par.collision_check_steps = COLLISION_STEPS

        feas_base = FeasibilityRRT(env, **feas_kwargs)
        feas_par  = SpeculativeFeasibilityRRT(env, pool, batch_size=BATCH_SIZE,
                                              **feas_kwargs)

        print(f"\n[4/4] Running {N_QUERIES} queries per planner …")
        t1 = run_planner("RRTStar (baseline)",         lambda s, g: rrt_base.plan(s, g),  queries)
        t2 = run_planner("ParallelEdgeRRTStar",        lambda s, g: rrt_par.plan(s, g),   queries)
        t3 = run_planner("FeasibilityRRT (baseline)",  lambda s, g: feas_base.plan(s, g), queries)
        t4 = run_planner("SpeculativeFeasibilityRRT",  lambda s, g: feas_par.plan(s, g),  queries)

        print("\n" + "=" * 60)
        print("Speedup (total time)")
        print(f"  ParallelEdgeRRTStar       : {t1.sum() / t2.sum():.2f}x")
        print(f"  SpeculativeFeasibilityRRT : {t3.sum() / t4.sum():.2f}x")
        print("=" * 60)

        pool.shutdown()

    finally:
        os.unlink(xml_path)


if __name__ == "__main__":
    main()
