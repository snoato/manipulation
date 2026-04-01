"""Parameter sweep: IK iterations/threshold vs RRT* iterations for feasibility checking.

Measures speed and reliability (false-negative rate vs a high-quality ground truth)
across a grid of parameter combinations.

Two state types are tested:
  Easy   — 1 isolated cylinder (should always be feasible).
            Any 'infeasible' result here is a false negative.
  Dense  — 5 cylinders (mix of feasible/infeasible).
            Compared against ground truth to count false negatives.

Usage:
    python examples/benchmark_feasibility_params.py
    python examples/benchmark_feasibility_params.py --states 15 --rng-seed 7
"""

import argparse
import itertools
import time
from pathlib import Path

import mujoco
import numpy as np

from tampanda import FrankaEnvironment, RRTStar, SCENE_SYMBOLIC
from tampanda.planners.grasp_planner import GraspPlanner
from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker

_XML = SCENE_SYMBOLIC

TABLE_Z: float = 0.27

# ---------------------------------------------------------------------------
# Parameter grid to sweep
# ---------------------------------------------------------------------------

IK_MAX_ITERS    = [100, 200, 400]       # MinkIK.max_iters
IK_POS_THRESH   = [0.002, 0.005, 0.010] # MinkIK.pos_threshold (metres)
RRT_MAX_ITERS   = [200, 500, 1000]      # ActionFeasibilityChecker.max_iterations

# Ground truth: high-quality settings used to label each state feasible/infeasible
GT_IK_MAX_ITERS   = 1000
GT_IK_POS_THRESH  = 0.002
GT_RRT_MAX_ITERS  = 2000


# ---------------------------------------------------------------------------
# Build shared environment
# ---------------------------------------------------------------------------

def build_env():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    planner = RRTStar(env)
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )

    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=TABLE_Z)

    # Checker owns the fast-step patch — build it once
    checker = ActionFeasibilityChecker(
        env, planner, state_manager, grasp_planner,
        max_iterations=GT_RRT_MAX_ITERS,
    )

    return env, planner, checker, state_manager


# ---------------------------------------------------------------------------
# Generate fixed test states
# ---------------------------------------------------------------------------

def generate_states(state_manager: StateManager,
                    n: int, n_cylinders: int,
                    rng: np.random.Generator) -> list[dict]:
    states = []
    while len(states) < n:
        seed = int(rng.integers(0, 2**31))
        state_manager.sample_random_state(n_cylinders=n_cylinders, seed=seed)
        state = state_manager.ground_state()
        cyls  = list(state["cylinders"].keys())
        if cyls:
            target = str(np.random.choice(cyls))
            states.append({"state": state, "target": target})
    return states


# ---------------------------------------------------------------------------
# Run one combo on a list of states
# ---------------------------------------------------------------------------

def run_combo(checker: ActionFeasibilityChecker,
              env: FrankaEnvironment,
              ik_max_iters: int,
              ik_pos_thresh: float,
              rrt_max_iters: int,
              states: list[dict]) -> list[dict]:
    env.ik.max_iters      = ik_max_iters
    env.ik.pos_threshold  = ik_pos_thresh
    checker.max_iterations = rrt_max_iters

    results = []
    for s in states:
        t0 = time.perf_counter()
        feasible, timing = checker.check("pick", s["state"], cylinder_name=s["target"])
        wall_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "feasible": feasible,
            "wall_ms":  wall_ms,
            "reason":   timing.get("reason", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(n_easy: int = 10, n_dense: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)

    print("Building environment...")
    env, planner, checker, state_manager = build_env()

    print(f"Generating {n_easy} easy + {n_dense} dense states...")
    easy_states  = generate_states(state_manager, n_easy,  n_cylinders=1, rng=rng)
    dense_states = generate_states(state_manager, n_dense, n_cylinders=5, rng=rng)

    # --- Ground truth -------------------------------------------------------
    print(f"\nRunning ground truth "
          f"(ik_iters={GT_IK_MAX_ITERS}, pos_thresh={GT_IK_POS_THRESH}, "
          f"rrt_iters={GT_RRT_MAX_ITERS})...")

    gt_easy  = run_combo(checker, env, GT_IK_MAX_ITERS, GT_IK_POS_THRESH,
                         GT_RRT_MAX_ITERS, easy_states)
    gt_dense = run_combo(checker, env, GT_IK_MAX_ITERS, GT_IK_POS_THRESH,
                         GT_RRT_MAX_ITERS, dense_states)

    gt_easy_ok  = [r["feasible"] for r in gt_easy]
    gt_dense_ok = [r["feasible"] for r in gt_dense]

    print(f"  Ground truth: easy {sum(gt_easy_ok)}/{n_easy} feasible, "
          f"dense {sum(gt_dense_ok)}/{n_dense} feasible")

    # --- Parameter sweep ----------------------------------------------------
    combos = list(itertools.product(IK_MAX_ITERS, IK_POS_THRESH, RRT_MAX_ITERS))
    total  = len(combos)

    print(f"\nSweeping {total} parameter combinations "
          f"({n_easy + n_dense} states each)...\n")

    rows = []
    for idx, (ik_i, ik_t, rrt_i) in enumerate(combos, 1):
        print(f"  [{idx:2d}/{total}] ik_iters={ik_i:4d}  "
              f"pos_thresh={ik_t:.3f}  rrt_iters={rrt_i:5d}", end="  ", flush=True)

        easy_res  = run_combo(checker, env, ik_i, ik_t, rrt_i, easy_states)
        dense_res = run_combo(checker, env, ik_i, ik_t, rrt_i, dense_states)

        # False negatives: gt says feasible but combo says infeasible
        fn_easy  = sum(1 for gt, r in zip(gt_easy_ok,  easy_res)
                       if gt and not r["feasible"])
        fn_dense = sum(1 for gt, r in zip(gt_dense_ok, dense_res)
                       if gt and not r["feasible"])

        all_times = [r["wall_ms"] for r in easy_res + dense_res]
        mean_ms   = np.mean(all_times)
        med_ms    = np.median(all_times)
        max_ms    = np.max(all_times)

        rows.append({
            "ik_iters":   ik_i,
            "pos_thresh": ik_t,
            "rrt_iters":  rrt_i,
            "fn_easy":    fn_easy,
            "fn_dense":   fn_dense,
            "fn_total":   fn_easy + fn_dense,
            "mean_ms":    mean_ms,
            "med_ms":     med_ms,
            "max_ms":     max_ms,
        })

        print(f"fn={fn_easy+fn_dense:2d}/{n_easy+n_dense}  "
              f"mean={mean_ms:6.0f}ms  max={max_ms:6.0f}ms")

    _print_report(rows, n_easy, n_dense)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(rows: list[dict], n_easy: int, n_dense: int):
    n_total = n_easy + n_dense
    print("\n" + "=" * 80)
    print("  PARAMETER SWEEP RESULTS  (sorted by false-negatives then speed)")
    print("=" * 80)
    print(f"  {'ik_iters':>8}  {'pos_thr':>7}  {'rrt_iters':>9}  "
          f"{'fn':>4}  {'fn_easy':>7}  {'fn_dense':>8}  "
          f"{'mean_ms':>7}  {'med_ms':>6}  {'max_ms':>6}")
    print("  " + "-" * 76)

    for r in sorted(rows, key=lambda x: (x["fn_total"], x["mean_ms"])):
        fn_pct = 100 * r["fn_total"] / max(n_total, 1)
        print(f"  {r['ik_iters']:>8d}  {r['pos_thresh']:>7.3f}  {r['rrt_iters']:>9d}  "
              f"{r['fn_total']:>3d}  {r['fn_easy']:>7d}  {r['fn_dense']:>8d}  "
              f"{r['mean_ms']:>7.0f}  {r['med_ms']:>6.0f}  {r['max_ms']:>6.0f}"
              f"  {'← fn=0' if r['fn_total'] == 0 else ''}")

    # Best zero-false-negative combo
    zero_fn = [r for r in rows if r["fn_total"] == 0]
    if zero_fn:
        best = min(zero_fn, key=lambda x: x["mean_ms"])
        print(f"\n  Fastest zero-false-negative combo:")
        print(f"    ik_iters={best['ik_iters']}  pos_thresh={best['pos_thresh']:.3f}  "
              f"rrt_iters={best['rrt_iters']}  → mean {best['mean_ms']:.0f}ms")
    else:
        print("\n  No zero-false-negative combo found — consider increasing ground-truth "
              "iterations or adding more states.")

    print("=" * 80)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--easy-states",  type=int, default=10)
    ap.add_argument("--dense-states", type=int, default=10)
    ap.add_argument("--rng-seed",     type=int, default=42)
    args = ap.parse_args()
    sweep(n_easy=args.easy_states, n_dense=args.dense_states, seed=args.rng_seed)
