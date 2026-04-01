"""Benchmark the tabletop action feasibility checker.

Generates random symbolic states and runs pick feasibility checks on them,
reporting timing broken down by phase (IK, RRT*) and grouped by outcome
(feasible vs infeasible).

Two scenario types are tested:
  Sparse  — 1 cylinder, no obstacles.  Should always be feasible.
  Dense   — many cylinders.  Some checks may fail due to cluttered paths.

With --visualize, infeasible states are displayed in the viewer after the run
(one at a time, pausing for --display-secs seconds each).

Usage:
    python examples/benchmark_feasibility.py
    python examples/benchmark_feasibility.py --trials 10 --dense-cylinders 6
    python examples/benchmark_feasibility.py --visualize
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tampanda import FrankaEnvironment, RRTStar, SCENE_SYMBOLIC
from tampanda.planners.grasp_planner import GraspPlanner
from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker

_XML = SCENE_SYMBOLIC

TABLE_Z: float = 0.27   # matches scene_symbolic.xml table surface height


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_checker(max_iterations: int):
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    planner = RRTStar(env)
    planner.max_iterations   = max_iterations
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grid = GridDomain(
        model=env.model,
        cell_size=0.04,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        # grid_offset_x=0.05 shifts x from [0.20,0.60] → [0.25,0.65]
        # grid_offset_y=0.25 shifts y from [0.22,0.52] → [0.45,0.75]
        # → FRONT approach y_min ≈ 0.32 at x_min ≈ 0.27, clear of joint limits
        grid_offset_x=0.05,
        grid_offset_y=0.25,
    )

    state_manager = StateManager(grid, env)
    grasp_planner = GraspPlanner(table_z=TABLE_Z)

    checker = ActionFeasibilityChecker(
        env, planner, state_manager, grasp_planner,
        max_iterations=max_iterations,
    )

    return env, checker, state_manager


# ---------------------------------------------------------------------------
# Trial helpers
# ---------------------------------------------------------------------------

def pick_random_cylinder(state: dict) -> str | None:
    """Return a random active cylinder name from the state dict."""
    cyls = list(state["cylinders"].keys())
    return np.random.choice(cyls) if cyls else None


def run_trial(checker: ActionFeasibilityChecker,
              state_manager: StateManager,
              n_cylinders: int,
              rng: np.random.Generator) -> dict | None:
    """Sample a random state and run a pick feasibility check."""
    seed = int(rng.integers(0, 2**31))
    state_manager.sample_random_state(n_cylinders=n_cylinders, seed=seed)
    state = state_manager.ground_state()

    target = pick_random_cylinder(state)
    if target is None:
        return None   # no cylinders placed — skip

    feasible, timing = checker.check("pick", state, cylinder_name=target)
    return {
        "feasible":    feasible,
        "n_cylinders": n_cylinders,
        "target":      target,
        "state_dict":  state,   # kept for visualization replay
        **timing,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(n_trials: int = 10,
              sparse_n: int = 1,
              dense_n:  int = 5,
              max_iterations: int = 1000,
              seed: int = 42,
              visualize: bool = False,
              display_secs: float = 4.0):

    rng = np.random.default_rng(seed)

    print(f"Building environment... (max_iterations={max_iterations})")
    env, checker, state_manager = build_checker(max_iterations)
    print("Ready.\n")

    scenarios = [
        ("sparse", sparse_n),
        ("dense",  dense_n),
    ]

    all_results: dict[str, list] = {name: [] for name, _ in scenarios}

    for scenario, n_cyls in scenarios:
        print(f"[{scenario}: {n_cyls} cylinder(s) per trial, {n_trials} trials]")
        for i in range(n_trials):
            res = run_trial(checker, state_manager, n_cyls, rng)
            if res is None:
                print(f"  trial {i+1}: skipped (no cylinders placed)")
                continue
            tag = "FEASIBLE" if res["feasible"] else f"INFEASIBLE ({res['reason']})"
            print(f"  trial {i+1}: {tag}  total={res['total_ms']:.0f}ms", end="")
            if "rrt_approach_ms" in res:
                print(f"  rrt_approach={res['rrt_approach_ms']:.0f}ms", end="")
            if "rrt_grasp_ms" in res:
                print(f"  rrt_grasp={res['rrt_grasp_ms']:.0f}ms", end="")
            print()
            all_results[scenario].append(res)
        print()

    _print_report(all_results, n_trials)

    if visualize:
        _show_infeasible(env, state_manager, all_results, display_secs)

    return env, checker, state_manager, all_results


# ---------------------------------------------------------------------------
# Grid reachability visualisation
# ---------------------------------------------------------------------------

def visualize_reachability(reachability: dict, grid) -> None:
    """Plot the reachability grid as a coloured heatmap."""
    cells_x = grid.cells_x
    cells_y = grid.cells_y

    grid_data = np.zeros((cells_y, cells_x), dtype=float)  # rows=y, cols=x

    for cell_id, reachable in reachability.items():
        # cell_id format: "cell_IX_IY"
        _, ix_s, iy_s = cell_id.split("_")
        ix, iy = int(ix_s), int(iy_s)
        grid_data[iy, ix] = 1.0 if reachable else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        grid_data,
        origin="lower",
        extent=[
            grid.table_bounds["min_x"], grid.table_bounds["max_x"],
            grid.table_bounds["min_y"], grid.table_bounds["max_y"],
        ],
        aspect="equal",
        cmap="RdYlGn",
        vmin=0, vmax=1,
    )

    n_ok   = sum(reachability.values())
    n_tot  = len(reachability)
    ax.set_title(f"Grid reachability — {n_ok}/{n_tot} cells reachable (IK check)")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)  ← robot side")

    green_patch = mpatches.Patch(color="green", label="Reachable")
    red_patch   = mpatches.Patch(color="red",   label="Unreachable")
    ax.legend(handles=[green_patch, red_patch], loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualization of infeasible states
# ---------------------------------------------------------------------------

def _show_infeasible(env: FrankaEnvironment,
                     state_manager: StateManager,
                     all_results: dict,
                     display_secs: float):
    """Replay every infeasible state in the passive viewer."""
    infeasible_cases = [
        (scenario, r)
        for scenario, results in all_results.items()
        for r in results
        if not r["feasible"]
    ]

    if not infeasible_cases:
        print("\nNo infeasible cases to display.")
        return

    print(f"\nDisplaying {len(infeasible_cases)} infeasible state(s) "
          f"({display_secs:.0f}s each)...")

    viewer = env.launch_viewer()

    for idx, (scenario, res) in enumerate(infeasible_cases):
        print(f"  [{idx+1}/{len(infeasible_cases)}] "
              f"scenario={scenario}  target={res['target']}  "
              f"reason={res['reason']}")

        # Load the state back into sim
        state_manager.set_from_grounded_state(res["state_dict"])

        # Move the IK target cube to the intended grasp pose so it's visible
        if "grasp_pos" in res:
            env.data.mocap_pos[0]  = res["grasp_pos"]
            env.data.mocap_quat[0] = res["grasp_quat"]

        mujoco.mj_forward(env.model, env.data)
        viewer.sync()

        deadline = time.time() + display_secs
        while time.time() < deadline and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

        if not viewer.is_running():
            break

    viewer.close()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(all_results: dict, n_trials: int):
    print("=" * 65)
    print("  FEASIBILITY CHECKER BENCHMARK")
    print("=" * 65)

    for scenario, results in all_results.items():
        if not results:
            continue

        feasible   = [r for r in results if r["feasible"]]
        infeasible = [r for r in results if not r["feasible"]]

        def ms(key, subset):
            vals = [r[key] for r in subset if key in r]
            return (np.mean(vals), np.median(vals), np.max(vals)) if vals else (0, 0, 0)

        print(f"\n  Scenario: {scenario} ({results[0]['n_cylinders']} cylinder(s))")
        print(f"    Outcome:  {len(feasible)}/{len(results)} feasible")

        for label, subset in [("feasible", feasible), ("infeasible", infeasible)]:
            if not subset:
                continue
            mu, med, mx = ms("total_ms", subset)
            print(f"    [{label}] n={len(subset)}  total: "
                  f"mean {mu:.0f}ms  median {med:.0f}ms  max {mx:.0f}ms")

            for phase in ("ik_approach_ms", "rrt_approach_ms",
                          "ik_grasp_ms",    "rrt_grasp_ms"):
                mu2, med2, _ = ms(phase, subset)
                if mu2 > 0:
                    name = phase.replace("_ms", "").replace("_", " ")
                    print(f"           {name:<20}: mean {mu2:.0f}ms  median {med2:.0f}ms")

        reasons: dict[str, int] = {}
        for r in infeasible:
            reasons[r.get("reason", "unknown")] = reasons.get(r.get("reason", "unknown"), 0) + 1
        if reasons:
            print(f"    Failure reasons: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",           type=int,   default=10)
    ap.add_argument("--sparse-cylinders", type=int,   default=1)
    ap.add_argument("--dense-cylinders",  type=int,   default=5)
    ap.add_argument("--max-iterations",   type=int,   default=1000)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--visualize",        action="store_true",
                    help="Show infeasible states in viewer after benchmark")
    ap.add_argument("--display-secs",     type=float, default=4.0,
                    help="Seconds to display each infeasible state")
    ap.add_argument("--verify",           action="store_true",
                    help="Run IK reachability check on every grid cell and plot the result")
    args = ap.parse_args()

    env, checker, state_manager, _ = benchmark(
        n_trials=args.trials,
        sparse_n=args.sparse_cylinders,
        dense_n=args.dense_cylinders,
        max_iterations=args.max_iterations,
        seed=args.seed,
        visualize=args.visualize,
        display_secs=args.display_secs,
    )

    if args.verify:
        print("\nRunning grid reachability verification...")
        reachability = checker.verify_reachability(verbose=True)
        visualize_reachability(reachability, state_manager.grid)
