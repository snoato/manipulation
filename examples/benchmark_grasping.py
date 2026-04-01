"""Headless grasping benchmark on scene_blocks.xml.

Tests blocks 0-11 (small 4 cm and medium 6 cm cubes) with GraspPlanner.
Runs as fast as possible (rate limiter bypassed) so 12 blocks x 3 trials
should complete in well under a minute.

Usage:
    python examples/benchmark_grasping.py
    python examples/benchmark_grasping.py --trials 5 --visualize
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from tampanda import FrankaEnvironment, RRTStar, ControllerStatus
from tampanda.planners.grasp_planner import GraspPlanner, GraspType
from tampanda.symbolic.domains.blocks.env_builder import make_blocks_builder

TABLE_SURFACE_Z: float = 0.27

# Placement region verified to be on the table and in robot workspace.
PLACE_X = (-0.20, 0.30)
PLACE_Y = (0.25,  0.55)

GRASPABLE_BLOCKS: dict = {
    **{f"block_{i}": np.array([0.02, 0.02, 0.02]) for i in range(6)},
    **{f"block_{i}": np.array([0.03, 0.03, 0.03]) for i in range(6, 12)},
}

LIFT_THRESHOLD: float = 0.08   # block must rise ≥ 8 cm


# ---------------------------------------------------------------------------
# Fast headless step – bypasses RateLimiter so the sim runs at max speed
# ---------------------------------------------------------------------------

def _patch_fast_step(env: FrankaEnvironment) -> None:
    """Replace env.step with a version that skips rate.sleep()."""
    _dt = env.model.opt.timestep

    def _fast_step():
        mujoco.mj_step(env.model, env.data)
        env.sim_time += _dt
        if env.viewer is not None:
            env.viewer.sync()
        return _dt

    env.step = _fast_step


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def sim_steps(env: FrankaEnvironment, n: int) -> None:
    """Run n simulation steps (no controller stepping)."""
    for _ in range(n):
        env.step()



def wait_idle(env: FrankaEnvironment, max_steps: int = 4000) -> bool:
    """Step controller + sim until IDLE; then run settling steps."""
    ok = env.wait_idle(max_steps=max_steps, settle_steps=200)
    if not ok:
        print(f"    [warn] wait_idle timed out after {max_steps} steps"
              f" (status={env.controller.get_status()})", flush=True)
    return ok


def get_site_pos(env: FrankaEnvironment) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return env.data.site_xpos[sid].copy()


def reset_robot(env: FrankaEnvironment) -> None:
    """Teleport arm to home, open gripper, force controller IDLE."""
    env.data.qpos[:8] = env.initial_qpos[:8]
    env.data.ctrl[:8] = env.initial_ctrl[:8]   # ctrl[7]=255 → gripper open
    env.data.qvel[:]  = 0.0
    # IMPORTANT: do NOT call open_gripper() here – it sets status=GRASPING
    # and settle() doesn't step the controller, leaving it permanently busy.
    env.controller.stop()                        # force IDLE, clear trajectory
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    sim_steps(env, 30)


def place_block(
    env:        FrankaEnvironment,
    name:       str,
    half_size:  np.ndarray,
    rng:        np.random.Generator,
) -> np.ndarray:
    """Teleport block to a random on-table pose; return settled centre."""
    x = rng.uniform(*PLACE_X)
    y = rng.uniform(*PLACE_Y)
    z = TABLE_SURFACE_Z + half_size[2]
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    env.set_object_pose(name, np.array([x, y, z]), quat)
    mujoco.mj_forward(env.model, env.data)
    sim_steps(env, 40)
    return env.get_object_position(name).copy()


def execute(env: FrankaEnvironment, planner: RRTStar, path) -> bool:
    """Smooth, interpolate, and execute a path. Returns True if completed."""
    env.execute_path(path, planner)
    return wait_idle(env)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    env:           FrankaEnvironment,
    planner:       RRTStar,
    grasp_planner: GraspPlanner,
    block_name:    str,
    half_size:     np.ndarray,
    verbose:       bool = True,
) -> dict:
    res = {
        "block":          block_name,
        "block_type":     "small" if half_size[0] < 0.025 else "medium",
        "ik_attempts":    0,
        "ik_successes":   0,
        "plan_attempts":  0,
        "plan_successes": 0,
        "grasp_success":  False,
        "grasp_type":     None,
        "pos_error_mm":   [],
    }

    block_pos  = env.get_object_position(block_name)
    block_quat = env.get_object_orientation(block_name)
    rest_z     = block_pos[2]
    candidates = grasp_planner.generate_candidates(block_pos, half_size, block_quat)

    if not candidates:
        return res

    for cand in candidates:
        if verbose:
            print(f"  {cand.grasp_type.value} (score={cand.score:.0f})", end=" ... ", flush=True)

        # 1. Approach
        res["ik_attempts"]   += 1
        res["plan_attempts"] += 1
        path = planner.plan_to_pose(
            cand.approach_pos, cand.grasp_quat, dt=0.005, max_iterations=1500)
        if path is None:
            if verbose: print("approach FAIL")
            reset_robot(env)
            continue
        res["ik_successes"]   += 1
        res["plan_successes"] += 1
        execute(env, planner, path)

        # 2. Grasp contact (block is not an obstacle for this move)
        env.add_collision_exception(block_name)
        res["ik_attempts"]   += 1
        res["plan_attempts"] += 1
        path = planner.plan_to_pose(
            cand.grasp_pos, cand.grasp_quat, dt=0.005, max_iterations=800)
        if path is None:
            if verbose: print("grasp plan FAIL")
            env.clear_collision_exceptions()
            reset_robot(env)
            continue
        res["ik_successes"]   += 1
        res["plan_successes"] += 1
        execute(env, planner, path)

        # Measure EE accuracy
        ee_pos   = get_site_pos(env)
        err_mm   = float(np.linalg.norm(ee_pos - cand.grasp_pos) * 1000)
        res["pos_error_mm"].append(err_mm)

        # 3. Close gripper + settle
        env.controller.close_gripper()
        wait_idle(env, max_steps=400)
        sim_steps(env, 40)

        # 4. Lift
        path = planner.plan_to_pose(
            cand.lift_pos, cand.grasp_quat, dt=0.005, max_iterations=1500)
        if path is not None:
            execute(env, planner, path)
        elif verbose:
            print("    [warn] lift plan FAIL", flush=True)

        # 5. Success check
        block_z = env.get_object_position(block_name)[2]
        success = block_z > rest_z + LIFT_THRESHOLD
        env.clear_collision_exceptions()

        if verbose:
            tag = "SUCCESS ✓" if success else f"FAIL (z={block_z:.3f})"
            print(f"EE err={err_mm:.1f}mm  {tag}")

        res["grasp_success"] = success
        res["grasp_type"]    = cand.grasp_type.value

        reset_robot(env)
        if success:
            break

    return res


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(trials_per_block: int = 3, visualize: bool = False, seed: int = 42):
    rng = np.random.default_rng(seed)

    env = make_blocks_builder().build_env(rate=200.0)
    _patch_fast_step(env)          # bypass rate.sleep() for headless speed

    if visualize:
        env.launch_viewer()

    planner = RRTStar(env)
    planner.max_iterations   = 2000
    planner.step_size        = 0.15
    planner.goal_sample_rate = 0.15

    grasp_planner = GraspPlanner(
        approach_dist=0.12,
        lift_height=0.22,
        table_z=TABLE_SURFACE_Z,
    )

    all_results = []
    t_start = time.time()

    for block_name, half_size in GRASPABLE_BLOCKS.items():
        btype = "small" if half_size[0] < 0.025 else "medium"
        print(f"\n[{block_name} / {btype}]")

        for trial in range(trials_per_block):
            env.reset()
            reset_robot(env)
            block_pos = place_block(env, block_name, half_size, rng)

            if block_pos[2] < TABLE_SURFACE_Z - 0.01:
                print(f"  trial {trial+1}: block off table (z={block_pos[2]:.3f}), skip")
                continue

            print(f"  trial {trial+1}: placed at {np.round(block_pos, 3)}", end="  ")
            res          = run_trial(env, planner, grasp_planner, block_name, half_size)
            res["trial"] = trial
            all_results.append(res)

    elapsed = time.time() - t_start
    _print_report(all_results, elapsed, trials_per_block)

    if visualize and env.viewer is not None:
        env.viewer.close()

    return all_results


# ---------------------------------------------------------------------------

def _print_report(results: list, elapsed: float, tpb: int):
    print("\n\n" + "=" * 65)
    print("  GRASPING BENCHMARK REPORT")
    print("=" * 65)

    n   = len(results)
    ok  = sum(r["grasp_success"] for r in results)
    ia  = sum(r["ik_attempts"]    for r in results)
    is_ = sum(r["ik_successes"]   for r in results)
    pa  = sum(r["plan_attempts"]  for r in results)
    ps  = sum(r["plan_successes"] for r in results)
    err = [e for r in results for e in r["pos_error_mm"]]

    print(f"\n  Trials total   : {n}  ({tpb}/block)")
    print(f"  Wall time      : {elapsed:.1f} s")
    print(f"\n  IK  success    : {is_}/{ia}  = {100*is_/max(ia,1):.1f}%")
    print(f"  Plan success   : {ps}/{pa}  = {100*ps/max(pa,1):.1f}%")
    print(f"  Grasp success  : {ok}/{n}  = {100*ok/max(n,1):.1f}%")

    if err:
        print(f"\n  EE pos error at grasp contact (lower = more accurate IK):")
        print(f"    mean   {np.mean(err):.2f} mm   median {np.median(err):.2f} mm"
              f"   p95 {np.percentile(err,95):.2f} mm   max {np.max(err):.2f} mm")

    print(f"\n  By grasp type:")
    for gt in [t.value for t in GraspType]:
        tr = [r for r in results if r.get("grasp_type") == gt]
        if not tr: continue
        ts = sum(r["grasp_success"] for r in tr)
        te = [e for r in tr for e in r["pos_error_mm"]]
        ee = f"  EE {np.mean(te):.1f}mm" if te else ""
        print(f"    {gt:<15} {ts}/{len(tr)} ({100*ts/len(tr):.0f}%){ee}")

    print(f"\n  By block type:")
    for bt in ("small", "medium"):
        br = [r for r in results if r["block_type"] == bt]
        if not br: continue
        bs = sum(r["grasp_success"] for r in br)
        print(f"    {bt:<10} {bs}/{len(br)} = {100*bs/len(br):.0f}%")

    print("=" * 65)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",    type=int,  default=3)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--seed",      type=int,  default=42)
    args = ap.parse_args()
    benchmark(trials_per_block=args.trials, visualize=args.visualize, seed=args.seed)
