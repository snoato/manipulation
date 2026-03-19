"""Headless benchmark comparing direct IK vs RRT* grasping on cylinders.

Both approaches use GraspPlanner to compute the target poses and IK to solve
for the goal joint configuration.  The difference is in how the robot moves
between poses:

  Direct IK  — linearly interpolates in joint space with no collision checking.
               Fast but will collide with obstacles if they are in the way.

  RRT*       — plans a collision-free path in joint space before executing.
               Slower due to planning overhead but safe in cluttered scenes.

Each trial places a cylinder at the same random position and runs both methods
so the comparison is fair.

Usage:
    python examples/benchmark_cylinder_grasping.py
    python examples/benchmark_cylinder_grasping.py --trials 5 --visualize
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from manipulation import FrankaEnvironment, RRTStar, GraspPlanner
from manipulation.planners.grasp_planner import GraspType

_HERE = Path(__file__).parent
_XML  = (
    _HERE / ".." / "manipulation" / "environments"
    / "assets" / "franka_emika_panda" / "scene_test.xml"
)

TABLE_SURFACE_Z: float = 0.27
LIFT_THRESHOLD:  float = 0.08   # cylinder must rise ≥ 8 cm

PLACE_X = (0.20, 0.55)
PLACE_Y = (0.35, 0.60)

CYLINDER_NAMES = ["cylinder1", "cylinder2", "cylinder3"]


# ---------------------------------------------------------------------------
# Headless fast step — bypass RateLimiter
# ---------------------------------------------------------------------------

def _patch_fast_step(env: FrankaEnvironment) -> None:
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
    for _ in range(n):
        env.step()


def _wait_idle(env: FrankaEnvironment, max_steps: int = 5000) -> bool:
    ok = env.wait_idle(max_steps=max_steps, settle_steps=100)
    if not ok:
        print(f"    [warn] wait_idle timed out", flush=True)
    return ok


def get_ee_pos(env: FrankaEnvironment) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return env.data.site_xpos[sid].copy()


def reset_robot(env: FrankaEnvironment) -> None:
    env.data.qpos[:8] = env.initial_qpos[:8]
    env.data.ctrl[:8] = env.initial_ctrl[:8]
    env.data.qvel[:]  = 0.0
    env.controller.stop()
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    sim_steps(env, 30)


def place_cylinder(env: FrankaEnvironment, name: str,
                   half_size: np.ndarray, x: float, y: float) -> np.ndarray:
    z = TABLE_SURFACE_Z + half_size[2]
    env.set_object_pose(name, np.array([x, y, z]))
    mujoco.mj_forward(env.model, env.data)
    sim_steps(env, 60)
    return env.get_object_position(name).copy()


def _pick_candidate(candidates):
    """Prefer front (side) grasp — top-down is unstable on tall thin cylinders."""
    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


# ---------------------------------------------------------------------------
# IK trial
# ---------------------------------------------------------------------------

def run_ik_trial(
    env:           FrankaEnvironment,
    grasp_planner: GraspPlanner,
    cyl_name:      str,
    cyl_pos:       np.ndarray,
    half_size:     np.ndarray,
    verbose:       bool = True,
) -> dict:
    res = {"method": "direct", "cylinder": cyl_name,
           "success": False, "ee_err_mm": None, "ik_ok": True}

    block_quat = env.get_object_orientation(cyl_name)
    candidate  = _pick_candidate(
        grasp_planner.generate_candidates(cyl_pos, half_size, block_quat)
    )
    if candidate is None:
        res["ik_ok"] = False
        return res

    rest_z = cyl_pos[2]
    dt     = env.model.opt.timestep

    # 1. Approach
    env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
    if not env.ik.converge_ik(dt):
        res["ik_ok"] = False
        if verbose: print("    approach IK FAIL")
        return res
    env.move_to_compensated(env.ik.configuration.q[:7])
    _wait_idle(env)

    # 2. Grasp descent (slow interpolation)
    env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
    if not env.ik.converge_ik(dt):
        res["ik_ok"] = False
        if verbose: print("    grasp IK FAIL")
        return res
    env.move_to_compensated(env.ik.configuration.q[:7], step_size=0.003)
    _wait_idle(env)

    res["ee_err_mm"] = float(np.linalg.norm(get_ee_pos(env) - candidate.grasp_pos) * 1000)

    # 3. Close gripper
    env.controller.close_gripper()
    _wait_idle(env, max_steps=400)
    sim_steps(env, 40)

    # 4. Lift
    env.attach_object_to_ee(cyl_name)
    env.ik.set_target_position(candidate.lift_pos, candidate.grasp_quat)
    if not env.ik.converge_ik(dt):
        res["ik_ok"] = False
        env.detach_object()
        if verbose: print("    lift IK FAIL")
        return res
    env.move_to_compensated(env.ik.configuration.q[:7], step_size=0.003)
    _wait_idle(env)
    env.detach_object()

    cyl_z = env.get_object_position(cyl_name)[2]
    res["success"] = cyl_z > rest_z + LIFT_THRESHOLD

    if verbose:
        tag = "SUCCESS" if res["success"] else f"FAIL (z={cyl_z:.3f})"
        print(f"    IK  EE err={res['ee_err_mm']:.1f}mm  {tag}")

    return res


# ---------------------------------------------------------------------------
# RRT trial
# ---------------------------------------------------------------------------

def run_rrt_trial(
    env:           FrankaEnvironment,
    planner:       RRTStar,
    grasp_planner: GraspPlanner,
    cyl_name:      str,
    cyl_pos:       np.ndarray,
    half_size:     np.ndarray,
    verbose:       bool = True,
) -> dict:
    res = {"method": "rrt", "cylinder": cyl_name,
           "success": False, "ee_err_mm": None,
           "plan_attempts": 0, "plan_ok": 0}  # noqa: E501

    block_quat = env.get_object_orientation(cyl_name)
    candidate  = _pick_candidate(
        grasp_planner.generate_candidates(cyl_pos, half_size, block_quat)
    )
    if candidate is None:
        return res

    rest_z = cyl_pos[2]
    dt     = env.model.opt.timestep

    # 1. Approach
    res["plan_attempts"] += 1
    path = planner.plan_to_pose(candidate.approach_pos, candidate.grasp_quat,
                                dt=dt, max_iterations=2000)
    if path is None:
        if verbose: print("    approach plan FAIL")
        return res
    res["plan_ok"] += 1
    env.execute_path(path, planner)
    _wait_idle(env)

    # 2. Grasp descent
    env.add_collision_exception(cyl_name)
    res["plan_attempts"] += 1
    path = planner.plan_to_pose(candidate.grasp_pos, candidate.grasp_quat,
                                dt=dt, max_iterations=800)
    if path is None:
        env.clear_collision_exceptions()
        if verbose: print("    grasp plan FAIL")
        return res
    res["plan_ok"] += 1
    env.execute_path(path, planner, step_size=0.003)
    _wait_idle(env)

    res["ee_err_mm"] = float(np.linalg.norm(get_ee_pos(env) - candidate.grasp_pos) * 1000)

    # 3. Close gripper
    env.controller.close_gripper()
    _wait_idle(env, max_steps=400)
    sim_steps(env, 40)

    # 4. Lift
    env.attach_object_to_ee(cyl_name)
    res["plan_attempts"] += 1
    path = planner.plan_to_pose(candidate.lift_pos, candidate.grasp_quat,
                                dt=dt, max_iterations=2000)
    if path is None:
        env.detach_object()
        env.clear_collision_exceptions()
        if verbose: print("    lift plan FAIL")
        return res
    res["plan_ok"] += 1
    env.execute_path(path, planner, step_size=0.003)
    _wait_idle(env)
    env.detach_object()
    env.clear_collision_exceptions()

    cyl_z = env.get_object_position(cyl_name)[2]
    res["success"] = cyl_z > rest_z + LIFT_THRESHOLD

    if verbose:
        tag = "SUCCESS" if res["success"] else f"FAIL (z={cyl_z:.3f})"
        print(f"    RRT EE err={res['ee_err_mm']:.1f}mm  {tag}")

    return res


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def benchmark(trials_per_cylinder: int = 3, visualize: bool = False, seed: int = 42):
    rng = np.random.default_rng(seed)

    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    if visualize:
        env.launch_viewer()
    else:
        # Bypass rate limiter only in headless mode
        _patch_fast_step(env)

    planner = RRTStar(env)
    planner.max_iterations   = 2000
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    grasp_planner = GraspPlanner(table_z=TABLE_SURFACE_Z)

    # Query cylinder half-sizes from model
    half_sizes = {name: env.get_object_half_size(name) for name in CYLINDER_NAMES}

    ik_results  = []
    rrt_results = []
    t_start = time.time()

    for cyl_name in CYLINDER_NAMES:
        half_size = half_sizes[cyl_name]
        print(f"\n[{cyl_name}  r={half_size[0]*100:.1f}cm  h={half_size[2]*100:.0f}cm]")

        for trial in range(trials_per_cylinder):
            x = rng.uniform(*PLACE_X)
            y = rng.uniform(*PLACE_Y)
            print(f"  trial {trial+1}: place at ({x:.3f}, {y:.3f})")

            # --- Direct IK trial ---
            env.reset()
            reset_robot(env)
            cyl_pos = place_cylinder(env, cyl_name, half_size, x, y)
            t0  = time.time()
            res = run_ik_trial(env, grasp_planner, cyl_name, cyl_pos, half_size)
            res["wall_time"] = time.time() - t0
            res["trial"] = trial
            ik_results.append(res)
            if visualize and not res["success"]:
                env.rest(2.0)   # pause so failure is visible before reset

            # --- RRT* trial (same position) ---
            env.reset()
            reset_robot(env)
            cyl_pos = place_cylinder(env, cyl_name, half_size, x, y)
            t0  = time.time()
            res = run_rrt_trial(env, planner, grasp_planner, cyl_name, cyl_pos, half_size)
            res["wall_time"] = time.time() - t0
            res["trial"] = trial
            rrt_results.append(res)
            if visualize and not res["success"]:
                env.rest(2.0)   # pause so failure is visible before reset

    elapsed = time.time() - t_start
    _print_report(ik_results, rrt_results, elapsed, trials_per_cylinder)

    if visualize and env.viewer is not None:
        env.viewer.close()

    return ik_results, rrt_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(ik_results, rrt_results, elapsed, tpb):
    print("\n\n" + "=" * 65)
    print("  CYLINDER GRASPING BENCHMARK")
    print("  Direct IK (linear interp, no collision check) vs RRT*")
    print("=" * 65)
    print(f"\n  Trials: {len(ik_results)}  ({tpb}/cylinder)   Wall time: {elapsed:.1f} s\n")

    for label, results in [("Direct", ik_results), ("RRT*  ", rrt_results)]:
        n   = len(results)
        ok  = sum(r["success"] for r in results)
        err = [r["ee_err_mm"] for r in results if r["ee_err_mm"] is not None]
        t   = [r["wall_time"] for r in results]

        print(f"  {label}  grasp success : {ok}/{n} = {100*ok/max(n,1):.0f}%")
        print(f"       wall time/trial : mean {np.mean(t):.2f}s  "
              f"median {np.median(t):.2f}s  max {np.max(t):.2f}s")
        if err:
            print(f"       EE pos error   : mean {np.mean(err):.2f}mm  "
                  f"median {np.median(err):.2f}mm  "
                  f"p95 {np.percentile(err, 95):.2f}mm")
        if label.strip() == "RRT*":
            pa = sum(r.get("plan_attempts", 0) for r in results)
            pk = sum(r.get("plan_ok", 0)       for r in results)
            print(f"       plan success   : {pk}/{pa} = {100*pk/max(pa,1):.0f}%")
        print()

    print("  By cylinder:")
    for cyl in CYLINDER_NAMES:
        ik_r  = [r for r in ik_results  if r["cylinder"] == cyl]
        rrt_r = [r for r in rrt_results if r["cylinder"] == cyl]
        ik_ok  = sum(r["success"] for r in ik_r)
        rrt_ok = sum(r["success"] for r in rrt_r)
        ik_t   = np.mean([r["wall_time"] for r in ik_r])
        rrt_t  = np.mean([r["wall_time"] for r in rrt_r])
        print(f"    {cyl:<12}  Direct {ik_ok}/{len(ik_r)} ({ik_t:.2f}s)   "
              f"RRT* {rrt_ok}/{len(rrt_r)} ({rrt_t:.2f}s)")

    print("=" * 65)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",    type=int,  default=3)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--seed",      type=int,  default=42)
    args = ap.parse_args()
    benchmark(trials_per_cylinder=args.trials, visualize=args.visualize, seed=args.seed)
