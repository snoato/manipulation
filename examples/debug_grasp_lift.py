"""
Headless debug script: diagnose and verify the EE dip fix.

Tests two modes of execute_path:
  MODE A: compensate only last waypoint  (old behaviour → shows dip)
  MODE B: compensate ALL waypoints       (fix → should show no dip)

What the numbers mean:
  ctrl_drop   — max |ctrl_before - ctrl_traj1| at the transition.
                With mode A traj[0] is immediately consumed (ctrl==ctrl_ref),
                so ctrl jumps to uncompensated traj[1].
                With mode B every waypoint is compensated so ctrl is stable.
  EE dip      — how many mm the EE drops below the grasp z before climbing.
                > 1 mm is noticeable; should be < 0.1 mm after the fix.

Run:
    python examples/debug_grasp_lift.py
"""

from pathlib import Path
import numpy as np
import mujoco

from tampanda import FrankaEnvironment, RRTStar, ControllerStatus, SCENE_BLOCKS
from tampanda.environments.franka_env import _EFF_KP

_XML = SCENE_BLOCKS

GRASP_POS  = np.array([0.45, 0.10, 0.31])
GRASP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
LIFT_POS   = GRASP_POS + np.array([0.0, 0.0, 0.18])

# ---- headless helpers -------------------------------------------------

def get_ee_pos(env: FrankaEnvironment) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return env.data.site_xpos[sid].copy()


def step_both(env: FrankaEnvironment):
    """Step controller + physics (the canonical pair used by blocks_world)."""
    env.controller.step()
    mujoco.mj_step(env.model, env.data)


def run_to_idle(env: FrankaEnvironment, max_steps: int = 6000) -> bool:
    for _ in range(max_steps):
        step_both(env)
        if env.controller.get_status() == ControllerStatus.IDLE:
            return True
    return False


def run_n(env: FrankaEnvironment, n: int):
    for _ in range(n):
        step_both(env)


# ---- gravity comp helpers (mirror of franka_env) -----------------------

def gravity_comp_one(env: FrankaEnvironment, goal_q: np.ndarray) -> np.ndarray:
    qpos_save = env.data.qpos.copy()
    qvel_save = env.data.qvel.copy()
    ctrl_save = env.data.ctrl.copy()
    env.data.qpos[:7] = goal_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    e_ss = env.data.qfrc_bias[:7] / _EFF_KP
    env.data.qpos[:] = qpos_save
    env.data.qvel[:] = qvel_save
    env.data.ctrl[:] = ctrl_save
    mujoco.mj_forward(env.model, env.data)
    return goal_q + e_ss


def gravity_comp_all(env: FrankaEnvironment, traj: list) -> list:
    """Compensate every waypoint in one save/restore cycle."""
    qpos_save = env.data.qpos.copy()
    qvel_save = env.data.qvel.copy()
    ctrl_save = env.data.ctrl.copy()
    env.data.qvel[:] = 0.0
    result = []
    for wp in traj:
        env.data.qpos[:7] = wp
        mujoco.mj_forward(env.model, env.data)
        result.append(wp + env.data.qfrc_bias[:7] / _EFF_KP)
    env.data.qpos[:] = qpos_save
    env.data.qvel[:] = qvel_save
    env.data.ctrl[:] = ctrl_save
    mujoco.mj_forward(env.model, env.data)
    return result


def execute_path_mode_a(env, path, planner, step_size=0.003):
    """OLD: compensate only last waypoint."""
    smoothed = planner.smooth_path(path)
    traj = env.controller.interpolate_linear_path(smoothed, step_size=step_size)
    traj[-1] = gravity_comp_one(env, traj[-1])
    env.controller.follow_trajectory(traj)
    return traj


def execute_path_mode_b(env, path, planner, step_size=0.003):
    """NEW: compensate ALL waypoints."""
    smoothed = planner.smooth_path(path)
    traj = env.controller.interpolate_linear_path(smoothed, step_size=step_size)
    traj = gravity_comp_all(env, traj)
    env.controller.follow_trajectory(traj)
    return traj


# ---- single trial ---------------------------------------------------------

def trial(env: FrankaEnvironment, planner: RRTStar, mode: str, n_record: int = 150) -> dict:
    """
    mode: 'A' = last only, 'B' = all waypoints
    """
    env.reset()
    env.controller.stop()   # clear any leftover trajectory from a previous trial
    mujoco.mj_forward(env.model, env.data)

    # Move to grasp position with the correct (mode B) approach so we start from
    # the same hold state in both trials.
    path = planner.plan_to_pose(GRASP_POS, GRASP_QUAT, dt=0.005, max_iterations=3000)
    if path is None:
        raise RuntimeError("Could not plan approach")
    execute_path_mode_b(env, path, planner, step_size=0.01)
    run_to_idle(env)
    run_n(env, 100)  # settle

    grasp_ee_z   = get_ee_pos(env)[2]
    ctrl_hold    = env.data.ctrl[:7].copy()
    qpos_hold    = env.data.qpos[:7].copy()
    bias_hold    = env.data.qfrc_bias[:7].copy()

    # simulate gripper close + settle (include controller.step() to clear GRASPING)
    env.controller.close_gripper()
    run_n(env, 80)   # step_both clears GRASPING on first tick

    assert env.controller.get_status() == ControllerStatus.IDLE, \
        "Controller not IDLE before lift — check GRASPING clearance"

    # Plan lift
    path_lift = planner.plan_to_pose(LIFT_POS, GRASP_QUAT, dt=0.005, max_iterations=3000)
    if path_lift is None:
        raise RuntimeError("Could not plan lift")

    ctrl_before = env.data.ctrl[:7].copy()

    # Hand off lift trajectory using chosen mode
    if mode == 'A':
        traj = execute_path_mode_a(env, path_lift, planner, step_size=0.003)
    else:
        traj = execute_path_mode_b(env, path_lift, planner, step_size=0.003)

    traj1 = np.array(traj[1]) if len(traj) > 1 else np.array(traj[0])

    # Step once so controller loads traj[0]
    step_both(env)
    ctrl_after_first_step = env.data.ctrl[:7].copy()

    # Record n_record steps
    ee_zs  = [get_ee_pos(env)[2]]
    ctrls  = [env.data.ctrl[:7].copy()]
    qposs  = [env.data.qpos[:7].copy()]

    for _ in range(n_record - 1):
        step_both(env)
        ee_zs.append(get_ee_pos(env)[2])
        ctrls.append(env.data.ctrl[:7].copy())
        qposs.append(env.data.qpos[:7].copy())

    ee_zs = np.array(ee_zs)
    ctrls = np.array(ctrls)

    ctrl_drop = ctrl_before - ctrl_after_first_step
    dip_mm    = (grasp_ee_z - ee_zs.min()) * 1000
    step_min  = int(np.argmin(ee_zs))

    return {
        "mode":               mode,
        "grasp_ee_z_mm":      grasp_ee_z * 1000,
        "ctrl_hold":          ctrl_hold,
        "qpos_hold":          qpos_hold,
        "bias_hold":          bias_hold,
        "ctrl_before":        ctrl_before,
        "ctrl_after_step1":   ctrl_after_first_step,
        "traj1":              traj1,
        "ctrl_drop":          ctrl_drop,
        "max_ctrl_drop_mrad": float(np.max(np.abs(ctrl_drop)) * 1000),
        "dip_mm":             float(dip_mm),
        "step_of_min_z":      step_min,
        "ee_zs_mm":           ee_zs * 1000,
        "n_traj_wps":         len(traj),
    }


def report(d: dict):
    mode = d['mode']
    print(f"\n{'='*60}")
    print(f"  MODE {mode}: {'last waypoint only' if mode=='A' else 'ALL waypoints compensated'}")
    print(f"{'='*60}")
    print(f"  Trajectory waypoints : {d['n_traj_wps']}")
    print(f"  Grasp EE z           : {d['grasp_ee_z_mm']:.2f} mm")
    print(f"\n  gravity bias (qfrc)  : {np.round(d['bias_hold'], 2)} Nm")
    print(f"  ctrl_hold - qpos     : {np.round(d['ctrl_hold'] - d['qpos_hold'], 4)} rad")
    print(f"\n  ctrl_drop at step 1  : {np.round(d['ctrl_drop'], 4)} rad")
    print(f"  max |ctrl_drop|      : {d['max_ctrl_drop_mrad']:.1f} mrad")
    print(f"\n  EE DIP below grasp   : {d['dip_mm']:.3f} mm  (step {d['step_of_min_z']})")
    print(f"  EE z first 10 steps  : {np.round(d['ee_zs_mm'][:10], 3)} mm")
    verdict = "OK (< 0.5 mm)" if d['dip_mm'] < 0.5 else f"FAIL — visible dip!"
    print(f"\n  >> {verdict}")


# ---- main ----------------------------------------------------------------

def main():
    print("Loading environment (headless)...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    print("Initialising planner...")
    planner = RRTStar(env)
    planner.max_iterations   = 2000
    planner.step_size        = 0.15
    planner.goal_sample_rate = 0.15

    print(f"\nGrasp: {GRASP_POS}  Lift: {LIFT_POS}")

    da = trial(env, planner, 'A')
    report(da)

    db = trial(env, planner, 'B')
    report(db)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Mode A dip : {da['dip_mm']:.3f} mm")
    print(f"  Mode B dip : {db['dip_mm']:.3f} mm")
    print(f"  Improvement: {da['dip_mm'] - db['dip_mm']:.3f} mm")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
