"""Focused diagnostic of action 39 — pick blocker_13 from shelf_top__1_0.

Runs the L4 return-all plan via FULL up to action 38, then snapshots
the env state, then runs action 39's chain phase-by-phase to see
WHERE the gripper actually goes vs where it's supposed to go.

The point is to answer: does the chain overreach in y (gripper ends
up near the +y neighbour cube), or does it position correctly (-y of
the target) and something else knocks the neighbour?
"""
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent))
from access19_full_validation import _setup, _OBJECT_NAMES
from tampanda.symbolic.domains.access19.templates import (
    canonical_18, source_layout as sl, goal_layout as gl,
)
from tampanda.symbolic.domains.access19.planner import phased_plan
from tampanda.symbolic.domains.access19.parallel import _layout_to_state
from tampanda.symbolic.domains.access19.feasibility import (
    check_action_sequence, _dispatch,
)
from tampanda.symbolic.domains.access19.state import restore_state
from tampanda.symbolic.workspace import Cell


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        env, ws, cfg, executor, pick_fn, put_fn, shelf_home = _setup(Path(td))
        template = canonical_18(return_blockers=True)
        init = sl(template)
        goal = gl(template)
        print("planning ...", flush=True)
        res = phased_plan(env, ws, cfg, init, goal, _OBJECT_NAMES,
                              pick_fn, put_fn, executor=executor,
                              home_qpos=shelf_home, fast=True,
                              phase2_time_budget_s=300.0,
                              phase2_max_states=30000)
        plan = list(res.plan)
        print(f"  plan: {len(plan)} actions")

        # Restore and run actions 1..38 via FULL.
        init_state = _layout_to_state(init, held=None)
        restore_state(env, ws, cfg, init_state, _OBJECT_NAMES,
                          home_qpos=shelf_home)
        prefix_res = check_action_sequence(
            env, ws, cfg, init_state, plan[:38], _OBJECT_NAMES,
            pick_fn, put_fn, executor=executor, fast=False,
            home_qpos=shelf_home, short_circuit=True,
        )
        print(f"  prefix: success={prefix_res['success']}")

        # Snapshot cube positions BEFORE action 39.
        print("\nCube positions after prefix (38 actions via FULL):")
        for name in ("blocker_10", "blocker_12", "blocker_13",
                          "blocker_15", "blocker_16", "ooi"):
            pos, quat = env.get_object_pose(name)
            print(f"  {name:<12} pos=({pos[0]:.3f}, {pos[1]:.3f}, "
                      f"{pos[2]:.3f})  quat=({quat[0]:.3f}, {quat[1]:.3f}, "
                      f"{quat[2]:.3f}, {quat[3]:.3f})")

        sym_target = ws.pose_for(Cell.parse("shelf_top__1_0"))
        sym_neighbor = ws.pose_for(Cell.parse("shelf_top__1_1"))
        print(f"\nSymbolic shelf_top__1_0 = "
                  f"({sym_target[0]:.3f}, {sym_target[1]:.3f}, "
                  f"{sym_target[2]:.3f})")
        print(f"Symbolic shelf_top__1_1 = "
                  f"({sym_neighbor[0]:.3f}, {sym_neighbor[1]:.3f}, "
                  f"{sym_neighbor[2]:.3f})")

        # Reset arm to home (as check_action_sequence does between actions).
        env.data.qpos[: len(shelf_home)] = shelf_home
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        if env._attached is not None:
            env._apply_attachment()
            mujoco.mj_forward(env.model, env.data)

        site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        # Patch execute_path AND wait_idle to log EE pre/target/post.
        # env.execute_path only queues the trajectory; the actual motion
        # happens during env.wait_idle which drives the controller +
        # physics.  So pre_EE+target_EE are captured at execute_path
        # entry, post_EE after wait_idle returns.
        orig_execute_path = env.execute_path
        orig_wait_idle = env.wait_idle
        state = {"counter": 0, "pre": None, "target": None}

        def logged_execute_path(path, planner, *args, **kwargs):
            state["counter"] += 1
            state["pre"] = env.data.site_xpos[site_id].copy()
            target_q = np.asarray(path[-1])
            save_q = env.data.qpos[:7].copy()
            env.data.qpos[:7] = target_q[:7]
            mujoco.mj_forward(env.model, env.data)
            state["target"] = env.data.site_xpos[site_id].copy()
            env.data.qpos[:7] = save_q
            mujoco.mj_forward(env.model, env.data)
            return orig_execute_path(path, planner, *args, **kwargs)

        def logged_wait_idle(*args, **kwargs):
            ret = orig_wait_idle(*args, **kwargs)
            post_ee = env.data.site_xpos[site_id].copy()
            # Gripper state at end of motion.
            grip_qpos = env.data.qpos[7:9].copy()
            grip_ctrl = env.data.ctrl[7]
            # EE rotation matrix (we want the approach axis direction).
            ee_rot = np.asarray(env.data.site_xmat[site_id],
                                       dtype=float).reshape(3, 3)
            # In palm-+y orientation, the local +z axis (approach
            # direction) should align with world +y, so ee_rot @ [0,0,1]
            # should equal [0, 1, 0].
            approach_world = ee_rot @ np.array([0.0, 0.0, 1.0])
            # Cube positions of interest.
            b13_pos, b13_quat = env.get_object_pose("blocker_13")
            b12_pos, b12_quat = env.get_object_pose("blocker_12")
            pre = state["pre"] if state["pre"] is not None else post_ee
            target = state["target"] if state["target"] is not None else post_ee
            err = float(np.linalg.norm(post_ee - target))
            print(f"  exec #{state['counter']}: "
                      f"pre=({pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f}) "
                      f"target=({target[0]:.3f},{target[1]:.3f},"
                      f"{target[2]:.3f}) "
                      f"post=({post_ee[0]:.3f},{post_ee[1]:.3f},"
                      f"{post_ee[2]:.3f}) err={err*1000:.0f}mm | "
                      f"grip=[{grip_qpos[0]:.3f},{grip_qpos[1]:.3f}] "
                      f"ctrl={grip_ctrl:.2f} | "
                      f"approach_axis=({approach_world[0]:+.2f},"
                      f"{approach_world[1]:+.2f},{approach_world[2]:+.2f})")
            print(f"         b13_pos=({b13_pos[0]:.3f},{b13_pos[1]:.3f},"
                      f"{b13_pos[2]:.3f}) "
                      f"b13_quat=({b13_quat[0]:.3f},{b13_quat[1]:.3f},"
                      f"{b13_quat[2]:.3f},{b13_quat[3]:.3f})")
            print(f"         b12_pos=({b12_pos[0]:.3f},{b12_pos[1]:.3f},"
                      f"{b12_pos[2]:.3f}) "
                      f"b12_quat=({b12_quat[0]:.3f},{b12_quat[1]:.3f},"
                      f"{b12_quat[2]:.3f},{b12_quat[3]:.3f})")
            return ret

        env.execute_path = logged_execute_path
        env.wait_idle = logged_wait_idle

        # Also wrap controller.close_gripper / open_gripper to log timing.
        orig_close = env.controller.close_gripper
        orig_open = env.controller.open_gripper

        def logged_close():
            grip_qpos = env.data.qpos[7:9].copy()
            print(f"  >> CLOSE_GRIPPER called (current qpos={grip_qpos[0]:.3f},"
                      f"{grip_qpos[1]:.3f})")
            return orig_close()

        def logged_open():
            grip_qpos = env.data.qpos[7:9].copy()
            print(f"  >> OPEN_GRIPPER called (current qpos={grip_qpos[0]:.3f},"
                      f"{grip_qpos[1]:.3f})")
            return orig_open()

        env.controller.close_gripper = logged_close
        env.controller.open_gripper = logged_open

        print(f"\n=== Running action 39: pick blocker_13 from "
                  f"shelf_top__1_0 ===")
        action = ('pick', 'blocker_13', 'shelf_top__1_0')
        pos, _ = env.get_object_pose('blocker_13')
        print(f"  blocker_13 current pos = "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  -> pick_fn target_y will be: {pos[1] - 0.014:.3f}")

        ok = _dispatch(env, ws, pick_fn, put_fn, action)
        print(f"\n  action result: {ok}")
        print("\nCube positions after action 39:")
        for name in ("blocker_10", "blocker_12", "blocker_13"):
            pos, quat = env.get_object_pose(name)
            print(f"  {name:<12} pos=({pos[0]:.3f}, {pos[1]:.3f}, "
                      f"{pos[2]:.3f})  quat=({quat[0]:.3f}, {quat[1]:.3f}, "
                      f"{quat[2]:.3f}, {quat[3]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
