"""Blocking pick-and-place executor built on FrankaEnvironment + RRTStar.

Provides a high-level, reusable interface for grasping and placing objects
using the GraspPlanner for grasp candidate generation, execute_path for smooth
gravity-compensated trajectory execution, and wait_idle for synchronous
blocking waits.

Typical usage:
    from manipulation import FrankaEnvironment, RRTStar, GraspPlanner
    from manipulation.planners import PickPlaceExecutor

    env = FrankaEnvironment(xml_path)
    planner = RRTStar(env)
    grasp_planner = GraspPlanner(table_z=0.27)
    executor = PickPlaceExecutor(env, planner, grasp_planner)

    block_pos  = env.get_object_position("block_0")
    half_size  = np.array([0.02, 0.02, 0.02])
    block_quat = env.get_object_orientation("block_0")

    success = executor.pick("block_0", block_pos, half_size, block_quat)
    if success:
        place_pos = np.array([0.5, 0.2, target_top_z + 0.02])
        executor.place(place_pos, np.array([0, 1, 0, 0]))
"""

import numpy as np
import mujoco
from typing import Optional

from manipulation.planners.grasp_planner import GraspPlanner, GraspCandidate, GRASP_CONTACT_OFFSET
from manipulation.controllers.position_controller import ControllerStatus


class PickPlaceExecutor:
    """Blocking pick-and-place state machine.

    All methods are synchronous: they run the simulation loop internally
    and return only when the action is complete (or has failed).

    Args:
        env: A FrankaEnvironment instance.
        planner: An RRTStar instance configured for this env.
        grasp_planner: A GraspPlanner instance.  If None, a default one is
            created (requires table_z to be passed per call).
        approach_step_size: Joint-space interpolation step for approach moves.
        grasp_step_size: Joint-space step for final grasp descent.
        lift_step_size: Joint-space step for lift (slow = less cube slip).
        place_step_size: Joint-space step for placement descent.
        retreat_step_size: Joint-space step for post-place retreat.
        max_plan_iters: RRT max iterations for each planning call.
        arm_settle_vel: qvel norm threshold — arm considered settled below this.
        gripper_start_vel: qvel[7] threshold — gripper started closing above this.
        gripper_stop_vel: qvel[7] threshold — gripper fully closed below this.
        settle_steps: Extra physics steps after wait_idle finishes.
        use_attachment: If True, kinematically attach the object to the EE during
            lift and place so it cannot slip.  Detached before gripper opens.
    """

    def __init__(
        self,
        env,
        planner,
        grasp_planner: Optional[GraspPlanner] = None,
        approach_step_size: float = 0.01,
        grasp_step_size: float = 0.003,
        lift_step_size: float = 0.003,
        place_step_size: float = 0.003,
        retreat_step_size: float = 0.01,
        max_plan_iters: int = 2000,
        arm_settle_vel: float = 0.01,
        gripper_start_vel: float = 0.0005,
        gripper_stop_vel: float = 0.0001,
        settle_steps: int = 80,
        use_attachment: bool = True,
    ):
        self.env = env
        self.planner = planner
        self.grasp_planner = grasp_planner or GraspPlanner()
        self.approach_step_size = approach_step_size
        self.grasp_step_size    = grasp_step_size
        self.lift_step_size     = lift_step_size
        self.place_step_size    = place_step_size
        self.retreat_step_size  = retreat_step_size
        self.max_plan_iters     = max_plan_iters
        self.arm_settle_vel     = arm_settle_vel
        self.gripper_start_vel  = gripper_start_vel
        self.gripper_stop_vel   = gripper_stop_vel
        self.settle_steps       = settle_steps
        self.use_attachment     = use_attachment

        self._ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pick(
        self,
        block_name: str,
        block_pos: np.ndarray,
        half_size: np.ndarray,
        block_quat: np.ndarray,
    ) -> bool:
        """Execute a full grasping sequence for the named block.

        Tries each GraspPlanner candidate in turn.  For each candidate:
          1. Plan and execute approach (above grasp position)
          2. Plan and execute grasp descent (collision-excepted)
          3. Wait for arm to settle, verify EE position
          4. Close gripper, wait for fingers to stop
          5. Optionally attach block to EE kinematically
          6. Execute lift

        Returns True on success, False if all candidates were exhausted.
        """
        candidates = self.grasp_planner.generate_candidates(
            block_pos, half_size, block_quat
        )
        if not candidates:
            print(f"[PickPlaceExecutor] No grasp candidates for {block_name}")
            return False

        for i, cand in enumerate(candidates):
            print(f"[PickPlaceExecutor] Trying candidate {i+1}/{len(candidates)}: "
                  f"{cand.grasp_type.value}")

            # 1. Approach
            path = self.planner.plan_to_pose(
                cand.approach_pos, cand.grasp_quat,
                dt=0.005, max_iterations=self.max_plan_iters,
            )
            if path is None:
                print("  approach plan failed — next candidate")
                continue
            self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            # 2. Grasp descent (block not an obstacle)
            self.env.add_collision_exception(block_name)
            path = self.planner.plan_to_pose(
                cand.grasp_pos, cand.grasp_quat,
                dt=0.005, max_iterations=self.max_plan_iters,
            )
            if path is None:
                print("  grasp descent plan failed — next candidate")
                self.env.remove_collision_exception(block_name)
                continue
            self.env.execute_path(path, self.planner, step_size=self.grasp_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            # 3. Arm settle check + EE accuracy report
            self._wait_arm_settle()
            ee_pos = self.env.data.site_xpos[self._ee_site_id].copy()
            err_mm = np.linalg.norm(ee_pos - cand.grasp_pos) * 1000
            print(f"  EE error at grasp: {err_mm:.1f} mm")

            # 4. Close gripper, wait for full closure
            self.env.controller.close_gripper()
            self._wait_gripper_closed()

            # 5. Kinematic attachment (prevents slip during lift)
            if self.use_attachment:
                self.env.attach_object_to_ee(block_name)

            # 6. Lift
            path = self.planner.plan_to_pose(
                cand.lift_pos, cand.grasp_quat,
                dt=0.005, max_iterations=self.max_plan_iters,
            )
            if path is None:
                print("  lift plan failed — pick aborted")
                self.env.detach_object()
                return False
            self.env.execute_path(path, self.planner, step_size=self.lift_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            print(f"  pick SUCCESS")
            return True

        print(f"[PickPlaceExecutor] All {len(candidates)} candidates exhausted for {block_name}")
        return False

    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: np.ndarray,
        target_block_name: Optional[str] = None,
        approach_height: float = 0.15,
        place_clearance: float = 0.003,
    ) -> bool:
        """Execute a blocking place sequence.

        The EE descends to the position that puts the held block's centre at
        place_block_center + place_clearance above the target surface
        (GRASP_CONTACT_OFFSET is added automatically).  After the gripper opens
        physics settles the block the remaining clearance onto the surface.

        The kinematic attachment (if active) is released before the descent so
        that contact forces are handled by physics rather than fighting the
        forced position — this prevents the block from being pushed into the
        target.

        Args:
            block_name: Name of the held block (for collision exceptions).
            place_block_center: 3D target position for the held block's centre
                                when resting on the target surface.
            ee_quat: EE orientation during placement.
            target_block_name: If given, add collision exception for the target.
            approach_height: Extra height above the placement EE position to
                             approach from.
            place_clearance: Extra z added to ee_place so the block is released
                             this far above the target surface (default 3 mm).
                             Physics settles it down naturally.

        Returns True on success.
        """
        # EE position when block centre is at place_block_center + clearance
        ee_place = place_block_center.copy()
        ee_place[2] += GRASP_CONTACT_OFFSET + place_clearance

        ee_approach = ee_place.copy()
        ee_approach[2] += approach_height

        # Add exceptions for held block and (optionally) target
        self.env.add_collision_exception(block_name)
        if target_block_name is not None:
            self.env.add_collision_exception(target_block_name)

        # 1. Approach above placement (attachment still active — block rides along)
        path = self.planner.plan_to_pose(
            ee_approach, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is None:
            print("[PickPlaceExecutor] place approach plan failed")
            self._clear_exceptions(block_name, target_block_name)
            return False
        self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)

        # 2. Detach before descent so contact forces are handled by physics
        if self.use_attachment:
            self.env.detach_object()

        # 3. Descend to placement position
        path = self.planner.plan_to_pose(
            ee_place, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is None:
            print("[PickPlaceExecutor] place descent plan failed")
            self._clear_exceptions(block_name, target_block_name)
            return False
        self.env.execute_path(path, self.planner, step_size=self.place_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)

        # 4. Open gripper, let physics settle the block onto the surface
        self.env.controller.open_gripper()
        self._wait_gripper_open()

        # 5. Retreat upward
        self._clear_exceptions(block_name, target_block_name)
        path = self.planner.plan_to_pose(
            ee_approach, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is not None:
            self.env.execute_path(path, self.planner, step_size=self.retreat_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

        print("[PickPlaceExecutor] place SUCCESS")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_arm_settle(self, max_steps: int = 500):
        """Spin until joint velocity is below arm_settle_vel (or timeout)."""
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            if np.linalg.norm(self.env.data.qvel[:7]) < self.arm_settle_vel:
                return

    def _wait_gripper_closed(self, max_steps: int = 1000):
        """Two-phase wait: fingers start moving, then stop (contact or limit)."""
        started = False
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            vel = abs(self.env.data.qvel[7])
            if not started:
                if vel > self.gripper_start_vel:
                    started = True
            else:
                if vel < self.gripper_stop_vel:
                    return

    def _wait_gripper_open(self, steps: int = 300):
        """Run physics for a fixed number of steps after opening the gripper."""
        for _ in range(steps):
            self.env.controller.step()
            self.env.step()

    def _clear_exceptions(self, block_name: str, target_block_name: Optional[str]):
        self.env.remove_collision_exception(block_name)
        if target_block_name is not None:
            self.env.remove_collision_exception(target_block_name)
