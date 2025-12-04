"""Inverse kinematics implementation using MINK."""

import mujoco
import numpy as np
import mink

from manipulation.core.base_ik import BaseIK


class MinkIK(BaseIK):
    """Inverse kinematics solver using the MINK library."""

    def __init__(self, model, data, target_name: str = "target"):
        self.model = model
        self.data = data
        self.aux_tasks = []
        self.solver = "quadprog"
        self.pos_threshold = 0.006
        self.ori_threshold = 1e-3
        self.max_iters = 600
        self.ee_task = None
        self.posture_task = None
        self.target_name = target_name

        self.configuration = mink.Configuration(model)

        self.set_ee_task()
        self.set_posture_task()

    def add_aux_task(self, task):
        self.aux_tasks.append(task)

    def set_ee_task(self, task=None):
        if task is None:
            self.ee_task = mink.FrameTask(
                frame_name="attachment_site",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
        else:
            self.ee_task = task

    def set_posture_task(self, task=None):
        if task is None:
            self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        else:
            self.posture_task = task

    def tasks(self):
        tasks = []
        if self.ee_task is not None:
            tasks.append(self.ee_task)
        if self.posture_task is not None:
            tasks.append(self.posture_task)
        tasks.extend(self.aux_tasks)
        return tasks

    def set_target_position(self, pos: np.ndarray, quat: np.ndarray):
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat

    def converge_ik(self, dt: float) -> bool:
        # Update the end effector task target from the mocap body
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, self.target_name)
        self.ee_task.set_target(T_wt)

        self.posture_task.set_target_from_configuration(self.configuration)
        for _ in range(self.max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks(), dt, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, dt)

            err = self.ee_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold

            if pos_achieved and ori_achieved:
                return True
        print("MINK IK failed to converge within the maximum iterations"
              f" (pos error: {np.linalg.norm(err[:3]):.6f},"
              f" ori error: {np.linalg.norm(err[3:]):.6f}))")
        return False
    
    def update_configuration(self, qpos: np.ndarray):
        self.configuration.update(qpos)
