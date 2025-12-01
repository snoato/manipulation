"""Franka Panda robot environment implementation."""

import mujoco
import numpy as np
from typing import Optional, List

from manipulation.core.base_env import BaseEnvironment
from manipulation.ik.mink_ik import MinkIK
from manipulation.controllers.position_controller import PositionController
from manipulation.utils.rate_limiter import RateLimiter


class FrankaEnvironment(BaseEnvironment):
    """Environment for Franka Emika Panda robot simulation."""

    def __init__(self, path: str, rate: float = 200.0, collision_bodies: Optional[List[str]] = None):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ik = MinkIK(self.model, self.data)
        self.controller = PositionController(self.model, self.data)
        
        # List of body names to check for collisions
        if collision_bodies is not None:
            self.collision_bodies = collision_bodies
        else:
            self.collision_bodies = [
                "link0", "link1", "link2", "link3", "link4", "link5", "link6",
                "hand", "right_finger", "left_finger"
            ]

        self.collision_exceptions = []

        # Set initial position
        self.data.qpos[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
        self.data.ctrl[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
        self.ik.update_configuration(self.data.qpos)
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        self.rate = RateLimiter(frequency=rate, warn=False)
        self.sim_time = 0.0
        self.viewer = None

    def get_model(self):
        return self.model
    
    def get_data(self):
        return self.data

    def get_ik(self):
        return self.ik
    
    def launch_viewer(self):
        self.sim_time = 0.0
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        mujoco.mj_forward(self.model, self.data)
        return self.viewer

    def reset(self):
        self.data.ctrl[:] = self.initial_ctrl
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.ik.update_configuration(self.data.qpos)
        self.sim_time = 0.0

    def step(self):
        mujoco.mj_step(self.model, self.data)
        dt = self.rate.dt
        self.sim_time += dt
        if self.viewer is not None:
            self.viewer.sync()
        self.rate.sleep()
        return dt

    def rest(self, duration: float):
        steps = int(duration / self.rate.dt)
        for _ in range(steps):
            self.step()

    def add_collision_exception(self, body_name: str):
        if body_name not in self.collision_exceptions:
            self.collision_exceptions.append(body_name)

    def remove_collision_exception(self, body_name: str):
        if body_name in self.collision_exceptions:
            self.collision_exceptions.remove(body_name)
    
    def clear_collision_exceptions(self):
        self.collision_exceptions = []

    def check_collisions(self) -> bool:
        collision_free = True
        if self.data.ncon > 0:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                body1 = self.model.geom_bodyid[contact.geom1]
                name1 = self.model.body(body1).name
                body2 = self.model.geom_bodyid[contact.geom2]
                name2 = self.model.body(body2).name

                if name1 in self.collision_exceptions or name2 in self.collision_exceptions:
                    continue
                
                if name1 in self.collision_bodies and name2 in self.collision_bodies:
                    continue
                if name1 not in self.collision_bodies and name2 not in self.collision_bodies:
                    continue
                if contact.dist < -1e-4:
                    collision_free = False
                    break
        return collision_free

    def is_collision_free(self, configuration: np.ndarray) -> bool:
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        
        self.data.qpos[:7] = configuration
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        collision_free = self.check_collisions()

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return collision_free

    def get_object_id(self, object_name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if object_id == -1:
            raise ValueError(f"Object '{object_name}' not found in the model.")
        return object_id

    def get_object_position(self, object_name: str) -> np.ndarray:
        object_id = self.get_object_id(object_name)
        return self.data.xpos[object_id].copy()
    
    def get_object_orientation(self, object_name: str) -> np.ndarray:
        object_id = self.get_object_id(object_name)
        return self.data.xquat[object_id].copy()

    def get_approach_pose(
        self,
        target: np.ndarray,
        offset: np.ndarray = np.array([0, -0.1, 0.1]),
        orientation: np.ndarray = np.array([-0.5, 0.5, 0.5, 0.5])
    ):
        pos = target + offset
        return pos, orientation
    
    def get_grasp_pose(
        self,
        target: np.ndarray,
        offset: np.ndarray = np.array([0, 0, 0.03]),
        orientation: np.ndarray = np.array([-0.5, 0.5, 0.5, 0.5])
    ):
        pos = target + offset
        return pos, orientation
    
    def get_lift_pose(
        self,
        target: np.ndarray,
        offset: np.ndarray = np.array([0, 0, 0.2]),
        orientation: np.ndarray = np.array([-0.5, 0.5, 0.5, 0.5])
    ):
        pos = target + offset
        return pos, orientation
    
    def get_dropoff_pose(self):
        return np.array([0.5, 0, 0.5]), np.array([0, 1, 0, 0])
