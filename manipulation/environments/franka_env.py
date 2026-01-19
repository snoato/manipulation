"""Franka Panda robot environment implementation."""

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
from typing import Optional, List, Tuple

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
        
        # Cache body IDs for fast collision checking (converted to set for O(1) lookup)
        self._collision_body_ids = set(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.collision_bodies
        )
        self._collision_exception_ids = set()

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
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self._collision_exception_ids.add(body_id)

    def remove_collision_exception(self, body_name: str):
        if body_name in self.collision_exceptions:
            self.collision_exceptions.remove(body_name)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self._collision_exception_ids.discard(body_id)
    
    def clear_collision_exceptions(self):
        self.collision_exceptions = []
        self._collision_exception_ids = set()
    
    def set_collision_exceptions(self, body_names: list):
        """Set collision exceptions and update cached IDs."""
        self.collision_exceptions = list(body_names)
        self._collision_exception_ids = set()
        for name in body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._collision_exception_ids.add(body_id)

    def check_collisions(self) -> bool:
        """Check for collisions using cached body IDs for fast lookup."""
        if self.data.ncon == 0:
            return True
            
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            # Skip if either body is in exceptions (O(1) set lookup)
            if body1 in self._collision_exception_ids or body2 in self._collision_exception_ids:
                continue
            
            # Check if collision involves robot and environment
            b1_is_robot = body1 in self._collision_body_ids
            b2_is_robot = body2 in self._collision_body_ids
            
            # Skip self-collisions and non-robot collisions
            if b1_is_robot == b2_is_robot:
                continue
                
            if contact.dist < -1e-4:
                return False
                
        return True

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
    
    def is_collision_free_no_restore(self, configuration: np.ndarray) -> bool:
        """Check collision without saving/restoring state. Faster for batch checks."""
        self.data.qpos[:7] = configuration
        self.data.qvel[:7] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.check_collisions()
    
    def is_path_collision_free(self, config1: np.ndarray, config2: np.ndarray, steps: int = 5) -> bool:
        """Check if path between two configs is collision-free. Optimized batch version."""
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        
        result = True
        for i in range(steps + 1):
            alpha = i / steps
            config = (1 - alpha) * config1 + alpha * config2
            if not self.is_collision_free_no_restore(config):
                result = False
                break
        
        # Restore state once at the end
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return result

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
    
    def set_object_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray = None):
        """
        Set pose of a free body in the scene.
        
        Args:
            body_name: Name of the body
            pos: Position [x, y, z]
            quat: Quaternion [w, x, y, z], defaults to identity
        """
        if quat is None:
            quat = np.array([1, 0, 0, 0])
        
        joint_name = f"{body_name}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id >= 0:
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[joint_qposadr:joint_qposadr+3] = pos
            self.data.qpos[joint_qposadr+3:joint_qposadr+7] = quat
            self.data.qvel[self.model.jnt_dofadr[joint_id]:self.model.jnt_dofadr[joint_id]+6] = 0
    
    def get_object_pose(self, body_name: str) -> tuple:
        """
        Get pose of a body in the scene.
        
        Args:
            body_name: Name of the body
            
        Returns:
            Tuple of (position, quaternion) or (None, None) if not found
        """
        joint_name = f"{body_name}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id >= 0:
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            pos = self.data.qpos[joint_qposadr:joint_qposadr+3].copy()
            quat = self.data.qpos[joint_qposadr+3:joint_qposadr+7].copy()
            return pos, quat
        
        return None, None
    
    def reset_velocities(self):
        """Zero out all velocities in the scene."""
        self.data.qvel[:] = 0
    
    def forward(self):
        """Forward kinematics to update derived quantities."""
        mujoco.mj_forward(self.model, self.data)
    
    def close(self):
        """Close the environment and release resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
