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
from manipulation.controllers.position_controller import PositionController, ControllerStatus
from manipulation.utils.rate_limiter import RateLimiter


# Empirically measured effective PD gain per joint for the Franka MuJoCo model.
# Discrete-time actuator dynamics reduce apparent stiffness below the XML gainprm
# values. Joint 1 (base rotation) has zero gravity torque so its gain is inf.
_EFF_KP = np.array([np.inf, 1000., 750., 750., 300., 300., 300.])


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion [w, x, y, z] (Shepperd's method)."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (mat[2, 1] - mat[1, 2]) * s,
                         (mat[0, 2] - mat[2, 0]) * s,
                         (mat[1, 0] - mat[0, 1]) * s])
    if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        return np.array([(mat[2, 1] - mat[1, 2]) / s, 0.25 * s,
                         (mat[0, 1] + mat[1, 0]) / s,
                         (mat[0, 2] + mat[2, 0]) / s])
    if mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        return np.array([(mat[0, 2] - mat[2, 0]) / s,
                         (mat[0, 1] + mat[1, 0]) / s, 0.25 * s,
                         (mat[1, 2] + mat[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
    return np.array([(mat[1, 0] - mat[0, 1]) / s,
                     (mat[0, 2] + mat[2, 0]) / s,
                     (mat[1, 2] + mat[2, 1]) / s, 0.25 * s])


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

        # Object attachment state (for kinematic gripper-object coupling)
        self._attached: Optional[dict] = None
        self._ee_site_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

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
        self.controller.stop()
        self._attached = None
        self.sim_time = 0.0

    def step(self):
        if self._attached is not None:
            self._apply_attachment()
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

    def get_object_half_size(self, body_name: str) -> np.ndarray:
        """Return half-extents [hx, hy, hz] for the first geom of a body.

        Works for box, cylinder, capsule, and sphere geoms.  For cylinders/
        capsules the radial dimension is used for both X and Y.
        """
        body_id = self.get_object_id(body_name)
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != body_id:
                continue
            gtype = self.model.geom_type[geom_id]
            size  = self.model.geom_size[geom_id]
            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                return size.copy()
            elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER,
                           mujoco.mjtGeom.mjGEOM_CAPSULE):
                return np.array([size[0], size[0], size[1]])
            else:  # sphere or other — treat all dims as size[0]
                return np.array([size[0], size[0], size[0]])
        raise ValueError(f"No geom found for body '{body_name}'")

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
    
    def execute_path(self, path, planner, step_size: float = 0.05):
        """Smooth, interpolate, and execute a planned path with gravity compensation.

        Every waypoint is gravity-compensated so that ctrl never drops between
        trajectory segments (which would cause the arm to sag).  The compensation
        is computed in a single save/restore cycle (O(n) mj_forward calls but no
        repeated state save/restore overhead).

        Args:
            path: List of joint configurations from the planner.
            planner: The RRTStar instance (used for smoothing).
            step_size: Interpolation step size in joint space (radians).
        """
        smoothed = planner.smooth_path(path)
        waypoints = self.controller.interpolate_linear_path(smoothed, step_size=step_size)
        traj = self._compensate_trajectory(waypoints)
        # Pass uncompensated waypoints separately so the advance condition
        # checks |wp - qpos| < delta rather than |ctrl_ref - qpos| < delta.
        # This makes the advance delta-independent of gravity magnitude.
        self.controller.follow_trajectory(traj, waypoints=waypoints)

    def wait_idle(self, max_steps: int = 5000, settle_steps: int = 200) -> bool:
        """Block until the controller reaches IDLE, then run extra settling steps.

        Each step advances both the controller and the physics simulation.
        Returns True if IDLE was reached within max_steps, False on timeout.
        """
        for _ in range(max_steps):
            self.controller.step()
            self.step()
            if self.controller.get_status() == ControllerStatus.IDLE:
                for _ in range(settle_steps):
                    self.step()
                return True
        self.step()  # one last step so the caller is not stuck
        return False

    def attach_object_to_ee(self, body_name: str):
        """Kinematically attach a free body to the end-effector.

        On every subsequent call to step(), the body is teleported to maintain
        its pose relative to the EE site at the moment of attachment.  This
        prevents slip during lifting without requiring a MuJoCo equality
        constraint defined in the XML.

        Call detach_object() before opening the gripper.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")

        joint_name = f"{body_name}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Free joint '{joint_name}' not found in model")

        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        ee_mat = self.data.site_xmat[self._ee_site_id].reshape(3, 3).copy()

        obj_pos = self.data.xpos[body_id].copy()
        obj_mat = self.data.xmat[body_id].reshape(3, 3).copy()

        # Relative pose in EE frame
        rel_pos = ee_mat.T @ (obj_pos - ee_pos)
        rel_mat = ee_mat.T @ obj_mat

        self._attached = {
            "body_name": body_name,
            "joint_id":  joint_id,
            "qadr":      self.model.jnt_qposadr[joint_id],
            "vadr":      self.model.jnt_dofadr[joint_id],
            "rel_pos":   rel_pos,
            "rel_mat":   rel_mat,
        }

    def detach_object(self):
        """Remove the kinematic attachment set by attach_object_to_ee()."""
        self._attached = None

    def _apply_attachment(self):
        """Update the attached body's pose to track the EE. Called inside step()."""
        a = self._attached
        ee_pos = self.data.site_xpos[self._ee_site_id]
        ee_mat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)

        new_pos  = ee_pos + ee_mat @ a["rel_pos"]
        new_mat  = ee_mat @ a["rel_mat"]
        new_quat = _mat2quat(new_mat)

        qadr = a["qadr"]
        self.data.qpos[qadr:qadr + 3] = new_pos
        self.data.qpos[qadr + 3:qadr + 7] = new_quat
        vadr = a["vadr"]
        self.data.qvel[vadr:vadr + 6] = 0.0

    def _compensate_trajectory(self, traj: list) -> list:
        """Apply gravity compensation to every waypoint in a single save/restore cycle.

        Without this, only the last waypoint holds the arm correctly against
        gravity.  When a new trajectory starts, the PD controller's ctrl drops
        from the compensated hold value to the uncompensated first waypoint,
        causing a transient sag of up to 14 mm (measured empirically).
        """
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        self.data.qvel[:] = 0.0
        result = []
        for wp in traj:
            self.data.qpos[:7] = wp
            mujoco.mj_forward(self.model, self.data)
            result.append(wp + self.data.qfrc_bias[:7] / _EFF_KP)

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_forward(self.model, self.data)
        return result

    def move_to_compensated(self, goal_q: np.ndarray, step_size: float = 0.01):
        """Interpolate from current qpos to goal_q with per-waypoint gravity compensation.

        IK-friendly equivalent of execute_path: produces a dense trajectory where
        every waypoint is individually compensated, so the arm never sags mid-move.
        The uncompensated waypoints are stored separately in the controller so the
        advance condition is delta-independent of gravity magnitude.

        Args:
            goal_q: Target joint configuration (7-DOF).
            step_size: Interpolation step size in joint space (radians).
        """
        start = self.data.qpos[:7].copy()
        waypoints = self.controller.interpolate_linear_points(start, goal_q, step_size)
        traj = self._compensate_trajectory(waypoints)
        self.controller.follow_trajectory(traj, waypoints=waypoints)

    def gravity_compensated_target(self, goal_q: np.ndarray) -> np.ndarray:
        """Return the ctrl target that makes the robot settle exactly at goal_q.

        Single-waypoint version of _compensate_trajectory, kept for external use
        (e.g. IK-based scripts that set ctrl directly).
        """
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        self.data.qpos[:7] = goal_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        e_ss = self.data.qfrc_bias[:7] / _EFF_KP

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_forward(self.model, self.data)
        return goal_q + e_ss

    def close(self):
        """Close the environment and release resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
