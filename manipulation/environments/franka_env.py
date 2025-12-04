"""Franka Panda robot environment implementation."""

import cv2
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
        
        # Camera rendering
        self.renderer = None
        self._camera_width = 640
        self._camera_height = 480

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
    
    def _ensure_renderer(self, width: Optional[int] = None, height: Optional[int] = None):
        """
        Lazy initialize renderer with specified dimensions.
        
        Args:
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
        """
        if width is not None:
            self._camera_width = width
        if height is not None:
            self._camera_height = height
        
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self._camera_height,
                width=self._camera_width
            )
    
    def _get_camera_intrinsics(self, camera_name: str, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Calculate camera intrinsic parameters for pixel-to-3D projection.
        
        Args:
            camera_name: Name of the camera
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (fx, fy, cx, cy) where fx, fy are focal lengths and cx, cy is principal point
        """
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")
        
        fovy_rad = np.deg2rad(self.model.cam_fovy[camera_id])
        
        # Calculate focal length in pixels from field of view
        fy = height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy  # Assuming square pixels (aspect ratio 1:1)
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        return fx, fy, cx, cy
    
    def render_camera(self, camera_name: str, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Capture RGB image from specified camera.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            RGB image as numpy array with shape (height, width, 3) and dtype uint8
        """
        self._ensure_renderer(width, height)
        
        # Disable depth rendering to get RGB
        self.renderer.disable_depth_rendering()
        
        # Update scene and render
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        
        return rgb
    
    def render_depth(self, camera_name: str, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Capture depth image from specified camera.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Depth image as numpy array with shape (height, width) and dtype float32.
            Values are in meters. Invalid depth is represented as infinity or NaN.
        """
        self._ensure_renderer(width, height)
        
        # Enable depth rendering and update scene
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        depth = self.renderer.render()
        
        # Disable depth rendering to restore normal state
        self.renderer.disable_depth_rendering()
        
        return depth
    
    def save_camera_image(self, camera_name: str, filepath: str, width: int = 640, height: int = 480):
        """
        Save RGB camera image to PNG file.
        
        Args:
            camera_name: Name of the camera to render from
            filepath: Output filepath (should end with .png)
            width: Image width in pixels
            height: Image height in pixels
        """
        rgb = self.render_camera(camera_name, width, height)
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Save to file
        success = cv2.imwrite(filepath, bgr)
        if not success:
            raise IOError(f"Failed to save image to {filepath}")
    
    def get_pointcloud(
        self,
        camera_name: str,
        width: int = 640,
        height: int = 480,
        min_depth: float = 0.1,
        max_depth: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pointcloud from RGB-D data with depth filtering.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels
            height: Image height in pixels
            min_depth: Minimum valid depth in meters (default: 0.1)
            max_depth: Maximum valid depth in meters (default: 2.0)
            
        Returns:
            Tuple of (points, colors) where:
            - points: Nx3 numpy array of 3D points in world coordinates
            - colors: Nx3 numpy array of RGB colors (uint8, range 0-255)
        """
        # Capture RGB and depth
        rgb = self.render_camera(camera_name, width, height)
        depth = self.render_depth(camera_name, width, height)
        
        # Get camera intrinsics
        fx, fy, cx, cy = self._get_camera_intrinsics(camera_name, width, height)
        
        # Create pixel coordinate grids
        u = np.arange(width)
        v = np.arange(height)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Flatten arrays
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        depth_flat = depth.flatten()
        rgb_flat = rgb.reshape(-1, 3)
        
        # Filter valid depths
        valid_mask = (
            np.isfinite(depth_flat) &
            (depth_flat >= min_depth) &
            (depth_flat <= max_depth)
        )
        
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        colors_valid = rgb_flat[valid_mask]
        
        # Unproject to camera coordinates
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        # Stack into Nx3 array (camera coordinates)
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Get camera pose to transform to world coordinates
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")
        
        # Get camera position and orientation in world frame
        cam_pos = self.data.cam_xpos[camera_id]
        cam_mat = self.data.cam_xmat[camera_id].reshape(3, 3)
        
        # Transform points from camera to world coordinates
        # Note: MuJoCo camera looks down -Z axis in camera frame
        points_world = (cam_mat @ points_cam.T).T + cam_pos
        
        return points_world, colors_valid
    
    def close(self):
        """Close the environment and release resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
