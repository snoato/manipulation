"""MuJoCo camera rendering and pointcloud generation utilities."""

import cv2
import mujoco
import numpy as np
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from manipulation.core.base_env import BaseEnvironment


class MujocoCamera:
    """
    Camera utilities for rendering RGB, depth, segmentation, and pointclouds.
    
    Handles all camera-related operations for MuJoCo simulations including:
    - RGB/depth/segmentation rendering
    - Camera intrinsics calculation
    - Point cloud generation from depth images
    - Camera pose extraction (position and orientation)
    """
    
    def __init__(
        self,
        env: "BaseEnvironment",
        width: int = 640,
        height: int = 480,
        default_excludes: Optional[List[str]] = None
    ):
        """
        Initialize camera utilities.
        
        Args:
            env: Robot environment with get_model() and get_data() methods
            width: Default image width in pixels
            height: Default image height in pixels
            default_excludes: Default patterns to exclude from segmentation
                             (e.g., robot parts, static scene). If None, uses
                             sensible defaults for Franka robot.
        """
        self.env = env
        self._width = width
        self._height = height
        self.renderer = None
        
        # Build object cache for segmentation
        if default_excludes is None:
            default_excludes = [
                # Robot body parts
                "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                "hand", "left_finger", "right_finger",
                # Static scene elements
                "floor", "world",
                # Mocap/visualization bodies
                "target_viz", "target", "mocap"
            ]
        
        self._default_excludes = default_excludes
        self._object_cache = None
        self._build_object_cache()
    
    def _ensure_renderer(self, width: Optional[int] = None, height: Optional[int] = None):
        """
        Lazy initialize renderer with specified dimensions.
        
        Args:
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
        """
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height
        
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.env.get_model(),
                height=self._height,
                width=self._width
            )
    
    def get_camera_intrinsics(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculate camera intrinsic parameters for pixel-to-3D projection.
        
        Args:
            camera_name: Name of the camera
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
            
        Returns:
            Tuple of (fx, fy, cx, cy) where fx, fy are focal lengths
            and cx, cy is principal point
        """
        width = width or self._width
        height = height or self._height
        
        model = self.env.get_model()
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")
        
        fovy_rad = np.deg2rad(model.cam_fovy[camera_id])
        
        # Calculate focal length in pixels from field of view
        fy = height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy  # Assuming square pixels (aspect ratio 1:1)
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        return fx, fy, cx, cy
    
    def render_rgb(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        Capture RGB image from specified camera.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
            
        Returns:
            RGB image as numpy array with shape (height, width, 3) and dtype uint8
        """
        self._ensure_renderer(width, height)
        
        # Disable depth rendering to get RGB
        self.renderer.disable_depth_rendering()
        
        # Update scene and render
        self.renderer.update_scene(self.env.get_data(), camera=camera_name)
        rgb = self.renderer.render()
        
        return rgb
    
    def render_depth(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        Capture depth image from specified camera.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
            
        Returns:
            Depth image as numpy array with shape (height, width) and dtype float32.
            Values are in meters. Invalid depth is represented as infinity or NaN.
        """
        self._ensure_renderer(width, height)
        
        # Enable depth rendering and update scene
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.env.get_data(), camera=camera_name)
        depth = self.renderer.render()
        
        # Disable depth rendering to restore normal state
        self.renderer.disable_depth_rendering()
        
        return depth
    
    def render_segmentation(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        """
        Capture segmentation image from specified camera.
        
        Args:
            camera_name: Name of the camera to render from
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
            
        Returns:
            Segmentation image as numpy array with shape (height, width, 2) and dtype int32.
            [:, :, 0] contains geom IDs, [:, :, 1] contains object IDs (unused, always -1).
        """
        self._ensure_renderer(width, height)
        
        # Temporarily enable segmentation rendering
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.env.get_data(), camera=camera_name)
        seg = self.renderer.render()
        
        # Disable segmentation rendering to restore normal state
        self.renderer.disable_segmentation_rendering()
        
        return seg
    
    def save_image(
        self,
        camera_name: str,
        filepath: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ):
        """
        Save RGB camera image to PNG file.
        
        Args:
            camera_name: Name of the camera to render from
            filepath: Output filepath (should end with .png)
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
        """
        rgb = self.render_rgb(camera_name, width, height)
        
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Save to file
        success = cv2.imwrite(filepath, bgr)
        if not success:
            raise IOError(f"Failed to save image to {filepath}")
    
    def _get_camera_pose(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and rotation matrix in world frame.
        
        Args:
            camera_name: Name of the camera
            
        Returns:
            Tuple of (position, rotation_matrix) where:
            - position is 3D vector (x, y, z) in meters
            - rotation_matrix is 3x3 rotation matrix from camera to world frame
        """
        model = self.env.get_model()
        data = self.env.get_data()
        
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in the model.")
        
        # Get camera position (3D vector)
        cam_pos = data.cam_xpos[camera_id].copy()
        
        # Get camera rotation matrix (3x3 matrix, stored as flat 9 elements)
        cam_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()
        
        return cam_pos, cam_mat
    
    def get_pointcloud(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_samples: int = 1000,
        min_depth: float = 0.3,
        max_depth: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point cloud from camera depth image.
        
        Samples points from depth map, unprojects to 3D using pinhole camera model,
        and transforms to world frame using camera pose.
        """
        # Capture RGB and depth images
        rgb = self.render_rgb(camera_name, width, height)
        depth = self.render_depth(camera_name, width, height)
        
        img_height, img_width = depth.shape
        
        # Get camera intrinsics
        fx, fy, cx, cy = self.get_camera_intrinsics(camera_name, img_width, img_height)
        
        # Create pixel coordinate grids
        u_coords, v_coords = np.meshgrid(
            np.arange(img_width, dtype=np.float32),
            np.arange(img_height, dtype=np.float32),
            indexing='xy'
        )
        
        # Flatten arrays
        u_flat = u_coords.ravel()
        v_flat = v_coords.ravel()
        depth_flat = depth.ravel()
        rgb_flat = rgb.reshape(-1, 3)
        
        # Filter valid depths
        valid_mask = (depth_flat >= min_depth) & (depth_flat <= max_depth) & np.isfinite(depth_flat)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        
        # Random sampling to reduce point count
        if len(valid_indices) > num_samples:
            sampled_indices = np.random.choice(valid_indices, size=num_samples, replace=False)
        else:
            sampled_indices = valid_indices
        
        # Get sampled data
        u_sampled = u_flat[sampled_indices]
        v_sampled = v_flat[sampled_indices]
        depth_sampled = depth_flat[sampled_indices]
        colors_sampled = rgb_flat[sampled_indices]
        
        # 1. Unproject to "Standard Vision" frame (Z-forward, Y-down, X-right)
        # We use positive depth here. We will correct the frame orientation later using matrices.
        x_cam = (u_sampled - cx) * depth_sampled / fx
        y_cam = (v_sampled - cy) * depth_sampled / fy
        z_cam = depth_sampled 
        
        # Stack into Nx3 array
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # 2. Get camera pose in world frame
        cam_pos, cam_rot = self._get_camera_pose(camera_name)
        
        # 3. Define the coordinate correction matrix
        # Rotates "Vision" frame (Z-forward, Y-down) to "MuJoCo/OpenGL" frame (Z-backward, Y-up)
        # This is a 180-degree rotation around the X-axis.
        R_correction = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)
        
        # 4. Transform to World Frame
        # Formula: P_world = R_mujoco * (R_correction * P_vision) + t_mujoco
        
        # Combine rotations (more efficient than rotating points twice)
        R_total = cam_rot @ R_correction
        
        # Apply transformation
        points_world = (R_total @ points_cam.T).T + cam_pos
        
        return points_world.astype(np.float32), colors_sampled.astype(np.uint8)
    
    def get_segmented_pointcloud(
        self,
        camera_name: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_samples: int = 1000,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate segmented point clouds for each object in the scene.
        
        Args:
            camera_name: Name of the camera
            width: Image width in pixels (uses default if None)
            height: Image height in pixels (uses default if None)
            num_samples: Maximum number of points to sample per object
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            exclude_patterns: Additional patterns to exclude from results
            
        Returns:
            Dictionary mapping object name to (points, colors) tuple.
            points: Nx3 float32 array of world coordinates
            colors: Nx3 uint8 array of RGB values
        """
        # Capture RGB, depth, and segmentation
        rgb = self.render_rgb(camera_name, width, height)
        depth = self.render_depth(camera_name, width, height)
        seg = self.render_segmentation(camera_name, width, height)
        
        img_height, img_width = depth.shape
        
        # Get camera intrinsics and pose
        fx, fy, cx, cy = self.get_camera_intrinsics(camera_name, img_width, img_height)
        cam_pos, cam_rot = self._get_camera_pose(camera_name)
        
        # Create pixel coordinate grids
        u_coords, v_coords = np.meshgrid(
            np.arange(img_width, dtype=np.float32),
            np.arange(img_height, dtype=np.float32),
            indexing='xy'
        )
        
        # Flatten arrays
        u_flat = u_coords.ravel()
        v_flat = v_coords.ravel()
        depth_flat = depth.ravel()
        rgb_flat = rgb.reshape(-1, 3)
        geom_ids_flat = seg[:, :, 0].ravel()
        
        # Filter valid depths
        valid_mask = (depth_flat >= min_depth) & (depth_flat <= max_depth) & np.isfinite(depth_flat)
        
        # Result dictionary
        segmented_clouds = {}
        
        # Identify unique geoms in the valid depth range
        present_geoms = np.unique(geom_ids_flat[valid_mask])
        
        # Pre-calculate coordinate correction matrix
        R_correction = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)
        R_total = cam_rot @ R_correction
        
        for geom_id in present_geoms:
            # Check if excluded
            if geom_id in self._object_cache['excluded_geoms']:
                continue
                
            # Get object name
            body_name = self._object_cache['geom_to_name'].get(geom_id)
            if not body_name:
                continue
                
            # Check exclude patterns
            if exclude_patterns and any(p in body_name for p in exclude_patterns):
                continue
            
            # Get indices for this geom
            geom_mask = (geom_ids_flat == geom_id) & valid_mask
            geom_indices = np.where(geom_mask)[0]
            
            if len(geom_indices) == 0:
                continue
            
            # Extract data
            u_sampled = u_flat[geom_indices]
            v_sampled = v_flat[geom_indices]
            depth_sampled = depth_flat[geom_indices]
            colors_sampled = rgb_flat[geom_indices]
            
            # Unproject to camera frame
            x_cam = (u_sampled - cx) * depth_sampled / fx
            y_cam = (v_sampled - cy) * depth_sampled / fy
            z_cam = depth_sampled;
            
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=1);
            
            # Transform to world frame
            points_world = (R_total @ points_cam.T).T + cam_pos;
            
            # Add to result (accumulate if multiple geoms belong to same body)
            if body_name in segmented_clouds:
                prev_pts, prev_cols = segmented_clouds[body_name]
                segmented_clouds[body_name] = (
                    np.vstack([prev_pts, points_world.astype(np.float32)]),
                    np.vstack([prev_cols, colors_sampled.astype(np.uint8)])
                )
            else:
                segmented_clouds[body_name] = (
                    points_world.astype(np.float32),
                    colors_sampled.astype(np.uint8)
                )
        
        # Final resampling to respect num_samples per object
        for name in list(segmented_clouds.keys()):
            pts, cols = segmented_clouds[name]
            if len(pts) > num_samples:
                indices = np.random.choice(len(pts), size=num_samples, replace=False)
                segmented_clouds[name] = (pts[indices], cols[indices])
                
        return segmented_clouds
    
    def get_multi_camera_segmented_pointcloud(
        self,
        camera_names: List[str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_samples_per_camera: int = 1000,
        min_depth: float = 0.3,
        max_depth: float = 3.0,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate and merge segmented point clouds from multiple cameras.
        
        Args:
            camera_names: List of camera names to use
            width: Image width in pixels
            height: Image height in pixels
            num_samples_per_camera: Max points per object per camera
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters
            exclude_patterns: Additional patterns to exclude
            
        Returns:
            Dictionary mapping object name to (points, colors) tuple.
        """
        merged_clouds = {}
        
        for cam_name in camera_names:
            try:
                clouds = self.get_segmented_pointcloud(
                    cam_name, width, height, num_samples_per_camera,
                    min_depth, max_depth, exclude_patterns
                )
                
                for obj_name, (points, colors) in clouds.items():
                    if obj_name in merged_clouds:
                        prev_pts, prev_cols = merged_clouds[obj_name]
                        merged_clouds[obj_name] = (
                            np.vstack([prev_pts, points]),
                            np.vstack([prev_cols, colors])
                        )
                    else:
                        merged_clouds[obj_name] = (points, colors)
            except ValueError as e:
                print(f"Warning: Skipping camera '{cam_name}': {e}")
                    
        return merged_clouds
    
    def _build_object_cache(self):
        """
        Build a cache of object information for efficient segmentation.
        Pre-computes geom-to-body mappings and object names.
        """
        self._object_cache = {
            'geom_to_body': {},  # geom_id -> body_id
            'geom_to_name': {},  # geom_id -> body_name
            'body_to_name': {},  # body_id -> body_name
            'segmentable_objects': set(),  # Set of body names that are segmentable
            'excluded_geoms': set(),  # Set of geom IDs to exclude
        }
        
        model = self.env.get_model()
        
        # Build geom mappings
        for geom_id in range(model.ngeom):
            body_id = model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            self._object_cache['geom_to_body'][geom_id] = body_id
            self._object_cache['geom_to_name'][geom_id] = body_name
            self._object_cache['body_to_name'][body_id] = body_name
            
            # Check if this body should be excluded
            if body_name and any(exclude in body_name for exclude in self._default_excludes):
                self._object_cache['excluded_geoms'].add(geom_id)
            elif body_name:  # Valid, non-excluded body
                self._object_cache['segmentable_objects'].add(body_name)
    
    def rebuild_object_cache(self):
        """
        Rebuild object cache (useful after scene changes or resets).
        """
        self._build_object_cache()
    
    def list_segmentable_objects(
        self,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """
        List all objects that can be segmented from pointclouds.
        
        Args:
            exclude_patterns: Additional patterns to exclude (e.g., ['table', 'wall']).
                             By default, robot parts and static scene elements are excluded.
        
        Returns:
            List of object body names that are available for segmentation
        """
        segmentable = self._object_cache['segmentable_objects'].copy()
        
        # Apply additional exclusions if provided
        if exclude_patterns:
            filtered = set()
            for obj_name in segmentable:
                if not any(pattern in obj_name for pattern in exclude_patterns):
                    filtered.add(obj_name)
            segmentable = filtered
        
        return sorted(list(segmentable))
    
    def close(self):
        """Close and release renderer resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
