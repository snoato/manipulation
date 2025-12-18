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
    - Pointcloud generation from RGB-D data
    - Object-segmented pointcloud extraction
    - Camera intrinsics calculation
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
