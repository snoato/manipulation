"""Observation and action space builders for TampandaGymEnv."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from gymnasium import spaces


VALID_OBS_KEYS = frozenset({
    "joints", "joint_vel", "ee_pose", "object_poses",
    "rgb", "depth", "pointcloud", "segmented_pointcloud", "multi_pointcloud",
})


def build_observation_space(
    obs_keys: List[str],
    n_objects: int = 0,
    image_size: Tuple[int, int] = (84, 84),
    n_points: int = 1024,
) -> spaces.Dict:
    """Build a Dict observation space from a list of obs key names.

    Args:
        obs_keys: Subset of ``VALID_OBS_KEYS``.
        n_objects: Number of manipulable objects (for ``"object_poses"`` shape).
        image_size: ``(height, width)`` in pixels.
        n_points: Fixed point count for all pointcloud keys.

    Returns:
        ``gymnasium.spaces.Dict`` with one sub-space per key.
    """
    h, w = image_size
    space_map: dict = {}

    for key in obs_keys:
        if key == "joints":
            space_map[key] = spaces.Box(-np.pi, np.pi, (7,), np.float32)
        elif key == "joint_vel":
            space_map[key] = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        elif key == "ee_pose":
            # pos (3) + quat (4) = 7
            space_map[key] = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        elif key == "object_poses":
            # pos (3) + quat (4) per object
            space_map[key] = spaces.Box(-np.inf, np.inf, (n_objects * 7,), np.float32)
        elif key == "rgb":
            space_map[key] = spaces.Box(0, 255, (h, w, 3), np.uint8)
        elif key == "depth":
            space_map[key] = spaces.Box(0.0, np.inf, (h, w), np.float32)
        elif key == "pointcloud":
            space_map[key] = spaces.Box(-np.inf, np.inf, (n_points, 3), np.float32)
        elif key == "segmented_pointcloud":
            # xyz (3) + integer segment label as float (1)
            space_map[key] = spaces.Box(-np.inf, np.inf, (n_points, 4), np.float32)
        elif key == "multi_pointcloud":
            space_map[key] = spaces.Box(-np.inf, np.inf, (n_points, 3), np.float32)
        else:
            raise ValueError(
                f"Unknown obs key {key!r}. Valid keys: {sorted(VALID_OBS_KEYS)}"
            )

    return spaces.Dict(space_map)


def build_action_space(action_space_type: str, include_gripper: bool) -> spaces.Box:
    """Build a Box action space.

    Args:
        action_space_type: ``"joint_delta"``, ``"joint_target"``, or
            ``"cartesian_delta"`` (3 pos + 3 rot axis-angle).
        include_gripper: If True, append one dimension in ``[-1, 1]``
            where ``-1`` = close and ``1`` = open.

    Returns:
        ``gymnasium.spaces.Box`` with all values in ``[-1, 1]``.
    """
    if action_space_type in ("joint_delta", "joint_target"):
        n_arm = 7
    elif action_space_type == "cartesian_delta":
        n_arm = 6  # 3 position + 3 axis-angle rotation
    else:
        raise ValueError(
            f"Unknown action_space_type {action_space_type!r}. "
            "Valid: 'joint_delta', 'joint_target', 'cartesian_delta'."
        )

    n_total = n_arm + (1 if include_gripper else 0)
    return spaces.Box(-1.0, 1.0, (n_total,), np.float32)
