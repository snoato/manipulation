"""Base motion planner class."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseMotionPlanner(ABC):
    """Abstract base class for motion planners."""

    @abstractmethod
    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None
    ) -> Optional[List[np.ndarray]]:
        """
        Plan a collision-free path from start to goal.
        
        Returns:
            List of configurations forming a path, or None if planning fails.
        """
        pass

    @abstractmethod
    def smooth_path(
        self,
        path: List[np.ndarray],
        max_iterations: int = 100
    ) -> List[np.ndarray]:
        """Smooth a path by removing unnecessary waypoints."""
        pass

    @abstractmethod
    def is_path_collision_free(
        self,
        config1: np.ndarray,
        config2: np.ndarray,
        steps: int = 10
    ) -> bool:
        """Check if a straight-line path between two configs is collision-free."""
        pass
