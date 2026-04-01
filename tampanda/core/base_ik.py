"""Base inverse kinematics class."""

from abc import ABC, abstractmethod
import numpy as np


class BaseIK(ABC):
    """Abstract base class for inverse kinematics solvers."""

    @abstractmethod
    def set_target_position(self, pos: np.ndarray, quat: np.ndarray):
        """Set the target end-effector position and orientation."""
        pass

    @abstractmethod
    def converge_ik(self, dt: float) -> bool:
        """
        Run IK solver to converge to target.
        
        Returns:
            True if converged, False otherwise.
        """
        pass

    @abstractmethod
    def update_configuration(self, qpos: np.ndarray):
        """Update the current robot configuration."""
        pass
