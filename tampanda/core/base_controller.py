"""Base controller class."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import enum


class ControllerStatus(enum.Enum):
    """Controller status enumeration."""
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    GRASPING = "grasping"


class BaseController(ABC):
    """Abstract base class for robot controllers."""

    @abstractmethod
    def step(self):
        """Execute one control step."""
        pass

    @abstractmethod
    def move_to(self, configuration: np.ndarray):
        """Move to a target configuration."""
        pass

    @abstractmethod
    def follow_trajectory(self, trajectory: List[np.ndarray]):
        """Follow a trajectory of configurations."""
        pass

    @abstractmethod
    def get_status(self) -> ControllerStatus:
        """Get the current controller status."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the controller."""
        pass

    @abstractmethod
    def open_gripper(self):
        """Open the gripper."""
        pass

    @abstractmethod
    def close_gripper(self):
        """Close the gripper."""
        pass
