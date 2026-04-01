"""Base environment class for robot manipulation."""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnvironment(ABC):
    """Abstract base class for robot environments."""

    @abstractmethod
    def get_model(self):
        """Return the MuJoCo model."""
        pass

    @abstractmethod
    def get_data(self):
        """Return the MuJoCo data."""
        pass

    @abstractmethod
    def get_ik(self):
        """Return the inverse kinematics solver."""
        pass

    @abstractmethod
    def launch_viewer(self):
        """Launch the MuJoCo viewer."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the environment to initial state."""
        pass

    @abstractmethod
    def step(self):
        """Step the simulation forward."""
        pass

    @abstractmethod
    def is_collision_free(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free."""
        pass

    @abstractmethod
    def get_object_position(self, object_name: str) -> np.ndarray:
        """Get the position of an object in the scene."""
        pass

    @abstractmethod
    def get_object_orientation(self, object_name: str) -> np.ndarray:
        """Get the orientation of an object in the scene."""
        pass

    def gravity_compensated_target(self, goal_q: np.ndarray) -> np.ndarray:
        """Return a ctrl target adjusted for gravity-induced steady-state error.

        The default implementation is a no-op (returns goal_q unchanged).
        Subclasses for specific robots should override this with an empirically
        calibrated implementation.
        """
        return goal_q
