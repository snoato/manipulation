"""
Base classes for symbolic planning domain integration.

This module defines abstract interfaces that all symbolic planning domains
should implement, enabling modular integration of different planning problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path


class BaseDomain(ABC):
    """
    Abstract base class for spatial domain discretization.
    
    A domain provides a discretized representation of continuous space
    for symbolic planning. Different domains can represent different
    types of problems (e.g., grid-based tabletop, hierarchical blocks).
    """
    
    @abstractmethod
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the domain configuration.
        
        Returns:
            Dictionary containing domain-specific configuration details
        """
        pass
    
    @abstractmethod
    def get_location_at_position(self, x: float, y: float) -> str:
        """
        Get the symbolic location identifier for a continuous position.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
        
        Returns:
            Location identifier string, or None if out of bounds
        """
        pass
    
    @abstractmethod
    def get_location_center(self, location_id: str) -> Tuple[float, float]:
        """
        Get the (x, y) center coordinates of a location.
        
        Args:
            location_id: Location identifier string
        
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        pass


class BaseStateManager(ABC):
    """
    Abstract base class for managing symbolic state and PDDL generation.
    
    A state manager bridges the gap between continuous simulation state
    and discrete symbolic representations, generating PDDL problems from
    the current world state.
    """
    
    def __init__(self, domain: BaseDomain, environment):
        """
        Initialize state manager.
        
        Args:
            domain: Domain discretization instance
            environment: Simulation environment
        """
        self.domain = domain
        self.env = environment
    
    @abstractmethod
    def ground_state(self) -> Dict[str, Any]:
        """
        Extract current symbolic state from the simulation.
        
        Returns:
            Dictionary containing grounded symbolic state
        """
        pass
    
    @abstractmethod
    def generate_pddl_problem(self, problem_name: str, output_path: Path) -> None:
        """
        Generate a PDDL problem file from current state.
        
        Args:
            problem_name: Name for the PDDL problem
            output_path: Path where to save the PDDL file
        """
        pass
    
    @abstractmethod
    def sample_random_state(self, **kwargs) -> None:
        """
        Sample a random valid state in the simulation.
        
        Args:
            **kwargs: Domain-specific parameters for state generation
        """
        pass
