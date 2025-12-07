"""
Blocks world domain with continuous working area (no grid discretization).

This module provides the BlocksDomain class that defines a restricted working
area on a tabletop for blocks world manipulation without spatial discretization.
"""

import numpy as np
from typing import Dict, Any, Tuple
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass

from manipulation.symbolic.base_domain import BaseDomain


class BlocksDomain(BaseDomain):
    """
    Defines a continuous working area on a tabletop for blocks world manipulation.
    
    Unlike the GridDomain which discretizes space into cells, this domain maintains
    continuous XY coordinates while restricting the working area to a specified
    region of the table. This is suitable for blocks world where precise positions
    matter less than stacking relationships.
    
    Attributes:
        model: MuJoCo model containing the table geometry
        working_area: Tuple of (width_x, width_y) in meters
        offset_x: Working area offset in x-direction (meters)
        offset_y: Working area offset in y-direction (meters)
        table_bounds: Dict with min_x, max_x, min_y, max_y, table_height
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        working_area: Tuple[float, float] = (0.4, 0.4),
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        table_body_name: str = "simple_table",
        table_geom_name: str = "table_surface"
    ):
        """
        Initialize blocks domain from MuJoCo model.
        
        Args:
            model: MuJoCo model containing table geometry
            working_area: (width_x, width_y) of working area in meters
            offset_x: Offset in x-direction in meters (default 0.0).
                     Positive values shift area right, negative shift left.
            offset_y: Offset in y-direction in meters (default 0.0).
                     Positive values shift area away from robot, negative toward robot.
            table_body_name: Name of the table body in MuJoCo model
            table_geom_name: Name of the table surface geom
        """
        self.model = model
        self.working_area = working_area
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.table_body_name = table_body_name
        self.table_geom_name = table_geom_name
        
        # Create temporary data to get global positions
        self._temp_data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._temp_data)
        
        # Parse table geometry from model
        self.table_bounds = self._parse_table_geometry()
        self.table_height = self.table_bounds['table_height']
    
    def _parse_table_geometry(self) -> Dict[str, float]:
        """
        Extract table bounds from MuJoCo model.
        
        Looks for the specified table geom to determine the working area center.
        
        Returns:
            Dictionary with keys: min_x, max_x, min_y, max_y, table_height
        """
        # Find table geom
        table_geom_id = None
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name == self.table_geom_name:
                table_geom_id = i
                break
        
        if table_geom_id is None:
            raise ValueError(f"Could not find '{self.table_geom_name}' geom in model")
        
        # Get geom global position
        geom_xpos = self._temp_data.geom_xpos[table_geom_id]
        geom_size = self.model.geom_size[table_geom_id]
        
        # Center working area on long axis (X), align to robot-facing edge on short axis (Y)
        # X: centered on table, then apply offset
        center_x = geom_xpos[0] + self.offset_x
        # Y: aligned with front edge (robot-facing side), then apply offset
        center_y = geom_xpos[1] - geom_size[1] + self.working_area[1] / 2.0 + self.offset_y
        
        table_height = geom_xpos[2] + geom_size[2]  # top of table
        
        half_width_x = self.working_area[0] / 2.0
        half_width_y = self.working_area[1] / 2.0
        
        return {
            'min_x': center_x - half_width_x,
            'max_x': center_x + half_width_x,
            'min_y': center_y - half_width_y,
            'max_y': center_y + half_width_y,
            'table_height': table_height
        }
    
    def get_working_bounds(self) -> Dict[str, float]:
        """
        Get the working area bounds.
        
        Returns:
            Dictionary with keys: min_x, max_x, min_y, max_y, table_height
        """
        return self.table_bounds.copy()
    
    def is_in_bounds(self, x: float, y: float) -> bool:
        """
        Check if a position is within the working area bounds.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            
        Returns:
            True if position is within bounds, False otherwise
        """
        return (self.table_bounds['min_x'] <= x <= self.table_bounds['max_x'] and
                self.table_bounds['min_y'] <= y <= self.table_bounds['max_y'])
    
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the domain configuration.
        
        Returns:
            Dictionary containing domain-specific configuration details
        """
        return {
            'type': 'blocks_world',
            'working_area': self.working_area,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'table_bounds': self.table_bounds,
            'table_height': self.table_height
        }
    
    def get_location_at_position(self, x: float, y: float) -> str:
        """
        Get the symbolic location identifier for a continuous position.
        
        For blocks world with no discretization, this returns a string
        representation of the continuous coordinates.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
        
        Returns:
            Location string in format "pos_X_Y" (rounded to mm)
        
        Raises:
            ValueError: If position is out of bounds
        """
        if not self.is_in_bounds(x, y):
            raise ValueError(f"Position ({x:.3f}, {y:.3f}) is outside working bounds")
        
        # Return position rounded to millimeter precision
        return f"pos_{int(x*1000)}_{int(y*1000)}"
    
    def get_location_center(self, location_id: str) -> Tuple[float, float]:
        """
        Get the (x, y) center coordinates of a location.
        
        For continuous positions, this parses the location string back to coordinates.
        
        Args:
            location_id: Location identifier string in format "pos_X_Y" (mm precision)
        
        Returns:
            Tuple of (x, y) coordinates in meters
        
        Raises:
            ValueError: If location_id format is invalid
        """
        if not location_id.startswith("pos_"):
            raise ValueError(f"Invalid location ID format: {location_id}")
        
        try:
            parts = location_id.split("_")
            x_mm = int(parts[1])
            y_mm = int(parts[2])
            return (x_mm / 1000.0, y_mm / 1000.0)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Cannot parse location ID: {location_id}") from e
