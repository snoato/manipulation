"""
Grid-based spatial discretization for symbolic planning.

This module provides the GridDomain class that discretizes a tabletop workspace
into a uniform grid of cells for PDDL-based symbolic planning.
"""

import numpy as np
from typing import Dict, List, Set, Tuple
import mujoco


class GridDomain:
    """
    Discretizes a continuous tabletop workspace into a uniform grid.
    
    The grid provides a symbolic abstraction of space for PDDL planning,
    with 1cm cell resolution and 4-way adjacency connectivity.
    
    Attributes:
        model: MuJoCo model containing the table geometry
        cells_x: Number of cells in x-direction (typically 40)
        cells_y: Number of cells in y-direction (typically 40)
        cell_size: Size of each cell in meters (0.01m = 1cm)
        table_bounds: Dict with min_x, max_x, min_y, max_y of working area
        cell_centers: Dict mapping cell IDs to (x, y) center coordinates
        cell_bounds: Dict mapping cell IDs to (min_x, max_x, min_y, max_y)
        adjacency: Dict mapping cell IDs to set of adjacent cell IDs
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        cell_size: float = 0.01,
        working_area: Tuple[float, float] = (0.4, 0.4),
        table_body_name: str = "simple_table",
        table_geom_name: str = "table_surface"
    ):
        """
        Initialize grid domain from MuJoCo model.
        
        Args:
            model: MuJoCo model containing table geometry
            cell_size: Size of each grid cell in meters (default 0.01 = 1cm)
            working_area: (width_x, width_y) of working area in meters
            table_body_name: Name of the table body in MuJoCo model
            table_geom_name: Name of the table surface geom
        """
        self.model = model
        self.cell_size = cell_size
        self.working_area = working_area
        self.table_body_name = table_body_name
        self.table_geom_name = table_geom_name
        
        # Create temporary data to get global positions
        self._temp_data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._temp_data)
        
        # Parse table geometry from model
        self.table_bounds = self._parse_table_geometry()
        self.table_height = self.table_bounds['table_height']
        
        # Compute grid dimensions
        self.cells_x = int(working_area[0] / cell_size)
        self.cells_y = int(working_area[1] / cell_size)
        
        # Generate cells with centers and bounds
        self.cells = self._generate_cells()
        
        # Precompute adjacency relationships (both legacy and directional)
        self.adjacency = self._compute_adjacency()
        self.directional_adjacency = self._compute_directional_adjacency()
    
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
        
        # Get geom global position (not local position)
        geom_xpos = self._temp_data.geom_xpos[table_geom_id]
        geom_size = self.model.geom_size[table_geom_id]
        
        # Center working area on long axis (X), align to robot-facing edge on short axis (Y)
        # X: centered on table
        center_x = geom_xpos[0]
        # Y: aligned with front edge (robot-facing side)
        # Front edge is at geom_xpos[1] - geom_size[1], shift grid center towards robot
        center_y = geom_xpos[1] - geom_size[1] + self.working_area[1] / 2.0
        
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
    
    def _generate_cells(self) -> Dict[str, Dict]:
        """
        Generate cells with centers and bounds for the grid.
        
        Returns:
            Dictionary mapping cell IDs to dict with 'center' and 'bounds' keys
        """
        cells = {}
        
        for ix in range(self.cells_x):
            for iy in range(self.cells_y):
                # Compute cell center
                center_x = self.table_bounds['min_x'] + (ix + 0.5) * self.cell_size
                center_y = self.table_bounds['min_y'] + (iy + 0.5) * self.cell_size
                
                # Compute cell bounds
                min_x = self.table_bounds['min_x'] + ix * self.cell_size
                max_x = min_x + self.cell_size
                min_y = self.table_bounds['min_y'] + iy * self.cell_size
                max_y = min_y + self.cell_size
                
                cell_id = f"cell_{ix}_{iy}"
                cells[cell_id] = {
                    'center': (center_x, center_y),
                    'bounds': (min_x, max_x, min_y, max_y)
                }
        
        return cells
    
    def _compute_adjacency(self) -> Dict[str, Set[str]]:
        """
        Precompute 4-way adjacency relationships.
        
        Each cell is connected to its neighbors in the 4 cardinal directions
        (north, south, east, west), if they exist within grid bounds.
        
        Returns:
            Dictionary mapping cell IDs to sets of adjacent cell IDs
        """
        adjacency = {}
        
        for ix in range(self.cells_x):
            for iy in range(self.cells_y):
                cell_id = f"cell_{ix}_{iy}"
                neighbors = set()
                
                # North (iy + 1)
                if iy + 1 < self.cells_y:
                    neighbors.add(f"cell_{ix}_{iy + 1}")
                
                # South (iy - 1)
                if iy - 1 >= 0:
                    neighbors.add(f"cell_{ix}_{iy - 1}")
                
                # East (ix + 1)
                if ix + 1 < self.cells_x:
                    neighbors.add(f"cell_{ix + 1}_{iy}")
                
                # West (ix - 1)
                if ix - 1 >= 0:
                    neighbors.add(f"cell_{ix - 1}_{iy}")
                
                adjacency[cell_id] = neighbors
        
        return adjacency
    
    def _compute_directional_adjacency(self) -> Dict[str, Dict[str, str]]:
        """
        Compute directional adjacency relationships between cells.
        
        Returns:
            Dict mapping each cell to a dict of {direction: neighbor_cell}.
            Example: {'cell_0_0': {'north': 'cell_0_1', 'east': 'cell_1_0'}, ...}
            Edge cells will only have valid directions (2-3 directions),
            corner cells will have exactly 2 directions.
        """
        adjacencies = {}
        
        for ix in range(self.cells_x):
            for iy in range(self.cells_y):
                cell_id = f"cell_{ix}_{iy}"
                cell_adjacencies = {}
                
                # North (increasing y)
                if iy + 1 < self.cells_y:
                    cell_adjacencies['north'] = f"cell_{ix}_{iy + 1}"
                
                # South (decreasing y)
                if iy - 1 >= 0:
                    cell_adjacencies['south'] = f"cell_{ix}_{iy - 1}"
                
                # East (increasing x)
                if ix + 1 < self.cells_x:
                    cell_adjacencies['east'] = f"cell_{ix + 1}_{iy}"
                
                # West (decreasing x)
                if ix - 1 >= 0:
                    cell_adjacencies['west'] = f"cell_{ix - 1}_{iy}"
                
                adjacencies[cell_id] = cell_adjacencies
        
        return adjacencies
    
    def get_cell_at_position(self, x: float, y: float) -> str:
        """
        Get the cell ID containing a given (x, y) position.
        
        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
        
        Returns:
            Cell ID string (e.g., 'cell_20_20'), or None if out of bounds
        """
        # Check if position is within table bounds
        if (x < self.table_bounds['min_x'] or x > self.table_bounds['max_x'] or
            y < self.table_bounds['min_y'] or y > self.table_bounds['max_y']):
            return None
        
        # Compute cell indices
        ix = int((x - self.table_bounds['min_x']) / self.cell_size)
        iy = int((y - self.table_bounds['min_y']) / self.cell_size)
        
        # Clamp to valid range (edge case for exactly max values)
        ix = min(ix, self.cells_x - 1)
        iy = min(iy, self.cells_y - 1)
        
        return f"cell_{ix}_{iy}"
    
    def get_cell_center(self, cell_id: str) -> Tuple[float, float]:
        """
        Get the (x, y) center coordinates of a cell.
        
        Args:
            cell_id: Cell ID string (e.g., 'cell_20_20')
        
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        return self.cells[cell_id]['center']
    
    def get_cell_bounds(self, cell_id: str) -> Tuple[float, float, float, float]:
        """
        Get the boundary coordinates of a cell.
        
        Args:
            cell_id: Cell ID string (e.g., 'cell_20_20')
        
        Returns:
            Tuple of (min_x, max_x, min_y, max_y) in meters
        """
        return self.cells[cell_id]['bounds']
    
    def get_grid_info(self) -> Dict:
        """
        Get comprehensive information about the grid configuration.
        
        Returns:
            Dictionary containing grid dimensions, cell size, working area, etc.
        """
        return {
            'grid_dimensions': (self.cells_x, self.cells_y),
            'total_cells': self.cells_x * self.cells_y,
            'cell_size': self.cell_size,
            'working_area': self.working_area,
            'table_height': self.table_bounds['table_height'],
            'table_bounds': self.table_bounds
        }
    