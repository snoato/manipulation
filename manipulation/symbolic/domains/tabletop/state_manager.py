"""State management for tabletop symbolic planning - grounding, initialization, and sampling."""

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re

from manipulation.symbolic.base_domain import BaseStateManager
from manipulation.symbolic.domains.tabletop.grid_domain import GridDomain


def extract_grid_dimensions_from_pddl(pddl_content: str) -> Tuple[int, int]:
    """
    Extract grid dimensions from PDDL cell objects.
    
    Parses cell names like 'cell_X_Y' to find maximum indices.
    
    Args:
        pddl_content: PDDL file content as string
        
    Returns:
        (cells_x, cells_y) tuple representing grid dimensions
        
    Raises:
        ValueError: If no cell objects found in PDDL file
    """
    cell_pattern = r'cell_(\d+)_(\d+)'
    matches = re.findall(cell_pattern, pddl_content)
    
    if not matches:
        raise ValueError("No cell objects found in PDDL file")
    
    max_x = max(int(x) for x, y in matches)
    max_y = max(int(y) for x, y in matches)
    
    return (max_x + 1, max_y + 1)


class StateManager(BaseStateManager):
    """Manages state grounding, initialization, and randomization for tabletop symbolic planning."""
    
    # Cylinder specifications: (radius, height) indexed by cylinder number
    CYLINDER_SPECS = {
        **{i: (0.0125, 0.08) for i in range(15)},     # 0-14: thin (1.25cm radius, 8cm height)
        **{i: (0.0175, 0.10) for i in range(15, 25)}, # 15-24: medium (1.75cm radius, 10cm height)
        **{i: (0.020, 0.12) for i in range(25, 30)}   # 25-29: thick (2.0cm radius, 12cm height)
    }
    
    def __init__(self, grid_domain: GridDomain, env):
        """
        Initialize state manager.
        
        Args:
            grid_domain: GridDomain instance
            env: FrankaEnvironment instance
        """
        self.grid = grid_domain
        self.env = env
        self.model = env.get_model()
        self.data = env.get_data()
        self.gripper_holding = None  # Track what gripper is holding
        
    
    def _get_cylinder_occupied_cells(self, cyl_idx: int) -> Set[str]:
        """
        Get all cells occupied by a cylinder (cell center within radius).
        
        Args:
            cyl_idx: Cylinder index (0-29)
            
        Returns:
            Set of cell names occupied by this cylinder
        """
        # Get cylinder position using environment wrapper
        body_name = f"cylinder_{cyl_idx}"
        pos, _ = self.env.get_object_pose(body_name)
        
        if pos is None:
            return set()
        
        cyl_x, cyl_y, cyl_z = pos
        
        # Skip if cylinder is hidden (far off to the side at x=100)
        if cyl_x > 50.0:
            return set()
        
        # Find the cell containing the cylinder's center position
        # We use abstract representation: one cylinder = one cell (its center cell)
        # This allows PDDL pick/place actions to work correctly
        try:
            center_cell = self.grid.get_cell_at_position(cyl_x, cyl_y)
            return {center_cell}
        except ValueError:
            # Cylinder is outside the grid
            return set()
    
    def ground_state(self) -> Dict[str, any]:
        """
        Ground current MuJoCo state to symbolic representation.
        
        Returns:
            Dictionary with cylinders, their occupied cells, and gripper state
        """
        state = {
            'cylinders': {},
            'gripper_empty': self.gripper_holding is None,
            'holding': self.gripper_holding
        }
        
        # Check each cylinder
        for cyl_idx in range(30):
            occupied_cells = self._get_cylinder_occupied_cells(cyl_idx)
            if occupied_cells:
                state['cylinders'][f"cylinder_{cyl_idx}"] = list(occupied_cells)
        
        return state
    
    def set_from_grounded_state(self, state: Dict[str, any]):
        """
        Set MuJoCo state from grounded symbolic representation.
        
        Takes the same dictionary format as returned by ground_state() and
        configures the MuJoCo simulation accordingly.
        
        Args:
            state: Dictionary with 'cylinders', 'gripper_empty', and 'holding' keys
        """
        # Hide all cylinders first
        for cyl_idx in range(30):
            self._hide_cylinder(cyl_idx)
        
        # Place cylinders at the centroid of their occupied cells
        for cyl_name, cells in state.get('cylinders', {}).items():
            # Extract cylinder index
            cyl_idx = int(cyl_name.split('_')[1])
            
            # Compute centroid of occupied cells
            cell_centers = [self.grid.cells[cell]['center'] for cell in cells]
            centroid_x = np.mean([c[0] for c in cell_centers])
            centroid_y = np.mean([c[1] for c in cell_centers])
            
            # Get cylinder height - position center slightly above table to avoid initial contact
            _, height = self.CYLINDER_SPECS[cyl_idx]
            centroid_z = self.grid.table_height + height / 2.0 + 0.003  # 0.3cm clearance
            
            # Set cylinder position
            self._set_cylinder_position(cyl_idx, centroid_x, centroid_y, centroid_z)
        
        # Update gripper holding state
        self.gripper_holding = state.get('holding', None)

        self.env.data.qpos[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
        self.env.data.ctrl[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
        
        # Zero out all velocities to prevent wobbling
        self.env.reset_velocities()
    
    def generate_pddl_problem(self, problem_name: str, output_path: Optional[str] = None, goal_string: Optional[str] = "") -> str:
        """
        Generate PDDL problem file from current state.
        
        Args:
            problem_name: Name for the problem
            output_path: Optional path to save problem file
            
        Returns:
            PDDL problem string
        """
        state = self.ground_state()
        
        # Build objects section
        objects = []
        
        # All cells
        cell_names = sorted(self.grid.cells.keys())
        objects.append(f"    {' '.join(cell_names)} - cell")
        
        # Active cylinders
        cylinder_names = sorted(state['cylinders'].keys())
        if cylinder_names:
            objects.append(f"    {' '.join(cylinder_names)} - cylinder")
        
        # Gripper
        objects.append("    gripper1 - gripper")
        
        # Build init section
        init_predicates = []
        
        # Adjacency (only include once per pair)
        added_adjacencies = set()
        for cell, directions in self.grid.directional_adjacency.items():
            for direction, neighbor in directions.items():
                init_predicates.append(f"    (adjacent {direction} {cell} {neighbor})")
        
        # Occupied cells
        occupied_cells = set()
        for cyl_name, cells in state['cylinders'].items():
            for cell in cells:
                init_predicates.append(f"    (occupied {cell} {cyl_name})")
                occupied_cells.add(cell)
        
        # Empty cells
        for cell in cell_names:
            if cell not in occupied_cells:
                init_predicates.append(f"    (empty {cell})")
        
        # Gripper state
        if state['gripper_empty']:
            init_predicates.append("    (gripper-empty gripper1)")
        else:
            init_predicates.append(f"    (holding gripper1 {state['holding']})")
        
        # Build PDDL problem
        problem = f"""(define (problem {problem_name})
  (:domain tabletop-manipulation)
  
  (:objects
{chr(10).join(objects)}
  )
  
  (:init
{chr(10).join(init_predicates)}
  )
  
  (:goal (and
    ; Define goal here
    {goal_string}
  ))
)
"""
        
        if output_path:
            Path(output_path).write_text(problem)
        
        return problem
    
    def init_from_pddl_state(self, pddl_state: str):
        """
        Initialize MuJoCo state from PDDL state description.
        
        Args:
            pddl_state: PDDL init section with (occupied cell cylinder) predicates
        """
        # Parse occupied predicates
        occupied_pattern = r'\(occupied\s+(\w+)\s+(\w+)\)'
        matches = re.findall(occupied_pattern, pddl_state)
        
        # Group cells by cylinder
        cylinder_cells = {}
        for cell_name, cyl_name in matches:
            if cyl_name not in cylinder_cells:
                cylinder_cells[cyl_name] = []
            cylinder_cells[cyl_name].append(cell_name)
        
        # Hide all cylinders first
        for cyl_idx in range(30):
            self._hide_cylinder(cyl_idx)
        
        # Place cylinders at centroid of their occupied cells
        for cyl_name, cells in cylinder_cells.items():
            # Extract cylinder index
            cyl_idx = int(cyl_name.split('_')[1])
            
            # Compute centroid of occupied cells
            cell_centers = [self.grid.cells[cell]['center'] for cell in cells]
            centroid_x = np.mean([c[0] for c in cell_centers])
            centroid_y = np.mean([c[1] for c in cell_centers])
            
            # Get cylinder height - position center slightly above table to avoid initial contact
            _, height = self.CYLINDER_SPECS[cyl_idx]
            centroid_z = self.grid.table_height + height / 2.0 + 0.3  # 0.5cm clearance
            
            # Set cylinder position
            self._set_cylinder_position(cyl_idx, centroid_x, centroid_y, centroid_z)
        
        # Zero out all velocities to prevent wobbling
        self.env.reset_velocities()
    
    def sample_random_state(self, 
                       n_cylinders: Optional[int] = 5,
                       seed: Optional[int] = None):
        """
        Generate and initialize a random valid state in MuJoCo.
        
        Args:
            n_cylinders: Number of cylinders to place (1-30)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_cylinders = max(1, min(30, n_cylinders))
        
        # Create occupancy grid (True = occupied, False = free)
        occupancy_grid = np.zeros((self.grid.cells_x, self.grid.cells_y), dtype=bool)
        
        # Track cylinder placements
        cylinder_positions = {}  # cyl_idx -> (cell_x, cell_y)
        
        # Select random cylinders
        selected_cylinders = np.random.choice(30, n_cylinders, replace=False)
        
        # Hide all cylinders first
        for cyl_idx in range(30):
            self._hide_cylinder(cyl_idx)
        
        for cyl_idx in selected_cylinders:
            # Get cylinder diameter in cells
            radius, _ = self.CYLINDER_SPECS[cyl_idx]
            diameter = 2 * radius
            diameter_cells = int(np.ceil(diameter / self.grid.cell_size))
            
            # Find all free cells (not occupied)
            free_cells = []
            for x in range(self.grid.cells_x):
                for y in range(self.grid.cells_y):
                    if not occupancy_grid[x, y]:
                        free_cells.append((x, y))
            
            if not free_cells:
                print(f"Warning: No free cells for cylinder_{cyl_idx}")
                continue
            
            # Try random free cells until we find one with free surroundings
            np.random.shuffle(free_cells)
            placed = False
            
            for cell_x, cell_y in free_cells:
                # Check if surrounding cells (within diameter) are also free
                all_clear = True
                for dx in range(-diameter_cells, diameter_cells + 1):
                    for dy in range(-diameter_cells, diameter_cells + 1):
                        check_x = cell_x + dx
                        check_y = cell_y + dy
                        
                        # Skip if out of bounds
                        if check_x < 0 or check_x >= self.grid.cells_x or \
                           check_y < 0 or check_y >= self.grid.cells_y:
                            continue
                        
                        # Check if this cell is occupied
                        if occupancy_grid[check_x, check_y]:
                            all_clear = False
                            break
                    
                    if not all_clear:
                        break
                
                if all_clear:
                    # Place cylinder at this cell center
                    cell_name = f"cell_{cell_x}_{cell_y}"
                    center_x, center_y = self.grid.cells[cell_name]['center']
                    _, height = self.CYLINDER_SPECS[cyl_idx]
                    center_z = self.grid.table_height + height / 2.0 + 0.013  # 0.3cm clearance
                    
                    self._set_cylinder_position(cyl_idx, center_x, center_y, center_z)
                    cylinder_positions[cyl_idx] = (cell_x, cell_y)
                    
                    # Mark surrounding cells as occupied (with spacing)
                    for dx in range(-diameter_cells, diameter_cells + 1):
                        for dy in range(-diameter_cells, diameter_cells + 1):
                            mark_x = cell_x + dx
                            mark_y = cell_y + dy
                            
                            if 0 <= mark_x < self.grid.cells_x and \
                               0 <= mark_y < self.grid.cells_y:
                                occupancy_grid[mark_x, mark_y] = True
                    
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place cylinder_{cyl_idx} - no valid location")
        
        # Zero velocities and update simulation
        self.env.reset_velocities()
        self.env.step()
        self.env.reset_velocities()
        self.env.forward()
        
        print(f"Placed {len(cylinder_positions)} cylinders successfully")
    
    def _hide_cylinder(self, cyl_idx: int):
        """Move cylinder far off to the side to hide it."""
        body_name = f"cylinder_{cyl_idx}"
        _, height = self.CYLINDER_SPECS[cyl_idx]
        self.env.set_object_pose(body_name, np.array([100.0, 0.0, height]))
    
    def _set_cylinder_position(self, cyl_idx: int, x: float, y: float, z: float):
        """Set cylinder position."""
        body_name = f"cylinder_{cyl_idx}"
        self.env.set_object_pose(body_name, np.array([x, y, z]))
