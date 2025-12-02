"""State management for symbolic planning - grounding, initialization, and sampling."""

import mujoco
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import re

from manipulation.symbolic.grid_domain import GridDomain


class StateManager:
    """Manages state grounding, initialization, and randomization for symbolic planning."""
    
    # Cylinder specifications: (radius, height) indexed by cylinder number
    CYLINDER_SPECS = {
        **{i: (0.015, 0.10) for i in range(15)},      # 0-14: thin
        **{i: (0.020, 0.12) for i in range(15, 25)},  # 15-24: medium
        **{i: (0.025, 0.16) for i in range(25, 30)}   # 25-29: thick
    }
    
    def __init__(self, grid_domain: GridDomain, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize state manager.
        
        Args:
            grid_domain: GridDomain instance
            model: MuJoCo model
            data: MuJoCo data
        """
        self.grid = grid_domain
        self.model = model
        self.data = data
        self.gripper_holding = None  # Track what gripper is holding
        
    def _compute_circle_rectangle_intersection_area(self, 
                                                     circle_x: float, 
                                                     circle_y: float,
                                                     circle_r: float,
                                                     rect_min_x: float,
                                                     rect_max_x: float,
                                                     rect_min_y: float,
                                                     rect_max_y: float) -> float:
        """
        Compute intersection area between circle and rectangle.
        Uses Monte Carlo sampling for approximation.
        
        Args:
            circle_x, circle_y, circle_r: Circle center and radius
            rect_min_x, rect_max_x, rect_min_y, rect_max_y: Rectangle bounds
            
        Returns:
            Intersection area
        """
        # Simple approximation: sample points in rectangle and check if in circle
        samples = 100
        count = 0
        
        for _ in range(samples):
            x = np.random.uniform(rect_min_x, rect_max_x)
            y = np.random.uniform(rect_min_y, rect_max_y)
            
            if (x - circle_x)**2 + (y - circle_y)**2 <= circle_r**2:
                count += 1
        
        rect_area = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y)
        intersection_area = (count / samples) * rect_area
        
        return intersection_area
    
    def _get_cylinder_occupied_cells(self, cyl_idx: int) -> Set[str]:
        """
        Get all cells occupied by a cylinder (≥50% overlap rule).
        
        Args:
            cyl_idx: Cylinder index (0-29)
            
        Returns:
            Set of cell names occupied by this cylinder
        """
        # Get cylinder position
        body_name = f"cylinder_{cyl_idx}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if body_id < 0:
            return set()
        
        # Get position from qpos (first 3 elements of the body's freejoint)
        joint_name = f"{body_name}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id < 0:
            return set()
        
        joint_qposadr = self.model.jnt_qposadr[joint_id]
        cyl_x = self.data.qpos[joint_qposadr]
        cyl_y = self.data.qpos[joint_qposadr + 1]
        cyl_z = self.data.qpos[joint_qposadr + 2]
        
        # Skip if cylinder is hidden (far off to the side at x=100)
        if cyl_x > 50.0:
            return set()
        
        # Get cylinder radius
        radius, _ = self.CYLINDER_SPECS[cyl_idx]
        
        # Find all cells with ≥50% overlap
        occupied = set()
        cell_area = self.grid.cell_size ** 2
        threshold_area = 0.5 * cell_area
        
        for cell_name, cell_data in self.grid.cells.items():
            rect_bounds = cell_data['bounds']
            intersection = self._compute_circle_rectangle_intersection_area(
                cyl_x, cyl_y, radius,
                rect_bounds[0], rect_bounds[1], rect_bounds[2], rect_bounds[3]
            )
            
            if intersection >= threshold_area:
                occupied.add(cell_name)
        
        return occupied
    
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
    
    def generate_pddl_problem(self, problem_name: str, output_path: Optional[str] = None) -> str:
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
        for cell, neighbors in self.grid.adjacency.items():
            for neighbor in neighbors:
                pair = tuple(sorted([cell, neighbor]))
                if pair not in added_adjacencies:
                    init_predicates.append(f"    (adjacent {pair[0]} {pair[1]})")
                    init_predicates.append(f"    (adjacent {pair[1]} {pair[0]})")
                    added_adjacencies.add(pair)
        
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
            centroid_z = self.grid.table_height + height / 2.0 + 0.1  # 0.5cm clearance
            
            # Set cylinder position
            self._set_cylinder_position(cyl_idx, centroid_x, centroid_y, centroid_z)
        
        # Zero out all velocities to prevent wobbling
        self.data.qvel[:] = 0
    
    def sample_random_state(self, 
                           n_cylinders: Optional[int] = None,
                           seed: Optional[int] = None) -> str:
        """
        Generate random valid state.
        
        Args:
            n_cylinders: Number of cylinders to place (1-30), random if None
            seed: Random seed
            
        Returns:
            PDDL init section string
        """
        if seed is not None:
            np.random.seed(seed)
        
        if n_cylinders is None:
            n_cylinders = np.random.randint(1, 31)
        
        n_cylinders = max(1, min(30, n_cylinders))
        
        # Select random cylinders
        selected_cylinders = np.random.choice(30, n_cylinders, replace=False)
        
        # Try to place cylinders without overlap
        cylinder_positions = {}
        occupied_cells_global = set()
        
        max_attempts = 1000
        placed = 0
        
        for attempt in range(max_attempts):
            if placed >= n_cylinders:
                break
            
            cyl_idx = selected_cylinders[placed]
            radius, _ = self.CYLINDER_SPECS[cyl_idx]
            
            # Sample random cell
            cell_name = np.random.choice(list(self.grid.cells.keys()))
            center_x, center_y = self.grid.cells[cell_name]['center']
            
            # Compute which cells this cylinder would occupy
            potential_occupied = set()
            cell_area = self.grid.cell_size ** 2
            threshold_area = 0.5 * cell_area
            
            for check_cell, cell_data in self.grid.cells.items():
                rect_bounds = cell_data['bounds']
                intersection = self._compute_circle_rectangle_intersection_area(
                    center_x, center_y, radius,
                    rect_bounds[0], rect_bounds[1], rect_bounds[2], rect_bounds[3]
                )
                
                if intersection >= threshold_area:
                    potential_occupied.add(check_cell)
            
            # Check if placement is valid (no overlap with existing cylinders)
            if not potential_occupied.intersection(occupied_cells_global):
                cylinder_positions[cyl_idx] = (center_x, center_y, potential_occupied)
                occupied_cells_global.update(potential_occupied)
                placed += 1
        
        # Generate PDDL init predicates
        predicates = []
        predicates.append("(gripper-empty gripper1)")
        
        for cyl_idx, (x, y, cells) in cylinder_positions.items():
            for cell in cells:
                predicates.append(f"(occupied {cell} cylinder_{cyl_idx})")
        
        # Mark empty cells
        for cell in self.grid.cells.keys():
            if cell not in occupied_cells_global:
                predicates.append(f"(empty {cell})")
        
        return "\n    ".join(predicates)
    
    def _hide_cylinder(self, cyl_idx: int):
        """Move cylinder far off to the side to hide it."""
        joint_name = f"cylinder_{cyl_idx}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id >= 0:
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            _, height = self.CYLINDER_SPECS[cyl_idx]
            self.data.qpos[joint_qposadr:joint_qposadr+3] = [100.0, 0.0, height]
            self.data.qpos[joint_qposadr+3:joint_qposadr+7] = [1, 0, 0, 0]  # identity quat
            self.data.qvel[self.model.jnt_dofadr[joint_id]:self.model.jnt_dofadr[joint_id]+6] = 0
    
    def _set_cylinder_position(self, cyl_idx: int, x: float, y: float, z: float):
        """Set cylinder position."""
        joint_name = f"cylinder_{cyl_idx}_freejoint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id >= 0:
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[joint_qposadr:joint_qposadr+3] = [x, y, z]
            self.data.qpos[joint_qposadr+3:joint_qposadr+7] = [1, 0, 0, 0]  # identity quat
            self.data.qvel[self.model.jnt_dofadr[joint_id]:self.model.jnt_dofadr[joint_id]+6] = 0
