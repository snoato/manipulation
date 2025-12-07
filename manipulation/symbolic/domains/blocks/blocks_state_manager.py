"""State management for blocks world symbolic planning - grounding, initialization, and sampling."""

import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import re

from manipulation.symbolic.base_domain import BaseStateManager
from manipulation.symbolic.domains.blocks.blocks_domain import BlocksDomain


class BlocksStateManager(BaseStateManager):
    """Manages state grounding, initialization, and randomization for blocks world symbolic planning."""
    
    # Block specifications: size in meters (width, depth, height)
    # Indices 0-5: Small cubes (4cm x 4cm x 4cm)
    # Indices 6-11: Medium cubes (6cm x 6cm x 6cm)
    # Indices 12-13: Platform cuboids (10cm x 10cm x 5cm)
    # Indices 14-15: Large platform cuboids (15cm x 10cm x 5cm)
    BLOCK_SPECS = {
        **{i: (0.04, 0.04, 0.04) for i in range(6)},      # 0-5: 4cm cubes
        **{i: (0.06, 0.06, 0.06) for i in range(6, 12)},  # 6-11: 6cm cubes
        12: (0.10, 0.10, 0.05),  # Platform 1
        13: (0.10, 0.10, 0.05),  # Platform 2
        14: (0.15, 0.10, 0.05),  # Large platform 1
        15: (0.15, 0.10, 0.05)   # Large platform 2
    }
    
    # Graspable blocks (exclude large platforms)
    GRASPABLE_BLOCKS = list(range(12))  # Only cubes are graspable
    
    # Platform blocks (used as bases, not graspable)
    PLATFORM_BLOCKS = [12, 13, 14, 15]
    
    def __init__(self, domain: BlocksDomain, env):
        """
        Initialize state manager.
        
        Args:
            domain: BlocksDomain instance
            env: FrankaEnvironment instance
        """
        self.domain = domain
        self.env = env
        self.model = env.get_model()
        self.data = env.get_data()
        self.gripper_holding = None  # Track what gripper is holding (block_idx or None)
    
    def _get_block_pose(self, block_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get block position and orientation.
        
        Args:
            block_idx: Block index (0-15)
            
        Returns:
            Tuple of (position, quaternion)
        """
        body_name = f"block_{block_idx}"
        pos, quat = self.env.get_object_pose(body_name)
        return pos, quat
    
    def _hide_block(self, block_idx: int):
        """
        Hide block by moving it far off to the side.
        
        Args:
            block_idx: Block index (0-15)
        """
        body_name = f"block_{block_idx}"
        width, depth, height = self.BLOCK_SPECS[block_idx]
        # Hide at x=100, position at half-height above ground
        self.env.set_object_pose(body_name, np.array([100.0, 0.0, height / 2.0]))
    
    def _set_block_position(self, block_idx: int, x: float, y: float, z: float, quat: Optional[np.ndarray] = None):
        """
        Set block position (and optionally orientation).
        
        Args:
            block_idx: Block index (0-15)
            x, y, z: Position coordinates in meters
            quat: Optional quaternion [w, x, y, z]. Defaults to identity.
        """
        body_name = f"block_{block_idx}"
        position = np.array([x, y, z])
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self.env.set_object_pose(body_name, position, quat)
    
    def _check_xy_overlap(self, block1_idx: int, block2_idx: int, threshold: float = 0.8) -> bool:
        """
        Check if two blocks have sufficient XY overlap for stacking.
        
        Uses bounding box overlap with configurable threshold.
        
        Args:
            block1_idx: First block index
            block2_idx: Second block index
            threshold: Overlap ratio threshold (0.8 = 80%)
            
        Returns:
            True if blocks overlap sufficiently in XY plane
        """
        pos1, _ = self._get_block_pose(block1_idx)
        pos2, _ = self._get_block_pose(block2_idx)
        
        # Skip if either block is hidden
        if pos1[0] > 50.0 or pos2[0] > 50.0:
            return False
        
        w1, d1, _ = self.BLOCK_SPECS[block1_idx]
        w2, d2, _ = self.BLOCK_SPECS[block2_idx]
        
        # Compute bounding boxes (assuming axis-aligned)
        x1_min, x1_max = pos1[0] - w1/2, pos1[0] + w1/2
        y1_min, y1_max = pos1[1] - d1/2, pos1[1] + d1/2
        
        x2_min, x2_max = pos2[0] - w2/2, pos2[0] + w2/2
        y2_min, y2_max = pos2[1] - d2/2, pos2[1] + d2/2
        
        # Compute intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        if x_overlap == 0 or y_overlap == 0:
            return False
        
        # Compute overlap area
        overlap_area = x_overlap * y_overlap
        
        # Use smaller block's area as reference
        area1 = w1 * d1
        area2 = w2 * d2
        min_area = min(area1, area2)
        
        overlap_ratio = overlap_area / min_area
        return overlap_ratio >= threshold
    
    def _is_on_block(self, top_block_idx: int, bottom_block_idx: int) -> bool:
        """
        Check if top_block is on bottom_block using Z-height and XY overlap.
        
        Args:
            top_block_idx: Index of potentially upper block
            bottom_block_idx: Index of potentially lower block
            
        Returns:
            True if top_block is on bottom_block
        """
        pos_top, _ = self._get_block_pose(top_block_idx)
        pos_bottom, _ = self._get_block_pose(bottom_block_idx)
        
        # Skip hidden blocks
        if pos_top[0] > 50.0 or pos_bottom[0] > 50.0:
            return False
        
        _, _, h_top = self.BLOCK_SPECS[top_block_idx]
        _, _, h_bottom = self.BLOCK_SPECS[bottom_block_idx]
        
        # Expected Z position if top is on bottom
        expected_z = pos_bottom[2] + h_bottom/2 + h_top/2
        
        # Check if Z positions match (within tolerance) and XY overlap is sufficient
        z_tolerance = 0.01  # 1cm tolerance
        z_match = abs(pos_top[2] - expected_z) < z_tolerance
        
        if not z_match:
            return False
        
        return self._check_xy_overlap(top_block_idx, bottom_block_idx, threshold=0.8)
    
    def _is_on_table(self, block_idx: int) -> bool:
        """
        Check if block is resting on the table.
        
        Args:
            block_idx: Block index
            
        Returns:
            True if block is on table (not on another block)
        """
        pos, _ = self._get_block_pose(block_idx)
        
        # Skip hidden blocks
        if pos[0] > 50.0:
            return False
        
        _, _, height = self.BLOCK_SPECS[block_idx]
        expected_z = self.domain.table_height + height / 2.0
        
        # Check if block is at table height (within tolerance)
        z_tolerance = 0.015  # 1.5cm tolerance for table contact
        return abs(pos[2] - expected_z) < z_tolerance
    
    def _is_clear(self, block_idx: int, active_blocks: List[int]) -> bool:
        """
        Check if a block has no other blocks on top of it.
        
        Args:
            block_idx: Block index to check
            active_blocks: List of all active (non-hidden) block indices
            
        Returns:
            True if no blocks are on top of this block
        """
        for other_idx in active_blocks:
            if other_idx != block_idx and self._is_on_block(other_idx, block_idx):
                return False
        return True
    
    def ground_state(self) -> Dict[str, Any]:
        """
        Extract current symbolic state from the simulation.
        
        Returns blocks-world predicates:
        - on(A, B): Block A is on block B
        - on-table(A): Block A is on the table
        - clear(A): Block A has nothing on top
        - holding(gripper, A): Gripper is holding block A
        - gripper-empty: Gripper is not holding anything
        
        Returns:
            Dictionary with grounded state predicates
        """
        state = {
            'blocks': {},      # block_name -> properties
            'on': [],          # (block_a, block_b) pairs
            'on_table': [],    # blocks on table
            'clear': [],       # clear blocks
            'gripper_empty': self.gripper_holding is None,
            'holding': self.gripper_holding  # block_idx or None
        }
        
        # Find all active blocks (not hidden)
        active_blocks = []
        for block_idx in range(len(self.BLOCK_SPECS)):
            pos, _ = self._get_block_pose(block_idx)
            if pos[0] < 50.0:  # Not hidden
                active_blocks.append(block_idx)
                state['blocks'][f'block_{block_idx}'] = {
                    'position': pos.tolist(),
                    'size': self.BLOCK_SPECS[block_idx]
                }
        
        # Compute on(A, B) relationships
        for top_idx in active_blocks:
            for bottom_idx in active_blocks:
                if top_idx != bottom_idx and self._is_on_block(top_idx, bottom_idx):
                    state['on'].append((f'block_{top_idx}', f'block_{bottom_idx}'))
        
        # Compute on-table(A)
        for block_idx in active_blocks:
            if self._is_on_table(block_idx):
                state['on_table'].append(f'block_{block_idx}')
        
        # Compute clear(A)
        for block_idx in active_blocks:
            if self._is_clear(block_idx, active_blocks):
                state['clear'].append(f'block_{block_idx}')
        
        return state
    
    def generate_pddl_problem(self, problem_name: str, output_path: Path, goal_predicates: Optional[List[str]] = None) -> None:
        """
        Generate a PDDL problem file from current state.
        
        Args:
            problem_name: Name for the PDDL problem
            output_path: Path where to save the PDDL file
            goal_predicates: Optional list of goal predicates (e.g., ["(on block_0 block_12)"])
        """
        state = self.ground_state()
        
        # Extract block names
        block_names = sorted(state['blocks'].keys())
        
        # Build PDDL problem file
        pddl_lines = [
            f"(define (problem {problem_name})",
            "  (:domain blocks-world)",
            "",
            "  (:objects",
            f"    {' '.join(block_names)} - block",
            "    gripper1 - gripper",
            "  )",
            "",
            "  (:init"
        ]
        
        # Add on(A, B) predicates
        for block_a, block_b in state['on']:
            pddl_lines.append(f"    (on {block_a} {block_b})")
        
        # Add on-table(A) predicates
        for block in state['on_table']:
            pddl_lines.append(f"    (on-table {block})")
        
        # Add clear(A) predicates
        for block in state['clear']:
            pddl_lines.append(f"    (clear {block})")
        
        # Add gripper state
        if state['gripper_empty']:
            pddl_lines.append("    (gripper-empty gripper1)")
        else:
            pddl_lines.append(f"    (holding gripper1 block_{state['holding']})")
        
        pddl_lines.append("  )")
        
        # Add goal if provided
        if goal_predicates:
            pddl_lines.append("")
            pddl_lines.append("  (:goal (and")
            for pred in goal_predicates:
                pddl_lines.append(f"    {pred}")
            pddl_lines.append("  ))")
        
        pddl_lines.append(")")
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(pddl_lines))
    
    def sample_random_state(self, 
                           n_blocks: Optional[int] = 5,
                           include_platforms: bool = True,
                           seed: Optional[int] = None):
        """
        Generate and initialize a random valid state in MuJoCo.
        
        Randomly places blocks on the table within the working area,
        ensuring no collisions.
        
        Args:
            n_blocks: Number of graspable blocks to place (1-12)
            include_platforms: Whether to include platform blocks
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Clamp n_blocks to valid range
        n_blocks = max(1, min(len(self.GRASPABLE_BLOCKS), n_blocks))
        
        # Hide all blocks first
        for block_idx in range(len(self.BLOCK_SPECS)):
            self._hide_block(block_idx)
        
        # Reset gripper state
        self.gripper_holding = None
        
        # Select blocks to place
        selected_blocks = list(np.random.choice(self.GRASPABLE_BLOCKS, n_blocks, replace=False))
        
        if include_platforms:
            selected_blocks.extend(self.PLATFORM_BLOCKS)
        
        # Track placed block positions for collision checking
        placed_blocks = []  # List of (x, y, width, depth) tuples
        
        bounds = self.domain.get_working_bounds()
        
        # Add margin to avoid placing blocks at edges
        margin = 0.05  # 5cm margin
        x_min = bounds['min_x'] + margin
        x_max = bounds['max_x'] - margin
        y_min = bounds['min_y'] + margin
        y_max = bounds['max_y'] - margin
        
        for block_idx in selected_blocks:
            width, depth, height = self.BLOCK_SPECS[block_idx]
            
            # Try to find a collision-free position
            max_attempts = 100
            placed = False
            
            for _ in range(max_attempts):
                # Random position within working area
                # Ensure block footprint fits entirely within bounds
                x = np.random.uniform(x_min + width/2, x_max - width/2)
                y = np.random.uniform(y_min + depth/2, y_max - depth/2)
                
                # Check collision with already placed blocks
                collision = False
                for px, py, pw, pd in placed_blocks:
                    # Check if bounding boxes overlap (with small clearance)
                    clearance = 0.01  # 1cm clearance between blocks
                    if (abs(x - px) < (width + pw) / 2 + clearance and
                        abs(y - py) < (depth + pd) / 2 + clearance):
                        collision = True
                        break
                
                if not collision:
                    # Place block on table with small clearance
                    z = self.domain.table_height + height / 2.0 + 0.003  # 3mm clearance
                    self._set_block_position(block_idx, x, y, z)
                    placed_blocks.append((x, y, width, depth))
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place block_{block_idx} without collision after {max_attempts} attempts")
        
        # Zero out all velocities and update physics
        self.env.reset_velocities()
        self.env.forward()
    
    def compute_pickup_pose(self, block_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute grasp pose for picking up a block.
        
        Returns top-center position with vertical (downward) approach.
        
        Args:
            block_idx: Block index to grasp
            
        Returns:
            Tuple of (position, quaternion) for end-effector
        """
        pos, _ = self._get_block_pose(block_idx)
        _, _, height = self.BLOCK_SPECS[block_idx]
        
        # Grasp at top-center of block
        grasp_x = pos[0]
        grasp_y = pos[1]
        grasp_z = pos[2] + height / 2.0 + 0.02  # Slightly above block top
        
        grasp_position = np.array([grasp_x, grasp_y, grasp_z])
        
        # Vertical downward orientation (gripper pointing down)
        # Quaternion for -90° rotation around Y-axis: [cos(-45°), 0, sin(-45°), 0] = [0.707, 0, -0.707, 0]
        # Actually for straight down: [0, 1, 0, 0] or [0.707, 0.707, 0, 0] depending on gripper frame
        # Standard downward grasp orientation
        grasp_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        
        return grasp_position, grasp_quaternion
    
    def compute_putdown_pose(self, target_x: float, target_y: float, target_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pose for placing a block at a target position.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            target_z: Target Z coordinate (center of block after placement)
            
        Returns:
            Tuple of (position, quaternion) for end-effector
        """
        # Place position is slightly above target
        place_position = np.array([target_x, target_y, target_z + 0.01])
        
        # Same downward orientation as pickup
        place_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        
        return place_position, place_quaternion
    
    def compute_stack_pose(self, target_block_idx: int, block_to_place_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pose for stacking a block on top of another block.
        
        Args:
            target_block_idx: Block to stack onto
            block_to_place_idx: Block being placed
            
        Returns:
            Tuple of (position, quaternion) for end-effector
        """
        target_pos, _ = self._get_block_pose(target_block_idx)
        _, _, target_height = self.BLOCK_SPECS[target_block_idx]
        _, _, place_height = self.BLOCK_SPECS[block_to_place_idx]
        
        # Stack at center of target block
        stack_x = target_pos[0]
        stack_y = target_pos[1]
        # Z position: top of target + half height of block being placed
        stack_z = target_pos[2] + target_height / 2.0 + place_height / 2.0
        
        return self.compute_putdown_pose(stack_x, stack_y, stack_z)
    
    def compute_table_pose(self, x: float, y: float, block_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pose for placing a block on the table.
        
        Args:
            x: Target X coordinate on table
            y: Target Y coordinate on table
            block_idx: Block being placed
            
        Returns:
            Tuple of (position, quaternion) for end-effector
        """
        _, _, height = self.BLOCK_SPECS[block_idx]
        z = self.domain.table_height + height / 2.0
        
        return self.compute_putdown_pose(x, y, z)
