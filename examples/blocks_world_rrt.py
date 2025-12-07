"""Example demonstrating blocks world with real robot stacking using RRT planning."""

from pathlib import Path
import time
import numpy as np

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus
from manipulation.symbolic.domains.blocks import BlocksDomain, BlocksStateManager

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_blocks.xml"
_OUTPUT_DIR = _HERE / "blocks_output"


class PickPlaceController:
    """State machine controller for pick and place actions."""
    def __init__(self, env, planner, state_manager):
        self.env = env
        self.planner = planner
        self.state_manager = state_manager
        self.action_queue = []
        self.current_action = None
        self.step = 0
        self.wait_timer = 0.0
        self.wait_duration = 0.0
        
    def execute_pick_from_table(self, block_idx):
        """Queue a pick-from-table action."""
        block_pos, _ = self.state_manager._get_block_pose(block_idx)
        self.action_queue.append({
            'type': 'pick',
            'block_idx': block_idx,
            'block_pos': block_pos.copy(),
            'block_name': f"block_{block_idx}"
        })
        
    def execute_stack(self, block_idx, target_block_idx, custom_offset=None):
        """Queue a stack action."""
        self.action_queue.append({
            'type': 'stack',
            'block_idx': block_idx,
            'target_block_idx': target_block_idx,
            'custom_offset': custom_offset
        })
        
    def is_busy(self):
        """Check if controller is executing actions."""
        return len(self.action_queue) > 0 or self.current_action is not None
        
    def update(self, dt):
        """Update state machine - call this in main loop."""
        # Update wait timer
        if self.wait_duration > 0:
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                self.wait_duration = 0.0
                self.wait_timer = 0.0
            else:
                return  # Still waiting
        
        # Start new action if idle
        if self.current_action is None and len(self.action_queue) > 0:
            self.current_action = self.action_queue.pop(0)
            self.step = 0
            
            if self.current_action['type'] == 'pick':
                print(f"\n   === Picking block_{self.current_action['block_idx']} from table ===")
            elif self.current_action['type'] == 'stack':
                print(f"\n   === Stacking block_{self.current_action['block_idx']} on block_{self.current_action['target_block_idx']} ===")
        
        # Process current action
        if self.current_action is not None:
            if self.current_action['type'] == 'pick':
                done = self._update_pick(dt)
                if done:
                    self.current_action = None
                    self.step = 0
            elif self.current_action['type'] == 'stack':
                done = self._update_stack(dt)
                if done:
                    self.current_action = None
                    self.step = 0
    
    def _update_pick(self, dt):
        """Update pick action state machine."""
        action = self.current_action
        
        if self.env.controller.get_status() == ControllerStatus.IDLE:
            self.step += 1
            
            if self.step == 1:
                print(f"     Step 1/4: Planning approach...")
                approach_offset = np.array([0, 0, 0.15])
                approach_pos = action['block_pos'] + approach_offset
                approach_quat = np.array([0, 1, 0, 0])
                
                path = self.planner.plan_to_pose(
                    approach_pos, approach_quat, dt=dt, max_iterations=2000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.01)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan approach!")
                    return True  # Abort
            
            elif self.step == 2:
                print(f"     Step 2/4: Planning grasp...")
                grasp_pos, grasp_quat = self.state_manager.compute_pickup_pose(action['block_idx'])
                
                path = self.planner.plan_to_pose(
                    grasp_pos, grasp_quat, dt=dt, max_iterations=1000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.001)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan grasp!")
                    return True
            
            elif self.step == 3:
                print(f"     Step 3/5: Closing gripper...")
                self.env.controller.close_gripper()
                self.wait_duration = 1.5  # Wait for gripper to close
                self.wait_timer = 0.0
                self.step += 1  # Move to waiting state
                return False  # Return to let wait timer run
            
            elif self.step == 4:
                print(f"     Step 4/5: Gripper closed, ready to lift...")
                # Wait completed, will proceed to step 5 on next IDLE check
            
            elif self.step == 5:
                print(f"     Step 5/5: Planning lift...")
                lift_pos = action['block_pos'] + np.array([0, 0, 0.2])
                lift_quat = np.array([0, 1, 0, 0])
                
                self.env.add_collision_exception(action['block_name'])
                
                path = self.planner.plan_to_pose(
                    lift_pos, lift_quat, dt=dt, max_iterations=2000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.01)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan lift!")
                    return True
            
            elif self.step == 6:
                print(f"     ✓ Pick complete!")
                self.state_manager.gripper_holding = action['block_idx']
                return True  # Done
        
        return False


    def _update_stack(self, dt):
        """Update stack action state machine."""
        action = self.current_action
        block_name = f"block_{action['block_idx']}"
        
        # Compute stack pose on first step
        if self.step == 0:
            if action['custom_offset'] is not None:
                target_pos, _ = self.state_manager._get_block_pose(action['target_block_idx'])
                _, _, target_h = self.state_manager.BLOCK_SPECS[action['target_block_idx']]
                _, _, block_h = self.state_manager.BLOCK_SPECS[action['block_idx']]
                
                stack_x = target_pos[0] + action['custom_offset'][0]
                stack_y = target_pos[1] + action['custom_offset'][1]
                stack_z = target_pos[2] + target_h/2 + block_h/2 + 0.04
                action['stack_pos'] = np.array([stack_x, stack_y, stack_z])
                action['stack_quat'] = np.array([0, 1, 0, 0])
            else:
                stack_pos, stack_quat = self.state_manager.compute_stack_pose(
                    action['target_block_idx'], action['block_idx']
                )
                action['stack_pos'] = stack_pos
                action['stack_quat'] = stack_quat
        
        if self.env.controller.get_status() == ControllerStatus.IDLE:
            self.step += 1
            
            if self.step == 1:
                print(f"     Step 1/5: Planning approach to stack position...")
                # Add collision exceptions for held block and target platform
                target_name = f"block_{action['target_block_idx']}"
                self.env.add_collision_exception(target_name)
                
                approach_pos = action['stack_pos'] + np.array([0, 0, 0.15])  # Higher approach
                approach_quat = action['stack_quat']
                
                path = self.planner.plan_to_pose(
                    approach_pos, approach_quat, dt=dt, max_iterations=3000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.01)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan stack approach!")
                    return True
            
            elif self.step == 2:
                print(f"     Step 2/5: Planning placement...")
                
                path = self.planner.plan_to_pose(
                    action['stack_pos'], action['stack_quat'], dt=dt, max_iterations=1000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.001)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan placement!")
                    return True
            
            elif self.step == 3:
                print(f"     Step 3/5: Opening gripper...")
                self.env.controller.open_gripper()
                self.wait_duration = 1.5  # Wait for gripper to open
                self.wait_timer = 0.0
                self.step += 1  # Move to waiting state
                return False  # Return to let wait timer run
            
            elif self.step == 4:
                print(f"     Step 4/5: Gripper opened, ready to retreat...")
                # Wait completed, will proceed to step 5 on next IDLE check
            
            elif self.step == 5:
                print(f"     Step 5/5: Planning retreat...")
                # Remove collision exceptions
                self.env.remove_collision_exception(block_name)
                target_name = f"block_{action['target_block_idx']}"
                self.env.remove_collision_exception(target_name)
                
                retreat_pos = action['stack_pos'] + np.array([0, 0, 0.15])
                retreat_quat = action['stack_quat']
                
                path = self.planner.plan_to_pose(
                    retreat_pos, retreat_quat, dt=dt, max_iterations=2000
                )
                
                if path is not None:
                    smoothed = self.planner.smooth_path(path)
                    interpolated = self.env.controller.interpolate_linear_path(smoothed, step_size=0.01)
                    self.env.controller.follow_trajectory(interpolated)
                else:
                    print(f"     ERROR: Failed to plan retreat!")
                    return True
            
            elif self.step == 6:
                print(f"     ✓ Stack complete!")
                self.state_manager.gripper_holding = None
                return True
        
        return False


def main():
    # Configuration
    RANDOMIZE = False  # Set to False for fixed configuration
    
    print("=" * 70)
    print("Blocks World Demo - Real Robot Stacking with RRT Planning")
    print("=" * 70)
    print(f"\nConfiguration: {'RANDOMIZED' if RANDOMIZE else 'FIXED'}")
    
    # Initialize environment
    print("\n1. Loading environment with blocks scene...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Initialize RRT* planner
    print("\n2. Initializing RRT* motion planner...")
    planner = RRTStar(env)
    planner.max_iterations = 3000
    planner.step_size = 0.2
    planner.goal_sample_rate = 0.2
    print(f"   Max iterations: {planner.max_iterations}")
    print(f"   Step size: {planner.step_size}")
    
    # Create blocks domain
    print("\n3. Creating blocks domain...")
    domain = BlocksDomain(
        model=env.model,
        working_area=(0.4, 0.4),  # 40cm x 40cm working area
        offset_x=0.0,
        offset_y=0.0,
        table_body_name="simple_table",
        table_geom_name="table_surface"
    )
    print(f"   Working area bounds: {domain.get_working_bounds()}")
    print(f"   Table height: {domain.table_height:.3f}m")
    
    # Create state manager
    print("\n4. Creating blocks state manager...")
    state_manager = BlocksStateManager(domain, env)
    
    # Setup initial configuration
    print("\n5. Setting up initial configuration...")
    print("   - 2 small cubes (graspable)")
    print("   - 1 large platform (target base)")
    
    # Hide all blocks first
    for i in range(16):
        state_manager._hide_block(i)
    
    # Select blocks
    if RANDOMIZE:
        cube1_idx = np.random.choice([0, 1, 2])  # Choose from small cubes
        cube2_idx = np.random.choice([3, 4, 5])  # Choose from other small cubes
        platform_idx = np.random.choice([14, 15])  # Choose from large platforms
    else:
        cube1_idx = 0  # Small cube (4cm)
        cube2_idx = 1  # Small cube (4cm)
        platform_idx = 14  # Large platform (15x10x5cm)
    
    # Place platform with some variation in position
    bounds = domain.get_working_bounds()
    center_x = (bounds['min_x'] + bounds['max_x']) / 2.0
    center_y = (bounds['min_y'] + bounds['max_y']) / 2.0
    
    # Place platform
    if RANDOMIZE:
        # Random offset (±5cm) and rotation
        platform_x = center_x + np.random.uniform(-0.05, 0.05)
        platform_y = center_y + np.random.uniform(-0.05, 0.05)
        platform_angle = np.random.uniform(0, 2 * np.pi)
        platform_quat = np.array([
            np.cos(platform_angle / 2),
            0,
            0,
            np.sin(platform_angle / 2)
        ])  # Quaternion for Z-axis rotation
        print(f"   Platform (block_{platform_idx}) placed at ({platform_x:.3f}, {platform_y:.3f}) with angle {np.degrees(platform_angle):.1f}°")
    else:
        # Fixed position at center
        platform_x = center_x
        platform_y = center_y
        platform_quat = None
        print(f"   Platform (block_{platform_idx}) placed at ({platform_x:.3f}, {platform_y:.3f})")
    
    platform_z = domain.table_height + 0.025 + 0.003  # height/2 + clearance
    state_manager._set_block_position(platform_idx, platform_x, platform_y, platform_z, platform_quat)
    
    # Place cube 1
    if RANDOMIZE:
        # Random position (10-15cm away, random angle)
        angle1 = np.random.uniform(0, 2 * np.pi)
        distance1 = np.random.uniform(0.10, 0.15)
        cube1_x = platform_x + distance1 * np.cos(angle1)
        cube1_y = platform_y + distance1 * np.sin(angle1)
        # Ensure cube1 is in bounds
        cube1_x = np.clip(cube1_x, bounds['min_x'] + 0.03, bounds['max_x'] - 0.03)
        cube1_y = np.clip(cube1_y, bounds['min_y'] + 0.03, bounds['max_y'] - 0.03)
    else:
        # Fixed position to the left
        cube1_x = platform_x - 0.12
        cube1_y = platform_y
    
    cube1_z = domain.table_height + 0.02 + 0.003  # height/2 + clearance
    state_manager._set_block_position(cube1_idx, cube1_x, cube1_y, cube1_z)
    print(f"   Cube 1 (block_{cube1_idx}) placed at ({cube1_x:.3f}, {cube1_y:.3f})")
    
    # Place cube 2
    if RANDOMIZE:
        # Random position (opposite side with variation)
        angle2 = angle1 + np.pi + np.random.uniform(-0.5, 0.5)  # Roughly opposite side
        distance2 = np.random.uniform(0.10, 0.15)
        cube2_x = platform_x + distance2 * np.cos(angle2)
        cube2_y = platform_y + distance2 * np.sin(angle2)
        # Ensure cube2 is in bounds
        cube2_x = np.clip(cube2_x, bounds['min_x'] + 0.03, bounds['max_x'] - 0.03)
        cube2_y = np.clip(cube2_y, bounds['min_y'] + 0.03, bounds['max_y'] - 0.03)
    else:
        # Fixed position to the right
        cube2_x = platform_x + 0.12
        cube2_y = platform_y
    
    cube2_z = domain.table_height + 0.02 + 0.003  # height/2 + clearance
    state_manager._set_block_position(cube2_idx, cube2_x, cube2_y, cube2_z)
    print(f"   Cube 2 (block_{cube2_idx}) placed at ({cube2_x:.3f}, {cube2_y:.3f})")
    
    # Update physics
    env.reset_velocities()
    env.forward()
    
    # Execute with viewer
    with env.launch_viewer() as viewer:
        # Let physics settle
        print("\n6. Letting physics settle...")
        env.rest(2.0)
        
        if not viewer.is_running():
            return
        
        # Ground state and generate PDDL
        print("\n7. Grounding symbolic state...")
        state = state_manager.ground_state()
        print(f"   Active blocks: {list(state['blocks'].keys())}")
        print(f"   On table: {state['on_table']}")
        print(f"   Clear blocks: {state['clear']}")
        
        print("\n8. Generating PDDL problem...")
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        problem_path = _OUTPUT_DIR / "problem_stack_demo_real.pddl"
        
        goal_predicates = [
            f"(on block_{cube1_idx} block_{platform_idx})",
            f"(on block_{cube2_idx} block_{platform_idx})"
        ]
        
        state_manager.generate_pddl_problem(
            problem_name="stack-demo-real",
            output_path=problem_path,
            goal_predicates=goal_predicates
        )
        print(f"   PDDL problem saved to: {problem_path}")
        print(f"   Goal: Stack both cubes on platform")
        
        # Create pick-place controller
        print("\n9. Setting up pick-and-place controller...")
        controller = PickPlaceController(env, planner, state_manager)
        
        # Queue all actions
        print("\n10. Queuing pick-and-stack actions...")
        print("   Plan: pick-from-table(cube1) -> stack(cube1, platform)")
        print("         pick-from-table(cube2) -> stack(cube2, platform)")
        
        print(f"\n{'='*70}")
        print(f"ACTION 1: pick-from-table(block_{cube1_idx})")
        print(f"{'='*70}")
        controller.execute_pick_from_table(cube1_idx)
        
        print(f"\n{'='*70}")
        print(f"ACTION 2: stack(block_{cube1_idx}, block_{platform_idx})")
        print(f"{'='*70}")
        controller.execute_stack(cube1_idx, platform_idx, custom_offset=(-0.03, 0))
        
        print(f"\n{'='*70}")
        print(f"ACTION 3: pick-from-table(block_{cube2_idx})")
        print(f"{'='*70}")
        controller.execute_pick_from_table(cube2_idx)
        
        print(f"\n{'='*70}")
        print(f"ACTION 4: stack(block_{cube2_idx}, block_{platform_idx})")
        print(f"{'='*70}")
        controller.execute_stack(cube2_idx, platform_idx, custom_offset=(0.04, 0))
        
        # Main execution loop
        print("\n11. Executing actions...")
        dt = 0.005
        
        while viewer.is_running() and controller.is_busy():
            controller.update(dt)
            env.controller.step()
            env.step()
        
        if not viewer.is_running():
            return
        
        # Let physics settle
        print("\n   Letting physics settle...")
        env.rest(2.0)
        
        # Verify final state
        print("\n12. Verifying final state...")
        final_state = state_manager.ground_state()
        print(f"   Active blocks: {list(final_state['blocks'].keys())}")
        print(f"   On table: {final_state['on_table']}")
        print(f"   Clear blocks: {final_state['clear']}")
        print(f"   On relationships: {final_state['on']}")
        
        # Check if goal is achieved
        goal_achieved = all([
            (f'block_{cube1_idx}', f'block_{platform_idx}') in final_state['on'],
            (f'block_{cube2_idx}', f'block_{platform_idx}') in final_state['on']
        ])
        
        print("\n" + "=" * 70)
        if goal_achieved:
            print("✓ SUCCESS: Goal achieved! Both cubes are on the platform.")
        else:
            print("⚠ PARTIAL: Some goals may not be fully satisfied.")
            print("  This could be due to:")
            print("  - XY overlap threshold not met (need 80% overlap)")
            print("  - Physics settling time insufficient")
            print("  - Block positions after placement")
        print("=" * 70)
        
        print("\n13. Keeping viewer open for inspection...")
        print("    Close viewer window to exit.")
        
        # Keep running until viewer is closed
        while viewer.is_running():
            env.step()
            time.sleep(0.01)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
