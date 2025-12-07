"""Example demonstrating blocks world symbolic planning with randomized problem."""

from pathlib import Path
import time
import numpy as np

from manipulation import FrankaEnvironment
from manipulation.symbolic.domains.blocks import BlocksDomain, BlocksStateManager

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_blocks.xml"
_DOMAIN_DIR = _HERE / ".." / "manipulation" / "symbolic" / "domains" / "blocks" / "pddl"
_OUTPUT_DIR = _HERE / "blocks_output"


def main():
    print("=" * 70)
    print("Blocks World Demo - Stacking Two Cubes on a Platform")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment with blocks scene...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Create blocks domain (continuous working area, no grid)
    print("\n2. Creating blocks domain...")
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
    print("\n3. Creating blocks state manager...")
    state_manager = BlocksStateManager(domain, env)
    
    # Sample random initial state
    print("\n4. Sampling random initial state...")
    print("   - 2 small cubes (graspable)")
    print("   - 1 large platform (target base)")
    
    # Hide all blocks first
    for i in range(16):
        state_manager._hide_block(i)
    
    # Manually place blocks for controlled demo
    # Select 2 small cubes and 1 large platform
    cube1_idx = 0  # Small cube (4cm)
    cube2_idx = 1  # Small cube (4cm)
    platform_idx = 14  # Large platform (15x10x5cm)
    
    # Place platform at center of working area
    bounds = domain.get_working_bounds()
    platform_x = (bounds['min_x'] + bounds['max_x']) / 2.0
    platform_y = (bounds['min_y'] + bounds['max_y']) / 2.0
    platform_z = domain.table_height + 0.025 + 0.003  # height/2 + clearance
    state_manager._set_block_position(platform_idx, platform_x, platform_y, platform_z)
    print(f"   Platform (block_{platform_idx}) placed at ({platform_x:.3f}, {platform_y:.3f})")
    
    # Place cube 1 to the left
    cube1_x = platform_x - 0.12
    cube1_y = platform_y
    cube1_z = domain.table_height + 0.02 + 0.003  # height/2 + clearance
    state_manager._set_block_position(cube1_idx, cube1_x, cube1_y, cube1_z)
    print(f"   Cube 1 (block_{cube1_idx}) placed at ({cube1_x:.3f}, {cube1_y:.3f})")
    
    # Place cube 2 to the right
    cube2_x = platform_x + 0.12
    cube2_y = platform_y
    cube2_z = domain.table_height + 0.02 + 0.003  # height/2 + clearance
    state_manager._set_block_position(cube2_idx, cube2_x, cube2_y, cube2_z)
    print(f"   Cube 2 (block_{cube2_idx}) placed at ({cube2_x:.3f}, {cube2_y:.3f})")
    
    # Update physics
    env.reset_velocities()
    env.forward()
    
    # Run setup and execute actions with viewer
    with env.launch_viewer() as viewer:
        # Let physics settle
        print("\n5. Letting physics settle...")
        for _ in range(100):
            if not viewer.is_running():
                return
            env.step()
            time.sleep(0.01)
        
        # Ground current state
        print("\n6. Grounding symbolic state...")
        state = state_manager.ground_state()
        print(f"   Active blocks: {list(state['blocks'].keys())}")
        print(f"   On table: {state['on_table']}")
        print(f"   Clear blocks: {state['clear']}")
        print(f"   On relationships: {state['on']}")
        
        # Generate PDDL problem file
        print("\n7. Generating PDDL problem file...")
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        problem_path = _OUTPUT_DIR / "problem_stack_demo.pddl"
        
        goal_predicates = [
            f"(on block_{cube1_idx} block_{platform_idx})",
            f"(on block_{cube2_idx} block_{platform_idx})"
        ]
        
        state_manager.generate_pddl_problem(
            problem_name="stack-demo",
            output_path=problem_path,
            goal_predicates=goal_predicates
        )
        print(f"   PDDL problem saved to: {problem_path}")
        print(f"   Goal: Both cubes should be on the platform")
        
        # Display PDDL problem
        with open(problem_path, 'r') as f:
            pddl_content = f.read()
        print("\n" + "=" * 70)
        print("Generated PDDL Problem:")
        print("=" * 70)
        print(pddl_content)
        print("=" * 70)
        
        # Execute pick-and-place actions to achieve goal
        print("\n8. Executing pick-and-place actions (no high-level planner)...")
        print("   Plan: pick cube1 -> stack on platform -> pick cube2 -> stack on platform")
        
        # Action 1: Pick cube 1 from table
        print(f"\n   Action 1: Pick block_{cube1_idx} from table")
        pickup_pose1, pickup_quat1 = state_manager.compute_pickup_pose(cube1_idx)
        print(f"   Pickup pose: pos={pickup_pose1}, quat={pickup_quat1}")
        
        # Simulate picking (for now, just update internal state)
        # In a full implementation, this would use IK + motion planning
        state_manager.gripper_holding = cube1_idx
        state_manager._hide_block(cube1_idx)  # Temporarily hide while "holding"
        print(f"   [Simulated] Gripper now holding block_{cube1_idx}")
        
        # Action 2: Stack cube 1 on platform
        print(f"\n   Action 2: Stack block_{cube1_idx} on block_{platform_idx}")
        stack_pose1, stack_quat1 = state_manager.compute_stack_pose(platform_idx, cube1_idx)
        print(f"   Stack pose: pos={stack_pose1}, quat={stack_quat1}")
        
        # Place cube on platform
        state_manager._set_block_position(cube1_idx, stack_pose1[0], stack_pose1[1], stack_pose1[2] - 0.01)
        state_manager.gripper_holding = None
        env.reset_velocities()
        env.forward()
        print(f"   [Simulated] Placed block_{cube1_idx} on block_{platform_idx}")
        
        # Let physics settle
        for _ in range(100):
            if not viewer.is_running():
                return
            env.step()
            time.sleep(0.01)
        
        # Action 3: Pick cube 2 from table
        print(f"\n   Action 3: Pick block_{cube2_idx} from table")
        pickup_pose2, pickup_quat2 = state_manager.compute_pickup_pose(cube2_idx)
        print(f"   Pickup pose: pos={pickup_pose2}, quat={pickup_quat2}")
        
        state_manager.gripper_holding = cube2_idx
        state_manager._hide_block(cube2_idx)
        print(f"   [Simulated] Gripper now holding block_{cube2_idx}")
        
        # Action 4: Stack cube 2 on platform (next to cube 1)
        print(f"\n   Action 4: Stack block_{cube2_idx} on block_{platform_idx}")
        
        # Compute stack position offset from cube 1 to avoid collision
        cube1_pos, _ = state_manager._get_block_pose(cube1_idx)
        stack_x = platform_x + 0.04  # Offset to the right of platform center
        stack_y = platform_y
        stack_z = platform_z + 0.025 + 0.02  # Platform half-height + cube half-height
        
        print(f"   Stack pose: pos=({stack_x:.3f}, {stack_y:.3f}, {stack_z:.3f})")
        
        state_manager._set_block_position(cube2_idx, stack_x, stack_y, stack_z - 0.01)
        state_manager.gripper_holding = None
        env.reset_velocities()
        env.forward()
        print(f"   [Simulated] Placed block_{cube2_idx} on block_{platform_idx}")
        
        # Let physics settle
        print("\n9. Letting physics settling after stacking...")
        for _ in range(200):
            if not viewer.is_running():
                return
            env.step()
            time.sleep(0.01)
        
        # Verify final state
        print("\n10. Verifying final state...")
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
            print("SUCCESS: Goal achieved! Both cubes are on the platform.")
        else:
            print("PARTIAL: Some goals may not be fully satisfied (check physics).")
        print("=" * 70)
        
        print("\n11. Keeping viewer open for inspection...")
        print("    Close viewer window or press Ctrl+C to exit.")
        
        # Keep running until viewer is closed
        while viewer.is_running():
            env.step()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
