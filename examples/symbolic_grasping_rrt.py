"""Example combining symbolic planning with grasping RRT."""

from pathlib import Path
import time
import numpy as np

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
_PROBLEM_DIR = _HERE / ".." / "manipulation" / "symbolic" / "problems"
_VIZ_DIR = _HERE / ".." / "manipulation" / "symbolic" / "viz"


def main():
    print("=" * 70)
    print("Symbolic Planning + Grasping RRT Example")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Initialize RRT* planner
    print("2. Initializing RRT* planner...")
    planner = RRTStar(env)
    planner.max_iterations = 1000
    planner.step_size = 0.2
    planner.goal_sample_rate = 0.2
    
    # Create grid domain (20x20 cells = 2cm resolution)
    print("3. Creating grid domain...")
    grid = GridDomain(
        model=env.model,
        cell_size=0.04,  # 2cm cells
        working_area=(0.4, 0.3),  # 40cm x 40cm
        table_body_name="simple_table",
        table_geom_name="table_surface"
    )
    
    info = grid.get_grid_info()
    print(f"   Grid: {info['grid_dimensions'][0]}x{info['grid_dimensions'][1]} cells ({info['cell_size']*100:.1f}cm)")
    
    # Create state manager
    state_manager = StateManager(grid, env)
    
    # Create output directories
    _PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    config_num = 0
    
    with env.launch_viewer() as viewer:
        targets = []
        target = None
        step = 0
        
        while viewer.is_running():
            # Generate new configuration if no targets
            if target is None and len(targets) == 0:
                config_num += 1
                print(f"\n{'='*70}")
                print(f"Configuration {config_num}: Generating random problem...")
                print(f"{'='*70}")
                
                # Sample random state
                state_manager.sample_random_state(n_cylinders=5)
                
                # Ground and visualize state
                grounded_state = state_manager.ground_state()
                print(f"   Active cylinders: {len(grounded_state['cylinders'])}")
                
                # Generate PDDL problem
                problem_path = _PROBLEM_DIR / f"config_{config_num}.pddl"
                state_manager.generate_pddl_problem(f"config-{config_num}", problem_path)
                print(f"   Saved PDDL: {problem_path}")
                
                # Visualize
                viz_path = _VIZ_DIR / f"config_{config_num}.png"
                visualize_grid_state(
                    state_manager,
                    save_path=viz_path,
                    title=f"Configuration {config_num}"
                )
                print(f"   Saved visualization: {viz_path}")
                
                # Build target list (all active cylinders)
                targets = list(grounded_state['cylinders'].keys())
                print(f"   Targets to grasp: {targets}")
                
                target = None
                env.rest(2.0)
            
            dt = env.step()
            
            # Start grasping sequence
            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target = targets[0]
                step = 0
            
            # Execute grasping sequence
            if target is not None:
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    
                    if step == 1:
                        print(f"\n  [{target}] Planning approach...")
                        target_pos = env.get_object_position(target)  # Get fresh position
                        target_pose, target_orientation = env.get_approach_pose(target_pos)
                        env.add_collision_exception(target)  # Ignore collisions with target object
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=2000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("    Failed to plan approach!")
                            step = 9
                    
                    elif step == 2:
                        print(f"  [{target}] Planning grasp...")
                        target_pos = env.get_object_position(target)  # Get fresh position
                        target_pose, target_orientation = env.get_grasp_pose(target_pos)
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=1000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.02  # Slower approach for more precision
                            )
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("    Failed to plan grasp!")
                            step = 9
                    
                    elif step == 3:
                        print(f"  [{target}] Waiting for arm to stabilize...")
                        # Check if arm is stationary before grasping
                        arm_velocity = np.linalg.norm(env.data.qvel[:7])
                        if arm_velocity < 0.01:  # Threshold for "stationary"
                            step += 1  # Move to next step immediately
                            env.controller.close_gripper()
                            print(f"  [{target}] Closing gripper...")
                        else:
                            step -= 1  # Stay in this step
                    
                    elif step == 4:
                        pass  # Gripper closing (handled by controller status)
                    
                    elif step == 5:
                        print(f"  [{target}] Planning lift...")
                        target_pos = env.get_object_position(target)  # Get fresh position
                        target_pose, target_orientation = env.get_lift_pose(target_pos)
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=2000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("    Failed to plan lift!")
                            step = 9
                    
                    elif step == 6:
                        print(f"  [{target}] Planning dropoff...")
                        target_pose, target_orientation = env.get_dropoff_pose()
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=3000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("    Failed to plan dropoff!")
                            step = 9
                    
                    elif step == 7:
                        print(f"  [{target}] Opening gripper...")
                        env.controller.open_gripper()
                        env.clear_collision_exceptions()
                    
                    elif step == 8:
                        print(f"  [{target}] Hiding object...")
                        # Move cylinder far away to avoid accidental collisions
                        env.set_object_pose(target, np.array([100.0, 0.0, 0.0]))
                    
                    elif step == 9:
                        print(f"  [{target}] Complete!")
                        target = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            print(f"\n  All cylinders cleared! Waiting 2 seconds...")
                            time.sleep(2.0)
                
                # Rest after gripper operations
                if step == 4 or step == 7:
                    env.rest(1.0)
            
            env.controller.step()
    
    print(f"\n{'='*70}")
    print(f"Generated {config_num} configurations")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
