"""
Example demonstrating RRT* motion planning with the Franka Panda robot.
"""
import time
import numpy as np
from env import FrankaEnvironment
from mp import MotionPlanner


def main():
    # Initialize environment
    print("Initializing environment...")
    env = FrankaEnvironment("envs/franka_emika_panda/scene.xml")
    
    # Create motion planner
    print("Creating motion planner...")
    planner = MotionPlanner(env)
    
    # Define start and goal configurations
    # Start: Home position
    start_config = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    
    # Goal: Different configuration
    goal_config = np.array([0.5, -0.3, 0.2, -1.2, 0.3, 1.8, -0.5])
    
    print(f"\nStart configuration: {start_config}")
    print(f"Goal configuration: {goal_config}")
    
    # Plan path
    print("\n" + "="*60)
    path = planner.plan(start_config, goal_config, max_iterations=3000)
    print("="*60)
    
    if path is None:
        print("\nFailed to find a path!")
        return
    
    # Smooth the path
    print("\nSmoothing path...")
    smoothed_path = planner.smooth_path(path, max_iterations=50)
    print(f"Path smoothed: {len(path)} -> {len(smoothed_path)} waypoints")
    
    # Interpolate for smoother motion
    print("\nInterpolating path...")
    interpolated_path = env.controller.interpolate_linear_path(smoothed_path, steps_per_segment=5)
    print(f"Path interpolated: {len(smoothed_path)} -> {len(interpolated_path)} waypoints")
    
    # Visualize the path
    print("\nLaunching viewer to visualize path...")
    viewer = env.launch_viewer()
    
    # Execute the path
    print("\nExecuting path...")
    for i, config in enumerate(interpolated_path):
        env.data.qpos[:7] = config
        env.step()
        
        time.sleep(0.5)

        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(interpolated_path)} waypoints")
    
    print("\nPath execution complete!")
    print("Close the viewer window to exit.")
    
    # Keep viewer open
    while viewer.is_running():
        env.step()


if __name__ == "__main__":
    main()
