"""Example demonstrating RRT* motion planning with the Franka Panda robot."""

from pathlib import Path
import numpy as np

from tampanda import FrankaEnvironment, RRTStar, SCENE_DEFAULT

_XML = SCENE_DEFAULT


def main():
    print("Initializing environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    print("Creating motion planner...")
    planner = RRTStar(env)

    start_config = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    goal_config  = np.array([0.5, -0.3, 0.2, -1.2, 0.3, 1.8, -0.5])

    print(f"\nStart: {start_config}")
    print(f"Goal:  {goal_config}")

    print("\n" + "=" * 60)
    path = planner.plan(start_config, goal_config, max_iterations=3000)
    print("=" * 60)

    if path is None:
        print("\nFailed to find a path!")
        return

    print(f"\nPath found: {len(path)} waypoints")

    with env.launch_viewer() as viewer:
        env.reset()

        print("\nExecuting path...")
        env.execute_path(path, planner)
        env.wait_idle()
        print("Path complete — holding for 3 seconds...")
        env.rest(3.0)

        print("Close the viewer window to exit.")
        while viewer.is_running():
            env.step()


if __name__ == "__main__":
    main()
