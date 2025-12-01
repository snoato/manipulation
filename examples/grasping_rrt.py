"""Example demonstrating grasping with RRT* motion planning."""

from pathlib import Path
import time

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_test.xml"


def main():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Initialize RRT* planner
    planner = RRTStar(env)
    planner.max_iterations = 1000
    planner.step_size = 0.2
    planner.goal_sample_rate = 0.2

    with env.launch_viewer() as viewer:
        targets = []
        target = None

        while viewer.is_running():
            if target is None and len(targets) == 0:
                targets = ["cylinder1", "cylinder2", "cylinder3"]
                target = None
                env.reset()
                env.rest(2.0)

            dt = env.step()

            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target = targets[0]
                target_pos = env.get_object_position(target)
                step = 0 
            
            if target is not None: 
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        print(f"Planning approach to {target}...")
                        target_pose, target_orientation = env.get_approach_pose(target_pos)
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=2000
                        )
                        
                        if path is not None:
                            print("Path found! Smoothing and executing...")
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            print(f"Executing path with {len(interpolated)} waypoints")
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("Failed to plan approach path!")
                            step = 7
                    
                    if step == 2:
                        print("Planning grasp...")
                        target_pose, target_orientation = env.get_grasp_pose(target_pos)
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=1000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            print(f"Executing path with {len(interpolated)} waypoints")
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("Failed to plan grasp path!")
                            step = 7
                    
                    if step == 3:
                        print("Closing gripper...")
                        env.controller.close_gripper()

                    if step == 4:
                        print("Planning lift...")
                        target_pose, target_orientation = env.get_lift_pose(target_pos)
                        env.add_collision_exception(target)
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=2000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            print(f"Executing path with {len(interpolated)} waypoints")
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("Failed to plan lift path!")
                            step = 7
                    
                    if step == 5:
                        print("Planning dropoff...")
                        target_pose, target_orientation = env.get_dropoff_pose()
                        path = planner.plan_to_pose(
                            target_pose, target_orientation, dt=dt, max_iterations=3000
                        )
                        
                        if path is not None:
                            smoothed = planner.smooth_path(path)
                            interpolated = env.controller.interpolate_linear_path(
                                smoothed, step_size=0.05
                            )
                            print(f"Executing path with {len(interpolated)} waypoints")
                            env.controller.follow_trajectory(interpolated)
                        else:
                            print("Failed to plan dropoff path!")
                            step = 7

                    if step == 6:
                        print("Opening gripper...")
                        env.controller.open_gripper()
                        env.clear_collision_exceptions()
                    
                    if step == 7:
                        target = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            time.sleep(2.0)

                if step == 3 or step == 6:
                    env.rest(2.0)

            env.controller.step()


if __name__ == "__main__":
    main()
