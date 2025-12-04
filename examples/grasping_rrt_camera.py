"""Example demonstrating grasping with RRT* and camera capture (headless)."""

from pathlib import Path

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_test.xml"
_OUTPUT_DIR = _HERE / "grasp_captures"


def main():
    # Create output directory for images
    _OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving images to: {_OUTPUT_DIR}")
    
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Initialize RRT* planner
    planner = RRTStar(env)
    planner.max_iterations = 1000
    planner.step_size = 0.2
    planner.goal_sample_rate = 0.2

    # Run headless (no viewer)
    targets = ["cylinder1", "cylinder2", "cylinder3"]
    target = None
    capture_count = 0
    running = True
    step = 0
    
    print(f"\nStarting grasping sequence for {len(targets)} objects...")
    print("="*60)
    
    env.rest(2.0)
    
    while running:
        if target is None and len(targets) == 0:
            print("\n" + "="*60)
            print("All objects grasped! Sequence complete.")
            print("="*60)
            running = False
            break

        dt = env.step()

        if env.sim_time > 0.0 and target is None and len(targets) > 0:
            target = targets[0]
            target_pos = env.get_object_position(target)
            step = 0
            print(f"\n>>> Target: {target}")
        
        if target is not None: 
            if env.controller.get_status() == ControllerStatus.IDLE:
                step += 1

                if step == 1:
                    print(f"  [1/7] Planning approach to {target}...")
                    target_pose, target_orientation = env.get_approach_pose(target_pos)
                    path = planner.plan_to_pose(
                        target_pose, target_orientation, dt=dt, max_iterations=2000
                    )
                    
                    if path is not None:
                        print(f"        Path found! Executing...")
                        smoothed = planner.smooth_path(path)
                        interpolated = env.controller.interpolate_linear_path(
                            smoothed, step_size=0.05
                        )
                        env.controller.follow_trajectory(interpolated)
                    else:
                        print("        ✗ Failed to plan approach path!")
                        step = 7
                
                if step == 2:
                    print(f"  [2/7] Planning grasp...")
                    target_pose, target_orientation = env.get_grasp_pose(target_pos)
                    path = planner.plan_to_pose(
                        target_pose, target_orientation, dt=dt, max_iterations=1000
                    )
                    
                    if path is not None:
                        smoothed = planner.smooth_path(path)
                        interpolated = env.controller.interpolate_linear_path(
                            smoothed, step_size=0.05
                        )
                        env.controller.follow_trajectory(interpolated)
                    else:
                        print("        ✗ Failed to plan grasp path!")
                        step = 7
                
                if step == 3:
                    print(f"  [3/7] Closing gripper...")
                    env.controller.close_gripper()

                if step == 4:
                    print(f"  [4/7] Planning lift...")
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
                        env.controller.follow_trajectory(interpolated)
                    else:
                        print("        ✗ Failed to plan lift path!")
                        step = 7
                
                if step == 5:
                    print(f"  [5/7] Planning dropoff...")
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
                        print("        ✗ Failed to plan dropoff path!")
                        step = 7

                if step == 6:
                    print(f"  [6/7] Opening gripper...")
                    env.controller.open_gripper()
                    env.clear_collision_exceptions()
                    
                    # Capture image after successful grasp
                    capture_count += 1
                    print(f"  [7/7] 📸 Capturing images...")
                    
                    # Capture from all three cameras
                    for cam_name in ["top_camera", "side_camera", "front_camera"]:
                        filename = _OUTPUT_DIR / f"grasp_{capture_count:02d}_{target}_{cam_name}.png"
                        env.save_camera_image(cam_name, str(filename))
                    
                    print(f"        ✓ Saved 3 images (grasp_{capture_count:02d}_*)")
                
                if step == 7:
                    target = None
                    step = 0
                    if len(targets) > 1:
                        targets = targets[1:]
                    else:
                        targets = []

            if step == 3 or step == 6:
                env.rest(2.0)

        env.controller.step()
    
    env.close()
    print(f"\n✓ Total images captured: {capture_count * 3}")
    print(f"✓ Saved to: {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
