"""Example demonstrating grasping with IK-based control (no motion planning)."""

from pathlib import Path
import time

from manipulation import FrankaEnvironment, ControllerStatus

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_test.xml"


def main():
    # Load model and data
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    ik = env.get_ik()

    with env.launch_viewer() as viewer:
        targets = []
        target = None

        while viewer.is_running():
            if target is None and len(targets) == 0:
                targets = ["cylinder1", "cylinder2", "cylinder3"]
                target = None
                env.reset()
                env.rest(2.0)

            # Update our local time
            dt = env.step()

            # High-level control logic
            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target = targets[0]
                target_pos = env.get_object_position(target)
                step = 0 
            
            if target is not None: 
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        print(f"Approaching {target}...")
                        target_pose, target_orientation = env.get_approach_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 2:
                        print("Moving to grasp pose...")
                        target_pose, target_orientation = env.get_grasp_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 3:
                        print("Closing gripper...")
                        env.controller.close_gripper()

                    if step == 4:
                        print("Lifting object...")
                        target_pose, target_orientation = env.get_lift_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 5:
                        print("Moving to dropoff...")
                        target_pose, target_orientation = env.get_dropoff_pose()
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])

                    if step == 6:
                        print("Opening gripper...")
                        env.controller.open_gripper()
                    
                    if step == 7:
                        print(f"Completed {target}!")
                        target = None
                        step = 0
                        if len(targets) > 1:
                            targets = targets[1:]
                        else:
                            targets = []
                            time.sleep(2.0)

                if step == 3 or step == 6:
                    env.rest(2.0)

            # Low-level control logic
            env.controller.step()


if __name__ == "__main__":
    main()
