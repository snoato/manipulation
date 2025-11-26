from pathlib import Path
import time

from controller import ControllerStatus
from ik import InverseKinematics
from env import FrankaEnvironment
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "envs" / "franka_emika_panda" / "scene_test.xml"


def main():
    # load model and data
    env = FrankaEnvironment(_XML.as_posix(), rate = 200.0)
    model = env.get_model()
    data = env.get_data()
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

            # high-level control logic
            if env.sim_time > 0.0 and target is None and len(targets) > 0:
                target = targets[0]
                target_pos = env.get_object_position(target)
                step = 0 
            if target is not None: 
                if env.controller.get_status() == ControllerStatus.IDLE:
                    step += 1
                    print(f"Time: {env.sim_time:.2f}, Step: {step}")

                    if step == 1:
                        target_pose, target_orientation = env.get_approach_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 2:
                        target_pose, target_orientation = env.get_grasp_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 3:
                        env.controller.close_gripper()

                    if step == 4:
                        target_pose, target_orientation = env.get_lift_pose(target_pos)
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])
                    
                    if step == 5:
                        target_pose, target_orientation = env.get_dropoff_pose()
                        ik.set_target_position(target_pose, target_orientation)
                        converged = ik.converge_ik(dt)
                        print("IK result:", "Converged" if converged else "Not converged")
                        env.controller.move_to_incremental(ik.configuration.q[:7])

                    if step == 6:
                        env.controller.open_gripper()
                    
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

            # low-level control logic
            env.controller.step()


main()