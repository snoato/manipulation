from pathlib import Path
import time

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
        targets = ["cylinder1", "cylinder2", "cylinder3"]
        target = None

        cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder2")
        target_pos = data.xpos[cylinder_id].copy()
        while viewer.is_running():
            # Update our local time
            dt = env.step()

            # high-level control logic


            # low-level control logic
            

            # set target
            # if env.sim_time > 2.0:
            #     ik.set_target_position(np.array([0.395, 0.300, 0.571]), np.array([-0.5, 0.5, 0.5, 0.5]))
            # if env.sim_time > 4.0:
            #     ik.set_target_position(np.array([0.395, 0.300, 0.571]), np.array([-0.5, 0.5, 0.5, 0.5]))
            # if env.sim_time > 6.0:
            #     ik.set_target_position(np.array([0.295, 0.550, 0.371]), np.array([-0.5, 0.5, 0.5, 0.5]))
            # if env.sim_time > 8.0:
            #     ik.set_target_position(np.array([0.495, 0.450, 0.471]), np.array([-0.5, 0.5, 0.5, 0.5]))
            # if env.sim_time > 10.0:
            #     ik.set_target_position(np.array([0.395, 0.400, 0.571]), np.array([-0.5, 0.5, 0.5, 0.5]))
            # if env.sim_time > 12.0:
            #     ik.set_target_position(np.array([0.225, 0.550, 0.371]), np.array([-0.5, 0.5, 0.5, 0.5]))
            if env.sim_time > 0.0 and target is None:
                target = targets[0]
                cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
                target_pos = data.xpos[cylinder_id].copy()
            if env.sim_time > 2.0:
                ik.set_target_position(target_pos-np.array([0, 0.1, -0.1]), np.array([-0.5, 0.5, 0.5, 0.5]))
            if env.sim_time > 4.0:
                ik.set_target_position(target_pos-np.array([0, 0.01, 0.0]), np.array([-0.5, 0.5, 0.5, 0.5]))
            if env.sim_time > 6.0:
                data.ctrl[7] = ik.configuration.q[8] = -0.2  # close gripper
            if env.sim_time > 8.0:
                ik.set_target_position(target_pos+np.array([0, 0.0, 0.2]), np.array([-0.5, 0.5, 0.5, 0.5]))
            if env.sim_time > 10.0:
                ik.set_target_position(target_pos+np.array([0, 0.0, 0.2])-np.array([0, 0.4, 0]), np.array([-0.5, 0.5, 0.5, 0.5]))
            if env.sim_time > 12.0:
                data.ctrl[7] = ik.configuration.q[8] = 0.4  # close gripper
                env.sim_time = 0.0  # reset time
                target = None
                if len(targets) > 1:
                    targets = targets[1:]
                else:
                    targets = []
                    time.sleep(2.0)



            # Compute target and control
            ik.converge_ik(dt)
            data.ctrl[:7] = ik.configuration.q[:7]



main()