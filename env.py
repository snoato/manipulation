import mujoco
import numpy as np
from ik import InverseKinematics
from loop_rate_limiters import RateLimiter

class FrankaEnvironment:
    def __init__(self, path, rate=200.0):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ik = InverseKinematics(self.model, self.data)

        # set initial pos
        self.data.qpos[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
        self.data.ctrl[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
        self.ik.update_configuration(self.data.qpos)

        self.rate = RateLimiter(frequency=rate, warn=False)

    def get_model(self):
        return self.model
    
    def get_data(self):
        return self.data

    def get_ik(self):
        return self.ik
    
    def launch_viewer(self):
        self.sim_time = 0.0
        self.viewer = mujoco.viewer.launch_passive(model=self.model, data=self.data, show_left_ui=False, show_right_ui=False)
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        mujoco.mj_forward(self.model, self.data)
        return self.viewer

    def step(self):
        mujoco.mj_step(self.model, self.data)

        dt = self.rate.dt
        self.sim_time += dt

        self.viewer.sync()
        self.rate.sleep()
        return dt
