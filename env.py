from controller import Controller
import mujoco
import numpy as np
from ik import InverseKinematics
from loop_rate_limiters import RateLimiter

class FrankaEnvironment:
    def __init__(self, path, rate=200.0):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ik = InverseKinematics(self.model, self.data)
        self.controller = Controller(self.model, self.data)

        # set initial pos
        self.data.qpos[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])
        self.data.ctrl[:8] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])
        self.ik.update_configuration(self.data.qpos)
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

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

    def reset(self):
        self.data.ctrl[:] = self.initial_ctrl
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.ik.update_configuration(self.data.qpos)
        self.sim_time = 0.0

    def step(self):
        mujoco.mj_step(self.model, self.data)

        dt = self.rate.dt
        self.sim_time += dt

        self.viewer.sync()
        self.rate.sleep()
        return dt

    def rest(self, duration):
        steps = int(duration / self.rate.dt)
        for _ in range(steps):
            self.step()

    def get_object_id(self, object_name):
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if object_id == -1:
            raise ValueError(f"Object '{object_name}' not found in the model.")
        return object_id

    def get_object_position(self, object_name):
        object_id = self.get_object_id(object_name)
        return self.data.xpos[object_id].copy()
    
    def get_object_orientation(self, object_name):
        object_id = self.get_object_id(object_name)
        return self.data.xquat[object_id].copy()

    def get_approach_pose(self, target, offset=np.array([0, -0.1, 0.1]), orientation=np.array([-0.5, 0.5, 0.5, 0.5])):
        pos = target + offset
        return pos, orientation
    
    def get_grasp_pose(self, target, offset=np.array([0, 0, 0.03]), orientation=np.array([-0.5, 0.5, 0.5, 0.5])):
        pos = target + offset
        return pos, orientation
    
    def get_lift_pose(self, target, offset=np.array([0, 0, 0.2]), orientation=np.array([-0.5, 0.5, 0.5, 0.5])):
        pos = target + offset
        return pos, orientation
    
    def get_dropoff_pose(self):
        return np.array([0.5, 0, 0.5]), np.array([0, 1, 0, 0])
# need to generate drop off positions for objects
# need to randomize object positions
# need to translate object positions into PDDL and back
# need to get list of objects in the scene (potentially filtered)

