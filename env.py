from controller import Controller
import mujoco
import numpy as np
from ik import InverseKinematics
from loop_rate_limiters import RateLimiter

class FrankaEnvironment:
    def __init__(self, path, rate=200.0, collision_bodies=None):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ik = InverseKinematics(self.model, self.data)
        self.controller = Controller(self.model, self.data)
        
        # List of body names to check for collisions (empty means check all)
        if collision_bodies is not None:
            self.collision_bodies = collision_bodies
        else:
            self.collision_bodies = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "hand", "right_finger", "left_finger"]

        self.collision_exceptions = []

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

    def add_collision_exception(self, body_name):
        if body_name not in self.collision_exceptions:
            self.collision_exceptions.append(body_name)

    def remove_collision_exception(self, body_name):
        if body_name in self.collision_exceptions:
            self.collision_exceptions.remove(body_name)
    
    def clear_collision_exceptions(self):
        self.collision_exceptions = []

    def check_collisions(self):
        collision_free = True
        if self.data.ncon > 0:
            # Check each contact
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # Get body names for the geometries
                body1 = self.model.geom_bodyid[contact.geom1]
                name1 = self.model.body(body1).name
                body2 = self.model.geom_bodyid[contact.geom2]
                name2 = self.model.body(body2).name

                # Ignore collisions involving bodies in the exceptions list
                if name1 in self.collision_exceptions or name2 in self.collision_exceptions:
                    continue
                
                # Ignore self-collisions within the collision list (typically parts of the robot)
                if name1 in self.collision_bodies and name2 in self.collision_bodies:
                    continue
                # Ignore collisions if neither body is in the collision_bodies list
                if name1 not in self.collision_bodies and name2 not in self.collision_bodies:
                    continue
                # If contact depth is significant, it's a collision
                if contact.dist < -1e-4:
                    collision_free = False
                    break
        return collision_free

    def is_collision_free(self, configuration):
        # Save current state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        
        # Set the configuration
        self.data.qpos[:7] = configuration
        self.data.qvel[:] = 0.0
        
        # Forward kinematics to update body positions
        mujoco.mj_forward(self.model, self.data)

        collision_free = self.check_collisions()

        # Restore state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return collision_free

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

