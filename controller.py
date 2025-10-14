import mujoco
import numpy as np
import mink

import enum

class ControllerStatus(enum.Enum):
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    GRASPING = "grasping"

class Controller:
    def linear_interpolate(start, end, step_size):
        steps = int(np.linalg.norm(end - start) / step_size)
        trajectory = [start + (end - start) * (i / steps) for i in range(steps + 1)]
        return trajectory
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.status = ControllerStatus.IDLE
        self.trajectory = []


    def step(self, delta=0.1, grasping_delta=0.3):
        if self.status == ControllerStatus.IDLE:
            if self.trajectory:
                self.status = ControllerStatus.MOVING
            else:
                return

        if self.status == ControllerStatus.MOVING:
            if self.trajectory:
                ctrl = self.data.ctrl.copy()[:7]
                qpos = self.data.qpos.copy()[:7]
                ctrl_ref = self.trajectory[0] 

                if np.linalg.norm(ctrl_ref - qpos) < delta and (ctrl == ctrl_ref).all():
                    self.trajectory.pop(0)
                    if not self.trajectory:
                        self.status = ControllerStatus.IDLE
                        return
                    ctrl_ref = self.trajectory[0]

                if (ctrl != ctrl_ref).any():
                    self.data.ctrl[:7] = ctrl_ref

            else:
                self.status = ControllerStatus.IDLE
                return
        
        if self.status == ControllerStatus.GRASPING:
            ctrl = self.data.ctrl[7]
            qpos = self.data.qpos[7]
            if np.linalg.norm(ctrl - qpos) < grasping_delta:
                self.status = ControllerStatus.IDLE
                return

    def move_to(self, configuration):
        if self.status != ControllerStatus.IDLE:
            print("Controller is busy. Cannot move to new position.")
            return
        self.trajectory = [configuration]

    def get_status(self):
        return self.status
    
    def stop(self):
        # Implement stop logic here
        self.status = ControllerStatus.IDLE
        self.trajectory = []

    def open_gripper(self):
        self.data.ctrl[7] = 0.04  # open gripper
        self.status = ControllerStatus.GRASPING

    def close_gripper(self):
        self.data.ctrl[7] = -0.2  # close gripper
        self.status = ControllerStatus.GRASPING

    # need trajectory follower
    def follow_trajectory(self, trajectory):
        if self.status != ControllerStatus.IDLE:
            print("Controller is busy. Cannot follow new trajectory.")
            return
        self.trajectory = trajectory

        for point in trajectory:
            position, orientation = point
            self.move_to(position, orientation)
            # Add delay or synchronization as needed

    # need velocity controller to follow trajectories
    def velocity_control(self, target_velocity, duration, dt):
        pass
        # steps = int(duration / dt)
        # for _ in range(steps):
        #     # Update position based on velocity
        #     self.data.mocap_pos[0] += target_velocity * dt
            # Add delay or synchronization as needed
