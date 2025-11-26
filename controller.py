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
    def interpolate_linear_points(start, end, step_size):
        steps = int(np.linalg.norm(end - start) / step_size)
        trajectory = [start + (end - start) * (i / steps) for i in range(steps + 1)]
        return trajectory

    def interpolate_linear_path(self, path, steps_per_segment = 10, step_size = None):
        if len(path) <= 1:
            return path
        
        interpolated = [path[0].copy()]
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            if step_size is not None:
                steps = max(1, int(np.ceil(np.linalg.norm(end - start) / step_size)))
            else:
                steps = steps_per_segment
            
            for step in range(1, steps + 1):
                alpha = step / steps
                interpolated.append((1 - alpha) * start + alpha * end)
        return interpolated
    
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

    def move_to_incremental(self, target_configuration, step_size=0.01):
        start = self.data.qpos.copy()[:7]
        end = target_configuration
        self.trajectory = Controller.interpolate_linear_points(start, end, step_size)

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

        # for point in trajectory:
        #     position, orientation = point
        #     self.move_to(position, orientation)
            # Add delay or synchronization as needed

    # need velocity controller to follow trajectories
    def velocity_control(self, target_velocity, duration, dt):
        pass
        # steps = int(duration / dt)
        # for _ in range(steps):
        #     # Update position based on velocity
        #     self.data.mocap_pos[0] += target_velocity * dt
            # Add delay or synchronization as needed
