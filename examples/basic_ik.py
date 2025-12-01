"""Example demonstrating basic IK control with the Franka Panda robot."""

from pathlib import Path
import numpy as np

from manipulation import FrankaEnvironment

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "mjx_scene.xml"


def main():
    # Load environment
    print("Initializing environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    print("Launching viewer...")
    with env.launch_viewer() as viewer:
        # Reset to home position
        env.reset()
        
        # Get IK solver
        ik = env.get_ik()
        
        # Get initial end-effector pose
        env.move_mocap_to_frame("target", "attachment_site", "site")
        initial_position = env.data.mocap_pos[0].copy()
        initial_orientation = env.data.mocap_quat[0].copy()
        
        print(f"\nInitial position: {initial_position}")
        print(f"Initial orientation: {initial_orientation}")
        
        # Define target pose
        target_position = np.array([0.695, 0.000, 0.621])
        target_orientation = np.array([0.001, 0.835, 0.002, 0.550])
        
        print(f"\nTarget position: {target_position}")
        print(f"Target orientation: {target_orientation}")
        
        # Update mocap target
        env.data.mocap_pos[0] = target_position
        env.data.mocap_quat[0] = target_orientation
        
        print("\nMoving to target pose...")
        
        # Main control loop
        while viewer.is_running():
            dt = env.step()
            
            # Update IK target from mocap body
            ik.set_target_from_mocap("target")
            
            # Solve IK
            converged = ik.converge_ik(dt)
            
            # Apply joint positions
            env.data.ctrl[:8] = ik.configuration.q[:8]
            
            if converged:
                print("Target reached!")
                
                # Hold position for a bit
                for _ in range(int(2.0 / env.rate_limiter.dt)):
                    env.step()
                    env.data.ctrl[:8] = ik.configuration.q[:8]
                
                print("Close the viewer window to exit.")
                
                # Keep holding position
                while viewer.is_running():
                    env.step()
                    env.data.ctrl[:8] = ik.configuration.q[:8]


if __name__ == "__main__":
    main()
