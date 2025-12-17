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
        
        # Define target pose
        target_position = np.array([0.695, 0.000, 0.621])
        target_orientation = np.array([0.001, 0.835, 0.002, 0.550])
        
        print(f"\nTarget position: {target_position}")
        print(f"Target orientation: {target_orientation}")
        
        # Set IK target
        ik.set_target_position(target_position, target_orientation)
        
        print("\nMoving to target pose...")
        
        # Solve IK
        dt = env.rate.dt
        converged = ik.converge_ik(dt)
        
        if converged:
            print("IK converged successfully!")
            
            # Apply joint positions and hold
            target_q = ik.configuration.q[:8].copy()
            
            while viewer.is_running():
                env.step()
                env.data.ctrl[:8] = target_q
        else:
            print("IK failed to converge!")
            print("Close the viewer window to exit.")
            
            # Keep current position
            while viewer.is_running():
                env.step()


if __name__ == "__main__":
    main()
