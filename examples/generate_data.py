"""Example combining symbolic planning with grasping RRT."""

from pathlib import Path
import time
import numpy as np

from manipulation import FrankaEnvironment, RRTStar, ControllerStatus
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
_PROBLEM_DIR = _HERE / ".." / "data" / "tabletop" / "problems"
_VIZ_DIR = _HERE / ".." / "data" / "viz" / "tabletop"

_EXAMPLES = 1000
_TARGETS_MAX = 7
_TARGETS_MIN = 3

_VIEWER = True

def main():
    print("=" * 70)
    print("Symbolic Planning Data Generation")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    
    # Create grid domain (20x20 cells = 2cm resolution)
    print("3. Creating grid domain...")
    grid = GridDomain(
        model=env.model,
        cell_size=0.04,  # 2cm cells
        working_area=(0.4, 0.3),  # 40cm x 40cm
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_y=0.02
    )
    
    info = grid.get_grid_info()
    print(f"   Grid: {info['grid_dimensions'][0]}x{info['grid_dimensions'][1]} cells ({info['cell_size']*100:.1f}cm)")
    
    # Create state manager
    state_manager = StateManager(grid, env)
    
    # Create output directories
    _PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    if _VIEWER:
        viewer = env.launch_viewer()
        if not viewer.is_running():
            raise Exception("no viewer available")

        # Generate new configuration if no targets
    for config_num in range(_EXAMPLES):
        config_num += 1
        print(f"\n{'='*70}")
        print(f"Configuration {config_num}: Generating random problem...")
        print(f"{'='*70}")
        
        # Sample random state
        n_cylinders = np.random.randint(_TARGETS_MIN, _TARGETS_MAX)
        state_manager.sample_random_state(n_cylinders=n_cylinders)
        
        # Ground and visualize state
        grounded_state = state_manager.ground_state()
        print(f"   Active cylinders: {len(grounded_state['cylinders'])}")
        
        # Generate PDDL problem
        cylinders = sorted(state_manager.ground_state()['cylinders'].keys())
        target_cylinder = np.random.choice(cylinders)
        problem_path = _PROBLEM_DIR / f"config_{config_num}.pddl"
        state_manager.generate_pddl_problem(f"config-{config_num}", problem_path, f"(holding gripper1 {target_cylinder})")
        print(f"   Saved PDDL: {problem_path}")
        
        # Visualize
        viz_path = _VIZ_DIR / f"config_{config_num}.png"
        visualize_grid_state(
            state_manager,
            save_path=viz_path,
            title=f"Configuration {config_num}"
        )
        print(f"   Saved visualization: {viz_path}")
        
        env.rest(2.0)

if __name__ == "__main__":
    main()
