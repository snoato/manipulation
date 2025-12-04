"""Example demonstrating symbolic planning with grid-based PDDL domain."""

from pathlib import Path
import time
import re
import matplotlib.pyplot as plt

from manipulation import FrankaEnvironment
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state
from manipulation.symbolic.domains.tabletop.state_manager import extract_grid_dimensions_from_pddl

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
_DOMAIN_DIR = _HERE / ".." / "manipulation" / "symbolic" / "domains" / "tabletop" / "pddl"
_PROBLEM_DIR = _HERE / ".." / "manipulation" / "symbolic" / "domains" / "tabletop" / "pddl" / "problems"
_VIZ_DIR = _HERE / ".." / "manipulation" / "symbolic" / "viz"


def main():
    print("=" * 70)
    print("Symbolic Planning Example - Grid-based PDDL Domain")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    print("\n2. Preparing to generate problems with varying grid dimensions...")
    print("   Grid sizes: 10x10, 20x20, 40x40, 60x60, 100x100")
    print("   Fixed working area: 0.4m x 0.4m")
    
    # Generate 5 different problems with varying grid dimensions
    print("\n4. Generating 5 random problem configurations with different grid sizes...")
    _PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define grid sizes: from 10x10 to 100x100
    grid_sizes = [10, 20, 40, 60, 100]
    working_area = (0.4, 0.4)  # 40cm x 40cm fixed working area
    
    problem_files = []
    grid_configs = []
    
    for i, grid_size in enumerate(grid_sizes):
        cell_size = working_area[0] / grid_size
        print(f"\n   Problem {i+1}: {grid_size}x{grid_size} grid (cell size: {cell_size*100:.2f}cm)")
        
        # Create grid domain for this size
        grid = GridDomain(
            model=env.model,
            cell_size=cell_size,
            working_area=working_area,
            table_body_name="simple_table",
            table_geom_name="table_surface"
        )
        
        # Create state manager
        state_manager = StateManager(grid, env)
        
        # Sample random state
        state_manager.sample_random_state(n_cylinders=10)
        
        # Ground current state
        grounded_state = state_manager.ground_state()
        print(f"     Active cylinders: {len(grounded_state['cylinders'])}")
        
        # Generate PDDL problem
        problem_path = _PROBLEM_DIR / f"problem_{i+1}.pddl"
        state_manager.generate_pddl_problem(f"problem-{i+1}", problem_path)
        problem_files.append(problem_path)
        grid_configs.append((grid, state_manager))
        print(f"     Saved to: {problem_path}")
        
        # Visualize state
        viz_path = _VIZ_DIR / f"grid_state_{i+1}.png"
        visualize_grid_state(
            state_manager,
            save_path=viz_path,
            title=f"Grid State - Problem {i+1} ({grid_size}x{grid_size})"
        )
    
    # Show all problems in MuJoCo viewer with 5 seconds between each
    print("\n5. Launching MuJoCo viewer...")
    print("    Will cycle through all 5 problems (5 seconds each)")
    
    with env.launch_viewer() as viewer:
        for i, (problem_path, (grid, state_manager)) in enumerate(zip(problem_files, grid_configs)):
            print(f"\n   Displaying Problem {i+1}...")
            
            # Read PDDL problem file
            pddl_content = problem_path.read_text()
            
            # Extract and verify grid dimensions
            cells_x, cells_y = extract_grid_dimensions_from_pddl(pddl_content)
            cell_size = 0.4 / cells_x  # Calculate cell size from grid dimensions
            print(f"     Grid from PDDL: {cells_x}x{cells_y} (cell size: {cell_size*100:.2f}cm)")
            
            # Extract init section and load state
            init_match = re.search(r'\(:init\s+(.*?)\s+\)', pddl_content, re.DOTALL)
            if init_match:
                init_section = init_match.group(1)
                state_manager.init_from_pddl_state(init_section)
            
            # Render for 5 seconds
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < 5.0:
                env.step()
                time.sleep(0.01)
            
            if not viewer.is_running():
                break
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("\nGenerated files:")
    print(f"  - 5 PDDL problems in: {_PROBLEM_DIR}")
    print(f"  - 5 Grid visualizations in: {_VIZ_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
