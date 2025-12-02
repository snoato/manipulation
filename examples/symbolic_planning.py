"""Example demonstrating symbolic planning with grid-based PDDL domain."""

from pathlib import Path
import matplotlib.pyplot as plt

from manipulation import FrankaEnvironment
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state, plot_grid_heatmap

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
_DOMAIN_DIR = _HERE / ".." / "manipulation" / "symbolic" / "domains"
_PROBLEM_DIR = _HERE / ".." / "manipulation" / "symbolic" / "problems"
_VIZ_DIR = _HERE / ".." / "manipulation" / "symbolic" / "viz"


def main():
    print("=" * 70)
    print("Symbolic Planning Example - Grid-based PDDL Domain")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Create grid domain with 40x40 cells (0.4m x 0.4m working area at 1cm resolution)
    print("\n2. Creating grid domain...")
    grid = GridDomain(
        model=env.model,
        cell_size=0.05,  # 2cm cells
        working_area=(0.4, 0.4),  # 40cm x 40cm
        table_body_name="simple_table",
        table_geom_name="table_surface"
    )
    
    # Print grid info
    info = grid.get_grid_info()
    print(f"   Grid dimensions: {info['grid_dimensions'][0]}x{info['grid_dimensions'][1]} cells")
    print(f"   Total cells: {info['total_cells']}")
    print(f"   Cell size: {info['cell_size']*100:.1f}cm")
    print(f"   Working area: {info['working_area'][0]}m x {info['working_area'][1]}m")
    print(f"   Table height: {info['table_height']:.3f}m")
    
    # Generate PDDL domain
    print("\n3. Generating PDDL domain...")
    domain_path = _DOMAIN_DIR / "tabletop.pddl"
    _DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    grid.generate_pddl_domain(domain_path)
    print(f"   Saved domain to: {domain_path}")
    
    # Create state manager
    print("\n4. Creating state manager...")
    state_manager = StateManager(grid, env.model, env.data)
    
    # Sample random state
    print("\n5. Sampling random state with 5 cylinders...")
    random_init = state_manager.sample_random_state(n_cylinders=5)
    print("   Sampled state (first few predicates):")
    for pred in random_init.split('\n')[:5]:
        print(f"     {pred}")
    print("     ...")
    
    # Initialize environment from random state
    print("\n6. Initializing environment from sampled state...")
    state_manager.init_from_pddl_state(random_init)
    # Forward dynamics to update derived quantities (don't reset, which would clear state)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    
    # Ground current state
    print("\n7. Grounding current state...")
    grounded_state = state_manager.ground_state()
    print(f"   Active cylinders: {len(grounded_state['cylinders'])}")
    for cyl_name, cells in list(grounded_state['cylinders'].items())[:3]:
        print(f"     {cyl_name}: occupies {len(cells)} cells")
    if len(grounded_state['cylinders']) > 3:
        print(f"     ... and {len(grounded_state['cylinders']) - 3} more")
    
    # Generate PDDL problem
    print("\n8. Generating PDDL problem file...")
    problem_path = _PROBLEM_DIR / "problem_example.pddl"
    _PROBLEM_DIR.mkdir(parents=True, exist_ok=True)
    state_manager.generate_pddl_problem("example-problem", problem_path)
    print(f"   Saved problem to: {problem_path}")
    
    # Visualize state
    print("\n9. Creating visualizations...")
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Grid state visualization
    # viz_path = _VIZ_DIR / "grid_state.png"
    # visualize_grid_state(
    #     state_manager,
    #     save_path=viz_path,
    #     title="Grid State - Random Configuration"
    # )
    
    # # Heatmap visualization
    # heatmap_path = _VIZ_DIR / "occupancy_heatmap.png"
    # plot_grid_heatmap(
    #     state_manager,
    #     save_path=heatmap_path,
    #     title="Grid Occupancy Heatmap"
    # )
    
    # print(f"   Saved grid visualization to: {viz_path}")
    # print(f"   Saved heatmap to: {heatmap_path}")
    
    # # Show in MuJoCo viewer
    # print("\n10. Launching MuJoCo viewer...")
    # print("    (Close viewer window to continue)")
    
    with env.launch_viewer() as viewer:
        
        # Keep viewer open
        while viewer.is_running():
            env.step()
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("\nGenerated files:")
    print(f"  - PDDL domain: {domain_path}")
    print(f"  - PDDL problem: {problem_path}")
    print(f"  - Grid visualization: {viz_path}")
    print(f"  - Occupancy heatmap: {heatmap_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
