"""Example demonstrating different grid offset configurations."""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from manipulation import FrankaEnvironment
from manipulation.symbolic import GridDomain, StateManager, visualize_grid_state

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"
_VIZ_DIR = _HERE / ".." / "manipulation" / "symbolic" / "viz"


def visualize_grid_with_table(state_manager: StateManager, 
                               save_path: str = None,
                               title: str = "Grid Configuration"):
    """
    Visualize grid state showing both table bounds and grid bounds.
    
    Args:
        state_manager: StateManager instance
        save_path: Optional path to save figure
        title: Figure title
    """
    grid = state_manager.grid
    state = state_manager.ground_state()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get table geometry from model
    model = state_manager.model
    temp_data = state_manager.data
    
    # Find table geom to draw full table bounds
    table_geom_id = None
    for i in range(model.ngeom):
        import mujoco
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name == grid.table_geom_name:
            table_geom_id = i
            break
    
    if table_geom_id is not None:
        geom_xpos = temp_data.geom_xpos[table_geom_id]
        geom_size = model.geom_size[table_geom_id]
        
        # Draw full table outline
        table_min_x = geom_xpos[0] - geom_size[0]
        table_max_x = geom_xpos[0] + geom_size[0]
        table_min_y = geom_xpos[1] - geom_size[1]
        table_max_y = geom_xpos[1] + geom_size[1]
        
        table_rect = patches.Rectangle(
            (table_min_x, table_min_y),
            table_max_x - table_min_x,
            table_max_y - table_min_y,
            linewidth=3, edgecolor='brown', facecolor='none',
            linestyle='--', alpha=0.7, label='Table bounds'
        )
        ax.add_patch(table_rect)
    
    # Draw grid bounds
    bounds = grid.table_bounds
    grid_width = bounds['max_x'] - bounds['min_x']
    grid_height = bounds['max_y'] - bounds['min_y']
    
    grid_rect = patches.Rectangle(
        (bounds['min_x'], bounds['min_y']),
        grid_width, grid_height,
        linewidth=2, edgecolor='blue', facecolor='lightblue',
        alpha=0.2, label='Grid bounds'
    )
    ax.add_patch(grid_rect)
    
    # Draw grid cells
    for cell_name, cell_data in grid.cells.items():
        cell_bounds = cell_data['bounds']
        width = cell_bounds[1] - cell_bounds[0]
        height = cell_bounds[3] - cell_bounds[2]
        
        rect = patches.Rectangle(
            (cell_bounds[0], cell_bounds[2]), width, height,
            linewidth=0.5, edgecolor='gray', facecolor='white', alpha=0.5
        )
        ax.add_patch(rect)
    
    # Draw cylinders
    for cyl_name, occupied_cells in state['cylinders'].items():
        cyl_idx = int(cyl_name.split('_')[1])
        
        # Compute centroid of occupied cells
        cell_centers = [grid.cells[cell]['center'] for cell in occupied_cells if cell in grid.cells]
        if not cell_centers:
            continue
        centroid_x = np.mean([c[0] for c in cell_centers])
        centroid_y = np.mean([c[1] for c in cell_centers])
        
        # Get cylinder radius from StateManager specs
        radius, _ = StateManager.CYLINDER_SPECS[cyl_idx]
        
        circle = patches.Circle(
            (centroid_x, centroid_y), radius,
            edgecolor='red', facecolor='orange', alpha=0.7, linewidth=2
        )
        ax.add_patch(circle)
        
        # Add label
        ax.text(centroid_x, centroid_y, str(cyl_idx),
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Set equal aspect and limits
    if table_geom_id is not None:
        ax.set_xlim(table_min_x - 0.05, table_max_x + 0.05)
        ax.set_ylim(table_min_y - 0.05, table_max_y + 0.05)
    else:
        ax.set_xlim(bounds['min_x'] - 0.05, bounds['max_x'] + 0.05)
        ax.set_ylim(bounds['min_y'] - 0.05, bounds['max_y'] + 0.05)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add info text
    info_text = f"Grid: {grid.cells_x}×{grid.cells_y} cells ({grid.cell_size*100:.1f}cm)\n"
    info_text += f"Offset: X={grid.grid_offset_x*100:+.1f}cm, Y={grid.grid_offset_y*100:+.1f}cm\n"
    info_text += f"Cylinders: {len(state['cylinders'])}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization: {save_path}")
    
    plt.close()


def main():
    print("=" * 70)
    print("Grid Offset Configurations Example")
    print("=" * 70)
    
    # Initialize environment
    print("\n1. Loading environment...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    
    # Create output directory
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define configurations with different offsets
    print("\n2. Defining grid offset configurations...")
    
    configurations = [
        {
            'name': 'Centered (Default)',
            'offset_x': 0.0,
            'offset_y': 0.0,
            'description': 'Grid centered on table (X), front-aligned (Y)'
        },
        {
            'name': 'Left Shifted',
            'offset_x': -0.10,
            'offset_y': 0.0,
            'description': 'Grid shifted 10cm to the left'
        },
        {
            'name': 'Right Shifted',
            'offset_x': 0.10,
            'offset_y': 0.0,
            'description': 'Grid shifted 10cm to the right'
        },
        {
            'name': 'Robot Near',
            'offset_x': 0.0,
            'offset_y': -0.08,
            'description': 'Grid shifted 8cm toward robot'
        },
        {
            'name': 'Robot Far',
            'offset_x': 0.0,
            'offset_y': 0.08,
            'description': 'Grid shifted 8cm away from robot'
        },
        {
            'name': 'Diagonal (Top-Right)',
            'offset_x': 0.08,
            'offset_y': 0.08,
            'description': 'Grid shifted 8cm right and 8cm away from robot'
        },
    ]
    
    # Fixed parameters for all configurations
    cell_size = 0.02  # 2cm cells
    working_area = (0.4, 0.4)  # 40cm x 40cm
    n_cylinders = 8
    
    print(f"   Grid: 20×20 cells ({cell_size*100:.0f}cm cell size)")
    print(f"   Working area: {working_area[0]*100:.0f}cm × {working_area[1]*100:.0f}cm")
    print(f"   Cylinders per config: {n_cylinders}")
    print(f"   Total configurations: {len(configurations)}")
    
    # Create grids and state managers for each configuration
    print("\n3. Creating grid configurations...")
    grid_configs = []
    
    for i, config in enumerate(configurations):
        print(f"\n   Configuration {i+1}: {config['name']}")
        print(f"      {config['description']}")
        print(f"      Offset: X={config['offset_x']*100:+.1f}cm, Y={config['offset_y']*100:+.1f}cm")
        
        # Create grid domain with offsets
        grid = GridDomain(
            model=env.model,
            cell_size=cell_size,
            working_area=working_area,
            grid_offset_x=config['offset_x'],
            grid_offset_y=config['offset_y'],
            table_body_name="simple_table",
            table_geom_name="table_surface"
        )
        
        # Verify grid info
        info = grid.get_domain_info()
        print(f"      Grid bounds: X=[{info['table_bounds']['min_x']:.3f}, {info['table_bounds']['max_x']:.3f}], "
              f"Y=[{info['table_bounds']['min_y']:.3f}, {info['table_bounds']['max_y']:.3f}]")
        
        # Create state manager
        state_manager = StateManager(grid, env)
        
        # Sample random state
        state_manager.sample_random_state(n_cylinders=n_cylinders, seed=42 + i)
        
        # Ground state
        grounded_state = state_manager.ground_state()
        print(f"      Active cylinders: {len(grounded_state['cylinders'])}")
        
        # Visualize with table bounds
        viz_path = _VIZ_DIR / f"offset_config_{i+1}_{config['name'].lower().replace(' ', '_')}.png"
        visualize_grid_with_table(
            state_manager,
            save_path=viz_path,
            title=f"Config {i+1}: {config['name']} (X={config['offset_x']*100:+.0f}cm, Y={config['offset_y']*100:+.0f}cm)"
        )
        
        grid_configs.append((config, grid, state_manager))
    
    # Display in MuJoCo viewer
    print("\n4. Launching MuJoCo viewer...")
    print(f"   Will cycle through all {len(configurations)} configurations (5 seconds each)")
    print("   Press ESC to exit early\n")
    
    with env.launch_viewer() as viewer:
        for i, (config, grid, state_manager) in enumerate(grid_configs):
            if not viewer.is_running():
                break
            
            print(f"   [{i+1}/{len(configurations)}] Displaying: {config['name']}")
            print(f"       Offset: X={config['offset_x']*100:+.1f}cm, Y={config['offset_y']*100:+.1f}cm")
            
            # Re-sample this configuration's state to refresh MuJoCo
            state_manager.sample_random_state(n_cylinders=n_cylinders, seed=42 + i)
            
            # Render for 5 seconds
            start_time = time.time()
            while viewer.is_running() and (time.time() - start_time) < 5.0:
                env.step()
                time.sleep(0.01)
            
            print()
    
    print("=" * 70)
    print("Grid Offset Configurations Example Complete!")
    print(f"Visualizations saved to: {_VIZ_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
