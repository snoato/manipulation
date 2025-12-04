"""Visualization utilities for symbolic planning."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for mjpython compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from manipulation.symbolic.grid_domain import GridDomain
from manipulation.symbolic.state_manager import StateManager


def visualize_grid_state(state_manager: StateManager, 
                         save_path: Optional[str] = None,
                         title: str = "Grid State",
                         figsize: tuple = (12, 10)):
    """
    Visualize grid state with matplotlib.
    
    Args:
        state_manager: StateManager instance
        save_path: Optional path to save figure
        title: Figure title
        figsize: Figure size
    """
    grid = state_manager.grid
    state = state_manager.ground_state()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid cells
    for cell_name, cell_data in grid.cells.items():
        bounds = cell_data['bounds']
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        
        rect = patches.Rectangle(
            (bounds[0], bounds[2]), width, height,
            linewidth=0.5, edgecolor='gray', facecolor='white', alpha=0.3
        )
        ax.add_patch(rect)
    
    # Color map matching MuJoCo materials (cyl_mat_0 through cyl_mat_9)
    # These colors match the rgba values in scene_symbolic.xml
    colors = [
        (1.0, 0.2, 0.2),  # cyl_mat_0: red
        (0.2, 1.0, 0.2),  # cyl_mat_1: green
        (0.2, 0.2, 1.0),  # cyl_mat_2: blue
        (1.0, 1.0, 0.2),  # cyl_mat_3: yellow
        (1.0, 0.2, 1.0),  # cyl_mat_4: magenta
        (0.2, 1.0, 1.0),  # cyl_mat_5: cyan
        (1.0, 0.5, 0.2),  # cyl_mat_6: orange
        (0.5, 0.2, 1.0),  # cyl_mat_7: purple
        (0.2, 1.0, 0.5),  # cyl_mat_8: light green
        (1.0, 0.8, 0.4),  # cyl_mat_9: gold
    ]
    
    # Draw occupied cells and cylinders
    drawn_cylinders = set()
    
    for cyl_name, occupied_cells in state['cylinders'].items():
        cyl_idx = int(cyl_name.split('_')[1])
        color_idx = cyl_idx % 10
        color = colors[color_idx]
        
        # Shade occupied cells
        for cell_name in occupied_cells:
            if cell_name is None or cell_name not in grid.cells:
                continue  # Skip cells outside grid bounds
            cell_data = grid.cells[cell_name]
            bounds = cell_data['bounds']
            width = bounds[1] - bounds[0]
            height = bounds[3] - bounds[2]
            
            rect = patches.Rectangle(
                (bounds[0], bounds[2]), width, height,
                linewidth=0.5, edgecolor='gray', facecolor=color, alpha=0.4
            )
            ax.add_patch(rect)
        
        # Draw cylinder circle (only once per cylinder)
        if cyl_name not in drawn_cylinders:
            # Compute centroid of occupied cells
            cell_centers = [grid.cells[cell]['center'] for cell in occupied_cells]
            centroid_x = np.mean([c[0] for c in cell_centers])
            centroid_y = np.mean([c[1] for c in cell_centers])
            
            # Get cylinder radius
            radius, _ = StateManager.CYLINDER_SPECS[cyl_idx]
            
            circle = patches.Circle(
                (centroid_x, centroid_y), radius,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(circle)
            
            # Add label
            ax.text(centroid_x, centroid_y, str(cyl_idx),
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            drawn_cylinders.add(cyl_name)
    
    # Set axis properties
    bounds = grid.table_bounds
    ax.set_xlim(bounds['min_x'] - 0.01, bounds['max_x'] + 0.01)
    ax.set_ylim(bounds['min_y'] - 0.01, bounds['max_y'] + 0.01)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(False)
    
    # Add info text
    info_text = f"Grid: {grid.cells_x}×{grid.cells_y} cells ({grid.cell_size*100:.1f}cm)\n"
    info_text += f"Cylinders: {len(state['cylinders'])}\n"
    info_text += f"Gripper: {'empty' if state['gripper_empty'] else f'holding {state['holding']}'}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close(fig)
    
    return fig, ax
