"""Visualization utilities for symbolic planning."""

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
    
    # Color map for cylinders (cycling through 10 colors)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Draw occupied cells and cylinders
    drawn_cylinders = set()
    
    for cyl_name, occupied_cells in state['cylinders'].items():
        cyl_idx = int(cyl_name.split('_')[1])
        color_idx = cyl_idx % 10
        color = colors[color_idx]
        
        # Shade occupied cells
        for cell_name in occupied_cells:
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
    min_x, max_x, min_y, max_y = grid.grid_bounds
    ax.set_xlim(min_x - 0.01, max_x + 0.01)
    ax.set_ylim(min_y - 0.01, max_y + 0.01)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add info text
    info_text = f"Grid: {grid.grid_width}×{grid.grid_height} cells ({grid.cell_size*100:.1f}cm)\n"
    info_text += f"Cylinders: {len(state['cylinders'])}\n"
    info_text += f"Gripper: {'empty' if state['gripper_empty'] else f'holding {state['holding']}'}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig, ax


def create_grid_overlay_geoms(grid: GridDomain) -> str:
    """
    Generate XML snippet for grid overlay in MuJoCo.
    
    Args:
        grid: GridDomain instance
        
    Returns:
        XML string with grid line geoms
    """
    min_x, max_x, min_y, max_y = grid.grid_bounds
    
    xml_lines = []
    xml_lines.append("<!-- Grid overlay (visual only) -->")
    
    # Vertical lines
    for i in range(grid.grid_width + 1):
        x = min_x + i * grid.cell_size
        y_center = (min_y + max_y) / 2
        half_height = (max_y - min_y) / 2
        
        xml_lines.append(
            f'<geom name="grid_v_{i}" type="box" '
            f'pos="{x} {y_center} {grid.table_height + 0.001}" '
            f'size="0.0005 {half_height} 0.0001" '
            f'rgba="0.3 0.3 0.3 0.5" contype="0" conaffinity="0"/>'
        )
    
    # Horizontal lines
    for j in range(grid.grid_height + 1):
        y = min_y + j * grid.cell_size
        x_center = (min_x + max_x) / 2
        half_width = (max_x - min_x) / 2
        
        xml_lines.append(
            f'<geom name="grid_h_{j}" type="box" '
            f'pos="{x_center} {y} {grid.table_height + 0.001}" '
            f'size="{half_width} 0.0005 0.0001" '
            f'rgba="0.3 0.3 0.3 0.5" contype="0" conaffinity="0"/>'
        )
    
    return "\n    ".join(xml_lines)


def plot_grid_heatmap(state_manager: StateManager,
                      save_path: Optional[str] = None,
                      title: str = "Grid Occupancy Heatmap",
                      figsize: tuple = (12, 10)):
    """
    Plot grid as heatmap showing occupancy density.
    
    Args:
        state_manager: StateManager instance
        save_path: Optional path to save figure
        title: Figure title
        figsize: Figure size
    """
    grid = state_manager.grid
    state = state_manager.ground_state()
    
    # Create occupancy matrix
    occupancy = np.zeros((grid.grid_height, grid.grid_width))
    
    for cyl_name, occupied_cells in state['cylinders'].items():
        for cell_name in occupied_cells:
            cell_data = grid.cells[cell_name]
            i, j = cell_data['index']
            occupancy[j, i] += 1  # Note: matrix is row-major, so y is first index
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(occupancy, cmap='Reds', origin='lower', aspect='auto',
                   extent=[0, grid.grid_width, 0, grid.grid_height])
    
    ax.set_xlabel('X cell index', fontsize=12)
    ax.set_ylabel('Y cell index', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of cylinders', fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(0, grid.grid_width, max(1, grid.grid_width // 10)))
    ax.set_yticks(np.arange(0, grid.grid_height, max(1, grid.grid_height // 10)))
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    return fig, ax
