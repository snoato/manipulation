"""Visualization test for point cloud generation using matplotlib."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from manipulation import FrankaEnvironment
from manipulation.perception import MujocoCamera

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"


def visualize_pointcloud(points, colors, title="Point Cloud", show_axes=True):
    """
    Visualize a point cloud using matplotlib 3D scatter plot.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255)
        title: Plot title
        show_axes: Whether to show axis labels and grid
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0, 1] for matplotlib
    colors_normalized = colors.astype(float) / 255.0
    
    # Create scatter plot
    ax.scatter(
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        c=colors_normalized,
        marker='.',
        s=1,  # Small point size
        alpha=0.6
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    if show_axes:
        ax.grid(True)
    
    # Set equal aspect ratio for better visualization
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ])
    
    mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
    mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
    mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    return fig, ax


def main():
    """Generate and visualize point clouds from multiple cameras."""
    print("=" * 70)
    print("Point Cloud Visualization Test")
    print("=" * 70)
    
    # Initialize environment
    print("\nInitializing environment with scene_symbolic.xml...")
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)
    env.reset()
    
    # Step simulation to ensure stable state
    for _ in range(10):
        env.step()
    
    # Initialize camera
    print("Initializing camera utilities...")
    camera = MujocoCamera(env, width=640, height=480)
    
    # Generate point clouds from different cameras
    cameras_to_visualize = [
        ('top_camera', 'Top View'),
        ('side_camera', 'Side View'),
        ('front_camera', 'Front View')
    ]
    
    print("\nGenerating point clouds...")
    
    for camera_name, view_name in cameras_to_visualize:
        print(f"  Processing {camera_name}...")
        
        # Generate point cloud with more samples for better visualization
        points, colors = camera.get_pointcloud(
            camera_name,
            num_samples=2000,  # More points for visualization
            min_depth=0.3,
            max_depth=3.0
        )
        
        if len(points) > 0:
            print(f"    Generated {len(points)} points")
            
            # Visualize
            visualize_pointcloud(points, colors, title=f"{view_name} - {camera_name}")
        else:
            print(f"    WARNING: No points generated from {camera_name}")
    
    # Create a combined view with all three point clouds in one figure
    print("\nCreating combined multi-camera view...")
    
    fig = plt.figure(figsize=(15, 5))
    
    for idx, (camera_name, view_name) in enumerate(cameras_to_visualize, 1):
        points, colors = camera.get_pointcloud(
            camera_name,
            num_samples=1000,
            min_depth=0.3,
            max_depth=3.0
        )
        
        if len(points) > 0:
            ax = fig.add_subplot(1, 3, idx, projection='3d')
            colors_normalized = colors.astype(float) / 255.0
            
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors_normalized,
                marker='.',
                s=1,
                alpha=0.6
            )
            
            ax.set_xlabel('X (m)', fontsize=8)
            ax.set_ylabel('Y (m)', fontsize=8)
            ax.set_zlabel('Z (m)', fontsize=8)
            ax.set_title(view_name, fontsize=10)
            ax.tick_params(labelsize=6)
            
            # Set consistent axis limits for comparison
            ax.set_xlim(-1, 2)
            ax.set_ylim(-1, 2)
            ax.set_zlim(-0.5, 1.5)
    
    plt.suptitle('Multi-Camera Point Cloud Comparison', fontsize=14)
    plt.tight_layout()
    
    camera.close()
    
    print("\n" + "="*70)
    print("Visualization complete! Close the plot windows to exit.")
    print("="*70 + "\n")
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
