"""Interactive 3D visualization of point clouds with matplotlib."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from manipulation import FrankaEnvironment
from manipulation.perception import MujocoCamera

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_symbolic.xml"


def main():
    """Generate point clouds and display in interactive 3D plot."""
    print("=" * 70)
    print("Interactive Point Cloud Visualization")
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
    
    # Let user choose camera
    print("\nAvailable cameras:")
    print("  1. top_camera")
    print("  2. side_camera")
    print("  3. front_camera")
    print("  4. All cameras (combined)")
    
    choice = input("\nSelect camera (1-4) [default=1]: ").strip()
    
    if choice == '2':
        cameras = [('side_camera', 'cyan')]
    elif choice == '3':
        cameras = [('front_camera', 'blue')]
    elif choice == '4':
        cameras = [
            ('top_camera', 'red'),
            ('side_camera', 'green'),
            ('front_camera', 'blue')
        ]
    else:
        cameras = [('top_camera', 'red')]
    
    # Generate point clouds
    print(f"\nGenerating point clouds from {len(cameras)} camera(s)...")
    all_points = []
    all_colors = []
    camera_positions = []
    
    for cam_name, _ in cameras:
        print(f"  Processing {cam_name}...")
        
        points, colors = camera.get_pointcloud(
            cam_name,
            num_samples=2000,
            min_depth=0.3,
            max_depth=3.0
        )
        
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
            
            cam_pos, _ = camera._get_camera_pose(cam_name)
            camera_positions.append((cam_pos, cam_name))
            
            print(f"    Generated {len(points)} points")
    
    print(f"\nTotal: {len(all_points)} point cloud(s) with {sum(len(p) for p in all_points)} total points")
    
    if not all_points:
        print("No points generated!")
        camera.close()
        return
    
    # Combine all point clouds
    points_combined = np.vstack(all_points)
    colors_combined = np.vstack(all_colors)
    colors_normalized = colors_combined.astype(float) / 255.0
    
    camera.close()
    
    # Create interactive 3D plot
    print(f"\nCreating interactive visualization with {len(points_combined)} points...")
    print("\nControls:")
    print("  - Click and drag to rotate")
    print("  - Right-click and drag to zoom")
    print("  - Middle-click and drag to pan")
    print("  - Close window to exit")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    scatter = ax.scatter(
        points_combined[:, 0],
        points_combined[:, 1],
        points_combined[:, 2],
        c=colors_normalized,
        marker='.',
        s=2,
        alpha=0.6,
        label='Point Cloud'
    )
    
    # Plot camera positions
    for cam_pos, cam_name in camera_positions:
        ax.scatter(
            [cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
            c='red',
            marker='o',
            s=200,
            edgecolors='black',
            linewidths=2,
            label=f'{cam_name}'
        )
        
        # Add text label for camera
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.1,
               cam_name.replace('_camera', ''),
               fontsize=9,
               weight='bold')
    
    # Plot table plane at z=0.25
    x_range = [np.min(points_combined[:, 0]), np.max(points_combined[:, 0])]
    y_range = [np.min(points_combined[:, 1]), np.max(points_combined[:, 1])]
    
    xx, yy = np.meshgrid(
        np.linspace(x_range[0] - 0.2, x_range[1] + 0.2, 10),
        np.linspace(y_range[0] - 0.2, y_range[1] + 0.2, 10)
    )
    zz_table = np.ones_like(xx) * 0.25
    ax.plot_surface(xx, yy, zz_table, alpha=0.2, color='brown', label='Table (z=0.25)')
    
    # Plot floor at z=0
    zz_floor = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz_floor, alpha=0.1, color='gray', label='Floor (z=0)')
    
    # Labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Interactive Point Cloud Visualization\n(Click and drag to rotate)', fontsize=14)
    
    # Set reasonable axis limits
    ax.set_xlim([x_range[0] - 0.3, x_range[1] + 0.3])
    ax.set_ylim([y_range[0] - 0.3, y_range[1] + 0.3])
    ax.set_zlim([-0.2, np.max(points_combined[:, 2]) + 0.3])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=8)
    
    # Add statistics text
    stats_text = f"Points: {len(points_combined)}\n"
    stats_text += f"X: [{np.min(points_combined[:, 0]):.2f}, {np.max(points_combined[:, 0]):.2f}]\n"
    stats_text += f"Y: [{np.min(points_combined[:, 1]):.2f}, {np.max(points_combined[:, 1]):.2f}]\n"
    stats_text += f"Z: [{np.min(points_combined[:, 2]):.2f}, {np.max(points_combined[:, 2]):.2f}]"
    
    ax.text2D(0.02, 0.98, stats_text,
             transform=ax.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show interactive plot
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization closed.")


if __name__ == "__main__":
    main()
