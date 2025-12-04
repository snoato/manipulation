import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from manipulation.environments.franka_env import FrankaEnvironment

# Create environment (headless - no viewer to avoid threading issues)
env = FrankaEnvironment("manipulation/environments/assets/franka_emika_panda/scene_symbolic.xml")

# Update scene
env.forward()
print("Environment loaded (headless mode)")

# Capture from all three cameras
cameras = ["top_camera", "side_camera", "front_camera"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Camera Views and Depth Maps', fontsize=16)

for idx, cam_name in enumerate(cameras):
    print(f"\nCapturing {cam_name}...")
    
    # Render RGB and depth
    rgb = env.render_camera(cam_name, width=640, height=480)
    depth = env.render_depth(cam_name, width=640, height=480)
    
    # Plot RGB
    axes[0, idx].imshow(rgb)
    axes[0, idx].set_title(f'{cam_name} - RGB')
    axes[0, idx].axis('off')
    
    # Plot depth (mask out invalid values)
    depth_masked = np.copy(depth)
    depth_masked[~np.isfinite(depth_masked)] = np.nan
    
    im = axes[1, idx].imshow(depth_masked, cmap='turbo', vmin=0, vmax=2.0)
    axes[1, idx].set_title(f'{cam_name} - Depth')
    axes[1, idx].axis('off')
    plt.colorbar(im, ax=axes[1, idx], label='Depth (m)')
    
    # Print stats
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    print(f"  RGB mean: {rgb.mean():.1f}, Depth range: {valid_depth.min():.3f}-{valid_depth.max():.3f}m")

plt.tight_layout()
plt.savefig('camera_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved camera_comparison.png")

# Generate pointclouds from all cameras
print("\n" + "="*60)
print("Generating pointclouds...")
print("="*60)

pointcloud_fig = plt.figure(figsize=(18, 6))

for idx, cam_name in enumerate(cameras):
    print(f"\nGenerating pointcloud from {cam_name}...")
    points, colors = env.get_pointcloud(cam_name, min_depth=0.3, max_depth=2.0)
    print(f"✓ Generated {points.shape[0]:,} points")
    
    if points.shape[0] > 0:
        # Show pointcloud statistics
        print(f"  Bounds: X=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"          Y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"          Z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Create 3D subplot for pointcloud
        ax = pointcloud_fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Downsample for visualization if too many points
        if points.shape[0] > 10000:
            indices = np.random.choice(points.shape[0], 10000, replace=False)
            plot_points = points[indices]
            plot_colors = colors[indices]
        else:
            plot_points = points
            plot_colors = colors
        
        # Plot pointcloud with colors
        ax.scatter(
            plot_points[:, 0],
            plot_points[:, 1],
            plot_points[:, 2],
            c=plot_colors / 255.0,  # Normalize to [0, 1]
            s=1,
            marker='.'
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{cam_name}\n({points.shape[0]:,} points)')
        
        # Set equal aspect ratio for better visualization
        max_range = np.array([
            plot_points[:, 0].max() - plot_points[:, 0].min(),
            plot_points[:, 1].max() - plot_points[:, 1].min(),
            plot_points[:, 2].max() - plot_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (plot_points[:, 0].max() + plot_points[:, 0].min()) * 0.5
        mid_y = (plot_points[:, 1].max() + plot_points[:, 1].min()) * 0.5
        mid_z = (plot_points[:, 2].max() + plot_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Better viewing angle
        ax.view_init(elev=20, azim=45)

plt.tight_layout()
pointcloud_fig.savefig('pointcloud_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved pointcloud_comparison.png")

env.close()
print("\n" + "="*60)
print("✓ Test complete!")
print("="*60)
print("\nGenerated files:")
print("  - camera_comparison.png (RGB and depth views)")
print("  - pointcloud_comparison.png (3D pointcloud visualizations)")
