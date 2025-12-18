"""
Demo script for segmented point cloud generation.
Shows how to retrieve and visualize point clouds for individual objects.
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from manipulation import FrankaEnvironment
from manipulation.perception import MujocoCamera

# Define path to scene XML
_HERE = Path(__file__).parent
_XML = _HERE / ".." / "manipulation" / "environments" / "assets" / "franka_emika_panda" / "scene_test.xml"

def visualize_segmented_clouds(segmented_clouds):
    """
    Visualize segmented point clouds in a single 3D plot.
    
    Args:
        segmented_clouds: Dictionary mapping object name to (points, colors)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Found {len(segmented_clouds)} objects:")
    
    # Generate distinct colors for legend if we want to override object colors
    # But here we will use the actual RGB colors from the camera
    
    for obj_name, (points, colors) in segmented_clouds.items():
        num_points = len(points)
        print(f"  - {obj_name}: {num_points} points")
        
        if num_points == 0:
            continue
            
        # Normalize colors to [0, 1] for matplotlib
        colors_normalized = colors.astype(float) / 255.0
        
        # Scatter plot for this object
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors_normalized,
            marker='.',
            s=5,
            alpha=0.8,
            label=obj_name
        )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Segmented Point Clouds")
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Set view to look at the table
    ax.view_init(elev=30, azim=135)
    
    # Set equal aspect ratio
    # 1. Collect all points to find global bounds
    all_points = []
    for pts, _ in segmented_clouds.values():
        if len(pts) > 0:
            all_points.append(pts)
            
    if all_points:
        all_points = np.vstack(all_points)
        min_vals = all_points.min(axis=0)
        max_vals = all_points.max(axis=0)
        mid_vals = (max_vals + min_vals) / 2
        
        # Find maximum range to create a cubic bounding box
        max_range = (max_vals - min_vals).max() / 2
        
        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        ax.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)
        
        # Also set box aspect to 1,1,1 so the cubic box is rendered as a cube
        try:
            ax.set_box_aspect([1, 1, 1])
        except:
            pass

    plt.tight_layout()
    plt.show()

def main():
    # 1. Initialize Environment
    print("Initializing environment...")
    env = FrankaEnvironment(path=str(_XML))
    env.reset()
    
    # 2. Initialize Camera
    # We use the 'front' camera or 'top_cam' depending on what's in the XML.
    # Usually 'top_cam' gives a good view of the table.
    print("Initializing camera...")
    camera = MujocoCamera(env)
    
    # Let the simulation settle a bit
    for _ in range(100):
        env.step()
    
    # 3. Get Segmented Point Clouds
    print("\nGenerating segmented point clouds...")
    start_time = time.time()
    
    # Use multiple cameras for better coverage
    camera_names = ["top_camera", "front_camera", "side_camera"]
    print(f"Using cameras: {camera_names}")
    
    segmented_clouds = camera.get_multi_camera_segmented_pointcloud(
        camera_names=camera_names,
        width=640,
        height=480,
        num_samples_per_camera=500,        # Points per object per camera
        min_depth=0.1,
        max_depth=2.0
    )
    
    elapsed = time.time() - start_time
    print(f"Generation took {elapsed:.3f} seconds")
    
    # 4. Visualize
    if len(segmented_clouds) == 0:
        print("No objects found! Check camera name and scene configuration.")
    else:
        visualize_segmented_clouds(segmented_clouds)
        
    # Cleanup
    camera.close()
    env.close()

if __name__ == "__main__":
    main()
