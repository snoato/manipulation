import numpy as np
from tampanda.environments.franka_env import FrankaEnvironment
from tampanda.perception import MujocoCamera
from tampanda import SCENE_SYMBOLIC

# Create environment (headless - no viewer)
env = FrankaEnvironment(str(SCENE_SYMBOLIC))

# Create camera instance
camera = MujocoCamera(env, width=640, height=480)

# Update scene
env.forward()

print("Testing camera capture (headless mode)\n")

# Test all three cameras
cameras = ["top_camera", "side_camera", "front_camera"]

for cam_name in cameras:
    print(f"=== {cam_name} ===")
    
    # Capture RGB
    rgb = camera.render_rgb(cam_name, width=640, height=480)
    print(f"RGB: shape={rgb.shape}, range=[{rgb.min()}, {rgb.max()}], mean={rgb.mean():.1f}")
    
    # Save snapshot
    filename = f"test_{cam_name}.png"
    camera.save_image(cam_name, filename)
    print(f"Saved: {filename}")
    
    # Capture depth
    depth = camera.render_depth(cam_name, width=640, height=480)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    print(f"Depth: range=[{valid.min():.3f}m, {valid.max():.3f}m], valid={len(valid)}/{depth.size}")

env.close()
print("✓ All images saved. Open them to verify the scene is visible.")
print("  - test_top_camera.png (overhead view)")
print("  - test_side_camera.png (side view)")
print("  - test_front_camera.png (front view)")
