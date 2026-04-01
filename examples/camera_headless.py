"""Headless camera test: capture RGB and depth from all three cameras."""

import numpy as np

from tampanda.perception import MujocoCamera
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder

env = make_symbolic_builder().build_env()

camera = MujocoCamera(env, width=640, height=480)

env.forward()

print("Testing camera capture (headless mode)\n")

cameras = ["top_camera", "side_camera", "front_camera"]

for cam_name in cameras:
    print(f"=== {cam_name} ===")

    rgb = camera.render_rgb(cam_name, width=640, height=480)
    print(f"RGB: shape={rgb.shape}, range=[{rgb.min()}, {rgb.max()}], mean={rgb.mean():.1f}")

    filename = f"test_{cam_name}.png"
    camera.save_image(cam_name, filename)
    print(f"Saved: {filename}")

    depth = camera.render_depth(cam_name, width=640, height=480)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    print(f"Depth: range=[{valid.min():.3f}m, {valid.max():.3f}m], valid={len(valid)}/{depth.size}")

env.close()
print("All images saved.")
print("  - test_top_camera.png")
print("  - test_side_camera.png")
print("  - test_front_camera.png")
