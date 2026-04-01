"""General-purpose point cloud demo with interactive visualization.

Three modes are available:

  camera  — interactive single/multi-camera RGB pointcloud with camera markers
  compare — multi-camera side-by-side comparison
  segment — segmented per-object pointclouds (one colour per object, all cameras merged)

Usage::

    cd examples
    python pointcloud_demo.py               # prompted mode selection
    python pointcloud_demo.py --mode camera
    python pointcloud_demo.py --mode compare
    python pointcloud_demo.py --mode segment
    python pointcloud_demo.py --mode all
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from tampanda.symbolic.domains.tabletop import GridDomain, StateManager
from tampanda.symbolic.domains.tabletop.env_builder import make_symbolic_builder
from tampanda.perception import MujocoCamera

_CAMERAS = ["top_camera", "side_camera", "front_camera"]
_N_CYLINDERS = 5

_CELL_SIZE     = 0.04
_GRID_OFFSET_X = 0.05
_GRID_OFFSET_Y = 0.25

_HOME_QPOS = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04]
_HOME_CTRL = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]


def _build_scene():
    """Build a symbolic tabletop env with cylinders placed on the table."""
    env = make_symbolic_builder().build_env(rate=200.0)
    grid = GridDomain(
        model=env.model,
        cell_size=_CELL_SIZE,
        working_area=(0.4, 0.3),
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
        grid_offset_x=_GRID_OFFSET_X,
        grid_offset_y=_GRID_OFFSET_Y,
    )
    state_manager = StateManager(grid, env)
    state_manager.sample_random_state(n_cylinders=_N_CYLINDERS)
    env.data.qpos[:8] = _HOME_QPOS
    env.data.ctrl[:8] = _HOME_CTRL
    env.reset_velocities()
    env.forward()
    for _ in range(100):
        env.step()
    return env


def _equal_aspect(ax, points):
    min_v = points.min(axis=0)
    max_v = points.max(axis=0)
    mid   = (max_v + min_v) / 2
    r     = (max_v - min_v).max() / 2
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


# ── Modes ──────────────────────────────────────────────────────────────────

def mode_camera(camera: MujocoCamera) -> None:
    """Interactive single/multi-camera RGB pointcloud with camera positions."""
    print("\nAvailable cameras:")
    print("  1. top_camera")
    print("  2. side_camera")
    print("  3. front_camera")
    print("  4. All cameras (combined)")
    choice = input("\nSelect camera (1-4) [default=1]: ").strip()

    options = {
        "2": [("side_camera",  "cyan")],
        "3": [("front_camera", "blue")],
        "4": [(c, col) for c, col in zip(_CAMERAS, ["red", "green", "blue"])],
    }
    selected = options.get(choice, [("top_camera", "red")])

    all_points, all_colors, cam_markers = [], [], []
    for cam_name, _ in selected:
        print(f"  Generating {cam_name}...")
        pts, cols = camera.get_pointcloud(cam_name, num_samples=4000,
                                          min_depth=0.3, max_depth=3.0)
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)
            cam_pos, _ = camera._get_camera_pose(cam_name)
            cam_markers.append((cam_pos, cam_name))
            print(f"    {len(pts)} points")

    if not all_points:
        print("No points generated.")
        return

    pts_all = np.vstack(all_points)
    col_all = np.vstack(all_colors).astype(float) / 255.0

    print("\nControls: drag to rotate, right-drag to zoom, middle-drag to pan")

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_all[:, 0], pts_all[:, 1], pts_all[:, 2],
               c=col_all, marker=".", s=2, alpha=0.6)

    for cam_pos, cam_name in cam_markers:
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                   c="red", marker="o", s=200, edgecolors="black", linewidths=2,
                   label=cam_name)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + 0.1,
                cam_name.replace("_camera", ""), fontsize=9, weight="bold")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Point Cloud — Interactive View\n(click and drag to rotate)", fontsize=13)
    _equal_aspect(ax, pts_all)

    stats = (f"Points: {len(pts_all)}\n"
             f"X: [{pts_all[:,0].min():.2f}, {pts_all[:,0].max():.2f}]\n"
             f"Y: [{pts_all[:,1].min():.2f}, {pts_all[:,1].max():.2f}]\n"
             f"Z: [{pts_all[:,2].min():.2f}, {pts_all[:,2].max():.2f}]")
    ax.text2D(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
              verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if cam_markers:
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def mode_compare(camera: MujocoCamera) -> None:
    """Multi-camera side-by-side comparison."""
    print("\nGenerating multi-camera comparison...")
    fig = plt.figure(figsize=(15, 5))

    for idx, cam_name in enumerate(_CAMERAS, 1):
        pts, cols = camera.get_pointcloud(cam_name, num_samples=1000,
                                          min_depth=0.3, max_depth=3.0)
        if len(pts) == 0:
            print(f"  {cam_name}: no points")
            continue
        print(f"  {cam_name}: {len(pts)} points")
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=cols.astype(float) / 255.0, marker=".", s=1, alpha=0.6)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(cam_name.replace("_", " ").title(), fontsize=10)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-0.5, 1.5)

    plt.suptitle("Multi-Camera Point Cloud Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()


def mode_segment(camera: MujocoCamera) -> None:
    """Segmented per-object pointclouds (one colour per object)."""
    print("\nGenerating segmented pointclouds from all cameras...")
    segmented = camera.get_multi_camera_segmented_pointcloud(
        camera_names=_CAMERAS,
        width=640, height=480,
        num_samples_per_camera=500,
        min_depth=0.1, max_depth=2.0,
    )

    if not segmented:
        print("No objects found.")
        return

    print(f"Found {len(segmented)} objects:")
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection="3d")

    all_pts = []
    for obj_name, (pts, cols) in segmented.items():
        print(f"  {obj_name}: {len(pts)} points")
        if len(pts) == 0:
            continue
        all_pts.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=cols.astype(float) / 255.0,
                   marker=".", s=5, alpha=0.8, label=obj_name)

    if all_pts:
        _equal_aspect(ax, np.vstack(all_pts))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Segmented Point Clouds")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.show()


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Point cloud demo")
    ap.add_argument("--mode", choices=["camera", "compare", "segment", "all"],
                    help="Visualization mode (default: prompt user)")
    args = ap.parse_args()

    print("Initialising environment...")
    env = _build_scene()

    print("Initialising camera...")
    camera = MujocoCamera(env, width=640, height=480)

    mode = args.mode
    if mode is None:
        print("\nVisualization modes:")
        print("  1. camera  — interactive RGB view with camera markers")
        print("  2. compare — multi-camera side-by-side comparison")
        print("  3. segment — segmented per-object pointclouds")
        print("  4. all     — run all modes in sequence")
        choice = input("\nSelect mode (1-4) [default=1]: ").strip()
        mode = {"2": "compare", "3": "segment", "4": "all"}.get(choice, "camera")

    if mode in ("camera", "all"):
        mode_camera(camera)
    if mode in ("compare", "all"):
        mode_compare(camera)
    if mode in ("segment", "all"):
        mode_segment(camera)

    camera.close()
    env.close()


if __name__ == "__main__":
    main()
