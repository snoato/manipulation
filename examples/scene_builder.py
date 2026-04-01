"""Demo: programmatic scene construction and hot-reload with live viewer.

Builds a table + cylinders scene from code, runs the simulation until
objects settle, then hot-reloads with extra cylinders added — preserving
the existing objects' positions and velocities.
"""

import time
import mujoco
import mujoco.viewer

from manipulation import FrankaEnvironment, SceneBuilder, SceneReloader
from manipulation.scenes import CYLINDER_TEMPLATE, TABLE_TEMPLATE


def make_builder(n_cylinders: int) -> SceneBuilder:
    import math

    b = SceneBuilder()
    b.add_resource("cylinder", CYLINDER_TEMPLATE)
    b.add_resource("table",    TABLE_TEMPLATE)
    b._options = {
        "timestep": "0.005",
        "integrator": "implicitfast",
        "solver": "Newton",
        "iterations": "20",
        "ls_iterations": "10",
        # Small viscous drag suppresses the slow contact-driven rocking mode
        # without affecting gross dynamics (air ≈ 1.8e-5 Pa·s; 0.01 is ~600×).
        "density": "1.2",
        "viscosity": "0.01",
    }

    b.add_object("table", pos=[0.45, 0.0, 0.0])

    colours = [
        [1.0, 0.2, 0.2, 1.0],  # red
        [0.2, 1.0, 0.2, 1.0],  # green
        [0.2, 0.2, 1.0, 1.0],  # blue
        [1.0, 1.0, 0.2, 1.0],  # yellow
        [1.0, 0.2, 1.0, 1.0],  # magenta
        [0.2, 1.0, 1.0, 1.0],  # cyan
        [1.0, 0.5, 0.2, 1.0],  # orange
        [0.5, 0.2, 1.0, 1.0],  # violet
    ]

    # Arrange cylinders in a circle on the table surface (z=0.35 ≈ table top + half-height)
    radius = 0.12
    for i in range(n_cylinders):
        angle = 2 * math.pi * i / max(n_cylinders, 1)
        x = 0.45 + radius * math.cos(angle)
        y = radius * math.sin(angle)
        b.add_object(
            "cylinder",
            name=f"cyl_{i}",
            pos=[x, y, 0.36],   # 1 cm above resting height (table top 0.27 + half-height 0.08)
            rgba=colours[i % len(colours)],
        )
    return b


def settle(env: FrankaEnvironment, seconds: float):
    steps = int(seconds / env.model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)


def main():
    reloader = SceneReloader()

    # --- Phase 1: 4 cylinders ---
    print("Building scene with 4 cylinders...")
    env = make_builder(4).build_env(rate=200.0)
    viewer = mujoco.viewer.launch_passive(
        model=env.model, data=env.data,
        show_left_ui=False, show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
    mujoco.mj_forward(env.model, env.data)

    print("Cylinders settling — close this window to add 3 more (state preserved).")
    while viewer.is_running():
        settle(env, 0.05)
        viewer.sync()

    # --- Phase 2: hot-reload with 7 cylinders ---
    # The first viewer is now fully closed (user closed it), so mjpython
    # allows a new one to open.
    print("Hot-reloading: adding 3 more cylinders (state preserved)...")
    env = reloader.reload(env, make_builder(7), rate=200.0)

    viewer = mujoco.viewer.launch_passive(
        model=env.model, data=env.data,
        show_left_ui=False, show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
    mujoco.mj_forward(env.model, env.data)
    viewer.sync()

    print("3 new cylinders falling in — close to exit.")
    while viewer.is_running():
        settle(env, 0.05)
        viewer.sync()

    print("Done.")


if __name__ == "__main__":
    main()
