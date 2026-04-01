"""Visual verification for the blocks domain SceneBuilder port.

Spawns the full 16-block scene, places a random selection of blocks on the
table, and opens the MuJoCo viewer.  Use this to confirm the table geometry,
block sizes, and robot home pose all look correct.

Run with::

    cd examples
    mjpython blocks_scene.py
"""

import mujoco
import mujoco.viewer

from manipulation.symbolic.domains.blocks import (
    BlocksDomain,
    BlocksStateManager,
    make_blocks_builder,
)


def main():
    print("Building blocks scene...")
    env = make_blocks_builder().build_env(rate=200.0)

    domain = BlocksDomain(
        model=env.model,
        working_area=(0.4, 0.4),
        offset_x=0.0,
        offset_y=0.0,
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
    )
    sm = BlocksStateManager(domain, env)

    # Place a representative sample: 2 small, 2 medium, 1 platform, 1 large platform
    active = [0, 1, 6, 7, 12, 14]
    positions = [
        (-0.10, -0.05),
        ( 0.05, -0.05),
        (-0.10,  0.10),
        ( 0.05,  0.10),
        (-0.05,  0.20),
        ( 0.05,  0.20),
    ]
    z = domain.table_height
    tb = domain.table_bounds
    cx0 = (tb["min_x"] + tb["max_x"]) / 2
    cy0 = (tb["min_y"] + tb["max_y"]) / 2
    for idx, (ox, oy) in zip(active, positions):
        _, _, h = BlocksStateManager.BLOCK_SPECS[idx]
        sm._set_block_position(idx, cx0 + ox, cy0 + oy, z + h / 2)

    mujoco.mj_forward(env.model, env.data)

    viewer = mujoco.viewer.launch_passive(
        model=env.model, data=env.data,
        show_left_ui=False, show_right_ui=False,
    )
    mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
    viewer.sync()

    print("Scene ready — close the viewer window to exit.")
    print(f"  Table height : {domain.table_height:.4f} m")
    print(f"  Active blocks: {active}")
    while viewer.is_running():
        mujoco.mj_step(env.model, env.data)
        viewer.sync()


if __name__ == "__main__":
    main()
