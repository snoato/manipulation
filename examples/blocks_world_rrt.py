"""Blocks-world demo: pick two cubes and place them evenly on a platform.

Uses PickPlaceExecutor for all grasping/placing so the main loop is clean.
"""

from pathlib import Path
import time
import numpy as np

from tampanda import RRTStar, GraspPlanner
from tampanda.planners import PickPlaceExecutor
from tampanda.symbolic.domains.blocks import BlocksDomain, BlocksStateManager
from tampanda.symbolic.domains.blocks.env_builder import make_blocks_builder

_OUTPUT_DIR = Path(__file__).parent / "blocks_output"


def main():
    print("=" * 70)
    print("Blocks World Demo  —  RRT* + PickPlaceExecutor")
    print("=" * 70)

    # ------------------------------------------------------------------ env
    print("\n1. Loading environment...")
    env = make_blocks_builder().build_env(rate=200.0)

    # ------------------------------------------------------------------ planner
    print("\n2. Initialising RRT* planner...")
    planner = RRTStar(env)
    planner.max_iterations   = 3000
    planner.step_size        = 0.2
    planner.goal_sample_rate = 0.2

    # ------------------------------------------------------------------ domain
    print("\n3. Creating blocks domain...")
    domain = BlocksDomain(
        model=env.model,
        working_area=(0.4, 0.4),
        offset_x=0.0,
        offset_y=0.0,
        table_body_name="simple_table",
        table_geom_name="simple_table_surface",
    )
    print(f"   Table height: {domain.table_height:.3f} m")

    state_manager = BlocksStateManager(domain, env)

    # ------------------------------------------------------------------ scene setup
    print("\n4. Setting up scene...")
    for i in range(16):
        state_manager._hide_block(i)

    cube1_idx    = 0   # small 4 cm cube
    cube2_idx    = 1   # small 4 cm cube
    platform_idx = 14  # 15 × 10 × 5 cm platform

    bounds   = domain.get_working_bounds()
    center_x = (bounds["min_x"] + bounds["max_x"]) / 2.0
    center_y = (bounds["min_y"] + bounds["max_y"]) / 2.0

    platform_z = domain.table_height + 0.025 + 0.003
    state_manager._set_block_position(platform_idx, center_x, center_y, platform_z)

    cube1_x = center_x - 0.12
    cube1_z = domain.table_height + 0.02 + 0.003
    state_manager._set_block_position(cube1_idx, cube1_x, center_y, cube1_z)

    cube2_x = center_x + 0.12
    cube2_z = domain.table_height + 0.02 + 0.003
    state_manager._set_block_position(cube2_idx, cube2_x, center_y, cube2_z)

    env.reset_velocities()
    env.forward()

    # ------------------------------------------------------------------ placement geometry
    # Even placement: 2 cubes symmetrically on the platform, equal margins.
    platform_w = state_manager.BLOCK_SPECS[platform_idx][0]   # 0.15 m
    cube_w     = state_manager.BLOCK_SPECS[cube1_idx][0]       # 0.04 m
    gap        = (platform_w - 2 * cube_w) / 3                # equal margin + inter-cube gap
    half_span  = platform_w / 2 - gap - cube_w / 2
    offsets    = [(-half_span, 0.0), (half_span, 0.0)]
    print(f"   Platform {platform_w*100:.0f} cm, cube {cube_w*100:.0f} cm → "
          f"offsets ±{half_span*100:.1f} cm")

    # ------------------------------------------------------------------ grasp planner + executor
    grasp_planner = GraspPlanner(table_z=domain.table_height)
    executor = PickPlaceExecutor(
        env, planner, grasp_planner,
        approach_step_size = 0.01,
        grasp_step_size    = 0.003,
        lift_step_size     = 0.003,
        place_step_size    = 0.003,
        use_attachment     = True,
    )

    # ------------------------------------------------------------------ viewer
    with env.launch_viewer() as viewer:
        print("\n5. Settling physics...")
        env.rest(2.0)

        if not viewer.is_running():
            return

        # ---- pick-and-place loop -----------------------------------------
        for trial_idx, (cube_idx, offset) in enumerate(
            [(cube1_idx, offsets[0]), (cube2_idx, offsets[1])]
        ):
            cube_name = f"block_{cube_idx}"
            w, d, h   = state_manager.BLOCK_SPECS[cube_idx]
            half_size = np.array([w / 2, d / 2, h / 2])

            print(f"\n{'='*70}")
            print(f"  ACTION {trial_idx*2+1}: pick {cube_name}")
            print(f"{'='*70}")

            block_pos  = env.get_object_position(cube_name)
            block_quat = env.get_object_orientation(cube_name)

            ok = executor.pick(cube_name, block_pos, half_size, block_quat)
            if not ok:
                print(f"  !! pick failed for {cube_name}, aborting demo")
                break

            if not viewer.is_running():
                return

            # Compute place position: block centre on top of platform
            print(f"\n{'='*70}")
            print(f"  ACTION {trial_idx*2+2}: place {cube_name} on block_{platform_idx}")
            print(f"{'='*70}")

            platform_pos = env.get_object_position(f"block_{platform_idx}")
            _, _, platform_h = state_manager.BLOCK_SPECS[platform_idx]
            # Block centre when placed = platform top + block_h/2
            place_center = np.array([
                platform_pos[0] + offset[0],
                platform_pos[1] + offset[1],
                platform_pos[2] + platform_h / 2 + h / 2,
            ])

            ok = executor.place(
                block_name        = cube_name,
                place_block_center = place_center,
                ee_quat           = np.array([0.0, 1.0, 0.0, 0.0]),
                target_block_name  = f"block_{platform_idx}",
            )
            if not ok:
                print(f"  !! place failed for {cube_name}")

            if not viewer.is_running():
                return

        # ------------------------------------------------------------------ verify
        print("\n6. Settling final state...")
        env.rest(2.0)

        print("\n7. Verifying final state...")
        final = state_manager.ground_state()
        print(f"   On relationships: {final['on']}")

        goal_ok = all([
            (f"block_{cube1_idx}", f"block_{platform_idx}") in final["on"],
            (f"block_{cube2_idx}", f"block_{platform_idx}") in final["on"],
        ])
        print("\n" + "=" * 70)
        if goal_ok:
            print("  SUCCESS: both cubes are on the platform!")
        else:
            print("  PARTIAL: goal may not be fully satisfied")
        print("=" * 70)

        print("\nClose viewer to exit.")
        while viewer.is_running():
            env.step()
            time.sleep(0.01)

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
