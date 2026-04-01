"""Visualise the holding-state load to verify cylinder placement at the EE.

Cycles through four views so you can confirm the cylinder ends up physically
inside the gripper when 'holding' is set in the symbolic state:

  Phase 1 — TABLE:   one cylinder on the table, gripper empty.
  Phase 2 — HOLDING: same cylinder registered as held; should appear at the EE.
  Phase 3 — HOME:    raw home config, no cylinders, so you can see the EE alone.
  Phase 4 — REPEAT

Usage::

    cd examples
    python verify_holding_state.py
"""

import time

import mujoco
import numpy as np

from tampanda import FrankaEnvironment, SCENE_SYMBOLIC
from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.symbolic.domains.tabletop.grid_domain import GridDomain
from tampanda.symbolic.domains.tabletop.state_manager import StateManager

_XML           = SCENE_SYMBOLIC
_GRID_WIDTH    = 0.4
_GRID_HEIGHT   = 0.3
_CELL_SIZE     = 0.04
_GRID_OFFSET_X = 0.05
_GRID_OFFSET_Y = 0.25
_PHASE_SECS    = 3.0   # seconds per phase


def _print_ee_and_cylinder(env, cylinder_name: str):
    """Print EE site position and cylinder body position side-by-side."""
    mujoco.mj_forward(env.model, env.data)

    ee_pos = env.data.site_xpos[env._ee_site_id].copy()
    ee_mat = env.data.site_xmat[env._ee_site_id].reshape(3, 3).copy()

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_name)
    cyl_pos = env.data.xpos[body_id].copy()

    expected = ee_pos + ee_mat @ np.array([0.0, 0.0, GRASP_CONTACT_OFFSET])
    error    = np.linalg.norm(cyl_pos - expected)

    print(f"  EE site pos      : {ee_pos}")
    print(f"  Cylinder pos     : {cyl_pos}")
    print(f"  Expected (EE+off): {expected}")
    print(f"  Position error   : {error*1000:.2f} mm")


def main():
    env = FrankaEnvironment(_XML.as_posix(), rate=200.0)

    grid = GridDomain(
        model=env.model,
        cell_size=_CELL_SIZE,
        working_area=(_GRID_WIDTH, _GRID_HEIGHT),
        table_body_name="simple_table",
        table_geom_name="table_surface",
        grid_offset_x=_GRID_OFFSET_X,
        grid_offset_y=_GRID_OFFSET_Y,
    )
    state_manager = StateManager(grid, env)

    # Use a thick cylinder (index 25) so it's easy to see in the viewer
    CYL = "cylinder_25"
    # Place it in the middle of the grid for Phase 1
    mid_cell = f"cell_{grid.cells_x // 2}_{grid.cells_y // 2}"

    state_on_table = {
        "cylinders": {CYL: [mid_cell]},
        "gripper_empty": True,
        "holding": None,
    }
    state_holding = {
        "cylinders": {},
        "gripper_empty": False,
        "holding": CYL,
    }
    state_home = {
        "cylinders": {},
        "gripper_empty": True,
        "holding": None,
    }

    phases = [
        ("TABLE  — cylinder on table, gripper empty", state_on_table),
        ("HOLDING — cylinder should be at the EE",    state_holding),
        ("HOME   — no cylinders, bare EE reference",  state_home),
    ]
    phase_idx = 0
    phase_start = None

    with env.launch_viewer() as viewer:
        while viewer.is_running():
            now = time.time()

            if phase_start is None or (now - phase_start) > _PHASE_SECS:
                label, state = phases[phase_idx % len(phases)]
                print(f"\n{'='*55}")
                print(f"  Phase {phase_idx % len(phases) + 1}: {label}")
                print(f"{'='*55}")

                state_manager.set_from_grounded_state(state)
                mujoco.mj_forward(env.model, env.data)

                if state.get("holding"):
                    _print_ee_and_cylinder(env, state["holding"])
                else:
                    ee_pos = env.data.site_xpos[env._ee_site_id].copy()
                    print(f"  EE site pos: {ee_pos}  (no cylinder held)")

                phase_idx  += 1
                phase_start = now

            env.step()


if __name__ == "__main__":
    main()
