![TAMPanda](assets/TAMPanda_banner.svg)

[![License](https://img.shields.io/badge/license-MIT-bf5fcf?style=flat-square&labelColor=0d0d12)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-bf5fcf?style=flat-square&labelColor=0d0d12)](pyproject.toml)
[![MuJoCo](https://img.shields.io/badge/mujoco-3.x-7eecdf?style=flat-square&labelColor=0d0d12)](https://mujoco.org)

**TAMPanda** is a MuJoCo-based task and motion planning library developed at the Chair of Machine Learning and Reasoning (i6) at RWTH Aachen University. It combines IK, RRT\* motion planning, and PDDL symbolic planning — primarily for the Franka Emika Panda — along with A\* navigation for mobile robots, a programmatic scene builder, and remote asset support. In other words, just another MuJoCo wrapper.

## Features

**Simulation & Control** — MuJoCo environments for the Franka Panda and a differential-drive mobile robot; collision detection, gravity compensation, position-based trajectory controller

**Planning** — IK via [MINK](https://github.com/kevinzakka/mink); RRT\* with path smoothing; geometry-aware grasp candidate ranking; A\* navigation with occupancy-grid obstacle inflation

**Manipulation** — `PickPlaceExecutor` for end-to-end pick-and-place with multi-candidate retry and kinematic object attachment; `PointCloudGraspPlanner` for rudimentary grasp pose computation on unseen objects from segmented point clouds (WIP)

**Symbolic Planning** — `DomainBridge` connects any PDDL domain to the continuous stack: register Python callables as predicate evaluators or action-tracked fluents, map PDDL actions to executors, and call `ground_state`, `plan`, and `execute_action` without domain-specific glue code; built-in tabletop (grid-based) and blocks-world domains; `ActionFeasibilityChecker` validates symbolic actions against the continuous planner before committing; parallel dataset generation with BFS and optional W&B logging

**Scene & Assets** — `SceneBuilder` assembles scenes from reusable MJCF templates at runtime with hot-reload; `YCBDownloader` / `GSODownloader` fetch ~80 YCB objects and ~1 030 Google Scanned Objects on demand; `MujocoCamera` for RGB, depth, segmentation, and pointcloud rendering

**Gymnasium Integration** — `tampanda.gym` wraps any TAMPanda scene as a standard `gymnasium.Env`; configurable observation spaces (joints, EE pose, object poses, RGB, depth, pointcloud, segmented pointcloud, multi-camera pointcloud); three action spaces (joint delta, joint target, Cartesian EE delta via IK); goal-conditioned `TampandaGoalEnv` with HER-compatible `compute_reward`; `DomainBridge` wired as `bridge_factory` for symbolic state in `info` and predicate-vector goals; `PseudoGraspWrapper` for kinematic grasp attachment; `ExpertActionWrapper` for imitation learning; spawn-safe `make_vec_env` for parallel rollouts

## Setup

```bash
pip install -e .
```

Dependencies: `mujoco>=3.0.0`, `mink>=0.0.1`, `numpy`, `loop-rate-limiters`, `matplotlib`, `opencv-python`

> **macOS:** The MuJoCo passive viewer requires `mjpython` instead of `python`. Headless scripts run fine with standard `python`.

## Quick Start

### Build a scene

```python
from tampanda import ArmSceneBuilder
from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE, CYLINDER_THIN_TEMPLATE

builder = ArmSceneBuilder()
builder.add_resource("table",    TABLE_SYMBOLIC_TEMPLATE)
builder.add_resource("cylinder", CYLINDER_THIN_TEMPLATE)
builder.add_resource("can",      {"type": "ycb", "name": "master_chef_can"})
builder.add_object("table",    pos=[0.45,  0.00, 0.00])
builder.add_object("cylinder", pos=[0.40,  0.10, 0.36], rgba=[0.8, 0.3, 0.2, 1.0])
builder.add_object("can",      pos=[0.50, -0.10, 0.33])

env = builder.build_env(rate=200.0)
with env.launch_viewer() as viewer:
    while viewer.is_running():
        env.step()
```

### Pick and place

```python
from tampanda import ArmSceneBuilder, RRTStar, GraspPlanner, PickPlaceExecutor
from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE, BLOCK_SMALL_TEMPLATE
import numpy as np

builder = ArmSceneBuilder()
builder.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)
builder.add_resource("block", BLOCK_SMALL_TEMPLATE)
builder.add_object("table", pos=[0.75, 0.80, 0.00])
builder.add_object("block", pos=[0.45, 0.40, 0.31], rgba=[0.2, 0.5, 0.9, 1.0], name="block_0")

env      = builder.build_env(rate=200.0)
planner  = RRTStar(env)
executor = PickPlaceExecutor(env, planner, GraspPlanner(table_z=0.27))

with env.launch_viewer() as viewer:
    env.rest(2.0)
    ok = executor.pick("block_0",
                       env.get_object_position("block_0"),
                       env.get_object_half_size("block_0"),
                       env.get_object_orientation("block_0"))
    if ok:
        executor.place("block_0", np.array([0.50, 0.25, 0.31]))
    while viewer.is_running():
        env.step()
```

For interactive walkthroughs see [`notebooks/franka_getting_started.ipynb`](notebooks/franka_getting_started.ipynb), [`notebooks/mobile_getting_started.ipynb`](notebooks/mobile_getting_started.ipynb), [`notebooks/domain_bridge_getting_started.ipynb`](notebooks/domain_bridge_getting_started.ipynb), and [`notebooks/gym_grasp_cube.ipynb`](notebooks/gym_grasp_cube.ipynb).

### Gymnasium RL environment

```python
from tampanda.gym import TampandaGymEnv
from tampanda.scenes import ArmSceneBuilder, TABLE_TEMPLATE, BLOCK_MEDIUM_TEMPLATE

builder = ArmSceneBuilder()
builder.add_resource("table", TABLE_TEMPLATE)
builder.add_resource("cube", BLOCK_MEDIUM_TEMPLATE)
builder.add_object("table", pos=[0.45, 0.0, 0.0])
builder.add_object("cube", name="cube_0", pos=[0.45, 0.0, 0.315])
builder.add_camera_orbit("workspace", target=[0.45, 0.0, 0.35], distance=0.75, elevation=45)

env = TampandaGymEnv(
    scene=builder,
    obs=["joints", "ee_pose", "object_poses", "rgb"],
    action_space_type="cartesian_delta",
    cameras=["workspace"],
    image_size=(64, 64),
    reward_fn="dense_grasp",
)
obs, info = env.reset()
```

## Examples

All examples are in `examples/`. On macOS, use `mjpython` for anything that opens a viewer.

**Arm — control and grasping**
- `basic_ik.py` — IK to a target pose, held in viewer
- `basic_rrt.py` — RRT\* between two joint configurations
- `grasping_ik_planner.py`, `grasping_rrt_planner.py` — geometry-aware grasping with ranked candidates
- `blocks_world_rrt.py` — pick two cubes onto a platform with `PickPlaceExecutor`

**Arm — symbolic planning (TAMP)**

The tabletop domain connects PDDL task planning to the continuous planner: symbolic actions (pick, put) are validated with IK + RRT\* before being committed to the plan. `demo_pick_put.py` runs the full loop end-to-end. `DomainBridge` provides a domain-agnostic version of the same pipeline — see the [notebook](notebooks/domain_bridge_getting_started.ipynb) for a walkthrough.

- `symbolic.py` — grid-based PDDL planning in viewer
- `tabletop_interactive.py` — real-time state grounding and interactive tabletop
- `demo_pick_put.py` — full TAMP execution pipeline
- `scene_builder.py` — programmatic scene construction with hot-reload

**Gymnasium / RL**
- `learn_stack_action.py` — SAC + HER to satisfy the `stack` action postconditions via `TampandaGoalEnv` and `DomainBridge`; geometric predicate grounding, shaped reward, `PseudoGraspWrapper`

**Mobile robot**
- `basic_navigation.py` — A\* through a slalom, Lidar/IMU readout at goal
- `square_drive.py` — drift measurement over multiple square laps

**Perception and assets**
- `camera_headless.py`, `pointcloud_demo.py` — RGB and pointcloud capture
- `object_browser.py` — browse, download, and preview YCB / Google Scanned Objects

**Benchmarks** (all headless) — `benchmark_grasping.py`, `benchmark_cylinder_grasping.py`, `benchmark_feasibility.py`, `benchmark_feasibility_params.py`, `benchmark_parallel_rrt.py`, `benchmark_ycb_grasp.py`

## Citation

If you use TAMPanda in your research, please cite:

```bibtex
@software{tampanda,
  title  = {{TAMPanda}: Task and Motion Planning for the Franka Emika Panda},
  author = {Swoboda, Daniel},
  year   = {2025},
  url    = {https://github.com/snoato/TAMPanda},
}
```

## Acknowledgements

- [MuJoCo](https://mujoco.org) (Google DeepMind) — physics engine
- [MINK](https://github.com/kevinzakka/mink) (Kevin Zakka) — differential IK library
- [elpis-lab/ycb_dataset](https://github.com/elpis-lab/ycb_dataset) — YCB object assets
- [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) — Google Scanned Objects assets
