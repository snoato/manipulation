# Manipulation

Just another wrapper for robotics manipulation built on top of [MuJoCo](https://github.com/google-deepmind/mujoco) and [MINK](https://github.com/kevinzakka/mink), featuring several grasping and manipulation scenarios. Supports motion planning via RRT implementations, as well as weld constraints with mocap targets.

## Features

- **Environment Simulation**: MuJoCo-based robot environments with collision detection
- **Inverse Kinematics**: Fast IK solving using MINK
- **Motion Planning**: RRT* motion planner with path smoothing
- **Robot Control**: Position-based controller with gravity-compensated trajectory following
- **Grasp Planning**: Geometry-aware `GraspPlanner` with ranked candidates (top-down, front approach), table-clearance and gripper-width checks
- **Pick and Place**: Reusable `PickPlaceExecutor` with multi-candidate retry and kinematic object attachment
- **Franka Panda Support**: Pre-configured support for Franka Emika Panda robot
- **Symbolic Planning**: Grid-based and blocks-world PDDL domains for task planning
- **Camera Support**: RGB, depth, segmentation, and pointcloud rendering via `MujocoCamera`

## Installation

```bash
cd manipulation
pip install -e .
```

## Examples

Run the example scripts:

```bash
cd examples

# Basic control
python basic_ik.py            # IK control — move end-effector to target pose
python basic_rrt.py           # RRT* motion planning — plan and execute collision-free path

# Grasping with hardcoded poses
python grasping_ik.py         # Pick and place with IK control
python grasping_rrt.py        # Pick and place with RRT* planning
python grasping_rrt_camera.py # RRT* grasping + camera capture (headless)

# Grasping with GraspPlanner (geometry-aware poses)
python grasping_ik_planner.py   # IK + GraspPlanner
python grasping_rrt_planner.py  # RRT* + GraspPlanner

# Blocks world
python blocks_world_rrt.py    # Pick two cubes onto a platform using PickPlaceExecutor
python blocks_world_demo.py   # Symbolic state grounding and PDDL generation

# Symbolic planning
python symbolic.py               # Grid-based PDDL planning
python symbolic_grasping_rrt.py  # Symbolic plan executed with RRT*

# Benchmarks
python benchmark_grasping.py           # GraspPlanner + RRT* on blocks (headless)
python benchmark_cylinder_grasping.py  # Direct IK vs RRT* on cylinders (headless)
```

## Package Structure

```
manipulation/
├── core/               # Abstract base classes
├── environments/       # Environment implementations
├── ik/                 # IK implementations
├── planners/           # Motion planners, GraspPlanner, PickPlaceExecutor
├── controllers/        # Controllers
├── perception/         # Camera and pointcloud
├── symbolic/           # Symbolic planning (PDDL)
└── utils/              # Utilities
```

## Quick Start

```python
import numpy as np
from manipulation import FrankaEnvironment, RRTStar

# Initialize environment
env = FrankaEnvironment("path/to/scene.xml")

# Create motion planner
planner = RRTStar(env)

# Plan and execute a path
path = planner.plan(start_config, goal_config)
if path is not None:
    env.execute_path(path, planner)
    env.wait_idle()
```

### Pick and place with GraspPlanner

```python
from manipulation import FrankaEnvironment, RRTStar, GraspPlanner, PickPlaceExecutor

env = FrankaEnvironment("path/to/scene.xml", rate=200.0)
planner = RRTStar(env)
executor = PickPlaceExecutor(env, planner, GraspPlanner(table_z=0.27))

block_pos  = env.get_object_position("block_0")
half_size  = env.get_object_half_size("block_0")
block_quat = env.get_object_orientation("block_0")

executor.pick("block_0", block_pos, half_size, block_quat)
executor.place("block_0", place_center, ee_quat=np.array([0, 1, 0, 0]))
```
