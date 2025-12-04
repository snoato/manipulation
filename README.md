# Manipulation

Just another wrapper for robotics manipulation built on top of [MuJoCo](https://github.com/google-deepmind/mujoco) and [MINK](https://github.com/kevinzakka/mink), featuring several grasping and manipulation scenarios. Supports motion planning via RRT implementations, as well as weld constraints with mocap targets.

## Features

- **Environment Simulation**: MuJoCo-based robot environments with collision detection
- **Inverse Kinematics**: Fast IK solving using MINK
- **Motion Planning**: RRT* motion planner with path smoothing
- **Robot Control**: Position-based controller with trajectory following
- **Franka Panda Support**: Pre-configured support for Franka Emika Panda robot
- **Symbolic Planning**: Grid-based PDDL domain for task planning with multi-cell occupancy

## Installation

```bash
cd manipulation
pip install -e .
```

## Examples

Run the example scripts:

```bash
cd examples

# Basic IK control - move end-effector to target pose
python basic_ik.py

# RRT* motion planning - plan collision-free path
python basic_rrt.py

# Pick and place with IK control - simple grasping
python grasping_ik.py

# Pick and place with RRT* - collision-aware grasping
python grasping_rrt.py

# Symbolic planning with grid-based PDDL domain
python symbolic_planning.py
```

## Package Structure

```
manipulation/
├── core/               # Abstract base classes
├── environments/       # Environment implementations
├── ik/                 # IK implementations
├── planners/           # Motion planners
├── controllers/        # Controllers
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

# Define start and goal configurations
start = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
goal = np.array([0.5, -0.3, 0.2, -1.2, 0.3, 1.8, -0.5])

# Plan path
path = planner.plan(start, goal)

if path is not None:
    # Smooth and execute
    smoothed = planner.smooth_path(path)
    interpolated = env.controller.interpolate_linear_path(smoothed)
    env.controller.follow_trajectory(interpolated)
```