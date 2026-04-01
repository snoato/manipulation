# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -e .
```

Dependencies: `mujoco>=3.0.0`, `mink>=0.0.1`, `numpy`, `loop-rate-limiters`, `matplotlib`, `opencv-python`

## Running Examples

There is no formal test suite. Validation is done by running example scripts:

```bash
cd examples
python basic_ik.py           # Basic IK control
python basic_rrt.py          # RRT* motion planning
python grasping_ik.py        # Pick-and-place with IK
python grasping_rrt.py       # Pick-and-place with RRT*
mjpython symbolic.py         # Grid-based PDDL planning (tabletop domain)
mjpython blocks_scene.py     # Blocks domain visual verification
mjpython scene_builder.py    # Programmatic scene construction + hot-reload demo
python -m manipulation.symbolic.domains.tabletop.generate_data  # Tabletop data generation (supports multiprocessing)
```

## Architecture

This is a robotics manipulation library wrapping MuJoCo and MINK for the Franka Emika Panda robot. It combines continuous motion planning with discrete symbolic (PDDL) planning.

### Core Components

**`SceneBuilder`** (`scenes/builder.py`) — Assembles MJCF scenes from named object templates (`scenes/templates/objects/`). Call `builder.build_env()` instead of `FrankaEnvironment(xml_path)` for any domain that has an `env_builder.py`. Each symbolic domain exports a factory: `make_symbolic_builder()` (tabletop) and `make_blocks_builder()` (blocks).

**`FrankaEnvironment`** (`environments/franka_env.py`) — The main entry point. Wraps MuJoCo simulation, manages collision detection, and provides access to IK and controller. Instantiate via a domain builder rather than a raw XML path.

**`RRTStar`** (`planners/rrt_star.py`) — RRT* motion planner. Plans collision-free paths in joint configuration space. Includes path smoothing. Key params: `max_iterations`, `step_size`, `search_radius`, `goal_threshold`.

**`MinkIK`** (`ik/mink_ik.py`) — IK solver using the MINK library. Solves end-effector pose tasks with configurable position/orientation thresholds.

**`PositionController`** (`controllers/position_controller.py`) — Trajectory-following controller with state machine (IDLE → MOVING → GRASPING). Uses linear interpolation between waypoints.

**`MujocoCamera`** (`perception/mujoco_camera.py`) — Renders RGB, depth, and segmentation images. Generates 3D pointclouds from depth, with object filtering by name pattern.

### Symbolic Planning

`symbolic/` integrates PDDL task planning with continuous robot control:

- **`GridDomain`** (`domains/tabletop/`) — Discretizes the workspace into a grid; supports multi-cell object occupancy. Generates PDDL problems from current world state.
- **`BlocksDomain`** (`domains/blocks/`) — Continuous blocks-world domain with a separate PDDL formulation.
- **`StateManager`** — Grounds symbolic state from MuJoCo simulation; maps PDDL predicates to physical object positions.

### Typical Planning + Execution Flow

1. `RRTStar.plan(start_q, goal_q)` → joint-space path
2. `RRTStar.smooth_path(path)` → smoothed path
3. `PositionController.interpolate_linear_path(path)` → dense trajectory
4. `PositionController.follow_trajectory(traj)` → executes in sim

For symbolic tasks: `StateManager` grounds the current state → PDDL problem generated → symbolic plan returned → each action dispatched to the continuous pipeline above.

### Abstract Base Classes (`core/`)

`BaseEnvironment`, `BaseMotionPlanner`, `BaseController`, `BaseIK` — extend these to add new environments or algorithms.
