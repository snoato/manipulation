# Manipulation

Just another wrapper for robotics manipulation built on top of [MuJoCo](https://github.com/google-deepmind/mujoco) and [MINK](https://github.com/kevinzakka/mink), featuring several grasping and manipulation scenarios. Supports motion planning via RRT*, symbolic task planning via PDDL, and dataset generation for learning-based methods.

## Features

- **Environment Simulation**: MuJoCo-based robot environments with collision detection
- **Inverse Kinematics**: Fast IK solving using MINK
- **Motion Planning**: RRT* with path smoothing
- **Robot Control**: Position-based controller with gravity-compensated trajectory following
- **Grasp Planning**: Geometry-aware `GraspPlanner` with ranked candidates (top-down, front approach), table-clearance and gripper-width checks
- **Pick and Place**: Reusable `PickPlaceExecutor` with multi-candidate retry and kinematic object attachment
- **Franka Panda Support**: Pre-configured support for Franka Emika Panda robot
- **Symbolic Planning**: Grid-based and blocks-world PDDL domains; `ActionFeasibilityChecker` for validating pick/drop with IK + RRT*
- **Dataset Generation**: Multiprocessing-capable data generation with BFS planning, feasibility validation, and optional W&B logging
- **Camera Support**: RGB, depth, segmentation, and pointcloud rendering via `MujocoCamera`
- **Bundled Scene Constants**: All scene XMLs accessible as `SCENE_SYMBOLIC`, `SCENE_BLOCKS`, etc. — no fragile path wrangling

## Installation

```bash
cd manipulation
pip install -e .
```

Dependencies: `mujoco>=3.0.0`, `mink>=0.0.1`, `numpy`, `loop-rate-limiters`, `matplotlib`, `opencv-python`

## Quick Start

### Motion planning

```python
import numpy as np
from manipulation import FrankaEnvironment, RRTStar, SCENE_DEFAULT

env = FrankaEnvironment(str(SCENE_DEFAULT), rate=200.0)
planner = RRTStar(env)

path = planner.plan(start_config, goal_config)
if path is not None:
    with env.launch_viewer() as viewer:
        env.execute_path(path, planner)
        env.wait_idle()
        env.rest(2.0)
```

### Pick and place

```python
from manipulation import FrankaEnvironment, RRTStar, GraspPlanner, PickPlaceExecutor, SCENE_BLOCKS

env      = FrankaEnvironment(str(SCENE_BLOCKS), rate=200.0)
planner  = RRTStar(env)
executor = PickPlaceExecutor(env, planner, GraspPlanner(table_z=env.table_height))

block_pos  = env.get_object_position("block_0")
half_size  = env.get_object_half_size("block_0")
block_quat = env.get_object_orientation("block_0")

executor.pick("block_0", block_pos, half_size, block_quat)
executor.place("block_0", place_center, ee_quat=np.array([0, 1, 0, 0]))
```

### Feasibility checking (headless)

```python
from manipulation import FrankaEnvironment, RRTStar, SCENE_SYMBOLIC
from manipulation.symbolic import GridDomain, StateManager
from manipulation.symbolic.domains.tabletop.feasibility import ActionFeasibilityChecker
from manipulation.planners.grasp_planner import GraspPlanner

env     = FrankaEnvironment(str(SCENE_SYMBOLIC), rate=200.0)
planner = RRTStar(env)
grid    = GridDomain(env.model, cell_size=0.04, working_area=(0.4, 0.3),
                     grid_offset_x=0.05, grid_offset_y=0.25)
state_manager = StateManager(grid, env)
checker = ActionFeasibilityChecker(env, planner, state_manager,
                                   GraspPlanner(table_z=grid.table_height))

feasible, reason = checker.check("pick", state, cylinder_name="cyl_0")
```

## Examples

```bash
cd examples

# Basic control
python basic_ik.py            # IK control
python basic_rrt.py           # RRT* motion planning

# Grasping with hardcoded poses
python grasping_ik.py         # Pick and place with IK
python grasping_rrt.py        # Pick and place with RRT*
python grasping_rrt_camera.py # RRT* grasping + camera capture (headless)

# Grasping with GraspPlanner (geometry-aware)
python grasping_ik_planner.py
python grasping_rrt_planner.py

# Blocks world
python blocks_world_rrt.py    # Pick two cubes onto a platform using PickPlaceExecutor
python blocks_world_demo.py   # Symbolic state grounding + PDDL generation

# Symbolic / tabletop
python symbolic.py               # Grid-based PDDL planning
python symbolic_grasping_rrt.py  # Symbolic plan executed with RRT*
python demo_solve.py             # Headless BFS planning → full physical execution in viewer

# Benchmarks (all headless, fast)
python benchmark_grasping.py              # GraspPlanner + RRT* on blocks
python benchmark_cylinder_grasping.py     # Direct IK vs RRT* on cylinders
python benchmark_feasibility.py           # ActionFeasibilityChecker correctness
python benchmark_feasibility_params.py    # RRT/IK parameter sweep (finds fastest zero-FN combo)
```

## Data Generation

Generate PDDL problems with motion-planning-validated feasibility labels:

```bash
python -m manipulation.symbolic.domains.tabletop.generate_data \
    --num-train 200 --num-val 20 --num-test 20 \
    --output-dir data/tabletop \
    --num-workers 0          # 0 = all available CPUs
```

Key flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--num-workers` | 1 | 0 = all CPUs |
| `--grid-offset-x` | 0.05 | calibrated reachable zone |
| `--grid-offset-y` | 0.25 | calibrated reachable zone |
| `--rrt-iters` | 1000 | benchmarked fastest zero-false-negative |
| `--ik-iters` | 100 | benchmarked fastest zero-false-negative |
| `--ik-pos-thresh` | 0.005 | benchmarked fastest zero-false-negative |
| `--wandb` | off | enable W&B logging |
| `--no-viz` | off | skip per-instance PNG (faster on headless) |

For cluster runs, see `slurm/generate_data.sbatch`.

## Package Structure

```
manipulation/
├── core/               # Abstract base classes
├── environments/
│   └── assets/         # Bundled scene XMLs + SCENE_* constants
├── ik/                 # MinkIK
├── planners/           # RRTStar, GraspPlanner, PickPlaceExecutor
├── controllers/        # PositionController
├── perception/         # MujocoCamera (RGB, depth, seg, pointcloud)
├── symbolic/
│   └── domains/
│       ├── tabletop/   # GridDomain, StateManager, feasibility, generate_data
│       └── blocks/     # BlocksDomain, BlocksStateManager
└── utils/
```
