# Blocks World Symbolic Domain

A blocks world symbolic planning domain for stacking cubes and cuboids with continuous spatial representation.

## Overview

This domain extends the manipulation framework with a classic blocks-world planning interface while maintaining continuous physics simulation. Unlike the grid-based tabletop domain, this domain does not discretize space into cells - objects maintain continuous XY coordinates within a bounded working area.

## Components

### BlocksDomain (`blocks_domain.py`)
- Defines a continuous working area on the table surface
- No grid discretization - pure continuous space
- Configurable working area size and offset
- Validates positions are within bounds

### BlocksStateManager (`blocks_state_manager.py`)
- Manages 16 blocks: 12 graspable cubes, 4 platform cuboids
- Block specifications:
  - 6× small cubes: 4×4×4 cm (blocks 0-5)
  - 6× medium cubes: 6×6×6 cm (blocks 6-11)
  - 2× platforms: 10×10×5 cm (blocks 12-13)
  - 2× large platforms: 15×10×5 cm (blocks 14-15)
- Implements blocks-world predicates:
  - `on(A, B)`: Block A is on block B (80% XY overlap + Z-height check)
  - `on-table(A)`: Block A is resting on the table
  - `clear(A)`: No blocks on top of A
  - `holding(gripper, A)`: Gripper is holding block A
  - `gripper-empty(gripper)`: Gripper is empty
- Provides pose computation methods:
  - `compute_pickup_pose()`: Top-down grasp at block center
  - `compute_stack_pose()`: Place on top of another block
  - `compute_table_pose()`: Place on table at coordinates
- Random state sampling with collision avoidance
- PDDL problem generation with goal specifications

### PDDL Domain (`pddl/blocks_domain.pddl`)
Classic blocks-world PDDL with 4 actions:
- `pick-from-table`: Pick up block from table
- `place-on-table`: Place held block on table
- `stack`: Stack held block on another block
- `unstack`: Remove block from top of another

### MuJoCo Scene (`scene_blocks.xml`)
- 16 block bodies with free joints
- Box geometries with proper inertial properties
- 10 cycling color materials
- Optimized contact parameters for stable stacking
- Initial hidden positions (x=100)

## Usage

### Basic State Sampling and PDDL Generation

```python
from manipulation import FrankaEnvironment
from manipulation.symbolic.domains.blocks import BlocksDomain, BlocksStateManager

# Initialize environment
env = FrankaEnvironment("scene_blocks.xml", rate=200.0)

# Create domain with 40×40cm working area
domain = BlocksDomain(
    model=env.model,
    working_area=(0.4, 0.4),
    offset_x=0.0,
    offset_y=0.0
)

# Create state manager
state_manager = BlocksStateManager(domain, env)

# Sample random configuration
state_manager.sample_random_state(
    n_blocks=5,  # 5 graspable cubes
    include_platforms=True,  # Include platform bases
    seed=42
)

# Ground symbolic state
state = state_manager.ground_state()
print(f"On table: {state['on_table']}")
print(f"On relationships: {state['on']}")
print(f"Clear blocks: {state['clear']}")

# Generate PDDL problem with goal
state_manager.generate_pddl_problem(
    problem_name="stack-task",
    output_path="problem.pddl",
    goal_predicates=[
        "(on block_0 block_12)",  # Stack cube 0 on platform 12
        "(clear block_0)"
    ]
)
```

### Pose Computation for Pick and Place

```python
# Compute pickup pose for a block
pickup_pos, pickup_quat = state_manager.compute_pickup_pose(block_idx=0)

# Compute stacking pose
stack_pos, stack_quat = state_manager.compute_stack_pose(
    target_block_idx=12,  # Stack onto platform
    block_to_place_idx=0  # Block being placed
)

# Compute table placement pose
table_pos, table_quat = state_manager.compute_table_pose(
    x=0.4, y=0.3,
    block_idx=0
)
```

## Examples

### `test_blocks_world.py`
Simple test demonstrating:
- Domain creation
- Random state sampling
- State grounding
- PDDL generation
- Pose computation

Run: `python examples/test_blocks_world.py`

### `blocks_world_demo.py`
Full demonstration with viewer showing:
- Manual block placement setup
- PDDL problem generation
- Simulated pick-and-place actions
- Goal achievement verification
- Physics validation

Run: `mjpython examples/blocks_world_demo.py`

## Key Design Decisions

1. **Continuous Space**: No grid discretization - maintains precise continuous positions
2. **Stacking Detection**: 80% XY overlap threshold + Z-height tolerance (1cm)
3. **Top-Down Grasping**: Only vertical approach grasps to simplify manipulation
4. **Platform Blocks**: Large cuboids (10×10×5, 15×10×5) serve as bases but aren't graspable
5. **Collision Avoidance**: Random sampling uses bounding box checks with 1cm clearance
6. **Physics Stability**: 3mm clearance above surfaces + contact parameters tuned for stable stacking

## Integration Notes

- Domain uses same table geometry detection as GridDomain
- Compatible with existing FrankaEnvironment interface
- Pose methods return positions/quaternions suitable for IK/motion planning
- PDDL domain follows STRIPS formalism for planner compatibility
- State manager tracks gripper state manually (not from physics)

## Limitations

1. **Orientation**: Assumes axis-aligned blocks (no rotation handling)
2. **Gripper Tracking**: Gripper state is manually managed, not physics-based
3. **Large Block Placement**: Large platforms may fail to place in crowded scenes
4. **No Side Grasps**: Only supports top-down vertical grasps
5. **Static Goals**: PDDL goals must be manually specified (no automatic goal generation)

## Future Extensions

- Support for oriented blocks and rotation
- Side grasp computation for blocks exceeding gripper width
- Automatic goal generation from initial/desired states
- Integration with motion planners (RRT*, etc.)
- Multi-robot coordination
- Dynamic re-planning based on physics state
