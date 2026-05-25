"""Multi-level wooden-blocks domain (Kulshrestha CoRL-2023, redesigned 2026-05).

Two-table workspace: a 2D ``parts`` grid behind the robot, a 3D ``stack``
grid in front modelled as five stacked 2D :class:`GridRegion`s
(``stack_L0`` … ``stack_L4``).  Blocks come in two shapes — 1×1 cubes
and 2×1 oblong blocks — and the oblong supports three placement
orientations (flat-x, flat-y, upright).

Public API::

    from tampanda.symbolic.domains.multilevel_blocks import (
        MultilevelBlocksConfig,
        make_multilevel_blocks_builder,
        cube_block_name,
        oblong_block_name,
        stack_region_name,
    )

The bridge and reachability modules are temporarily not exported here
while the symbolic-side rewrite is in progress (Phase 2).
"""

from tampanda.symbolic.domains.multilevel_blocks.env_builder import (
    MultilevelBlocksConfig,
    apply_runtime_tweaks,
    cube_block_name,
    long_block_name,
    make_multilevel_blocks_builder,
    oblong_block_name,
    stack_region_name,
)
from tampanda.symbolic.domains.multilevel_blocks.bridge import (
    make_multilevel_blocks_bridge,
)
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    MultilevelBlocksExecutor,
    register_executor,
)
from tampanda.symbolic.domains.multilevel_blocks.parallel import (
    ParallelFeasibilityChecker,
)
from tampanda.symbolic.domains.multilevel_blocks.state import (
    check_stability,
    ground_to_block_layout,
    held_block_in_state,
    restore_state,
)

__all__ = [
    "MultilevelBlocksConfig",
    "MultilevelBlocksExecutor",
    "ParallelFeasibilityChecker",
    "apply_runtime_tweaks",
    "check_stability",
    "cube_block_name",
    "ground_to_block_layout",
    "held_block_in_state",
    "long_block_name",
    "make_multilevel_blocks_builder",
    "make_multilevel_blocks_bridge",
    "oblong_block_name",
    "register_executor",
    "restore_state",
    "stack_region_name",
]
