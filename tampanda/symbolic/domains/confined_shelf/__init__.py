"""Confined-shelf rearrangement domain (Wang ICAPS-2022).

A free-standing closed cubicle with one open face, populated with N
identical cylinders in K colour groups.  The robot reaches in through
the open face only, and the goal is typically a colour-by-column
arrangement of the cylinders.

Public API::

    from tampanda.symbolic.domains.confined_shelf import (
        ConfinedShelfConfig,
        make_confined_shelf_builder,
        make_confined_shelf_bridge,
        set_cylinders_at_cells,
    )

See ``DIVERGENCES.md`` in this package for the documented differences
from the source paper.
"""

from tampanda.symbolic.domains.confined_shelf.env_builder import (
    STAGING_HOME_QPOS,
    ConfinedShelfConfig,
    apply_runtime_tweaks,
    default_color_sort_goal,
    default_initial_layout,
    make_confined_shelf_builder,
)
from tampanda.symbolic.domains.confined_shelf.bridge import (
    make_confined_shelf_bridge,
    set_cylinders_at_cells,
)
from tampanda.symbolic.domains.confined_shelf.state import (
    check_stability,
    ground_to_object_cells,
    held_object_in_state,
    restore_state,
)
from tampanda.symbolic.domains.confined_shelf.feasibility import (
    check_action,
    check_action_sequence,
    check_fast,
    check_full,
    prefilter_reject,
)
from tampanda.symbolic.domains.confined_shelf.parallel import (
    ParallelFeasibilityChecker,
)

__all__ = [
    "STAGING_HOME_QPOS",
    "ConfinedShelfConfig",
    "apply_runtime_tweaks",
    "default_color_sort_goal",
    "default_initial_layout",
    "make_confined_shelf_builder",
    "make_confined_shelf_bridge",
    "set_cylinders_at_cells",
    # state
    "restore_state",
    "ground_to_object_cells",
    "held_object_in_state",
    "check_stability",
    # feasibility
    "check_action",
    "check_action_sequence",
    "check_fast",
    "check_full",
    "prefilter_reject",
    # parallel
    "ParallelFeasibilityChecker",
]
