"""Tabletop-access domain (Bouhsain HAL 2025).

Two scene variants share one PDDL family:

* ``access`` — free-standing 3-tier shelf with YCB-proxy items.
* ``access-19`` — deck-style open-tunnel shelf with 19 generic cubes.

Two PDDL modes are available per the planning discussion:

* ``"filter"`` — face is a refinement detail; PDDL has
  ``(pick ?obj ?cel)`` and ``(put ?obj ?cel)`` only.
* ``"face"`` — PDDL parameterises actions on the chosen face and
  evaluates ``(face-grasp-clear ?obj ?face)`` from the live state.

Public API::

    from tampanda.symbolic.domains.tabletop_access import (
        TabletopAccessConfig,
        make_access_builder,
        make_access19_builder,
        make_tabletop_access_bridge,
        set_objects_at_cells,
    )
"""

from tampanda.symbolic.domains.tabletop_access.env_builder import (
    TabletopAccessConfig,
    access_default_goal,
    access_default_initial_layout,
    apply_runtime_tweaks,
    make_access_builder,
    make_access19_builder,
)
from tampanda.symbolic.domains.tabletop_access.bridge import (
    make_tabletop_access_bridge,
    set_objects_at_cells,
)
from tampanda.symbolic.domains.tabletop_access.chains import (
    make_access19_pick_fn,
    make_access19_put_fn,
)

__all__ = [
    "TabletopAccessConfig",
    "access_default_goal",
    "access_default_initial_layout",
    "apply_runtime_tweaks",
    "make_access_builder",
    "make_access19_builder",
    "make_access19_pick_fn",
    "make_access19_put_fn",
    "make_tabletop_access_bridge",
    "set_objects_at_cells",
]
