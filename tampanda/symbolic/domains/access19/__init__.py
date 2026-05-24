"""access-19 domain (Bouhsain HAL 2025).

Forked from :mod:`tampanda.symbolic.domains.tabletop_access` so the
deck-style closed-cubicle variant evolves independently of the
3-tier ``access`` scene.

Public API::

    from tampanda.symbolic.domains.access19 import (
        Access19Config,
        make_access19_builder,
        apply_runtime_tweaks,
        make_tabletop_access_bridge,
        set_objects_at_cells,
        make_access19_pick_fn,
        make_access19_put_fn,
    )
"""

from tampanda.symbolic.domains.access19.env_builder import (
    Access19Config,
    apply_runtime_tweaks,
    make_access19_builder,
)
from tampanda.symbolic.domains.access19.bridge import (
    make_tabletop_access_bridge,
    set_objects_at_cells,
)
from tampanda.symbolic.domains.access19.chains import (
    make_access19_pick_fn,
    make_access19_put_fn,
)

__all__ = [
    "Access19Config",
    "apply_runtime_tweaks",
    "make_access19_builder",
    "make_access19_pick_fn",
    "make_access19_put_fn",
    "make_tabletop_access_bridge",
    "set_objects_at_cells",
]
