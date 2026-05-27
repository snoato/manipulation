"""Dense YCB tabletop-access domain (fork of ``tabletop_access:access``).

Same free-standing 3-tier shelf simulation (Bouhsain HAL 2025), but with
real GSO/YCB meshes, a finer 3 cm grid, multi-cell rectangular footprints,
and per-footprint-size ``pick_<W>x<H>`` / ``put_<W>x<H>`` actions for tight
arrangements.  Two placement regions: ``middle_deck`` + ``top_deck``.

See ``DESIGN.md`` for the full design and ``RGNET_HANDOFF.md`` for the
rgnet integration.  Public API::

    from tampanda.symbolic.domains.tabletop_access_ycb import (
        TabletopAccessYcbConfig, make_tabletop_access_ycb_builder,
        apply_runtime_tweaks, make_tabletop_access_ycb_bridge,
        make_ycb_access_chains, restore_state, build_setup, solve,
    )
    from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
        compute_all_footprints,
    )
    from tampanda.symbolic.domains.tabletop_access_ycb.feasibility import (
        check_action, check_action_sequence,
    )
"""

from tampanda.symbolic.domains.tabletop_access_ycb.env_builder import (
    DEFAULT_ROSTER,
    TabletopAccessYcbConfig,
    apply_runtime_tweaks,
    make_tabletop_access_ycb_builder,
)
from tampanda.symbolic.domains.tabletop_access_ycb.footprint import (
    ObjectFootprint,
    compute_all_footprints,
    compute_footprint,
)
from tampanda.symbolic.domains.tabletop_access_ycb.bridge import (
    make_tabletop_access_ycb_bridge,
)
from tampanda.symbolic.domains.tabletop_access_ycb.state import restore_state
from tampanda.symbolic.domains.tabletop_access_ycb.chains import (
    make_ycb_access_chains,
)
from tampanda.symbolic.domains.tabletop_access_ycb.setup import (
    build_setup,
    YcbAccessSetup,
)
from tampanda.symbolic.domains.tabletop_access_ycb.planner import (
    solve,
    make_fast_oracle,
)

__all__ = [
    "DEFAULT_ROSTER",
    "TabletopAccessYcbConfig",
    "apply_runtime_tweaks",
    "make_tabletop_access_ycb_builder",
    "ObjectFootprint",
    "compute_all_footprints",
    "compute_footprint",
    "make_tabletop_access_ycb_bridge",
    "restore_state",
    "make_ycb_access_chains",
    "build_setup",
    "YcbAccessSetup",
    "solve",
    "make_fast_oracle",
]
