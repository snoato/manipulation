"""SceneBuilder factory for the blocks world environment.

Produces a scene equivalent to scene_blocks.xml: 16 pre-loaded blocks
(6 small cubes, 6 medium cubes, 2 platforms, 2 large platforms) parked
off-screen, plus the symbolic table.

Usage::

    from tampanda.symbolic.domains.blocks.env_builder import make_blocks_builder

    env    = make_blocks_builder().build_env(rate=200.0)
    domain = BlocksDomain(model=env.model, ..., table_geom_name="simple_table_surface")
"""

from tampanda.scenes import (
    SceneBuilder,
    TABLE_SYMBOLIC_TEMPLATE,
    BLOCK_SMALL_TEMPLATE,
    BLOCK_MEDIUM_TEMPLATE,
    BLOCK_PLATFORM_TEMPLATE,
    BLOCK_LARGE_PLATFORM_TEMPLATE,
)

# 10-colour palette matching block_mat_0 … block_mat_9 from scene_blocks.xml
_COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # block_mat_0  red
    [0.2, 1.0, 0.2, 1.0],  # block_mat_1  green
    [0.2, 0.2, 1.0, 1.0],  # block_mat_2  blue
    [1.0, 1.0, 0.2, 1.0],  # block_mat_3  yellow
    [1.0, 0.2, 1.0, 1.0],  # block_mat_4  magenta
    [0.2, 1.0, 1.0, 1.0],  # block_mat_5  cyan
    [1.0, 0.5, 0.2, 1.0],  # block_mat_6  orange
    [0.5, 0.2, 1.0, 1.0],  # block_mat_7  violet
    [0.2, 1.0, 0.5, 1.0],  # block_mat_8  mint
    [1.0, 0.8, 0.4, 1.0],  # block_mat_9  gold
]

# (template_key, hide_z, count, first_index)
# hide_z = half-height of the block so it rests flush at x=100
_POOL = [
    ("block_small",         0.020, 6,  0),   # blocks 0–5
    ("block_medium",        0.030, 6,  6),   # blocks 6–11
    ("block_platform",      0.025, 2, 12),   # blocks 12–13
    ("block_large_platform",0.025, 2, 14),   # blocks 14–15
]


def make_blocks_builder() -> SceneBuilder:
    """Return a SceneBuilder configured for the blocks world domain.

    The generated scene is functionally equivalent to scene_blocks.xml:
    - Table body named ``simple_table``; surface geom ``simple_table_surface``
    - 16 blocks ``block_0`` … ``block_15`` parked at x=100

    Pass ``table_geom_name="simple_table_surface"`` when constructing BlocksDomain.
    """
    b = SceneBuilder()
    b.add_resource("table_symbolic",      TABLE_SYMBOLIC_TEMPLATE)
    b.add_resource("block_small",         BLOCK_SMALL_TEMPLATE)
    b.add_resource("block_medium",        BLOCK_MEDIUM_TEMPLATE)
    b.add_resource("block_platform",      BLOCK_PLATFORM_TEMPLATE)
    b.add_resource("block_large_platform",BLOCK_LARGE_PLATFORM_TEMPLATE)

    b._options = {
        "timestep":      "0.005",
        "iterations":    "5",
        "ls_iterations": "8",
        "integrator":    "implicitfast",
        "solver":        "Newton",
        "density":       "1.2",
        "viscosity":     "0.01",
    }
    b._option_flags    = {"eulerdamp": "disable"}
    b._custom_numerics = {"max_contact_points": "12"}

    # Table: same world position and orientation as scene_blocks.xml
    b.add_object("table_symbolic", name="simple_table",
                 pos=[0.0, 0.4, 0.0], quat=[0.0, 0.0, 0.0, 1.0])

    # Block pool
    color_idx = 0
    for tmpl, hide_z, count, first in _POOL:
        for k in range(count):
            idx = first + k
            b.add_object(tmpl, name=f"block_{idx}",
                         pos=[100.0, 0.0, hide_z],
                         rgba=_COLORS[color_idx % len(_COLORS)])
            color_idx += 1

    return b
