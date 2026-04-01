"""SceneBuilder factory for the tabletop symbolic environment.

Produces a scene equivalent to scene_symbolic.xml: 30 pre-loaded cylinders
(15 thin, 10 medium, 5 thick) parked off-screen, plus the symbolic table.
The pool pattern (hide-at-distance) is preserved so StateManager works
without modification.

Usage::

    from manipulation.symbolic.domains.tabletop.env_builder import make_symbolic_builder

    env  = make_symbolic_builder().build_env(rate=200.0)
    grid = GridDomain(model=env.model, ..., table_geom_name="simple_table_surface")
"""

from manipulation.scenes import (
    SceneBuilder,
    CYLINDER_THIN_TEMPLATE,
    CYLINDER_MEDIUM_TEMPLATE,
    CYLINDER_THICK_TEMPLATE,
    TABLE_SYMBOLIC_TEMPLATE,
)

# 10 distinct colours cycling across the 30-cylinder pool,
# matching the cyl_mat_0 … cyl_mat_9 palette from scene_symbolic.xml.
_COLORS = [
    [1.0, 0.2, 0.2, 1.0],  # cyl_mat_0  red
    [0.2, 1.0, 0.2, 1.0],  # cyl_mat_1  green
    [0.2, 0.2, 1.0, 1.0],  # cyl_mat_2  blue
    [1.0, 1.0, 0.2, 1.0],  # cyl_mat_3  yellow
    [1.0, 0.2, 1.0, 1.0],  # cyl_mat_4  magenta
    [0.2, 1.0, 1.0, 1.0],  # cyl_mat_5  cyan
    [1.0, 0.5, 0.2, 1.0],  # cyl_mat_6  orange
    [0.5, 0.2, 1.0, 1.0],  # cyl_mat_7  violet
    [0.2, 1.0, 0.5, 1.0],  # cyl_mat_8  mint
    [1.0, 0.8, 0.4, 1.0],  # cyl_mat_9  gold
]

# Hide positions match scene_symbolic.xml: x=100, z = half-height + 0.02
_HIDE = {
    "thin":   [100.0, 0.0, 0.10],
    "medium": [100.0, 0.0, 0.12],
    "thick":  [100.0, 0.0, 0.16],
}


def make_symbolic_builder() -> SceneBuilder:
    """Return a SceneBuilder configured for the tabletop symbolic domain.

    The generated scene is functionally equivalent to scene_symbolic.xml:
    - Table body named ``simple_table``; surface geom ``simple_table_surface``
    - 30 cylinders ``cylinder_0`` … ``cylinder_29`` parked at x=100
    - Newton solver + mild viscosity to suppress long-run contact-driven drift

    Pass ``table_geom_name="simple_table_surface"`` when constructing GridDomain.
    """
    b = SceneBuilder()
    b.add_resource("table_symbolic",   TABLE_SYMBOLIC_TEMPLATE)
    b.add_resource("cylinder_thin",    CYLINDER_THIN_TEMPLATE)
    b.add_resource("cylinder_medium",  CYLINDER_MEDIUM_TEMPLATE)
    b.add_resource("cylinder_thick",   CYLINDER_THICK_TEMPLATE)

    b._options = {
        "timestep":      "0.005",
        "iterations":    "5",
        "ls_iterations": "8",
        "integrator":    "implicitfast",
        "solver":        "Newton",
        "density":       "1.2",
        "viscosity":     "0.01",
    }
    b._option_flags     = {"eulerdamp": "disable"}
    b._custom_numerics  = {"max_contact_points": "12"}

    # Table: same world position and orientation as scene_symbolic.xml
    b.add_object("table_symbolic", name="simple_table",
                 pos=[0.0, 0.4, 0.0], quat=[0.0, 0.0, 0.0, 1.0])

    # Thin cylinders: 0–14
    for i in range(15):
        b.add_object("cylinder_thin", name=f"cylinder_{i}",
                     pos=_HIDE["thin"], rgba=_COLORS[i % len(_COLORS)])

    # Medium cylinders: 15–24
    for i in range(15, 25):
        b.add_object("cylinder_medium", name=f"cylinder_{i}",
                     pos=_HIDE["medium"], rgba=_COLORS[i % len(_COLORS)])

    # Thick cylinders: 25–29
    for i in range(25, 30):
        b.add_object("cylinder_thick", name=f"cylinder_{i}",
                     pos=_HIDE["thick"], rgba=_COLORS[i % len(_COLORS)])

    return b
