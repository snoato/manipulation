"""Project-wide scene constants — robot-relative defaults.

These values codify a convention used across the multi-grid domains
(``confined_shelf``, ``confined_pickonly``, ``tabletop_access``,
``multilevel_blocks``).  They reflect Franka kinematics, not paper
geometry — fiddling with them changes which cells the robot can
actually reach, so don't override per-domain unless you've measured
the consequences with ``examples/check_executability.py``.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Workspace offset — default x-position of multi-grid scene origins.
# ---------------------------------------------------------------------------

WORKSPACE_X_OFFSET: float = 0.30
"""Default world-x offset for multi-grid scene origins.

Real Franka robot setups mount the arm at the corner of a workspace,
not at the centre.  Centre placement (x=0) lands shelf cells inside a
narrow palm-+y dead zone where ``q4`` hits its joint limit at
z ≈ 0.30–0.55, making FRONT grasps unreachable on the centre column
of any shelf.

The +0.30 default puts every cell of typical multi-grid shelves
(width ≤ 0.6 m) firmly in the Franka's reachable green zone.  The
empirical map this is derived from is produced by
``examples/build_reachability_map.py``; see ``/tmp/reach_map.png``
when that script is run.

Domains that legitimately want centre placement (e.g., open-table
top-down stacking where palm-down works everywhere) can opt out by
setting their ``table_pos.x`` explicitly to 0.0.
"""


# ---------------------------------------------------------------------------
# Shelf-mount default Y — typical "in front of robot" depth.
# ---------------------------------------------------------------------------

WORKSPACE_Y_OFFSET: float = 0.40
"""Default world-y position for shelf bodies in front of the robot.

The Franka comfortably reaches y in [0.25, 0.65] at table-comfortable
heights.  The default 0.40 sits in the middle of that band; bumping
to 0.45–0.50 is fine for shallower domains.
"""


# ---------------------------------------------------------------------------
# Pedestal — additional z-lift for shelves to put cube centres in the
# palm-+y IK convergence band when FRONT grasps are required.
# ---------------------------------------------------------------------------

DEFAULT_SHELF_PEDESTAL: float = 0.18
"""Default pedestal height for closed-top shelves on a table.

With ``table_top_z = 0.27`` and a 4-6 cm cube on the shelf floor, this
puts the cube centre at z ≈ 0.49 — inside the Franka palm-+y IK
convergence band (z ∈ [0.45, 0.55]).  Open-top shelves where TOP_DOWN
is the primary grasp can skip the pedestal entirely.
"""
