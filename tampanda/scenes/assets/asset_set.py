"""Per-domain asset sets — generic boxes and YCB-proxy boxes.

An :class:`Asset` is a parameterised description of one rigid object that
will appear in a scene: half-extents, color, label, optional mass override.
An :class:`AssetSet` is an ordered collection of named assets that a domain
factory hands to a SceneBuilder.

The point of this module is the **swap mechanism**: a domain factory takes
an :class:`AssetSet` argument, so the same domain can be instantiated with
generic boxes today and with YCB meshes later, without changing scene-graph
construction code.

Usage::

    from tampanda.scenes.assets import AssetSet, make_generic_boxes

    boxes = make_generic_boxes(
        prefix="blocker",
        sizes=[(0.05, 0.05, 0.05)] * 5,
    )
    with boxes.materialised() as resources:
        # resources is {asset_id: path_to_template_xml}
        for asset_id, path in resources.items():
            builder.add_resource(asset_id, str(path))
        # ... add objects, build env ...

The materialisation helper writes one MJCF template fragment per asset to a
temporary directory and yields ``{asset_id: path}``.  After the ``with``
block exits, the tempdir is cleaned up — by then the SceneBuilder has
already loaded the template into the assembled scene XML.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

# Conservative default: identity matrix (slightly inflated) prevents tipping
# of tall thin boxes on contact, mirroring the trick used for thin cylinders.
_DEFAULT_DIAGINERTIA = (0.0250, 0.0250, 0.0125)


@dataclass(frozen=True)
class Asset:
    """One parameterised rigid object instantiated from a template fragment.

    Attributes:
        asset_id: Logical identifier — used as the resource name when
            registering with a SceneBuilder.  Must be a valid PDDL atom
            (letters, digits, underscore; no leading digit, no spaces).
        half_extents: Box half-sizes ``(lx/2, ly/2, lz/2)`` in metres.
            For YCB-proxy assets these are the bounding-box half-extents
            of the real mesh; for generic boxes the user picks them.
        color: Optional RGBA tuple; ``None`` keeps the template default.
        label: Free-text human description (for logs and DIVERGENCES.md).
        mass: Optional mass override in kg.  Default 0.05 matches the
            existing cylinder template (intentionally light to keep
            contact dynamics gentle).
        friction: ``(slide, spin, roll)`` friction triple matching MJCF.
    """

    asset_id: str
    half_extents: Tuple[float, float, float]
    color: Optional[Tuple[float, float, float, float]] = None
    label: str = ""
    mass: float = 0.05
    friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if "__" in self.asset_id:
            raise ValueError(f"asset_id must not contain '__': {self.asset_id!r}")
        if any(s <= 0 for s in self.half_extents):
            raise ValueError(f"half_extents must be positive: {self.half_extents}")
        if self.mass <= 0:
            raise ValueError(f"mass must be positive: {self.mass}")

    @property
    def full_size(self) -> Tuple[float, float, float]:
        return (2 * self.half_extents[0], 2 * self.half_extents[1], 2 * self.half_extents[2])

    def render_template_xml(self) -> str:
        """Return an MJCF body fragment string for this asset.

        The fragment uses the SceneBuilder ``_``-prefix convention so that
        per-instance positions / orientations / colors / names are applied
        by the scene builder at instantiation time.
        """
        sx, sy, sz = self.half_extents
        ix, iy, iz = _DEFAULT_DIAGINERTIA
        fx, fy, fz = self.friction
        rgba_attr = ""
        if self.color is not None:
            rgba_attr = f' rgba="{self.color[0]:.4f} {self.color[1]:.4f} ' \
                        f'{self.color[2]:.4f} {self.color[3]:.4f}"'
        return (
            f'<body pos="0 0 0">\n'
            f'  <joint name="_freejoint" type="free"/>\n'
            f'  <inertial mass="{self.mass:.6g}" pos="0 0 0" '
            f'diaginertia="{ix:.6g} {iy:.6g} {iz:.6g}"/>\n'
            f'  <geom name="_geom" type="box" size="{sx:.6g} {sy:.6g} {sz:.6g}"'
            f'{rgba_attr}\n'
            f'        solimp="0.9 0.99 0.001" solref="0.02 1"\n'
            f'        friction="{fx:.6g} {fy:.6g} {fz:.6g}" condim="4"/>\n'
            f'</body>\n'
        )


_SHELF_OPEN_FACES = ("+x", "-x", "+y", "-y", "+z", "-z")


@dataclass(frozen=True)
class Shelf:
    """Static four-walled shelf with one open face, plus optional floor/top.

    The shelf is rendered as a single MJCF body with up to six box geoms,
    one per face.  Faces listed in ``open_faces`` are omitted so the robot
    can reach in.  This covers two distinct shelf shapes seen in the papers:

    * **Confined cubicle** (Wang ICAPS-2022, Saxena E-M4M): one open face
      (``open_faces=("+x",)``); the other five faces are walls.  Robot
      reaches in from the front only.
    * **Open shelf / bookshelf level** (HAL access, access-19): two open
      faces (``open_faces=("+x", "-x")``) — front and back are open, the
      remaining four faces (top deck, bottom deck, two side panels) form
      a tunnel.  The exterior of the top deck is the second placement
      grid for the access-19 environment.

    Coordinates are local to the shelf body — pose it via the SceneBuilder's
    ``add_object("<id>", pos=..., relative_to=...)`` call.

    Attributes:
        asset_id:        Logical identifier (resource key).
        interior_size:   ``(depth_x, width_y, height_z)`` of the empty volume
                         enclosed by the walls, in metres.
        wall_thickness:  Wall plate thickness in metres.
        open_faces:      Tuple of faces with no wall.  Each face is one of
                         ``+x``, ``-x``, ``+y``, ``-y``, ``+z``, ``-z``.
                         Default ``("+x",)`` — front-only opening.
        color:           Wall RGBA.
    """

    asset_id: str
    interior_size: Tuple[float, float, float]
    wall_thickness: float = 0.01
    open_faces: Tuple[str, ...] = ("+x",)
    color: Tuple[float, float, float, float] = (0.92, 0.92, 0.92, 1.0)
    # ``pedestal_height`` extends the bottom (-z) wall downward by this
    # amount so the shelf sits raised above whatever surface it's
    # mounted on.  Used to put cube/cylinder centres in the Franka's
    # comfortable wrist-bend envelope (~z=0.45) when the parent table
    # top is at the standard z=0.27.  Default 0 (no pedestal).
    pedestal_height: float = 0.0

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if "__" in self.asset_id:
            raise ValueError(f"asset_id must not contain '__': {self.asset_id!r}")
        if any(s <= 0 for s in self.interior_size):
            raise ValueError(f"interior_size must be positive: {self.interior_size}")
        if self.wall_thickness <= 0:
            raise ValueError(f"wall_thickness must be positive: {self.wall_thickness}")
        if self.pedestal_height < 0:
            raise ValueError(
                f"pedestal_height must be >= 0: {self.pedestal_height}")
        if not isinstance(self.open_faces, tuple):
            raise TypeError(
                f"open_faces must be a tuple (got {type(self.open_faces).__name__})"
            )
        if len(self.open_faces) >= 6:
            raise ValueError("at least one face must remain closed")
        for face in self.open_faces:
            if face not in _SHELF_OPEN_FACES:
                raise ValueError(
                    f"open face {face!r} not in {_SHELF_OPEN_FACES}"
                )
        if len(set(self.open_faces)) != len(self.open_faces):
            raise ValueError(f"duplicate face in open_faces: {self.open_faces}")

    @property
    def interior_floor_z(self) -> float:
        """Local z of the inner floor surface (where placed objects rest)."""
        return -self.interior_size[2] / 2

    @property
    def exterior_size(self) -> Tuple[float, float, float]:
        t = self.wall_thickness
        sx, sy, sz = self.interior_size
        return (sx + 2 * t, sy + 2 * t, sz + 2 * t)

    def render_template_xml(self) -> str:
        """Return MJCF body fragment for this shelf (static — no freejoint)."""
        sx, sy, sz = self.interior_size
        t = self.wall_thickness
        r, g, b, a = self.color
        rgba_attr = f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.4f}"'

        # Each entry: (name, face, half-extents, position).  The wall is
        # emitted unless ``face`` appears in ``open_faces``.
        # The bottom (-z) "floor" wall is thickened by ``pedestal_height``
        # so the entire shelf sits raised by that amount.  Body-local
        # ``z=0`` remains the centre of the interior cavity.
        ph = self.pedestal_height
        floor_thickness = t + ph
        wall_specs = (
            ("_floor",      "-z",
             (sx / 2 + t, sy / 2 + t, floor_thickness / 2),
             (0.0, 0.0, -sz / 2 - floor_thickness / 2)),
            ("_top",        "+z", (sx / 2 + t, sy / 2 + t, t / 2),     (0.0, 0.0,  sz / 2 + t / 2)),
            ("_wall_neg_x", "-x", (t / 2, sy / 2, sz / 2),             (-sx / 2 - t / 2, 0.0, 0.0)),
            ("_wall_pos_x", "+x", (t / 2, sy / 2, sz / 2),             ( sx / 2 + t / 2, 0.0, 0.0)),
            ("_wall_neg_y", "-y", (sx / 2, t / 2, sz / 2),             (0.0, -sy / 2 - t / 2, 0.0)),
            ("_wall_pos_y", "+y", (sx / 2, t / 2, sz / 2),             (0.0,  sy / 2 + t / 2, 0.0)),
        )

        walls = [
            (name, half, pos)
            for name, face, half, pos in wall_specs
            if face not in self.open_faces
        ]

        lines: List[str] = ['<body pos="0 0 0">']
        lines.append(f'  <!-- shelf {self.asset_id} interior={self.interior_size} '
                     f'open_faces={self.open_faces} '
                     f'pedestal={self.pedestal_height} -->')
        for name, (hx, hy, hz), (x, y, z) in walls:
            lines.append(
                f'  <geom name="{name}" type="box" '
                f'size="{hx:.6g} {hy:.6g} {hz:.6g}" '
                f'pos="{x:.6g} {y:.6g} {z:.6g}" {rgba_attr} '
                f'contype="1" conaffinity="1" mass="100"/>'
            )
        lines.append('</body>')
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class MultiTierShelf:
    """Free-standing multi-deck shelf supported by four corner legs.

    Used for the HAL ``access`` environment, where placement regions exist
    on multiple horizontal levels:

    * **Floor** between the legs (the world floor at ``z = 0``).
    * Top surface of each deck slab.

    Cavities between decks are open on all four sides — the robot reaches
    in horizontally.  Legs are inset from the deck edges by ``leg_inset``
    so objects on the floor can sit beside the legs without overlap.

    Coordinates are local to the shelf body — the body origin sits at the
    floor.  Pose it via ``add_object("<id>", pos=[x, y, 0], …)``.

    Attributes:
        asset_id:        Resource key.
        deck_size:       ``(lx, ly)`` full footprint of each deck slab in m.
        deck_thickness:  Slab thickness in m.
        deck_levels:     Tuple of deck-TOP z-heights above the floor in m,
                         in ascending order.  The HAL access shelf uses
                         two decks: e.g. ``(0.40, 0.62)``.
        leg_size:        ``(lx, ly)`` cross-section of each leg in m.
        leg_inset:       Distance from deck corner inwards to the leg
                         (lets objects on the floor sit beside the legs).
        color:           RGBA.
    """

    asset_id: str
    deck_size: Tuple[float, float]
    deck_thickness: float = 0.015
    deck_levels: Tuple[float, ...] = (0.40, 0.62)
    leg_size: Tuple[float, float] = (0.012, 0.012)
    leg_inset: float = 0.015
    color: Tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    # Optional vertical wall along the body-X axis (i.e., perpendicular to
    # the front face of the shelf) running from the floor up to just below
    # the first deck.  Splits the bottom compartment into left/right
    # halves — used by HAL's standard ``access`` problem.
    floor_separator: bool = False
    floor_separator_thickness: float = 0.005
    # ``base_height`` extends the legs DOWNWARD from the body floor (z=0)
    # by this amount — gives the whole shelf an effective stand so the
    # bottom compartment can be raised into the Franka's reach envelope
    # while keeping the body origin (and hence the grids) at the
    # original z values.  Default 0 (legs start at z=0).
    base_height: float = 0.0
    # ``bottom_plate`` adds an extra deck slab at the body floor (z=0).
    # When True, the lowest workspace tier (``level_z = shelf_pos.z``)
    # has a real surface for objects to rest on instead of floating in
    # mid-air between the legs.  Required for ``access``-style scenes
    # whose 4-region workspace includes ``floor_left`` / ``floor_right``
    # tiers at the body origin level.
    bottom_plate: bool = False

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be non-empty")
        if "__" in self.asset_id:
            raise ValueError(f"asset_id must not contain '__': {self.asset_id!r}")
        if any(s <= 0 for s in self.deck_size):
            raise ValueError(f"deck_size must be positive: {self.deck_size}")
        if self.deck_thickness <= 0:
            raise ValueError(f"deck_thickness must be positive: {self.deck_thickness}")
        if not self.deck_levels:
            raise ValueError("deck_levels must be non-empty")
        if any(z <= 0 for z in self.deck_levels):
            raise ValueError(f"deck_levels must be positive: {self.deck_levels}")
        if list(self.deck_levels) != sorted(self.deck_levels):
            raise ValueError(f"deck_levels must be ascending: {self.deck_levels}")
        if any(s <= 0 for s in self.leg_size):
            raise ValueError(f"leg_size must be positive: {self.leg_size}")
        if self.leg_inset < 0:
            raise ValueError(f"leg_inset must be >= 0: {self.leg_inset}")

    @property
    def top_deck_top_z(self) -> float:
        return max(self.deck_levels)

    def deck_top_z(self, level_idx: int) -> float:
        return self.deck_levels[level_idx]

    def render_template_xml(self) -> str:
        dx, dy = self.deck_size
        t = self.deck_thickness
        leg_lx, leg_ly = self.leg_size
        r, g, b, a = self.color
        rgba_attr = f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.4f}"'

        top_z = self.top_deck_top_z

        leg_corners = [
            ( dx / 2 - self.leg_inset - leg_lx / 2,
              dy / 2 - self.leg_inset - leg_ly / 2),
            (-dx / 2 + self.leg_inset + leg_lx / 2,
              dy / 2 - self.leg_inset - leg_ly / 2),
            ( dx / 2 - self.leg_inset - leg_lx / 2,
             -dy / 2 + self.leg_inset + leg_ly / 2),
            (-dx / 2 + self.leg_inset + leg_lx / 2,
             -dy / 2 + self.leg_inset + leg_ly / 2),
        ]

        lines: List[str] = ['<body pos="0 0 0">']
        lines.append(f'  <!-- multi-tier shelf {self.asset_id} '
                     f'levels={self.deck_levels} -->')

        # Legs span from -base_height (extension below body floor) up to
        # top_z (top deck).  Body origin (z=0) is the bottom-deck level.
        leg_bottom = -self.base_height
        leg_total_z = top_z - leg_bottom
        leg_centre_z = (leg_bottom + top_z) / 2
        for i, (lx, ly) in enumerate(leg_corners):
            lines.append(
                f'  <geom name="_leg_{i}" type="box" '
                f'size="{leg_lx / 2:.6g} {leg_ly / 2:.6g} {leg_total_z / 2:.6g}" '
                f'pos="{lx:.6g} {ly:.6g} {leg_centre_z:.6g}" {rgba_attr} '
                f'contype="1" conaffinity="1" mass="100"/>'
            )

        if self.bottom_plate:
            # Deck slab at the body floor (z=0) so the lowest workspace
            # tier has a real surface.  Its TOP is at z=0; its centre
            # sits at -t/2 so its top edge lines up with the
            # workspace's ``level_z = shelf_pos.z``.
            lines.append(
                f'  <geom name="_deck_bottom" type="box" '
                f'size="{dx / 2:.6g} {dy / 2:.6g} {t / 2:.6g}" '
                f'pos="0 0 {-t / 2:.6g}" {rgba_attr} '
                f'contype="1" conaffinity="1" mass="100"/>'
            )

        for i, deck_top in enumerate(self.deck_levels):
            centre_z = deck_top - t / 2
            lines.append(
                f'  <geom name="_deck_{i}" type="box" '
                f'size="{dx / 2:.6g} {dy / 2:.6g} {t / 2:.6g}" '
                f'pos="0 0 {centre_z:.6g}" {rgba_attr} '
                f'contype="1" conaffinity="1" mass="100"/>'
            )

        if self.floor_separator:
            # Vertical wall splitting the bottom compartment along x=0,
            # spanning the full y-depth and from the floor up to just
            # under the bottom deck.  Tinted slightly darker than the
            # rest of the shelf for visual distinguishability.
            sep_t = self.floor_separator_thickness
            sep_top = self.deck_levels[0] - t
            centre_z = sep_top / 2
            sep_r = max(0.0, r - 0.20)
            sep_g = max(0.0, g - 0.20)
            sep_b = max(0.0, b - 0.20)
            sep_rgba = (
                f'rgba="{sep_r:.4f} {sep_g:.4f} {sep_b:.4f} {a:.4f}"'
            )
            lines.append(
                f'  <geom name="_floor_separator" type="box" '
                f'size="{sep_t / 2:.6g} {dy / 2:.6g} {sep_top / 2:.6g}" '
                f'pos="0 0 {centre_z:.6g}" {sep_rgba} '
                f'contype="1" conaffinity="1" mass="100"/>'
            )

        lines.append('</body>')
        return "\n".join(lines) + "\n"


# Anything stored inside an AssetSet must expose at least these two attributes.
# Using duck typing (rather than a Protocol) to keep Python 3.7 compatibility.
TemplateLike = Union[Asset, Shelf, MultiTierShelf]


class AssetSet:
    """Ordered collection of :class:`Asset` / :class:`Shelf` records."""

    def __init__(self, assets: Iterable[TemplateLike]) -> None:
        self._assets: Dict[str, TemplateLike] = {}
        for a in assets:
            if a.asset_id in self._assets:
                raise ValueError(f"duplicate asset_id {a.asset_id!r}")
            self._assets[a.asset_id] = a

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[TemplateLike]:
        return iter(self._assets.values())

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._assets

    def __getitem__(self, asset_id: str) -> TemplateLike:
        return self._assets[asset_id]

    def ids(self) -> List[str]:
        return list(self._assets)

    def filter(self, prefix: str) -> "AssetSet":
        """Return a sub-set whose ids begin with ``prefix``."""
        return AssetSet(a for a in self if a.asset_id.startswith(prefix))

    @contextlib.contextmanager
    def materialised(self) -> Iterator[Dict[str, Path]]:
        """Yield ``{asset_id: path}`` for the duration of a tempdir.

        Each asset's :meth:`Asset.render_template_xml` is written to a file
        named ``<asset_id>.xml`` inside the tempdir.  Use the returned mapping
        to register templates with a SceneBuilder, then build the env *inside*
        the ``with`` block — the tempdir is deleted as soon as the block exits.
        """
        with tempfile.TemporaryDirectory(prefix="tampanda_assets_") as tmpdir:
            paths: Dict[str, Path] = {}
            tmp = Path(tmpdir)
            for asset in self._assets.values():
                p = tmp / f"{asset.asset_id}.xml"
                p.write_text(asset.render_template_xml())
                paths[asset.asset_id] = p
            yield paths


# ----------------------------------------------------------------------
# Pre-built asset sets
# ----------------------------------------------------------------------

# YCB-bounding-box-sized half-extents (metres).  Sourced from the YCB Object
# Set documentation — values rounded to mm.  These stand in for the real
# meshes until full mesh swap is added.  Names match the YCB labels.
_YCB_PROXY_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    # HALF-extent triples (lx/2, ly/2, lz/2), scaled down from the
    # real YCB dimensions so every item is graspable by the Franka:
    # the gripper max opening is ``MAX_GRIPPER_WIDTH = 0.08 m``, and
    # for the FRONT_X grasp the finger axis is world-x, so each
    # item's ``half_x`` must be ≤ 0.035 m (block width ≤ 0.07 m
    # leaves a small margin).  Height is capped at ``half_z ≤ 0.06``
    # so items fit in the access shelf's 25 cm-tall middle
    # compartment with ~10 cm hand clearance.
    # Height (half_z) is also bounded BELOW by ~0.033 m so the
    # gripper can grasp from above palm-+y without link7 clipping
    # the deck (link7 capsule hangs ~5.5 cm below the EE; grasp z is
    # cube_top - 0.010, so cube_top - EE = 0.010 vs link7 ≈ 0.055
    # below EE → block height ≥ 0.066 → half_z ≥ 0.033).  Items
    # naturally below that are bumped up so every item in the set
    # is graspable.
    "meat_can":         (0.030, 0.029, 0.041),  # potted meat can
    "tomato_soup_can":  (0.0335, 0.0335, 0.050),  # cylindrical soup can
    "mustard_bottle":   (0.030, 0.030, 0.060),  # squat bottle
    "cracker_box":      (0.035, 0.030, 0.060),  # cracker box (shortened)
    "sugar_box":        (0.030, 0.018, 0.060),  # sugar box
    "gelatin_box":      (0.035, 0.014, 0.040),  # gelatin box (bumped from 0.034)
    "pudding_box":      (0.035, 0.020, 0.045),  # pudding box
    "tuna_can":         (0.035, 0.035, 0.040),  # bumped from 0.018 (real tuna
                                                 # is flat but ungraspable
                                                 # palm-+y at that height)
}


def make_generic_boxes(
    prefix: str,
    sizes: List[Tuple[float, float, float]],
    colors: Optional[List[Tuple[float, float, float, float]]] = None,
) -> AssetSet:
    """Build a set of generic boxes.

    Args:
        prefix: Asset id prefix; assets get suffixes ``_0, _1, …``.
        sizes:  List of half-extents ``(lx/2, ly/2, lz/2)`` per box.
        colors: Optional per-box RGBA list.  Cycles if shorter than ``sizes``.
    """
    palette = colors if colors else [
        (0.85, 0.20, 0.20, 1.0),
        (0.20, 0.65, 0.85, 1.0),
        (0.30, 0.70, 0.30, 1.0),
        (0.95, 0.75, 0.20, 1.0),
        (0.55, 0.30, 0.70, 1.0),
        (0.85, 0.50, 0.25, 1.0),
        (0.30, 0.55, 0.50, 1.0),
        (0.70, 0.70, 0.70, 1.0),
    ]
    assets: List[Asset] = []
    for i, half in enumerate(sizes):
        assets.append(Asset(
            asset_id=f"{prefix}_{i}",
            half_extents=half,
            color=palette[i % len(palette)],
            label=f"generic box {i}",
        ))
    return AssetSet(assets)


def make_ycb_proxy(items: List[str]) -> AssetSet:
    """Build a YCB-proxy asset set from a list of YCB names.

    Each name must be a key in :data:`_YCB_PROXY_DIMENSIONS`.  The asset_id
    matches the YCB name.  Colors are set deterministically per-item so
    different runs render consistently.
    """
    palette = [
        (0.80, 0.30, 0.30, 1.0),  # red — tomato/meat
        (0.95, 0.85, 0.30, 1.0),  # yellow — mustard
        (0.85, 0.55, 0.25, 1.0),  # orange — cracker
        (0.85, 0.85, 0.85, 1.0),  # white — sugar
        (0.55, 0.30, 0.20, 1.0),  # brown — pudding
        (0.30, 0.65, 0.75, 1.0),  # cyan — gelatin
        (0.65, 0.65, 0.30, 1.0),  # olive — tuna
        (0.55, 0.55, 0.80, 1.0),  # blue — soup
    ]
    assets: List[Asset] = []
    for i, name in enumerate(items):
        if name not in _YCB_PROXY_DIMENSIONS:
            raise KeyError(
                f"unknown YCB proxy item {name!r}; available: "
                f"{sorted(_YCB_PROXY_DIMENSIONS)}"
            )
        assets.append(Asset(
            asset_id=name,
            half_extents=_YCB_PROXY_DIMENSIONS[name],
            color=palette[i % len(palette)],
            label=f"YCB-proxy box ({name})",
        ))
    return AssetSet(assets)


# Re-exported for convenience.
YCB_PROXY_ITEMS: Tuple[str, ...] = tuple(_YCB_PROXY_DIMENSIONS)
