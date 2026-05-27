"""SceneBuilder factory for the dense YCB tabletop-access variant.

A clean fork of ``tabletop_access:access`` (Bouhsain HAL 2025, the
free-standing 3-tier shelf).  Differences from the parent:

* **Real GSO/YCB meshes** replace the uniform placeholder boxes — a
  validated, graspable, handle-free roster spanning several footprint
  sizes.  Mesh mass + inertia are overridden at build time to match the
  old placeholder boxes (light + contact-gentle) since the raw meshes
  carry density-derived masses of 0.7–5.7 kg.
* **Finer 3 cm grid** so larger objects occupy more than one cell,
  enabling tighter arrangements.
* **Two placement regions only** — ``middle_deck`` + ``top_deck``.  The
  bottom floor compartments are dropped (cramped under-deck reach +
  documented cross-region put basin coupling); the middle→top retrieval
  is the paper's core task and gives ample scratch space.

The shelf body itself is byte-for-byte the parent ``access`` shelf, so
this is genuinely "the same simulation environment".

Public entry point: :func:`make_tabletop_access_ycb_builder` returning
``(builder, workspace, config)`` — same shape as the parent factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tampanda.scenes import ArmSceneBuilder
from tampanda.scenes.assets.asset_set import MultiTierShelf
from tampanda.symbolic.workspace import GridRegion, Workspace


# ----------------------------------------------------------------------
# Roster — (body_id, ycb_source).  body_id is the PDDL atom / scene name;
# ycb_source is the cached YCB object the mesh is loaded from.  The OoI is
# sourced from a distinctive object but named ``ooi`` so the bridge finds
# it (mirrors the parent domains).
#
# Every entry was validated by examples/ta_ycb_measure_probe.py:
# stable rest, no protruding handle, min horizontal extent < 8 cm (the
# Franka gripper opening).  Tall items at risk in the middle compartment
# are pruned by the Phase-2 reachability sweep, not here.
# ----------------------------------------------------------------------

DEFAULT_ROSTER: Tuple[Tuple[str, str], ...] = (
    ("ooi", "tomato_soup_can"),       # OoI — distinctive red can; FULL-passing
    ("a_lego_duplo", "a_lego_duplo"),
    ("b_lego_duplo", "b_lego_duplo"),
    ("c_lego_duplo", "c_lego_duplo"),
    ("d_lego_duplo", "d_lego_duplo"),
    ("e_lego_duplo", "e_lego_duplo"),
    ("f_lego_duplo", "f_lego_duplo"),
    ("strawberry", "strawberry"),
    ("orange", "orange"),
    ("potted_meat_can", "potted_meat_can"),
    ("master_chef_can", "master_chef_can"),
)
# Excluded from the roster (validated unsuitable):
#   mustard_bottle — FAST-only: tall (9.9 cm) tapered bottle; FULL
#     put-to-top fails at a top-deck row-step (held-object drift).
#   mug / pitcher_base — protruding handles.
#   a_colored_wood_blocks — 13.5 cm wide (> gripper).
#   peach / pear — unstable rollers.  bleach_cleanser / g_lego_duplo — too tall.


@dataclass(frozen=True)
class TabletopAccessYcbConfig:
    """Parameters for the dense-YCB tabletop-access scene.

    Attributes:
        object_ids:     Ordered scene/PDDL body ids of all movables.
        ooi_id:         Id of the distinguished target.
        cell_size:      Fine grid resolution (m).  3 cm default.
        shelf_pos:      World XYZ of the shelf body origin.
        deck_levels:    Deck-TOP z-heights above the shelf base.
        hide_far_x:     Sentinel x for parked objects.
        hand_capsule_radius_override:  Runtime shrink of the Franka wrist
            guard (None disables).
        match_mass / match_diaginertia / match_identity_iquat:  Inertial
            override applied to every movable so meshes behave like the
            old placeholder boxes (light, contact-gentle, tip-resistant).
    """

    object_ids: Tuple[str, ...]
    roster: Tuple[Tuple[str, str], ...]
    ooi_id: str = "ooi"
    cell_size: float = 0.03
    shelf_pos: Tuple[float, float, float] = (0.25, 0.70, 0.0)
    deck_size: Tuple[float, float] = (0.60, 0.45)
    deck_levels: Tuple[float, ...] = (0.30, 0.55)
    hide_far_x: float = 100.0
    hand_capsule_radius_override: Optional[float] = 0.02
    # Placeholder-matched inertial (see tabletop_access asset_set
    # _DEFAULT_DIAGINERTIA + Asset.mass default).
    match_mass: float = 0.05
    match_diaginertia: Tuple[float, float, float] = (0.0250, 0.0250, 0.0125)
    match_identity_iquat: bool = True


def _materialise_ycb_mesh(ycb_source: str, body_id: str, scratch_dir: Path) -> Path:
    """Resolve the cached YCB MJCF and rewrite it so the SceneBuilder injects
    a named ``{body_id}_freejoint``.

    The YCB downloader's generated MJCF carries an **unnamed** ``<joint
    type="free">``; the builder only injects its named freejoint when the
    body has none, so that unnamed joint leaves ``attach_object_to_ee``
    unable to find ``{body}_freejoint``.  We strip the free joint (the
    builder then injects the named one) and pin an absolute ``meshdir`` so
    the mesh ``file=`` paths still resolve from the cache dir.
    """
    import xml.etree.ElementTree as ET
    from tampanda.scenes.assets.downloaders.ycb import YCBDownloader

    src = YCBDownloader().get(ycb_source)        # cached path (downloads if needed)
    cache_dir = Path(src).parent
    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(cache_dir.resolve()))
    for body in root.iter("body"):
        for child in list(body):
            if child.tag == "freejoint" or (
                child.tag == "joint" and child.get("type") == "free"
            ):
                body.remove(child)
    out = Path(scratch_dir) / f"{body_id}.xml"
    tree.write(out)
    return out


def make_tabletop_access_ycb_builder(
    scratch_dir: Path,
    roster: Optional[Sequence[Tuple[str, str]]] = None,
    cell_size: float = 0.03,
    shelf_pos: Tuple[float, float, float] = (0.25, 0.70, 0.0),
    deck_size: Tuple[float, float] = (0.60, 0.45),
    deck_levels: Tuple[float, ...] = (0.30, 0.55),
) -> Tuple[ArmSceneBuilder, Workspace, TabletopAccessYcbConfig]:
    """Build the dense-YCB 3-tier-shelf scene (middle + top regions).

    Args:
        scratch_dir: Directory where the shelf template is materialised;
            must outlive ``builder.build_env()``.
        roster:      Sequence of ``(body_id, ycb_source)`` pairs.  Defaults
            to :data:`DEFAULT_ROSTER`.  All ``ycb_source`` names must be
            available to the YCB downloader (cached or fetchable).
        cell_size:   Fine grid resolution.
        shelf_pos / deck_size / deck_levels:  Shelf geometry — defaults
            match the parent ``access`` shelf.

    Returns:
        ``(builder, workspace, config)``.
    """
    roster = tuple(roster) if roster is not None else DEFAULT_ROSTER
    object_ids = tuple(bid for bid, _ in roster)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"duplicate body id in roster: {object_ids}")

    cfg = TabletopAccessYcbConfig(
        object_ids=object_ids,
        roster=roster,
        cell_size=cell_size,
        shelf_pos=shelf_pos,
        deck_size=deck_size,
        deck_levels=deck_levels,
    )

    # ---- Shelf body: identical to the parent access shelf ----
    shelf = MultiTierShelf(
        asset_id="access_shelf",
        deck_size=deck_size,
        deck_thickness=0.012,
        deck_levels=deck_levels,
        leg_size=(0.018, 0.018),
        leg_inset=0.020,
        floor_separator=True,
        floor_separator_thickness=0.010,
        base_height=0.0,
        bottom_plate=False,
    )

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shelf_path = scratch_dir / f"{shelf.asset_id}.xml"
    shelf_path.write_text(shelf.render_template_xml())

    b = ArmSceneBuilder()
    b.add_resource(shelf.asset_id, str(shelf_path))
    b.add_object(shelf.asset_id, name="shelf", pos=list(shelf_pos))

    # ---- Movables: real YCB meshes, parked off-screen ----
    for body_id, ycb_source in roster:
        mesh_xml = _materialise_ycb_mesh(ycb_source, body_id, scratch_dir)
        b.add_resource(body_id, str(mesh_xml))
        b.add_object(body_id, name=body_id,
                     pos=[cfg.hide_far_x, 0.0, 0.10])

    # ---- Workspace: middle_deck + top_deck only ----
    sx, sy, _sz = shelf_pos
    dx, dy = deck_size
    leg_inset, leg_lx = 0.020, 0.018
    leg_inside_x = dx / 2 - leg_inset - leg_lx
    leg_inside_y = dy / 2 - leg_inset - leg_lx
    hand_clearance = 0.08
    upper_origin = (sx - leg_inside_x + hand_clearance,
                    sy - leg_inside_y + hand_clearance / 2)
    upper_extent = (2 * (leg_inside_x - hand_clearance),
                    2 * leg_inside_y - hand_clearance)

    # ``level_z`` here is the deck SURFACE z (not item-centre, since item
    # heights vary).  Placement code adds each object's half-height.
    middle_surface_z = shelf_pos[2] + deck_levels[0]
    top_surface_z = shelf_pos[2] + deck_levels[1]

    # Reachability sweep (examples/ta_ycb_validate.py --reachmap, FAST):
    # middle_deck pick 80/80 anchors, top_deck put 79/80 (only the extreme
    # back-right anchor fails).  The per-action feasibility check is the
    # gate (it rejects that corner), so excluded_cells stays empty —
    # matching the sister domains' feasibility-gated convention.
    workspace = Workspace([
        GridRegion(
            name="middle_deck",
            origin=upper_origin,
            extent=upper_extent,
            cell_size=cell_size,
            level_z=middle_surface_z,
            access_modes=("front", "top_down"),
            excluded_cells=frozenset(),
        ),
        GridRegion(
            name="top_deck",
            origin=upper_origin,
            extent=upper_extent,
            cell_size=cell_size,
            level_z=top_surface_z,
            access_modes=("top_down", "front"),
            excluded_cells=frozenset(),
        ),
    ])
    return b, workspace, cfg


def apply_runtime_tweaks(env, cfg: TabletopAccessYcbConfig) -> None:
    """Apply runtime model tweaks the static MJCF can't express.

    Two tweaks, both idempotent; call AFTER ``builder.build_env(...)`` and
    BEFORE the first interaction:

    1. **Inertial override** — every movable's mass is set to
       ``cfg.match_mass`` and its diagonal inertia to
       ``cfg.match_diaginertia`` (optionally with identity principal-axis
       quaternion).  The cached YCB MJCFs derive mass from mesh volume ×
       default density → 0.7–5.7 kg, which makes contacts violent and
       knocks neighbours over.  Matching the placeholder box inertia keeps
       contact dynamics gentle and the objects tip-resistant, exactly as
       the parent domain's boxes behaved.

    2. **hand_capsule shrink** — shrinks the 4 cm Franka wrist guard to
       ``cfg.hand_capsule_radius_override`` so it doesn't clip shelf walls
       during the joint-lerp descent (same as the parent domains).
    """
    import mujoco

    for body_id in cfg.object_ids:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if bid < 0:
            raise RuntimeError(f"movable body {body_id!r} not found on env.model")
        env.model.body_mass[bid] = float(cfg.match_mass)
        env.model.body_inertia[bid] = list(cfg.match_diaginertia)
        if cfg.match_identity_iquat:
            env.model.body_iquat[bid] = [1.0, 0.0, 0.0, 0.0]

    radius = cfg.hand_capsule_radius_override
    if radius is not None:
        for gid in range(env.model.ngeom):
            if env.model.geom(gid).name == "hand_capsule":
                env.model.geom_size[gid][0] = float(radius)
                break
        else:
            raise RuntimeError("hand_capsule geom not found on env.model")

    mujoco.mj_forward(env.model, env.data)
