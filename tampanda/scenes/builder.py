"""Programmatic MuJoCo scene builder.

Assembles a full scene XML from a base robot model, a set of named resources
(template XML fragments), and a list of object instances — each with an
explicit position and optional rotation.

Typical usage::

    builder = SceneBuilder.from_json("my_scene.json")
    env = builder.build_env()

Or programmatically::

    builder = SceneBuilder()
    builder.add_resource("cylinder", "scenes/templates/objects/cylinder.xml")
    builder.add_object("cylinder", pos=[0.4, 0.0, 0.42], rgba=[1, 0.2, 0.2, 1])
    builder.add_object("cylinder", pos=[0.5, 0.1, 0.42], euler=[0, 0, 45])
    env = builder.build_env()

Objects can be placed relative to another named object using ``relative_to``.
The offset ``pos`` is interpreted in the anchor's local frame (rotated by its
orientation) and the rotations are composed::

    builder.add_object("table", name="table", pos=[0.5, 0.0, 0.0])
    builder.add_object("cylinder", pos=[0.0, 0.0, 0.45], relative_to="table")
"""

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

from tampanda.scenes.registry import AssetRegistry

_ASSETS_DIR = (
    Path(__file__).parent.parent / "environments" / "assets" / "franka_emika_panda"
)

#: Path to the bundled Franka Panda base XML.  Pass this to
#: :class:`SceneBuilder` when building Panda arm scenes.
PANDA_BASE_XML = _ASSETS_DIR / "base_panda.xml"

#: Path to the bundled differential-drive mobile robot base XML.  Pass this
#: to :class:`SceneBuilder` when building navigation scenes.
DIFFBOT_BASE_XML = (
    Path(__file__).parent.parent / "environments" / "assets" / "diffbot" / "diffbot.xml"
)


# ------------------------------------------------------------------
# Vector helpers (pure math, no numpy dependency)
# ------------------------------------------------------------------

def _normalize(v: List[float]) -> List[float]:
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


def _cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _quat_mul(q1: List[float], q2: List[float]) -> List[float]:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def _quat_rotate(q: List[float], v: List[float]) -> List[float]:
    """Rotate vector v by quaternion q [w, x, y, z] using Rodrigues' formula."""
    u = q[1:]          # vector part
    t = _cross(u, v)   # u × v
    s = _cross(u, t)   # u × (u × v)
    return [v[i] + 2 * q[0] * t[i] + 2 * s[i] for i in range(3)]


def _look_at_xyaxes(cam_pos: List[float], target_pos: List[float]) -> str:
    """Return MuJoCo xyaxes string for a camera at cam_pos looking at target_pos."""
    # Camera -Z points toward scene; Z points away
    z = _normalize([cam_pos[i] - target_pos[i] for i in range(3)])
    up = [0.0, 0.0, 1.0]
    if abs(_dot(z, up)) > 0.999:
        up = [0.0, 1.0, 0.0]
    x = _normalize(_cross(up, z))
    y = _normalize(_cross(z, x))
    return " ".join(f"{v:.6f}" for v in x + y)


def _orbit_pos(
    target_pos: List[float], distance: float, elevation: float, azimuth: float
) -> List[float]:
    """Compute camera world position for an orbit-style spec (angles in degrees)."""
    el = math.radians(elevation)
    az = math.radians(azimuth)
    dx = distance * math.cos(el) * math.cos(az)
    dy = distance * math.cos(el) * math.sin(az)
    dz = distance * math.sin(el)
    return [target_pos[0] + dx, target_pos[1] + dy, target_pos[2] + dz]


@dataclass
class ObjectSpec:
    type_name: str
    name: str
    pos: List[float]                    # [x, y, z] — offset in anchor frame when relative_to is set
    quat: List[float]                   # [w, x, y, z] — rotation in anchor frame when relative_to is set
    rgba: Optional[List[float]] = None  # [r, g, b, a] — applied to named geoms
    relative_to: Optional[str] = None  # name of anchor object; None means world frame


@dataclass
class CameraSpec:
    name: str
    fovy: float = 60.0
    # Raw mode — direct MJCF attributes
    pos: Optional[List[float]] = None
    xyaxes: Optional[str] = None
    euler: Optional[List[float]] = None
    # Orbit mode — target is [x,y,z] coords or a body name string
    target: Optional[Union[List[float], str]] = None
    distance: Optional[float] = None
    elevation: float = 30.0   # degrees below horizontal (positive = camera above target)
    azimuth: float = 0.0      # degrees around Z axis


def _euler_xyz_to_quat(euler_deg: List[float]) -> List[float]:
    """Intrinsic XYZ Euler angles (degrees) → quaternion [w, x, y, z]."""
    rx, ry, rz = (math.radians(e) for e in euler_deg)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


class SceneBuilder:
    """Builds a MuJoCo MJCF scene from named resource templates.

    Prefer the concrete subclasses :class:`ArmSceneBuilder` (Franka Panda) or
    :class:`MobileSceneBuilder` (differential-drive robot).  Use the base class
    directly only when you need a custom base XML and will always pass
    ``env_class`` explicitly to :meth:`build_env`.

    Args:
        base: Path to the base robot XML.
    """

    def __init__(self, base: Union[str, Path]):
        self._base = Path(base)
        self._resources: dict[str, Union[str, dict]] = {}
        self._objects: List[ObjectSpec] = []
        self._cameras: List[CameraSpec] = []
        self._options: dict = {}
        self._option_flags: dict = {}     # emitted as <flag k="v"/> inside <option>
        self._custom_numerics: dict = {}  # emitted as <custom><numeric name=k data=v/></custom>
        self._type_counters: dict[str, int] = {}
        self._registry_base: Path = Path.cwd()

    # ------------------------------------------------------------------
    # Loading from JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SceneBuilder":
        """Load a scene descriptor from a JSON file.

        The JSON file may contain:
          - ``"base"``: path to the base robot XML (optional)
          - ``"resources"``: mapping of type name → source (path or dict)
          - ``"objects"``: list of instance descriptors
          - ``"options"``: dict of MuJoCo ``<option>`` attributes

        Resource paths and relative file paths in ``"base"`` are resolved
        relative to the JSON file's directory.

        Example JSON::

            {
              "resources": {
                "cylinder": "objects/cylinder.xml",
                "table":    "objects/table.xml"
              },
              "objects": [
                {"type": "table",    "pos": [0.4, 0.0, 0.0]},
                {"type": "cylinder", "pos": [0.4, 0.0, 0.42], "rgba": [1, 0.2, 0.2, 1]},
                {"type": "cylinder", "pos": [0.5, 0.1, 0.42], "euler": [0, 0, 45]},
                {"type": "cylinder", "name": "special", "pos": [0.3, -0.1, 0.42]}
              ],
              "options": {"timestep": 0.005}
            }
        """
        path = Path(path).resolve()
        with open(path) as f:
            desc = json.load(f)

        raw_base = desc.get("base")
        if raw_base is not None:
            base = (path.parent / raw_base).resolve()
        else:
            base = PANDA_BASE_XML  # backward-compatible default for JSON files
        builder = cls(base=base)
        builder._registry_base = path.parent

        for name, source in desc.get("resources", {}).items():
            builder.add_resource(name, source)

        for obj in desc.get("objects", []):
            obj = dict(obj)
            type_name = obj.pop("type")
            pos = obj.pop("pos")
            name = obj.pop("name", None)
            euler = obj.pop("euler", None)
            quat = obj.pop("quat", None)
            rgba = obj.pop("rgba", None)
            relative_to = obj.pop("relative_to", None)
            builder.add_object(type_name, pos, name=name, euler=euler, quat=quat, rgba=rgba,
                               relative_to=relative_to)

        for cam in desc.get("cameras", []):
            cam = dict(cam)
            name = cam.pop("name")
            target = cam.pop("target", None)
            if target is not None:
                builder.add_camera_orbit(
                    name,
                    target=target,
                    distance=cam.pop("distance"),
                    elevation=cam.pop("elevation", 30.0),
                    azimuth=cam.pop("azimuth", 0.0),
                    fovy=cam.pop("fovy", 60.0),
                )
            else:
                builder.add_camera(
                    name,
                    pos=cam.pop("pos"),
                    fovy=cam.pop("fovy", 60.0),
                    xyaxes=cam.pop("xyaxes", None),
                    euler=cam.pop("euler", None),
                )

        builder._options = desc.get("options", {})
        return builder

    # ------------------------------------------------------------------
    # Building up the scene
    # ------------------------------------------------------------------

    def add_resource(self, name: str, source: Union[str, dict]):
        """Register a resource type.

        Args:
            name:   The type identifier used in add_object() and JSON ``"type"`` fields.
            source: A file path string (resolved relative to registry_base) or a
                    source dict with a ``"type"`` key.  Supported types today:
                    ``"local"`` (explicit path) and ``"builtin"`` (bundled templates).
                    ``"url"`` and ``"menagerie"`` are reserved for future use.
        """
        self._resources[name] = source

    def add_object(
        self,
        type_name: str,
        pos: List[float],
        *,
        name: Optional[str] = None,
        euler: Optional[List[float]] = None,
        quat: Optional[List[float]] = None,
        rgba: Optional[List[float]] = None,
        relative_to: Optional[str] = None,
    ):
        """Add one object instance to the scene.

        Args:
            type_name:   Resource key registered via add_resource().
            pos:         Position [x, y, z].  World frame by default; local frame
                         of *relative_to* when that is specified.
            name:        Body name.  Auto-generated as ``{type_name}_{n}`` if omitted.
            euler:       XYZ Euler angles in degrees [rx, ry, rz].  Mutually exclusive
                         with ``quat``.
            quat:        Quaternion [w, x, y, z].  Identity if neither is given.
            rgba:        RGBA colour override [r, g, b, a] applied to primary (named)
                         geoms in the template.
            relative_to: Name of a previously-added object whose pose is used as the
                         reference frame.  The offset ``pos`` is rotated by the anchor's
                         orientation and added to its world position; the rotations are
                         composed.  The anchor must be added before build_xml() is called
                         but does not need to precede this call.
        """
        if type_name not in self._resources:
            raise ValueError(
                f"Unknown resource {type_name!r}. "
                f"Register it with add_resource() first."
            )
        if euler is not None and quat is not None:
            raise ValueError("Specify either euler or quat, not both.")

        if name is None:
            n = self._type_counters.get(type_name, 0)
            name = f"{type_name}_{n}"
            self._type_counters[type_name] = n + 1

        if euler is not None:
            quat = _euler_xyz_to_quat(euler)
        elif quat is None:
            quat = [1.0, 0.0, 0.0, 0.0]

        self._objects.append(
            ObjectSpec(
                type_name=type_name,
                name=name,
                pos=list(pos),
                quat=list(quat),
                rgba=list(rgba) if rgba is not None else None,
                relative_to=relative_to,
            )
        )

    def add_camera(
        self,
        name: str,
        *,
        pos: List[float],
        fovy: float = 60.0,
        xyaxes: Optional[str] = None,
        euler: Optional[List[float]] = None,
    ):
        """Add a camera with explicit MJCF position and orientation.

        Args:
            name:    Camera name (used in MujocoCamera calls).
            pos:     World position [x, y, z].
            fovy:    Vertical field of view in degrees.
            xyaxes:  MuJoCo xyaxes string "x0 x1 x2 y0 y1 y2" (mutually exclusive with euler).
            euler:   Euler angles in degrees [rx, ry, rz] (mutually exclusive with xyaxes).
        """
        if xyaxes is not None and euler is not None:
            raise ValueError("Specify either xyaxes or euler, not both.")
        self._cameras.append(CameraSpec(name=name, pos=pos, fovy=fovy, xyaxes=xyaxes, euler=euler))

    def add_camera_orbit(
        self,
        name: str,
        *,
        target: Union[List[float], str],
        distance: float,
        elevation: float = 30.0,
        azimuth: float = 0.0,
        fovy: float = 60.0,
    ):
        """Add a camera positioned on an orbit around a target point or body.

        The camera position is computed from spherical coordinates relative to
        the target.  When ``target`` is a body name, MuJoCo's native ``target``
        attribute is used so the camera dynamically tracks the body.

        Args:
            name:      Camera name.
            target:    [x, y, z] world coordinates, or the name of a scene body.
            distance:  Distance from target in metres.
            elevation: Degrees below horizontal (positive = camera above target,
                       looking down).  E.g. 30 gives a 30° downward view.
            azimuth:   Degrees around the world Z axis (0 = along +X).
            fovy:      Vertical field of view in degrees.
        """
        self._cameras.append(CameraSpec(
            name=name, fovy=fovy,
            target=target, distance=distance,
            elevation=elevation, azimuth=azimuth,
        ))

    def _resolve_world_poses(self) -> "dict[str, tuple[List[float], List[float]]]":
        """Resolve each object's world-frame (pos, quat), following relative_to chains.

        Raises ValueError on unknown anchors or cycles.
        """
        by_name = {obj.name: obj for obj in self._objects}
        resolved: dict[str, tuple[List[float], List[float]]] = {}
        in_progress: set[str] = set()

        def resolve(name: str) -> tuple[List[float], List[float]]:
            if name in resolved:
                return resolved[name]
            if name in in_progress:
                raise ValueError(f"Cycle detected in relative_to chain at {name!r}")
            obj = by_name.get(name)
            if obj is None:
                raise ValueError(
                    f"relative_to references unknown object {name!r}. "
                    f"Add it with add_object() before calling build_xml()."
                )
            if obj.relative_to is None:
                resolved[name] = (obj.pos, obj.quat)
                return resolved[name]
            in_progress.add(name)
            anchor_pos, anchor_quat = resolve(obj.relative_to)
            world_pos = [anchor_pos[i] + _quat_rotate(anchor_quat, obj.pos)[i] for i in range(3)]
            world_quat = _quat_mul(anchor_quat, obj.quat)
            in_progress.discard(name)
            resolved[name] = (world_pos, world_quat)
            return resolved[name]

        for obj in self._objects:
            resolve(obj.name)
        return resolved

    def _get_object_pos(self, name: str, world_poses: "dict | None" = None) -> List[float]:
        """Return the world position of an object by body name."""
        if world_poses is not None and name in world_poses:
            return world_poses[name][0]
        for obj in self._objects:
            if obj.name == name:
                return obj.pos
        raise ValueError(
            f"No object named {name!r} found in scene. "
            f"Add it with add_object() before referencing it in a camera."
        )

    def _camera_to_xml_attrs(self, spec: CameraSpec, world_poses: "dict | None" = None) -> dict:
        """Convert a CameraSpec to a dict of MJCF <camera> attributes."""
        attrs: dict = {"name": spec.name, "fovy": str(spec.fovy)}

        if spec.target is not None:
            # Orbit mode
            if isinstance(spec.target, str):
                target_pos = self._get_object_pos(spec.target, world_poses)
                cam_pos = _orbit_pos(target_pos, spec.distance, spec.elevation, spec.azimuth)
                attrs["pos"] = " ".join(f"{v:.6g}" for v in cam_pos)
                attrs["target"] = spec.target  # MuJoCo tracks the body dynamically
            else:
                target_pos = spec.target
                cam_pos = _orbit_pos(target_pos, spec.distance, spec.elevation, spec.azimuth)
                attrs["pos"] = " ".join(f"{v:.6g}" for v in cam_pos)
                attrs["xyaxes"] = _look_at_xyaxes(cam_pos, target_pos)
        else:
            # Raw mode
            if spec.pos is not None:
                attrs["pos"] = " ".join(f"{v:.6g}" for v in spec.pos)
            if spec.xyaxes is not None:
                attrs["xyaxes"] = spec.xyaxes
            if spec.euler is not None:
                attrs["euler"] = " ".join(f"{v:.6g}" for v in spec.euler)

        return attrs

    # ------------------------------------------------------------------
    # XML generation
    # ------------------------------------------------------------------

    def _load_template(self, source: Union[str, dict]) -> Tuple[ET.Element, List[ET.Element]]:
        """Parse a template file.

        Returns ``(body_element, extra_assets)`` where *extra_assets* is a
        (possibly empty) list of ``<asset>`` child elements to be merged into
        the scene's ``<asset>`` block.

        Supports two formats:

        * **Body fragment** — the existing format: a bare ``<body>`` element
          (optionally with ``_``-prefixed child names for auto-renaming).
          Returns an empty *extra_assets* list.

        * **Full MJCF** — a complete ``<mujoco>`` document as produced by
          downloaded YCB / GSO assets.  Asset names and cross-references are
          namespaced by *instance name*; relative ``file=`` paths are rewritten
          to absolute paths so the scene XML can be written anywhere.
        """
        registry = AssetRegistry(self._registry_base)
        path = registry.resolve(source)
        text = path.read_text(encoding="utf-8")
        stripped = text.strip()

        if stripped.startswith("<mujoco") or stripped.startswith("<?xml"):
            root = ET.fromstring(text)
            return root, path  # deferred — processed in _instantiate_object
        else:
            wrapper = ET.fromstring(f"<_root>{text}</_root>")
            bodies = [c for c in wrapper if c.tag == "body"]
            if len(bodies) != 1:
                raise ValueError(
                    f"Template {path} must contain exactly one top-level <body> element, "
                    f"found {len(bodies)}."
                )
            return bodies[0], None  # None signals "fragment" mode

    # ------------------------------------------------------------------
    # Full-MJCF merging helpers
    # ------------------------------------------------------------------

    def _extract_from_full_mjcf(
        self, root: ET.Element, xml_dir: Path, prefix: str
    ) -> Tuple[ET.Element, List[ET.Element]]:
        """Extract and namespace a body + assets from a full MJCF document.

        * All asset ``name=`` attributes are prefixed with ``{prefix}_``.
        * Relative ``file=`` paths are rewritten to absolute using the XML's
          directory (or ``<compiler meshdir>`` when present).
        * Cross-references (``material.texture``, ``geom.mesh``, etc.) are
          updated to match renamed assets.
        * A ``<freejoint>`` is injected if the body has no free joint.

        Returns ``(body_element, list_of_asset_elements)``.
        """
        # Determine effective mesh directory
        meshdir = xml_dir
        compiler = root.find("compiler")
        if compiler is not None and "meshdir" in compiler.attrib:
            md = Path(compiler.attrib["meshdir"])
            meshdir = md if md.is_absolute() else (xml_dir / md).resolve()

        # --- Pass 1: build name_map from all named assets ---
        name_map: dict[str, str] = {}
        asset_elem = root.find("asset")
        if asset_elem is not None:
            for child in asset_elem:
                old = child.get("name", "")
                if old:
                    name_map[old] = f"{prefix}_{old}"

        # --- Pass 2: deep-copy + transform asset elements ---
        transformed_assets: List[ET.Element] = []
        if asset_elem is not None:
            for child in asset_elem:
                el = copy.deepcopy(child)
                # Rewrite relative file paths to absolute
                if "file" in el.attrib:
                    fp = Path(el.attrib["file"])
                    if not fp.is_absolute():
                        el.attrib["file"] = str((meshdir / fp).resolve())
                # Rename self
                old = el.get("name", "")
                if old:
                    el.set("name", f"{prefix}_{old}")
                # Rewrite internal cross-references (e.g. material.texture)
                for ref_attr in ("texture", "mesh", "material"):
                    ref = el.get(ref_attr, "")
                    if ref in name_map:
                        el.set(ref_attr, name_map[ref])
                transformed_assets.append(el)

        # --- Extract body from worldbody ---
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Full MJCF document has no <worldbody>")
        body_elems = [e for e in worldbody if e.tag == "body"]
        if not body_elems:
            raise ValueError("Full MJCF <worldbody> contains no <body>")
        body = copy.deepcopy(body_elems[0])

        # --- Rewrite body: rename elements, update asset refs ---
        for elem in body.iter():
            # Update asset cross-references
            for ref_attr in ("mesh", "material", "texture"):
                ref = elem.get(ref_attr, "")
                if ref in name_map:
                    elem.set(ref_attr, name_map[ref])
            # Prefix element names
            old = elem.get("name", "")
            if old:
                elem.set("name", f"{prefix}_{old}")

        # --- Inject freejoint if absent ---
        has_free = any(
            (c.tag == "joint" and c.get("type") == "free") or c.tag == "freejoint"
            for c in body
        )
        if not has_free:
            fj = ET.Element("freejoint")
            fj.set("name", f"{prefix}_freejoint")
            body.insert(0, fj)

        return body, transformed_assets

    # ------------------------------------------------------------------
    # Object instantiation
    # ------------------------------------------------------------------

    def _instantiate_object(self, spec: ObjectSpec) -> Tuple[ET.Element, List[ET.Element]]:
        """Return ``(body_element, extra_asset_elements)`` for the given spec."""
        raw, hint = self._load_template(self._resources[spec.type_name])

        # ---- Full MJCF mode (downloaded YCB / GSO objects) ----
        if hint is not None:
            # hint == resolved Path when full MJCF
            xml_path: Path = hint
            body, extra_assets = self._extract_from_full_mjcf(raw, xml_path.parent, spec.name)
            body.set("name", spec.name)
            body.set("pos", " ".join(f"{v:.6g}" for v in spec.pos))
            body.set("quat", " ".join(f"{v:.6g}" for v in spec.quat))
            if spec.rgba is not None:
                for elem in body.iter():
                    if elem.tag == "geom":
                        elem.set("rgba", " ".join(f"{v:.4g}" for v in spec.rgba))
            return body, extra_assets

        # ---- Fragment mode (existing behaviour) ----
        body = raw
        body.set("name", spec.name)
        body.set("pos", " ".join(f"{v:.6g}" for v in spec.pos))
        body.set("quat", " ".join(f"{v:.6g}" for v in spec.quat))

        # Rename all descendants whose name starts with "_":
        #   "_freejoint" → "{spec.name}_freejoint"
        #   "_geom"      → "{spec.name}_geom"
        #   "_surface"   → "{spec.name}_surface"   etc.
        for elem in body.iter():
            if elem is body:
                continue
            old_name = elem.get("name", "")
            if old_name.startswith("_"):
                elem.set("name", f"{spec.name}{old_name}")
                if elem.tag == "geom" and spec.rgba is not None:
                    elem.set("rgba", " ".join(f"{v:.4g}" for v in spec.rgba))

        return body, []

    def build_xml(self) -> str:
        """Generate the full scene MJCF XML as a string.

        The XML uses ``<include>`` for the base robot model, so it must be
        written to the same directory as the base XML before loading
        (handled automatically by build_env()).
        """
        root = ET.Element("mujoco")
        root.set("model", "generated scene")

        # Robot base via include (resolved relative to output file location)
        ET.SubElement(root, "include", file=self._base.name)

        ET.SubElement(root, "statistic", center="0.3 0 0.4", extent="1")

        if self._options or self._option_flags:
            opt = ET.SubElement(root, "option", **{k: str(v) for k, v in self._options.items()})
            for k, v in self._option_flags.items():
                ET.SubElement(opt, "flag", **{k: str(v)})

        if self._custom_numerics:
            custom = ET.SubElement(root, "custom")
            for name, data in self._custom_numerics.items():
                ET.SubElement(custom, "numeric", name=name, data=str(data))

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight",
                      diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
        ET.SubElement(visual, "global", azimuth="120", elevation="-20")

        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                      rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        ET.SubElement(asset, "texture", type="2d", name="groundplane",
                      builtin="checker", mark="edge",
                      rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                      markrgb="0.8 0.8 0.8", width="300", height="300")
        ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                      texuniform="true", texrepeat="5 5", reflectance="0.2")

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light",
                      pos="0 0 1.5", dir="0 0 -1", directional="true")
        ET.SubElement(worldbody, "geom", name="floor",
                      size="0 0 0.05", type="plane", material="groundplane", contype="1")
        self._add_scene_extras(worldbody)

        world_poses = self._resolve_world_poses()

        if self._cameras:
            for cam_spec in self._cameras:
                ET.SubElement(worldbody, "camera", **self._camera_to_xml_attrs(cam_spec, world_poses))
        else:
            # Default cameras when none are configured
            ET.SubElement(worldbody, "camera", name="top_camera",
                          pos="0.4 0.4 1.5", xyaxes="1 0 0 0 1 0", fovy="60")
            ET.SubElement(worldbody, "camera", name="side_camera",
                          pos="1.6 0.55 0.8", euler="0 1.07 1.57", fovy="55")
            ET.SubElement(worldbody, "camera", name="front_camera",
                          pos="0.4 -0.4 0.7", xyaxes="1 0 0 0 0.4 0.8", fovy="60")
        for spec in self._objects:
            world_pos, world_quat = world_poses[spec.name]
            resolved_spec = ObjectSpec(
                type_name=spec.type_name,
                name=spec.name,
                pos=world_pos,
                quat=world_quat,
                rgba=spec.rgba,
            )
            body, extra_assets = self._instantiate_object(resolved_spec)
            worldbody.append(body)
            for elem in extra_assets:
                asset.append(elem)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _add_scene_extras(self, worldbody: ET.Element) -> None:
        """Inject robot-type-specific elements into worldbody.

        Called during :meth:`build_xml` after the floor geom is added.
        Base implementation is a no-op; subclasses override as needed.
        """

    # ------------------------------------------------------------------
    # Loading into MuJoCo
    # ------------------------------------------------------------------

    def _default_env_class(self):
        """Return the default environment class for this builder type.

        Subclasses override to supply their default.  The base class raises
        so callers know they must pass ``env_class`` explicitly.
        """
        raise TypeError(
            "Cannot infer env_class from the base SceneBuilder. "
            "Use ArmSceneBuilder or MobileSceneBuilder, or pass env_class= explicitly."
        )

    def build_env(self, env_class=None, rate: float = 200.0, **kwargs):
        """Build the scene XML and return a ready environment instance.

        The XML is written to a temporary file next to the base robot XML so
        that ``<include>`` directives and mesh relative paths resolve correctly.
        The file is deleted immediately after loading.

        Args:
            env_class:  Environment class to instantiate.  Defaults to the
                        class provided by the concrete subclass
                        (:class:`FrankaEnvironment` for :class:`ArmSceneBuilder`,
                        :class:`MobileEnvironment` for :class:`MobileSceneBuilder`).
            rate:       Simulation rate in Hz.
            **kwargs:   Extra keyword arguments forwarded to the environment
                        constructor.
        """
        if env_class is None:
            env_class = self._default_env_class()

        xml = self.build_xml()
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=self._base.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(xml)
            env = env_class(tmp_path, rate=rate, **kwargs)
        finally:
            os.unlink(tmp_path)
        return env


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class ArmSceneBuilder(SceneBuilder):
    """SceneBuilder for Franka Panda arm scenes.

    Adds a mocap ``target`` body used by :class:`MinkIK` as the end-effector
    visualisation target.  Defaults to :class:`FrankaEnvironment`.

    Args:
        base: Path to the base arm XML.  Defaults to the bundled Panda model.
    """

    def __init__(self, base: Union[str, Path] = PANDA_BASE_XML):
        super().__init__(base)

    def _add_scene_extras(self, worldbody: ET.Element) -> None:
        target = ET.SubElement(worldbody, "body", name="target",
                               pos="0.5 0 0.5", quat="0 1 0 0", mocap="true")
        ET.SubElement(target, "geom", type="box", size=".05 .05 .05",
                      contype="0", conaffinity="0", rgba=".6 .3 .3 .2")

    def _default_env_class(self):
        from tampanda.environments.franka_env import FrankaEnvironment
        return FrankaEnvironment


class MobileSceneBuilder(SceneBuilder):
    """SceneBuilder for differential-drive mobile robot scenes.

    Defaults to :class:`MobileEnvironment`.

    Args:
        base: Path to the base mobile robot XML.  Defaults to the bundled
              diffbot model.
    """

    def __init__(self, base: Union[str, Path] = DIFFBOT_BASE_XML):
        super().__init__(base)

    def _default_env_class(self):
        from tampanda.environments.mobile_env import MobileEnvironment
        return MobileEnvironment
