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
_DEFAULT_BASE = _ASSETS_DIR / "base_panda.xml"


@dataclass
class ObjectSpec:
    type_name: str
    name: str
    pos: List[float]          # [x, y, z]
    quat: List[float]         # [w, x, y, z]
    rgba: Optional[List[float]] = None  # [r, g, b, a] — applied to named geoms


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

    Args:
        base: Path to the base robot XML (default: bundled base_panda.xml).
    """

    def __init__(self, base: Union[str, Path, None] = None):
        self._base = Path(base) if base else _DEFAULT_BASE
        self._resources: dict[str, Union[str, dict]] = {}
        self._objects: List[ObjectSpec] = []
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

        base = desc.get("base")
        if base is not None:
            base = (path.parent / base).resolve()
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
            builder.add_object(type_name, pos, name=name, euler=euler, quat=quat, rgba=rgba)

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
    ):
        """Add one object instance to the scene.

        Args:
            type_name: Resource key registered via add_resource().
            pos:       World position [x, y, z].
            name:      Body name.  Auto-generated as ``{type_name}_{n}`` if omitted.
            euler:     XYZ Euler angles in degrees [rx, ry, rz].  Mutually exclusive
                       with ``quat``.
            quat:      Quaternion [w, x, y, z].  Identity if neither is given.
            rgba:      RGBA colour override [r, g, b, a] applied to primary (named)
                       geoms in the template.
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
            )
        )

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
        text = path.read_text()
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
        # Mocap body used by MinkIK as the end-effector target
        target = ET.SubElement(worldbody, "body", name="target",
                               pos="0.5 0 0.5", quat="0 1 0 0", mocap="true")
        ET.SubElement(target, "geom", type="box", size=".05 .05 .05",
                      contype="0", conaffinity="0", rgba=".6 .3 .3 .2")
        ET.SubElement(worldbody, "camera", name="top_camera",
                      pos="0.4 0.4 1.5", xyaxes="1 0 0 0 1 0", fovy="60")
        ET.SubElement(worldbody, "camera", name="side_camera",
                      pos="1.6 0.55 0.8", euler="0 1.07 1.57", fovy="55")
        ET.SubElement(worldbody, "camera", name="front_camera",
                      pos="0.4 -0.4 0.7", xyaxes="1 0 0 0 0.4 0.8", fovy="60")

        for spec in self._objects:
            body, extra_assets = self._instantiate_object(spec)
            worldbody.append(body)
            for elem in extra_assets:
                asset.append(elem)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Loading into MuJoCo
    # ------------------------------------------------------------------

    def build_env(self, rate: float = 200.0, collision_bodies=None):
        """Build the scene XML and return a ready FrankaEnvironment.

        The XML is written to a temporary file next to base_panda.xml so that
        the ``<include>`` directive and mesh relative paths resolve correctly.
        The file is deleted immediately after loading.

        Args:
            rate:             Simulation rate in Hz.
            collision_bodies: Robot body names used for collision checking.
                              Defaults to the full Panda arm + hand.
        """
        from tampanda.environments.franka_env import FrankaEnvironment

        xml = self.build_xml()
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=self._base.parent)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(xml)
            env = FrankaEnvironment(tmp_path, rate=rate,
                                    collision_bodies=collision_bodies)
        finally:
            os.unlink(tmp_path)
        return env
