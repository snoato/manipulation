# Scene Builder

`SceneBuilder` assembles a MuJoCo MJCF scene from a base robot XML, a set of
named *resource* templates, and a list of object instances.  It handles asset
namespacing, coordinate transforms, and temporary-file management so that
`.build_env()` returns a ready simulation in one call.

## Quick start

```python
from tampanda.scenes import ArmSceneBuilder

builder = ArmSceneBuilder()
builder.add_resource("table",    {"type": "builtin", "name": "table"})
builder.add_resource("cylinder", {"type": "builtin", "name": "cylinder"})

builder.add_object("table",    name="table",  pos=[0.5, 0.0, 0.0])
builder.add_object("cylinder", name="obj_0",  pos=[0.0, 0.0, 0.45], relative_to="table")
builder.add_object("cylinder", name="obj_1",  pos=[0.1, 0.0, 0.45], relative_to="table",
                   rgba=[0.2, 0.8, 0.2, 1])

env = builder.build_env()
```

Or load from a JSON scene file:

```python
env = ArmSceneBuilder.from_json("my_scene.json").build_env()
```

## JSON format

```json
{
  "base": "path/to/robot.xml",
  "resources": {
    "table":    {"type": "builtin", "name": "table"},
    "cylinder": {"type": "builtin", "name": "cylinder"},
    "mug":      "local/assets/mug.xml"
  },
  "objects": [
    {"type": "table",    "name": "table",  "pos": [0.5, 0.0, 0.0]},
    {"type": "cylinder", "name": "obj_0",  "pos": [0.0, 0.0, 0.45], "relative_to": "table"},
    {"type": "cylinder", "name": "obj_1",  "pos": [0.1, 0.0, 0.45], "relative_to": "table",
     "euler": [0, 0, 45], "rgba": [0.2, 0.8, 0.2, 1]},
    {"type": "mug",      "pos": [0.5, 0.2, 0.45]}
  ],
  "cameras": [
    {"name": "top",  "target": [0.5, 0.0, 0.3], "distance": 1.2, "elevation": 60},
    {"name": "side", "target": "obj_0",          "distance": 0.8, "elevation": 20, "azimuth": 90}
  ],
  "options": {"timestep": 0.005}
}
```

`"base"` is optional — defaults to the bundled Franka Panda model.  All paths
are resolved relative to the JSON file's directory.

### Object fields

| Field          | Type              | Default    | Description |
|----------------|-------------------|------------|-------------|
| `type`         | string            | required   | Resource key from `"resources"` |
| `pos`          | `[x, y, z]`       | required   | Position (world frame, or local frame of `relative_to`) |
| `name`         | string            | auto       | Body name; auto-generated as `{type}_{n}` if omitted |
| `euler`        | `[rx, ry, rz]`    | identity   | XYZ Euler angles in degrees (mutually exclusive with `quat`) |
| `quat`         | `[w, x, y, z]`    | identity   | Quaternion (mutually exclusive with `euler`) |
| `rgba`         | `[r, g, b, a]`    | template   | Colour override applied to primary geoms |
| `relative_to`  | string            | `null`     | Name of anchor object (see below) |

## Relative positioning

When `relative_to` is set, `pos` and the rotation are interpreted in the
**anchor object's local frame**:

- The offset `pos` is rotated by the anchor's orientation, then added to its
  world position.
- The rotations are composed (`anchor_quat × local_quat`).

This makes it easy to place objects on a surface without knowing its absolute
world coordinates:

```python
builder.add_object("table",    name="table",  pos=[0.5, 0.0, 0.0], euler=[0, 0, 30])
builder.add_object("cylinder", pos=[0.0, 0.0, 0.45], relative_to="table")
# The cylinder lands on the table surface regardless of the table's rotation.
```

Chains are supported (A relative to B relative to C).  Cycles raise a
`ValueError` at `build_xml()` time.  The anchor does not need to be added
before the dependent object — resolution is deferred to `build_xml()`.

## Resource types

Resources are registered with `add_resource(name, source)`.  The `source` can be:

| Form | Example | Description |
|------|---------|-------------|
| Plain path string | `"objects/mug.xml"` | Path relative to the JSON file's directory (or `registry_base`) |
| `{"type": "builtin", "name": "…"}` | `{"type": "builtin", "name": "cylinder"}` | Bundled template in `scenes/templates/objects/` |
| `{"type": "local", "path": "…"}` | `{"type": "local", "path": "/abs/path/obj.xml"}` | Explicit local path |

## Adding new assets

### Option 1 — Builtin template (body fragment)

Create an XML file in `tampanda/scenes/templates/objects/` containing exactly
one top-level `<body>` element.  Name all child elements with a leading `_` so
the builder can auto-rename them per instance:

```xml
<body pos="0 0 0">
  <joint name="_freejoint" type="free"/>
  <geom name="_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.4 0.1 1"/>
</body>
```

Then register it as a builtin:

```python
builder.add_resource("mybox", {"type": "builtin", "name": "mybox"})
```

The body name, freejoint name, and all `_`-prefixed names are automatically
rewritten to `{instance_name}_freejoint`, `{instance_name}_geom`, etc.

### Option 2 — Full MJCF (downloaded assets)

Full MJCF documents (e.g. from the YCB or Google Scanned Objects datasets) are
supported directly.  The builder:

- Prefixes all asset names with the instance name to avoid collisions.
- Rewrites relative `file=` paths to absolute paths.
- Injects a `<freejoint>` if the top body lacks one.

Use the `AssetCache` / `YCBDownloader` / `GSODownloader` helpers in
`scenes/assets/` to download and cache these objects, then point a resource at
the resulting XML path.

### Option 3 — Local XML

Any `<body>` fragment or full MJCF document can be registered by file path:

```python
builder.add_resource("mug", "/path/to/mug.xml")
# or in JSON:
# "mug": {"type": "local", "path": "/path/to/mug.xml"}
```

## Cameras

Two placement modes are available:

**Raw** — explicit MJCF attributes:

```python
builder.add_camera("front", pos=[0.4, -0.4, 0.7], euler=[0, -20, 0])
```

**Orbit** — spherical coordinates around a target point or body:

```python
builder.add_camera_orbit("top", target=[0.5, 0.0, 0.3], distance=1.2, elevation=60)
builder.add_camera_orbit("side", target="obj_0", distance=0.8, elevation=20, azimuth=90)
```

When `target` is a body name, MuJoCo's native `target` attribute is used so
the camera tracks the body dynamically at runtime.  World-pose resolution runs
before camera XML is emitted, so orbit cameras targeting `relative_to` objects
always get the correct world position.
