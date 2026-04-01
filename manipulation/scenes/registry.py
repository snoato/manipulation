"""Asset registry: resolves resource names to local file paths.

Designed for extension — the resolve() method dispatches on source type,
so URL / Menagerie / YCB sources can be added without touching the builder.
"""

from pathlib import Path
from typing import Union

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class AssetRegistry:
    """Resolves a resource source description to an absolute local path.

    Args:
        base_dir: Directory used to resolve relative path strings.
                  Defaults to the current working directory.
    """

    def __init__(self, base_dir: Path = None):
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()

    def resolve(self, source: Union[str, dict]) -> Path:
        """Return the absolute path to a template XML file.

        Source forms:
            "path/to/template.xml"          — path relative to base_dir
            {"type": "local", "path": "…"}  — explicit local path
            {"type": "builtin", "name": "…"}— relative to bundled templates/
            {"type": "url", "url": "…"}     — future, not yet implemented
            {"type": "menagerie", …}        — future, not yet implemented
        """
        if isinstance(source, (str, Path)):
            return self._resolve_path(str(source))

        src_type = source.get("type", "local")
        if src_type == "local":
            return self._resolve_path(source["path"])
        if src_type == "builtin":
            return (_TEMPLATES_DIR / source["name"]).resolve()
        if src_type == "url":
            raise NotImplementedError("URL sources are not yet supported")
        if src_type == "menagerie":
            raise NotImplementedError("Menagerie sources are not yet supported")
        raise ValueError(f"Unknown source type: {src_type!r}")

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        candidate = (self._base_dir / p).resolve()
        if candidate.exists():
            return candidate
        # Fall back to bundled templates directory
        builtin = (_TEMPLATES_DIR / p).resolve()
        if builtin.exists():
            return builtin
        # Return base_dir-relative path (will raise a clear error when read)
        return candidate
