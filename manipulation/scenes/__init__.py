"""Programmatic scene construction for MuJoCo environments."""

from pathlib import Path

from manipulation.scenes.builder import SceneBuilder
from manipulation.scenes.registry import AssetRegistry
from manipulation.scenes.reloader import SceneReloader

TEMPLATES_DIR = Path(__file__).parent / "templates"
CYLINDER_TEMPLATE        = TEMPLATES_DIR / "objects" / "cylinder.xml"
CYLINDER_THIN_TEMPLATE   = TEMPLATES_DIR / "objects" / "cylinder_thin.xml"
CYLINDER_MEDIUM_TEMPLATE = TEMPLATES_DIR / "objects" / "cylinder_medium.xml"
CYLINDER_THICK_TEMPLATE  = TEMPLATES_DIR / "objects" / "cylinder_thick.xml"
TABLE_TEMPLATE              = TEMPLATES_DIR / "objects" / "table.xml"
TABLE_SYMBOLIC_TEMPLATE     = TEMPLATES_DIR / "objects" / "table_symbolic.xml"
BLOCK_SMALL_TEMPLATE        = TEMPLATES_DIR / "objects" / "block_small.xml"
BLOCK_MEDIUM_TEMPLATE       = TEMPLATES_DIR / "objects" / "block_medium.xml"
BLOCK_PLATFORM_TEMPLATE     = TEMPLATES_DIR / "objects" / "block_platform.xml"
BLOCK_LARGE_PLATFORM_TEMPLATE = TEMPLATES_DIR / "objects" / "block_large_platform.xml"

__all__ = [
    "SceneBuilder",
    "AssetRegistry",
    "SceneReloader",
    "TEMPLATES_DIR",
    "CYLINDER_TEMPLATE",
    "CYLINDER_THIN_TEMPLATE",
    "CYLINDER_MEDIUM_TEMPLATE",
    "CYLINDER_THICK_TEMPLATE",
    "TABLE_TEMPLATE",
    "TABLE_SYMBOLIC_TEMPLATE",
    "BLOCK_SMALL_TEMPLATE",
    "BLOCK_MEDIUM_TEMPLATE",
    "BLOCK_PLATFORM_TEMPLATE",
    "BLOCK_LARGE_PLATFORM_TEMPLATE",
]
