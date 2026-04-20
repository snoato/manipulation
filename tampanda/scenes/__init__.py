"""Programmatic scene construction for MuJoCo environments."""

from pathlib import Path

from tampanda.scenes.builder import (
    SceneBuilder,
    ArmSceneBuilder,
    MobileSceneBuilder,
    PANDA_BASE_XML,
    DIFFBOT_BASE_XML,
)
from tampanda.scenes.registry import AssetRegistry
from tampanda.scenes.reloader import SceneReloader
from tampanda.scenes.assets import AssetCache, YCBDownloader, GSODownloader

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
WALL_TEMPLATE                 = TEMPLATES_DIR / "objects" / "wall.xml"
PILLAR_TEMPLATE               = TEMPLATES_DIR / "objects" / "pillar.xml"
BALL_TEMPLATE                 = TEMPLATES_DIR / "objects" / "ball.xml"
BIN_TEMPLATE                  = TEMPLATES_DIR / "objects" / "bin.xml"

__all__ = [
    "SceneBuilder",
    "ArmSceneBuilder",
    "MobileSceneBuilder",
    "PANDA_BASE_XML",
    "DIFFBOT_BASE_XML",
    "AssetRegistry",
    "SceneReloader",
    "AssetCache",
    "YCBDownloader",
    "GSODownloader",
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
    "WALL_TEMPLATE",
    "PILLAR_TEMPLATE",
    "BALL_TEMPLATE",
    "BIN_TEMPLATE",
]
