"""Paths to bundled MuJoCo scene XML files."""

from pathlib import Path

_PANDA = Path(__file__).parent / "franka_emika_panda"

SCENE_DEFAULT  = _PANDA / "scene.xml"
SCENE_SYMBOLIC = _PANDA / "scene_symbolic.xml"
SCENE_BLOCKS   = _PANDA / "scene_blocks.xml"
SCENE_MAMO     = _PANDA / "scene_mamo.xml"
SCENE_TEST     = _PANDA / "scene_test.xml"
SCENE_MJX      = _PANDA / "mjx_scene.xml"

__all__ = [
    "SCENE_DEFAULT",
    "SCENE_SYMBOLIC",
    "SCENE_BLOCKS",
    "SCENE_MAMO",
    "SCENE_TEST",
    "SCENE_MJX",
]
