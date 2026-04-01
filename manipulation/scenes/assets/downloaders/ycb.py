"""Downloader for elpis-lab/ycb_dataset — YCB objects in MuJoCo MJCF format."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from manipulation.scenes.assets.downloaders.base import BaseDownloader, _download_file

# Repo layout: elpis-lab/ycb_dataset/ycb/<object_name>/
_OBJECTS_SUBDIR = "ycb"


class YCBDownloader(BaseDownloader):
    """Downloads YCB objects from ``elpis-lab/ycb_dataset`` on GitHub.

    Objects live under the ``ycb/`` subdirectory of the repo (e.g.
    ``ycb/002_master_chef_can/``).  The downloader fetches the full directory
    tree for the requested object into the local cache.

    Usage::

        dl = YCBDownloader()
        available = dl.list_available()   # e.g. ['002_master_chef_can', ...]
        xml_path  = dl.get("002_master_chef_can")

    Set ``GITHUB_TOKEN`` in the environment to raise the API rate limit from
    60 to 5 000 requests/hour.
    """

    REPO_OWNER = "elpis-lab"
    REPO_NAME = "ycb_dataset"
    BRANCHES = ["main", "master"]

    _available_cache: Optional[List[str]] = None
    _objects_subdir: str = _OBJECTS_SUBDIR

    @property
    def source_name(self) -> str:
        return "ycb"

    def list_available(self) -> List[str]:
        if YCBDownloader._available_cache is not None:
            return list(YCBDownloader._available_cache)
        branch = self._active_branch()
        entries = self._github_get(self._contents_url(_OBJECTS_SUBDIR, branch))
        dirs = sorted(
            e["name"] for e in entries
            if e["type"] == "dir" and not e["name"].startswith(".")
        )
        YCBDownloader._available_cache = dirs
        return dirs

    def _download(self, name: str, dest: Path) -> None:
        branch = self._active_branch()
        remote_path = f"{_OBJECTS_SUBDIR}/{name}"
        print(f"[YCB] Downloading '{name}' from {self.REPO_OWNER}/{self.REPO_NAME}/{remote_path} …")
        self._download_tree(remote_path, dest, branch)
        print(f"[YCB] '{name}' cached at {dest}")

    def _mjcf_path(self, cache_dir: Path) -> Path:
        """Return path to MJCF, generating it from meshes if it doesn't exist yet."""
        xml_path = cache_dir / f"{cache_dir.name}.xml"
        if not xml_path.exists():
            xml_path = self._generate_mjcf(cache_dir)
        return xml_path

    @staticmethod
    def _generate_mjcf(cache_dir: Path) -> Path:
        """Generate a MuJoCo MJCF from the downloaded mesh/texture files.

        Expected layout (elpis-lab/ycb_dataset convention):
            textured.obj          — visual mesh
            textured_coacd_N.stl  — collision meshes (N = 0, 1, …)
            texture_map.png       — colour texture  (optional)
        """
        name = cache_dir.name

        root = ET.Element("mujoco", model=name)
        # meshdir="." so all file= paths are relative to this XML
        ET.SubElement(root, "compiler", meshdir=".")

        asset = ET.SubElement(root, "asset")
        has_texture = (cache_dir / "texture_map.png").exists()
        if has_texture:
            ET.SubElement(asset, "texture", name="tex", type="2d", file="texture_map.png")
            ET.SubElement(asset, "material", name="mat", texture="tex",
                          specular="0.5", shininess="0.25")

        has_visual = (cache_dir / "textured.obj").exists()
        if has_visual:
            ET.SubElement(asset, "mesh", name="visual", file="textured.obj")

        col_meshes = sorted(cache_dir.glob("textured_coacd_*.stl"))
        for i, stl in enumerate(col_meshes):
            ET.SubElement(asset, "mesh", name=f"col_{i}", file=stl.name)

        worldbody = ET.SubElement(root, "worldbody")
        body = ET.SubElement(worldbody, "body", name=name)

        if has_visual:
            geom_kw = dict(type="mesh", mesh="visual", contype="0", conaffinity="0")
            if has_texture:
                geom_kw["material"] = "mat"
            ET.SubElement(body, "geom", **geom_kw)

        for i in range(len(col_meshes)):
            ET.SubElement(body, "geom", type="mesh", mesh=f"col_{i}", group="3")

        if not has_visual and not col_meshes:
            # Fallback: tiny sphere so the body isn't empty
            ET.SubElement(body, "geom", type="sphere", size="0.03")

        ET.indent(root, space="  ")
        xml_path = cache_dir / f"{name}.xml"
        xml_path.write_text(ET.tostring(root, encoding="unicode"))
        return xml_path
