"""Downloader for kevinzakka/mujoco_scanned_objects — Google Scanned Objects."""

from pathlib import Path
from typing import List, Optional

from manipulation.scenes.assets.downloaders.base import BaseDownloader, _download_file


class GSODownloader(BaseDownloader):
    """Downloads Google Scanned Objects from ``kevinzakka/mujoco_scanned_objects``.

    The repo contains ~1 030 everyday objects converted to MJCF with CoACD
    collision meshes.  Each object lives in its own subdirectory.

    Usage::

        dl = GSODownloader()
        available = dl.list_available()   # e.g. ['Alarm_Clock', 'Apple', ...]
        xml_path  = dl.get("Alarm_Clock")

    Set ``GITHUB_TOKEN`` in the environment to raise the API rate limit from
    60 to 5 000 requests/hour.
    """

    REPO_OWNER = "kevinzakka"
    REPO_NAME = "mujoco_scanned_objects"
    BRANCHES = ["main", "master"]

    # The objects live inside a subdirectory of the repo
    _OBJECTS_SUBDIR = "models"

    _available_cache: Optional[List[str]] = None

    @property
    def source_name(self) -> str:
        return "gso"

    def list_available(self) -> List[str]:
        if GSODownloader._available_cache is not None:
            return list(GSODownloader._available_cache)
        branch = self._active_branch()
        entries = self._github_get(self._contents_url(self._OBJECTS_SUBDIR, branch))
        dirs = sorted(
            e["name"] for e in entries
            if e["type"] == "dir" and not e["name"].startswith(".")
        )
        GSODownloader._available_cache = dirs
        return dirs

    def _download(self, name: str, dest: Path) -> None:
        branch = self._active_branch()
        subdir = self._OBJECTS_SUBDIR
        remote_path = f"{subdir}/{name}" if subdir else name
        print(
            f"[GSO] Downloading '{name}' from {self.REPO_OWNER}/{self.REPO_NAME} …"
        )
        self._download_tree(remote_path, dest, branch)
        print(f"[GSO] '{name}' cached at {dest}")
