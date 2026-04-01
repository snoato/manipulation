"""Abstract base class for remote asset downloaders."""

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class BaseDownloader(ABC):
    """Downloads named MuJoCo objects from a remote source into the local cache.

    Subclasses implement :meth:`_download`, :meth:`list_available`, and
    declare a :attr:`source_name` property.
    """

    # GitHub repo coordinates — override in subclasses
    REPO_OWNER: str = ""
    REPO_NAME: str = ""
    BRANCHES: List[str] = ["main", "master"]

    # Cached after first successful probe
    _branch_cache: Optional[str] = None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier used as the first-level cache directory (e.g. 'ycb')."""

    @abstractmethod
    def list_available(self) -> List[str]:
        """Return names of all available objects in the remote library."""

    @abstractmethod
    def _download(self, name: str, dest: Path) -> None:
        """Download all files for *name* into the *dest* directory."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_cached(self, name: str) -> bool:
        from tampanda.scenes.assets.cache import AssetCache
        return AssetCache().is_cached(self.source_name, name)

    def get(self, name: str) -> Path:
        """Return the path to the object's main MJCF XML, downloading if needed."""
        from tampanda.scenes.assets.cache import AssetCache
        dest = AssetCache().ensure(
            self.source_name, name,
            lambda d: self._download(name, d),
        )
        return self._mjcf_path(dest)

    def mjcf_path(self, name: str) -> Path:
        """Like :meth:`get` but raises if not yet cached."""
        from tampanda.scenes.assets.cache import AssetCache
        cache = AssetCache()
        if not cache.is_cached(self.source_name, name):
            raise FileNotFoundError(
                f"{self.source_name}/{name} is not cached. "
                f"Call get('{name}') first to download it."
            )
        return self._mjcf_path(cache.path(self.source_name, name))

    # ------------------------------------------------------------------
    # GitHub helpers
    # ------------------------------------------------------------------

    def _github_get(self, url: str):
        """Make a GitHub API GET request and return the parsed JSON."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "manipulation-library/1.0",
        }
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _contents_url(self, path: str = "", branch: str = "main") -> str:
        base = (
            f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}"
            f"/contents"
        )
        if path:
            base += f"/{path}"
        return f"{base}?ref={branch}"

    def _active_branch(self) -> str:
        """Return the repo's default branch (cached after first call).

        Tries each entry in BRANCHES in order, using a cheap HEAD request
        so we don't burn a Contents API slot just for branch detection.
        """
        if self._branch_cache is not None:
            return self._branch_cache
        import urllib.error
        for branch in self.BRANCHES:
            url = (
                f"https://api.github.com/repos/{self.REPO_OWNER}/{self.REPO_NAME}"
                f"/git/refs/heads/{branch}"
            )
            try:
                self._github_get(url)
                BaseDownloader._branch_cache = branch
                return branch
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
        # Default to first branch if detection fails
        BaseDownloader._branch_cache = self.BRANCHES[0]
        return self.BRANCHES[0]

    def _download_tree(self, remote_path: str, dest: Path, branch: str) -> None:
        """Recursively download a directory tree from GitHub."""
        entries = self._github_get(self._contents_url(remote_path, branch))
        for entry in entries:
            if entry["type"] == "file":
                local = dest / entry["name"]
                _download_file(entry["download_url"], local)
            elif entry["type"] == "dir":
                sub = dest / entry["name"]
                sub.mkdir(parents=True, exist_ok=True)
                self._download_tree(f"{remote_path}/{entry['name']}", sub, branch)

    # ------------------------------------------------------------------
    # XML path discovery
    # ------------------------------------------------------------------

    def _mjcf_path(self, cache_dir: Path) -> Path:
        """Find the main MJCF XML file inside a cached object directory."""
        xml_files = sorted(cache_dir.glob("*.xml"))
        if not xml_files:
            # Search one level deeper
            xml_files = sorted(cache_dir.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No .xml file found under {cache_dir}")
        # Prefer file named after the directory or named model.xml
        dir_name = cache_dir.name
        for candidate_name in (f"{dir_name}.xml", "model.xml"):
            for f in xml_files:
                if f.name == candidate_name:
                    return f
        return xml_files[0]


def _download_file(url: str, dest: Path) -> None:
    """Download a single file, showing a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "manipulation-library/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        while True:
            block = resp.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
