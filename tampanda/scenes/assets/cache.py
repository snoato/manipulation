"""Filesystem cache for downloaded MuJoCo assets."""

import os
import shutil
from pathlib import Path


class AssetCache:
    """Local cache for downloaded assets.

    The cache root defaults to ``~/.cache/tampanda/assets`` and can be
    overridden with the ``TAMPANDA_ASSETS_CACHE`` environment variable or
    by passing ``base_dir`` directly.

    Layout::

        <base>/
          ycb/
            002_master_chef_can/
              .ok            ← sentinel: download complete
              model.xml
              meshes/
                ...
          gso/
            Alarm_Clock/
              .ok
              Alarm_Clock.xml
              ...
    """

    def __init__(self, base_dir=None):
        self._base = Path(
            base_dir
            or os.environ.get(
                "TAMPANDA_ASSETS_CACHE",
                Path.home() / ".cache/tampanda/assets",
            )
        )

    def path(self, source: str, name: str) -> Path:
        """Return cache directory for a given source + object name."""
        return self._base / source / name

    def is_cached(self, source: str, name: str) -> bool:
        """Return True if the object has been fully downloaded."""
        return (self.path(source, name) / ".ok").exists()

    def ensure(self, source: str, name: str, download_fn) -> Path:
        """Return the cache dir for (source, name), downloading if needed.

        ``download_fn(dest: Path)`` is called with the destination directory
        when a download is required.  If it raises, the partial directory is
        removed so the next call retries.
        """
        dest = self.path(source, name)
        if not (dest / ".ok").exists():
            dest.mkdir(parents=True, exist_ok=True)
            try:
                download_fn(dest)
                (dest / ".ok").touch()
            except Exception:
                shutil.rmtree(dest, ignore_errors=True)
                raise
        return dest
