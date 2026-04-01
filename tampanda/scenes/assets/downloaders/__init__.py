"""Asset downloaders for remote MuJoCo object libraries."""

from tampanda.scenes.assets.downloaders.base import BaseDownloader
from tampanda.scenes.assets.downloaders.ycb import YCBDownloader
from tampanda.scenes.assets.downloaders.gso import GSODownloader

__all__ = ["BaseDownloader", "YCBDownloader", "GSODownloader"]
