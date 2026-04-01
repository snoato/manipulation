"""Remote asset downloading and local caching for YCB and GSO objects."""

from tampanda.scenes.assets.cache import AssetCache
from tampanda.scenes.assets.downloaders.ycb import YCBDownloader
from tampanda.scenes.assets.downloaders.gso import GSODownloader

__all__ = ["AssetCache", "YCBDownloader", "GSODownloader"]
