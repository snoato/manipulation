"""Backwards-compatibility shim for the old `manipulation` package name.

The package has been renamed to `tampanda`. This shim re-exports everything
so that existing code continues to work, but will emit a DeprecationWarning
to encourage migration.

    # Old (still works, shows warning):
    from manipulation import RRTStar

    # New:
    from tampanda import RRTStar
"""

import warnings

warnings.warn(
    "The 'manipulation' package has been renamed to 'tampanda'. "
    "Please update your imports: `from manipulation import X` → `from tampanda import X`. "
    "This compatibility shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from tampanda import *  # noqa: F401, F403, E402
from tampanda import __version__, __all__  # noqa: F401, E402
