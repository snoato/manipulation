"""IK seed LUT loader for multilevel_blocks.

Loads the ``ik_seed_lut.npz`` produced by
``examples/precompute_ik_seeds.py`` (or a fresh run on the workstation)
and exposes a fast in-memory lookup:

    lut = IKSeedLUT.from_default_path()  # raises if .npz missing
    arm_q = lut.lookup(cell_id="stack_L1__5_5",
                            family="upright",
                            quat=np.array([-0.5, 0.5, 0.5, 0.5]))
    # arm_q is a 7-DOF np.float64 array, or None on cache miss.

Used by the executor to seed mink IK before slow IK phases (most
notably the put_upright column-align step).  With a good seed mink
converges in ~10-30 iterations instead of running to max_iters at
~200 ms a pop.

The LUT is OPTIONAL — if the .npz file isn't found at load time, the
executor falls back to cold-start IK (current behaviour, just slower
on put_upright).  Use ``IKSeedLUT.from_default_path(strict=False)``
to suppress the FileNotFoundError.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


_DEFAULT_PATH = Path(__file__).parent / "ik_seed_lut.npz"


def _quat_key(q) -> str:
    """Stable string key for an array quat (rounded to match
    precompute_ik_seeds.py)."""
    rounded = np.round(np.asarray(q, dtype=float), 4)
    return "_".join(f"{v:+.4f}" for v in rounded)


class IKSeedLUT:
    """Lookup table: ``(cell_id, family, quat_key) -> arm_q (7-DOF)``.

    Build with :meth:`from_default_path` (loads the bundled .npz) or
    :meth:`from_path` for a custom location.  ``lookup()`` returns
    ``None`` on cache miss so callers don't need try/except.

    The LUT is read-only at runtime; no thread / process synchronisation
    needed.  Memory footprint ~150-300 KB for the default 4000-entry
    LUT.
    """

    def __init__(self, entries: Dict):
        self._entries = entries
        self.n = len(entries)

    @classmethod
    def from_path(cls, path: Path) -> "IKSeedLUT":
        if not path.exists():
            raise FileNotFoundError(
                f"IK seed LUT not found at {path}.  Run "
                f"`python examples/precompute_ik_seeds.py` to build it."
            )
        with np.load(path, allow_pickle=True) as data:
            cells = data["cells"]
            families = data["families"]
            quat_keys = data["quat_keys"]
            qs = data["qs"]
        entries: Dict = {}
        for i in range(len(cells)):
            # Normalize cell_id casing at load time.  precompute_ik_seeds.py
            # builds with raw PDDL casing (capital-L: stack_L0__...).  Callers
            # arrive with either casing — tampanda native uses capital, but
            # rgnet (via pymimir/xmimir) lowercases all symbols.  Storing
            # lowercase keys and lowercasing the lookup argument makes the
            # LUT case-insensitive without forcing all callers to normalize.
            entries[(str(cells[i]).lower(), str(families[i]),
                          str(quat_keys[i]))] = (
                np.asarray(qs[i], dtype=np.float64)
            )
        return cls(entries)

    @classmethod
    def from_default_path(cls, strict: bool = True) -> "Optional[IKSeedLUT]":
        if not _DEFAULT_PATH.exists():
            if strict:
                raise FileNotFoundError(
                    f"IK seed LUT not found at {_DEFAULT_PATH}.  Run "
                    f"`python examples/precompute_ik_seeds.py` to build it."
                )
            return None
        return cls.from_path(_DEFAULT_PATH)

    def lookup(self, cell_id: str, family: str,
                  quat: np.ndarray) -> Optional[np.ndarray]:
        """Return cached arm_q (7-DOF) for the (cell, family, quat)
        triple, or ``None`` on cache miss.

        ``family`` is ``"top_down"`` (cube / flat / long picks and
        puts) or ``"upright"`` (upright pick / put / long-upright).
        ``quat`` is matched by rounded key (4-digit precision).
        ``cell_id`` is matched case-insensitively — see ``from_path``."""
        return self._entries.get((cell_id.lower(), family, _quat_key(quat)))

    def has(self, cell_id: str, family: str, quat: np.ndarray) -> bool:
        return (cell_id.lower(), family, _quat_key(quat)) in self._entries

    def __len__(self) -> int:
        return self.n


def load_default(strict: bool = False) -> "Optional[IKSeedLUT]":
    """Convenience: load the LUT if it exists, else return None.

    Used by the executor's __init__ — when None, downstream IK code
    just skips seeding (current behaviour).
    """
    return IKSeedLUT.from_default_path(strict=strict)
