"""Deterministically subsample a multilevel_blocks train set.

Given a source directory ``data/multilevel_blocks/train/`` containing
``config_*.pddl`` + ``config_*.pddl.plan`` (+ optional ``viz/``), produce
a smaller subset directory preserving per-level proportions and using
a seeded shuffle so the same call always gives the same subset.

Per-level proportions are computed from the source set's plan-file
``; difficulty:`` metadata trailer, and a budget of ``--size`` examples
is allocated proportionally.

Usage::

    python examples/sample_dataset_subset.py \\
        --source data/multilevel_blocks/train \\
        --dest   data/multilevel_blocks/train_200 \\
        --size   200 \\
        --seed   42
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _parse_difficulty(plan_path: Path) -> int:
    """Read difficulty from the plan file's metadata trailer."""
    text = plan_path.read_text()
    for line in text.splitlines():
        if line.startswith("; difficulty:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return -1   # unknown


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = args.source.resolve()
    if not src.is_dir():
        raise SystemExit(f"source not a directory: {src}")
    dst = args.dest.resolve()
    if dst.exists():
        raise SystemExit(f"dest already exists: {dst}")

    # Index source by difficulty.
    by_level: Dict[int, List[Path]] = defaultdict(list)
    for plan_path in sorted(src.glob("config_*.pddl.plan")):
        diff = _parse_difficulty(plan_path)
        by_level[diff].append(plan_path)
    if not by_level:
        raise SystemExit(f"no plans found in {src}")

    total = sum(len(v) for v in by_level.values())
    print(f"Source: {src}  total={total}")
    for lvl in sorted(by_level):
        print(f"  L{lvl}: {len(by_level[lvl])} examples")

    # Allocate per-level counts proportionally.
    targets: Dict[int, int] = {}
    remainder = args.size
    levels_sorted = sorted(by_level)
    for lvl in levels_sorted[:-1]:
        targets[lvl] = round(len(by_level[lvl]) / total * args.size)
        remainder -= targets[lvl]
    targets[levels_sorted[-1]] = max(0, remainder)

    # Sample deterministic.
    rng = np.random.default_rng(args.seed)
    dst.mkdir(parents=True)
    dst_viz = dst / "viz"
    src_viz = src / "viz"
    if src_viz.is_dir():
        dst_viz.mkdir()

    counter = 1
    summary = {}
    for lvl in levels_sorted:
        available = by_level[lvl]
        want = min(targets[lvl], len(available))
        idx = rng.choice(len(available), size=want, replace=False)
        chosen = [available[i] for i in sorted(idx)]
        summary[lvl] = len(chosen)
        for plan_p in chosen:
            stem = plan_p.name.rsplit(".pddl.plan", 1)[0]
            pddl_p = src / f"{stem}.pddl"
            new_stem = f"config_{counter}"
            shutil.copy(plan_p, dst / f"{new_stem}.pddl.plan")
            shutil.copy(pddl_p, dst / f"{new_stem}.pddl")
            viz_p = src_viz / f"{stem}.png"
            if viz_p.exists():
                shutil.copy(viz_p, dst_viz / f"{new_stem}.png")
            counter += 1

    print(f"\nDest: {dst}  total={counter - 1}")
    for lvl in sorted(summary):
        print(f"  L{lvl}: {summary[lvl]} examples")

    # Also copy domain.pddl from source's parent if present.  Skip if
    # source and dest share the same parent — domain.pddl would copy to
    # itself.
    domain = src.parent / "domain.pddl"
    dest_domain = dst.parent / "domain.pddl"
    if domain.exists() and domain.resolve() != dest_domain.resolve():
        shutil.copy(domain, dest_domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
