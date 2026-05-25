"""Smoke test: restore_state(on_held='attach') attaches the right block
at the right offset for every held-* fluent family.

Builds a fresh multilevel_blocks env, then for each of the four held-*
fluents repeats:

  1. Call restore_state with that fluent + block.
  2. Verify env._attached refers to the right body.
  3. Verify executor._held_block + _held_offset are populated.
  4. Verify world-frame block-vs-EE geometry matches the canonical
     grasp pose:
       * held-cube / held-flat-x / held-flat-y  → top-down handoff,
         block 14mm below EE in world z.
       * held-upright (2-cell oblong)            → front handoff,
         block 0.5*cube_size below EE in world z.
       * held-upright (3-cell long)              → front handoff,
         block 1.0*cube_size below EE in world z.

Catches the original silent bug (restore left the gripper empty when
the state had a held-* fluent).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    make_multilevel_blocks_builder,
    make_multilevel_blocks_bridge,
)
from tampanda.symbolic.domains.multilevel_blocks.feasibility import (
    _make_executor,
)
from tampanda.symbolic.domains.multilevel_blocks.state import restore_state


def _check_one(env, ws, cfg, executor, *, fluent: str, block: str,
                  expected_world_offset: np.ndarray,
                  xy_tol: float = 5e-3, z_tol: float = 1e-3) -> bool:
    """Restore a state with the given held-* fluent and assert the
    resulting kinematic state matches expectations."""
    state = {(fluent, block): True}

    # Detach anything from a prior round to make sure restore is what
    # actually performs the new attach.
    if getattr(env, "_attached", None) is not None:
        env.detach_object()
    executor._held_block = None
    executor._held_offset = np.zeros(3)

    restore_state(env, ws, cfg, state,
                       on_held="attach", executor=executor)

    att = getattr(env, "_attached", None)
    assert att is not None and att["body_name"] == block, (
        f"[{fluent}] attached body should be {block!r}, got "
        f"{att and att['body_name']!r}"
    )
    assert executor._held_block == block, (
        f"[{fluent}] executor._held_block should be {block!r}, got "
        f"{executor._held_block!r}"
    )

    pos, _ = env.get_object_pose(block)
    ee_pos = env.data.site_xpos[executor.ee_site_id]
    delta = np.asarray(pos) - np.asarray(ee_pos)

    z_ok = abs(delta[2] - expected_world_offset[2]) < z_tol
    xy_ok = (abs(delta[0] - expected_world_offset[0]) < xy_tol
                  and abs(delta[1] - expected_world_offset[1]) < xy_tol)

    held_off = executor._held_offset
    held_off_ok = np.allclose(held_off, delta, atol=1e-6)

    mark = "✓" if (z_ok and xy_ok and held_off_ok) else "✗"
    print(f"  {mark} {fluent:14s} block={block:10s}  "
              f"delta=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})  "
              f"expected=({expected_world_offset[0]:+.4f}, "
              f"{expected_world_offset[1]:+.4f}, {expected_world_offset[2]:+.4f})")
    if not (z_ok and xy_ok):
        print(f"    geometry mismatch (xy_tol={xy_tol}, z_tol={z_tol})")
    if not held_off_ok:
        print(f"    executor._held_offset={held_off} disagrees with delta")
    return z_ok and xy_ok and held_off_ok


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=4, n_oblong=2, n_long=1)
    with tempfile.TemporaryDirectory(prefix="mlb_attach_smoke_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)
        bridge, objects = make_multilevel_blocks_bridge(env, ws, cfg)
        executor = _make_executor(env, ws, cfg, fast=True)

        cube_size = cfg.cube_size
        # Top-down: 14mm below EE in world z.
        top_down = np.array([0.0, 0.0, -0.014])
        # Front grasp: 2-cell oblong upright = cube_size/2 below; 3-cell
        # long upright = cube_size below.
        front_oblong = np.array([0.0, 0.0, -cube_size / 2.0])
        front_long   = np.array([0.0, 0.0, -cube_size])

        cases: list[Tuple[str, str, np.ndarray]] = [
            ("held-cube",    "cube_0",   top_down),
            ("held-flat-x",  "oblong_0", top_down),
            ("held-flat-y",  "oblong_1", top_down),
            ("held-upright", "oblong_0", front_oblong),
            ("held-upright", "long_0",   front_long),
        ]

        print(f"Running {len(cases)} restore-state attach cases:\n")
        all_pass = True
        for fluent, block, expected in cases:
            ok = _check_one(env, ws, cfg, executor,
                                fluent=fluent, block=block,
                                expected_world_offset=expected)
            all_pass = all_pass and ok

        print()
        if all_pass:
            print("ALL CHECKS PASSED")
            return 0
        print("ONE OR MORE CHECKS FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
