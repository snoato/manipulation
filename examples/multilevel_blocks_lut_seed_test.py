"""Test the seed-then-short-converge hypothesis for the IK LUT.

For every cached (cell, "upright", quat) in the LUT, try IK at the
TWO put_upright probe targets (column_align_preview and
final_descent_preview), seeded from the cached arm_q at cell_centre.

Reports per-target:
  * how many iters mink needed
  * convergence rate
  * comparison vs no-seed (seed from HOME_STACK)

If the seeded version converges in <15-30 iters reliably, the LUT can
replace Tier 0 IK in put_upright with a much shorter probe.
"""
from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import (
    MultilevelBlocksConfig,
    make_multilevel_blocks_builder,
)
from tampanda.symbolic.domains.multilevel_blocks.executor import (
    _HOME_STACK,
    _TABLE_TOP_LOCAL_Z,
)
from tampanda.symbolic.domains.multilevel_blocks.ik_seed_lut import (
    IKSeedLUT,
)
from tampanda.symbolic.workspace import Cell


def _converge_with_iters(env, target_pos, target_quat, max_iters):
    """Run mink converge_ik with a custom cap.  Returns (ok, iters_used)."""
    import mink
    ik = env.ik
    T_wt = mink.SE3.from_mocap_name(ik.model, ik.data, ik.target_name)
    ik.ee_task.set_target(T_wt)
    ik.posture_task.set_target_from_configuration(ik.configuration)
    for i in range(max_iters):
        vel = mink.solve_ik(ik.configuration, ik.tasks(), 0.005,
                                      ik.solver, 1e-3)
        ik.configuration.integrate_inplace(vel, 0.005)
        err = ik.ee_task.compute_error(ik.configuration)
        pos_ok = np.linalg.norm(err[:3]) <= ik.pos_threshold
        ori_ok = np.linalg.norm(err[3:]) <= ik.ori_threshold
        if pos_ok and ori_ok:
            return True, i + 1
    return False, max_iters


def _trial(env, target_pos, target_quat, seed_q, max_iters):
    """Set qpos to seed, set target, converge.  Returns (ok, iters)."""
    env.data.qpos[:7] = seed_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(target_pos, target_quat)
    return _converge_with_iters(env, target_pos, target_quat, max_iters)


def main() -> int:
    cfg = MultilevelBlocksConfig(n_cubes=20, n_oblong=12, n_long=6)
    lut = IKSeedLUT.from_default_path()

    # Find upright entries with cached arm_q.
    upright = [(cell, qkey, arm_q)
                  for (cell, fam, qkey), arm_q in lut._entries.items()
                  if fam == "upright"]
    print(f"upright LUT entries: {len(upright)}")
    print()

    with tempfile.TemporaryDirectory(prefix="lut_seed_test_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )
        env = builder.build_env(rate=10000.0)

        # Constants
        stack_table_top_z = cfg.stack_table_pos[2] + _TABLE_TOP_LOCAL_Z
        max_stack_height = cfg.stack_grid_cells[2] * cfg.cube_size
        traverse_z = stack_table_top_z + max_stack_height + 0.08

        # Test a SUBSET — first 50 entries (mix of cells / yaws).
        sample = upright[: min(50, len(upright))]

        # For each: try LUT-seeded short-converge at column_align_preview.
        # Then compare with HOME_STACK-seeded long-converge.
        print(f"{'cell':>20s}  {'mode':>16s}  {'pos_z':>6s}  {'ok':>3s}  "
                  f"{'iters':>5s}")
        print("-" * 80)
        seeded_iters_ca = []
        seeded_iters_fd = []
        seeded_ok_ca = Counter()
        seeded_ok_fd = Counter()
        cold_iters_ca = []
        cold_ok_ca = Counter()

        # parse quat from key
        def _parse_quat(qkey: str) -> np.ndarray:
            return np.array([float(x) for x in qkey.split("_")],
                                dtype=float)

        for cell_id, qkey, arm_q in sample:
            quat = _parse_quat(qkey)
            cell = Cell.parse(cell_id)
            cell_centre = np.asarray(ws.pose_for(cell), dtype=float)
            # column_align_preview: high above target
            ca_target = np.array([cell_centre[0], cell_centre[1], traverse_z])
            # final_descent_preview: at place pose (z = anchor.z + 0.04)
            fd_target = np.array([cell_centre[0], cell_centre[1],
                                          cell_centre[2] + 0.04])

            # Seeded short-converge at column_align
            ok, iters = _trial(env, ca_target, quat, arm_q, max_iters=30)
            seeded_iters_ca.append(iters)
            seeded_ok_ca[ok] += 1

            # Seeded short-converge at final_descent
            ok, iters = _trial(env, fd_target, quat, arm_q, max_iters=30)
            seeded_iters_fd.append(iters)
            seeded_ok_fd[ok] += 1

            # COLD (HOME_STACK seed) at column_align — for comparison
            ok_cold, iters_cold = _trial(env, ca_target, quat, _HOME_STACK,
                                                          max_iters=200)
            cold_iters_ca.append(iters_cold)
            cold_ok_ca[ok_cold] += 1

        print(f"\nSeeded probe at column_align_preview (cap=30):")
        print(f"  ok rate:   {seeded_ok_ca[True]}/{len(sample)}")
        print(f"  iter dist: min={min(seeded_iters_ca)}  "
                  f"median={int(np.median(seeded_iters_ca))}  "
                  f"max={max(seeded_iters_ca)}  "
                  f"mean={np.mean(seeded_iters_ca):.1f}")
        bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
        for lo, hi in bins:
            n = sum(1 for it in seeded_iters_ca if lo <= it < hi)
            print(f"    {lo:>2d}-{hi:<2d}: {n}")

        print(f"\nSeeded probe at final_descent_preview (cap=30):")
        print(f"  ok rate:   {seeded_ok_fd[True]}/{len(sample)}")
        print(f"  iter dist: min={min(seeded_iters_fd)}  "
                  f"median={int(np.median(seeded_iters_fd))}  "
                  f"max={max(seeded_iters_fd)}  "
                  f"mean={np.mean(seeded_iters_fd):.1f}")
        for lo, hi in bins:
            n = sum(1 for it in seeded_iters_fd if lo <= it < hi)
            print(f"    {lo:>2d}-{hi:<2d}: {n}")

        print(f"\nCOLD (HOME_STACK seed) probe at column_align (cap=200):")
        print(f"  ok rate:   {cold_ok_ca[True]}/{len(sample)}")
        print(f"  iter dist: min={min(cold_iters_ca)}  "
                  f"median={int(np.median(cold_iters_ca))}  "
                  f"max={max(cold_iters_ca)}  "
                  f"mean={np.mean(cold_iters_ca):.1f}")
        bins_long = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 200)]
        for lo, hi in bins_long:
            n = sum(1 for it in cold_iters_ca if lo <= it < hi)
            print(f"    {lo:>3d}-{hi:<3d}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
