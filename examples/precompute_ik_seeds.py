"""Precompute IK seed configs per (cell, yaw) for multilevel_blocks.

The fast-mode put_upright chain spends ~80-95 % of its time inside
mink's ``converge_ik``.  Most of that is wasted on cold-start IK —
seeded from the handoff or HOME pose, mink takes hundreds of
iterations to converge at a far cell with a tight front-grasp.
Seeded from an arm config that's ALREADY at the target cell, mink
converges in 10-30 iterations.

This script builds a lookup table: for every (cell, yaw) the chain
might IK to, store the converged arm config.  At runtime, the
executor seeds ``env.ik`` with the cached arm_q before slow phases
(column-align, descent), turning a 200-iter probe into a 20-iter one.

Output: ``ik_seed_lut.npz`` in the multilevel_blocks package, loaded
once per worker at executor construction.

Cell × yaw combinations covered:

* Every stack cell ``stack_L{k}__{ix}_{iy}`` × every upright grasp quat
  (8 yaws — covers held-upright placement).
* Every stack cell × every top-down quat (4 yaws — cube / flat-x /
  flat-y put / pick).
* Every parts cell × every top-down quat (cube / flat-x / flat-y pick
  from parts).

Multiprocessed.  On a 4-core Mac with --num-workers 4 finishes in
~5-10 min; on a workstation with 96 cores and --num-workers 32 it
finishes in well under a minute.

Run from repo root::

  # Local (Mac)
  python examples/precompute_ik_seeds.py --num-workers 4

  # On desk-01 workstation (headless MuJoCo + many cores)
  conda activate rgnet_fresh
  export MUJOCO_GL=egl
  python examples/precompute_ik_seeds.py --num-workers 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Output path relative to repo root.
_OUT_PATH = (Path(__file__).parent.parent
              / "tampanda" / "symbolic" / "domains"
              / "multilevel_blocks" / "ik_seed_lut.npz")


def _quat_key(q) -> str:
    """Stable string key for a quat (rounded so float drift doesn't
    cause cache misses)."""
    rounded = np.round(np.asarray(q, dtype=float), 4)
    return "_".join(f"{v:+.4f}" for v in rounded)


# ---------------------------------------------------------------------------
# Worker side — runs in its own subprocess with its own MuJoCo env.
# ---------------------------------------------------------------------------


# Worker-local persistent state.  ``_worker_init`` populates these once
# per spawned process; ``_worker_solve_chunk`` consumes them.
_W: Dict[str, object] = {}


def _worker_init(config_dict: dict) -> None:
    """Build a fresh env + mink IK in this worker process.  Runs once
    per worker at pool startup."""
    import mujoco
    from tampanda.symbolic.domains.multilevel_blocks import (
        MultilevelBlocksConfig, make_multilevel_blocks_builder,
    )
    from tampanda.symbolic.domains.multilevel_blocks.executor import (
        _HOME_PARTS, _HOME_STACK,
    )

    cfg = MultilevelBlocksConfig(**config_dict)
    scratch = tempfile.TemporaryDirectory(prefix=f"ikseed_w{os.getpid()}_")
    builder, ws, cfg = make_multilevel_blocks_builder(
        scratch_dir=Path(scratch.name), config=cfg,
    )
    env = builder.build_env(rate=10000.0)

    _W["scratch"] = scratch
    _W["env"] = env
    _W["ws"] = ws
    _W["cfg"] = cfg
    _W["HOME_PARTS"] = _HOME_PARTS
    _W["HOME_STACK"] = _HOME_STACK
    _W["mujoco"] = mujoco


def _worker_solve_one(payload: Tuple[str, str, str,
                                            np.ndarray, np.ndarray, np.ndarray,
                                            int]) -> Tuple[str, str, str,
                                                              bool, np.ndarray]:
    """Solve one IK target.  Payload:
        (family, cell_id, quat_key, target_pos, quat, seed_q, max_iters)
    Returns:
        (family, cell_id, quat_key, converged, q7)
    """
    family, cell_id, qkey, pos, quat, seed_q, max_iters = payload
    env = _W["env"]
    mujoco = _W["mujoco"]

    env.data.qpos[:7] = seed_q
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.update_configuration(env.data.qpos)
    env.ik.set_target_position(pos, quat)

    saved = env.ik.max_iters
    env.ik.max_iters = int(max_iters)
    try:
        ok = env.ik.converge_ik(0.005)
        q7 = env.ik.configuration.q[:7].copy()
    finally:
        env.ik.max_iters = saved
    return (family, cell_id, qkey, ok, q7)


# ---------------------------------------------------------------------------
# Main side — builds the target list + drives the pool.
# ---------------------------------------------------------------------------


def _build_targets(cfg_dict: dict, max_iters: int
                       ) -> List[Tuple]:
    """Enumerate every (cell, family, quat) IK target up-front and
    package as serialisable payloads.  This runs ONCE in the main
    process; workers consume the resulting list.
    """
    # Local imports so the main process can also use it standalone if needed.
    from tampanda.symbolic.domains.multilevel_blocks import (
        MultilevelBlocksConfig, make_multilevel_blocks_builder,
    )
    from tampanda.symbolic.domains.multilevel_blocks.executor import (
        _cube_grasp_quats, _flat_x_grasp_quats, _flat_y_grasp_quats,
        _upright_grasp_quats, _HOME_PARTS, _HOME_STACK,
        _EE_TO_BLOCK_CENTRE_Z,
    )

    cfg = MultilevelBlocksConfig(**cfg_dict)
    with tempfile.TemporaryDirectory(prefix="ikseed_targets_") as scratch:
        builder, ws, cfg = make_multilevel_blocks_builder(
            scratch_dir=Path(scratch), config=cfg,
        )

        # Quat dictionaries (one entry per orientation).
        top_down_quats: Dict[str, np.ndarray] = {}
        for q in _cube_grasp_quats():
            top_down_quats[_quat_key(q)] = np.asarray(q, dtype=float)
        for q in _flat_x_grasp_quats():
            top_down_quats[_quat_key(q)] = np.asarray(q, dtype=float)
        for q in _flat_y_grasp_quats():
            top_down_quats[_quat_key(q)] = np.asarray(q, dtype=float)

        upright_quats: Dict[str, np.ndarray] = {
            _quat_key(q): np.asarray(q, dtype=float)
            for q in _upright_grasp_quats()
        }

        targets: List[Tuple] = []
        for region_name in ws.regions:
            region = ws[region_name]
            home_seed = (_HOME_PARTS if region_name == "parts"
                              else _HOME_STACK)
            for cell in region.cells():
                cell_centre = np.asarray(ws.pose_for(cell), dtype=float)
                top_down_anchor = cell_centre + np.array(
                    [0.0, 0.0, _EE_TO_BLOCK_CENTRE_Z]
                )
                for qkey, q in top_down_quats.items():
                    targets.append(
                        ("top_down", cell.id, qkey,
                         top_down_anchor.copy(), q.copy(),
                         home_seed.copy(), max_iters)
                    )
                if region_name.startswith("stack_L"):
                    for qkey, q in upright_quats.items():
                        targets.append(
                            ("upright", cell.id, qkey,
                             cell_centre.copy(), q.copy(),
                             _HOME_STACK.copy(), max_iters)
                        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int,
                            default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--max-iters", type=int, default=500,
                            help="mink IK max_iters per probe")
    parser.add_argument("--chunksize", type=int, default=8,
                            help="targets per pool dispatch")
    parser.add_argument("--n-cubes", type=int, default=20)
    parser.add_argument("--n-oblong", type=int, default=12)
    parser.add_argument("--n-long", type=int, default=6)
    parser.add_argument("--out", type=Path, default=_OUT_PATH,
                            help="output .npz path")
    args = parser.parse_args()

    cfg_dict = dict(n_cubes=args.n_cubes,
                          n_oblong=args.n_oblong,
                          n_long=args.n_long)

    print(f"Config: {cfg_dict}")
    print(f"Output: {args.out}")
    print(f"Workers: {args.num_workers}")
    print(f"mink max_iters: {args.max_iters}")
    print()

    print("Building target list ...")
    t_build0 = time.perf_counter()
    targets = _build_targets(cfg_dict, args.max_iters)
    print(f"  {len(targets)} targets enumerated "
              f"({time.perf_counter() - t_build0:.1f}s)")
    print()

    # Dispatch across workers.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        args.num_workers,
        initializer=_worker_init,
        initargs=(cfg_dict,),
    )

    print(f"Solving {len(targets)} IK targets across {args.num_workers} workers ...")
    t_run0 = time.perf_counter()
    results: List[Tuple[str, str, str, bool, np.ndarray]] = []
    n_done = 0
    for r in pool.imap_unordered(_worker_solve_one, targets,
                                          chunksize=args.chunksize):
        results.append(r)
        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.perf_counter() - t_run0
            eta = (elapsed / n_done) * (len(targets) - n_done)
            n_ok = sum(1 for x in results if x[3])
            print(f"  [{n_done:>4d}/{len(targets)}]  "
                      f"ok={n_ok}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    pool.close()
    pool.join()

    elapsed = time.perf_counter() - t_run0
    n_ok = sum(1 for x in results if x[3])
    n_fail = len(results) - n_ok
    print()
    print(f"Done.  ok={n_ok}  fail={n_fail}  elapsed={elapsed:.0f}s "
              f"({elapsed/max(1, len(results))*1000:.0f} ms/target avg)")

    if n_ok == 0:
        print("ERROR: no converged seeds — LUT would be empty.")
        return 1

    # Keep only converged entries.
    converged = [r for r in results if r[3]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        cells=np.array([r[1] for r in converged], dtype=object),
        families=np.array([r[0] for r in converged], dtype=object),
        quat_keys=np.array([r[2] for r in converged], dtype=object),
        qs=np.array([r[4] for r in converged], dtype=np.float64),
    )
    size_kb = args.out.stat().st_size / 1024.0
    print(f"Wrote {args.out}  ({size_kb:.1f} KB, {len(converged)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
