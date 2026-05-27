"""Per-cell deck-traverse trajectory LUT for access-19.

Phase-1 optimisation: the dominant cost in ``check_action`` is the
horizontal Cartesian traverse over the shelf top at ``safe_z``.  This
LUT pre-computes the natural mink trajectory once per deck cell × quat
(in an empty environment) so runtime calls can:

* **Short-circuit** infeasible targets (``LUTEntry.trajectory is None``)
  without invoking mink at all.
* **Replay** the natural trajectory in the current env — substep
  collisions are still checked against blockers, but the kinematics
  are pre-solved, eliminating 16 mink convergence iterations per call.
* **Order quats** by per-cell historical success so the first attempt
  hits the most-likely-feasible quat.

Alignment guarantee: mink in ``plan_to_pose`` is collision-unaware —
it solves IK purely against kinematics, then the substep collision
check rejects in-collision configs.  Replaying the empty-env-recorded
trajectory in a populated env therefore produces the same substep
qpos sequence; only the collision verdict differs (which is what we
want).  Validated on the L4 stress set (366 actions FAST vs FULL).

Public API::

    lut = Access19IKSeedLUT()
    lut.precompute(env, ws, cfg, lik, shelf_home, cube_half_z)

    # Pass into the chain factories:
    make_access19_put_fn(..., seed_lut=lut)
    make_access19_pick_fn(..., seed_lut=lut)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from tampanda.planners.grasp_planner import GRASP_CONTACT_OFFSET
from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.symbolic.workspace import Workspace

from tampanda.symbolic.domains.access19.env_builder import Access19Config


# Mirror chains.py — duplicated here to avoid a chains.py import cycle.
_FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])
_FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])
_QUATS: Tuple[np.ndarray, ...] = (_FRONT_QUAT, _FRONT_QUAT_FLIPPED)
_SAFE_Z_ABOVE_SHELF_TOP = 0.10


@dataclass
class LUTEntry:
    """One precomputed traverse trajectory.

    ``trajectory`` is the substep-by-substep joint config sequence
    (length = ``n_substeps + 1`` with the start config first), or
    ``None`` if the empty-env mink couldn't converge / pass collision
    checks for this (cell, quat).
    """
    trajectory: Optional[List[np.ndarray]] = None
    mink_ms: float = 0.0


@dataclass
class Access19IKSeedLUT:
    """LUT keyed by ``(cell_id, quat_idx) → LUTEntry`` for the deck
    horizontal traverse (chains.py: ``_put_deck`` line 427 and
    ``_pick_deck`` line 788).

    ``cell_id`` is the ``shelf_top__<ix>_<iy>`` id.  ``quat_idx`` is
    the index into ``chains._QUATS``.

    ``quat_priority[cell_id]`` is the per-cell quat preference order
    (success first, then mink-time ascending) populated by
    :meth:`precompute`.
    """
    cache: Dict[Tuple[str, int], LUTEntry] = field(default_factory=dict)
    quat_priority: Dict[str, List[int]] = field(default_factory=dict)
    n_substeps: int = 16
    _post_lift_qpos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_priority(self, cell_id: str) -> List[int]:
        """Return per-cell quat order, or default ``[0, 1, ...]`` if
        the cell wasn't precomputed (caller falls back to natural mink).
        """
        return self.quat_priority.get(cell_id, list(range(len(_QUATS))))

    def get_trajectory(
        self, cell_id: str, quat_idx: int,
    ) -> Optional[List[np.ndarray]]:
        entry = self.cache.get((cell_id, quat_idx))
        if entry is None:
            return None
        return entry.trajectory

    def all_unreachable(self, cell_id: str) -> bool:
        """True iff every quat probed in precompute came back None.

        ``_try_cartesian`` should short-circuit to ``None`` immediately
        — no mink, no collision checks.  Soundness: unreachable in
        empty env → unreachable in any populated env.
        """
        if cell_id not in self.quat_priority:
            return False  # cell not in LUT (e.g. non-deck) → don't short-circuit
        return all(
            self.cache.get((cell_id, qi), LUTEntry()).trajectory is None
            for qi in range(len(_QUATS))
        )

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute(
        self,
        env,
        workspace: Workspace,
        config: Access19Config,
        lik: LinearIKPlanner,
        shelf_home: np.ndarray,
        cube_half_z: float = 0.040,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Build the LUT.  Must be called with the env in a clean
        empty-layout state (no blockers near the deck).  Caller is
        responsible for restoring whatever env state they want after.

        Returns a small stats dict ``{"precompute_s": …,
        "n_reachable": …, "n_cells": …}``.
        """
        t0 = time.perf_counter()

        region_t = workspace["shelf_top"]
        cube_top_z = region_t.level_z - cube_half_z
        safe_z = cube_top_z + _SAFE_Z_ABOVE_SHELF_TOP

        ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # 1. Establish the post-lift basin once.  Mirror the chains'
        # vertical lift from ``shelf_home`` to ``(home_xy, safe_z)``.
        shelf_home_arm = np.asarray(shelf_home, dtype=float)[:7]
        env.data.qpos[:7] = shelf_home_arm
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        home_ee = env.data.site_xpos[ee_site_id].copy()
        lift_target = np.array([home_ee[0], home_ee[1], safe_z])
        lift_path = lik.plan_joint_lerp(
            lift_target, _QUATS[0], n_substeps=16)
        if lift_path is None:
            # If we can't even lift from staging-home, the LUT can't
            # help.  Leave cache empty; runtime falls back to natural
            # mink.
            return {"precompute_s": time.perf_counter() - t0,
                    "n_reachable": 0, "n_cells": 0,
                    "error": "lift from shelf_home failed"}
        self._post_lift_qpos = np.asarray(lift_path[-1], dtype=float).copy()

        # 2. Probe every deck cell × every quat.
        n_cells = 0
        n_reachable = 0
        for ix in range(region_t.cells_x):
            for iy in range(region_t.cells_y):
                cell_id = f"shelf_top__{ix}_{iy}"
                cell = workspace.cell(cell_id)
                target_pos = np.asarray(workspace.pose_for(cell))
                col_x = float(target_pos[0])
                target_y = float(target_pos[1]) - GRASP_CONTACT_OFFSET
                traverse_target = np.array([col_x, target_y, safe_z])

                per_quat: List[Tuple[int, LUTEntry]] = []
                for q_idx, quat in enumerate(_QUATS):
                    env.data.qpos[:7] = self._post_lift_qpos
                    env.data.qvel[:] = 0.0
                    mujoco.mj_forward(env.model, env.data)
                    tq0 = time.perf_counter()
                    traj = lik.plan_to_pose(
                        traverse_target, quat,
                        n_substeps=self.n_substeps,
                        slerp_orientation=False,
                    )
                    dt_ms = (time.perf_counter() - tq0) * 1000.0
                    entry = LUTEntry(trajectory=traj, mink_ms=dt_ms)
                    self.cache[(cell_id, q_idx)] = entry
                    per_quat.append((q_idx, entry))
                    if traj is not None:
                        n_reachable += 1

                # Quat priority: feasible first (None last), then
                # ascending mink cost.
                self.quat_priority[cell_id] = [
                    qi for qi, _ in sorted(
                        per_quat,
                        key=lambda qe: (qe[1].trajectory is None,
                                          qe[1].mink_ms),
                    )
                ]
                n_cells += 1
                if verbose and n_cells % 10 == 0:
                    print(f"  [LUT] {n_cells} cells precomputed, "
                          f"{n_reachable} reachable so far")

        # Reset arm to shelf_home so caller's next operation starts clean.
        shelf_home_arm = np.asarray(shelf_home, dtype=float)[:7]
        env.data.qpos[:7] = shelf_home_arm
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        return {
            "precompute_s": time.perf_counter() - t0,
            "n_reachable": n_reachable,
            "n_cells": n_cells * len(_QUATS),
            "n_unique_cells": n_cells,
        }


# ---------------------------------------------------------------------------
# Runtime replay
# ---------------------------------------------------------------------------


def replay_cartesian_trajectory(
    env, lik: LinearIKPlanner, trajectory: List[np.ndarray],
) -> Optional[List[np.ndarray]]:
    """Validate ``trajectory`` against the current env's collision
    world.  Returns ``trajectory`` unchanged if every substep config
    is collision-free and joint-segment interpolation between
    consecutive configs is collision-free.  Returns ``None`` otherwise.

    Mirrors the inner-loop collision checks of
    ``LinearIKPlanner.plan_to_pose`` exactly — only the mink calls are
    skipped.  qpos / qvel are restored on exit (same contract as
    ``plan_to_pose``).
    """
    if trajectory is None or len(trajectory) < 2:
        return None

    qpos_save = env.data.qpos.copy()
    qvel_save = env.data.qvel.copy()
    try:
        prev_q = np.asarray(trajectory[0], dtype=float)
        # First config is the entry qpos — assume it's already
        # collision-free (the chain wouldn't have got here otherwise).
        for q_k in trajectory[1:]:
            q_k = np.asarray(q_k, dtype=float)
            if not env.is_collision_free(q_k):
                return None
            if not lik._segment_collision_free_n(
                prev_q, q_k, lik.joint_check_steps,
            ):
                return None
            prev_q = q_k
        return list(trajectory)
    finally:
        env.data.qpos[:] = qpos_save
        env.data.qvel[:] = qvel_save
        mujoco.mj_forward(env.model, env.data)
