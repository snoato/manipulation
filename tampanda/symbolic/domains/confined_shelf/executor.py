"""Domain-local executor for confined_shelf.

``PickPlaceExecutor.place`` plans its approach/descent/retreat with
``self.planner.plan_to_pose`` — and the shared planner is RRT*, so every
FULL put pays full RRT* (~1.6 s) even though ``pick`` resolves most moves
on the cheap LinearIK ladder (~0.3 s).

Rather than edit the shared ``PickPlaceExecutor`` / ``RRTStar`` (which
other domains rely on), we subclass ``RRTStar`` here so its
``plan_to_pose`` tries the **same joint-lerp -> Cartesian-linear-IK
ladder pick uses**, falling back to real RRT* only when the straight
paths clip.  Swapping this planner into the executor upgrades place's
three planning calls (approach, descent, retreat) without touching any
shared module — keeping parallel sessions on other domains safe.
"""
from __future__ import annotations

import numpy as np

from tampanda.planners.grasp_planner import GraspPlanner, GraspType
from tampanda.planners.linear_ik import LinearIKPlanner
from tampanda.planners.pick_place import PickPlaceExecutor
from tampanda.planners.rrt_star import RRTStar

_LERP_STEP_M = 0.02         # Cartesian waypoint spacing for the dense rung
_RRT_FALLBACK_ITERS = 1500  # tight cap so the fallback can't thrash (was
                            # blowing up to 140 s on edge cells); a cell that
                            # needs more than this is treated as infeasible


class LadderRRT(RRTStar):
    """RRT* whose ``plan_to_pose`` tries the cheap LinearIK ladder first.

    Order:
      1. ``plan_joint_lerp`` — IK once + joint-space straight line (cheapest).
      2. ``plan_to_pose`` — Cartesian straight line with **~2 cm waypoints**
         (``_LERP_STEP_M``), IK seeded per step.  The fine spacing lets
         edge/deep reaches (e.g. the left-edge column) resolve here instead
         of falling through to the flaky RRT*.
      3. real RRT* — curved, correctness-preserving last resort.

    Same-orientation confined-shelf FRONT moves resolve on rung 1 or 2 for
    essentially all cells; RRT* should now rarely fire.
    """

    def __init__(self, env, lik: LinearIKPlanner, **rrt_kwargs):
        super().__init__(env, **rrt_kwargs)
        self._lik = lik

    def plan_to_pose(self, target_pos, target_quat, dt: float = 0.01,
                     max_iterations: Optional[int] = None,
                     max_ik_retries: int = 3):
        path = self._lik.plan_joint_lerp(target_pos, target_quat, dt=dt)
        if path is not None:
            return path
        # Dense Cartesian waypoints (~2 cm), from the current EE pose.
        cur_ee = self.data.site_xpos[self._lik._ee_site_id]
        dist = float(np.linalg.norm(np.asarray(target_pos, float) - cur_ee))
        n_sub = max(8, int(np.ceil(dist / _LERP_STEP_M)))
        path = self._lik.plan_to_pose(target_pos, target_quat, dt=dt,
                                      n_substeps=n_sub, slerp_orientation=False)
        if path is not None:
            return path
        # Tightly-bounded RRT* last resort (never thrash).
        cap = min(max_iterations or _RRT_FALLBACK_ITERS, _RRT_FALLBACK_ITERS)
        return super().plan_to_pose(target_pos, target_quat, dt=dt,
                                    max_iterations=cap, max_ik_retries=1)


class ConfinedShelfExecutor(PickPlaceExecutor):
    """PickPlaceExecutor with confined-shelf-tuned ``place`` defaults.

    Two defaults differ from the shared executor:

    * **8 cm approach standoff** (vs 15 cm).  Combined with the LadderRRT
      planner that pushes the held bottle in along a straight path, a 15 cm
      push-in clips near the side walls on a few cells and falls through to
      a slow RRT* descent (seconds → minutes).  An 8 cm standoff (matching
      the pick approach) keeps the descent short so the joint-lerp resolves
      it directly.

    * **8 mm place clearance** (vs 3 mm).  At 3 mm the held cylinder's
      bottom sits only ~0.8-2.1 mm above the shelf floor (the EE→cylinder
      offset eats ~1-2 mm, more at deep cells where the IK config tilts the
      bottle), and the joint-space descent dips a fraction of a millimetre
      mid-path — enough that the bottle grazes ``shelf_floor`` by a few
      microns and *every* planning rung rejects the descent.  An 8 mm
      clearance lands the bottle bottom ~5-7 mm up so the descent clears
      with margin even at the worst cell (the deep right-column (7,3),
      whose tilt drops the bottle ~2 mm); the gripper opens those few mm
      above the floor and physics settles the bottle down on release (a
      flat-bottomed cylinder dropping <7 mm cannot topple).

    Only the defaults change; the placement logic is the shared
    implementation.
    """

    def place(self, block_name, place_block_center, ee_quat=None,
              target_block_name=None, approach_height: float = 0.08,
              place_clearance: float = 0.008, retreat_lift=None) -> bool:
        return super().place(
            block_name, place_block_center, ee_quat=ee_quat,
            target_block_name=target_block_name,
            approach_height=approach_height, place_clearance=place_clearance,
            retreat_lift=retreat_lift,
        )


def build_confined_shelf_executor(
    env,
    table_z: float,
    *,
    lik: Optional[LinearIKPlanner] = None,
    max_iterations: int = 12000,
) -> PickPlaceExecutor:
    """Build the confined_shelf PickPlaceExecutor with the ladder planner.

    FRONT-only grasps (closed-top cubicle), tight 8 cm standoff, short
    4 cm lift (fits the interior).  The shared ``linear_ik_planner`` is
    reused by both the executor (descent/lift) and ``LadderRRT`` (place
    approach), so they convergence-seed identically.
    """
    env.ik.pos_threshold = 0.005
    env.ik.ori_threshold = 5e-3
    lik = lik or LinearIKPlanner(env, n_substeps=8, joint_check_steps=8)
    rrt = LadderRRT(env, lik, max_iterations=max_iterations)
    grasp = GraspPlanner(
        table_z=table_z, allowed_types=[GraspType.FRONT],
        approach_dist=0.08, lift_height=0.04,
    )
    return ConfinedShelfExecutor(env, rrt, grasp, use_attachment=True,
                                 max_plan_iters=max_iterations,
                                 linear_ik_planner=lik)
