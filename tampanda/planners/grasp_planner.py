"""Grasp pose planner for the Franka Panda gripper.

Gripper geometry reference (base_panda.xml):
  - attachment_site:        z = 0.0900 m in hand frame
  - fingertip_pad center:   z = 0.0584 (finger_base) + 0.04525 (pad_pos_z) = 0.1037 m
  - Effective contact offset from attachment_site along hand +Z: 0.0137 m
  - Fingers slide along hand Y axis; max opening = 0.08 m

Grasp-position formula
  contact_centre_world = attach_site_world + R @ [0, 0, CONTACT_OFFSET]
  → attach_site_world  = block_pos        − R @ [0, 0, CONTACT_OFFSET]

where R is the rotation matrix of the desired attachment_site orientation (body→world).
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# Fingertip contact centre offset from attachment_site along hand +Z (metres).
GRASP_CONTACT_OFFSET: float = 0.0137

# Distance from attachment_site to the lowest finger geom bottom (metres).
# Measured empirically with the Panda hand: 0.0226 m.
FINGER_TIP_BELOW: float = 0.0226

# Hand capsule radius in body-Z / body-X directions (from hand_capsule geom).
# Used for table-clearance checks in front-approach candidates.
HAND_CAPSULE_RADIUS: float = 0.040


class GraspType(Enum):
    TOP_DOWN_X = "top_down_x"  # Approach from above; fingers close along world ±X
    TOP_DOWN_Y = "top_down_y"  # Approach from above; fingers close along world ±Y
    FRONT      = "front"       # Approach from −Y (robot side); fingers along world ±X


# ---------------------------------------------------------------------------
# Precomputed quaternions (WXYZ, MuJoCo convention) with rotation matrices
#
# TOP_DOWN_X  →  R = [[0,1,0],[1,0,0],[0,0,−1]]
#   col-1 (body-Y in world) = [1,0,0]  → fingers along world +X
#   col-2 (body-Z in world) = [0,0,−1] → approach from above ✓
_QUAT_TOP_DOWN_X = np.array([0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])

# TOP_DOWN_Y  →  R = [[1,0,0],[0,−1,0],[0,0,−1]]
#   col-1 (body-Y in world) = [0,−1,0] → fingers along world ±Y
#   col-2 (body-Z in world) = [0, 0,−1] → approach from above ✓
_QUAT_TOP_DOWN_Y = np.array([0.0, 1.0, 0.0, 0.0])

# FRONT  →  R = [[0,1,0],[0,0,1],[1,0,0]]  (same as the legacy −0.5,0.5,0.5,0.5)
#   col-1 (body-Y in world) = [1,0,0] → fingers along world +X
#   col-2 (body-Z in world) = [0,1,0] → approach from −Y direction ✓
_QUAT_FRONT = np.array([-0.5, 0.5, 0.5, 0.5])

_QUATS: dict = {
    GraspType.TOP_DOWN_X: _QUAT_TOP_DOWN_X,
    GraspType.TOP_DOWN_Y: _QUAT_TOP_DOWN_Y,
    GraspType.FRONT:      _QUAT_FRONT,
}
# ---------------------------------------------------------------------------


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """WXYZ quaternion → 3×3 rotation matrix  (v_world = R @ v_body)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


@dataclass
class GraspCandidate:
    """A single grasp pose candidate for the Franka end-effector."""
    grasp_type:   GraspType
    grasp_quat:   np.ndarray   # desired attachment_site orientation (WXYZ)
    approach_pos: np.ndarray   # attachment_site world pos for pre-grasp standoff
    grasp_pos:    np.ndarray   # attachment_site world pos at grasp contact
    lift_pos:     np.ndarray   # attachment_site world pos for post-grasp lift
    score:        float = 0.0


class GraspPlanner:
    """Generate and rank grasp pose candidates for objects on a table.

    Usage::

        planner = GraspPlanner()
        candidates = planner.generate_candidates(block_pos, block_half_size)
        for c in candidates:           # sorted best-first
            path = rrt.plan_to_pose(c.approach_pos, c.grasp_quat, ...)
    """

    # Maximum gripper opening with a 5 % safety margin (Franka max = 0.08 m).
    MAX_GRIPPER_WIDTH: float = 0.076

    def __init__(
        self,
        approach_dist:        float = 0.12,
        lift_height:          float = 0.20,
        table_z:              float = 0.27,
        table_clearance:      float = 0.025,
    ):
        """
        Args:
            approach_dist:   Standoff distance (m) along −hand-Z from the grasp point.
            lift_height:     How far above the grasp point to lift the object (m).
            table_z:         World Z coordinate of the table surface.
            table_clearance: Minimum gap (m) between finger-tip bottom and table_z.
                             Should exceed the expected IK position error (~0.025 m).
        """
        self.approach_dist   = float(approach_dist)
        self.lift_height     = float(lift_height)
        self.table_z         = float(table_z)
        self.table_clearance = float(table_clearance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        block_pos:       np.ndarray,
        block_half_size: Optional[np.ndarray] = None,
        block_quat:      Optional[np.ndarray] = None,
    ) -> List[GraspCandidate]:
        """Return grasp candidates sorted best-first by score.

        Args:
            block_pos:       Block centre in world frame [x, y, z].
            block_half_size: Half-extents of the block [hx, hy, hz].
                             Defaults to a 4 cm cube if None.
            block_quat:      Block world orientation (WXYZ).  When provided the
                             top-down finger axes are aligned with the block's
                             local X and Y axes, minimising the effective gripper
                             width needed.  Ignored for FRONT grasps.
        """
        if block_half_size is None:
            block_half_size = np.array([0.02, 0.02, 0.02])

        candidates: List[GraspCandidate] = []
        for gtype in (GraspType.TOP_DOWN_X, GraspType.TOP_DOWN_Y, GraspType.FRONT):
            c = self._make_candidate(gtype, block_pos, block_half_size, block_quat)
            if c is not None:
                c.score = self._score(c, block_pos)
                candidates.append(c)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_candidate(
        self,
        gtype:           GraspType,
        block_pos:       np.ndarray,
        block_half_size: np.ndarray,
        block_quat:      Optional[np.ndarray] = None,
    ) -> Optional[GraspCandidate]:
        # For top-down grasps, align the finger axis with the block's local
        # horizontal axes when the block orientation is known.  This ensures the
        # gripper closes along the block's body-X or body-Y, keeping the
        # effective width minimal even for yaw-rotated blocks.
        if block_quat is not None and gtype in (GraspType.TOP_DOWN_X, GraspType.TOP_DOWN_Y):
            R_block  = quat_to_rotmat(block_quat)
            # block_x_h / block_y_h: horizontal (XY) block axes, normalised
            block_x_h = R_block[:, 0].copy(); block_x_h[2] = 0.0
            block_y_h = R_block[:, 1].copy(); block_y_h[2] = 0.0
            # Normalise (guard against near-zero after zeroing Z)
            nx = np.linalg.norm(block_x_h); ny = np.linalg.norm(block_y_h)
            if nx < 1e-6 or ny < 1e-6:
                block_quat = None   # fall back to world-aligned quaternion
            else:
                block_x_h /= nx; block_y_h /= ny
                if gtype == GraspType.TOP_DOWN_X:
                    finger_h = block_x_h   # close along block X
                else:
                    finger_h = block_y_h   # close along block Y
                # Build rotation matrix: body-Y → finger_h, body-Z → -world-Z
                approach_dir = np.array([0.0, 0.0, -1.0])    # hand Z in world
                hand_y  = finger_h
                hand_z  = approach_dir
                hand_x  = np.cross(hand_y, hand_z)
                hand_x /= np.linalg.norm(hand_x)
                R = np.column_stack([hand_x, hand_y, hand_z])
                # Convert rotation matrix to WXYZ quaternion
                quat = self._rotmat_to_quat(R)

        if block_quat is None or gtype == GraspType.FRONT:
            quat = _QUATS[gtype]
            R    = quat_to_rotmat(quat)

        # ---- gripper-opening feasibility ---------------------------------
        # Project the rotated block bounding box onto the finger axis to get
        # the true effective width (accounts for yaw-rotated blocks).
        finger_axis_world = R[:, 1]
        if block_quat is not None and gtype in (GraspType.TOP_DOWN_X, GraspType.TOP_DOWN_Y):
            R_block = quat_to_rotmat(block_quat)
            # Support function: max projection of any vertex onto finger_axis
            half = block_half_size
            max_proj = sum(abs(np.dot(R_block[:, i], finger_axis_world)) * half[i]
                           for i in range(3))
            block_width_in_finger_dir = 2.0 * max_proj
        else:
            block_width_in_finger_dir = 2.0 * float(
                np.dot(np.abs(finger_axis_world), block_half_size)
            )
        if block_width_in_finger_dir > self.MAX_GRIPPER_WIDTH:
            return None

        # ---- grasp position ----------------------------------------------
        # Place attachment_site so the finger contact centre lands on block_pos.
        hand_z_world         = R[:, 2]
        contact_offset_world = hand_z_world * GRASP_CONTACT_OFFSET
        grasp_pos            = block_pos - contact_offset_world

        # ---- finger-tip table-clearance floor (all top-down grasps) ----------
        # Finger tips extend FINGER_TIP_BELOW metres below the attachment_site.
        # If the IK error pushes the EE down, those tips can collide with the table.
        # Raise grasp_pos.z so that even with IK error there is table_clearance margin.
        if gtype in (GraspType.TOP_DOWN_X, GraspType.TOP_DOWN_Y):
            min_grasp_z = self.table_z + FINGER_TIP_BELOW + self.table_clearance
            if grasp_pos[2] < min_grasp_z:
                # Check the raised contact point is still within the block.
                raised_contact_z = min_grasp_z + hand_z_world[2] * GRASP_CONTACT_OFFSET
                block_top_z = block_pos[2] + block_half_size[2]
                if raised_contact_z > block_top_z - 0.005:
                    return None   # block too short to grasp without table collision
                grasp_pos = grasp_pos.copy()
                grasp_pos[2] = min_grasp_z

        # ---- table-clearance check for front approach --------------------
        if gtype == GraspType.FRONT:
            min_safe_z = self.table_z + HAND_CAPSULE_RADIUS + 0.01
            if grasp_pos[2] < min_safe_z:
                # Try to raise the grasp to just clear the table.
                raise_z   = min_safe_z - grasp_pos[2]
                grasp_pos = grasp_pos + np.array([0.0, 0.0, raise_z])
                block_top = block_pos[2] + block_half_size[2]
                # Discard if the raised grasp is above the block entirely.
                if grasp_pos[2] > block_top - 0.005:
                    return None

        # ---- pre-grasp (approach) position -------------------------------
        # Move attachment_site along −hand-Z by approach_dist.
        approach_pos = grasp_pos - hand_z_world * self.approach_dist

        # ---- lift position -----------------------------------------------
        lift_pos = grasp_pos + np.array([0.0, 0.0, self.lift_height])

        return GraspCandidate(
            grasp_type=gtype,
            grasp_quat=quat.copy(),
            approach_pos=approach_pos,
            grasp_pos=grasp_pos,
            lift_pos=lift_pos,
        )

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """3×3 rotation matrix → WXYZ quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def _score(self, c: GraspCandidate, block_pos: np.ndarray) -> float:
        """Heuristic score — higher is better."""
        score = 0.0

        # Top-down grasps are more stable for objects on flat surfaces.
        if c.grasp_type in (GraspType.TOP_DOWN_X, GraspType.TOP_DOWN_Y):
            score += 20.0

        # Prefer approach positions comfortably inside robot workspace.
        xy_dist = float(np.linalg.norm(c.approach_pos[:2]))
        if 0.20 <= xy_dist <= 0.75:
            score += 5.0

        # Penalise approaches that risk table collision.
        if c.approach_pos[2] < self.table_z + 0.05:
            score -= 15.0

        return score
