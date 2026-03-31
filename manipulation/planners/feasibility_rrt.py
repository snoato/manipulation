"""Fast bidirectional RRT (RRT-Connect) for feasibility checking.

Use this in place of RRTStar when you only need to know *whether* a
collision-free path exists, not its quality.  It is a drop-in for
RRTStar: same attribute names, same plan / plan_to_pose / smooth_path
interface.

Optimisations over RRTStar
--------------------------
1. **No rewiring** — plain RRT; O(1) node insertion vs O(k) for RRT*.
2. **Bidirectional search (RRT-Connect)** — grows two trees simultaneously
   and greedily connects them; finds paths in roughly √n iterations.
3. **Vectorised nearest-neighbour** — numpy argmin on a contiguous
   (n, 7) float64 array; ~50× faster than a Python loop.
   (A KD-tree requires an O(n log n) rebuild per insertion — for the
   tree sizes here the numpy scan is strictly faster.)
4. **Binary-subdivision edge check** — checks the midpoint of an edge
   first, then recursively halves; finds obstacles near the centre with
   fewer mj_forward calls on average.  The start-point of each edge is
   skipped because the parent node is already known collision-free.

Parameter correspondence with RRTStar
--------------------------------------
step_size, goal_threshold, collision_check_steps — identical semantics.
Set them to match the RRTStar instance used for execution so that a
feasibility-OK result here implies the same for the motion planner.
max_iterations — same budget; expect earlier termination in practice.
goal_sample_rate, search_radius — accepted for API compatibility, unused.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np


class FeasibilityRRT:
    """Bidirectional RRT (RRT-Connect) optimised for existence queries."""

    # Class-level cache: steps -> check-order array (shared across instances)
    _order_cache: dict[int, np.ndarray] = {}

    def __init__(
        self,
        environment,
        step_size: float = 0.1,
        goal_threshold: float = 0.05,
        collision_check_steps: int = 5,
        max_iterations: int = 1000,
    ) -> None:
        self.env = environment

        self.step_size             = step_size
        self.goal_threshold        = goal_threshold
        self.collision_check_steps = collision_check_steps
        self.max_iterations        = max_iterations

        # Accepted for API compat with RRTStar; not used by this planner.
        self.goal_sample_rate = 0.1
        self.search_radius    = 0.5

        self.joint_limits_low  = environment.model.jnt_range[:7, 0].copy()
        self.joint_limits_high = environment.model.jnt_range[:7, 1].copy()

    # ------------------------------------------------------------------
    # Binary subdivision order (class-level cache, keyed by steps)
    # ------------------------------------------------------------------

    @classmethod
    def _get_check_order(cls, steps: int) -> np.ndarray:
        if steps not in cls._order_cache:
            cls._order_cache[steps] = cls._make_binary_order(steps)
        return cls._order_cache[steps]

    @staticmethod
    def _make_binary_order(steps: int) -> np.ndarray:
        """Return indices 1..steps in binary-subdivision order.

        Index 0 (the edge start) is excluded — the parent config is
        already verified collision-free by the previous iteration.

        Example (steps=5): [3, 1, 4, 2, 5]
        The midpoint (3) is checked first; collisions near the centre
        of an edge are detected after fewer mj_forward calls on average.
        """
        order: list[int] = []
        queue: list[tuple[int, int]] = [(1, steps)]
        while queue:
            lo, hi = queue.pop(0)
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            order.append(mid)
            queue.append((lo, mid - 1))
            queue.append((mid + 1, hi))
        return np.array(order, dtype=np.int32)

    # ------------------------------------------------------------------
    # Internal tree (numpy-backed for vectorised nearest-neighbour)
    # ------------------------------------------------------------------

    class _Tree:
        """Preallocated tree backed by contiguous NumPy arrays.

        Doubles capacity on overflow so the nearest-neighbour slice
        always operates on a contiguous block.
        """

        __slots__ = ("configs", "parents", "n")

        def __init__(self, root: np.ndarray, capacity: int) -> None:
            self.configs = np.empty((capacity, root.shape[0]), dtype=np.float64)
            self.parents = np.full(capacity, -1, dtype=np.int32)
            self.configs[0] = root
            self.n = 1

        def _grow(self) -> None:
            new_cap  = len(self.configs) * 2
            new_cfg  = np.empty((new_cap, self.configs.shape[1]), dtype=np.float64)
            new_par  = np.full(new_cap, -1, dtype=np.int32)
            new_cfg[: self.n] = self.configs[: self.n]
            new_par[: self.n] = self.parents[: self.n]
            self.configs = new_cfg
            self.parents = new_par

        def add(self, config: np.ndarray, parent_idx: int) -> int:
            if self.n >= len(self.configs):
                self._grow()
            idx = self.n
            self.configs[idx] = config
            self.parents[idx] = parent_idx
            self.n += 1
            return idx

        def nearest_idx(self, query: np.ndarray) -> int:
            """Vectorised O(n) nearest-neighbour — no sqrt needed."""
            diff = self.configs[: self.n] - query
            sq_dists = np.einsum("ij,ij->i", diff, diff)
            return int(np.argmin(sq_dists))

        def trace(self, node_idx: int) -> list[np.ndarray]:
            """Walk from node to root; returns [node, ..., root]."""
            path: list[np.ndarray] = []
            idx = node_idx
            while idx != -1:
                path.append(self.configs[idx].copy())
                idx = int(self.parents[idx])
            return path

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _is_edge_free(self, c1: np.ndarray, c2: np.ndarray) -> bool:
        """Check edge c1→c2 using binary-subdivision order.

        State save/restore is NOT done here — the caller wraps the full
        planning loop in a single save/restore.  c1 is not re-checked
        (parent node is already known collision-free).
        """
        steps  = self.collision_check_steps
        order  = self._get_check_order(steps)
        is_cf  = self.env.is_collision_free_no_restore
        inv    = 1.0 / steps
        delta  = c2 - c1
        for idx in order:
            if not is_cf(c1 + (idx * inv) * delta):
                return False
        return True

    # ------------------------------------------------------------------
    # Public API — mirrors RRTStar
    # ------------------------------------------------------------------

    def plan(
        self,
        start_config: np.ndarray,
        goal_config:  np.ndarray,
        max_iterations: Optional[int] = None,
    ) -> Optional[list[np.ndarray]]:
        """RRT-Connect bidirectional search.

        Returns a joint-config path from start to goal, or None if no
        path was found within the iteration budget.
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        # Check start/goal (these save/restore internally)
        if not self.env.is_collision_free(start_config):
            return None
        if not self.env.is_collision_free(goal_config):
            return None

        # Each tree can receive up to ~max_iterations extensions from the
        # extend step plus greedy-connect additions from the other tree.
        # In the worst case the connect traverses the full config-space
        # diameter; 3× gives a generous safety margin.
        capacity = max_iterations * 3 + 10

        tree_s = self._Tree(start_config, capacity)
        tree_g = self._Tree(goal_config,  capacity)

        # Save MuJoCo state once; all edge checks use the no-restore
        # variant throughout; we restore at the very end.
        data      = self.env.data
        model     = self.env.model
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        lo_lim = self.joint_limits_low
        hi_lim = self.joint_limits_high
        step   = self.step_size
        thr    = self.goal_threshold

        tree_a, tree_b = tree_s, tree_g
        a_is_start     = True
        path: Optional[list[np.ndarray]] = None

        for _ in range(max_iterations):
            # ── Extend tree_a toward a random configuration ──────────
            q_rand     = np.random.uniform(lo_lim, hi_lim)
            near_a_idx = tree_a.nearest_idx(q_rand)
            near_a     = tree_a.configs[near_a_idx]

            diff_a = q_rand - near_a
            dist_a = np.linalg.norm(diff_a)
            q_new  = (near_a + diff_a * (step / dist_a)
                      if dist_a > step else q_rand.copy())

            if not self._is_edge_free(near_a, q_new):
                tree_a, tree_b = tree_b, tree_a
                a_is_start = not a_is_start
                continue

            new_a_idx = tree_a.add(q_new, near_a_idx)

            # ── Greedy connect: grow tree_b toward q_new ─────────────
            # After each step we continue from the newly added node
            # (always closer to q_new), so we never need to call
            # nearest_idx again inside this loop.
            near_b_idx = tree_b.nearest_idx(q_new)

            while True:
                near_b = tree_b.configs[near_b_idx]
                diff_b = q_new - near_b
                dist_b = np.linalg.norm(diff_b)

                if dist_b < thr:
                    # Trees joined — assemble the full path
                    path = self._build_path(
                        tree_a, new_a_idx,
                        tree_b, near_b_idx,
                        a_is_start,
                    )
                    break

                # Clamp to q_new when within one step — avoids infinite
                # oscillation from overshooting the target.
                alpha = min(step / dist_b, 1.0)
                q_ext = near_b + diff_b * alpha
                if not self._is_edge_free(near_b, q_ext):
                    break
                near_b_idx = tree_b.add(q_ext, near_b_idx)

            if path is not None:
                break

            # Alternate which tree does the random extend
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start

        # Restore MuJoCo state
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)

        return path

    def plan_to_pose(
        self,
        target_pos:  np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.01,
        max_iterations: Optional[int] = None,
        max_ik_retries: int = 3,
    ) -> Optional[list[np.ndarray]]:
        """IK-solve target pose, then plan with RRT-Connect.

        Mirrors RRTStar.plan_to_pose exactly so callers are interchangeable.
        """
        start_config = self.env.data.qpos[:7].copy()

        for attempt in range(max_ik_retries):
            if attempt == 0:
                self.env.ik.update_configuration(self.env.data.qpos)
            else:
                neutral  = (self.joint_limits_low + self.joint_limits_high) / 2.0
                alpha    = 0.3 * attempt
                fallback = (1.0 - alpha) * start_config + alpha * neutral
                self.env.data.qpos[:7] = fallback
                mujoco.mj_forward(self.env.model, self.env.data)
                self.env.ik.update_configuration(self.env.data.qpos)

            self.env.ik.set_target_position(target_pos, target_quat)
            if self.env.ik.converge_ik(dt):
                goal_config = self.env.ik.configuration.q[:7].copy()
                self.env.data.qpos[:7] = start_config
                mujoco.mj_forward(self.env.model, self.env.data)
                # Continue to next IK seed if planning fails — different seeds
                # produce different joint-space goals with different reachability.
                path = self.plan(start_config, goal_config, max_iterations)
                if path is not None:
                    return path

        self.env.data.qpos[:7] = start_config
        mujoco.mj_forward(self.env.model, self.env.data)
        return None

    def smooth_path(
        self,
        path: list[np.ndarray],
        max_iterations: int = 100,
    ) -> list[np.ndarray]:
        """No-op: path quality is irrelevant for feasibility checking."""
        return path

    # ------------------------------------------------------------------
    # Path assembly
    # ------------------------------------------------------------------

    def _build_path(
        self,
        tree_a: "_Tree", node_a_idx: int,
        tree_b: "_Tree", node_b_idx: int,
        a_is_start: bool,
    ) -> list[np.ndarray]:
        """Concatenate traces from both trees into start→goal order.

        trace_a = [node_a, ..., root_a]  (walk from meeting node to root)
        trace_b = [node_b, ..., root_b]

        a_is_start=True:  root_a=start, root_b=goal
            path = reversed(trace_a) + trace_b
                 = [start, ..., node_a]  +  [node_b, ..., goal]

        a_is_start=False: root_a=goal, root_b=start
            path = reversed(trace_b) + trace_a
                 = [start, ..., node_b]  +  [node_a, ..., goal]
        """
        trace_a = tree_a.trace(node_a_idx)
        trace_b = tree_b.trace(node_b_idx)

        if a_is_start:
            return list(reversed(trace_a)) + trace_b
        else:
            return list(reversed(trace_b)) + trace_a
