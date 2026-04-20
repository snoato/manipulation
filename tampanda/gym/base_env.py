"""TampandaGymEnv — Gymnasium wrapper around FrankaEnvironment."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import gymnasium

from tampanda.gym.spaces import build_observation_space, build_action_space
from tampanda.perception.mujoco_camera import MujocoCamera


# Franka Panda joint limits (radians)
_Q_LOW  = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0175, -2.9671], np.float32)
_Q_HIGH = np.array([ 2.9671,  1.8326,  2.9671, -0.0698,  2.9671,  3.7525,  2.9671], np.float32)

# Per-step action scales
_JOINT_DELTA_SCALE = 0.05   # radians
_EE_POS_SCALE      = 0.02   # metres
_EE_ROT_SCALE      = 0.1    # radians


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [w, x, y, z] (Shepperd's method)."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (mat[2, 1] - mat[1, 2]) * s,
                         (mat[0, 2] - mat[2, 0]) * s,
                         (mat[1, 0] - mat[0, 1]) * s], np.float32)
    if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        return np.array([(mat[2, 1] - mat[1, 2]) / s, 0.25 * s,
                         (mat[0, 1] + mat[1, 0]) / s,
                         (mat[0, 2] + mat[2, 0]) / s], np.float32)
    if mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        return np.array([(mat[0, 2] - mat[2, 0]) / s,
                         (mat[0, 1] + mat[1, 0]) / s, 0.25 * s,
                         (mat[1, 2] + mat[2, 1]) / s], np.float32)
    s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
    return np.array([(mat[1, 0] - mat[0, 1]) / s,
                     (mat[0, 2] + mat[2, 0]) / s,
                     (mat[1, 2] + mat[2, 1]) / s, 0.25 * s], np.float32)


def _infer_object_names(model) -> List[str]:
    """Return body names of all free-joint objects (manipulable scene objects)."""
    names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jnt_name and jnt_name.endswith("_freejoint"):
                names.append(jnt_name[: -len("_freejoint")])
    return names


def _pad_points(pts: np.ndarray, n: int, n_cols: int = 3) -> np.ndarray:
    """Pad or subsample an (M, n_cols) pointcloud to exactly (n, n_cols)."""
    if len(pts) == 0:
        return np.zeros((n, n_cols), np.float32)
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx].astype(np.float32)
    repeats = n // len(pts) + 1
    return np.tile(pts, (repeats, 1))[:n].astype(np.float32)


class TampandaGymEnv(gymnasium.Env):
    """Gymnasium environment wrapping a TAMPanda ``FrankaEnvironment``.

    Args:
        scene: An ``ArmSceneBuilder`` (or subclass) instance.  ``build_env()``
            is called internally, so the builder must not have been built yet.
        obs: Observation keys to include.  Any subset of:
            ``"joints"``, ``"joint_vel"``, ``"ee_pose"``, ``"object_poses"``,
            ``"rgb"``, ``"depth"``, ``"pointcloud"``, ``"segmented_pointcloud"``,
            ``"multi_pointcloud"``.
        action_space_type: ``"joint_delta"`` (default), ``"joint_target"``, or
            ``"cartesian_delta"``.  All produce a ``[-1, 1]`` Box.
        include_gripper: Append one gripper dim (``-1`` = close, ``1`` = open).
        reward_fn: Built-in name (``"dense_grasp"``, ``"sparse_grasp"``,
            ``"dense_ee_distance"``) or a callable ``(sim, sym_state) -> float``.
        cameras: Camera names used for visual observations (must exist in the
            built scene).  First camera is used for single-camera obs keys.
        image_size: ``(height, width)`` in pixels.
        pointcloud_n_points: Fixed point count after downsampling / padding.
        pointcloud_filter: Body-name pattern passed as ``exclude_patterns`` to
            ``get_segmented_pointcloud`` (e.g. ``"table"``).
        bridge_factory: Callable ``() -> DomainBridge`` — called fresh on every
            ``reset()`` so parallel workers never share bridge state.
        bridge_objects: ``{type: [names]}`` dict forwarded to ``ground_state``
            and ``plan``.
        bridge_goals: Goal predicate list for ``bridge.plan()``.  If provided,
            a symbolic plan is computed on every ``reset()`` and surfaced in
            ``info["symbolic_plan"]``.
        object_names: Explicit list of object body names for the
            ``"object_poses"`` observation and reward computation.  Inferred
            from free joints in the scene if ``None``.
        max_episode_steps: Episode truncation horizon.
        n_substeps: Number of raw physics steps per ``gym.step()`` call.
        render_mode: ``"rgb_array"`` or ``None``.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scene,
        obs: List[str] = ("joints", "ee_pose"),
        action_space_type: str = "joint_delta",
        include_gripper: bool = True,
        reward_fn = "dense_grasp",
        cameras: List[str] = ("workspace",),
        image_size: Tuple[int, int] = (84, 84),
        pointcloud_n_points: int = 1024,
        pointcloud_filter: Optional[str] = None,
        bridge_factory: Optional[Callable] = None,
        bridge_objects: Optional[Dict] = None,
        bridge_goals: Optional[List] = None,
        object_names: Optional[List[str]] = None,
        max_episode_steps: int = 500,
        n_substeps: int = 5,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self._obs_keys = list(obs)
        self._action_space_type = action_space_type
        self._include_gripper = include_gripper
        self._reward_fn = reward_fn
        self._cameras = list(cameras)
        self._image_size = image_size
        self._n_pts = pointcloud_n_points
        self._pcloud_filter = pointcloud_filter
        self._bridge_factory = bridge_factory
        self._bridge_objects = bridge_objects or {}
        self._bridge_goals = bridge_goals
        self._max_episode_steps = max_episode_steps
        self._n_substeps = n_substeps
        self.render_mode = render_mode

        # Build the underlying FrankaEnvironment (rate limit disabled for RL)
        self._sim = scene.build_env(rate=10_000.0)

        # Resolve object names
        self._object_names: List[str] = (
            object_names if object_names is not None
            else _infer_object_names(self._sim.get_model())
        )

        # Cache EE site id
        self._ee_site_id: int = mujoco.mj_name2id(
            self._sim.get_model(), mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        # Gymnasium spaces
        self.observation_space = build_observation_space(
            self._obs_keys,
            n_objects=len(self._object_names),
            image_size=self._image_size,
            n_points=self._n_pts,
        )
        self.action_space = build_action_space(action_space_type, include_gripper)

        # Lazy-initialised camera helper (renderer created on first use)
        self._mj_camera = MujocoCamera(
            self._sim,
            width=image_size[1],
            height=image_size[0],
        )

        # Episode state
        self._bridge = None
        self._current_plan: Optional[List] = None
        self._plan_step: int = 0
        self._step_count: int = 0
        self._initial_obj_z: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._sim.reset()

        # Fresh bridge instance per episode (satisfies Option-A contract)
        if self._bridge_factory is not None:
            self._bridge = self._bridge_factory()
            self._bridge.env = self._sim
            self._bridge._fluent_state.clear()
        else:
            self._bridge = None

        # Allow objects to settle before recording initial heights
        for _ in range(50):
            self._physics_step()

        self._initial_obj_z = {
            name: float(self._sim.get_object_position(name)[2])
            for name in self._object_names
        }

        # Symbolic plan (optional)
        self._current_plan = None
        self._plan_step = 0
        if self._bridge is not None and self._bridge_goals is not None:
            try:
                self._current_plan = self._bridge.plan(
                    self._bridge_objects, self._bridge_goals
                )
            except Exception:
                pass

        self._step_count = 0
        obs = self._get_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self._n_substeps):
            self._physics_step()

        self._step_count += 1

        sym_state = None
        if self._bridge is not None:
            sym_state = self._bridge.ground_state(self._bridge_objects)

        obs = self._get_obs()
        reward = float(self._compute_reward(sym_state))
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_episode_steps

        info = self._build_info(sym_state)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or not self._cameras:
            return None
        h, w = self._image_size
        return self._mj_camera.render_rgb(self._cameras[0], w, h)

    def close(self):
        self._mj_camera.close()
        if getattr(self._sim, "viewer", None) is not None:
            self._sim.viewer.close()

    # ------------------------------------------------------------------
    # Physics stepping (rate-limit free)
    # ------------------------------------------------------------------

    def _physics_step(self):
        """Advance one MuJoCo timestep without wall-clock rate limiting."""
        sim = self._sim
        # Honour kinematic attachment if an object is being held
        if getattr(sim, "_attached", None) is not None:
            sim._apply_attachment()
        mujoco.mj_step(sim.model, sim.data)
        sim.sim_time += sim.rate.dt

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray):
        data = self._sim.data

        if self._action_space_type == "joint_delta":
            current = data.qpos[:7].copy()
            target = np.clip(
                current + action[:7] * _JOINT_DELTA_SCALE, _Q_LOW, _Q_HIGH
            )
            data.ctrl[:7] = target

        elif self._action_space_type == "joint_target":
            # action in [-1, 1] linearly mapped to [Q_LOW, Q_HIGH]
            target = _Q_LOW + (action[:7] + 1.0) / 2.0 * (_Q_HIGH - _Q_LOW)
            data.ctrl[:7] = np.clip(target, _Q_LOW, _Q_HIGH)

        elif self._action_space_type == "cartesian_delta":
            self._apply_cartesian_delta(
                action[:3] * _EE_POS_SCALE,
                action[3:6] * _EE_ROT_SCALE,
            )

        if self._include_gripper:
            # [-1, 1] → [0, 255] (0=closed, 255=open for Franka actuator)
            gripper_val = (float(action[-1]) + 1.0) / 2.0 * 255.0
            data.ctrl[7] = gripper_val

    def _apply_cartesian_delta(self, pos_delta: np.ndarray, rot_delta: np.ndarray):
        """Solve a short IK burst for EE_pose + delta, then commit to ctrl."""
        import mink

        data = self._sim.data
        model = self._sim.get_model()
        ik = self._sim.get_ik()

        # Current EE pose
        ee_pos = data.site_xpos[self._ee_site_id].copy()
        ee_mat = data.site_xmat[self._ee_site_id].reshape(3, 3).copy()

        target_pos = ee_pos + pos_delta

        # Apply axis-angle rotation delta
        angle = float(np.linalg.norm(rot_delta))
        if angle > 1e-6:
            axis = rot_delta / angle
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            target_mat = R_delta @ ee_mat
        else:
            target_mat = ee_mat

        target_quat = _mat2quat(target_mat)  # [w, x, y, z]

        # Point mocap body at target so IK task picks it up
        ik.set_target_position(target_pos, target_quat)
        mujoco.mj_forward(model, data)

        # Short IK solve (20 iters — fast and good enough for RL)
        ik.update_configuration(data.qpos)
        T_wt = mink.SE3.from_mocap_name(model, data, ik.target_name)
        ik.ee_task.set_target(T_wt)
        ik.posture_task.set_target_from_configuration(ik.configuration)
        dt = 1.0 / 200.0
        for _ in range(20):
            vel = mink.solve_ik(ik.configuration, ik.tasks(), dt, ik.solver, 1e-3)
            ik.configuration.integrate_inplace(vel, dt)

        q_sol = np.clip(ik.configuration.q[:7], _Q_LOW, _Q_HIGH)
        data.ctrl[:7] = q_sol.astype(np.float64)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        data = self._sim.data
        cam = self._cameras[0] if self._cameras else None
        h, w = self._image_size

        if "joints" in self._obs_keys:
            obs["joints"] = data.qpos[:7].astype(np.float32).copy()

        if "joint_vel" in self._obs_keys:
            obs["joint_vel"] = data.qvel[:7].astype(np.float32).copy()

        if "ee_pose" in self._obs_keys:
            ee_pos = data.site_xpos[self._ee_site_id].copy()
            ee_mat = data.site_xmat[self._ee_site_id].reshape(3, 3)
            obs["ee_pose"] = np.concatenate([ee_pos, _mat2quat(ee_mat)]).astype(np.float32)

        if "object_poses" in self._obs_keys:
            parts = []
            for name in self._object_names:
                parts.append(self._sim.get_object_position(name).astype(np.float32))
                parts.append(self._sim.get_object_orientation(name).astype(np.float32))
            obs["object_poses"] = (
                np.concatenate(parts) if parts else np.zeros((0,), np.float32)
            )

        if "rgb" in self._obs_keys and cam:
            obs["rgb"] = self._mj_camera.render_rgb(cam, w, h)

        if "depth" in self._obs_keys and cam:
            obs["depth"] = self._mj_camera.render_depth(cam, w, h).astype(np.float32)

        if "pointcloud" in self._obs_keys and cam:
            pts, _ = self._mj_camera.get_pointcloud(cam, num_samples=self._n_pts)
            obs["pointcloud"] = _pad_points(pts, self._n_pts, 3)

        if "segmented_pointcloud" in self._obs_keys and cam:
            exclude = [self._pcloud_filter] if self._pcloud_filter else None
            clouds = self._mj_camera.get_segmented_pointcloud(
                cam, num_samples=self._n_pts, exclude_patterns=exclude
            )
            xyzl_parts = []
            for label_i, (pts, _) in enumerate(clouds.values()):
                labels = np.full((len(pts), 1), float(label_i), np.float32)
                xyzl_parts.append(np.hstack([pts.astype(np.float32), labels]))
            merged = np.vstack(xyzl_parts) if xyzl_parts else np.zeros((0, 4), np.float32)
            obs["segmented_pointcloud"] = _pad_points(merged, self._n_pts, 4)

        if "multi_pointcloud" in self._obs_keys:
            n_per_cam = max(1, self._n_pts // len(self._cameras))
            clouds = self._mj_camera.get_multi_camera_segmented_pointcloud(
                self._cameras, total_samples_per_object=n_per_cam
            )
            all_pts = [pts for pts, _ in clouds.values()]
            merged = np.vstack(all_pts) if all_pts else np.zeros((0, 3), np.float32)
            obs["multi_pointcloud"] = _pad_points(merged, self._n_pts, 3)

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, sym_state) -> float:
        if callable(self._reward_fn):
            return float(self._reward_fn(self._sim, sym_state))
        if self._reward_fn == "dense_grasp":
            return self._reward_dense_grasp()
        if self._reward_fn == "sparse_grasp":
            return self._reward_sparse_grasp()
        if self._reward_fn == "dense_ee_distance":
            return self._reward_dense_ee_distance()
        raise ValueError(f"Unknown reward_fn: {self._reward_fn!r}")

    def _reward_dense_grasp(self) -> float:
        if not self._object_names:
            return 0.0
        ee_pos = self._sim.data.site_xpos[self._ee_site_id]
        obj_pos = self._sim.get_object_position(self._object_names[0])
        dist = float(np.linalg.norm(ee_pos - obj_pos))
        init_z = self._initial_obj_z.get(self._object_names[0], 0.0)
        lift = max(0.0, float(obj_pos[2]) - init_z)
        return -dist + lift * 10.0

    def _reward_sparse_grasp(self) -> float:
        if not self._object_names:
            return -1.0
        obj_pos = self._sim.get_object_position(self._object_names[0])
        init_z = self._initial_obj_z.get(self._object_names[0], 0.0)
        return 0.0 if float(obj_pos[2]) > init_z + 0.05 else -1.0

    def _reward_dense_ee_distance(self) -> float:
        if not self._object_names:
            return 0.0
        ee_pos = self._sim.data.site_xpos[self._ee_site_id]
        dists = [
            float(np.linalg.norm(ee_pos - self._sim.get_object_position(n)))
            for n in self._object_names
        ]
        return -min(dists)

    # ------------------------------------------------------------------
    # Termination / info
    # ------------------------------------------------------------------

    def _is_terminated(self) -> bool:
        if not self._object_names:
            return False
        obj_pos = self._sim.get_object_position(self._object_names[0])
        init_z = self._initial_obj_z.get(self._object_names[0], 0.0)
        return float(obj_pos[2]) > init_z + 0.05

    def _build_info(self, sym_state=None) -> Dict[str, Any]:
        return {
            "symbolic_state": sym_state,
            "symbolic_plan":  self._current_plan,
            "plan_step":      self._plan_step,
            "step_count":     self._step_count,
        }

    # ------------------------------------------------------------------
    # Accessors for subclasses and wrappers
    # ------------------------------------------------------------------

    @property
    def sim(self):
        return self._sim

    @property
    def bridge(self):
        return self._bridge

    @property
    def object_names(self) -> List[str]:
        return self._object_names

    def get_ee_pos(self) -> np.ndarray:
        return self._sim.data.site_xpos[self._ee_site_id].copy()
