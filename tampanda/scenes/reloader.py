"""Scene reloader: rebuild a scene while preserving physics state.

Objects that appear in both the old and new scene (matched by body name) get
their position, orientation, and velocity transplanted into the new model.
The robot joint state is always preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from tampanda.environments.franka_env import FrankaEnvironment
    from tampanda.scenes.builder import SceneBuilder

# Number of robot DOF to preserve: 7 arm joints + 2 finger joints.
_ROBOT_NDOF = 9


class SceneReloader:
    """Rebuilds a scene while carrying physics state across the reload.

    Usage::

        reloader = SceneReloader()
        state = reloader.snapshot(env)          # capture state
        new_env = reloader.reload(env, builder) # one-shot rebuild + restore

    Or manually::

        state = reloader.snapshot(env)
        # ... modify builder ...
        new_env = builder.build_env()
        reloader.restore(new_env, state)
    """

    def snapshot(self, env: FrankaEnvironment) -> dict:
        """Capture the full dynamics state of an environment.

        Returns a dict with:
          - ``robot_qpos``: arm + finger joint positions (9-vector)
          - ``robot_qvel``: arm + finger joint velocities (9-vector)
          - ``robot_ctrl``: full actuator control vector
          - ``objects``:    mapping of body name → {qpos, qvel} for every
                            free-joint body in the scene
        """
        state: dict = {
            "robot_qpos": env.data.qpos[:_ROBOT_NDOF].copy(),
            "robot_qvel": env.data.qvel[:_ROBOT_NDOF].copy(),
            "robot_ctrl": env.data.ctrl.copy(),
            "objects": {},
        }
        for joint_id in range(env.model.njnt):
            if env.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body_id = env.model.jnt_bodyid[joint_id]
            name = mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            qadr = env.model.jnt_qposadr[joint_id]
            vadr = env.model.jnt_dofadr[joint_id]
            state["objects"][name] = {
                "qpos": env.data.qpos[qadr: qadr + 7].copy(),
                "qvel": env.data.qvel[vadr: vadr + 6].copy(),
            }
        return state

    def restore(self, env: FrankaEnvironment, state: dict):
        """Write a previously captured state into a (new) environment.

        Bodies absent from the new scene are silently skipped.
        Bodies present in the new scene but absent from the snapshot keep
        the positions they were given at construction time.
        """
        n_robot = min(_ROBOT_NDOF, env.model.nq)
        env.data.qpos[:n_robot] = state["robot_qpos"][:n_robot]
        env.data.qvel[:n_robot] = state["robot_qvel"][:n_robot]
        n_ctrl = min(len(state["robot_ctrl"]), env.model.nu)
        env.data.ctrl[:n_ctrl] = state["robot_ctrl"][:n_ctrl]

        for joint_id in range(env.model.njnt):
            if env.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body_id = env.model.jnt_bodyid[joint_id]
            name = mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name not in state["objects"]:
                continue
            obj = state["objects"][name]
            qadr = env.model.jnt_qposadr[joint_id]
            vadr = env.model.jnt_dofadr[joint_id]
            env.data.qpos[qadr: qadr + 7] = obj["qpos"]
            env.data.qvel[vadr: vadr + 6] = obj["qvel"]

        env.ik.update_configuration(env.data.qpos)
        mujoco.mj_forward(env.model, env.data)

    def reload(
        self,
        env: FrankaEnvironment,
        builder: SceneBuilder,
        **build_kwargs,
    ) -> FrankaEnvironment:
        """Snapshot ``env``, rebuild with ``builder``, restore state.

        Args:
            env:           The environment to snapshot and replace.
            builder:       SceneBuilder describing the new scene.
            **build_kwargs: Forwarded to ``builder.build_env()``.

        Returns:
            A new FrankaEnvironment with state carried over from ``env``.
        """
        state = self.snapshot(env)
        new_env = builder.build_env(**build_kwargs)
        self.restore(new_env, state)
        return new_env
