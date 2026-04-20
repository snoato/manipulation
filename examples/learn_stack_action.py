"""Learn to execute the 'stack' action via SAC + HER.

Goal: train a policy that satisfies the postconditions of
stack(gripper1, block_0, block_1) — place block_0 on top of block_1 —
using TampandaGoalEnv driven by the blocks-world DomainBridge.

The bridge grounds the 'on(block_0, block_1)' predicate geometrically; HER
relabels failed episodes as successes for whichever goal was actually reached,
making sparse rewards tractable.

Usage::

    python examples/learn_stack_action.py               # train
    python examples/learn_stack_action.py --eval        # evaluate saved model
    python examples/learn_stack_action.py --timesteps 1000000

Requires: stable-baselines3  (pip install stable-baselines3)
"""

import argparse
import numpy as np


# ── Scene factory ─────────────────────────────────────────────────────────────

def make_scene():
    from tampanda.scenes import ArmSceneBuilder, TABLE_TEMPLATE
    from tampanda.scenes import BLOCK_SMALL_TEMPLATE

    b = ArmSceneBuilder()
    b.add_resource("table", TABLE_TEMPLATE)
    b.add_resource("block", BLOCK_SMALL_TEMPLATE)

    b.add_object("table", pos=[0.45, 0.0, 0.0])
    # Two blocks on the table surface (table top ≈ 0.27 m, block half-height ≈ 0.02 m)
    b.add_object("block", name="block_0", pos=[0.40, -0.08, 0.295], rgba=[0.9, 0.2, 0.2, 1.0])
    b.add_object("block", name="block_1", pos=[0.50,  0.08, 0.295], rgba=[0.2, 0.5, 0.9, 1.0])

    b.add_camera_orbit(
        "workspace",
        target=[0.45, 0.0, 0.35],
        distance=0.8,
        elevation=40,
        azimuth=30,
        fovy=60,
    )
    return b


# ── Bridge factory ────────────────────────────────────────────────────────────

# Block geometry: small cube, half-size 0.02 m
_BLOCK_HALF = np.array([0.02, 0.02, 0.02])
_TABLE_Z = 0.27  # table surface height


def _xy_overlap(pos1, pos2, threshold=0.8):
    """True if two same-size blocks overlap enough in XY to count as stacked."""
    hw = _BLOCK_HALF[0]
    xi = max(0.0, min(pos1[0]+hw, pos2[0]+hw) - max(pos1[0]-hw, pos2[0]-hw))
    yi = max(0.0, min(pos1[1]+hw, pos2[1]+hw) - max(pos1[1]-hw, pos2[1]-hw))
    if xi == 0.0 or yi == 0.0:
        return False
    return (xi * yi) / (2*hw)**2 >= threshold


def make_bridge_factory(block_names, table_z=_TABLE_Z):
    """Return a no-arg callable that builds a fresh DomainBridge each call."""
    from pathlib import Path
    from tampanda.tamp import DomainBridge

    _PDDL_PATH = (
        Path(__file__).parent.parent
        / "tampanda/symbolic/domains/blocks/pddl/blocks_domain.pddl"
    )

    def factory():
        bridge = DomainBridge(_PDDL_PATH, None)  # env injected by GoalEnv

        @bridge.predicate("on")
        def eval_on(env, fluents, block_top, block_bot):
            if block_top == block_bot:
                return False
            pos_top, _ = env.get_object_pose(block_top)
            pos_bot, _ = env.get_object_pose(block_bot)
            h = _BLOCK_HALF[2]
            expected_z = pos_bot[2] + h + h
            if abs(pos_top[2] - expected_z) > 0.012:
                return False
            return _xy_overlap(pos_top, pos_bot)

        @bridge.predicate("on-table")
        def eval_on_table(env, fluents, block):
            pos, _ = env.get_object_pose(block)
            return abs(pos[2] - (table_z + _BLOCK_HALF[2])) < 0.015

        @bridge.predicate("clear")
        def eval_clear(env, fluents, block):
            pos_b, _ = env.get_object_pose(block)
            for other in block_names:
                if other == block:
                    continue
                pos_o, _ = env.get_object_pose(other)
                h = _BLOCK_HALF[2]
                expected_z = pos_b[2] + h + h
                if abs(pos_o[2] - expected_z) < 0.012 and _xy_overlap(pos_o, pos_b):
                    return False
            return True

        bridge.fluent("holding",       initial=None)
        bridge.fluent("gripper-empty", initial=[("gripper1",)])

        return bridge

    return factory


# ── Environment factory ───────────────────────────────────────────────────────

BLOCK_NAMES    = ["block_0", "block_1"]
BRIDGE_OBJECTS = {"block": BLOCK_NAMES, "gripper": ["gripper1"]}
BRIDGE_GOALS   = [("on", "block_0", "block_1")]

# block_0 initial z (table top 0.27 + half-height 0.02 + small settling offset)
_BLOCK_INIT_Z = 0.295
# height of block_0 centre when stacked on block_1 (two half-heights)
_STACK_OFFSET = _BLOCK_HALF[2] * 2   # 0.04 m


def stack_reward(sim, _sym_state):
    """Staged dense reward that guides approach → grasp → lift → transport → place."""
    import mujoco

    ee_site_id = mujoco.mj_name2id(
        sim.get_model(), mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
    )
    ee_pos = sim.data.site_xpos[ee_site_id]
    b0_pos = sim.get_object_position("block_0")
    b1_pos = sim.get_object_position("block_1")

    reach     = -np.linalg.norm(ee_pos - b0_pos)
    lift      =  5.0 * max(0.0, b0_pos[2] - _BLOCK_INIT_Z)
    transport = -np.linalg.norm(b0_pos[:2] - b1_pos[:2])
    place_z   = -abs(b0_pos[2] - (b1_pos[2] + _STACK_OFFSET))

    return reach + lift + transport + place_z


def make_env(render_mode=None):
    from tampanda.gym import TampandaGoalEnv, PseudoGraspWrapper

    bridge_factory = make_bridge_factory(BLOCK_NAMES)

    env = TampandaGoalEnv(
        scene=make_scene(),
        obs=["joints", "ee_pose", "object_poses"],
        action_space_type="cartesian_delta",
        include_gripper=True,
        reward_fn=stack_reward,
        cameras=["workspace"],
        object_names=BLOCK_NAMES,
        max_episode_steps=400,
        n_substeps=5,
        bridge_factory=bridge_factory,
        bridge_objects=BRIDGE_OBJECTS,
        bridge_goals=BRIDGE_GOALS,
        goal_type="symbolic_predicates",
        goal_threshold=0.02,
        render_mode=render_mode,
    )
    return PseudoGraspWrapper(env, grasp_threshold=0.08, close_threshold=-0.3)



# ── Training ──────────────────────────────────────────────────────────────────

def train(total_timesteps: int = 500_000):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    print("Building training environment …")
    train_env = make_env()
    eval_env  = make_env(render_mode="rgb_array")

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=20,
        best_model_save_path="./logs/stack_action/",
        verbose=1,
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        learning_rate=1e-3,
        buffer_size=300_000,
        batch_size=256,
        learning_starts=1_000,  # must be > max_episode_steps (400) for HER
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
    )

    print(f"Training SAC+HER for {total_timesteps:,} steps …")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save("stack_action_sac_her")
    print("Saved model to stack_action_sac_her.zip")

    train_env.close()
    eval_env.close()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model_path: str = "stack_action_sac_her", n_episodes: int = 100):
    from stable_baselines3 import SAC

    print(f"Loading model from {model_path} …")
    env   = make_env(render_mode="rgb_array")
    model = SAC.load(model_path, env=env)

    successes = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                success = True

        successes.append(success)
        episode_lengths.append(steps)

        if (ep + 1) % 20 == 0:
            sr = np.mean(successes) * 100
            print(f"  [{ep+1}/{n_episodes}]  Success rate: {sr:.1f}%  "
                  f"Mean ep length: {np.mean(episode_lengths):.0f}")

    sr = np.mean(successes) * 100
    print(f"\nFinal success rate over {n_episodes} episodes: {sr:.1f}%")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")

    env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--model",      default="stack_action_sac_her")
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model, args.n_episodes)
    else:
        train(args.timesteps)
