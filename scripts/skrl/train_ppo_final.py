#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Vision-Based Docking — PPO Training (FIXED VERSION)
Fixed PPO update triggering to ensure learning happens.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# FORCE headless environment BEFORE anything else
os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"
os.environ["DISPLAY"] = ""

# Create custom args for AppLauncher with cameras enabled
custom_args = argparse.Namespace(
    headless=True,
    enable_cameras=True,
    experience='',
    livestream=0
)

# Initialize AppLauncher with custom args that include enable_cameras
app_launcher = AppLauncher(custom_args)
simulation_app = app_launcher.app

# NOW parse our training arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--timesteps", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint_every", type=int, default=5_000)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------

from datetime import datetime
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder


# ======================================================================
# Wrappers
# ======================================================================
class RGBBoxWrapper:
    def __init__(self, env):
        self.env = env
        h, w = env.cfg.camera.height, env.cfg.camera.width
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)
        self.action_space = env.action_space
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)

    def reset(self, *a, **kw):
        obs, info = self.env.reset(*a, **kw)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", next(iter(obs.values()))))
        return obs.float(), info

    def step(self, actions):
        obs, r, t, tr, info = self.env.step(actions)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", next(iter(obs.values()))))
        return obs.float(), r, t, tr, info

    def __getattr__(self, n):
        return getattr(self.env, n)


class ActionBoxWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.device = getattr(env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = env.observation_space

    def reset(self, *a, **kw):
        return self.env.reset(*a, **kw)

    def step(self, actions):
        if isinstance(actions, tuple):
            actions = actions[0]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)
        return self.env.step(actions)

    def __getattr__(self, n):
        return getattr(self.env, n)


# ======================================================================
# PPO Models
# ======================================================================
class PolicyNet(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        c, h, w = observation_space.shape
        self.h, self.w = int(h), int(w)
        self.encoder = create_visual_encoder("simple", 256, False)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * self.h * self.w:
            x = x.view(x.shape[0], 3, self.h, self.w)
        return self.head(self.encoder(x)), self.log_std, {}


class ValueNet(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        c, h, w = observation_space.shape
        self.h, self.w = int(h), int(w)
        self.encoder = create_visual_encoder("simple", 256, False)
        self.v = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * self.h * self.w:
            x = x.view(x.shape[0], 3, self.h, self.w)
        return self.v(self.encoder(x)), {}


# ======================================================================
# Main
# ======================================================================
def main():
    print("\n" + "=" * 78)
    print("🚀 TEKO Vision-Based Docking — FIXED PPO Training")
    print("   Fixed update triggering for proper learning")
    print("=" * 78 + "\n")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    base_env = TekoEnv(cfg=cfg)

    def _safe_pre_physics_step(actions):
        if isinstance(actions, tuple):
            actions = actions[0]
        if not hasattr(base_env, 'prev_actions') or base_env.prev_actions is None:
            base_env.prev_actions = torch.zeros_like(actions)
        elif base_env.prev_actions.shape != actions.shape:
            base_env.prev_actions = torch.zeros_like(actions)
        else:
            prev = getattr(base_env, 'actions', actions)
            if isinstance(prev, tuple):
                prev = prev[0]
            base_env.prev_actions.copy_(prev)
        base_env.actions = actions
        base_env._lazy_init_articulation()

    base_env._pre_physics_step = _safe_pre_physics_step
    print("⚙️ Applied runtime patch: tuple-safe prev_actions")

    env = RGBBoxWrapper(base_env)
    env = ActionBoxWrapper(env)
    env = wrap_env(env, wrapper="gymnasium")

    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space:      {env.action_space}")

    policy = PolicyNet(env.observation_space, env.action_space, device)
    value = ValueNet(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

    # --- FIXED PPO CONFIG ---
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts": 64,
        "learning_epochs": 8,
        "mini_batches": 8,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 1e-4,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.05,
        "value_loss_scale": 0.25,
        "state_preprocessor": None,
        "state_preprocessor_kwargs": {},
        "value_preprocessor": None,
        "value_preprocessor_kwargs": {},
        "rewards_shaper": None,  # ADD THIS LINE
        "experiment": {
            "write_interval": 200,
            "checkpoint_interval": 0,
        },
    })

    # Force memory size to match rollouts exactly
    memory = RandomMemory(
        memory_size=64,  # Explicit size
        num_envs=args.num_envs, 
        device=device
    )

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        cfg=ppo_cfg,
        device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/workspace/teko/runs/teko_ppo_fixed_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    print(f"✓ TensorBoard: tensorboard --logdir /workspace/teko/runs")

    original_post = agent.post_interaction

    def logged_post_interaction(*, timestep, timesteps):
        original_post(timestep=timestep, timesteps=timesteps)

        try:
            rewards = agent.memory.get_tensor_by_name("rewards")
            reward_mean = float(rewards.mean().item()) if rewards.numel() else 0.0
        except Exception:
            reward_mean = 0.0

        policy_loss = 0.0
        value_loss = 0.0
        td = getattr(agent, "tracking_data", {})
        if isinstance(td, dict):
            v = td.get("Loss / Policy loss", 0.0)
            policy_loss = float(v[-1] if isinstance(v, list) and v else v)
            v = td.get("Loss / Value loss", 0.0)
            value_loss = float(v[-1] if isinstance(v, list) and v else v)

        if timestep % 200 == 0:
            writer.add_scalar("Training/Reward_mean", reward_mean, timestep)
            writer.add_scalar("Training/Policy_loss", policy_loss, timestep)
            writer.add_scalar("Training/Value_loss", value_loss, timestep)

            rc = getattr(base_env, "reward_components", None)
            if isinstance(rc, dict):
                for k in ["distance", "progress", "alignment",
                          "velocity_penalty", "oscillation_penalty",
                          "collision_penalty", "wall_penalty"]:
                    vals = rc.get(k, [])
                    if vals:
                        writer.add_scalar(f"Rewards/{k}", float(np.mean(vals[-50:])), timestep)

            pct = 100.0 * timestep / timesteps
            print(f"[{timestep:7d}/{timesteps}] {pct:5.1f}% | "
                  f"R̄ {reward_mean:8.3f} | π-loss {policy_loss:8.5f} | v-loss {value_loss:7.2f}")

        if args.checkpoint_every > 0 and timestep > 0 and timestep % args.checkpoint_every == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_step_{timestep}.pt")
            agent.save(ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    agent.post_interaction = logged_post_interaction

    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": True},
        env=env,
        agents=agent
    )

    print(f"\n✓ Training: {args.num_envs} envs, {args.timesteps:,} steps")
    print(f"✓ Save dir: {save_dir}\n")
    print("=" * 78)
    print("🎓 Starting FIXED training...")
    print("=" * 78 + "\n")

    trainer.train()

    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\n✅ Training complete!\n💾 Model: {final_path}\n")

    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        simulation_app.close()