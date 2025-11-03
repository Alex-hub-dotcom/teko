#!/usr/bin/env python3
"""
Minimal PPO Training Script for TEKO Docking
=============================================
Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train_ppo_minimal.py --num_envs 1
"""

from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder

TARGET_H, TARGET_W = 480, 640


class TensorActionWrapper(gym.Wrapper):
    def step(self, action):
        if isinstance(action, (list, tuple)):
            action = action[0] if len(action) == 1 else torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in action])
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        device = getattr(self.unwrapped, "device", "cuda:0")
        return super().step(action.to(device))


class ObsToTensorWrapper:
    """Force observations to fixed [B, 3, 480, 640] tensor"""
    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = getattr(env, "num_envs", 1)
        self.num_agents = getattr(env, "num_agents", 1)

    def reset(self):
        obs, info = self.env.reset()
        return self._fix_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._fix_obs(obs), reward, terminated, truncated, info

    def _fix_obs(self, obs):
        # Extract RGB from nested dict
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], dict) and "rgb" in obs["policy"]:
                rgb = obs["policy"]["rgb"]
            elif "rgb" in obs:
                rgb = obs["rgb"]
            else:
                raise ValueError(f"Cannot find RGB in obs: {obs.keys()}")
        else:
            rgb = obs

        # Convert to tensor
        if not torch.is_tensor(rgb):
            rgb = torch.as_tensor(rgb)

        rgb = rgb.to(device=self.device, dtype=torch.float32)

        # Normalize if needed
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        # Handle different shapes
        if rgb.ndim == 3:  # [H, W, C]
            if rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # -> [1, C, H, W]
        elif rgb.ndim == 4:  # [B, H, W, C] or [B, C, H, W]
            if rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)  # -> [B, C, H, W]

        # Force resize to target resolution
        if rgb.shape[-2:] != (TARGET_H, TARGET_W):
            rgb = F.interpolate(rgb, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)

        return rgb

    def render(self):
        pass

    def close(self):
        self.env.close()


class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.policy = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        rgb = inputs["states"]  # Already [B, 3, 480, 640]
        features = self.encoder(rgb)
        actions = self.policy(features)
        return actions, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        rgb = inputs["states"]
        features = self.encoder(rgb)
        return self.value(features), {}


def train():
    print("\n" + "="*70)
    print("🚀 TEKO Vision-Based Docking – PPO Training")
    print("="*70 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")
    env = TensorActionWrapper(env)
    env = ObsToTensorWrapper(env, device=device)

    print("✓ Environment created")
    print(f"  Observation will be forced to: [B, 3, {TARGET_H}, {TARGET_W}]")
    print(f"  Action space: {env.action_space}")

    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"✓ Value:  {sum(p.numel() for p in value.parameters()):,} parameters")

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts": 16,
        "learning_epochs": 8,
        "mini_batches": 2,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0,
        "state_preprocessor": None,
    })

    memory = RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=args.num_envs, device=device)
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"runs/teko_ppo_{args.num_envs}envs_{timestamp}"
    trainer_cfg = {"timesteps": 50000, "headless": args.headless}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print("\n" + "="*70)
    print("🎓 Starting training...")
    print("="*70)
    print(f"Total timesteps: {trainer_cfg['timesteps']:,}")
    print(f"Rollout steps: {ppo_cfg['rollouts']}")
    print("="*70 + "\n")

    trainer.train()

    print("\n✅ Training complete!")
    agent.save(f"{exp_dir}/final_model.pt")
    print(f"💾 Model saved: {exp_dir}/final_model.pt\n")
    env.close()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()