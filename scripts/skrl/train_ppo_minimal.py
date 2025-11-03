#!/usr/bin/env python3
"""
Minimal PPO Training Script for TEKO Docking – FULL FIXED VERSION
=================================================================
Handles multi-environment setup (num_envs > 1), flattens camera
observations, and ensures reward/done tensors have shape [num_envs].
"""

from isaacsim import SimulationApp
import argparse

# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder


# ------------------------------------------------------------
# Wrapper for SKRL
# ------------------------------------------------------------
class SimpleEnvWrapper:
    """Expose a single-agent SKRL-compatible API for multi-env TEKO."""
    def __init__(self, env):
        self.env = env
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.num_agents = 1
        self.device = getattr(env, "device", "cuda:0")

        H, W = env.cfg.camera.height, env.cfg.camera.width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, H, W), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        obs, info = self.env.reset()
        rgb = self._extract_rgb(obs).to(self.device).float()
        rgb = rgb.view(rgb.shape[0], -1)
        return rgb, info

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device)

        obs, reward, terminated, truncated, info = self.env.step(action)

        rgb = self._extract_rgb(obs).to(self.device).float()
        rgb = rgb.view(rgb.shape[0], -1)  # flatten camera frames

         # ✅ force everything to [num_envs, 1] (what SKRL memory expects)
        reward = reward.reshape(-1, 1).contiguous()
        terminated = terminated.reshape(-1, 1).contiguous()
        truncated = truncated.reshape(-1, 1).contiguous()

        # Debug
        # print(f"[DEBUG shapes after fix] obs:{rgb.shape}, reward:{reward.shape}, term:{terminated.shape}")

        return rgb, reward, terminated, truncated, info

    def _extract_rgb(self, obs):
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], dict):
                return obs["policy"]["rgb"]
            elif "rgb" in obs:
                return obs["rgb"]
        return obs

    def render(self):
        pass

    def close(self):
        self.env.close()


# ------------------------------------------------------------
# PPO Networks
# ------------------------------------------------------------
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
        rgb = inputs["states"]
        if rgb.ndim == 2 and rgb.shape[1] == 3 * 480 * 640:
            rgb = rgb.view(rgb.shape[0], 3, 480, 640)
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
        if rgb.ndim == 2 and rgb.shape[1] == 3 * 480 * 640:
            rgb = rgb.view(rgb.shape[0], 3, 480, 640)
        features = self.encoder(rgb)
        return self.value(features), {}


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train():
    print("\n" + "=" * 70)
    print("🚀 TEKO Vision-Based Docking – PPO Training (FINAL-FIXED)")
    print("=" * 70 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Environment setup ---
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")

    # Warm-up to make sure cameras deliver frames
    print("[INFO] Warming up simulation for camera readiness...")
    for _ in range(10):
        env.sim.step()
    env._init_observation_space()
    _ = env.reset()

    env = SimpleEnvWrapper(env)
    print(f"✓ Environment created with {env.num_envs} envs")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    obs, _ = env.reset()
    print(f"[DEBUG] First obs shape seen by policy: {tuple(obs.shape)}")

    # --- PPO setup ---
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

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
    exp_dir = f"runs/teko_ppo_{timestamp}"
    trainer_cfg = {"timesteps": 50000, "headless": args.headless}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print("\n" + "=" * 70)
    print("🎓 Starting training...")
    print("=" * 70 + "\n")

    trainer.train()

    print("\n✅ Training complete!")
    agent.save(f"{exp_dir}/final_model.pt")
    print(f"💾 Model saved to {exp_dir}/final_model.pt\n")
    env.close()


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
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
