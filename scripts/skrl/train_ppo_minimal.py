#!/usr/bin/env python3
"""
Minimal PPO Training Script for TEKO Docking
=============================================
Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train_ppo_minimal.py --num_envs 1
"""

# CRITICAL: Import SimulationApp FIRST
from isaacsim import SimulationApp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

# Initialize Isaac Sim
simulation_app = SimulationApp({"headless": args.headless})

# ======================================================================
# Imports
# ======================================================================
import torch
import torch.nn as nn
import numpy as np
import gym
from datetime import datetime

# SKRL imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Your environment
from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

# Your CNN
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder


# ======================================================================
#   Action wrapper to handle SKRL's list/tuple actions
# ======================================================================
class TensorActionWrapper(gym.Wrapper):
    """Convert SKRL's action format to tensor for Isaac Lab"""
    def step(self, action):
        # Handle different action formats
        if isinstance(action, (list, tuple)):
            if len(action) == 1:
                action = action[0]
            else:
                action = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in action])
        
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        
        # Ensure 2D: [batch, action_dim]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        
        # Move to device
        device = getattr(self.unwrapped, "device", "cuda:0")
        action = action.to(device)
        
        # Call parent's step (goes through gym.Wrapper chain properly)
        return super().step(action)


# ======================================================================
#   Policy Network
# ======================================================================
class PolicyNetwork(GaussianMixin, Model):
    """Policy: RGB -> wheel velocities"""
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)

        self.encoder = create_visual_encoder(
            architecture="simple",
            feature_dim=256,
            pretrained=False
        )

        self.policy = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if isinstance(states, dict):
            rgb = states.get("policy", {}).get("rgb", None)
            if rgb is None:
                rgb = states.get("rgb", None)
            if rgb is None:
                raise ValueError(f"Cannot find 'rgb' in states: {states.keys()}")
        else:
            rgb = states

        # Reshape to [B, 3, 480, 640] if needed
        if rgb.dim() == 2:
            total_pixels = 480 * 640
            b = rgb.shape[0] // total_pixels
            rgb = rgb.view(b, total_pixels, 3).permute(0, 2, 1).view(b, 3, 480, 640)
        elif rgb.dim() == 3:
            b = rgb.shape[0]
            rgb = rgb.permute(0, 2, 1).view(b, 3, 480, 640)

        features = self.encoder(rgb)
        actions = self.policy(features)
        return actions, self.log_std, {}


# ======================================================================
#   Value Network
# ======================================================================
class ValueNetwork(DeterministicMixin, Model):
    """Value: RGB -> state value"""
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)

        self.encoder = create_visual_encoder(
            architecture="simple",
            feature_dim=256,
            pretrained=False
        )

        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        if isinstance(states, dict):
            rgb = states.get("policy", {}).get("rgb", None)
            if rgb is None:
                rgb = states.get("rgb", None)
            if rgb is None:
                raise ValueError(f"Cannot find 'rgb' in states: {states.keys()}")
        else:
            rgb = states

        if rgb.dim() == 2:
            total_pixels = 480 * 640
            b = rgb.shape[0] // total_pixels
            rgb = rgb.view(b, total_pixels, 3).permute(0, 2, 1).view(b, 3, 480, 640)
        elif rgb.dim() == 3:
            b = rgb.shape[0]
            rgb = rgb.permute(0, 2, 1).view(b, 3, 480, 640)

        features = self.encoder(rgb)
        value = self.value(features)
        return value, {}


# ======================================================================
#   Training
# ======================================================================
def train():
    print("\n" + "="*70)
    print("🚀 TEKO Vision-Based Docking - PPO Training")
    print("="*70 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n📦 Creating {args.num_envs} environment(s)...")
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)

    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")
    env = wrap_env(env, wrapper="isaaclab")
    env = TensorActionWrapper(env)

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    print("\n🧠 Building networks...")
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value  = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"✓ Value:  {sum(p.numel() for p in value.parameters()):,} parameters")

    print("\n⚙️  Configuring PPO...")
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
        device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"runs/teko_ppo_{args.num_envs}envs_{timestamp}"
    print(f"\n📁 Experiment: {exp_dir}")

    trainer_cfg = {"timesteps": 50000, "headless": args.headless}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    agent.tracking_data = ["Reward / Instantaneous", "Loss / Policy loss", "Loss / Value loss"]
    agent.write_interval = 100
    agent.checkpoint_interval = 5000

    print("\n" + "="*70)
    print("🎓 Starting training...")
    print("="*70)
    print(f"Total timesteps: {trainer_cfg['timesteps']:,}")
    print(f"Rollout steps: {ppo_cfg['rollouts']}")
    print(f"Learning epochs: {ppo_cfg['learning_epochs']}")
    print(f"Mini batches: {ppo_cfg['mini_batches']}")
    print("="*70 + "\n")

    trainer.train()

    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70 + "\n")

    agent.save(f"{exp_dir}/final_model.pt")
    print(f"💾 Model saved: {exp_dir}/final_model.pt\n")

    env.close()


# ======================================================================
#   Main
# ======================================================================
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()