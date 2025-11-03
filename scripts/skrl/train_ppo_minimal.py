#!/usr/bin/env python3
"""
Minimal PPO Training Script for TEKO Docking
=============================================
Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/skrl/train_ppo_minimal.py --num_envs 1
"""

# ======================================================================
# Isaac Sim initialization (must be first)
# ======================================================================
from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# ======================================================================
# Imports
# ======================================================================
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

# TEKO environment + CNN
from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder

# Target resolution (matches your observation space)
TARGET_H, TARGET_W = 480, 640


# ======================================================================
# Helper – ensure SKRL's list/tuple actions become tensors
# ======================================================================
class TensorActionWrapper(gym.Wrapper):
    def step(self, action):
        if isinstance(action, (list, tuple)):
            if len(action) == 1:
                action = action[0]
            else:
                action = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in action])
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        device = getattr(self.unwrapped, "device", "cuda:0")
        return super().step(action.to(device))


# ======================================================================
# Helper – robust RGB extraction + hard resize
# ======================================================================
def extract_rgb_any(x):
    """Extract an RGB tensor/array from nested dicts/tuples/lists."""
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if isinstance(x, dict):
        if "rgb" in x:
            return extract_rgb_any(x["rgb"])
        if "policy" in x and isinstance(x["policy"], dict) and "rgb" in x["policy"]:
            return extract_rgb_any(x["policy"]["rgb"])
        for v in x.values():
            out = extract_rgb_any(v)
            if out is not None:
                return out
        return None
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    try:
        return torch.as_tensor(x)
    except Exception:
        return None


def ensure_bchw_resized(rgb, device):
    """Return [B, 3, TARGET_H, TARGET_W] float32 tensor on device."""
    if rgb is None:
        rgb = torch.zeros((1, 3, TARGET_H, TARGET_W), dtype=torch.float32, device=device)
        return rgb

    if not torch.is_tensor(rgb):
        rgb = torch.as_tensor(rgb)

    # [H, W, C] -> [1, C, H, W]
    if rgb.ndim == 3:
        # If last dim is 3, assume HWC
        if rgb.shape[-1] == 3 and (rgb.shape[0] != 3):
            rgb = rgb.permute(2, 0, 1)  # HWC -> CHW
        rgb = rgb.unsqueeze(0)

    # [B, H, W, C] -> [B, C, H, W]
    if rgb.ndim == 4 and rgb.shape[-1] == 3 and rgb.shape[1] != 3:
        rgb = rgb.permute(0, 3, 1, 2)

    rgb = rgb.to(device=device, dtype=torch.float32)

    # Normalize if needed
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    # Hard enforce spatial size
    if rgb.shape[-2] != TARGET_H or rgb.shape[-1] != TARGET_W:
        rgb = F.interpolate(rgb, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)

    # If multiple envs, SKRL can handle B>1; otherwise OK
    return rgb


# ======================================================================
# Preprocessor class for SKRL (agent-side)
# ======================================================================
class RGBPreprocessor:
    def __init__(self, device="cuda:0", normalize=True):
        self.device = device
        self.normalize = normalize

    def __call__(self, x):
        rgb = extract_rgb_any(x)
        rgb = ensure_bchw_resized(rgb, self.device)
        return rgb


# ======================================================================
# Observation wrapper (env-side) – critical fix
# ======================================================================
class ObsToTensorWrapper:
    """Convert dict/list observations into fixed-size torch tensors before SKRL sees them."""
    def __init__(self, env, device="cuda:0"):
        self.env = env
        self.device = device
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = getattr(env, "num_envs", 1)
        self.num_agents = getattr(env, "num_agents", 1)

    def reset(self):
        obs, info = self.env.reset()
        if isinstance(obs, list) and len(obs) == 1:
            obs = obs[0]
        return self._to_tensor(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, list) and len(obs) == 1:
            obs = obs[0]
        return self._to_tensor(obs), reward, terminated, truncated, info

    def _to_tensor(self, obs):
        rgb = extract_rgb_any(obs)
        rgb = ensure_bchw_resized(rgb, self.device)
        return rgb

    def render(self):
        pass

    def close(self):
        self.env.close()


# ======================================================================
# Policy / Value Networks
# ======================================================================
class PolicyNetwork(GaussianMixin, Model):
    """Policy: RGB → wheel velocities"""
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
        rgb = inputs["states"]  # already [B, 3, 480, 640] float32
        features = self.encoder(rgb)
        actions = self.policy(features)
        return actions, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    """Value: RGB → scalar state value"""
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


# ======================================================================
# Training loop
# ======================================================================
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

    # ----- Environment -----
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")
    env.num_agents = 1
    env = TensorActionWrapper(env)
    env = ObsToTensorWrapper(env, device=device)  # ✅ always outputs [B,3,480,640]

    # Single-agent adapter (SKRL compatibility)
    class SingleAgentWrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.num_envs = getattr(env, "num_envs", 1)
            self.num_agents = getattr(env, "num_agents", 1)
        def reset(self):
            obs, info = self.env.reset()
            return obs, info
        def step(self, action):
            return self.env.step(action)
        def render(self):
            pass
        def close(self):
            self.env.close()

    env = SingleAgentWrapper(env)
    print("✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # ----- Networks -----
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value  = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value:  {sum(p.numel() for p in value.parameters()):,}")

    # ----- PPO Config -----
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
        # A preprocessor is still set for agent-side safety (mirrors env wrapper)
        "state_preprocessor": RGBPreprocessor,
        "state_preprocessor_kwargs": {"device": str(device), "normalize": True},
        "next_state_preprocessor": RGBPreprocessor,
        "next_state_preprocessor_kwargs": {"device": str(device), "normalize": True},
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

    # ----- Trainer -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"runs/teko_ppo_{args.num_envs}envs_{timestamp}"
    trainer_cfg = {"timesteps": 50000, "headless": args.headless}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print("\n" + "="*70)
    print("🎓 Starting training...")
    print("="*70)
    print(f"Total timesteps: {trainer_cfg['timesteps']:,}")
    print(f"Rollout steps:  {ppo_cfg['rollouts']}")
    print(f"Learning epochs:{ppo_cfg['learning_epochs']}")
    print(f"Mini batches:   {ppo_cfg['mini_batches']}")
    print("="*70 + "\n")

    trainer.train()

    print("\n✅ Training complete!")
    agent.save(f"{exp_dir}/final_model.pt")
    print(f"💾 Model saved to {exp_dir}/final_model.pt\n")
    env.close()


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        simulation_app.close()
