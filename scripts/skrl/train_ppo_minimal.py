#!/usr/bin/env python3
"""
Improved PPO Training for Vision-Based Docking
===============================================
Fixed version (2025-11-05)
- Handles flattened RGB observations to avoid shape mismatch
- Keeps CNN encoder intact
- Compatible with SKRL memory allocation
- Simple console reward logger every 500 env steps (inside the env wrapper)
"""

from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
parser.add_argument("--debug_camera", action="store_true", help="Save camera frames for debugging")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime
import os
from pathlib import Path

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.schedulers.torch import KLAdaptiveRL

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder


# =====================================================================
# ENV WRAPPER (Flattened observations for SKRL compatibility)
# =====================================================================
class SimpleEnvWrapper:
    def __init__(self, env, debug_camera=False):
        self.env = env
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.num_agents = 1
        self.device = getattr(env, "device", "cuda:0")
        self.debug_camera = debug_camera
        self.frame_count = 0

        H, W = env.cfg.camera.height, env.cfg.camera.width
        self.H, self.W, self.C = H, W, 3
        self.flat_dim = self.C * H * W

        # flatten the RGB image (3*H*W)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.flat_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ---------------- REWARD CONSOLE LOGGER ----------------
        self._log_interval = 500               # print every 500 env steps
        self._global_step = 0
        # keep running sums on device to avoid host sync
        self._acc_sum = torch.zeros(1, device=self.device)
        self._acc_sq_sum = torch.zeros(1, device=self.device)
        self._acc_count = 0
        # -------------------------------------------------------

    def reset(self):
        obs, info = self.env.reset()
        rgb = self._extract_rgb(obs).to(self.device).float()  # (B,3,H,W)
        if self.debug_camera and self.frame_count == 0:
            self._save_debug_frame(rgb[0], "reset")
        return rgb.view(self.num_envs, -1).contiguous(), info  # flatten

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device)

        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = self._extract_rgb(obs).to(self.device).float()  # (B,3,H,W)

        if self.debug_camera and self.frame_count % 100 == 0:
            self._save_debug_frame(rgb[0], f"step_{self.frame_count}")
        self.frame_count += 1

        # ensure shapes
        reward = reward.reshape(-1, 1).contiguous()
        terminated = terminated.reshape(-1, 1).contiguous()
        truncated = truncated.reshape(-1, 1).contiguous()

        # ---------------- REWARD CONSOLE LOGGER ----------------
        # accumulate stats across all envs for this step
        r = reward.view(-1)  # [B]
        self._acc_sum += r.sum()
        self._acc_sq_sum += (r * r).sum()
        self._acc_count += r.numel()

        self._global_step += 1
        if (self._global_step % self._log_interval) == 0 and self._acc_count > 0:
            mean = (self._acc_sum / self._acc_count)
            var = (self._acc_sq_sum / self._acc_count) - (mean * mean)
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var)
            # move just the scalars to host for printing
            if (self._global_step // self._log_interval) % 10 == 0:
                print("\n─────────────────────────────────────────────────────────────")
            print(f"[Step {self._global_step:7d}]  🎯 mean reward: {float(mean):8.3f} ± {float(std):6.3f}")

            # reset accumulators
            self._acc_sum.zero_()
            self._acc_sq_sum.zero_()
            self._acc_count = 0
        # -------------------------------------------------------

        return rgb.view(self.num_envs, -1).contiguous(), reward, terminated, truncated, info

    def _extract_rgb(self, obs):
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], dict):
                return obs["policy"]["rgb"]
            elif "rgb" in obs:
                return obs["rgb"]
        return obs

    def _save_debug_frame(self, rgb_tensor, name):
        import cv2
        os.makedirs("/workspace/teko/debug_frames", exist_ok=True)
        frame = rgb_tensor.cpu().permute(1, 2, 0).numpy() * 255
        frame = frame.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        save_path = f"/workspace/teko/debug_frames/{name}.png"
        cv2.imwrite(save_path, frame_bgr)
        print(f"[DEBUG] Saved camera frame: {save_path}")

    def render(self):
        pass

    def close(self):
        self.env.close()


# =====================================================================
# MODELS (reshape flat → image before CNN)
# =====================================================================
class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        self.C, self.H, self.W = 3, 480, 640
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.policy = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]  # (B, flat)
        x = x.view(x.shape[0], self.C, self.H, self.W)
        feat = self.encoder(x)
        act = self.policy(feat)
        return act, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        self.C, self.H, self.W = 3, 480, 640
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        x = x.view(x.shape[0], self.C, self.H, self.W)
        feat = self.encoder(x)
        return self.value(feat), {}


# =====================================================================
# CURRICULUM (unchanged)
# =====================================================================
class CurriculumScheduler:
    def __init__(self, env, enabled=True):
        self.env = env
        self.enabled = enabled
        self.current_level = 0
        self.success_threshold = 0.7
        self.window_size = 1000
        self.episode_outcomes = []

    def update(self, success):
        if not self.enabled:
            return
        self.episode_outcomes.append(success)
        if len(self.episode_outcomes) > self.window_size:
            self.episode_outcomes.pop(0)
        if len(self.episode_outcomes) >= self.window_size:
            success_rate = sum(self.episode_outcomes) / len(self.episode_outcomes)
            if success_rate >= self.success_threshold and self.current_level < 2:
                self.current_level += 1
                self.env.env.set_curriculum_level(self.current_level)
                self.episode_outcomes = []
                print(f"\n{'='*70}")
                print(f"🎓 CURRICULUM ADVANCED TO LEVEL {self.current_level}")
                print(f"   Success rate: {success_rate:.1%}")
                print(f"{'='*70}\n")


# =====================================================================
# TRAIN FUNCTION
# =====================================================================
def train():
    print("\n" + "=" * 70)
    print("🚀 TEKO Vision-Based Docking – Improved PPO Training (Fixed)")
    print("=" * 70 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/workspace/teko/runs/teko_ppo_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")

    print("[INFO] Warming up simulation...")
    for _ in range(10):
        env.sim.step()
    env._init_observation_space()
    _ = env.reset()

    env = SimpleEnvWrapper(env, debug_camera=args.debug_camera)
    print(f"✓ Environment created with {env.num_envs} envs")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts": 32,
        "learning_epochs": 10,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
        "kl_threshold": 0,
        "learning_rate_scheduler": KLAdaptiveRL,
        "learning_rate_scheduler_kwargs": {
            "kl_threshold": 0.008,
            "min_lr": 1e-5,
            "max_lr": 1e-3,
        }
    })

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=args.num_envs, device=device),
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    curriculum = CurriculumScheduler(env, enabled=args.curriculum)
    if args.curriculum:
        print("✓ Curriculum learning enabled")

    trainer_cfg = {
        "timesteps": 200000,
        "headless": args.headless,
        "disable_progressbar": True,   # ✅ disable tqdm bar
        "close_environment_at_exit": True,
    }

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print("\n" + "=" * 70)
    print("🎓 Starting training...")
    if args.curriculum:
        print("   Curriculum: Level 0 (close spawn)")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    print("\n✅ Training complete!")
    agent.save(f"{exp_dir}/final_model.pt")
    print(f"💾 Model saved to {exp_dir}/final_model.pt")

    with open(f"{exp_dir}/training_summary.txt", "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Num Envs: {args.num_envs}\n")
        f.write(f"Total Timesteps: {trainer_cfg['timesteps']}\n")
        f.write(f"Curriculum: {args.curriculum}\n")
        f.write(f"Final Level: {curriculum.current_level if args.curriculum else 'N/A'}\n")

    env.close()
    print(f"\n📊 Training logs saved to: {exp_dir}\n")


# =====================================================================
# MAIN
# =====================================================================
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
