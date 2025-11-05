#!/usr/bin/env python3
"""
Improved PPO Training with Comprehensive Logging
================================================
Features:
- TensorBoard logging with detailed metrics
- Episode statistics tracking
- Reward component breakdown
- Success rate monitoring
- Anti-oscillation detection
- Curriculum progress tracking
"""

from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
parser.add_argument("--debug_camera", action="store_true", help="Save camera frames")
parser.add_argument("--log_dir", type=str, default="/workspace/teko/runs", help="TensorBoard log directory")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime
import os
import json
from pathlib import Path

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.schedulers.torch import KLAdaptiveRL

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder


# =====================================================================
# LOGGING WRAPPER
# =====================================================================
class LoggingEnvWrapper:
    """Environment wrapper with comprehensive logging."""
    
    def __init__(self, env, writer: SummaryWriter, debug_camera=False):
        self.env = env
        self.writer = writer
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.num_agents = 1
        self.device = getattr(env, "device", "cuda:0")
        self.debug_camera = debug_camera
        
        H, W = env.cfg.camera.height, env.cfg.camera.width
        self.H, self.W, self.C = H, W, 3
        self.flat_dim = self.C * H * W
        
        # Observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.flat_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Episode tracking
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        self.total_episodes = 0
        self.global_step = 0
        
        # Reward component tracking
        self.episode_reward_components = {
            'distance': [0.0] * self.num_envs,
            'alignment': [0.0] * self.num_envs,
            'orientation': [0.0] * self.num_envs,
            'velocity_penalty': [0.0] * self.num_envs,
            'oscillation_penalty': [0.0] * self.num_envs,
        }
        
        # Success tracking
        self.success_history = []
        self.recent_success_rate = 0.0
        
        # Console logging
        self.log_interval = 500
        self.last_console_log = 0

    def reset(self):
        obs, info = self.env.reset()
        rgb = self._extract_rgb(obs).to(self.device).float()
        
        # Reset episode trackers
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        for key in self.episode_reward_components:
            self.episode_reward_components[key] = [0.0] * self.num_envs
        
        return rgb.view(self.num_envs, -1).contiguous(), info

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device)

        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = self._extract_rgb(obs).to(self.device).float()
        
        # Ensure proper shapes
        reward = reward.reshape(-1, 1).contiguous()
        terminated = terminated.reshape(-1, 1).contiguous()
        truncated = truncated.reshape(-1, 1).contiguous()
        
        # Update episode statistics
        self.global_step += 1
        for env_idx in range(self.num_envs):
            self.episode_rewards[env_idx] += reward[env_idx].item()
            self.episode_lengths[env_idx] += 1
            
            # Track reward components if available
            if hasattr(self.env, 'reward_components'):
                for key in self.episode_reward_components:
                    if key in self.env.reward_components and self.env.reward_components[key]:
                        self.episode_reward_components[key][env_idx] += \
                            self.env.reward_components[key][-1]
            
            # Log completed episodes
            if terminated[env_idx] or truncated[env_idx]:
                self._log_episode(env_idx)
        
        # Periodic console logging
        if self.global_step - self.last_console_log >= self.log_interval:
            self._console_log()
            self.last_console_log = self.global_step
        
        return rgb.view(self.num_envs, -1).contiguous(), reward, terminated, truncated, info

    def _log_episode(self, env_idx):
        """Log completed episode to TensorBoard."""
        episode_reward = self.episode_rewards[env_idx]
        episode_length = self.episode_lengths[env_idx]
        
        # Log to TensorBoard
        self.writer.add_scalar('Episode/Reward', episode_reward, self.total_episodes)
        self.writer.add_scalar('Episode/Length', episode_length, self.total_episodes)
        
        # Log reward components
        for key, values in self.episode_reward_components.items():
            self.writer.add_scalar(f'Reward_Components/{key}', values[env_idx], self.total_episodes)
        
        # Update success tracking (check if reward is high enough)
        success = episode_reward > 20.0  # Threshold for success
        self.success_history.append(1.0 if success else 0.0)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        self.recent_success_rate = np.mean(self.success_history) if self.success_history else 0.0
        self.writer.add_scalar('Episode/Success_Rate', self.recent_success_rate, self.total_episodes)
        
        self.total_episodes += 1
        
        # Reset episode trackers
        self.episode_rewards[env_idx] = 0.0
        self.episode_lengths[env_idx] = 0
        for key in self.episode_reward_components:
            self.episode_reward_components[key][env_idx] = 0.0

    def _console_log(self):
        """Print progress to console."""
        if self.total_episodes == 0:
            return
        
        non_zero_rewards = [r for r in self.episode_rewards if r != 0]
        non_zero_lengths = [l for l in self.episode_lengths if l != 0]
        avg_reward = np.mean(non_zero_rewards) if non_zero_rewards else 0.0
        avg_length = np.mean(non_zero_lengths) if non_zero_lengths else 0.0
        
        if self.global_step % (self.log_interval * 10) == 0:
            print("\n" + "="*70)
        
        print(f"[Step {self.global_step:7d}] Episodes: {self.total_episodes:5d} | "
              f"Success: {self.recent_success_rate:.1%} | "
              f"Reward: {avg_reward:7.2f}")

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
        self.writer.close()


# =====================================================================
# MODELS (unchanged)
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
        x = inputs["states"]
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
# CURRICULUM SCHEDULER
# =====================================================================
class CurriculumScheduler:
    def __init__(self, env, writer: SummaryWriter, enabled=True):
        self.env = env
        self.writer = writer
        self.enabled = enabled
        self.current_level = 0
        self.success_threshold = 0.7
        self.window_size = 1000
        self.episode_outcomes = []

    def update(self, success_rate):
        if not self.enabled:
            return
        
        if success_rate >= self.success_threshold and self.current_level < 2:
            self.current_level += 1
            self.env.env.set_curriculum_level(self.current_level)
            self.episode_outcomes = []
            
            print(f"\n{'='*70}")
            print(f"🎓 CURRICULUM ADVANCED TO LEVEL {self.current_level}")
            print(f"   Success rate: {success_rate:.1%}")
            print(f"{'='*70}\n")
            
            self.writer.add_scalar('Curriculum/Level', self.current_level, self.current_level)


# =====================================================================
# TRAIN FUNCTION
# =====================================================================
def train():
    print("\n" + "=" * 70)
    print("🚀 TEKO Vision-Based Docking – Improved Training with Logging")
    print("=" * 70 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.log_dir, f"teko_ppo_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=exp_dir)
    print(f"TensorBoard logs: {exp_dir}")
    print(f"  Run: tensorboard --logdir {args.log_dir}")

    # Create environment
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")

    print("[INFO] Warming up simulation...")
    for _ in range(10):
        env.sim.step()
    env._init_observation_space()
    _ = env.reset()

    # Wrap environment with logging
    env = LoggingEnvWrapper(env, writer, debug_camera=args.debug_camera)
    print(f"✓ Environment created with {env.num_envs} envs")

    # Create models
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

    # PPO configuration
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

    # Create agent
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=args.num_envs, device=device),
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Curriculum
    curriculum = CurriculumScheduler(env, writer, enabled=args.curriculum)
    if args.curriculum:
        print("✓ Curriculum learning enabled")

    # Trainer configuration
    trainer_cfg = {
        "timesteps": 200000,
        "headless": args.headless,
        "disable_progressbar": True,
        "close_environment_at_exit": True,
    }

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    # Save configuration
    config_dict = {
        'timestamp': timestamp,
        'num_envs': args.num_envs,
        'curriculum': args.curriculum,
        'ppo_config': {k: str(v) for k, v in ppo_cfg.items()},
        'total_timesteps': trainer_cfg['timesteps'],
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "=" * 70)
    print("🎓 Starting training...")
    if args.curriculum:
        print("   Curriculum: Level 0 (close spawn, 0.5-0.8m)")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    print("\n✅ Training complete!")
    
    # Save final model
    model_path = os.path.join(exp_dir, "final_model.pt")
    agent.save(model_path)
    print(f"💾 Model saved to {model_path}")

    # Save training summary
    summary_path = os.path.join(exp_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"TEKO Vision-Based Docking Training Summary\n")
        f.write(f"==========================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Num Envs: {args.num_envs}\n")
        f.write(f"Total Timesteps: {trainer_cfg['timesteps']}\n")
        f.write(f"Total Episodes: {env.total_episodes}\n")
        f.write(f"Final Success Rate: {env.recent_success_rate:.2%}\n")
        f.write(f"Curriculum Enabled: {args.curriculum}\n")
        if args.curriculum:
            f.write(f"Final Curriculum Level: {curriculum.current_level}\n")
        f.write(f"\nModel saved to: {model_path}\n")
        f.write(f"TensorBoard logs: {exp_dir}\n")
    
    print(f"📊 Summary saved to {summary_path}")

    # Close resources
    writer.close()
    env.close()
    
    print(f"\n📊 View training progress:")
    print(f"   tensorboard --logdir {args.log_dir}\n")


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