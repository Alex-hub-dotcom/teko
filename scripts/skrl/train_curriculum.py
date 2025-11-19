#!/usr/bin/env python3
"""
TEKO - 12-STAGE ULTRA-GRADUAL CURRICULUM TRAINING
==================================================
2,000,000 steps total
15k minimum steps per stage before checking advancement
85% success rate threshold
Fixed curriculum advancement bug
"""

import argparse
import os
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
parser.add_argument("--steps", type=int, default=2000000, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--rollout_len", type=int, default=64, help="Rollout length")
parser.add_argument("--epochs", type=int, default=8, help="PPO epochs per update")
parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args)
sim = app.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
from teko.tasks.direct.teko.curriculum.curriculum_manager import should_advance_curriculum, STAGE_NAMES


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = create_visual_encoder("simple", 256, False)
        
        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(2))
    
    def forward(self, obs):
        feat = self.encoder(obs)
        return self.actor(feat), self.critic(feat), self.log_std
    
    def act(self, obs):
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value.squeeze(-1)
    
    def evaluate(self, obs, actions):
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, value.squeeze(-1), entropy


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        next_value = 0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
    
    return advantages, advantages + values


def ppo_update(policy, optimizer, obs, actions, logp_old, advantages, returns,
               epochs=8, batch_size=256, clip_ratio=0.2, value_clip=0.2):
    T, N = obs.shape[0], obs.shape[1]
    total_samples = T * N
    
    obs_flat = obs.view(total_samples, 3, 480, 640)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)
    
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    
    policy_losses, value_losses, entropies = [], [], []
    
    for epoch in range(epochs):
        indices = torch.randperm(total_samples, device=obs.device)
        
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            mb_idx = indices[start:end]
            
            logp, value, entropy = policy.evaluate(obs_flat[mb_idx], actions_flat[mb_idx])
            
            ratio = (logp - logp_old_flat[mb_idx]).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_flat[mb_idx]
            policy_loss = -torch.min(ratio * advantages_flat[mb_idx], clip_adv).mean()
            
            value_pred = torch.clamp(value, returns_flat[mb_idx] - value_clip, returns_flat[mb_idx] + value_clip) if value_clip else value
            value_loss = F.mse_loss(value_pred, returns_flat[mb_idx])
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
    
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


def main():
    print("\n" + "="*70)
    print("🎓 TEKO - 12-STAGE ULTRA-GRADUAL CURRICULUM")
    print("="*70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps: {args.steps:,}")
    print(f"Stages: 12 (ultra-gradual)")
    print(f"Advancement: 85% success, min 15k steps/stage")
    print("="*70 + "\n")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_curriculum = True
    env = TekoEnv(cfg=cfg)
    
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    log_dir = f"teko_curriculum/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"✓ Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"📊 TensorBoard: tensorboard --logdir teko_curriculum")
    print(f"💾 Checkpoints: {log_dir}\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["rgb"]
    
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    stage_success_window = deque(maxlen=200)
    
    current_episode_reward = torch.zeros(args.num_envs, device=device)
    current_episode_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
    
    step = 0
    steps_in_current_stage = 0  # FIXED: Track steps in CURRENT stage
    
    print(f"[CURRICULUM] {STAGE_NAMES[0]}\n")
    
    while step < args.steps:
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        
        for t in range(args.rollout_len):
            with torch.no_grad():
                action, logp, value = policy.act(obs)
            
            obs_dict, reward, term, trunc, info = env.step(action)
            next_obs = obs_dict["rgb"]
            done = term | trunc
            
            current_episode_reward += reward
            current_episode_length += 1
            
            for i in range(args.num_envs):
                if done[i]:
                    episode_rewards.append(current_episode_reward[i].item())
                    episode_lengths.append(current_episode_length[i].item())
                    success = reward[i] > 50
                    episode_successes.append(1.0 if success else 0.0)
                    stage_success_window.append(1.0 if success else 0.0)
                    current_episode_reward[i] = 0
                    current_episode_length[i] = 0
            
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done.float())
            
            obs = next_obs
            step += args.num_envs
            steps_in_current_stage += args.num_envs  # FIXED: increment per-stage counter
        
        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        val_t = torch.stack(val_buf)
        logp_t = torch.stack(logp_buf)
        done_t = torch.stack(done_buf)
        
        with torch.no_grad():
            advantages, returns = compute_gae(rew_t, val_t, done_t)
        
        policy_loss, value_loss, entropy = ppo_update(
            policy, optimizer, obs_t, act_t, logp_t, advantages, returns,
            epochs=args.epochs, batch_size=args.batch_size
        )
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        success_rate = np.mean(episode_successes) if episode_successes else 0
        stage_success = np.mean(stage_success_window) if stage_success_window else 0
        
        writer.add_scalar("train/reward", mean_reward, step)
        writer.add_scalar("train/episode_length", mean_length, step)
        writer.add_scalar("train/success_rate", success_rate, step)
        writer.add_scalar("train/stage_success", stage_success, step)
        writer.add_scalar("train/curriculum_stage", env.curriculum_level, step)
        writer.add_scalar("train/policy_loss", policy_loss, step)
        writer.add_scalar("train/value_loss", value_loss, step)
        writer.add_scalar("train/steps_in_stage", steps_in_current_stage, step)
        
        print(f"[{step:7d}] S{env.curriculum_level:02d} | R={mean_reward:6.1f} | "
              f"Len={mean_length:4.0f} | SR={success_rate*100:4.1f}% | "
              f"SSR={stage_success*100:4.1f}% | StageSteps={steps_in_current_stage:5d}")
        
        # FIXED: Check advancement only after minimum steps in THIS stage
        if steps_in_current_stage >= 15000:  # Minimum 15k steps per stage
            if should_advance_curriculum(stage_success, env.curriculum_level):
                env.set_curriculum_level(env.curriculum_level + 1)
                stage_success_window.clear()
                steps_in_current_stage = 0  # FIXED: Reset counter when advancing
        
        # Checkpoint every 50k steps
        if step % 50000 == 0 and step > 0:
            torch.save({
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'curriculum_level': env.curriculum_level,
                'steps_in_stage': steps_in_current_stage,
            }, f"{log_dir}/ckpt_{step}.pt")
            print(f"💾 Checkpoint: ckpt_{step}.pt")
    
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'curriculum_level': env.curriculum_level,
    }, f"{log_dir}/final.pt")
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"Final stage: {STAGE_NAMES[env.curriculum_level]}")
    print(f"💾 Model: {log_dir}/final.pt")
    print(f"{'='*70}\n")
    
    writer.close()
    env.close()
    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        sim.close()
