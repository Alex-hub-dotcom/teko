#!/usr/bin/env python3
"""
TEKO Vision Docking - Production PPO Training
==============================================
- Proper GAE (Generalized Advantage Estimation)
- Multi-epoch PPO updates
- Comprehensive logging (success rate, episode length, rewards)
- Checkpointing every 10k steps
- TensorBoard visualization
"""

import argparse
import os
from datetime import datetime
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
parser.add_argument("--steps", type=int, default=100000, help="Total training steps")
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


# ============================================================================
# PPO Policy Network
# ============================================================================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = create_visual_encoder("simple", 256, False)
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh()
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(2))
    
    def forward(self, obs):
        """Forward pass returning action mean, value, and log_std."""
        feat = self.encoder(obs)
        return self.actor(feat), self.critic(feat), self.log_std
    
    def act(self, obs):
        """Sample action from policy."""
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value.squeeze(-1)
    
    def evaluate(self, obs, actions):
        """Evaluate actions (for PPO update)."""
        mean, value, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, value.squeeze(-1), entropy


# ============================================================================
# GAE (Generalized Advantage Estimation)
# ============================================================================
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Compute GAE advantages and returns.
    
    Args:
        rewards: (T, N) tensor
        values: (T, N) tensor  
        dones: (T, N) tensor
        gamma: discount factor
        lambda_: GAE lambda
    
    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return advantages, returns


# ============================================================================
# PPO Update
# ============================================================================
def ppo_update(policy, optimizer, obs, actions, logp_old, advantages, returns,
               epochs=8, batch_size=256, clip_ratio=0.2, value_clip=0.2):
    """
    PPO update with clipping.
    
    Returns:
        policy_loss, value_loss, entropy (mean over all updates)
    """
    T, N = obs.shape[0], obs.shape[1]
    total_samples = T * N
    
    # Flatten everything
    obs_flat = obs.view(total_samples, 3, 480, 640)
    actions_flat = actions.view(total_samples, 2)
    logp_old_flat = logp_old.view(-1)
    advantages_flat = advantages.view(-1)
    returns_flat = returns.view(-1)
    
    # Normalize advantages
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
    
    policy_losses = []
    value_losses = []
    entropies = []
    
    for epoch in range(epochs):
        # Random permutation for minibatches
        indices = torch.randperm(total_samples, device=obs.device)
        
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            if end > total_samples:
                break
            
            mb_idx = indices[start:end]
            
            # Minibatch data
            mb_obs = obs_flat[mb_idx]
            mb_actions = actions_flat[mb_idx]
            mb_logp_old = logp_old_flat[mb_idx]
            mb_adv = advantages_flat[mb_idx]
            mb_returns = returns_flat[mb_idx]
            
            # Evaluate
            logp, value, entropy = policy.evaluate(mb_obs, mb_actions)
            
            # Policy loss (clipped)
            ratio = (logp - mb_logp_old).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
            policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean()
            
            # Value loss (clipped)
            value_pred_clipped = value
            if value_clip is not None:
                value_pred_clipped = torch.clamp(
                    value,
                    mb_returns - value_clip,
                    mb_returns + value_clip
                )
            value_loss = F.mse_loss(value_pred_clipped, mb_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
    
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("\n" + "="*70)
    print("🚀 TEKO VISION DOCKING - PRODUCTION PPO")
    print("="*70)
    print(f"Environments: {args.num_envs}")
    print(f"Total steps:  {args.steps:,}")
    print(f"Learning rate: {args.lr}")
    print(f"Rollout length: {args.rollout_len}")
    print(f"PPO epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70 + "\n")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    
    # Environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = TekoEnv(cfg=cfg)
    
    # Model
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}\n")
    
    # Logging
    log_dir = f"teko_production/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"📊 TensorBoard: tensorboard --logdir teko_production")
    print(f"💾 Checkpoints: {log_dir}\n")
    print("="*70)
    print("TRAINING STARTED")
    print("="*70 + "\n")
    
    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["rgb"]
    
    # Episode tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    current_episode_reward = torch.zeros(args.num_envs, device=device)
    current_episode_length = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
    
    step = 0
    update_count = 0
    
    while step < args.steps:
        # Collect rollout
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        
        for t in range(args.rollout_len):
            with torch.no_grad():
                action, logp, value = policy.act(obs)
            
            obs_dict, reward, term, trunc, info = env.step(action)
            next_obs = obs_dict["rgb"]
            done = term | trunc
            
            # Track episodes
            current_episode_reward += reward
            current_episode_length += 1
            
            # Log completed episodes
            for i in range(args.num_envs):
                if done[i]:
                    episode_rewards.append(current_episode_reward[i].item())
                    episode_lengths.append(current_episode_length[i].item())
                    # Check if success (from info or high reward)
                    success = reward[i] > 50  # Success bonus in reward
                    episode_successes.append(1.0 if success else 0.0)
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
        
        # Stack rollout
        obs_t = torch.stack(obs_buf)  # (T, N, 3, 480, 640)
        act_t = torch.stack(act_buf)  # (T, N, 2)
        rew_t = torch.stack(rew_buf)  # (T, N)
        val_t = torch.stack(val_buf)  # (T, N)
        logp_t = torch.stack(logp_buf)  # (T, N)
        done_t = torch.stack(done_buf)  # (T, N)
        
        # Compute GAE
        with torch.no_grad():
            advantages, returns = compute_gae(rew_t, val_t, done_t)
        
        # PPO update
        policy_loss, value_loss, entropy = ppo_update(
            policy, optimizer, obs_t, act_t, logp_t, advantages, returns,
            epochs=args.epochs, batch_size=args.batch_size
        )
        
        update_count += 1
        
         # Logging (print every update)
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        success_rate = np.mean(episode_successes) if episode_successes else 0
            
        writer.add_scalar("train/reward", mean_reward, step)
        writer.add_scalar("train/episode_length", mean_length, step)
        writer.add_scalar("train/success_rate", success_rate, step)
        writer.add_scalar("train/policy_loss", policy_loss, step)
        writer.add_scalar("train/value_loss", value_loss, step)
        writer.add_scalar("train/entropy", entropy, step)
            
        print(f"[{step:7d}] R={mean_reward:7.2f} | Len={mean_length:5.1f} | "
            f"Success={success_rate*100:4.1f}% | πL={policy_loss:.4f} | vL={value_loss:.3f}")
        # Checkpoint
        if step % 10000 == 0 and step > 0:
            torch.save({
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
            }, f"{log_dir}/ckpt_{step}.pt")
            print(f"💾 Checkpoint saved: ckpt_{step}.pt")
    
    # Final save
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, f"{log_dir}/final.pt")
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"💾 Final model: {log_dir}/final.pt")
    print(f"📊 TensorBoard: tensorboard --logdir teko_production")
    print(f"{'='*70}\n")
    
    writer.close()
    env.close()
    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sim.close()