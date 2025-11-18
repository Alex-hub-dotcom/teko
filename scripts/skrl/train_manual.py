#!/usr/bin/env python3
"""TEKO - Manual PPO (NO SKRL TRAINER)"""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args)
sim = app.app

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder

# Simple CNN-based Policy
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


def main():
    print("🚀 TEKO Manual PPO")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    
    # Env
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = TekoEnv(cfg=cfg)
    
    print(f"✓ Envs: {args.num_envs}")
    
    # Model
    policy = Policy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    print(f"✓ Params: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Logging
    log_dir = f"teko_manual/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"✓ Logs: {log_dir}\n")
    print("="*70)
    print("TRAINING...")
    print("="*70 + "\n")
    
    # Reset
    obs_dict, _ = env.reset()
    obs = obs_dict["rgb"]  # (num_envs, 3, 480, 640)
    
    step = 0
    rollout_len = 64
    
    while step < args.steps:
        # Collect rollout
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        
        for t in range(rollout_len):
            with torch.no_grad():
                mean, value, log_std = policy(obs)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
            
            obs_dict, reward, term, trunc, _ = env.step(action)
            next_obs = obs_dict["rgb"]
            
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value.squeeze(-1))
            logp_buf.append(logp)
            
            obs = next_obs
            step += args.num_envs
            
            if step % 200 == 0:
                print(f"[{step:6d}] R={reward.mean().item():7.2f}")
        
        # PPO Update (simplified - just 1 epoch for speed)
        obs_t = torch.stack(obs_buf)  # (T, N, 3, 480, 640)
        act_t = torch.stack(act_buf)  # (T, N, 2)
        rew_t = torch.stack(rew_buf)  # (T, N)
        val_t = torch.stack(val_buf)  # (T, N)
        logp_old = torch.stack(logp_buf)  # (T, N)
        
        # Flatten
        obs_flat = obs_t.view(-1, 3, 480, 640)
        act_flat = act_t.view(-1, 2)
        
        # Forward
        mean, value, log_std = policy(obs_flat)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp_new = dist.log_prob(act_flat).sum(-1)
        
        # Simple advantage (no GAE)
        adv = rew_t.flatten() - val_t.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Losses
        ratio = (logp_new - logp_old.flatten()).exp()
        clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        value_loss = (value.squeeze() - rew_t.flatten()).pow(2).mean()
        
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        if step % 1000 == 0:
            writer.add_scalar("policy_loss", policy_loss.item(), step)
            writer.add_scalar("value_loss", value_loss.item(), step)
            torch.save(policy.state_dict(), f"{log_dir}/ckpt_{step}.pt")
    
    torch.save(policy.state_dict(), f"{log_dir}/final.pt")
    print(f"\n✅ Done! {log_dir}/final.pt")
    
    writer.close()
    env.close()
    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Stop")
        sim.close()