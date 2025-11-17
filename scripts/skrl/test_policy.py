#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Test TEKO trained policy with visualization
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
args, _ = parser.parse_known_args()

# GUI mode for visualization
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder


# ======================================================================
# Policy Network (same as training)
# ======================================================================
class PolicyNet(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.encoder = create_visual_encoder("simple", 256, False)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
    
    def forward(self, x):
        # x shape: (batch, 3, 480, 640)
        features = self.encoder(x)
        actions = self.head(features)
        return actions


# ======================================================================
# Main
# ======================================================================
def main():
    print("\n" + "=" * 78)
    print("🎮 TEKO Policy Test - Visual Evaluation")
    print("=" * 78 + "\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup environment (1 env for visualization)
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    env = TekoEnv(cfg=cfg, render_mode="human")

    print(f"✓ Environment created")
    print(f"✓ Loading checkpoint: {args.checkpoint}")

    # Load policy
    policy = PolicyNet(action_dim=2).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract policy state dict from checkpoint
    if "policy" in checkpoint:
        policy_state = checkpoint["policy"]
        if "model" in policy_state:
            policy.load_state_dict(policy_state["model"], strict=False)
        else:
            policy.load_state_dict(policy_state, strict=False)
    else:
        policy.load_state_dict(checkpoint, strict=False)
    
    policy.eval()
    print(f"✓ Policy loaded successfully\n")

   # Run episodes
    total_successes = 0
    total_steps = 0

    for episode in range(args.num_episodes):
        print(f"{'='*78}")
        print(f"Episode {episode + 1}/{args.num_episodes}")
        print(f"{'='*78}")
        
        obs, _ = env.reset()
        obs = obs["policy"]  # Extract RGB observation (shape: [1, 3, 480, 640])
        
        episode_reward = 0.0
        episode_steps = 0
        success = False
        
        for step in range(1000):  # Max 1000 steps per episode
            # Get action from policy
            with torch.no_grad():
                # obs is already [1, 3, 480, 640], don't add extra dimension
                action = policy(obs).squeeze(0)  # Output: [2]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
            obs = obs["policy"]  # shape: [1, 3, 480, 640]
            
            # Get distance info
            _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
            distance = surface_xy[0].item()
            
            episode_reward += reward[0].item()
            episode_steps += 1
            total_steps += 1
            
            # Print every 50 steps
            if step % 50 == 0:
                print(f"  Step {step:4d} | Distance: {distance:.4f}m | Reward: {reward[0].item():7.3f}")
            
            # Check success
            if distance < 0.03:
                success = True
                total_successes += 1
                print(f"  ✅ SUCCESS at step {step}! Distance: {distance:.4f}m")
            
            # Check termination
            if terminated[0] or truncated[0]:
                if not success:
                    print(f"  ❌ Episode terminated without success. Final distance: {distance:.4f}m")
                break
        
        print(f"\nEpisode Summary:")
        print(f"  Total Reward: {episode_reward:8.2f}")
        print(f"  Steps:        {episode_steps}")
        print(f"  Success:      {'✅ YES' if success else '❌ NO'}")
        print()

    # Final statistics
    print(f"\n{'='*78}")
    print(f"📊 FINAL STATISTICS")
    print(f"{'='*78}")
    print(f"  Total Episodes:    {args.num_episodes}")
    print(f"  Successes:         {total_successes}")
    print(f"  Success Rate:      {100.0 * total_successes / args.num_episodes:.1f}%")
    print(f"  Avg Steps/Episode: {total_steps / args.num_episodes:.1f}")
    print(f"{'='*78}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()