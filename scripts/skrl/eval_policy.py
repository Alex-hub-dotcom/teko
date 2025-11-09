#!/usr/bin/env python3
"""Evaluate trained PPO policy"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg

def evaluate():
    # Load environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    env = TekoEnv(cfg=cfg)
    
    # Load model
    from skrl.agents.torch.ppo import PPO
    agent = PPO.load(args.model)
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*70}\n")
    
    successes = 0
    total_rewards = []
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 1000:  # Max 1000 steps per episode
            # Get action from policy
            with torch.no_grad():
                actions = agent.act(obs, deterministic=True)[0]
            
            obs, reward, done, truncated, _ = env.step(actions)
            episode_reward += reward.item()
            steps += 1
            
            if done or truncated:
                break
        
        # Check success (reward > 5.0 means close to goal)
        success = episode_reward > 5.0
        successes += success
        total_rewards.append(episode_reward)
        
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Steps={steps}, Success={'✅' if success else '❌'}")
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Success Rate: {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)")
    print(f"  Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"{'='*70}\n")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    evaluate()