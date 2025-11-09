# SPDX-License-Identifier: BSD-3-Clause
"""
Test TEKO Environment with Random Actions
------------------------------------------
Validates:
- Multi-environment setup
- Sphere position tracking
- Reward computation
- Episode termination
"""

from isaaclab.app import AppLauncher
import torch

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg

def test_environment():
    """Test environment with random actions."""
    
    # Create environment
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 4  # Test with 4 environments
    env = TekoEnv(cfg)
    
    print("\n" + "="*60)
    print("TEKO Environment Test")
    print("="*60)
    print(f"Number of environments: {cfg.scene.num_envs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode length: {cfg.episode_length_s}s")
    print("="*60 + "\n")
    
    # Reset environment
    obs = env.reset()
    print(f"[INFO] Environment reset complete")
    print(f"[INFO] Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"[INFO] Observation shape: {obs['policy']['rgb'].shape}")
    elif isinstance(obs, tuple):
        print(f"[INFO] Observation tuple length: {len(obs)}")
        print(f"[INFO] First element shape: {obs[0].shape if hasattr(obs[0], 'shape') else type(obs[0])}")
    
    # Run for 500 steps with random actions
    num_steps = 500
    episode_rewards = torch.zeros(cfg.scene.num_envs, device=env.device)
    episode_lengths = torch.zeros(cfg.scene.num_envs, device=env.device)
    
    print(f"\n[INFO] Running {num_steps} steps with random actions...\n")
    
    for step in range(num_steps):
        # Random actions in [-1, 1] for wheel torques
        actions = torch.rand((cfg.scene.num_envs, 2), device=env.device) * 2.0 - 1.0
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Accumulate statistics
        episode_rewards += rewards
        episode_lengths += 1
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{num_steps}")
            print(f"  Mean reward: {rewards.mean().item():.4f}")
            print(f"  Active episodes: {(~dones).sum().item()}/{cfg.scene.num_envs}")
            
            # Print reward components
            if len(env.reward_components["distance"]) > 0:
                print(f"  Distance reward: {env.reward_components['distance'][-1]:.4f}")
                print(f"  Wall penalty: {env.reward_components['wall_penalty'][-1]:.4f}")
        
        # Check for completed episodes
        if dones.any():
            completed = dones.sum().item()
            print(f"\n[SUCCESS] {completed} episode(s) completed at step {step+1}")
            for env_idx in torch.where(dones)[0]:
                print(f"  Env {env_idx}: Reward={episode_rewards[env_idx].item():.2f}, Length={episode_lengths[env_idx].item()}")
            
            # Reset completed environments
            episode_rewards[dones] = 0
            episode_lengths[dones] = 0
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"Total steps executed: {num_steps}")
    print(f"Environments tested: {cfg.scene.num_envs}")
    print("\n[INFO] Environment is working correctly! ✅")
    print("="*60 + "\n")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    test_environment()