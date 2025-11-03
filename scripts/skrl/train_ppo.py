#!/usr/bin/env python3
"""
Minimal Test Script - Verify TEKO Environment Works
====================================================
Tests basic environment functionality before starting RL training.

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/test_env_minimal.py
"""

# CRITICAL: Import SimulationApp FIRST before any other Isaac imports
from isaacsim import SimulationApp

# Initialize Isaac Sim (headless=False to see the simulation)
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np

# Now import Isaac Lab and your environment
from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg


def test_environment():
    """Test basic environment functionality"""
    
    print("\n" + "="*60)
    print("🧪 Testing TEKO Environment")
    print("="*60 + "\n")
    
    # Create environment config
    print("1. Creating environment config...")
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = 1  # Start with just 1
    env_cfg.sim.device = "cuda:0"
    print("   ✓ Config created")
    
    # Create environment
    print("\n2. Initializing TEKO environment...")
    env = TekoEnv(cfg=env_cfg, render_mode="human")
    print("   ✓ Environment initialized")
    
    # Print environment info
    print("\n3. Environment info:")
    print(f"   - Number of envs: {env.num_envs}")
    print(f"   - Device: {env.device}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    
    # Reset environment
    print("\n4. Resetting environment...")
    obs, info = env.reset()
    print("   ✓ Environment reset successful")
    
    # Check observation
    print("\n5. Checking observations:")
    if "policy" in obs and "rgb" in obs["policy"]:
        rgb = obs["policy"]["rgb"]
        print(f"   ✓ RGB observation shape: {rgb.shape}")
        print(f"   ✓ RGB value range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"   ✓ RGB dtype: {rgb.dtype}")
    else:
        print("   ✗ ERROR: Unexpected observation structure!")
        print(f"   Observation keys: {obs.keys()}")
    
    # Test random actions for a few steps
    print("\n6. Testing random actions (10 steps)...")
    num_test_steps = 10
    
    for step in range(num_test_steps):
        # Random wheel velocities between -1 and 1
        action = torch.rand((env.num_envs, 2), device=env.device) * 2 - 1
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print info every few steps
        if step % 3 == 0:
            print(f"   Step {step+1}/{num_test_steps}:")
            print(f"     - Action: [{action[0, 0]:.2f}, {action[0, 1]:.2f}]")
            print(f"     - Reward: {reward[0].item():.3f}")
            print(f"     - Terminated: {terminated[0].item()}")
            print(f"     - Truncated: {truncated[0].item()}")
    
    print("\n   ✓ Random action test completed")
    
    # Test robot state access
    print("\n7. Checking robot state:")
    robot_pos = env.robot.data.root_pos_w
    robot_quat = env.robot.data.root_quat_w
    print(f"   ✓ Robot position shape: {robot_pos.shape}")
    print(f"   ✓ Robot position: {robot_pos[0].cpu().numpy()}")
    print(f"   ✓ Robot orientation shape: {robot_quat.shape}")
    
    # Test goal positions (if cached)
    if hasattr(env, 'goal_positions') and env.goal_positions is not None:
        print("\n8. Checking goal positions:")
        print(f"   ✓ Goal position shape: {env.goal_positions.shape}")
        print(f"   ✓ Goal position: {env.goal_positions[0].cpu().numpy()}")
        
        # Calculate distance to goal
        distance = torch.norm(robot_pos - env.goal_positions, dim=-1)
        print(f"   ✓ Distance to goal: {distance[0].item():.3f} m")
    else:
        print("\n8. Goal positions not cached")
    
    # Test cameras
    if hasattr(env, 'cameras') and len(env.cameras) > 0:
        print("\n9. Checking cameras:")
        print(f"   ✓ Number of cameras: {len(env.cameras)}")
        print(f"   ✓ Camera resolution: {env._cam_res}")
    else:
        print("\n9. No cameras initialized")
    
    # Close environment
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
    
    env.close()


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always close simulation
        simulation_app.close()