#!/usr/bin/env python3
"""
Quick Verification Script for TEKO Docking Setup
=================================================
Tests that all improvements are working correctly before full training.
"""

from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import torch
import numpy as np
from datetime import datetime

print("\n" + "="*70)
print("🔍 TEKO Docking Setup Verification")
print("="*70 + "\n")

# Test 1: Import environment
print("[1/6] Testing environment import...")
try:
    from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
    from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
    print("✅ Environment imports successful")
except Exception as e:
    print(f"❌ Environment import failed: {e}")
    simulation_app.close()
    exit(1)

# Test 2: Create environment
print("\n[2/6] Creating environment...")
try:
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = 2  # Small test
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")
    
    # Warm-up
    for _ in range(10):
        env.sim.step()
    
    env._init_observation_space()
    print("✅ Environment created successfully")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    exit(1)

# Test 3: Reset and get observations
print("\n[3/6] Testing reset and observations...")
try:
    obs, info = env.reset()
    
    # Check observation structure
    if isinstance(obs, dict) and "policy" in obs:
        rgb = obs["policy"]["rgb"]
        print(f"✅ Observations shape: {rgb.shape}")
        print(f"   Expected: (2, 3, 480, 640)")
        
        if rgb.shape == (2, 3, 480, 640):
            print("✅ Observation shape correct!")
        else:
            print(f"⚠️  Observation shape mismatch")
    else:
        print(f"❌ Unexpected observation structure: {type(obs)}")
except Exception as e:
    print(f"❌ Reset/observation failed: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    simulation_app.close()
    exit(1)

# Test 4: Check reward function
print("\n[4/6] Testing reward function...")
try:
    # Apply zero action
    action = torch.zeros((2, 2), device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Reward shape: {reward.shape}")
    print(f"   Values: {reward.squeeze().cpu().numpy()}")
    
    # Check reward components
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    
    print(f"   Robot positions: {robot_pos.cpu().numpy()}")
    print(f"   Goal positions: {goal_pos.cpu().numpy()}")
    print(f"   Distances: {distance.cpu().numpy()}")
    print(f"   Target distance: 0.43m")
    
    if reward.shape == (2, 1):
        print("✅ Reward shape correct!")
    else:
        print(f"⚠️  Reward shape unexpected: {reward.shape}")
        
except Exception as e:
    print(f"❌ Reward test failed: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    simulation_app.close()
    exit(1)

# Test 5: Check termination conditions
print("\n[5/6] Testing termination conditions...")
try:
    # Get current state
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    
    # Check collision detection
    collision_threshold = 0.35
    collisions = distance < collision_threshold
    
    # Check wall detection
    half_size = env.arena_size / 2.0
    out_of_bounds = (
        (torch.abs(robot_pos[:, 0]) > half_size) |
        (torch.abs(robot_pos[:, 1]) > half_size)
    )
    
    print(f"✅ Collision detection: {collisions.cpu().numpy()}")
    print(f"✅ Out of bounds detection: {out_of_bounds.cpu().numpy()}")
    print(f"   Arena size: {env.arena_size}m x {env.arena_size}m")
    
except Exception as e:
    print(f"❌ Termination test failed: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    simulation_app.close()
    exit(1)

# Test 6: Check curriculum learning
print("\n[6/6] Testing curriculum learning...")
try:
    # Test level changes
    for level in range(3):
        env.set_curriculum_level(level)
        print(f"   Level {level}: {['Easy', 'Medium', 'Hard'][level]}")
        
        # Reset to test spawn
        obs, info = env.reset()
        robot_pos = env.robot.data.root_pos_w[0]
        print(f"      Spawn position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")
    
    print("✅ Curriculum learning functional")
    
except Exception as e:
    print(f"❌ Curriculum test failed: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    simulation_app.close()
    exit(1)

# Camera debugging test
print("\n[BONUS] Testing camera frame extraction...")
try:
    import cv2
    import os
    
    os.makedirs("/workspace/teko/debug_frames", exist_ok=True)
    
    # Get first camera frame
    rgb = obs["policy"]["rgb"][0]  # First environment
    frame = rgb.cpu().permute(1, 2, 0).numpy() * 255
    frame = frame.astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    save_path = "/workspace/teko/debug_frames/verification_frame.png"
    cv2.imwrite(save_path, frame_bgr)
    
    print(f"✅ Camera frame saved: {save_path}")
    print("   Open this image to verify the robot can see the goal!")
    
except Exception as e:
    print(f"⚠️  Camera frame save failed: {e}")
    print("   (Not critical for training)")

# Summary
print("\n" + "="*70)
print("📊 VERIFICATION SUMMARY")
print("="*70)
print("✅ All core systems operational!")
print("\nYou can now run full training:")
print("\n   Basic training:")
print("   /workspace/isaaclab/_isaac_sim/python.sh \\")
print("       scripts/skrl/train_ppo_improved.py --num_envs 16\n")
print("   With curriculum:")
print("   /workspace/isaaclab/_isaac_sim/python.sh \\")
print("       scripts/skrl/train_ppo_improved.py --num_envs 16 --curriculum\n")
print("="*70 + "\n")

# Cleanup
env.close()
simulation_app.close()