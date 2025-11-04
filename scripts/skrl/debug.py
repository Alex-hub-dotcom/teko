#!/usr/bin/env python3
"""
Quick Debug: Test Observation Shapes
====================================
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch

print("\n" + "="*70)
print("🔍 Testing Observation Shapes")
print("="*70 + "\n")

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

env_cfg = TekoEnvCfg()
env_cfg.scene.num_envs = 2
env = TekoEnv(cfg=env_cfg, render_mode=None)

# Warm-up
for _ in range(10):
    env.sim.step()

env._init_observation_space()
obs, info = env.reset()

print(f"Observation type: {type(obs)}")
print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")

if isinstance(obs, dict) and "policy" in obs:
    policy_obs = obs["policy"]
    print(f"Policy obs type: {type(policy_obs)}")
    print(f"Policy obs keys: {policy_obs.keys() if isinstance(policy_obs, dict) else 'N/A'}")
    
    if isinstance(policy_obs, dict) and "rgb" in policy_obs:
        rgb = policy_obs["rgb"]
        print(f"\n✅ RGB Shape: {rgb.shape}")
        print(f"   Expected: (2, 3, 480, 640)")
        print(f"   Dtype: {rgb.dtype}")
        print(f"   Device: {rgb.device}")
        print(f"   Min/Max: {rgb.min():.3f} / {rgb.max():.3f}")
        
        if rgb.shape == (2, 3, 480, 640):
            print("\n🎉 Shape is CORRECT!")
        else:
            print(f"\n❌ Shape is WRONG!")
            print(f"   Got: {rgb.shape}")
            print(f"   Expected: (2, 3, 480, 640)")

env.close()
simulation_app.close()