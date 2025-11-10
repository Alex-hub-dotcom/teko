#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Vision-Based Docking — PPO Training (Final with Logging)
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--timesteps", type=int, default=5000)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli=args)
simulation_app = app_launcher.app

import os
from datetime import datetime
import torch, torch.nn as nn, gymnasium as gym, numpy as np
from torch.utils.tensorboard import SummaryWriter

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder


# ======================================================================
# Wrappers
# ======================================================================
class RGBBoxWrapper:
    def __init__(self, env):
        self.env = env
        h, w = env.cfg.camera.height, env.cfg.camera.width
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)
        self.action_space = env.action_space
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)

    def reset(self, *a, **kw):
        obs, info = self.env.reset(*a, **kw)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", obs))
        if isinstance(obs, torch.Tensor):
            obs = obs.float()
        return obs, info

    def step(self, actions):
        obs, r, t, tr, i = self.env.step(actions)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", obs))
        if isinstance(obs, torch.Tensor):
            obs = obs.float()
        return obs, r, t, tr, i

    def __getattr__(self, n):
        return getattr(self.env, n)


class ActionBoxWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.device = getattr(env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = env.observation_space

    def reset(self, *a, **kw):
        return self.env.reset(*a, **kw)

    def step(self, actions):
        if isinstance(actions, tuple):
            actions = actions[0]

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        actions = actions.to(self.device, dtype=torch.float32)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)
        return self.env.step(actions)

    def __getattr__(self, n):
        return getattr(self.env, n)


# ======================================================================
# PPO Models
# ======================================================================
class PolicyNet(GaussianMixin, Model):
    def __init__(self, obs, act, dev, **kw):
        Model.__init__(self, obs, act, dev)
        GaussianMixin.__init__(self, **kw)
        self.encoder = create_visual_encoder("simple", 256, False)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, self.num_actions), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * 480 * 640:
            x = x.view(x.shape[0], 3, 480, 640)
        return self.head(self.encoder(x)), self.log_std, {}


class ValueNet(DeterministicMixin, Model):
    def __init__(self, obs, act, dev, **kw):
        Model.__init__(self, obs, act, dev)
        DeterministicMixin.__init__(self, **kw)
        self.encoder = create_visual_encoder("simple", 256, False)
        self.v = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                               nn.Linear(128, 64), nn.ReLU(),
                               nn.Linear(64, 1))

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * 480 * 640:
            x = x.view(x.shape[0], 3, 480, 640)
        return self.v(self.encoder(x)), {}


# ======================================================================
# Main
# ======================================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 TEKO Vision-Based Docking - PPO Training")
    print("=" * 80 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    base_env = TekoEnv(cfg=cfg)

    # --- FIX: robust tuple-safe prev_actions patch ---
    def _safe_pre_physics_step(actions):
        if isinstance(actions, tuple):
            actions = actions[0]
        
        if not hasattr(base_env, 'prev_actions') or base_env.prev_actions is None:
            base_env.prev_actions = torch.zeros_like(actions)
        elif base_env.prev_actions.shape != actions.shape:
            base_env.prev_actions = torch.zeros_like(actions)
        else:
            prev = base_env.actions if hasattr(base_env, 'actions') else actions
            if isinstance(prev, tuple):
                prev = prev[0]
            base_env.prev_actions.copy_(prev)
        
        base_env.actions = actions
        base_env._lazy_init_articulation()

    base_env._pre_physics_step = _safe_pre_physics_step
    print("⚙️ Applied runtime patch: prev_actions alignment")

    env = RGBBoxWrapper(base_env)
    env = ActionBoxWrapper(env)
    env = wrap_env(env, wrapper="gymnasium")

    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")

    policy = PolicyNet(env.observation_space, env.action_space, device)
    value = ValueNet(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value params: {sum(p.numel() for p in value.parameters()):,}")

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "rollouts": 32,
        "learning_epochs": 10,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4,
        "grad_norm_clip": 0.5,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 0.5,
    })

    memory = RandomMemory(memory_size=ppo_cfg["rollouts"], num_envs=args.num_envs, device=device)

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        cfg=ppo_cfg,
        device=device
    )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/workspace/teko/runs/teko_ppo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=save_dir)
    print(f"✓ TensorBoard: tensorboard --logdir /workspace/teko/runs")

    # Add logging callback
    original_post_interaction = agent.post_interaction
    
    def logged_post_interaction(timestep, timesteps):
        original_post_interaction(timestep, timesteps)
        
        if timestep % 100 == 0 and hasattr(agent, 'tracking_data'):
            data = agent.tracking_data
            
            def get_scalar(key, default=0):
                val = data.get(key, default)
                if isinstance(val, list) and len(val) > 0:
                    return val[-1]
                return val if not isinstance(val, list) else default
            
            reward_mean = get_scalar('Reward / Total reward (mean)')
            policy_loss = get_scalar('Loss / Policy loss')
            value_loss = get_scalar('Loss / Value loss')
            
            writer.add_scalar('Training/Reward', float(reward_mean), timestep)
            writer.add_scalar('Training/Policy_Loss', float(policy_loss), timestep)
            writer.add_scalar('Training/Value_Loss', float(value_loss), timestep)
            
            progress = timestep / timesteps * 100
            print(f"[{timestep:7d}/{timesteps}] {progress:5.1f}% | "
                  f"Reward: {reward_mean:7.2f} | Loss: {policy_loss:.4f}")
    
    agent.post_interaction = logged_post_interaction

    trainer = SequentialTrainer(
        cfg={"timesteps": args.timesteps, "headless": args.headless},
        env=env,
        agents=agent
    )

    print(f"\n✓ Training: {args.num_envs} envs, {args.timesteps:,} steps")
    print(f"✓ Save dir: {save_dir}")
    print("\n" + "=" * 80)
    print("🎓 Starting training...")
    print("=" * 80 + "\n")

    trainer.train()

    model_path = os.path.join(save_dir, "final_model.pt")
    agent.save(model_path)
    
    print(f"\n✅ Training complete!")
    print(f"💾 Model: {model_path}\n")

    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        simulation_app.close()