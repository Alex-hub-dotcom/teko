#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Vision-Based Docking — PPO Training (Final, No Env Edits)
--------------------------------------------------------------
Runs PPO training with automatic environment patch to keep
actions consistent (fixes oscillation_penalty crash).
"""

import argparse
from isaaclab.app import AppLauncher

# -------------------------------------------------------
# 1. Parse args and launch IsaacSim first
# -------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli=args)
simulation_app = app_launcher.app  # PhysX, omni.* loaded here

# -------------------------------------------------------
# 2. Imports (safe after app start)
# -------------------------------------------------------
import torch
import torch.nn as nn
import os
from datetime import datetime

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder


# =========================================================================================
# Observation Wrapper
# =========================================================================================
class RGBExtractorWrapper:
    """Extracts plain RGB tensor (N, 3, H, W) from env observations."""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", obs))
        return obs, info

    def step(self, actions):
        obs, reward, done, truncated, info = self.env.step(actions)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("rgb", obs))
        return obs, reward, done, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# =========================================================================================
# PPO Models
# =========================================================================================
class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_actions), nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * 480 * 640:
            B = x.shape[0]
            x = x.view(B, 3, 480, 640)
        feats = self.encoder(x)
        mu = self.policy_head(feats)
        return mu, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        if x.ndim == 2 and x.shape[1] == 3 * 480 * 640:
            B = x.shape[0]
            x = x.view(B, 3, 480, 640)
        feats = self.encoder(x)
        v = self.value_head(feats)
        return v, {}


# =========================================================================================
# MAIN
# =========================================================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 TEKO Vision-Based Docking - PPO Training (Final, No Env Edit)")
    print("=" * 80 + "\n")

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------------------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs
    base_env = TekoEnv(cfg=cfg)
    env = RGBExtractorWrapper(base_env)

    # ✅ Runtime hotfix to align actions for oscillation penalty
    def safe_pre_physics_step(actions):
        if getattr(env, "actions", None) is not None:
            prev = getattr(env, "prev_actions", None)
            if prev is None or prev.shape != actions.shape:
                env.prev_actions = torch.zeros_like(actions)
            else:
                env.prev_actions = env.actions.clone()
        env.actions = actions
        env._lazy_init_articulation()
    env._pre_physics_step = safe_pre_physics_step.__get__(env, type(env))
    print("⚙️ Applied runtime patch: safe_pre_physics_step (no env edits)")

    # Wrap for SKRL compatibility
    env = wrap_env(env)

    print(f"[DEBUG] Observation space: {env.observation_space}")
    print(f"[DEBUG] Action space     : {env.action_space}")

    obs, _ = env.reset()
    print(f"[DEBUG] First obs type/shape: {type(obs)} {getattr(obs, 'shape', None)}")

    # -----------------------------------------------------------------------------
    # PPO Setup
    # -----------------------------------------------------------------------------
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

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

    memory = RandomMemory(
        memory_size=ppo_cfg["rollouts"],
        num_envs=args.num_envs,
        device=device
    )

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        cfg=ppo_cfg,
        device=device
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/workspace/teko/runs/teko_ppo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    trainer_cfg = {
        "timesteps": 200_000,
        "headless": args.headless,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"\n✓ Checkpoints will be saved to: {save_dir}")
    print("\n" + "=" * 80)
    print("🎓 Starting training...")
    print("=" * 80 + "\n")

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    trainer.train()

    # -----------------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------------
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\n✅ Training complete!\n💾 Model saved to: {final_path}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        simulation_app.close()
