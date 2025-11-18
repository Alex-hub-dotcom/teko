#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Vision-Based Docking — PPO Training (Camera-only, YAML-driven)
"""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# Argument parsing (training + AppLauncher)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="TEKO Vision-Based Docking — PPO Training (Camera-only)"
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--checkpoint_every", type=int, default=5000, help="Checkpoint interval (timesteps)")

# Add Isaac Lab / AppLauncher arguments (device, headless, experience, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse CLI
args_cli, _ = parser.parse_known_args()

# Always enable cameras (we are vision-based)
if not hasattr(args_cli, "enable_cameras") or args_cli.enable_cameras is False:
    args_cli.enable_cameras = True

# Launch Isaac app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports that require Isaac Sim to be initialized
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg
from teko.tasks.direct.teko.teko_brain.cnn_model import create_visual_encoder
# If you have dedicated PolicyNetwork/ValueNetwork classes, you can import them instead:
# from teko.tasks.direct.teko.teko_brain.ppo_policy import PolicyNetwork, ValueNetwork


# -----------------------------------------------------------------------------
# Simple vision-based models (if you prefer using this instead of ppo_policy.py)
# -----------------------------------------------------------------------------
class PolicyNetwork(GaussianMixin, Model):
    """Gaussian policy over 2D continuous actions using CNN encoder on RGB frames."""

    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)

        # CNN encoder from your teko_brain module
        self.encoder = create_visual_encoder("simple", 256, False)

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh(),  # actions in [-1, 1]
        )

        # Learnable log-std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # inputs["states"]: (num_envs, 3, H, W)
        x = inputs["states"]
        features = self.encoder(x)
        actions = self.head(features)
        return actions, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    """State-value function using same CNN encoder."""

    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)

        self.encoder = create_visual_encoder("simple", 256, False)

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        features = self.encoder(x)
        value = self.value_head(features)
        return value, {}


# -----------------------------------------------------------------------------
# Wrappers: extract RGB image and ensure correct action format
# -----------------------------------------------------------------------------
class RGBBoxWrapper:
    """Expose only the RGB camera as observation: (3, H, W) in [0, 1]."""

    def __init__(self, env):
        self.env = env
        h, w = env.cfg.camera.height, env.cfg.camera.width
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, h, w),
            dtype=np.float32,
        )
        # Use the existing action_space from the Isaac env
        self.action_space = env.action_space
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if isinstance(obs, dict):
            # "rgb" key contains the camera tensor: (num_envs, 3, H, W)
            obs = obs.get("rgb", next(iter(obs.values())))
        # make sure it's float32
        obs = obs.to(torch.float32)
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        if isinstance(obs, dict):
            obs = obs.get("rgb", next(iter(obs.values())))
        obs = obs.to(torch.float32)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        # Delegate everything else to the base env
        return getattr(self.env, name)


class ActionBoxWrapper:
    """Make sure actions are a Box(-1, 1, (2,)) and properly shaped tensors."""

    def __init__(self, env):
        self.env = env
        self.num_envs = getattr(env.scene.cfg, "num_envs", 1)
        self.device = getattr(
            env, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = env.observation_space

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        # Handle different formats (tuple, numpy, etc.)
        if isinstance(actions, tuple):
            actions = actions[0]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)

        actions = actions.to(self.device, dtype=torch.float32)

        # Ensure shape: (num_envs, 2)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).repeat(self.num_envs, 1)

        return self.env.step(actions)

    def __getattr__(self, name):
        return getattr(self.env, name)


# -----------------------------------------------------------------------------
# YAML helper
# -----------------------------------------------------------------------------
def load_yaml_cfg():
    """Load PPO/skrl config from YAML if present."""
    # /workspace/teko/scripts/skrl/train_teko_direct.py
    # -> /workspace/teko/source/teko/teko/tasks/direct/teko/teko_brain/skrl_ppo_cfg.yaml
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "source",
        "teko",
        "teko",
        "tasks",
        "direct",
        "teko",
        "teko_brain",
        "skrl_ppo_cfg.yaml",
    )
    yaml_path = os.path.normpath(yaml_path)

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[INFO] Loaded PPO config from YAML: {yaml_path}")
        return cfg
    else:
        print("[WARN] YAML PPO config not found; using PPO defaults")
        return {}


# -----------------------------------------------------------------------------
# Main Training
# -----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 78)
    print("🚀 TEKO Vision-Based Docking — PPO Training (Camera-only)")
    print("=" * 78 + "\n")

    # -------------------------------------------------------------------------
    # Seeds
    # -------------------------------------------------------------------------
    set_seed(args_cli.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"Setting seed: {args_cli.seed}")

    # -------------------------------------------------------------------------
    # Create Isaac Lab environment (TEKO)
    # -------------------------------------------------------------------------
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Set sim device from CLI if present
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    base_env = TekoEnv(cfg=env_cfg)

    # Patch: tuple-safe prev_actions in _pre_physics_step
    def _safe_pre_physics_step(actions):
        if isinstance(actions, tuple):
            actions = actions[0]
        if not hasattr(base_env, "prev_actions") or base_env.prev_actions is None:
            base_env.prev_actions = torch.zeros_like(actions)
        elif base_env.prev_actions.shape != actions.shape:
            base_env.prev_actions = torch.zeros_like(actions)
        else:
            prev = getattr(base_env, "actions", actions)
            if isinstance(prev, tuple):
                prev = prev[0]
            base_env.prev_actions.copy_(prev)

        base_env.actions = actions
        base_env._lazy_init_articulation()

    base_env._pre_physics_step = _safe_pre_physics_step
    print("⚙️ Applied runtime patch: tuple-safe prev_actions")

    # Wrap environment for vision and SKRL
    env = RGBBoxWrapper(base_env)
    env = ActionBoxWrapper(env)
    env = wrap_env(env, wrapper="gymnasium")  # SkrlVecEnvWrapper equivalent

    print(f"[skrl:INFO] Environment wrapper: gymnasium")
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space:      {env.action_space}")
    print(f"✓ Num envs:          {env.num_envs}\n")

    # -------------------------------------------------------------------------
    # Load YAML config and build PPO config
    # -------------------------------------------------------------------------
    yaml_cfg = load_yaml_cfg()

    # Base PPO config
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()

    # Merge "agent" config from YAML (except "class")
    yaml_agent = yaml_cfg.get("agent", {}) or {}
    yaml_agent = {k: v for k, v in yaml_agent.items() if k != "class"}
    ppo_cfg.update(yaml_agent)

    # Configure experiment logging
    exp_cfg = ppo_cfg.get("experiment", {}) or {}
    if not exp_cfg.get("directory"):
        exp_cfg["directory"] = "teko_vision_docking"
    if not exp_cfg.get("experiment_name"):
        exp_cfg["experiment_name"] = f"teko_ppo_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ppo_cfg["experiment"] = exp_cfg

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------
    # If you want to pass YAML model kwargs (clip_actions, log_std, etc.):
    models_yaml = yaml_cfg.get("models", {}) or {}
    policy_kwargs = {k: v for k, v in models_yaml.get("policy", {}).items() if k != "class"}
    value_kwargs = {k: v for k, v in models_yaml.get("value", {}).items() if k != "class"}

    policy = PolicyNetwork(env.observation_space, env.action_space, device, **policy_kwargs)
    value = ValueNetwork(env.observation_space, env.action_space, device, **value_kwargs)

    print(f"✓ Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"✓ Value  params: {sum(p.numel() for p in value.parameters()):,}")

    # -------------------------------------------------------------------------
    # Memory & Agent
    # -------------------------------------------------------------------------
    rollouts = int(ppo_cfg.get("rollouts", 64))
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=ppo_cfg,
    )

    log_root = agent.experiment_dir  # e.g. "teko_vision_docking"
    log_dir = os.path.join(log_root, agent.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"✓ TensorBoard: tensorboard --logdir {log_root}")
    print(f"✓ Training: {env.num_envs} envs, {args_cli.timesteps:,} steps")
    print(f"✓ Save dir: {log_dir}\n")

    writer = SummaryWriter(log_dir=log_dir)

    # -------------------------------------------------------------------------
    # Custom logging hook into PPO (for progress + TensorBoard)
    # -------------------------------------------------------------------------
    original_post_interaction = agent.post_interaction

    def logged_post_interaction(*, timestep, timesteps):
        # Let PPO do its normal bookkeeping / updates
        original_post_interaction(timestep=timestep, timesteps=timesteps)

        # Log every 200 steps (and t=0 for the initial NaN)
        if timestep % 200 != 0 and timestep != 0:
            return

        # Reward mean over last batch in memory
        reward_mean = 0.0
        try:
            rewards = agent.memory.get_tensor_by_name("rewards")
            if rewards is not None and rewards.numel() > 0:
                reward_mean = float(rewards.mean().item())
        except Exception:
            reward_mean = 0.0

        # Get latest policy / value losses from tracking_data
        policy_loss = 0.0
        value_loss = 0.0
        td = getattr(agent, "tracking_data", None)
        if isinstance(td, dict):
            pl = td.get("Loss / Policy loss", None)
            vl = td.get("Loss / Value loss", None)
            if isinstance(pl, (list, tuple)) and len(pl) > 0:
                policy_loss = float(pl[-1])
            if isinstance(vl, (list, tuple)) and len(vl) > 0:
                value_loss = float(vl[-1])

        # TensorBoard scalars
        writer.add_scalar("Training/Reward_mean", reward_mean, timestep)
        writer.add_scalar("Training/Policy_loss", policy_loss, timestep)
        writer.add_scalar("Training/Value_loss", value_loss, timestep)

        # Optional: log reward components if env exposes them
        rc = getattr(base_env, "reward_components", None)
        if isinstance(rc, dict):
            for name, values in rc.items():
                if not values:
                    continue
                writer.add_scalar(
                    f"Rewards/{name}",
                    float(np.mean(values[-50:])),
                    timestep,
                )

        pct = 100.0 * timestep / timesteps if timesteps > 0 else 0.0
        print(
            f"[{timestep:7d}/{timesteps}] {pct:5.1f}% | "
            f"R̄ {reward_mean:8.3f} | π-loss {policy_loss:8.5f} | v-loss {value_loss:7.2f}"
        )

        # Checkpoints
        if (
            args_cli.checkpoint_every > 0
            and timestep > 0
            and timestep % args_cli.checkpoint_every == 0
        ):
            ckpt_path = os.path.join(log_dir, f"ckpt_step_{timestep}.pt")
            agent.save(ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

    agent.post_interaction = logged_post_interaction

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer_cfg = {
        "timesteps": args_cli.timesteps,
        "headless": args_cli.headless,
    }

    print("=" * 78)
    print("🎓 Starting PPO vision training...")
    print("=" * 78 + "\n")

    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    final_path = os.path.join(log_dir, "final_model.pt")
    agent.save(final_path)
    print("\n✅ Training complete!")
    print(f"💾 Model: {final_path}\n")

    writer.close()
    env.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    finally:
        simulation_app.close()
