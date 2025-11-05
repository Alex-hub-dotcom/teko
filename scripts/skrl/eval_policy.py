#!/usr/bin/env python3
"""
Evaluate Trained TEKO Docking Policy
====================================
Tests the trained model and reports success rate, average rewards, etc.
"""

from isaacsim import SimulationApp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of test episodes")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--render_video", action="store_true", help="Save video of episodes")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import torch
import numpy as np
from datetime import datetime

from source.teko.teko.tasks.direct.teko.teko_env import TekoEnv
from source.teko.teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import gymnasium as gym
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder

print("\n" + "="*70)
print("🔍 TEKO Docking Policy Evaluation")
print("="*70 + "\n")

# =====================================================================
# Wrapper (same as training)
# =====================================================================
class SimpleEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_envs = 1  # Single env for evaluation
        self.num_agents = 1
        self.device = getattr(env, "device", "cuda:0")

        H, W = env.cfg.camera.height, env.cfg.camera.width
        self.H, self.W, self.C = H, W, 3
        self.flat_dim = self.C * H * W

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.flat_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        obs, info = self.env.reset()
        rgb = self._extract_rgb(obs).to(self.device).float()
        return rgb.view(self.num_envs, -1).contiguous(), info

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device)

        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = self._extract_rgb(obs).to(self.device).float()
        
        reward = reward.reshape(-1, 1).contiguous()
        terminated = terminated.reshape(-1, 1).contiguous()
        truncated = truncated.reshape(-1, 1).contiguous()

        return rgb.view(self.num_envs, -1).contiguous(), reward, terminated, truncated, info

    def _extract_rgb(self, obs):
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], dict):
                return obs["policy"]["rgb"]
            elif "rgb" in obs:
                return obs["rgb"]
        return obs

    def render(self):
        pass

    def close(self):
        self.env.close()


# =====================================================================
# Models (same as training)
# =====================================================================
class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        self.C, self.H, self.W = 3, 480, 640
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_actions), torch.nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]
        x = x.view(x.shape[0], self.C, self.H, self.W)
        feat = self.encoder(x)
        act = self.policy(feat)
        return act, self.log_std, {}


class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        self.C, self.H, self.W = 3, 480, 640
        self.encoder = create_visual_encoder("simple", feature_dim=256, pretrained=False)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        x = x.view(x.shape[0], self.C, self.H, self.W)
        feat = self.encoder(x)
        return self.value(feat), {}


# =====================================================================
# Evaluation
# =====================================================================
def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.num_episodes}\n")

    # Create environment (single env for evaluation)
    env_cfg = TekoEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = str(device)
    env = TekoEnv(cfg=env_cfg, render_mode=None if args.headless else "human")

    # Warm-up
    for _ in range(10):
        env.sim.step()
    env._init_observation_space()
    _ = env.reset()

    env = SimpleEnvWrapper(env)
    print("✓ Environment created")

    # Create agent and load model
    policy = PolicyNetwork(env.observation_space, env.action_space, device)
    value = ValueNetwork(env.observation_space, env.action_space, device)
    
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    memory = RandomMemory(memory_size=16, num_envs=1, device=device)
    
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=memory,
        cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # Load trained weights
    print(f"Loading model from: {args.model_path}")
    agent.load(args.model_path)
    print("✓ Model loaded\n")

    # Evaluation metrics
    successes = 0
    collisions = 0
    out_of_bounds = 0
    episode_rewards = []
    episode_lengths = []
    final_distances = []

    print("="*70)
    print("Running evaluation...")
    print("="*70 + "\n")

    for episode in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from policy (deterministic = no exploration)
            with torch.no_grad():
                action, _, _ = agent.act(obs, timestep=0, timesteps=1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward.item()
            steps += 1
            done = terminated.item() or truncated.item()
            
            # Update visualization
            if not args.headless:
                simulation_app.update()
        
        # Get final state
        robot_pos = env.env.robot.data.root_pos_w[0].cpu().numpy()
        goal_pos = env.env.goal_positions[0].cpu().numpy()
        final_distance = np.linalg.norm(robot_pos - goal_pos)
        
        # Check termination reason
        target_distance = 0.43
        tolerance = 0.01
        success = abs(final_distance - target_distance) < tolerance
        collision = final_distance < 0.35
        oob = (abs(robot_pos[0]) > 1.0) or (abs(robot_pos[1]) > 1.0)
        
        if success:
            successes += 1
            status = "✅ SUCCESS"
        elif collision:
            collisions += 1
            status = "💥 COLLISION"
        elif oob:
            out_of_bounds += 1
            status = "🚫 OUT OF BOUNDS"
        else:
            status = "❌ FAILED"
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        final_distances.append(final_distance)
        
        print(f"Episode {episode+1:3d}/{args.num_episodes} | "
              f"Reward: {episode_reward:6.2f} | "
              f"Steps: {steps:4d} | "
              f"Dist: {final_distance:.3f}m | "
              f"{status}")

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Episodes:      {args.num_episodes}")
    print(f"Successes:           {successes} ({100*successes/args.num_episodes:.1f}%)")
    print(f"Collisions:          {collisions} ({100*collisions/args.num_episodes:.1f}%)")
    print(f"Out of Bounds:       {out_of_bounds} ({100*out_of_bounds/args.num_episodes:.1f}%)")
    print(f"Other Failures:      {args.num_episodes - successes - collisions - out_of_bounds}")
    print(f"\nAverage Reward:      {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Len: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Final Dist:  {np.mean(final_distances):.3f}m ± {np.std(final_distances):.3f}m")
    print(f"Target Distance:     0.430m ± 0.010m")
    print("="*70 + "\n")

    # Save results
    results_file = args.model_path.replace("final_model.pt", "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Episodes: {args.num_episodes}\n\n")
        f.write(f"Success Rate: {100*successes/args.num_episodes:.1f}%\n")
        f.write(f"Collision Rate: {100*collisions/args.num_episodes:.1f}%\n")
        f.write(f"Average Reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"Average Final Distance: {np.mean(final_distances):.3f}m\n")
    
    print(f"Results saved to: {results_file}")

    env.close()


if __name__ == "__main__":
    try:
        evaluate()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()