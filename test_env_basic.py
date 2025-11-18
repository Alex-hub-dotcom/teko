from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless', '--enable_cameras'])
app = AppLauncher(args)
sim = app.app

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg

cfg = TekoEnvCfg()
cfg.scene.num_envs = 2
env = TekoEnv(cfg=cfg)

print("Resetting...")
obs, info = env.reset()
print(f"Obs keys: {obs.keys()}")
print(f"RGB shape: {obs['rgb'].shape}")

print("Stepping...")
import torch
actions = torch.zeros((2, 2), device=env.device)
obs, r, term, trunc, info = env.step(actions)
print(f"✅ Step successful! Reward: {r}")

env.close()
sim.close()
