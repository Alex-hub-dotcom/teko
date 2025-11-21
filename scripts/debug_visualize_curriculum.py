#!/usr/bin/env python3
"""
Debug script: visualize all curriculum stages.

- Runs TEKO env with 2 parallel envs
- For each curriculum stage:
    * sets env.curriculum_level
    * reset()
    * takes one zero-action step
    * saves camera images for env 0 and 1 to PNG

Run with:

/workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/teko/scripts/debug_visualize_curriculum.py
"""

import os
import torch
import imageio

# 1) Launch Isaac app FIRST
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(
        headless=False,
        enable_cameras=True,   # <--- THIS is the key line
    )
app = app_launcher.app

# 2) Only now import TEKO / IsaacLab stuff
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES


def main():
    # --- Configure env for debugging ---
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 2
    cfg.enable_curriculum = True

    # If you added these flags in TekoEnvCfg:
    # cfg.debug_boundaries = True      # red arena lines
    # cfg.debug_goal_halo = True       # red halo around static robot

    # Render with GUI
    env = TekoEnv(cfg=cfg, render_mode="human")

    out_dir = os.path.join("/workspace/teko", "debug_stage_views")
    os.makedirs(out_dir, exist_ok=True)

    num_envs = cfg.scene.num_envs
    action_dim = cfg.action_space[0]
    num_stages = len(STAGE_NAMES)

    print(f"[DEBUG] Will capture {num_stages} stages with {num_envs} envs.")
    print(f"[DEBUG] Output folder: {out_dir}")

    for stage in range(num_stages):
        print("\n" + "=" * 70)
        print(f"[STAGE {stage}] {STAGE_NAMES[stage]}")
        print("=" * 70)

        # Set curriculum level and reset
        env.set_curriculum_level(stage)
        obs, _ = env.reset()

        # One zero-action step (physics + camera update)
        zero_action = torch.zeros((num_envs, action_dim), device=env.device)
        obs, _, _, _, _ = env.step(zero_action)

        # Get RGB observations: [num_envs, 3, H, W] in [0, 1]
        rgb = obs["rgb"].detach().cpu()

        for env_id in range(num_envs):
            img = rgb[env_id]  # [3, H, W]
            img = (img * 255.0).clamp(0, 255).byte()
            img = img.permute(1, 2, 0).numpy()  # [H, W, 3]

            fname = os.path.join(
                out_dir,
                f"stage_{stage:02d}_env_{env_id}_cam.png",
            )
            imageio.imwrite(fname, img)
            print(f"[SAVED] {fname}")

    env.close()


if __name__ == "__main__":
    main()
    app.close()
