#!/usr/bin/env python3
"""
Debug script: visualize all curriculum stages.

- Runs TEKO env with 2 parallel envs
- For each curriculum stage:
    * sets env.curriculum_level
    * reset()
    * takes one zero-action step
    * saves camera images for env 0 and 1 to PNG

Run from isaaclab root (or wherever you usually run training):

/workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/teko/scripts/debug_visualize_curriculum.py
"""

import os
import torch
import imageio

from isaaclab.app import AppLauncher

from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv
from teko.tasks.direct.teko.curriculum.curriculum_manager import STAGE_NAMES


def main():
    # We want the GUI so you can also look at the scene
    app = AppLauncher(headless=False).app

    # --- Configure env for debugging ---
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 2
    cfg.enable_curriculum = True

    # If you add the debug flags in the cfg (see section 2):
    # cfg.debug_boundaries = True
    # cfg.debug_goal_halo = True

    env = TekoEnv(cfg=cfg)

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

        # One zero-action step (to let physics & camera update)
        zero_action = torch.zeros((num_envs, action_dim), device=env.device)
        obs, _, _, _, _ = env.step(zero_action)

        # Get RGB observations: [num_envs, 3, H, W] in [0, 1]
        rgb = obs["rgb"].detach().cpu()  # tensor

        for env_id in range(num_envs):
            img = rgb[env_id]  # [3, H, W]
            # convert back to [H, W, 3] uint8 for saving
            img = (img * 255.0).clamp(0, 255).byte()
            img = img.permute(1, 2, 0).numpy()

            fname = os.path.join(
                out_dir,
                f"stage_{stage:02d}_env_{env_id}_cam.png",
            )
            imageio.imwrite(fname, img)
            print(f"[SAVED] {fname}")

        # OPTIONAL: if you want to also save a viewport screenshot,
        # you can plug in Isaac/Kit viewport API here (pseudo-code):
        #
        # import omni.kit.viewport_legacy as vp
        # viewport = vp.get_default_viewport_window()
        # tex = viewport.get_texture()
        # tex.save(os.path.join(out_dir, f"stage_{stage:02d}_viewport.png"))

    env.close()
    app.close()


if __name__ == "__main__":
    main()
