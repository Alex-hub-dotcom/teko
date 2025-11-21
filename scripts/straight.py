#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Torque & Motion Sanity Test (v,w)
======================================

- Uses your TekoEnv and TekoEnvCfg exactly as defined.
- Action is [v, w] in [-1, 1], where:
    v = forward/backward command
    w = turning command

We run several phases:
    1) idle
    2) forward
    3) backward
    4) left turn (in place)
    5) right turn (in place)
    6) idle

If the robot doesn't move or values barely change, torque/control is not
doing what we expect.
"""

import time
import random
import argparse

import torch
import numpy as np
from isaaclab.app import AppLauncher


STEP_HZ = 60  # simulation / control frequency


def run_torque_test(headless: bool = False):
    # --------------------------------------------------------------
    # 1. Launch Isaac Lab app
    # --------------------------------------------------------------
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=True,  # fine to keep this on
    )
    app = app_launcher.app

    # Import AFTER app init
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # --------------------------------------------------------------
    # 2. Deterministic setup
    # --------------------------------------------------------------
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.dt = 1.0 / STEP_HZ
    cfg.sim.render_interval = 1

    env = TekoEnv(cfg)
    env.reset()
    device = env.device

    print("[INFO] TEKO environment ready (v,w torque/motion test).")

    # --------------------------------------------------------------
    # 3. Define [v, w] commands
    # --------------------------------------------------------------
    # v = forward/backward  in [-1, 1]
    # w = turn (left/right) in [-1, 1]

    idle_cmd     = torch.tensor([[0.0,  0.0]], device=device)
    forward_cmd  = torch.tensor([[1.0,  0.0]], device=device)   # forward
    backward_cmd = torch.tensor([[-1.0, 0.0]], device=device)   # backward
    left_turn    = torch.tensor([[0.0,  1.0]], device=device)   # rotate left
    right_turn   = torch.tensor([[0.0, -1.0]], device=device)   # rotate right

    phases = [
        ("idle_start", idle_cmd,     2.0),
        ("forward",    forward_cmd,  4.0),
        ("backward",   backward_cmd, 4.0),
        ("left_turn",  left_turn,    4.0),
        ("right_turn", right_turn,   4.0),
        ("idle_end",   idle_cmd,     2.0),
    ]

    print("[INFO] Starting motion phases...")
    t0 = time.time()

    for phase_name, cmd, duration in phases:
        n_steps = int(duration * STEP_HZ)
        print(f"\n[PHASE] {phase_name} – {duration:.1f}s ({n_steps} steps)")

        for step in range(n_steps):
            # IMPORTANT: DirectRLEnv.step expects actions with shape [num_envs, 2]
            env.step(cmd)
            app.update()

            # Print once per second
            if step % STEP_HZ == 0:
                pos = env.robot.data.root_pos_w[0].cpu().numpy()
                lin_vel = env.robot.data.root_lin_vel_w[0].cpu().numpy()
                ang_vel = env.robot.data.root_ang_vel_w[0].cpu().numpy()
                elapsed = time.time() - t0

                print(
                    f"[{elapsed:05.2f}s] "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                    f"lin_vel=({lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}) | "
                    f"ang_vel=({ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f})"
                )

            # (optional) try to run roughly in real time
            time.sleep(1.0 / STEP_HZ)

    print("\n[INFO] Torque & motion test finished. Closing app.")
    app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()

    run_torque_test(headless=args.headless)
