# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Torque Test – Straight Line
================================
Drives the TEKO robot forward at constant torque to verify wheel actuation,
friction, and ground interaction. Compatible with Isaac Lab 0.47.1 and
the torque-driven TEKO environment.
"""

from isaaclab.app import AppLauncher
import torch
import time
import random
import numpy as np

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
FORWARD_TORQUE = 0.5          # Normalized torque command [-1, 1]
DURATION = 10.0               # Duration in seconds
STEP_HZ = 60                  # Simulation frequency (Hz)
HEADLESS = False              # Set True for offscreen/headless run


def main(headless=HEADLESS):
    # ------------------------------------------------------------------
    # 1. Launch Isaac Lab application (with cameras enabled)
    # ------------------------------------------------------------------
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=True,   # IMPORTANT: allow camera sensors to spawn
    )
    app = app_launcher.app

    # ------------------------------------------------------------------
    # 2. Import environment definitions AFTER app init
    # ------------------------------------------------------------------
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # ------------------------------------------------------------------
    # 3. Initialize deterministic environment
    # ------------------------------------------------------------------
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
    print("[INFO] TEKO environment ready (torque test).")

    # ------------------------------------------------------------------
    # 4. Prepare constant torque command
    # ------------------------------------------------------------------
    torque_cmd = torch.tensor([[FORWARD_TORQUE, FORWARD_TORQUE]], device=env.device)
    print(f"[INFO] Applying constant forward torque: {FORWARD_TORQUE:.2f}")

    # ------------------------------------------------------------------
    # 5. Simulation loop
    # ------------------------------------------------------------------
    steps = int(DURATION * STEP_HZ)
    print(f"[INFO] Running simulation for {steps} steps ({DURATION}s) ...")

    t0 = time.time()
    for step in range(steps):
        env.step(torque_cmd)
        app.update()

        # Periodic state print (1x per second)
        if step % STEP_HZ == 0:
            pos = env.robot.data.root_pos_w[0].cpu().numpy()
            lin_vel = env.robot.data.root_lin_vel_w[0].cpu().numpy()
            ang_vel = env.robot.data.root_ang_vel_w[0].cpu().numpy()
            elapsed = time.time() - t0
            print(
                f"[{elapsed:05.2f}s] "
                f"pos(x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}) | "
                f"lin_vel=({lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}) | "
                f"ang_vel=({ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f})"
            )

        # Optional: run approximately in real time
        time.sleep(1.0 / STEP_HZ)

    # ------------------------------------------------------------------
    # 6. Clean shutdown
    # ------------------------------------------------------------------
    print("[INFO] Shutting down simulation.")
    app.close()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main(headless=HEADLESS)
