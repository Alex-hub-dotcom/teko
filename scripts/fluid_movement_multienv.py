# SPDX-License-Identifier: BSD-3-Clause
"""
Fluid movement test for TEKO robot (Multi-Environment).
--------------------------------------------------------
Drives all robots forward while applying a slight angular difference
between left and right wheels to create smooth circular trajectories.
Compatible with Isaac Lab 0.47.1 and TekoEnv (multi-env + camera).
"""
from isaaclab.app import AppLauncher
import argparse
import time
import torch

# Motion parameters
LINEAR_SPEED = 1.2          # Forward linear velocity (m/s equivalent)
ANGULAR_FACTOR = 0.7        # <1.0 = turn left, >1.0 = turn right
DURATION = 50.0             # Total runtime (seconds)
STEP_HZ = 30                # Simulation step frequency


def main():
    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="TEKO fluid movement test")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Launch Isaac Lab
    # ------------------------------------------------------------------
    app_launcher = AppLauncher(args_cli=["--headless"] if args.headless else [])
    app = app_launcher.app

    # Import environment after Isaac Sim initializes
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # ------------------------------------------------------------------
    # Initialize environment with multiple envs
    # ------------------------------------------------------------------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs  # Override number of environments
    
    print(f"[INFO] Creating {cfg.scene.num_envs} parallel environments...")
    env = TekoEnv(cfg)
    env.reset()
    print(f"[INFO] TEKO environment ready with {cfg.scene.num_envs} robots.")

    # ------------------------------------------------------------------
    # Define constant circular motion for ALL environments
    # ------------------------------------------------------------------
    N = cfg.scene.num_envs
    left_speed = LINEAR_SPEED * 1.0
    right_speed = LINEAR_SPEED * ANGULAR_FACTOR
    
    # Create actions for all environments: shape (num_envs, 2)
    actions = torch.tensor([[left_speed, right_speed]], device=env.device).repeat(N, 1)
    
    print(f"[INFO] Starting circular motion test:")
    print(f"       - Environments: {N}")
    print(f"       - Left wheel:   {left_speed:.2f} m/s")
    print(f"       - Right wheel:  {right_speed:.2f} m/s")
    print(f"       - Duration:     {DURATION:.1f}s")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    t0 = time.time()
    step = 0
    
    try:
        while app.is_running() and time.time() - t0 < DURATION:
            # Step all environments with same action
            env.step(actions)
            app.update()

            # Log progress every ~1 second
            if step % STEP_HZ == 0:
                elapsed = time.time() - t0
                obs = env._get_observations()
                
                # Check camera observations
                rgb_ok = "rgb" in obs and obs["rgb"].sum() > 0
                rgb_shape = obs["rgb"].shape if "rgb" in obs else None
                
                # Check GPU memory
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                    mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
                else:
                    mem_allocated = mem_reserved = 0.0
                
                print(f"Step {step:04d} | Time: {elapsed:5.1f}s | "
                      f"RGB: {rgb_ok} {rgb_shape} | "
                      f"GPU: {mem_allocated:.2f}/{mem_reserved:.2f} GB")
            
            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted manually.")
    
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        elapsed = time.time() - t0
        print(f"\n[INFO] Simulation completed:")
        print(f"       - Total steps: {step}")
        print(f"       - Total time:  {elapsed:.1f}s")
        print(f"       - Avg FPS:     {step/elapsed:.1f}")
        print("[INFO] Shutting down...")
        env.close()
        app.close()


if __name__ == "__main__":
    main()

# Usage:
# /workspace/isaaclab/_isaac_sim/python.sh scripts/fluid_movement_multienv.py
# /workspace/isaaclab/_isaac_sim/python.sh scripts/fluid_movement_multienv.py --num_envs 32
# /workspace/isaaclab/_isaac_sim/python.sh scripts/fluid_movement_multienv.py --headless --num_envs 64
