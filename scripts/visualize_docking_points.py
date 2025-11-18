#!/usr/bin/env python3
"""Visualize TEKO docking measurement points"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--enable_cameras'])  # Force enable cameras

app = AppLauncher(args)
sim = app.app

import torch
from pxr import Gf, UsdGeom, Sdf
from omni.usd import get_context

from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg


def create_debug_sphere(path, position, radius, color):
    """Create a colored sphere at given position."""
    stage = get_context().get_stage()
    
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*position))
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    return sphere


def main():
    print("🔍 TEKO Docking Point Visualization\n")
    
    # Create environment with 1 robot
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    env = TekoEnv(cfg=cfg)
    
    obs, _ = env.reset()
    
    # Get measurement points
    female_pos, male_pos, dist_xy, dist_3d = env.get_sphere_distances_from_physics()
    
    print(f"Active robot female connector: {female_pos[0].cpu().numpy()}")
    print(f"Goal robot male connector:     {male_pos[0].cpu().numpy()}")
    print(f"Distance (XY):  {dist_xy[0].item():.4f} m")
    print(f"Distance (3D):  {dist_3d[0].item():.4f} m")
    print(f"Target distance: 0.03 m (3cm)\n")
    
    # Create visual debug spheres
    # RED = Female connector (on active robot)
    create_debug_sphere(
        "/World/Debug/FemalePoint",
        female_pos[0].cpu().tolist(),
        0.02,  # 2cm sphere for visibility
        (1.0, 0.0, 0.0)  # Red
    )
    
    # BLUE = Male connector (on goal robot)
    create_debug_sphere(
        "/World/Debug/MalePoint",
        male_pos[0].cpu().tolist(),
        0.02,  # 2cm sphere
        (0.0, 0.0, 1.0)  # Blue
    )
    
    print("✅ Debug spheres created!")
    print("   🔴 RED sphere  = Female connector (active robot)")
    print("   🔵 BLUE sphere = Male connector (goal robot)")
    print("\nThese should be at the docking points.")
    print("Success = when spheres are within 3cm of each other\n")
    print("Press Ctrl+C to exit...")
    
    # Keep simulation running
    try:
        while True:
            env.step(torch.zeros((1, 2), device=env.device))
    except KeyboardInterrupt:
        print("\n👋 Closing...")
    
    env.close()
    sim.close()


if __name__ == "__main__":
    main()