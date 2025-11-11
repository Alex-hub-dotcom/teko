# SPDX-License-Identifier: BSD-3-Clause
"""
red_dots.py — visualize docking sphere positions (Isaac Lab 0.47.1)
-------------------------------------------------------------------
Creates two colored spheres (radius 0.005 m) marking the female and
male connector centers for visual offset validation, using live data
from get_sphere_distances_from_physics().
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# Import after Isaac Sim starts
# ---------------------------------------------------------------------
import torch
import omni.usd
import omni.kit.commands
from pxr import UsdGeom, Gf
from isaaclab.sim import SimulationContext

# Import TEKO environment
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DOT_RADIUS = 0.005  # 5 mm
DOT_COLOR_FEMALE = (1.0, 0.0, 0.0)  # red
DOT_COLOR_MALE   = (0.0, 0.0, 1.0)  # blue

# ---------------------------------------------------------------------
# Initialize environment
# ---------------------------------------------------------------------
cfg = TekoEnvCfg()
env = TekoEnv(cfg, render_mode="human")

sim = SimulationContext.instance()
sim.step()  # settle physics

# ---------------------------------------------------------------------
# Get connector positions (from physics) - NO DELTAS
# ---------------------------------------------------------------------
female_pos, male_pos, surface_xy, surface_3d = env.get_sphere_distances_from_physics()

print(f"[DEBUG] female_pos[0]: {female_pos[0]}")
print(f"[DEBUG] male_pos[0]:   {male_pos[0]}")
print(f"[DEBUG] surface_xy={surface_xy[0]:.4f} m | surface_3d={surface_3d[0]:.4f} m")

# ---------------------------------------------------------------------
# Helper to create colored spheres
# ---------------------------------------------------------------------
def create_colored_sphere(name: str, position, color):
    """Spawn a small colored sphere at the given position."""
    stage = omni.usd.get_context().get_stage()
    path = f"/World/Debug/{name}"

    # Create sphere primitive
    omni.kit.commands.execute(
        "CreateMeshPrimWithDefaultXform",
        prim_type="Sphere",
        prim_path=path,
    )

    prim = stage.GetPrimAtPath(path)
    xformable = UsdGeom.Xformable(prim)

    # --- Set translation ---
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(*position))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))

    # --- Set scale (radius) ---
    scale_vec = Gf.Vec3f(DOT_RADIUS, DOT_RADIUS, DOT_RADIUS)
    if "xformOp:scale" in ops:
        ops["xformOp:scale"].Set(scale_vec)
    else:
        xformable.AddScaleOp().Set(scale_vec)

    # --- Set color ---
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    print(f"[INFO] Created {name} at {position} color={color} radius={DOT_RADIUS} m")

# ---------------------------------------------------------------------
# Create the debug dots
# ---------------------------------------------------------------------
create_colored_sphere("FemaleSphere", female_pos[0].tolist(), DOT_COLOR_FEMALE)
create_colored_sphere("MaleSphere",   male_pos[0].tolist(),   DOT_COLOR_MALE)

print("[INFO] ✅ Red (female) and blue (male) spheres created.")
print("👉 Press PLAY in Isaac Sim to view them.")

# ---------------------------------------------------------------------
# Keep simulation running
# ---------------------------------------------------------------------
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()