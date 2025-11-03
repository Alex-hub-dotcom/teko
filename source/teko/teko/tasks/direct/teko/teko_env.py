############ ENV - Multi-Environment Compatible with Rewards (v3)
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Isaac Lab 0.47.1 (Multi-Environment Support + RL Ready)
---------------------------------------------------------------------------
Active TEKO robot (RL agent) + static RobotGoal with emissive ArUco marker.
Includes reward computation and episode termination logic.
"""

from __future__ import annotations
import numpy as np
import torch
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdShade

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaacsim.sensors.camera import Camera

from .teko_env_cfg import TekoEnvCfg
from .robots.teko import TEKO_CONFIGURATION


class TekoEnv(DirectRLEnv):
    """TEKO environment: Multiple parallel robots + static goals with ArUco."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        self._cam_res = (cfg.camera.width, cfg.camera.height)
        self._max_wheel_speed = cfg.max_wheel_speed
        self.actions = None
        self.dof_idx = None
        self.cameras = []  # One camera per environment
        
        # Docking parameters
        self.target_distance = 0.475  # 47.5cm
        self.tolerance = 0.01  # 1cm
        self.max_distance = 3.0  # Max distance before episode fails
        
        # Goal positions (will be set after scene setup)
        self.goal_positions = None
        self.goal_orientations = None
        
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Setup da cena
    # ------------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        # --- Global lighting (shared across all environments) ---
        self._setup_global_lighting(stage)

        # --- Active robot - use scene config ---
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # --- Clone environments ---
        self.scene.clone_environments(copy_from_source=False)
        
        # --- Spawn per-environment assets AFTER cloning ---
        self._setup_per_environment_assets(stage)

        # --- Setup cameras ---
        self._setup_cameras()
        
        # --- Store goal positions for reward calculation ---
        self._cache_goal_transforms(stage)

    def _setup_global_lighting(self, stage):
        """Setup global lighting (dome + sun) - shared across all envs."""
        # Remove default blue dome
        if stage.GetPrimAtPath("/World/DomeLight"):
            stage.RemovePrim("/World/DomeLight")

        # Ambient dome light
        ambient = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AmbientLight"))
        ambient.CreateIntensityAttr(4000.0)
        ambient.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 0.95))
        ambient.CreateTextureFileAttr("")

        # Directional sun light
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(2000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-50.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

        print("[INFO] Global lighting setup complete.")

    def _setup_per_environment_assets(self, stage):
        """Spawn arena, goal robot + ArUco marker for EACH environment."""
        num_envs = self.scene.cfg.num_envs
        
        ARENA_USD_PATH = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
        TEKO_USD_PATH = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            
            # Arena per environment
            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")
            
            # Position active robot
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, 0.43))
                xf_robot.AddRotateZOp().Set(180.0)

            # Goal robot
            goal_path = f"{env_path}/RobotGoal"
            robot_goal = stage.DefinePrim(goal_path, "Xform")
            robot_goal.GetReferences().AddReference(TEKO_USD_PATH)
            
            xf_goal = UsdGeom.Xformable(robot_goal)
            xf_goal.ClearXformOpOrder()
            xf_goal.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.40))
            xf_goal.AddRotateZOp().Set(180.0)

            # ArUco marker
            self._create_aruco_marker(stage, goal_path, ARUCO_IMG_PATH)

        print(f"[INFO] Created {num_envs} environments with arenas, robots, and ArUco markers.")

    def _create_aruco_marker(self, stage, goal_path: str, aruco_img_path: str):
        """Create ArUco marker mesh with texture for a specific goal robot."""
        size = 0.05
        half = size * 0.5
        aruco_prim_path = f"{goal_path}/Aruco"

        # Mesh geometry
        mesh = UsdGeom.Mesh.Define(stage, aruco_prim_path)
        mesh.CreatePointsAttr([
            Gf.Vec3f(0.0, -half, -half),
            Gf.Vec3f(0.0,  half, -half),
            Gf.Vec3f(0.0,  half,  half),
            Gf.Vec3f(0.0, -half,  half),
        ])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        # Position
        xf_aruco = UsdGeom.Xformable(mesh)
        xf_aruco.ClearXformOpOrder()
        xf_aruco.AddTranslateOp().Set(Gf.Vec3f(0.17, 0.0, -0.045))

        # UV coordinates
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([
            Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)
        ])

        # Material with texture
        looks_path = f"{goal_path}/Looks/ArucoMaterial"
        material = UsdShade.Material.Define(stage, looks_path)

        tex = UsdShade.Shader.Define(stage, looks_path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(aruco_img_path))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

        st_reader = UsdShade.Shader.Define(stage, looks_path + "/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader_output)

        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader = UsdShade.Shader.Define(stage, looks_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)

        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

    def _setup_cameras(self):
        """Initialize one camera per environment."""
        sim = SimulationContext.instance()
        num_envs = self.scene.cfg.num_envs

        for env_idx in range(num_envs):
            cam_path = f"/World/envs/env_{env_idx}/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera"
            
            cam_prim = sim.stage.GetPrimAtPath(cam_path)
            if not cam_prim.IsValid():
                print(f"[WARN] Camera not found at {cam_path}")
                continue

            camera = Camera(
                prim_path=cam_path,
                resolution=self._cam_res,
                frequency=self.cfg.camera.frequency_hz,
            )
            camera.initialize()
            self.cameras.append(camera)

        print(f"[INFO] Initialized {len(self.cameras)} cameras.")

    def _cache_goal_transforms(self, stage):
        """Cache goal robot positions and orientations for reward calculation."""
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        self.goal_orientations = torch.zeros((num_envs, 4), device=self.device)
        
        for env_idx in range(num_envs):
            goal_path = f"/World/envs/env_{env_idx}/RobotGoal"
            goal_prim = stage.GetPrimAtPath(goal_path)
            
            if goal_prim.IsValid():
                xformable = UsdGeom.Xformable(goal_prim)
                local_transform = xformable.GetLocalTransformation()
                translation = local_transform.ExtractTranslation()
                
                # Store position (convert from USD Gf.Vec3d to torch)
                self.goal_positions[env_idx] = torch.tensor(
                    [translation[0], translation[1], translation[2]],
                    device=self.device
                )
                
                # For orientation, we'll use the default (180° rotation)
                # Quaternion for 180° around Z: [0, 0, 1, 0] (x, y, z, w format)
                self.goal_orientations[env_idx] = torch.tensor(
                    [0.0, 0.0, 1.0, 0.0],
                    device=self.device
                )
        
        print(f"[INFO] Cached {num_envs} goal transforms")

    # ------------------------------------------------------------------
    # Física / Observações / Ações
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        """Initialize joint indices once the articulation is ready."""
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        
        # Find indices for the wheel joints
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = []
        for dof_name in self.cfg.dof_names:
            if dof_name in name_to_idx:
                indices.append(name_to_idx[dof_name])
            else:
                print(f"[WARN] Joint '{dof_name}' not found. Available: {self.robot.joint_names}")
        
        if len(indices) == 0:
            raise RuntimeError(f"No valid DOF names found! Available: {self.robot.joint_names}")
        
        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions before physics step."""
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        """Apply wheel velocities to all environments."""
        if self.dof_idx is None or self.actions is None:
            return

        num_envs = self.scene.cfg.num_envs
        
        # actions shape: (num_envs, 2) -> [left_vel, right_vel] per env
        left_vel = self.actions[:, 0]
        right_vel = self.actions[:, 1]

        # Expand to 4 wheels: [front_left, front_right, back_left, back_right]
        targets = torch.stack([left_vel, right_vel, left_vel, right_vel], dim=1) * self._max_wheel_speed
        
        # Apply wheel polarity
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)
        targets = targets * polarity

        # Apply to all environments
        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_velocity_target(targets, env_ids=env_ids, joint_ids=self.dof_idx)

    def _get_observations(self):
        """Get RGB observations from all cameras."""
        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]
        
        obs = {"rgb": torch.zeros((num_envs, 3, h, w), device=self.device)}

        for env_idx, camera in enumerate(self.cameras):
            if env_idx >= num_envs:
                break
                
            try:
                rgba = camera.get_rgba()
                if isinstance(rgba, np.ndarray) and rgba.size > 0:
                    rgb = (torch.from_numpy(rgba[..., :3])
                           .to(self.device)
                           .permute(2, 0, 1)
                           .float() / 255.0)
                    obs["rgb"][env_idx] = rgb
            except Exception as e:
                print(f"[WARN] Camera {env_idx} failed: {e}")
                continue

        return obs

    def _get_rewards(self):
        """Compute docking rewards based on distance to goal."""
        # Get robot positions and orientations
        robot_pos = self.robot.data.root_pos_w  # (num_envs, 3)
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4) - [x, y, z, w]
        
        # Get joint velocities
        joint_vel = self.robot.data.joint_vel  # (num_envs, num_joints)
        
        # Calculate distance to goal
        distance = torch.norm(robot_pos - self.goal_positions, dim=-1)
        
        # Distance error from target
        error = torch.abs(distance - self.target_distance)
        
        # Main reward: exponential decay with distance error
        # Perfect alignment (0cm error) = 10.0
        # 1cm error = ~5.0
        # 5cm error = ~1.0
        distance_reward = 10.0 * torch.exp(-error / 0.02)
        
        # Success bonus when within tolerance
        success_mask = error < self.tolerance
        success_bonus = torch.where(success_mask, 
                                    torch.ones_like(error) * 100.0,
                                    torch.zeros_like(error))
        
        # Velocity penalty (encourage smooth, slow approach)
        vel_magnitude = torch.norm(joint_vel, dim=-1)
        velocity_penalty = -0.01 * (vel_magnitude / self._max_wheel_speed) ** 2
        
        # Total reward
        total_reward = distance_reward + success_bonus + velocity_penalty
        
        return total_reward

    def _get_dones(self):
        """Determine episode termination conditions."""
        num_envs = self.scene.cfg.num_envs
        
        # Get robot positions
        robot_pos = self.robot.data.root_pos_w
        
        # Calculate distance to goal
        distance = torch.norm(robot_pos - self.goal_positions, dim=-1)
        distance_error = torch.abs(distance - self.target_distance)
        
        # Success: within tolerance
        success = distance_error < self.tolerance
        
        # Failure: too far from goal
        too_far = distance > self.max_distance
        
        # Terminate on success or failure
        terminated = success | too_far
        
        # Time limit handled by DirectRLEnv
        time_out = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        return terminated, time_out

    def _reset_idx(self, env_ids):
        """Reset specific environments."""
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
        
        # Optional: Randomize initial robot positions slightly
        if len(env_ids) > 0:
            # Small random perturbations to starting position
            num_resets = len(env_ids)
            pos_noise = torch.randn(num_resets, 3, device=self.device) * 0.05
            pos_noise[:, 2] = 0  # Don't randomize Z (height)
            
            current_pos = self.robot.data.root_pos_w[env_ids]
            new_pos = current_pos + pos_noise
            
            self.robot.write_root_pose_to_sim(
                torch.cat([new_pos, self.robot.data.root_quat_w[env_ids]], dim=-1),
                env_ids=env_ids
            )