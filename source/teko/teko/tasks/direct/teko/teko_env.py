############ IMPROVED TEKO ENV - Vision-Based Docking with Collision Penalties
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Improved for Vision-Based Docking
----------------------------------------------------
Features:
- Collision detection (walls + static robot)
- Orientation-aware rewards
- Curriculum learning support
- Random spawn positions
"""

from __future__ import annotations
import numpy as np
import torch
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf, UsdShade

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaacsim.sensors.camera import Camera

from .teko_env_cfg import TekoEnvCfg


class TekoEnv(DirectRLEnv):
    """TEKO environment with collision penalties and improved rewards."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        self._cam_res = (cfg.camera.width, cfg.camera.height)
        self._max_wheel_speed = cfg.max_wheel_speed
        self.actions = None
        self.dof_idx = None
        self.cameras = []
        self.goal_positions = None
        self.num_agents = 1
        
        # Curriculum learning
        self.curriculum_level = 0  # 0=easy (close), 1=medium, 2=hard (far+random)
        self.spawn_distance_range = (0.8, 1.5)  # Start close
        
        # Collision tracking
        self.prev_robot_pos = None
        self.collision_cooldown = torch.zeros(1, dtype=torch.int32)
        
        # Arena boundaries (from your arena dimensions)
        self.arena_size = 2.0  # Assuming 2m x 2m arena
        
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        if not stage.GetPrimAtPath("/World/envs/env_0"):
            stage.DefinePrim("/World/envs/env_0", "Xform")

        self._setup_global_lighting(stage)

        # Active robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # Per-environment assets
        self._setup_per_environment_assets(stage)

        # Cameras + cached goal positions
        self._setup_cameras()
        self._cache_goal_transforms()

    def _init_observation_space(self):
        """Initialize observation space based on camera resolution."""
        import gymnasium as gym
        frame_shape = (3, self.cfg.camera.height, self.cfg.camera.width)
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=frame_shape, dtype=np.uint8)
        })
        print(f"[INFO] Observation space set to {frame_shape}")

    def _setup_global_lighting(self, stage):
        if stage.GetPrimAtPath("/World/DomeLight"):
            stage.RemovePrim("/World/DomeLight")

        ambient = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/AmbientLight"))
        ambient.CreateIntensityAttr(4000.0)
        ambient.CreateColorAttr(Gf.Vec3f(0.95, 0.95, 0.95))

        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
        sun.CreateIntensityAttr(2000.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.Xformable(sun).AddRotateXOp().Set(-50.0)
        UsdGeom.Xformable(sun).AddRotateYOp().Set(30.0)

        print("[INFO] Global lighting setup complete.")

    def _setup_per_environment_assets(self, stage):
        num_envs = self.scene.cfg.num_envs
        ARENA_USD_PATH = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
        TEKO_USD_PATH = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            # Arena
            try:
                arena_path = f"{env_path}/Arena"
                arena_prim = stage.DefinePrim(arena_path, "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            # Robot positioning (will be randomized in reset)
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, 0.43))
                xf_robot.AddRotateZOp().Set(180.0)

            # Goal robot (static)
            goal_path = f"{env_path}/RobotGoal"
            goal_prim = stage.DefinePrim(goal_path, "Xform")
            goal_prim.GetReferences().AddReference(TEKO_USD_PATH)

            xf_goal = UsdGeom.Xformable(goal_prim)
            xf_goal.ClearXformOpOrder()
            xf_goal.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.40))
            xf_goal.AddRotateZOp().Set(180.0)

            # ArUco marker
            self._create_aruco_marker(stage, goal_path, ARUCO_IMG_PATH)

        print(f"[INFO] Created {num_envs} environments.")

    def _create_aruco_marker(self, stage, goal_path: str, aruco_img_path: str):
        size = 0.05
        half = size * 0.5
        aruco_prim_path = f"{goal_path}/Aruco"

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

        xf_aruco = UsdGeom.Xformable(mesh)
        xf_aruco.ClearXformOpOrder()
        xf_aruco.AddTranslateOp().Set(Gf.Vec3f(0.17, 0.0, -0.045))

        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set([
            Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)
        ])

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
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_out)

        material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        UsdShade.MaterialBindingAPI(mesh).Bind(material)

    def _setup_cameras(self):
        sim = SimulationContext.instance()
        num_envs = self.scene.cfg.num_envs

        for env_idx in range(num_envs):
            cam_path = (
                f"/World/envs/env_{env_idx}/Robot/teko_urdf/TEKO_Body/"
                "TEKO_WallBack/TEKO_Camera/RearCamera"
            )
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

    def _cache_goal_transforms(self):
        """Cache goal positions with proper environment offsets."""
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        
        # ✅ Get environment origins from Isaac Lab scene
        # Each env is offset by env_spacing (6.0m in your config)
        env_origins = self.scene.env_origins  # This is provided by Isaac Lab!
        
        for env_idx in range(num_envs):
            # Goal position RELATIVE to environment origin
            local_goal_pos = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            # Add environment offset
            self.goal_positions[env_idx] = env_origins[env_idx] + local_goal_pos
        
        print(f"[INFO] Cached {num_envs} goal positions with environment offsets")
        print(f"[DEBUG] First goal at: {self.goal_positions[0]}")
        if num_envs > 1:
            print(f"[DEBUG] Second goal at: {self.goal_positions[1]}")

    # ------------------------------------------------------------------
    # Physics / Actions
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return

        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = []
        for dof_name in self.cfg.dof_names:
            if dof_name in name_to_idx:
                indices.append(name_to_idx[dof_name])

        if len(indices) == 0:
            raise RuntimeError(f"No valid DOF names found! Available: {self.robot.joint_names}")

        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx}")

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        if self.dof_idx is None or self.actions is None:
            return

        num_envs = self.scene.cfg.num_envs
        left_vel = self.actions[:, 0]
        right_vel = self.actions[:, 1]

        targets = torch.stack([left_vel, right_vel, left_vel, right_vel], dim=1) * self._max_wheel_speed
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)
        targets = targets * polarity

        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_velocity_target(targets, env_ids=env_ids, joint_ids=self.dof_idx)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self):
        import torch.nn.functional as F

        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]  # h=480, w=640

        rgb_obs = torch.zeros((num_envs, 3, h, w), device=self.device, dtype=torch.float32)
        
        for env_idx, camera in enumerate(self.cameras):
            if env_idx >= num_envs:
                break
            try:
                rgba = camera.get_rgba()
                if isinstance(rgba, np.ndarray) and rgba.size > 0:
                    # rgba shape is (H, W, 4) - NumPy format
                    rgb_np = rgba[..., :3]  # Take RGB, drop alpha -> (H, W, 3)
                    
                    # Convert to tensor and move to device
                    rgb = torch.from_numpy(rgb_np).to(self.device).float()  # (H, W, 3)
                    
                    # Permute to PyTorch format: (H, W, C) -> (C, H, W)
                    rgb = rgb.permute(2, 0, 1)  # (3, H, W)
                    
                    # Normalize to [0, 1]
                    rgb = rgb / 255.0
                    
                    # Resize if needed
                    if rgb.shape[1] != h or rgb.shape[2] != w:
                        rgb = F.interpolate(
                            rgb.unsqueeze(0), 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    
                    # Verify shape before assignment
                    assert rgb.shape == (3, h, w), f"Expected (3, {h}, {w}), got {rgb.shape}"
                    
                    rgb_obs[env_idx] = rgb
                    
            except Exception as e:
                print(f"[WARN] Camera {env_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Final shape check
        assert rgb_obs.shape == (num_envs, 3, h, w), f"Expected ({num_envs}, 3, {h}, {w}), got {rgb_obs.shape}"
        
        return {"policy": {"rgb": rgb_obs}}

    # ------------------------------------------------------------------
    # Rewards (IMPROVED)
    # ------------------------------------------------------------------
    def _get_rewards(self):
        robot_pos = self.robot.data.root_pos_w           # (num_envs, 3)
        robot_quat = self.robot.data.root_quat_w         # (num_envs, 4) [w,x,y,z]
        goal_pos = self.goal_positions                    # (num_envs, 3)
        
        # === 1. Distance Reward (center-to-center = 43cm when docked) ===
        distance = torch.norm(robot_pos - goal_pos, dim=-1)
        target_distance = 0.43  # 43cm from ground truth
        distance_error = torch.abs(distance - target_distance)
        # ✅ Make this the PRIMARY reward
        distance_reward = 20.0 * torch.exp(-distance_error / 0.05)  # Increased from 10.0
        
        # === 2. Y-Axis Alignment (lateral centering) ===
        y_error = torch.abs(robot_pos[:, 1] - goal_pos[:, 1])
        y_reward = 5.0 * torch.exp(-y_error / 0.05)
        
        # === 3. Orientation Alignment (both should face yaw≈180°) ===
        robot_yaw = torch.atan2(
            2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
            1.0 - 2.0 * (robot_quat[:, 2]**2 + robot_quat[:, 3]**2)
        )
        target_yaw = torch.tensor(np.pi, device=self.device)  # 180°
        yaw_error = torch.abs(robot_yaw - target_yaw)
        yaw_error = torch.min(yaw_error, 2*np.pi - yaw_error)  # Handle wrapping
        yaw_reward = 5.0 * torch.exp(-yaw_error / 0.2)
        
        # === 4. Collision Penalties ===
        collision_penalty = self._compute_collision_penalty()
        
        # === 5. Wall Boundary Penalty ===
        wall_penalty = self._compute_wall_penalty()
        
        # === 6. Simple approach incentive (NOT cumulative) ===
        # Give small bonus for being closer than starting distance
        approach_bonus = torch.where(
            distance < 1.0,  # If closer than 1m
            torch.tensor(2.0, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        
        # === Total Reward (NO unbounded progress reward) ===
        total_reward = (
            distance_reward 
            + y_reward 
            + yaw_reward 
            + approach_bonus
            - collision_penalty 
            - wall_penalty
        )
        
        return total_reward

    def _compute_collision_penalty(self):
        """Detect collisions with static robot."""
        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        
        distance = torch.norm(robot_pos - goal_pos, dim=-1)
        
        # Collision if center-to-center distance < 35cm (robots are ~40cm long)
        collision_threshold = 0.35
        collision = distance < collision_threshold
        
        # Large penalty for collision
        penalty = torch.where(collision, 
                             torch.tensor(20.0, device=self.device),
                             torch.tensor(0.0, device=self.device))
        
        return penalty

    def _compute_wall_penalty(self):
        """Penalize getting close to arena walls."""
        robot_pos = self.robot.data.root_pos_w
        
        # Arena boundaries (assuming centered at origin)
        half_size = self.arena_size / 2.0
        
        # Distance to nearest wall
        x_margin = half_size - torch.abs(robot_pos[:, 0])
        y_margin = half_size - torch.abs(robot_pos[:, 1])
        min_margin = torch.min(x_margin, y_margin)
        
        # Penalty if within 20cm of wall
        wall_threshold = 0.20
        penalty = torch.where(min_margin < wall_threshold,
                             10.0 * (wall_threshold - min_margin),
                             torch.tensor(0.0, device=self.device))
        
        return penalty

    def _compute_progress_reward(self):
        """Reward moving toward goal (capped to prevent exploitation)."""
        if self.prev_robot_pos is None:
            self.prev_robot_pos = self.robot.data.root_pos_w.clone()
            return torch.zeros(self.scene.cfg.num_envs, device=self.device)
        
        curr_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        
        prev_dist = torch.norm(self.prev_robot_pos - goal_pos, dim=-1)
        curr_dist = torch.norm(curr_pos - goal_pos, dim=-1)
        
        progress = prev_dist - curr_dist
        # ✅ Cap progress reward to prevent exploitation
        progress_reward = torch.clamp(progress * 2.0, -1.0, 1.0)
        
        self.prev_robot_pos = curr_pos.clone()
        
        return progress_reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self):
        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        distance = torch.norm(robot_pos - goal_pos, dim=-1)

        # Success: within 1cm of target distance (43cm)
        target_distance = 0.43
        tolerance = 0.01
        error = torch.abs(distance - target_distance)
        success = (error < tolerance)

        # Failure: collision with goal robot
        collision = distance < 0.35

        # Failure: out of arena
        half_size = self.arena_size / 2.0
        out_of_bounds = (
            (torch.abs(robot_pos[:, 0]) > half_size) |
            (torch.abs(robot_pos[:, 1]) > half_size)
        )

        terminated = (success | collision | out_of_bounds).squeeze(-1)
        time_out = torch.zeros_like(terminated)
        
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
        
        # Randomize spawn based on curriculum level
        if self.curriculum_level == 0:
            # Easy: close and aligned
            self._reset_close(env_ids)
        elif self.curriculum_level == 1:
            # Medium: varied distance
            self._reset_medium(env_ids)
        else:
            # Hard: random position and orientation
            self._reset_hard(env_ids)
        
        # Reset tracking variables
        self.prev_robot_pos = None
        self.collision_cooldown = torch.zeros(len(env_ids), dtype=torch.int32, device=self.device)

    def _reset_close(self, env_ids):
        """Easy curriculum: spawn close to goal."""
        num_reset = len(env_ids)
        
        # Distance from goal: 0.8m to 1.2m
        spawn_distance = torch.rand(num_reset, device=self.device) * 0.4 + 0.8
        
        # Always facing goal (yaw ≈ 180°)
        spawn_yaw = torch.ones(num_reset, device=self.device) * np.pi
        
        # Calculate position
        spawn_x = self.goal_positions[env_ids, 0] - spawn_distance
        spawn_y = self.goal_positions[env_ids, 1]
        # ✅ FIX: Use same Z as goal (0.40m) + small offset for robot height
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40  # Match goal Z
        
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
        spawn_quat = self._yaw_to_quat(spawn_yaw)
        
        self.robot.write_root_pose_to_sim(
            torch.cat([spawn_pos, spawn_quat], dim=1),
            env_ids=env_ids
        )

    def _reset_medium(self, env_ids):
        """Medium curriculum: varied distance, slight angle variation."""
        num_reset = len(env_ids)
        
        # Distance: 1.0m to 2.0m
        spawn_distance = torch.rand(num_reset, device=self.device) * 1.0 + 1.0
        
        # Yaw: 180° ± 30°
        spawn_yaw = torch.rand(num_reset, device=self.device) * (np.pi/3) - (np.pi/6) + np.pi
        
        spawn_x = self.goal_positions[env_ids, 0] - spawn_distance
        spawn_y = self.goal_positions[env_ids, 1] + (torch.rand(num_reset, device=self.device) * 0.4 - 0.2)
        # ✅ FIX: Use same Z as goal
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40
        
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
        spawn_quat = self._yaw_to_quat(spawn_yaw)
        
        self.robot.write_root_pose_to_sim(
            torch.cat([spawn_pos, spawn_quat], dim=1),
            env_ids=env_ids
        )

    def _reset_hard(self, env_ids):
        """Hard curriculum: random position in arena."""
        num_reset = len(env_ids)
        
        # Random position in arena (avoiding goal region)
        spawn_x = torch.rand(num_reset, device=self.device) * (self.arena_size - 0.5) - (self.arena_size/2 - 0.25)
        spawn_y = torch.rand(num_reset, device=self.device) * (self.arena_size - 0.5) - (self.arena_size/2 - 0.25)
        # ✅ FIX: Use same Z as goal
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40
        
        # Random orientation
        spawn_yaw = torch.rand(num_reset, device=self.device) * 2 * np.pi
        
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
        spawn_quat = self._yaw_to_quat(spawn_yaw)
        
        self.robot.write_root_pose_to_sim(
            torch.cat([spawn_pos, spawn_quat], dim=1),
            env_ids=env_ids
        )

    def _yaw_to_quat(self, yaw):
        """Convert yaw angle to quaternion [w, x, y, z]."""
        half_yaw = yaw / 2.0
        w = torch.cos(half_yaw)
        x = torch.zeros_like(yaw)
        y = torch.zeros_like(yaw)
        z = torch.sin(half_yaw)
        return torch.stack([w, x, y, z], dim=1)
    
    def set_curriculum_level(self, level: int):
        """Set curriculum difficulty level (0=easy, 1=medium, 2=hard)."""
        self.curriculum_level = max(0, min(2, level))
        print(f"[INFO] Curriculum level set to {self.curriculum_level}")