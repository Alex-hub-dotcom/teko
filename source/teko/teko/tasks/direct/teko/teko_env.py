############ IMPROVED TEKO ENV - Anti-Oscillation + Comprehensive Logging
# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment — Improved Version with Anti-Oscillation
---------------------------------------------------------
Key improvements:
- Velocity penalties to prevent back-and-forth behavior
- Better reward shaping (no unbounded distance rewards)
- Progress tracking with directional awareness
- Comprehensive episode logging
- Success rate metrics
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
    """TEKO environment with anti-oscillation and comprehensive logging."""

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
        self.curriculum_level = 0
        self.spawn_distance_range = (0.5, 0.8)  # Start even closer
        
        # Anti-oscillation tracking
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.movement_history = []  # Track movement patterns
        self.step_count = torch.zeros(1, dtype=torch.int32)
        
        # Arena boundaries
        self.arena_size = 2.0
        
        # Episode statistics for logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_components = {
            'distance': [],
            'alignment': [],
            'orientation': [],
            'velocity_penalty': [],
            'oscillation_penalty': [],
            'collision_penalty': [],
            'wall_penalty': []
        }
        
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Scene setup (unchanged from your version)
    # ------------------------------------------------------------------
    def _setup_scene(self):
        stage = get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not initialized")

        if not stage.GetPrimAtPath("/World/envs/env_0"):
            stage.DefinePrim("/World/envs/env_0", "Xform")

        self._setup_global_lighting(stage)

        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.scene.clone_environments(copy_from_source=True)
        self._setup_per_environment_assets(stage)
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
        
        from pxr import UsdPhysics, PhysxSchema

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            try:
                arena_path = f"{env_path}/Arena"
                arena_prim = stage.DefinePrim(arena_path, "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, 0.43))
                xf_robot.AddRotateZOp().Set(180.0)

            # CRITICAL: Goal robot must be static (no physics simulation)
            goal_path = f"{env_path}/RobotGoal"
            goal_prim = stage.DefinePrim(goal_path, "Xform")
            goal_prim.GetReferences().AddReference(TEKO_USD_PATH)

            xf_goal = UsdGeom.Xformable(goal_prim)
            xf_goal.ClearXformOpOrder()
            xf_goal.AddTranslateOp().Set(Gf.Vec3f(1.0, 0.0, 0.40))
            xf_goal.AddRotateZOp().Set(180.0)
            
            # Disable ALL physics on goal robot and its children
            import omni.usd
            from pxr import Usd
            for prim in Usd.PrimRange(goal_prim):
                # Remove rigid body
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                # Disable collisions
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                # Remove articulation
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

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
        
        env_origins = self.scene.env_origins
        
        for env_idx in range(num_envs):
            local_goal_pos = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = env_origins[env_idx] + local_goal_pos
        
        print(f"[INFO] Cached {num_envs} goal positions with environment offsets")

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
        h, w = self._cam_res[1], self._cam_res[0]

        rgb_obs = torch.zeros((num_envs, 3, h, w), device=self.device, dtype=torch.float32)
        
        for env_idx, camera in enumerate(self.cameras):
            if env_idx >= num_envs:
                break
            try:
                rgba = camera.get_rgba()
                if isinstance(rgba, np.ndarray) and rgba.size > 0:
                    rgb_np = rgba[..., :3]
                    rgb = torch.from_numpy(rgb_np).to(self.device).float()
                    rgb = rgb.permute(2, 0, 1)
                    rgb = rgb / 255.0
                    
                    if rgb.shape[1] != h or rgb.shape[2] != w:
                        rgb = F.interpolate(
                            rgb.unsqueeze(0), 
                            size=(h, w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                    
                    rgb_obs[env_idx] = rgb
                    
            except Exception as e:
                print(f"[WARN] Camera {env_idx} failed: {e}")
                continue

        return {"policy": {"rgb": rgb_obs}}

    # ------------------------------------------------------------------
    # IMPROVED REWARDS (Anti-Oscillation)
    # ------------------------------------------------------------------
    def _get_rewards(self):
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        goal_pos = self.goal_positions
        
        # Calculate current distance
        distance = torch.norm(robot_pos - goal_pos, dim=-1)
        target_distance = 0.43
        distance_error = torch.abs(distance - target_distance)
        
        # === 1. Distance Reward (bounded, exponential decay) ===
        # Maximum reward when at perfect distance, decays exponentially
        distance_reward = 15.0 * torch.exp(-distance_error / 0.05)
        
        # === 2. Y-Axis Alignment (lateral centering) ===
        y_error = torch.abs(robot_pos[:, 1] - goal_pos[:, 1])
        y_reward = 5.0 * torch.exp(-y_error / 0.05)
        
        # === 3. Orientation Alignment ===
        robot_yaw = torch.atan2(
            2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
            1.0 - 2.0 * (robot_quat[:, 2]**2 + robot_quat[:, 3]**2)
        )
        target_yaw = torch.tensor(np.pi, device=self.device)
        yaw_error = torch.abs(robot_yaw - target_yaw)
        yaw_error = torch.min(yaw_error, 2*np.pi - yaw_error)
        yaw_reward = 8.0 * torch.exp(-yaw_error / 0.2)
        
        # === 4. Velocity Penalty (penalize excessive movement) ===
        if self.actions is not None:
            action_magnitude = torch.norm(self.actions, dim=-1)
            # Penalize high velocities when close to goal
            distance_factor = torch.clamp(distance_error / 0.2, 0.0, 1.0)
            velocity_penalty = action_magnitude * (1.0 - distance_factor) * 3.0  # Reduced from 10
        else:
            velocity_penalty = torch.zeros_like(distance)
        
        # === 5. Oscillation Detection and Penalty ===
        oscillation_penalty = self._compute_oscillation_penalty()
        
        # === 6. Directional Progress Reward ===
        progress_reward = self._compute_smart_progress_reward(distance)
        
        # === 7. Collision & Wall Penalties ===
        collision_penalty = self._compute_collision_penalty()
        wall_penalty = self._compute_wall_penalty()
        
        # === 8. Success Bonus ===
        success_bonus = torch.where(
            distance_error < 0.02,  # Within 2cm of target
            torch.tensor(50.0, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        
        # === Total Reward ===
        total_reward = (
            distance_reward 
            + y_reward 
            + yaw_reward 
            + progress_reward
            + success_bonus
            - velocity_penalty
            - oscillation_penalty 
            - collision_penalty 
            - wall_penalty
        )
        
        # Store reward components for logging
        self.reward_components['distance'].append(distance_reward.mean().item())
        self.reward_components['alignment'].append(y_reward.mean().item())
        self.reward_components['orientation'].append(yaw_reward.mean().item())
        self.reward_components['velocity_penalty'].append(velocity_penalty.mean().item())
        self.reward_components['oscillation_penalty'].append(oscillation_penalty.mean().item())
        self.reward_components['collision_penalty'].append(collision_penalty.mean().item())
        self.reward_components['wall_penalty'].append(wall_penalty.mean().item())
        
        return total_reward

    def _compute_oscillation_penalty(self):
        """Detect and penalize back-and-forth oscillations."""
        if self.prev_actions is None or self.actions is None:
            self.prev_actions = self.actions.clone() if self.actions is not None else None
            return torch.zeros(self.scene.cfg.num_envs, device=self.device)
        
        # Check if actions reversed direction
        action_product = self.actions * self.prev_actions
        direction_reversal = (action_product < -0.5).any(dim=-1).float()
        
        # Stronger penalty if reversing frequently
        oscillation_penalty = direction_reversal * 8.0  # Reduced from 20
        
        self.prev_actions = self.actions.clone()
        
        return oscillation_penalty

    def _compute_smart_progress_reward(self, current_distance):
        """Reward progress toward goal, but prevent exploitation."""
        if self.prev_distance is None:
            self.prev_distance = current_distance.clone()
            return torch.zeros_like(current_distance)
        
        # Calculate progress
        progress = self.prev_distance - current_distance
        
        # Only reward forward progress (moving closer)
        progress_reward = torch.where(
            progress > 0,
            torch.clamp(progress * 10.0, 0.0, 2.0),  # Cap at +2
            torch.clamp(progress * 5.0, -1.0, 0.0)   # Small penalty for moving away
        )
        
        self.prev_distance = current_distance.clone()
        
        return progress_reward

    def _compute_collision_penalty(self):
        """Detect collisions with static robot."""
        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        
        distance = torch.norm(robot_pos - goal_pos, dim=-1)
        collision_threshold = 0.35
        collision = distance < collision_threshold
        
        penalty = torch.where(collision, 
                             torch.tensor(50.0, device=self.device),  # Reduced from 100
                             torch.tensor(0.0, device=self.device))
        
        return penalty

    def _compute_wall_penalty(self):
        """Penalize getting close to arena walls."""
        robot_pos = self.robot.data.root_pos_w
        half_size = self.arena_size / 2.0
        
        x_margin = half_size - torch.abs(robot_pos[:, 0])
        y_margin = half_size - torch.abs(robot_pos[:, 1])
        min_margin = torch.min(x_margin, y_margin)
        
        wall_threshold = 0.20
        penalty = torch.where(min_margin < wall_threshold,
                             15.0 * (wall_threshold - min_margin),
                             torch.tensor(0.0, device=self.device))
        
        return penalty

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self):
        robot_pos = self.robot.data.root_pos_w
        goal_pos = self.goal_positions
        distance = torch.norm(robot_pos - goal_pos, dim=-1)

        # Success
        target_distance = 0.43
        tolerance = 0.02  # 2cm tolerance
        error = torch.abs(distance - target_distance)
        success = (error < tolerance)

        # Failure: collision
        collision = distance < 0.35

        # Failure: out of bounds
        half_size = self.arena_size / 2.0
        out_of_bounds = (
            (torch.abs(robot_pos[:, 0]) > half_size) |
            (torch.abs(robot_pos[:, 1]) > half_size)
        )

        terminated = (success | collision | out_of_bounds).squeeze(-1)
        time_out = torch.zeros_like(terminated)
        
        # Log episode success
        if success.any():
            self.episode_successes.append(True)
        
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
        
        # Curriculum-based reset
        if self.curriculum_level == 0:
            self._reset_close(env_ids)
        elif self.curriculum_level == 1:
            self._reset_medium(env_ids)
        else:
            self._reset_hard(env_ids)
        
        # Reset tracking
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.step_count.zero_()

    def _reset_close(self, env_ids):
        """Very close spawn for initial learning."""
        num_reset = len(env_ids)
        
        spawn_distance = torch.rand(num_reset, device=self.device) * 0.3 + 0.5  # 0.5-0.8m
        spawn_yaw = torch.ones(num_reset, device=self.device) * np.pi + \
                    (torch.rand(num_reset, device=self.device) * 0.2 - 0.1)  # ±6°
        
        spawn_x = self.goal_positions[env_ids, 0] - spawn_distance
        spawn_y = self.goal_positions[env_ids, 1] + (torch.rand(num_reset, device=self.device) * 0.1 - 0.05)
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40
        
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
        spawn_quat = self._yaw_to_quat(spawn_yaw)
        
        self.robot.write_root_pose_to_sim(
            torch.cat([spawn_pos, spawn_quat], dim=1),
            env_ids=env_ids
        )

    def _reset_medium(self, env_ids):
        """Medium curriculum."""
        num_reset = len(env_ids)
        
        spawn_distance = torch.rand(num_reset, device=self.device) * 0.8 + 0.8  # 0.8-1.6m
        spawn_yaw = torch.rand(num_reset, device=self.device) * (np.pi/4) - (np.pi/8) + np.pi  # ±22.5°
        
        spawn_x = self.goal_positions[env_ids, 0] - spawn_distance
        spawn_y = self.goal_positions[env_ids, 1] + (torch.rand(num_reset, device=self.device) * 0.4 - 0.2)
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40
        
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
        spawn_quat = self._yaw_to_quat(spawn_yaw)
        
        self.robot.write_root_pose_to_sim(
            torch.cat([spawn_pos, spawn_quat], dim=1),
            env_ids=env_ids
        )

    def _reset_hard(self, env_ids):
        """Hard curriculum."""
        num_reset = len(env_ids)
        
        spawn_x = torch.rand(num_reset, device=self.device) * (self.arena_size - 0.5) - (self.arena_size/2 - 0.25)
        spawn_y = torch.rand(num_reset, device=self.device) * (self.arena_size - 0.5) - (self.arena_size/2 - 0.25)
        spawn_z = torch.ones(num_reset, device=self.device) * 0.40
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
        """Set curriculum difficulty level."""
        self.curriculum_level = max(0, min(2, level))
        print(f"[INFO] Curriculum level set to {self.curriculum_level}")
    
    def get_episode_statistics(self):
        """Return episode statistics for logging."""
        if len(self.episode_rewards) == 0:
            return {}
        
        stats = {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'success_rate': np.mean(self.episode_successes[-100:]) if self.episode_successes else 0.0,
            'reward_components': {k: np.mean(v[-100:]) if v else 0.0 for k, v in self.reward_components.items()}
        }
        return stats