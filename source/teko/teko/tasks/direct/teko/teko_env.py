# SPDX-License-Identifier: BSD-3-Clause
#
# TEKO Environment — Modular Version (Torque-driven, Multi-env)
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import torch
from omni.usd import get_context
from pxr import Sdf, UsdGeom, UsdLux, Gf
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.sensors import Camera, CameraCfg  # FIXED IMPORT

from .teko_env_cfg import TekoEnvCfg
from .rewards.reward_functions import compute_total_reward
from .curriculum.curriculum_manager import reset_environment_curriculum, set_curriculum_level
from .utils.geometry_utils import yaw_to_quat
from .utils.logging_utils import collect_episode_stats
from .robots.teko_static import TEKOStatic


class TekoEnv(DirectRLEnv):
    """Torque-driven TEKO environment with multi-env support."""

    cfg: TekoEnvCfg

    def __init__(self, cfg: TekoEnvCfg, render_mode: str | None = None, **kwargs):
        self._cam_res = (cfg.camera.width, cfg.camera.height)
        self._max_wheel_torque = cfg.max_wheel_torque
        self.actions = None
        self.dof_idx = None
        self.cameras = []
        self.goal_positions = None
        self.num_agents = 1

        # Curriculum learning
        self.curriculum_level = 0

        # State tracking
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.step_count = None

        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_components = {
            "distance": [], "progress": [], "alignment": [],
            "velocity_penalty": [], "oscillation_penalty": [],
            "collision_penalty": [], "wall_penalty": []
        }

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
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=True)
        self._setup_per_environment_assets(stage)
        self._setup_cameras()
        self._cache_goal_transforms()

    def _init_observation_space(self):
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
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"

            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            # Active robot start
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.0, 0.4))
                xf_robot.AddRotateZOp().Set(180.0)

            # Static goal robot
            try:
                TEKOStatic(
                    prim_path=f"{env_path}/RobotGoal",
                    aruco_path=ARUCO_IMG_PATH,
                )
                print(f"[INFO] Spawned static TEKO goal in env_{env_idx}")
            except Exception as e:
                print(f"[WARN] Failed to create static TEKO goal in env_{env_idx}: {e}")

        print(f"[INFO] Created {num_envs} environments.")

    def _setup_cameras(self):
        """Setup cameras using Isaac Lab's Camera sensor."""
        for env_idx in range(self.scene.cfg.num_envs):
            cam_path = (
                f"/World/envs/env_{env_idx}/Robot/teko_urdf/TEKO_Body/"
                "TEKO_WallBack/TEKO_Camera/RearCamera"
            )
            
            # Create camera config
            cam_cfg = CameraCfg(
                prim_path=cam_path,
                update_period=0,  # Update every step
                height=self._cam_res[1],
                width=self._cam_res[0],
                data_types=["rgb"],
                spawn=None,  # Camera already exists in USD
            )
            
            # Create camera sensor
            camera = Camera(cfg=cam_cfg)
            self.cameras.append(camera)
        
        print(f"[INFO] Initialized {len(self.cameras)} cameras.")

    def _cache_goal_transforms(self):
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        for env_idx, origin in enumerate(self.scene.env_origins):
            local_goal = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = origin + local_goal
        print(f"[INFO] Cached {num_envs} goal positions.")

    # ------------------------------------------------------------------
    # Sphere position computation
    # ------------------------------------------------------------------
    def get_sphere_distances_from_physics(self):
        """Get sphere distances using fixed world-space offsets (no rotation needed)."""
        # Calibrated offsets in world space
        FEMALE_OFFSET = torch.tensor([0.24, 0.0, -0.08], device=self.device)
        MALE_OFFSET = torch.tensor([0.22667, -0.00144, -0.08815], device=self.device)

        # Robot positions
        active_pos = self.robot.data.root_pos_w  # (num_envs, 3)
        static_pos = self.goal_positions  # (num_envs, 3)
        
        # Apply offsets directly (no rotation - both robots face same direction)
        female_pos = active_pos + FEMALE_OFFSET.unsqueeze(0).expand(active_pos.shape[0], 3)
        male_pos = static_pos + MALE_OFFSET.unsqueeze(0).expand(static_pos.shape[0], 3)
        
        # Compute distances
        diff = female_pos - male_pos
        dist_3d = torch.norm(diff, dim=-1)  # (num_envs,)
        dist_xy = torch.norm(diff[:, :2], dim=-1)  # (num_envs,)
        
        # Subtract sphere radii
        R_FEMALE = 0.005
        R_MALE = 0.005
        surface_3d = torch.clamp(dist_3d - (R_FEMALE + R_MALE), min=0.0)
        surface_xy = torch.clamp(dist_xy - (R_FEMALE + R_MALE), min=0.0)
        
        return female_pos, male_pos, surface_xy, surface_3d

    # ------------------------------------------------------------------
    # Actions (Torque control)
    # ------------------------------------------------------------------
    def _lazy_init_articulation(self):
        if self.dof_idx is not None or getattr(self.robot, "root_physx_view", None) is None:
            return
        name_to_idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        indices = [name_to_idx[n] for n in self.cfg.dof_names if n in name_to_idx]
        if not indices:
            raise RuntimeError(f"No valid DOF names found: {self.robot.joint_names}")
        self.dof_idx = torch.tensor(indices, dtype=torch.long, device=self.device)
        print(f"[INFO] DOF indices: {self.dof_idx}")

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self._lazy_init_articulation()

    def _apply_action(self):
        if self.dof_idx is None or self.actions is None:
            return
        num_envs = self.scene.cfg.num_envs
        torque_targets = torch.stack(
            [self.actions[:, 0], self.actions[:, 1], self.actions[:, 0], self.actions[:, 1]], dim=1
        ) * self._max_wheel_torque
        polarity = torch.tensor(self.cfg.wheel_polarity, device=self.device).unsqueeze(0)
        torque_targets = torque_targets * polarity
        env_ids = torch.arange(num_envs, device=self.device)
        self.robot.set_joint_effort_target(torque_targets, env_ids=env_ids, joint_ids=self.dof_idx)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Get RGB observations from cameras."""
        import torch.nn.functional as F
        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]
        rgb_obs = torch.zeros((num_envs, 3, h, w), device=self.device)

        for env_idx, cam in enumerate(self.cameras):
            # Update camera data
            cam.update(dt=0.0)
            
            # Get RGB data
            rgb_data = cam.data.output["rgb"]
            if rgb_data is not None and rgb_data.numel() > 0:
                # rgb_data shape: (1, H, W, 4) or (1, H, W, 3)
                # Remove batch dimension if present
                if rgb_data.ndim == 4:
                    rgb_data = rgb_data.squeeze(0)  # Now (H, W, C)
                
                # Remove alpha channel if present
                if rgb_data.shape[-1] == 4:
                    rgb_data = rgb_data[..., :3]  # Now (H, W, 3)
                
                # Convert to (3, H, W) and normalize
                rgb = rgb_data.permute(2, 0, 1).float() / 255.0
                
                # Resize if needed
                if rgb.shape[1] != h or rgb.shape[2] != w:
                    rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                
                rgb_obs[env_idx] = rgb

        return {"rgb": rgb_obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self):
        from teko.tasks.direct.teko.rewards.reward_functions import compute_total_reward
        return compute_total_reward(self)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self):
        """Terminate when spheres touch or robot leaves the arena."""
        _, _, surface_xy, _ = self.get_sphere_distances_from_physics()

        success_threshold = 0.03  # 3 cm contact
        success = surface_xy < success_threshold

        # Local coordinates for arena boundaries
        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins
        out_of_bounds = (
            (torch.abs(robot_pos_local[:, 0]) > 1.4) |
            (torch.abs(robot_pos_local[:, 1]) > 2.4)
        )

        terminated = success | out_of_bounds
        time_out = torch.zeros_like(terminated, device=self.device)

        if success.any():
            self.episode_successes.append(success.sum().item())
            print(f"[SUCCESS] {success.sum().item()} successful docking(s).")

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
        
        # Reset state tracking
        if self.prev_distance is None:
            self.prev_distance = torch.zeros(self.scene.cfg.num_envs, device=self.device)
        if self.prev_actions is None:
            self.prev_actions = torch.zeros((self.scene.cfg.num_envs, 2), device=self.device)
        if self.step_count is None:
            self.step_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int32, device=self.device)
        
        self.prev_distance[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.step_count[env_ids] = 0

    def set_curriculum_level(self, level: int):
        set_curriculum_level(self, level)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def get_episode_statistics(self):
        return collect_episode_stats(self)