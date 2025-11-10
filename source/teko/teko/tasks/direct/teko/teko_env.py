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
from isaacsim.sensors.camera import Camera

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

        # State tracking (for all envs)
        self.prev_robot_pos = None
        self.prev_distance = None
        self.prev_actions = None
        self.step_count = None

        # Episode statistics for logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_components = {
            "distance": [], "alignment": [],
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
        """Initialize observation space based on camera resolution."""
        import gymnasium as gym
        frame_shape = (3, self.cfg.camera.height, self.cfg.camera.width)
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, shape=frame_shape, dtype=np.uint8)
        })
        print(f"[INFO] Observation space set to {frame_shape}")

    def _setup_global_lighting(self, stage):
        """Configure ambient and directional lighting."""
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
        """Load arena, robot, and static goal for each environment."""
        num_envs = self.scene.cfg.num_envs
        ARENA_USD_PATH = "/workspace/teko/documents/CAD/USD/stage_arena.usd"
        ARUCO_IMG_PATH = "/workspace/teko/documents/Aruco/test_marker.png"

        for env_idx in range(num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            
            # Arena
            try:
                arena_prim = stage.DefinePrim(f"{env_path}/Arena", "Xform")
                arena_prim.GetReferences().AddReference(ARENA_USD_PATH)
            except Exception as e:
                print(f"[WARN] Arena failed for env_{env_idx}: {e}")

            # Active robot (articulated)
            robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
            if robot_prim.IsValid():
                xf_robot = UsdGeom.Xformable(robot_prim)
                xf_robot.ClearXformOpOrder()
                xf_robot.AddTranslateOp().Set(Gf.Vec3d(-0.2, 0.0, 0.4))
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
        """Attach Isaac Sim camera objects to each robot."""
        sim = SimulationContext.instance()
        for env_idx in range(self.scene.cfg.num_envs):
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
        """Precompute goal positions for faster access."""
        num_envs = self.scene.cfg.num_envs
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        for env_idx, origin in enumerate(self.scene.env_origins):
            local_goal = torch.tensor([1.0, 0.0, 0.40], device=self.device)
            self.goal_positions[env_idx] = origin + local_goal
        print(f"[INFO] Cached {num_envs} goal positions.")

    # ------------------------------------------------------------------
    # Sphere position computation (PhysX-based)
    # ------------------------------------------------------------------
    def get_sphere_distances_from_physics(self):
        """
        Get sphere distances using PhysX body positions + fixed offsets.
        Uses live robot physics state instead of USD hierarchy.
        """
        # Fixed sphere offsets from robot base (calibrated manually)
        FEMALE_OFFSET = torch.tensor([-0.245, 0.0, -0.07], device=self.device)
        MALE_OFFSET = torch.tensor([0.22667, -0.00144, -0.08815], device=self.device)
        
        # Active robot position (updates with physics every step)
        active_pos = self.robot.data.root_pos_w  # (num_envs, 3)
        active_quat = self.robot.data.root_quat_w  # (num_envs, 4)
        
        # Convert quaternion to rotation matrix for active robot
        def quat_to_rot_matrix(q):
            """Convert batch of quaternions (N,4) to rotation matrices (N,3,3)."""
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            
            R = torch.zeros((q.shape[0], 3, 3), device=q.device)
            R[:, 0, 0] = 1 - 2*(y*y + z*z)
            R[:, 0, 1] = 2*(x*y - w*z)
            R[:, 0, 2] = 2*(x*z + w*y)
            R[:, 1, 0] = 2*(x*y + w*z)
            R[:, 1, 1] = 1 - 2*(x*x + z*z)
            R[:, 1, 2] = 2*(y*z - w*x)
            R[:, 2, 0] = 2*(x*z - w*y)
            R[:, 2, 1] = 2*(y*z + w*x)
            R[:, 2, 2] = 1 - 2*(x*x + y*y)
            return R
        
        active_rot = quat_to_rot_matrix(active_quat)  # (num_envs, 3, 3)
        
        # Apply rotation to offset and add to position
        female_offset_rotated = torch.bmm(active_rot, FEMALE_OFFSET.unsqueeze(-1).expand(active_pos.shape[0], 3, 1))
        female_pos = active_pos + female_offset_rotated.squeeze(-1)
        
        # Static robot position (goal is stationary, no rotation needed - always facing same way)
        static_pos = self.goal_positions  # (num_envs, 3)
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
    # Debug visualization
    # ------------------------------------------------------------------
    def _debug_visualize_spheres(self):
        """Add red/blue debug spheres to visualize connector positions."""
        stage = get_context().get_stage()
        
        # Get actual sphere positions
        female_pos, male_pos, _, _ = self.get_sphere_distances_from_physics()
        
        for env_idx in range(self.scene.cfg.num_envs):
            # Female connector (red) - half size
            female_debug_path = f"/World/envs/env_{env_idx}/DebugFemale"
            if not stage.GetPrimAtPath(female_debug_path):
                sphere_prim = UsdGeom.Sphere.Define(stage, female_debug_path)
                sphere_prim.CreateRadiusAttr(0.01)  # 1cm radius (half of previous 2cm)
                sphere_prim.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])  # Red
            
            sphere_xform = UsdGeom.Xformable(stage.GetPrimAtPath(female_debug_path))
            pos = female_pos[env_idx].cpu().numpy()
            sphere_xform.ClearXformOpOrder()
            sphere_xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
            
            # Male connector (blue) - half size
            male_debug_path = f"/World/envs/env_{env_idx}/DebugMale"
            if not stage.GetPrimAtPath(male_debug_path):
                sphere_prim = UsdGeom.Sphere.Define(stage, male_debug_path)
                sphere_prim.CreateRadiusAttr(0.01)  # 1cm radius
                sphere_prim.CreateDisplayColorAttr([(0.0, 0.0, 1.0)])  # Blue
            
            sphere_xform = UsdGeom.Xformable(stage.GetPrimAtPath(male_debug_path))
            pos = male_pos[env_idx].cpu().numpy()
            sphere_xform.ClearXformOpOrder()
            sphere_xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

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
        """Apply torque-based wheel control."""
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
        
        # DEBUG: Visualize connector spheres every step
        self._debug_visualize_spheres()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Get RGB camera observations from all environments."""
        import torch.nn.functional as F
        num_envs = self.scene.cfg.num_envs
        h, w = self._cam_res[1], self._cam_res[0]
        rgb_obs = torch.zeros((num_envs, 3, h, w), device=self.device)
        
        for env_idx, cam in enumerate(self.cameras):
            rgba = cam.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                rgb = torch.from_numpy(rgba[..., :3]).to(self.device).float().permute(2, 0, 1) / 255.0
                rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                rgb_obs[env_idx] = rgb
        
        return {"policy": rgb_obs}

    # ------------------------------------------------------------------
    # Rewards (modular)
    # ------------------------------------------------------------------
    def _get_rewards(self):
        """Compute rewards using modular reward functions."""
        return compute_total_reward(self)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self):
        """Check for termination using physics-based sphere distances."""
        num_envs = self.scene.cfg.num_envs
        
        # Get sphere distances using PhysX positions
        female_pos, male_pos, surface_xy, surface_3d = self.get_sphere_distances_from_physics()
        
        # Success: surface_xy < 3cm
        success_threshold = 0.03
        success = surface_xy < success_threshold
        
        # Out of bounds - use LOCAL coordinates
        robot_pos_global = self.robot.data.root_pos_w
        env_origins = self.scene.env_origins
        robot_pos_local = robot_pos_global - env_origins
        
        # Extended arena boundaries: ±4.0m x ±4.0m
        out_of_bounds = (
            (torch.abs(robot_pos_local[:, 0]) > 4.0) |
            (torch.abs(robot_pos_local[:, 1]) > 4.0)
        )
        
        terminated = success | out_of_bounds
        time_out = torch.zeros_like(terminated)
        
        if success.any():
            self.episode_successes.append(success.sum().item())
        
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset and curriculum
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._lazy_init_articulation()
        # reset_environment_curriculum(self, env_ids)  # COMMENTED OUT FOR TESTING
        
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

        # Lift active robot slightly
        if hasattr(self.cfg, "robot_spawn_z_offset"):
            root_state = self.robot.data.root_state_w.clone()
            root_state[env_ids, 2] += self.cfg.robot_spawn_z_offset
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def set_curriculum_level(self, level: int):
        """Set curriculum difficulty level."""
        set_curriculum_level(self, level)

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def get_episode_statistics(self):
        """Collect episode statistics for logging."""
        return collect_episode_stats(self)