################# cfg - Multi-Environment Compatible
# SPDX-License-Identifier: BSD-3-Clause
"""
TekoEnvCfg — environment configuration for TEKO with RGB camera.
Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
Supports 1 to 100+ parallel environments.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Configuration for TEKO environment (multiple parallel robots + static goals)."""

    # --- General parameters -------------------------------------------
    decimation = 2                      # Subsampling for smooth rendering
    episode_length_s = 30.0

    # --- Simulation ---------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
        
    )

    # --- Scene configuration ------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, 
        env_spacing=6.0,
        replicate_physics=True,
    )
    
    # --- Active robot (with regex pattern for cloning) ---------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"  # Regex pattern for all envs
    )

    # --- Degrees of freedom (Wheels) ----------------------------------
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # --- Action configuration -----------------------------------------
    action_scale = 1.0
    max_wheel_speed = 6.0
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]  # Polarity for differential drive

    # --- Camera configuration -----------------------------------------
    class CameraCfg:
        # Path will be per-environment in the code
        prim_path = (
            "/World/Robot/teko_urdf/TEKO_Body/TEKO_WallBack/TEKO_Camera/RearCamera"
        )
        width = 640
        height = 480
        frequency_hz = 15                 # Reasonable frequency
        focal_length = 3.6               # Close to real RPi v2 camera
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 16.0
        focus_distance = 2.0             # Reasonable focal distance

    camera = CameraCfg()

    # --- Goal robot configuration (with ArUco) ------------------------
    class GoalCfg:
        usd_path = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        prim_path = "{ENV_REGEX_NS}/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/test_marker.png"
        position = (1.0, 0.0, 0.0)
        aruco_offset = (0.1675, 0.0, -0.025)
        aruco_size = 0.05                # Small but readable

    goal = GoalCfg()

    # --- Observation and action spaces --------------------------------
    action_space = (2,)  # [left, right] wheel velocities
    observation_space = {
        "rgb": (3, 480, 640),  # PyTorch format: (C, H, W)
    }
