################# cfg - Multi-Environment Compatible (FINAL)
# SPDX-License-Identifier: BSD-3-Clause
"""
TekoEnvCfg — configuration for TEKO robot docking with RGB camera.
Fully compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
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
    """Configuration for TEKO environment (multi-environment)."""

    # --- General parameters -------------------------------------------
    decimation = 2                      # Physics/render step ratio
    episode_length_s = 30.0             # Duration of one episode

    # --- Simulation ---------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # --- Scene configuration ------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,                     # Default (overridden by --num_envs in trainer)
        env_spacing=6.0,                # Distance between cloned arenas
        replicate_physics=True,
    )

    # --- Active robot -------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"   # Regex pattern for all envs
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
        """RGB camera mounted on the robot."""
        prim_path = (
            "/World/envs/env_.*/Robot/teko_urdf/TEKO_Body/"
            "TEKO_WallBack/TEKO_Camera/RearCamera"
        )
        width = 640
        height = 480
        frequency_hz = 15
        focal_length = 3.6
        horiz_aperture = 4.8
        vert_aperture = 3.6
        f_stop = 16.0
        focus_distance = 2.0

    camera = CameraCfg()

    # --- Goal robot configuration (with ArUco marker) -----------------
    class GoalCfg:
        """Static goal robot used as docking target."""
        usd_path = "/workspace/teko/documents/CAD/USD/teko_goal.usd"
        prim_path = "/World/envs/env_.*/RobotGoal"  # multi-env regex path
        aruco_texture = "/workspace/teko/documents/Aruco/test_marker.png"
        position = (1.0, 0.0, 0.0)
        aruco_offset = (0.1675, 0.0, -0.025)
        aruco_size = 0.05  # meters

    goal = GoalCfg()

    # --- Observation and action spaces --------------------------------
    action_space = (2,)  # [left, right] wheel velocities
    observation_space = {
        "rgb": (3, 480, 640),  # PyTorch format: (C, H, W)
    }
