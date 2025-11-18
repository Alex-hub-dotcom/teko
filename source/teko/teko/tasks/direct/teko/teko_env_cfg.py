# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Environment Configuration (Torque-driven, Modular)
-------------------------------------------------------
Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.

Provides configuration for:
- Active torque-driven robot (spawn offset added to prevent floor clipping)
- Static goal robot (with ArUco marker)
- Camera setup
- Scene, simulation, and observation/action specs
"""

from __future__ import annotations
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from teko.tasks.direct.teko.robots.teko import TEKO_CONFIGURATION


@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Environment configuration for torque-driven TEKO robot."""

    # ------------------------------------------------------------------
    # General parameters
    # ------------------------------------------------------------------
    decimation = 2
    episode_length_s = 15.0  # SHORTER: 15 seconds instead of 30

    # ------------------------------------------------------------------
    # Simulation setup
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
    )

    # ------------------------------------------------------------------
    # Scene configuration
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Spawn offset (active robot only)
    # ------------------------------------------------------------------
    # Prevents the robot's wheels from spawning intersecting with the ground.
    # The robot will spawn 3 cm above the ground and settle naturally with gravity.
    robot_spawn_z_offset = 0.03

    # ------------------------------------------------------------------
    # Active robot configuration
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = TEKO_CONFIGURATION.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Wheel joints used for torque control
    dof_names = [
        "TEKO_Chassi_JointWheelFrontLeft",
        "TEKO_Chassi_JointWheelFrontRight",
        "TEKO_Chassi_JointWheelBackLeft",
        "TEKO_Chassi_JointWheelBackRight",
    ]

    # ------------------------------------------------------------------
    # Actuation parameters
    # ------------------------------------------------------------------
    action_scale = 1.0
    max_wheel_torque = 2.0  # INCREASED: 3.0 Nm for faster movement
    wheel_polarity = [1.0, 1.0, 1.0, 1.0]  # Left/Right differential polarity

    # ------------------------------------------------------------------
    # Camera configuration (rear RGB camera)
    # ------------------------------------------------------------------
    @configclass
    class CameraCfg:
        """RGB camera mounted on the back wall of TEKO."""
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

    # ------------------------------------------------------------------
    # Static goal robot configuration (with ArUco marker)
    # ------------------------------------------------------------------
    @configclass
    class GoalCfg:
        """Static goal robot used as docking target."""
        usd_path = "/workspace/teko/documents/CAD/USD/teko.usd"
        prim_path = "/World/envs/env_.*/RobotGoal"
        aruco_texture = "/workspace/teko/documents/Aruco/test_marker.png"
        position = (1.0, 0.0, 0.40)
        aruco_offset = (0.17, 0.0, -0.045)
        aruco_size = 0.05  # meters

    goal = GoalCfg()

    # ------------------------------------------------------------------
    # Observation and action spaces
    # ------------------------------------------------------------------
    action_space = (2,)  # [left, right] torque control inputs
    observation_space = {
        "rgb": (3, 480, 640),
    }

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    # Actions in [-1, 1] are scaled by max_wheel_torque:
    #   [1.0, 1.0]  → +3.0 Nm forward (faster than before)
    #   [-1.0, -1.0] → -3.0 Nm reverse
    #   [1.0, -1.0] → rotate in place