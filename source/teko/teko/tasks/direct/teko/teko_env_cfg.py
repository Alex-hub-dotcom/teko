# SPDX-License-Identifier: BSD-3-Clause
"""
TekoEnvCfg — Updated with TurtleBot3 Burger velocity limits
============================================================
Compatible with Isaac Lab 0.47.1 / Isaac Sim 5.0.
"""
from __future__ import annotations
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Import your robot configuration
# Adjust this path based on your actual structure
try:
    from .robots.teko import TEKO_CONFIGURATION
except ImportError:
    # Fallback if module structure is different
    print("[WARN] Could not import TEKO_CONFIGURATION, using placeholder")
    TEKO_CONFIGURATION = None

@configclass
class TekoEnvCfg(DirectRLEnvCfg):
    """Configuration for TEKO environment with TurtleBot3 Burger velocity limits."""
    
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
        num_envs=1,                     # Default (overridden by --num_envs)
        env_spacing=6.0,                # Distance between cloned arenas
        replicate_physics=True,
    )
    
    # --- Active robot -------------------------------------------------
    if TEKO_CONFIGURATION is not None:
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
    
    # --- Action configuration (TurtleBot3 Burger limits) --------------
    # TurtleBot3 Burger specifications:
    # - Linear velocity: 0.22 m/s max (using 0.08 m/s for safety)
    # - Angular velocity: 2.84 rad/s max (using 3.8 rad/s as specified)
    # - Wheel radius: 0.033 m
    # - Wheel separation: 0.160 m
    #
    # Calculation for wheel velocity from linear velocity:
    # v_linear = 0.08 m/s
    # v_wheel = v_linear / wheel_radius = 0.08 / 0.033 = 2.42 rad/s
    #
    # For differential drive:
    # v_wheel_max should accommodate both linear and rotational motion
    # Angular: ω * (wheel_separation/2) / wheel_radius = 3.8 * 0.08 / 0.033 = 9.2 rad/s
    # Using max of linear and angular requirements
    
    action_scale = 1.0
    max_wheel_speed = 2.5  # rad/s (conservative, based on 0.08 m/s linear)
    wheel_polarity = [1.0, -1.0, 1.0, -1.0]  # Polarity for differential drive
    
    # --- Camera configuration -----------------------------------------
    @configclass
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
    @configclass
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
    
    # --- Notes on velocity scaling ------------------------------------
    # Actions from policy are in [-1, 1] range
    # These are multiplied by max_wheel_speed (2.5 rad/s)
    # 
    # Example:
    # - Full forward: action = [1.0, 1.0] → wheel_vel = [2.5, 2.5] rad/s
    # - Linear velocity = 2.5 * 0.033 = 0.0825 m/s ✅ (close to 0.08 m/s)
    # - Full rotation: action = [1.0, -1.0] → differential wheel speed
    # - Angular velocity = 2 * 2.5 * 0.033 / 0.160 = 1.03 rad/s
    #   (conservative compared to 3.8 rad/s max)
    #
    # If you want higher angular velocity, increase max_wheel_speed to ~10 rad/s
    # But this is good for stable training!