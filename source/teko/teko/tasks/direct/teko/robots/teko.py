# SPDX-License-Identifier: BSD-3-Clause
# TEKO Robot Configuration — clean and simulation-safe
# No omni.usd or pxr imports, only Isaac Lab API calls.

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

TEKO_PATH = "/workspace/teko/documents/CAD/USD/teko.usd"

WHEEL_JOINTS = [
    "TEKO_Chassi_JointWheelFrontLeft",
    "TEKO_Chassi_JointWheelFrontRight",
    "TEKO_Chassi_JointWheelBackLeft",
    "TEKO_Chassi_JointWheelBackRight",
]

TEKO_CONFIGURATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TEKO_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,  # Higher for stability
            solver_velocity_iteration_count=4,
            sleep_threshold=0.001,
            stabilization_threshold=0.0001,
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.3,      # was 0.1
            angular_damping=0.3,     # was 0.15
            max_linear_velocity=2.0,
            max_angular_velocity=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,    # a bit more tolerance
            rest_offset=0.0,
            torsional_patch_radius=0.003,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={joint: 0.0 for joint in WHEEL_JOINTS},
        joint_vel={joint: 0.0 for joint in WHEEL_JOINTS},
    ),
    actuators={
        "wheel_actuators": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            effort_limit_sim=1.0,  # ↓ less torque
            stiffness=0.0,
            damping=0.5,           # slightly softer
            friction=0.1,          # less joint friction
            armature=0.001,        # a bit more inertia
        ),
    },
)