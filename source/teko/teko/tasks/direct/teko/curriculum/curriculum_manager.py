# SPDX-License-Identifier: BSD-3-Clause
"""
6-Stage Curriculum Manager for TEKO Environment
================================================
Stage 0: Baby Steps (0.15-0.25m, 0° yaw)
Stage 1: Forward Drive (0.3-0.6m, 0° yaw)
Stage 2: Lateral Alignment (0.3-0.6m, ±20cm, ±20°)
Stage 3: 180° Turn (0.3-0.6m, 180° yaw)
Stage 4: Arena Search (0.8-1.5m, random yaw)
Stage 5: Full Autonomy (random position, random yaw)
"""

import torch
import numpy as np
from ..utils.geometry_utils import yaw_to_quat


STAGE_NAMES = [
    "Stage 0: Baby Steps",
    "Stage 1: Forward Drive", 
    "Stage 2: Lateral Alignment",
    "Stage 3: 180° Turn",
    "Stage 4: Arena Search",
    "Stage 5: Full Autonomy"
]


def reset_environment_curriculum(env, env_ids):
    """Reset robot according to current curriculum stage."""
    if env.curriculum_level == 0:
        _reset_stage0_baby_steps(env, env_ids)
    elif env.curriculum_level == 1:
        _reset_stage1_forward_drive(env, env_ids)
    elif env.curriculum_level == 2:
        _reset_stage2_lateral_alignment(env, env_ids)
    elif env.curriculum_level == 3:
        _reset_stage3_180_turn(env, env_ids)
    elif env.curriculum_level == 4:
        _reset_stage4_arena_search(env, env_ids)
    else:
        _reset_stage5_full_autonomy(env, env_ids)


# ============================================================================
# Stage 0: Baby Steps (0.15-0.25m, 0° yaw)
# ============================================================================
def _reset_stage0_baby_steps(env, env_ids):
    """Very close, perfectly aligned - just learn to move forward."""
    num_reset = len(env_ids)
    
    # Distance: 0.15-0.25m
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.10 + 0.15
    
    # Yaw: 180° (facing goal, perfect alignment)
    spawn_yaw = torch.ones(num_reset, device=env.device) * np.pi
    
    # Position
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1]  # No lateral offset
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1), 
        env_ids=env_ids
    )


# ============================================================================
# Stage 1: Forward Drive (0.3-0.6m, 0° yaw)
# ============================================================================
def _reset_stage1_forward_drive(env, env_ids):
    """Medium distance, still aligned - learn speed control."""
    num_reset = len(env_ids)
    
    # Distance: 0.3-0.6m
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.30 + 0.30
    
    # Yaw: 180° (perfect alignment)
    spawn_yaw = torch.ones(num_reset, device=env.device) * np.pi
    
    # Position
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1]
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1),
        env_ids=env_ids
    )


# ============================================================================
# Stage 2: Lateral Alignment (0.3-0.6m, ±20cm offset, ±20° yaw)
# ============================================================================
def _reset_stage2_lateral_alignment(env, env_ids):
    """Close but misaligned - learn steering corrections."""
    num_reset = len(env_ids)
    
    # Distance: 0.3-0.6m
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.30 + 0.30
    
    # Yaw: 180° ± 20° (small misalignment)
    spawn_yaw = np.pi + (torch.rand(num_reset, device=env.device) * 0.70 - 0.35)
    
    # Position with lateral offset ±20cm
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1] + \
              (torch.rand(num_reset, device=env.device) * 0.40 - 0.20)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1),
        env_ids=env_ids
    )


# ============================================================================
# Stage 3: 180° Turn (0.3-0.6m, 180° yaw)
# ============================================================================
def _reset_stage3_180_turn(env, env_ids):
    """Close but facing AWAY - learn to turn around and dock."""
    num_reset = len(env_ids)
    
    # Distance: 0.3-0.6m
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.30 + 0.30
    
    # Yaw: 0° (facing AWAY from goal)
    spawn_yaw = torch.zeros(num_reset, device=env.device)
    
    # Position (directly behind, no offset)
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1]
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1),
        env_ids=env_ids
    )


# ============================================================================
# Stage 4: Arena Search (0.8-1.5m, random yaw, goal visible)
# ============================================================================
def _reset_stage4_arena_search(env, env_ids):
    """Further away, random orientation - learn to search and approach."""
    num_reset = len(env_ids)
    
    # Distance: 0.8-1.5m (still in line of sight)
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.70 + 0.80
    
    # Yaw: fully random
    spawn_yaw = torch.rand(num_reset, device=env.device) * 2 * np.pi
    
    # Position with lateral variation
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1] + \
              (torch.rand(num_reset, device=env.device) * 0.60 - 0.30)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1),
        env_ids=env_ids
    )


# ============================================================================
# Stage 5: Full Autonomy (random spawn, random yaw)
# ============================================================================
def _reset_stage5_full_autonomy(env, env_ids):
    """Production scenario - anywhere in arena, any orientation."""
    num_reset = len(env_ids)
    
    # Arena bounds (assuming 3m x 5m arena centered on origin)
    # X: ±1.4m, Y: ±2.4m
    spawn_x = env.goal_positions[env_ids, 0] + \
              (torch.rand(num_reset, device=env.device) * 2.6 - 1.3)
    spawn_y = env.goal_positions[env_ids, 1] + \
              (torch.rand(num_reset, device=env.device) * 4.6 - 2.3)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    
    # Fully random orientation
    spawn_yaw = torch.rand(num_reset, device=env.device) * 2 * np.pi
    
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    
    env.robot.write_root_pose_to_sim(
        torch.cat([spawn_pos, spawn_quat], dim=1),
        env_ids=env_ids
    )


# ============================================================================
# Curriculum Progression
# ============================================================================
def set_curriculum_level(env, level: int):
    """Set curriculum stage (0-5)."""
    env.curriculum_level = max(0, min(5, level))
    print(f"[CURRICULUM] {STAGE_NAMES[env.curriculum_level]}")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Check if should advance to next stage (80% success threshold)."""
    if current_level >= 5:  # Already at final stage
        return False
    return success_rate >= 0.80