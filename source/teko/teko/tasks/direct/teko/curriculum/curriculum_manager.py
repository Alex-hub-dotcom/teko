# SPDX-License-Identifier: BSD-3-Clause
"""
12-STAGE ULTRA-GRADUAL CURRICULUM FOR TEKO
===========================================
Stage 0:  Baby Steps        (0.05-0.15m, 0°, 0cm)
Stage 1:  Close Forward     (0.15-0.30m, 0°, 0cm)
Stage 2:  Medium Forward    (0.30-0.50m, 0°, 0cm)
Stage 3:  Tiny Offset       (0.30-0.50m, ±3°, ±3cm)
Stage 4:  Small Offset      (0.30-0.50m, ±6°, ±6cm)
Stage 5:  Medium Offset     (0.30-0.50m, ±10°, ±10cm)
Stage 6:  Large Offset      (0.30-0.50m, ±15°, ±15cm)
Stage 7:  Full Lateral      (0.30-0.50m, ±20°, ±20cm)
Stage 8:  180° Close        (0.30-0.50m, 180°, 0cm)
Stage 9:  180° Offset       (0.30-0.50m, 180°±10°, ±10cm)
Stage 10: Arena Search      (0.80-1.50m, random yaw)
Stage 11: Full Autonomy     (random position, random yaw)

Advances at 85% success rate, minimum 15k steps per stage
"""

import torch
import numpy as np
from ..utils.geometry_utils import yaw_to_quat


STAGE_NAMES = [
    "Stage 0: Baby Steps (5-15cm)",
    "Stage 1: Close Forward (15-30cm)",
    "Stage 2: Medium Forward (30-50cm)",
    "Stage 3: Tiny Offset (±3°, ±3cm)",
    "Stage 4: Small Offset (±6°, ±6cm)",
    "Stage 5: Medium Offset (±10°, ±10cm)",
    "Stage 6: Large Offset (±15°, ±15cm)",
    "Stage 7: Full Lateral (±20°, ±20cm)",
    "Stage 8: 180° Close (turn around)",
    "Stage 9: 180° Offset (turn + align)",
    "Stage 10: Arena Search (far + random)",
    "Stage 11: Full Autonomy (production)"
]


def reset_environment_curriculum(env, env_ids):
    """Reset robot according to current curriculum stage."""
    stage = env.curriculum_level
    
    if stage == 0:
        _reset_stage0(env, env_ids)
    elif stage == 1:
        _reset_stage1(env, env_ids)
    elif stage == 2:
        _reset_stage2(env, env_ids)
    elif stage == 3:
        _reset_stage3(env, env_ids)
    elif stage == 4:
        _reset_stage4(env, env_ids)
    elif stage == 5:
        _reset_stage5(env, env_ids)
    elif stage == 6:
        _reset_stage6(env, env_ids)
    elif stage == 7:
        _reset_stage7(env, env_ids)
    elif stage == 8:
        _reset_stage8(env, env_ids)
    elif stage == 9:
        _reset_stage9(env, env_ids)
    elif stage == 10:
        _reset_stage10(env, env_ids)
    else:  # stage == 11
        _reset_stage11(env, env_ids)


# =============================================================================
# STAGE 0: Baby Steps (0.05-0.15m, 0°, 0cm)
# =============================================================================
def _reset_stage0(env, env_ids):
    """Super close, perfect alignment - just breathe forward."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.10 + 0.05  # 5-15cm
    yaw = torch.ones(num, device=env.device) * np.pi        # 180° (facing goal)
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]                       # No offset
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 1: Close Forward (0.15-0.30m, 0°, 0cm)
# =============================================================================
def _reset_stage1(env, env_ids):
    """Close range, perfect alignment - learn gentle approach."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.15 + 0.15  # 15-30cm
    yaw = torch.ones(num, device=env.device) * np.pi
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 2: Medium Forward (0.30-0.50m, 0°, 0cm)
# =============================================================================
def _reset_stage2(env, env_ids):
    """Medium distance, perfect alignment - speed control."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 30-50cm
    yaw = torch.ones(num, device=env.device) * np.pi
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 3: Tiny Offset (0.30-0.50m, ±3°, ±3cm)
# =============================================================================
def _reset_stage3(env, env_ids):
    """First misalignment - tiny offset and rotation."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = np.pi + (torch.rand(num, device=env.device) * 0.105 - 0.0525)  # ±3° = ±0.0524 rad
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.06 - 0.03)  # ±3cm
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 4: Small Offset (0.30-0.50m, ±6°, ±6cm)
# =============================================================================
def _reset_stage4(env, env_ids):
    """Small misalignment - learn basic steering."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = np.pi + (torch.rand(num, device=env.device) * 0.21 - 0.105)  # ±6° = ±0.105 rad
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.12 - 0.06)  # ±6cm
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 5: Medium Offset (0.30-0.50m, ±10°, ±10cm)
# =============================================================================
def _reset_stage5(env, env_ids):
    """Medium misalignment - moderate corrections."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = np.pi + (torch.rand(num, device=env.device) * 0.35 - 0.175)  # ±10° = ±0.175 rad
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.20 - 0.10)  # ±10cm
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 6: Large Offset (0.30-0.50m, ±15°, ±15cm)
# =============================================================================
def _reset_stage6(env, env_ids):
    """Large misalignment - aggressive corrections."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = np.pi + (torch.rand(num, device=env.device) * 0.524 - 0.262)  # ±15° = ±0.262 rad
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.30 - 0.15)  # ±15cm
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 7: Full Lateral (0.30-0.50m, ±20°, ±20cm)
# =============================================================================
def _reset_stage7(env, env_ids):
    """Maximum lateral challenge - master alignment."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = np.pi + (torch.rand(num, device=env.device) * 0.698 - 0.349)  # ±20° = ±0.349 rad
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.40 - 0.20)  # ±20cm
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 8: 180° Close (0.30-0.50m, 180°, 0cm)
# =============================================================================
def _reset_stage8(env, env_ids):
    """Facing AWAY - learn to turn around and dock."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = torch.zeros(num, device=env.device)  # 0° = facing away!
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 9: 180° Offset (0.30-0.50m, 180°±10°, ±10cm)
# =============================================================================
def _reset_stage9(env, env_ids):
    """Facing away + misaligned - turn and align."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = (torch.rand(num, device=env.device) * 0.35 - 0.175)  # 0° ± 10°
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.20 - 0.10)
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 10: Arena Search (0.80-1.50m, random yaw)
# =============================================================================
def _reset_stage10(env, env_ids):
    """Far away, random orientation - search and approach."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.70 + 0.80  # 80-150cm
    yaw = torch.rand(num, device=env.device) * 2 * np.pi
    
    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.60 - 0.30)
    z = torch.ones(num, device=env.device) * 0.40
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# STAGE 11: Full Autonomy (random position, random yaw)
# =============================================================================
def _reset_stage11(env, env_ids):
    """Production mode - anywhere in arena."""
    num = len(env_ids)
    
    # Random position in 3m x 5m arena
    x = env.goal_positions[env_ids, 0] + (torch.rand(num, device=env.device) * 2.6 - 1.3)
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 4.6 - 2.3)
    z = torch.ones(num, device=env.device) * 0.40
    
    yaw = torch.rand(num, device=env.device) * 2 * np.pi
    
    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# Curriculum Control
# =============================================================================
def set_curriculum_level(env, level: int):
    """Set curriculum stage (0-11)."""
    env.curriculum_level = max(0, min(11, level))
    print(f"\n{'='*70}")
    print(f"[CURRICULUM] {STAGE_NAMES[env.curriculum_level]}")
    print(f"{'='*70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Check if should advance (85% success threshold)."""
    if current_level >= 11:
        return False
    return success_rate >= 0.85
