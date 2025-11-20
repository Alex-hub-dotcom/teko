# SPDX-License-Identifier: BSD-3-Clause
"""
16-STAGE ULTRA-SMOOTH CURRICULUM FOR TEKO
=========================================
Stage 0:  Baby Steps        (0.05-0.15m, 0°, 0cm)
Stage 1:  Close Forward     (0.15-0.30m, 0°, 0cm)
Stage 2:  Medium Forward    (0.30-0.50m, 0°, 0cm)
Stage 3:  Tiny Offset       (0.30-0.50m, ±3°,   ±3cm)
Stage 4:  Tiny+ Offset      (0.30-0.50m, ±4.5°, ±4.5cm)
Stage 5:  Small Offset      (0.30-0.50m, ±6°,   ±6cm)
Stage 6:  Small+ Offset     (0.30-0.50m, ±8°,   ±8cm)
Stage 7:  Medium Offset     (0.30-0.50m, ±10°,  ±10cm)
Stage 8:  Medium+ Offset    (0.30-0.50m, ±13°,  ±13cm)
Stage 9:  Large Offset      (0.30-0.50m, ±15°,  ±15cm)
Stage 10: Large+ Offset     (0.30-0.50m, ±18°,  ±18cm)
Stage 11: Full Lateral      (0.30-0.50m, ±20°,  ±20cm)
Stage 12: 180° Close        (0.30-0.50m, 180°,  0cm)
Stage 13: 180° Offset       (0.30-0.50m, 180°±10°, ±10cm)
Stage 14: Arena Search      (0.80-1.50m, random yaw)
Stage 15: Full Autonomy     (random position, random yaw)

Advances at 85% success rate (plus your per-stage min steps in the trainer).
"""

import torch
import numpy as np
from ..utils.geometry_utils import yaw_to_quat


STAGE_NAMES = [
    "Stage 0:  Baby Steps (5-15cm)",
    "Stage 1:  Close Forward (15-30cm)",
    "Stage 2:  Medium Forward (30-50cm)",
    "Stage 3:  Tiny Offset (±3°, ±3cm)",
    "Stage 4:  Tiny+ Offset (±4.5°, ±4.5cm)",
    "Stage 5:  Small Offset (±6°, ±6cm)",
    "Stage 6:  Small+ Offset (±8°, ±8cm)",
    "Stage 7:  Medium Offset (±10°, ±10cm)",
    "Stage 8:  Medium+ Offset (±13°, ±13cm)",
    "Stage 9:  Large Offset (±15°, ±15cm)",
    "Stage 10: Large+ Offset (±18°, ±18cm)",
    "Stage 11: Full Lateral (±20°, ±20cm)",
    "Stage 12: 180° Close (turn around)",
    "Stage 13: 180° Offset (turn + align)",
    "Stage 14: Arena Search (far + random)",
    "Stage 15: Full Autonomy (production)",
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids):
    """Reset robot according to current curriculum stage."""
    stage = int(env.curriculum_level)

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
    elif stage == 11:
        _reset_stage11(env, env_ids)
    elif stage == 12:
        _reset_stage12(env, env_ids)
    elif stage == 13:
        _reset_stage13(env, env_ids)
    elif stage == 14:
        _reset_stage14(env, env_ids)
    elif stage == 15:
        _reset_stage15(env, env_ids)
    else:
        raise ValueError(f"Invalid curriculum stage: {stage}")


# =============================================================================
# Helpers
# =============================================================================

def _base_forward_reset(env, env_ids, min_dist: float, max_dist: float, yaw: torch.Tensor):
    """Helper: place robot in front of goal at given distance range."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * (max_dist - min_dist) + min_dist

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


def _offset_reset(env, env_ids, angle_deg: float, lateral_m: float):
    """Generic lateral-offset stage around 180°."""
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 0.30–0.50 m

    max_yaw = np.deg2rad(angle_deg)
    yaw = np.pi + (torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw)

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (
        torch.rand(num, device=env.device) * (2 * lateral_m) - lateral_m
    )
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# =============================================================================
# Stage implementations
# =============================================================================

# Stage 0: Baby Steps (0.05–0.15 m, straight)
def _reset_stage0(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi  # facing goal
    _base_forward_reset(env, env_ids, 0.05, 0.15, yaw)


# Stage 1: Close Forward (0.15–0.30 m)
def _reset_stage1(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.15, 0.30, yaw)


# Stage 2: Medium Forward (0.30–0.50 m)
def _reset_stage2(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi
    _base_forward_reset(env, env_ids, 0.30, 0.50, yaw)


# Stage 3: Tiny Offset (±3°, ±3 cm)
def _reset_stage3(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=3.0, lateral_m=0.03)


# Stage 4: Tiny+ Offset (±4.5°, ±4.5 cm)
def _reset_stage4(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=4.5, lateral_m=0.045)


# Stage 5: Small Offset (±6°, ±6 cm)
def _reset_stage5(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=6.0, lateral_m=0.06)


# Stage 6: Small+ Offset (±8°, ±8 cm)
def _reset_stage6(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=8.0, lateral_m=0.08)


# Stage 7: Medium Offset (±10°, ±10 cm)
def _reset_stage7(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=10.0, lateral_m=0.10)


# Stage 8: Medium+ Offset (±13°, ±13 cm)
def _reset_stage8(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=13.0, lateral_m=0.13)


# Stage 9: Large Offset (±15°, ±15 cm)
def _reset_stage9(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=15.0, lateral_m=0.15)


# Stage 10: Large+ Offset (±18°, ±18 cm)
def _reset_stage10(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=18.0, lateral_m=0.18)


# Stage 11: Full Lateral (±20°, ±20 cm)
def _reset_stage11(env, env_ids):
    _offset_reset(env, env_ids, angle_deg=20.0, lateral_m=0.20)


# Stage 12: 180° Close (facing away, 0 offset)
def _reset_stage12(env, env_ids):
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30
    yaw = torch.zeros(num, device=env.device)  # facing away

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 13: 180° Offset (facing away ±10°, ±10 cm)
def _reset_stage13(env, env_ids):
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30

    max_yaw = np.deg2rad(10.0)
    yaw = (torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw)  # 0° ± 10°

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.20 - 0.10)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 14: Arena Search (0.80–1.50 m, random yaw)
def _reset_stage14(env, env_ids):
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.70 + 0.80  # 0.80–1.50 m
    yaw = torch.rand(num, device=env.device) * 2 * np.pi

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.60 - 0.30)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 15: Full Autonomy (random in arena)
def _reset_stage15(env, env_ids):
    num = len(env_ids)

    # Example: 3m x 5m arena around goal
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
    """Set curriculum stage (0–15)."""
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """Return True if we should advance to the next curriculum stage."""
    max_level = len(STAGE_NAMES) - 1
    if current_level >= max_level:
        return False
    return success_rate >= 0.85
