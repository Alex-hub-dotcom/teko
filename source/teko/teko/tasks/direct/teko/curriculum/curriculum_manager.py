# SPDX-License-Identifier: BSD-3-Clause
"""
16-STAGE ULTRA-SMOOTH CURRICULUM FOR TEKO
=========================================

This file defines how the TEKO robot is spawned for each curriculum stage.
Only the *initial pose* (position + yaw) is changed per stage; rewards,
actions, etc. are defined elsewhere.

Important convention:
- yaw = π  -> "dock-ready" orientation (rear side / camera facing the goal)
- yaw = 0  -> robot turned 180° away from dock (needs to turn around)

Stage 0:  Baby Steps        (0.05–0.15 m, yaw ≈ π,       0 cm)
Stage 1:  Close Forward     (0.15–0.30 m, yaw ≈ π,       0 cm)
Stage 2:  Medium Forward    (0.30–0.50 m, yaw ≈ π,       0 cm)
Stage 3:  Tiny Offset       (0.30–0.50 m, π±3°,          ±3 cm)
Stage 4:  Tiny+ Offset      (0.30–0.50 m, π±4.5°,        ±4.5 cm)
Stage 5:  Small Offset      (0.30–0.50 m, π±6°,          ±6 cm)
Stage 6:  Small+ Offset     (0.30–0.50 m, π±8°,          ±8 cm)
Stage 7:  Medium Offset     (0.30–0.50 m, π±10°,         ±10 cm)
Stage 8:  Medium+ Offset    (0.30–0.50 m, π±13°,         ±13 cm)
Stage 9:  Large Offset      (0.30–0.50 m, π±15°,         ±15 cm)
Stage 10: Large+ Offset     (0.30–0.50 m, π±18°,         ±18 cm)
Stage 11: Full Lateral      (0.30–0.50 m, π±20°,         ±20 cm)
Stage 12: 180° Close        (0.30–0.50 m, yaw ≈ 0,       0 cm)
Stage 13: 180° Offset       (0.30–0.50 m, 0°±10°,        ±10 cm)
Stage 14: Arena Search      (0.80–1.50 m, random yaw)
Stage 15: Full Autonomy     (random position, random yaw)

Advancement:
- Logic to move to the next stage is implemented in the trainer
  (e.g., success rate >= 85% + minimum steps per stage).
"""

import numpy as np
import torch

from ..utils.geometry_utils import yaw_to_quat


# Descriptive names for pretty logging
STAGE_NAMES = [
    "Stage 0:  Baby Steps (5–15 cm)",
    "Stage 1:  Close Forward (15–30 cm)",
    "Stage 2:  Medium Forward (30–50 cm)",
    "Stage 3:  Tiny Offset (±3°, ±3 cm)",
    "Stage 4:  Tiny+ Offset (±4.5°, ±4.5 cm)",
    "Stage 5:  Small Offset (±6°, ±6 cm)",
    "Stage 6:  Small+ Offset (±8°, ±8 cm)",
    "Stage 7:  Medium Offset (±10°, ±10 cm)",
    "Stage 8:  Medium+ Offset (±13°, ±13 cm)",
    "Stage 9:  Large Offset (±15°, ±15 cm)",
    "Stage 10: Large+ Offset (±18°, ±18 cm)",
    "Stage 11: Full Lateral (±20°, ±20 cm)",
    "Stage 12: 180° Close (turn around)",
    "Stage 13: 180° Offset (turn + align)",
    "Stage 14: Arena Search (far + random)",
    "Stage 15: Full Autonomy (production)",
]


# =============================================================================
# Dispatcher
# =============================================================================

def reset_environment_curriculum(env, env_ids):
    """
    Reset selected environments according to the current curriculum stage.

    - `env.curriculum_level` (0–15) chooses which reset function is used.
    - Only the robot's root pose is modified here; goal positions are fixed.
    """
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
    """
    Helper for "forward docking" stages.

    Places the robot directly in front of the goal, at a random distance
    in [min_dist, max_dist], with a fixed yaw.

    Convention:
    - yaw ≈ π  -> rear side (camera) faces the goal (dock-ready)
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * (max_dist - min_dist) + min_dist

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40  # TEKO root height

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


def _offset_reset(env, env_ids, angle_deg: float, lateral_m: float):
    """
    Helper for lateral-offset stages around yaw ≈ π (rear facing goal).

    - Distance: 0.30–0.50 m from goal
    - Yaw:      π ± angle_deg
    - Lateral:  ± lateral_m in Y
    """
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

# Stage 0: Baby Steps (0.05–0.15 m, nearly perfect alignment)
def _reset_stage0(env, env_ids):
    num = len(env_ids)
    yaw = torch.ones(num, device=env.device) * np.pi  # dock-ready
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
    """
    Robot starts close but facing AWAY from the goal (needs to turn ~180°).
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 0.30–0.50 m

    yaw = torch.zeros(num, device=env.device)  # front towards goal, rear away

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1]
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 13: 180° Offset (facing away ±10°, ±10 cm)
def _reset_stage13(env, env_ids):
    """
    Robot starts close, facing away with a small yaw + lateral offset.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.20 + 0.30  # 0.30–0.50 m

    max_yaw = np.deg2rad(10.0)
    yaw = torch.rand(num, device=env.device) * (2 * max_yaw) - max_yaw  # 0° ± 10°

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.20 - 0.10)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 14: Arena Search (0.80–1.50 m, random yaw)
def _reset_stage14(env, env_ids):
    """
    Robot starts farther away and may need to search for the goal.
    """
    num = len(env_ids)
    dist = torch.rand(num, device=env.device) * 0.70 + 0.80  # 0.80–1.50 m
    yaw = torch.rand(num, device=env.device) * 2 * np.pi       # fully random

    x = env.goal_positions[env_ids, 0] - dist
    y = env.goal_positions[env_ids, 1] + (torch.rand(num, device=env.device) * 0.60 - 0.30)
    z = torch.ones(num, device=env.device) * 0.40

    pos = torch.stack([x, y, z], dim=1)
    quat = yaw_to_quat(yaw)
    env.robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=1), env_ids=env_ids)


# Stage 15: Full Autonomy (random in arena)
def _reset_stage15(env, env_ids):
    """
    Robot spawns anywhere in a 3 x 5 m area around the goal, random yaw.
    This approximates a production-like setup.
    """
    num = len(env_ids)

    # Example: ~3m (X) x ~5m (Y) area around the goal
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
    """
    Set the curriculum stage (0–15) on the environment and print a nice log.
    """
    max_level = len(STAGE_NAMES) - 1
    level = max(0, min(max_level, int(level)))
    env.curriculum_level = level

    print(f"\n{'=' * 70}")
    print(f"[CURRICULUM] {STAGE_NAMES[level]}")
    print(f"{'=' * 70}\n")


def should_advance_curriculum(success_rate: float, current_level: int) -> bool:
    """
    Decide whether to advance to the next curriculum stage.

    NOTE:
    - This function *only* checks the success rate.
    - The trainer enforces a minimum number of steps per stage.
    """
    max_level = len(STAGE_NAMES) - 1
    if current_level >= max_level:
        return False

    return success_rate >= 0.85
