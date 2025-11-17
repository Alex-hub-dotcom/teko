# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (v3.2 FINAL)
# ================================================

from __future__ import annotations
import torch
import numpy as np

from teko.tasks.direct.teko.penalties.penalties import (
    compute_time_penalty,
    compute_velocity_penalty_when_close,
    compute_oscillation_penalty,
    compute_wall_collision_penalty,
    compute_robot_collision_penalty
)

def compute_total_reward(env) -> torch.Tensor:  # REMOVED TYPE HINT
    """Compute total reward with STRONGER progress incentive."""
    
    # Get positions
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    robot_pos = env.robot.data.root_pos_w
    
    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    # ===== 1. DISTANCE REWARD (primary signal) =====
    # INCREASED weight from 2.0 to 5.0
    distance_reward = -5.0 * surface_xy
    
    # ===== 2. PROGRESS REWARD (approach bonus) =====
    progress = env.prev_distance - surface_xy
    # INCREASED multiplier from 10.0 to 30.0
    progress_reward = torch.where(
        progress > 0,
        30.0 * progress,  # BIG reward for getting closer
        5.0 * progress    # Small penalty for moving away
    )
    env.prev_distance = surface_xy.clone()
    
    # ===== 3. ALIGNMENT REWARD =====
    active_yaw = torch.atan2(
        env.robot.data.root_quat_w[:, 2],
        env.robot.data.root_quat_w[:, 3]
    ) * 2.0
    alignment_reward = 0.5 * torch.cos(active_yaw)
    
    # ===== 4. VELOCITY PENALTY (REDUCED) =====
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    # REDUCED from 0.05 to 0.01
    velocity_penalty = -0.01 * speed
    
    # ===== 5. OSCILLATION PENALTY (REDUCED) =====
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    # REDUCED from 0.1 to 0.02
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()
    
    # ===== 6. COLLISION PENALTY =====
    collision_penalty = torch.zeros_like(surface_xy)
    
    # ===== 7. SUCCESS BONUS =====
    success_bonus = torch.where(
        surface_xy < 0.03,
        torch.tensor(50.0, device=env.device),  # HUGE bonus for docking!
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== TOTAL REWARD =====
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        velocity_penalty +
        oscillation_penalty +
        collision_penalty +
        success_bonus
    )
    
    # Logging
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["progress"].append(progress_reward.mean().item())
    env.reward_components["alignment"].append(alignment_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    
    return total_reward

# ------------------------------------------------------------
# Alignment Bonus
# ------------------------------------------------------------
def compute_alignment_bonus(env, surface_xy):
    """Small bonus when close (<30 cm) and yaw error < 15°."""
    robot_quat = env.robot.data.root_quat_w

    # Extract yaw from quaternion
    robot_yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    )

    # Target yaw (facing the docking goal)
    target_yaw = torch.tensor(np.pi, device=env.device)
    yaw_error = torch.abs(robot_yaw - target_yaw)
    yaw_error = torch.min(yaw_error, 2 * np.pi - yaw_error)

    # Bonus if both close and well-aligned
    is_close = surface_xy < 0.30
    is_aligned = yaw_error < (15.0 * np.pi / 180.0)
    return torch.where(is_close & is_aligned,
                       torch.tensor(5.0, device=env.device),
                       torch.tensor(0.0, device=env.device))