# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment.
# =========================================
# Uses sphere connector positions for accurate docking rewards.
# Multi-environment support.

import torch
import numpy as np

from teko.tasks.direct.teko.penalties.penalties import (
    compute_time_penalty,
    compute_velocity_penalty_when_close,
    compute_oscillation_penalty,
    compute_wall_collision_penalty,
    compute_robot_collision_penalty
)


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward based on sphere connector positions.
    
    Reward structure:
    - Time penalty: -0.01 per step
    - Distance reward: +reward for getting closer
    - Alignment bonus: extra reward when close and aligned
    - Success bonus: +10.0 when docked (surface_xy < 3cm)
    - Penalties: walls (-5.0), robot collision (-2.0), oscillation, high velocity when close
    """
    
    # Get sphere positions and distances using PhysX
    female_pos, male_pos, surface_xy, surface_3d = env.get_sphere_distances_from_physics()
    
    # ------------------------------------------------------------------
    # 1. Time penalty (encourage faster completion)
    # ------------------------------------------------------------------
    time_penalty = compute_time_penalty(env)
    
    # ------------------------------------------------------------------
    # 2. Distance reward (dense shaping)
    # ------------------------------------------------------------------
    distance_reward = compute_distance_reward(env, surface_xy)
    
    # ------------------------------------------------------------------
    # 3. Alignment bonus (extra reward when close and well-aligned)
    # ------------------------------------------------------------------
    alignment_bonus = compute_alignment_bonus(env, surface_xy)
    
    # ------------------------------------------------------------------
    # 4. Success bonus (terminal reward)
    # ------------------------------------------------------------------
    success_threshold = 0.03  # 3cm
    success_bonus = torch.where(
        surface_xy < success_threshold,
        torch.tensor(10.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    # ------------------------------------------------------------------
    # 5. Penalties
    # ------------------------------------------------------------------
    velocity_penalty = compute_velocity_penalty_when_close(env, surface_xy)
    oscillation_penalty = compute_oscillation_penalty(env)
    wall_penalty = compute_wall_collision_penalty(env)
    robot_collision_penalty = compute_robot_collision_penalty(env, surface_xy)
    
    # ------------------------------------------------------------------
    # 6. Combine all terms
    # ------------------------------------------------------------------
    total_reward = (
        distance_reward
        + alignment_bonus
        + success_bonus
        - time_penalty
        - velocity_penalty
        - oscillation_penalty
        - wall_penalty
        - robot_collision_penalty
    )
    
    # Logging for diagnostics
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["alignment"].append(alignment_bonus.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    env.reward_components["wall_penalty"].append(wall_penalty.mean().item())
    env.reward_components["collision_penalty"].append(robot_collision_penalty.mean().item())
    
    return total_reward


def compute_distance_reward(env, surface_xy):
    """
    Reward for reducing distance to goal.
    Uses delta distance to encourage continuous progress.
    
    Args:
        surface_xy: (num_envs,) - planar distance for each environment
    """
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
        return torch.zeros_like(surface_xy)
    
    # Progress = reduction in distance
    progress = env.prev_distance - surface_xy
    
    # Reward progress, small penalty for regression
    reward = torch.where(
        progress > 0,
        progress * 20.0,  # Strong reward for getting closer
        progress * 5.0    # Mild penalty for moving away
    )
    
    env.prev_distance = surface_xy.clone()
    return reward


def compute_alignment_bonus(env, surface_xy):
    """
    Extra reward when robot is close and well-aligned.
    Encourages final precision.
    
    Args:
        surface_xy: (num_envs,) - planar distance for each environment
    """
    # Get robot orientation (num_envs, 4) quaternion
    robot_quat = env.robot.data.root_quat_w
    
    # Compute yaw from quaternion
    robot_yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    )
    
    # Target yaw (facing goal) - π radians (180 degrees)
    target_yaw = torch.tensor(np.pi, device=env.device)
    yaw_error = torch.abs(robot_yaw - target_yaw)
    yaw_error = torch.min(yaw_error, 2 * np.pi - yaw_error)
    
    # Bonus only when close (< 20cm) and well-aligned (< 10 degrees)
    is_close = surface_xy < 0.20
    is_aligned = yaw_error < (10.0 * np.pi / 180.0)
    
    bonus = torch.where(
        is_close & is_aligned,
        torch.tensor(2.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    return bonus