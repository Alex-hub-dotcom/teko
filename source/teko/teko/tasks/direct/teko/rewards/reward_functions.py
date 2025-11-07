# SPDX-License-Identifier: BSD-3-Clause

# Reward functions for the TEKO environment.
# =========================================
# This module defines all positive reinforcement terms and combines
# them with external penalties into the total reward signal.


import torch
import numpy as np

# Import penalty functions
from teko.tasks.direct.teko.penalties.penalties import (
    compute_velocity_penalty,
    compute_oscillation_penalty,
    compute_collision_penalty,
    compute_wall_penalty,
)


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute total reward combining reward shaping and penalties.
    """
    robot_pos = env.robot.data.root_pos_w
    robot_quat = env.robot.data.root_quat_w
    goal_pos = env.goal_positions

    # --- Distance metrics ---
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    target_distance = 0.43
    distance_error = torch.abs(distance - target_distance)

    # --- 1. Distance reward (bounded exponential) ---
    distance_reward = 15.0 * torch.exp(-distance_error / 0.05)

    # --- 2. Lateral alignment reward (Y-axis) ---
    y_error = torch.abs(robot_pos[:, 1] - goal_pos[:, 1])
    y_reward = 5.0 * torch.exp(-y_error / 0.05)

    # --- 3. Orientation alignment reward ---
    robot_yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2)
    )
    target_yaw = torch.tensor(np.pi, device=env.device)
    yaw_error = torch.abs(robot_yaw - target_yaw)
    yaw_error = torch.min(yaw_error, 2 * np.pi - yaw_error)
    yaw_reward = 8.0 * torch.exp(-yaw_error / 0.2)

    # --- External penalties ---
    velocity_penalty = compute_velocity_penalty(env, distance_error)
    oscillation_penalty = compute_oscillation_penalty(env)
    collision_penalty = compute_collision_penalty(env)
    wall_penalty = compute_wall_penalty(env)

    # --- 4. Progress reward (directional progress) ---
    progress_reward = compute_smart_progress_reward(env, distance)

    # --- 5. Success bonus ---
    success_bonus = torch.where(
        distance_error < 0.02,  # Within 2 cm
        torch.tensor(50.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )

    # --- Combine all terms ---
    total_reward = (
        distance_reward
        + y_reward
        + yaw_reward
        + progress_reward
        + success_bonus
        - velocity_penalty
        - oscillation_penalty
        - collision_penalty
        - wall_penalty
    )

    # --- Logging for diagnostics ---
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["alignment"].append(y_reward.mean().item())
    env.reward_components["orientation"].append(yaw_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    env.reward_components["collision_penalty"].append(collision_penalty.mean().item())
    env.reward_components["wall_penalty"].append(wall_penalty.mean().item())

    return total_reward


def compute_smart_progress_reward(env, current_distance: torch.Tensor) -> torch.Tensor:
    """
    Reward movement toward the target while discouraging oscillation.
    """
    if env.prev_distance is None:
        env.prev_distance = current_distance.clone()
        return torch.zeros_like(current_distance)

    progress = env.prev_distance - current_distance
    reward = torch.where(
        progress > 0,
        torch.clamp(progress * 10.0, 0.0, 2.0),  # Cap positive reward
        torch.clamp(progress * 5.0, -1.0, 0.0)   # Small penalty for moving away
    )

    env.prev_distance = current_distance.clone()
    return reward
