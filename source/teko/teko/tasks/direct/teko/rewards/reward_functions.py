# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment.
# =========================================
# Simplified and compatible with Isaac Lab base container.
# Uses connector tip distance only via env data (no pxr).
# Focuses on forward progress (X-axis), lateral alignment (Y),
# and stable orientation, with explicit penalties.

import torch
import numpy as np

# Import penalty functions
from teko.tasks.direct.teko.penalties.penalties import (
    compute_velocity_penalty,
    compute_oscillation_penalty,
    compute_collision_penalty,
    compute_wall_penalty,
)


# ------------------------------------------------------------------
# Total reward
# ------------------------------------------------------------------
def compute_total_reward(env) -> torch.Tensor:
    """Compute total reward combining reward shaping and penalties."""

    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    robot_quat = env.robot.data.root_quat_w

    # ------------------------------------------------------------------
    # 1. Axial (X-axis) distance
    # ------------------------------------------------------------------
    target_distance = 0.43  # ideal dock offset between robot bases
    dock_distance = torch.abs(robot_pos[:, 0] - goal_pos[:, 0]) - target_distance
    dock_distance = torch.clamp(dock_distance, 0.0, None)
    distance_error = dock_distance

    # ------------------------------------------------------------------
    # 2. Distance reward (exponential decay)
    # ------------------------------------------------------------------
    distance_reward = 15.0 * torch.exp(-distance_error / 0.05)

    # ------------------------------------------------------------------
    # 3. Lateral alignment (Y-axis)
    # ------------------------------------------------------------------
    y_error = torch.abs(robot_pos[:, 1] - goal_pos[:, 1])
    y_reward = 5.0 * torch.exp(-y_error / 0.05)

    # ------------------------------------------------------------------
    # 4. Orientation alignment (Yaw)
    # ------------------------------------------------------------------
    robot_yaw = torch.atan2(
        2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
    )
    target_yaw = torch.tensor(np.pi, device=env.device)  # facing goal
    yaw_error = torch.abs(robot_yaw - target_yaw)
    yaw_error = torch.min(yaw_error, 2 * np.pi - yaw_error)
    yaw_reward = 8.0 * torch.exp(-yaw_error / 0.2)

    # ------------------------------------------------------------------
    # 5. Progress reward
    # ------------------------------------------------------------------
    progress_reward = compute_smart_progress_reward(env, dock_distance)

    # ------------------------------------------------------------------
    # 6. Success bonus
    # ------------------------------------------------------------------
    success_bonus = torch.where(
        distance_error < 0.02,  # within 2 cm of ideal X distance
        torch.tensor(50.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )

    # ------------------------------------------------------------------
    # 7. External penalties
    # ------------------------------------------------------------------
    velocity_penalty = compute_velocity_penalty(env, distance_error)
    oscillation_penalty = compute_oscillation_penalty(env)
    collision_penalty = compute_collision_penalty(env)
    wall_penalty = compute_wall_penalty(env)

    # ------------------------------------------------------------------
    # 8. Combine all terms
    # ------------------------------------------------------------------
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

    # Logging for diagnostics
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["alignment"].append(y_reward.mean().item())
    env.reward_components["orientation"].append(yaw_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    env.reward_components["collision_penalty"].append(collision_penalty.mean().item())
    env.reward_components["wall_penalty"].append(wall_penalty.mean().item())

    return total_reward


# ------------------------------------------------------------------
# Progress shaping
# ------------------------------------------------------------------
def compute_smart_progress_reward(env, current_distance: torch.Tensor) -> torch.Tensor:
    """Reward reduction of distance toward the target along X-axis."""
    if env.prev_distance is None:
        env.prev_distance = current_distance.clone()
        return torch.zeros_like(current_distance)

    progress = env.prev_distance - current_distance
    reward = torch.where(
        progress > 0,
        torch.clamp(progress * 10.0, 0.0, 2.0),  # cap positive reward
        torch.clamp(progress * 5.0, -1.0, 0.0),  # mild penalty when regressing
    )

    env.prev_distance = current_distance.clone()
    return reward
