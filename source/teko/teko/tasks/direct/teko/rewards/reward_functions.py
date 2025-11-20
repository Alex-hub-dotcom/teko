# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (OPTIMIZED v6.1 - ALIGNMENT FIX)
# ==========================================================================
#
# Components:
#   1. Distance reward      – encourages being close to the docking point
#   2. Progress reward      – rewards getting closer over time
#   3. Alignment reward     – rear of robot aligned with goal
#   4. Velocity penalty     – discourages excessive speed
#   5. Oscillation penalty  – discourages jerky actions
#   6. Collision penalty    – HUGE penalty for crashes (anti-exploit)
#   7. Boundary penalty     – HUGE penalty for leaving arena
#   8. Success bonus        – large bonus for successful docking
#   9. Proximity bonus      – extra reward when very close
#  10. Survival bonus       – per-step reward to make crashing worse than timing out
#

from __future__ import annotations
import torch


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to yaw (rotation around Z axis).

    Args:
        quat: [N, 4] tensor with components [x, y, z, w]

    Returns:
        yaw: [N] tensor with yaw in radians
    """
    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to [-pi, pi] for stable error computation.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute balanced reward for vision-based docking.
    """
    device = env.device

    # ------------------------------------------------------------------
    # 0. Distance from docking spheres (our main geometric signal)
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (linear penalty)
    # ------------------------------------------------------------------
    distance_reward = -surface_xy  # ~[-2, 0] in typical range
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)

    # ------------------------------------------------------------------
    # 2. Progress reward (difference in distance)
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress
    progress_reward = torch.clamp(progress_reward, min=-2.0, max=2.0)
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. Alignment reward (rear of robot facing goal)
    # ------------------------------------------------------------------
    # Robot orientation (yaw from quaternion)
    robot_quat = env.robot.data.root_quat_w  # [N, 4] (x, y, z, w)
    robot_yaw = _quat_to_yaw(robot_quat)     # front yaw in world frame

    # Direction from robot to goal
    robot_pos = env.robot.data.root_pos_w    # [N, 3]
    goal_pos = env.goal_positions            # [N, 3]
    vec_to_goal = goal_pos - robot_pos       # vector in XY plane

    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])

    # Rear yaw = front yaw + pi (camera + connector are at the back)
    rear_yaw = robot_yaw + torch.pi

    # Alignment error = rear_yaw - goal_yaw (wrapped to [-pi, pi])
    yaw_error = _angle_wrap(rear_yaw - goal_yaw)

    # cos(0) = 1 when perfectly aligned, cos(pi) = -1 when opposite
    alignment_reward = 0.5 * torch.cos(yaw_error)

    # ------------------------------------------------------------------
    # 4. Velocity penalty (small)
    # ------------------------------------------------------------------
    lin_vel = env.robot.data.root_lin_vel_w  # [N, 3]
    speed = torch.norm(lin_vel[:, :2], dim=-1)  # XY speed
    velocity_penalty = -0.01 * speed

    # ------------------------------------------------------------------
    # 5. Oscillation penalty (small)
    # ------------------------------------------------------------------
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)

    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()

    # ------------------------------------------------------------------
    # 6. Collision penalty (ANTI-EXPLOIT)
    # ------------------------------------------------------------------
    # "Crash" = close AND fast BUT not successfully docked
    raw_success = surface_xy < 0.03
    collision = (surface_xy < 0.10) & (speed > 0.3) & (~raw_success)

    collision_penalty = torch.where(
        collision,
        torch.tensor(-500.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 7. Boundary penalty (out of arena)
    # ------------------------------------------------------------------
    robot_pos_global = env.robot.data.root_pos_w
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins

    out_of_bounds = (
        (torch.abs(robot_pos_local[:, 0]) > 1.4) |
        (torch.abs(robot_pos_local[:, 1]) > 2.4)
    )

    boundary_penalty = torch.where(
        out_of_bounds,
        torch.tensor(-500.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 8. Success bonus
    # ------------------------------------------------------------------
    success = raw_success  # within 3 cm
    success_bonus = torch.where(
        success,
        torch.tensor(100.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 9. Proximity bonus (helps final approach)
    # ------------------------------------------------------------------
    close = (surface_xy < 0.10) & (surface_xy >= 0.03) & (~collision)
    proximity_bonus = torch.where(
        close,
        torch.tensor(2.0, device=device),
        torch.tensor(0.0, device=device),
    )

    # ------------------------------------------------------------------
    # 10. Survival bonus (anti-crash exploit)
    # ------------------------------------------------------------------
    survival_bonus = torch.full_like(surface_xy, 0.3)

    # ------------------------------------------------------------------
    # TOTAL REWARD
    # ------------------------------------------------------------------
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        velocity_penalty +
        oscillation_penalty +
        collision_penalty +
        boundary_penalty +
        success_bonus +
        proximity_bonus +
        survival_bonus
    )

    # Allow nuclear -500 for crashes, ~100+ for success
    total_reward = torch.clamp(total_reward, min=-500.0, max=400.0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["progress"].append(progress_reward.mean().item())
    env.reward_components["alignment"].append(alignment_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    env.reward_components["collision_penalty"].append(collision_penalty.mean().item())
    env.reward_components["wall_penalty"].append(boundary_penalty.mean().item())

    return total_reward
