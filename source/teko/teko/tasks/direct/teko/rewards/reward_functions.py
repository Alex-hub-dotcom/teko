# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (OPTIMIZED v5.0)
# ===========================================================

from __future__ import annotations
import torch


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute balanced reward for vision-based docking.
    
    Components:
    1. Distance reward: Encourages approaching the goal
    2. Progress reward: Rewards getting closer over time
    3. Alignment reward: Encourages correct orientation
    4. Velocity penalty: Discourages excessive speed
    5. Oscillation penalty: Discourages jerky movements
    6. Boundary penalty: Heavily penalizes leaving arena
    7. Success bonus: Large reward for successful docking
    """
    
    # Get sphere distances
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    
    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    # ===== 1. DISTANCE REWARD (linear, simpler) =====
    # Linear penalty based on distance - easier to tune
    distance_reward = -surface_xy  # Range: ~[-2, 0] for typical distances
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)
    
    # ===== 2. PROGRESS REWARD (symmetric) =====
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress  # Scale up for significance
    progress_reward = torch.clamp(progress_reward, min=-2.0, max=2.0)
    env.prev_distance = surface_xy.clone()
    
    # ===== 3. ALIGNMENT REWARD =====
    # Reward for facing the goal (yaw = 0 or 180 degrees)
    active_yaw = torch.atan2(
        env.robot.data.root_quat_w[:, 2],
        env.robot.data.root_quat_w[:, 3]
    ) * 2.0
    alignment_reward = 0.5 * torch.cos(active_yaw)
    
    # ===== 4. VELOCITY PENALTY (small) =====
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.01 * speed  # Small penalty for speed
    
    # ===== 5. OSCILLATION PENALTY (small) =====
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()
    
    # ===== 6. BOUNDARY PENALTY =====
    robot_pos_global = env.robot.data.root_pos_w
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins
    
    out_of_bounds = (
        (torch.abs(robot_pos_local[:, 0]) > 1.4) |
        (torch.abs(robot_pos_local[:, 1]) > 2.4)
    )
    
    boundary_penalty = torch.where(
        out_of_bounds,
        torch.tensor(-50.0, device=env.device),  # Large penalty
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== 7. SUCCESS BONUS =====
    success = surface_xy < 0.03  # Within 3cm = success
    success_bonus = torch.where(
        success,
        torch.tensor(100.0, device=env.device),  # Huge reward!
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== 8. PROXIMITY BONUS (helps final approach) =====
    # Extra reward when very close to encourage final docking
    close = (surface_xy < 0.10) & (surface_xy >= 0.03)
    proximity_bonus = torch.where(
        close,
        torch.tensor(2.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== TOTAL REWARD =====
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        velocity_penalty +
        oscillation_penalty +
        boundary_penalty +
        success_bonus +
        proximity_bonus
    )
    
    # Safety clamp (should rarely trigger with balanced components)
    total_reward = torch.clamp(total_reward, min=-50.0, max=100.0)
    
    # ===== LOGGING =====
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["progress"].append(progress_reward.mean().item())
    env.reward_components["alignment"].append(alignment_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    
    return total_reward