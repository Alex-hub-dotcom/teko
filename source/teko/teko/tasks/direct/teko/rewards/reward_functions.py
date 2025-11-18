# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (BALANCED v4.0)
# ==========================================================

from __future__ import annotations
import torch
import numpy as np

def compute_total_reward(env) -> torch.Tensor:
    """Compute BALANCED reward - prevent explosion."""
    
    # Get positions
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    
    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    # ===== 1. DISTANCE REWARD (normalized, capped) =====
    # Use negative exponential to keep magnitude reasonable
    distance_reward = -torch.exp(surface_xy) + 1.0  # Range: [-inf, 1]
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=1.0)  # Cap at -10
    
    # ===== 2. PROGRESS REWARD (moderate, symmetric) =====
    progress = env.prev_distance - surface_xy
    progress_reward = 10.0 * progress  # Same weight for approach/retreat
    progress_reward = torch.clamp(progress_reward, min=-2.0, max=2.0)  # Cap magnitude
    env.prev_distance = surface_xy.clone()
    
    # ===== 3. ALIGNMENT REWARD =====
    active_yaw = torch.atan2(
        env.robot.data.root_quat_w[:, 2],
        env.robot.data.root_quat_w[:, 3]
    ) * 2.0
    alignment_reward = 0.5 * torch.cos(active_yaw)
    
    # ===== 4. VELOCITY PENALTY (very small) =====
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.005 * speed  # VERY small
    
    # ===== 5. OSCILLATION PENALTY (very small) =====
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.01 * action_diff
    env.prev_actions = env.actions.clone()
    
    # ===== 6. OUT OF BOUNDS PENALTY =====
    robot_pos_global = env.robot.data.root_pos_w
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins
    out_of_bounds = (
        (torch.abs(robot_pos_local[:, 0]) > 1.4) |
        (torch.abs(robot_pos_local[:, 1]) > 2.4)
    )
    boundary_penalty = torch.where(
        out_of_bounds,
        torch.tensor(-20.0, device=env.device),  # Big one-time penalty
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== 7. SUCCESS BONUS =====
    success_bonus = torch.where(
        surface_xy < 0.03,
        torch.tensor(100.0, device=env.device),  # HUGE bonus!
        torch.tensor(0.0, device=env.device)
    )
    
    # ===== TOTAL REWARD (CAPPED) =====
    total_reward = (
        distance_reward +
        progress_reward +
        alignment_reward +
        velocity_penalty +
        oscillation_penalty +
        boundary_penalty +
        success_bonus
    )
    
    # CRITICAL: Cap total reward to prevent explosion
    total_reward = torch.clamp(total_reward, min=-20.0, max=100.0)
    
    # Logging
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["progress"].append(progress_reward.mean().item())
    env.reward_components["alignment"].append(alignment_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    
    return total_reward