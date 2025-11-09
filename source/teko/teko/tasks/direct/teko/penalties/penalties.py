# SPDX-License-Identifier: BSD-3-Clause
#
# Penalty functions for the TEKO environment.
# -------------------------------------------
# Multi-environment support.

import torch


def compute_time_penalty(env):
    """Small penalty for each step to encourage faster completion."""
    return torch.full((env.scene.cfg.num_envs,), 0.01, device=env.device)


def compute_velocity_penalty_when_close(env, surface_xy_distance):
    """
    Penalize high velocity when close to goal (encourages careful alignment).
    
    Args:
        surface_xy_distance: (num_envs,) - planar distance for each environment
    """
    # Get linear velocity magnitude for all environments
    velocity = torch.norm(env.robot.data.root_lin_vel_w, dim=-1)  # (num_envs,)
    
    # Only penalize when very close to goal (< 20cm)
    close_threshold = 0.20
    is_close = surface_xy_distance < close_threshold
    
    # Penalty scales with velocity when close
    penalty = torch.where(
        is_close,
        velocity * 2.0,  # Higher velocity = higher penalty when close
        torch.tensor(0.0, device=env.device)
    )
    
    return penalty


def compute_oscillation_penalty(env):
    """
    Penalize frequent reversals of wheel direction.
    Detects when action signs flip between steps.
    """
    if env.prev_actions is None or env.actions is None:
        if env.actions is not None:
            env.prev_actions = env.actions.clone()
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)
    
    # Check if action product is negative (sign reversal)
    action_product = env.actions * env.prev_actions
    reversal = (action_product < -0.5).any(dim=-1).float()  # (num_envs,)
    penalty = reversal * 5.0
    
    env.prev_actions = env.actions.clone()
    return penalty


def compute_wall_collision_penalty(env):
    """Large penalty for hitting arena walls with extended boundaries."""
    robot_pos_global = env.robot.data.root_pos_w
    env_origins = env.scene.env_origins
    robot_pos_local = robot_pos_global - env_origins
    
    # Extended boundaries: ±4.0m x ±4.0m
    out_of_bounds = (
        (torch.abs(robot_pos_local[:, 0]) > 4.0) |
        (torch.abs(robot_pos_local[:, 1]) > 4.0)
    )
    
    penalty = torch.where(
        out_of_bounds,
        torch.tensor(5.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    return penalty


# Robot collision penalty is computed inline in reward function
def compute_robot_collision_penalty(env, surface_xy_distance):
    """Gentle penalty for colliding with static robot."""
    collision_threshold = 0.05  # 5cm
    success_threshold = 0.03    # 3cm
    
    collision = (surface_xy_distance < collision_threshold) & (surface_xy_distance >= success_threshold)
    
    penalty = torch.where(
        collision,
        torch.tensor(2.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    return penalty