# SPDX-License-Identifier: BSD-3-Clause

# Penalty functions for TEKO environment.

import torch

def compute_velocity_penalty(env, distance_error):
    """Penalize excessive wheel torque near goal."""
    if env.actions is None:
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)
    action_magnitude = torch.norm(env.actions, dim=-1)
    distance_factor = torch.clamp(distance_error / 0.2, 0.0, 1.0)
    return action_magnitude * (1.0 - distance_factor) * 3.0


def compute_oscillation_penalty(env):
    """Detect frequent reversals in wheel torque (anti-oscillation)."""
    if env.prev_actions is None or env.actions is None:
        env.prev_actions = env.actions.clone() if env.actions is not None else None
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)

    action_product = env.actions * env.prev_actions
    direction_reversal = (action_product < -0.5).any(dim=-1).float()
    penalty = direction_reversal * 8.0
    env.prev_actions = env.actions.clone()
    return penalty


def compute_collision_penalty(env):
    """Penalize collisions with the goal robot."""
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    collision = distance < 0.35
    return torch.where(collision, torch.tensor(50.0, device=env.device), torch.tensor(0.0, device=env.device))


def compute_wall_penalty(env):
    """Penalize approaching arena boundaries."""
    robot_pos = env.robot.data.root_pos_w
    half_size = env.arena_size / 2.0
    x_margin = half_size - torch.abs(robot_pos[:, 0])
    y_margin = half_size - torch.abs(robot_pos[:, 1])
    min_margin = torch.min(x_margin, y_margin)
    wall_threshold = 0.20
    penalty = torch.where(
        min_margin < wall_threshold,
        15.0 * (wall_threshold - min_margin),
        torch.tensor(0.0, device=env.device),
    )
    return penalty
