# SPDX-License-Identifier: BSD-3-Clause
#
# Penalty functions for the TEKO environment.
# -------------------------------------------
# Fully Isaac Lab compatible (no pxr imports).
# Uses robot and goal data directly for all safety penalties.

import torch


# ------------------------------------------------------------------
# Velocity penalty
# ------------------------------------------------------------------
def compute_velocity_penalty(env, distance_error):
    """Penalize excessive torque near goal."""
    if env.actions is None:
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)
    action_magnitude = torch.norm(env.actions, dim=-1)
    distance_factor = torch.clamp(distance_error / 0.2, 0.0, 1.0)
    return action_magnitude * (1.0 - distance_factor) * 3.0


# ------------------------------------------------------------------
# Oscillation penalty
# ------------------------------------------------------------------
def compute_oscillation_penalty(env):
    """Penalize frequent reversals of wheel direction."""
    if env.prev_actions is None or env.actions is None:
        env.prev_actions = env.actions.clone() if env.actions is not None else None
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)

    action_product = env.actions * env.prev_actions
    reversal = (action_product < -0.5).any(dim=-1).float()
    penalty = reversal * 8.0
    env.prev_actions = env.actions.clone()
    return penalty


# ------------------------------------------------------------------
# Collision penalty
# ------------------------------------------------------------------
def compute_collision_penalty(env):
    """Penalize if robot gets too close to the goal (connector overlap)."""
    robot_pos = env.robot.data.root_pos_w
    goal_pos = env.goal_positions
    dock_distance = torch.norm(robot_pos - goal_pos, dim=-1)
    collision = dock_distance < 0.35  # approximate physical contact

    return torch.where(
        collision,
        torch.tensor(50.0, device=env.device),
        torch.tensor(0.0, device=env.device),
    )


# ------------------------------------------------------------------
# Wall penalty
# ------------------------------------------------------------------
def compute_wall_penalty(env):
    """Penalize approaching arena boundaries."""
    robot_pos = env.robot.data.root_pos_w
    half = env.arena_size / 2.0
    margin_x = half - torch.abs(robot_pos[:, 0])
    margin_y = half - torch.abs(robot_pos[:, 1])
    min_margin = torch.min(margin_x, margin_y)

    wall_threshold = 0.20
    penalty = torch.where(
        min_margin < wall_threshold,
        15.0 * (wall_threshold - min_margin),
        torch.tensor(0.0, device=env.device),
    )
    return penalty
