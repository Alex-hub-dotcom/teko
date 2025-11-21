# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for the TEKO environment (v6.2 – AABB / arena-consistent)
==========================================================================

This module defines the scalar reward used in the TEKO vision-based docking task.
It decomposes the total reward into several interpretable components:

1. Distance reward
   - Uses the *connector sphere distance* in the XY plane (surface_xy).
   - Simply penalizes being far from the docking point:  r_dist ≈ -distance.
   - Encourages the agent to stay close to the docking interface.

2. Progress reward
   - Looks at the *change* in distance between steps.
   - If the robot moves closer (distance_t < distance_{t-1}), reward > 0.
   - If it moves away, reward < 0.
   - Strongly encourages monotonic approach behavior.

3. Alignment reward
   - Uses the robot’s orientation and the vector to the goal.
   - Measures how well the *rear* of the robot is aligned with the goal
     (because the connector + camera are on the back).
   - Uses cos(angular error), so perfectly aligned ≈ +0.5, opposite ≈ -0.5.

4. Velocity penalty
   - Small penalty proportional to the planar speed of the robot.
   - Discourages driving too fast (which often leads to crashes),
     but does not forbid moving.

5. Oscillation penalty
   - Penalizes large changes between consecutive actions.
   - Encourages smoother control instead of twitchy, rapidly-changing commands.

6. Collision penalty (connector-zone “crash”)
   - Detects “hard hits” near the docking interface:
       * distance < 10 cm
       * speed > 0.3 m/s
       * not already a successful dock
   - Applies a large negative reward (-500) to make crashing very undesirable.
   - Works together with the survival bonus to prevent "suicide for fast reset".

7. Boundary penalty (leaving the arena)
   - Uses the same arena limits as the environment:
       |x_local| > arena_half_x or |y_local| > arena_half_y
   - Applies -500 when the robot leaves the allowed region.
   - Fully consistent with the red boundary walls.

8. Success bonus
   - When the docking connectors are within 3 cm in the XY plane,
     the agent receives a big +100 reward.
   - This is the main sparse “task completion” signal.

9. Proximity bonus
   - Between 3 cm and 10 cm (near but not docked), gives a small +2 bonus.
   - Encourages reaching and staying near the docking point while it refines
     alignment and approach.

10. Survival bonus
    - Constant +0.3 every step.
    - Ensures that “surviving” the full episode without crashing is better
      than crashing early (even if crashes give some early progress reward).

The final reward is the sum of all components, clamped to [-500, 400].
"""

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
    Compute the scalar reward for the TEKO vision-based docking task.

    Args:
        env: TekoEnv instance (gives access to physics state and buffers)

    Returns:
        total_reward: [N] tensor of per-environment rewards
    """
    device = env.device

    # ------------------------------------------------------------------
    # 0. Distance from docking spheres (main geometric signal)
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()

    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 1. Distance reward (linear penalty)
    # ------------------------------------------------------------------
    # Negative of the distance in XY between connector spheres.
    distance_reward = -surface_xy
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

    # Direction from robot to goal in XY
    robot_pos = env.robot.data.root_pos_w    # [N, 3]
    goal_pos = env.goal_positions            # [N, 3]
    vec_to_goal = goal_pos - robot_pos
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
    # 6. Collision penalty (ANTI-EXPLOIT near connector)
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

    # Use the SAME arena limits as the environment & debug walls
    hx = float(env._arena_half_x)
    hy = float(env._arena_half_y)

    out_of_bounds = (
        (robot_pos_local[:, 0].abs() > hx) |
        (robot_pos_local[:, 1].abs() > hy)
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
