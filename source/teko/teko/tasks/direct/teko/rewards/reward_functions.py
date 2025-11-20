# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (OPTIMIZED v6.1 - ANTI-EXPLOIT + ORIENTATION FIX)
# ==========================================================================================

from __future__ import annotations
import torch


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute balanced reward for vision-based docking.
    
    Components:
    1. Distance reward: Encourages approaching the goal
    2. Progress reward: Rewards getting closer over time
    3. Alignment reward: Encourages correct orientation (rear towards goal)
    4. Velocity penalty: Discourages excessive speed
    5. Oscillation penalty: Discourages jerky movements
    6. Collision penalty: HUGE penalty for crashing (prevents exploit)
    7. Boundary penalty: HUGE penalty for leaving arena
    8. Success bonus: Large reward for successful docking
    9. Proximity bonus: Encourages final approach
    10. Survival bonus: Rewards staying alive (timeout >> crash)
    """
    
    # ------------------------------------------------------------------
    # 0. Geometry (sphere distances)
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    
    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
    
    # ------------------------------------------------------------------
    # 1. DISTANCE REWARD (linear)
    #    Penalize being far from docking connector (in XY).
    # ------------------------------------------------------------------
    distance_reward = -surface_xy  # ~[-2, 0] typically
    distance_reward = torch.clamp(distance_reward, min=-10.0, max=0.0)
    
    # ------------------------------------------------------------------
    # 2. PROGRESS REWARD (symmetric)
    #    Positive if we get closer, negative if we move away.
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy
    progress_reward = 5.0 * progress  # was 10.0, softer for PPO stability
    progress_reward = torch.clamp(progress_reward, min=-2.0, max=2.0)
    env.prev_distance = surface_xy.clone()
    
    # ------------------------------------------------------------------
    # 3. ALIGNMENT REWARD (rear pointing to goal)
    # ------------------------------------------------------------------
    # Robot yaw from quaternion (world frame)
    # yaw = 2 * atan2(qz, qw)
    quat = env.robot.data.root_quat_w
    active_yaw = torch.atan2(quat[:, 2], quat[:, 3]) * 2.0

    # Goal direction in world frame (vector from robot base to goal connector)
    robot_pos = env.robot.data.root_pos_w
    goal_pos  = env.goal_positions
    goal_vec  = goal_pos - robot_pos
    goal_yaw  = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])

    # TEKO docks backwards: rear of robot should point towards goal,
    # assuming robot's "forward" is +x in its local frame.
    desired_yaw = goal_yaw + torch.pi

    # Wrap yaw error to [-pi, pi]
    yaw_error = active_yaw - desired_yaw
    yaw_error = (yaw_error + torch.pi) % (2 * torch.pi) - torch.pi

    # Alignment reward: 1 when perfectly aligned, -1 when opposite
    alignment_reward = 0.5 * torch.cos(yaw_error)

    # ------------------------------------------------------------------
    # 4. VELOCITY PENALTY (small)
    # ------------------------------------------------------------------
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)
    velocity_penalty = -0.01 * speed  # small penalty for speed
    
    # ------------------------------------------------------------------
    # 5. OSCILLATION PENALTY (small)
    #    Penalize big changes in actions from one step to the next.
    # ------------------------------------------------------------------
    if env.prev_actions is None:
        env.prev_actions = torch.zeros_like(env.actions)
    
    action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
    oscillation_penalty = -0.02 * action_diff
    env.prev_actions = env.actions.clone()
    
    # ------------------------------------------------------------------
    # 6. COLLISION PENALTY (ANTI-EXPLOIT)
    # ------------------------------------------------------------------
    # Detect collision: too close + moving fast = crash
    collision = (surface_xy < 0.10) & (speed > 0.3) & (surface_xy >= 0.03)
    
    collision_penalty = torch.where(
        collision,
        torch.tensor(-500.0, device=env.device),  # NUCLEAR penalty!
        torch.tensor(0.0, device=env.device)
    )
    
    # ------------------------------------------------------------------
    # 7. BOUNDARY PENALTY (ANTI-EXPLOIT)
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
        torch.tensor(-500.0, device=env.device),  # NUCLEAR penalty!
        torch.tensor(0.0, device=env.device)
    )
    
    # ------------------------------------------------------------------
    # 8. SUCCESS BONUS
    # ------------------------------------------------------------------
    success = surface_xy < 0.03  # Within 3 cm = success
    success_bonus = torch.where(
        success,
        torch.tensor(100.0, device=env.device),  # large reward!
        torch.tensor(0.0, device=env.device)
    )
    
    # ------------------------------------------------------------------
    # 9. PROXIMITY BONUS (helps final approach)
    # ------------------------------------------------------------------
    close = (surface_xy < 0.10) & (surface_xy >= 0.03) & ~collision
    proximity_bonus = torch.where(
        close,
        torch.tensor(2.0, device=env.device),
        torch.tensor(0.0, device=env.device)
    )
    
    # ------------------------------------------------------------------
    # 10. SURVIVAL BONUS (anti-crash exploit)
    # ------------------------------------------------------------------
    # Slightly strong reward per step survived - makes timeout >> crash,
    # but not so strong that "camping" is better than docking.
    survival_bonus = torch.full_like(surface_xy, 0.2)  # was 0.3
    
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
    
    # Safety clamp (collision/boundary can go to -500; success+survival can exceed 100)
    total_reward = torch.clamp(total_reward, min=-500.0, max=400.0)
    
    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    env.reward_components["distance"].append(distance_reward.mean().item())
    env.reward_components["progress"].append(progress_reward.mean().item())
    env.reward_components["alignment"].append(alignment_reward.mean().item())
    env.reward_components["velocity_penalty"].append(velocity_penalty.mean().item())
    env.reward_components["oscillation_penalty"].append(oscillation_penalty.mean().item())
    env.reward_components["collision_penalty"].append(collision_penalty.mean().item())
    env.reward_components["wall_penalty"].append(boundary_penalty.mean().item())
    
    return total_reward
