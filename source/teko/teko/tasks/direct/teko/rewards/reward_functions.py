# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (OPTIMIZED v6.1 - ANTI-EXPLOIT)
# ===========================================================================

from __future__ import annotations
import torch

# Shared thresholds – keep these consistent with _get_dones
SUCCESS_RADIUS = 0.03      # 3 cm → success
CLOSE_RADIUS   = 0.10      # 10 cm → "close"
MIN_SUCCESS_STEPS   = 5    # must survive at least this many steps to "succeed"
MIN_COLLISION_STEPS = 2    # don't punish collisions at step 0/1 (spawn glitches)


def compute_total_reward(env) -> torch.Tensor:
    """
    Compute balanced reward for vision-based docking.
    
    Components:
    1. Distance reward: Encourages approaching the goal
    2. Progress reward: Rewards getting closer over time
    3. Alignment reward: Encourages correct orientation
    4. Velocity penalty: Discourages excessive speed
    5. Oscillation penalty: Discourages jerky movements
    6. Collision penalty: HUGE penalty for crashing (prevents exploit)
    7. Boundary penalty: HUGE penalty for leaving arena
    8. Success bonus: Large reward for successful docking
    9. Proximity bonus: Extra reward when very close
    """
    device = env.device

    # ------------------------------------------------------------------
    # 0. Basic state: distances, episode length, velocities
    # ------------------------------------------------------------------
    _, _, surface_xy, _ = env.get_sphere_distances_from_physics()
    episode_len = env.episode_length_buf.to(surface_xy.device)

    # Initialize prev_distance if needed
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()

    # Linear velocity in XY
    lin_vel = env.robot.data.root_lin_vel_w
    speed = torch.norm(lin_vel[:, :2], dim=-1)

    # Raw geometric success (distance only)
    raw_success = surface_xy < SUCCESS_RADIUS

    # ------------------------------------------------------------------
    # 1. DISTANCE REWARD (linear, simpler)
    # ------------------------------------------------------------------
    distance_reward = -surface_xy                      # ~[-2, 0] for typical distances
    distance_reward = torch.clamp(distance_reward, -10.0, 0.0)

    # ------------------------------------------------------------------
    # 2. PROGRESS REWARD (symmetric)
    # ------------------------------------------------------------------
    progress = env.prev_distance - surface_xy          # positive if we got closer
    progress_reward = 10.0 * progress                  # scale up
    progress_reward = torch.clamp(progress_reward, -2.0, 2.0)
    env.prev_distance = surface_xy.clone()

    # ------------------------------------------------------------------
    # 3. ALIGNMENT REWARD (face the goal)
    # ------------------------------------------------------------------
    # Approximated from yaw; works because robot moves mainly in +X / -X
    active_yaw = torch.atan2(
        env.robot.data.root_quat_w[:, 2],
        env.robot.data.root_quat_w[:, 3]
    ) * 2.0
    alignment_reward = 0.5 * torch.cos(active_yaw)

    # ------------------------------------------------------------------
    # 4. VELOCITY PENALTY (small global)
    # ------------------------------------------------------------------
    velocity_penalty = -0.01 * speed

    # ------------------------------------------------------------------
    # 5. OSCILLATION PENALTY (small, based on action changes)
    # ------------------------------------------------------------------
    if env.prev_actions is None or env.actions is None:
        # First step: no oscillation info yet
        oscillation_penalty = torch.zeros_like(surface_xy)
        env.prev_actions = torch.zeros(
            (surface_xy.shape[0], 2), device=device
        )
    else:
        action_diff = torch.norm(env.actions - env.prev_actions, dim=-1)
        oscillation_penalty = -0.02 * action_diff
        env.prev_actions = env.actions.clone()

    # ------------------------------------------------------------------
    # 6. COLLISION PENALTY (ANTI-EXPLOIT!)
    # ------------------------------------------------------------------
    # Only consider collision if we've run at least MIN_COLLISION_STEPS,
    # and only when not inside the raw success radius.
    collision_mask = (
        (surface_xy < CLOSE_RADIUS) &
        (speed > 0.3) &
        ~raw_success &
        (episode_len >= MIN_COLLISION_STEPS)
    )

    collision_penalty = torch.where(
        collision_mask,
        torch.tensor(-200.0, device=device),
        torch.tensor(0.0, device=device)
    )

    # ------------------------------------------------------------------
    # 7. BOUNDARY PENALTY (arena walls)
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
        torch.tensor(-200.0, device=device),
        torch.tensor(0.0, device=device)
    )

    # ------------------------------------------------------------------
    # 8. SUCCESS BONUS (step-gated)
    # ------------------------------------------------------------------
    success_mask = raw_success & (episode_len >= MIN_SUCCESS_STEPS)

    success_bonus = torch.where(
        success_mask,
        torch.tensor(100.0, device=device),
        torch.tensor(0.0, device=device)
    )

    # ------------------------------------------------------------------
    # 9. PROXIMITY BONUS (to help final approach)
    # ------------------------------------------------------------------
    close_mask = (
        (surface_xy < CLOSE_RADIUS) &
        (surface_xy >= SUCCESS_RADIUS) &
        ~collision_mask
    )

    proximity_bonus = torch.where(
        close_mask,
        torch.tensor(2.0, device=device),
        torch.tensor(0.0, device=device)
    )

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
        proximity_bonus
    )

    # Safety clamp (collision/boundary can go to -200; success to +100)
    total_reward = torch.clamp(total_reward, min=-200.0, max=100.0)

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
