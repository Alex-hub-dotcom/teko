# SPDX-License-Identifier: BSD-3-Clause
#
# Reward functions for the TEKO environment (v3.1)
# ================================================

import torch
import numpy as np

from teko.tasks.direct.teko.penalties.penalties import (
    compute_time_penalty,
    compute_velocity_penalty_when_close,
    compute_oscillation_penalty,
    compute_wall_collision_penalty,
    compute_robot_collision_penalty
)

def compute_total_reward(env) -> torch.Tensor:
    female_pos, male_pos, surface_xy, surface_3d = env.get_sphere_distances_from_physics()

    # Debug counter (scalar) — safe
    dbg = getattr(env, "_dbg_counter", 0)
    if dbg % 50 == 0:
        try:
            print(f"[REWARD DEBUG] surface_xy={float(surface_xy[0].item()) :.4f}")
        except Exception:
            pass
    setattr(env, "_dbg_counter", dbg + 1)

    # 1) Distance reward
    surface_xy = torch.nan_to_num(surface_xy, nan=10.0, posinf=10.0, neginf=0.0)
    distance_reward = 2.0 * (1.0 - torch.tanh(surface_xy * 1.5)) - 2.0

    # 2) Progress reward
    if env.prev_distance is None:
        env.prev_distance = surface_xy.clone()
        progress_reward = torch.zeros_like(surface_xy)
    else:
        delta = env.prev_distance - surface_xy
        progress_reward = torch.clamp(delta * 40.0, min=-5.0, max=5.0)
        env.prev_distance = surface_xy.clone()

    # 3) Alignment
    alignment_bonus = compute_alignment_bonus(env, surface_xy)

    # 4) Success
    success_threshold = 0.03
    success_bonus = torch.where(surface_xy < success_threshold,
                                torch.tensor(100.0, device=env.device),
                                torch.tensor(0.0, device=env.device))

    # 5) Penalties (each already NaN-safe)
    time_penalty          = compute_time_penalty(env)
    velocity_penalty      = compute_velocity_penalty_when_close(env, surface_xy)
    oscillation_penalty   = compute_oscillation_penalty(env)
    wall_penalty          = compute_wall_collision_penalty(env)
    robot_collision_penalty = compute_robot_collision_penalty(env, surface_xy)

    # 6) Combine
    total_reward = (
        distance_reward
        + progress_reward
        + alignment_bonus
        + success_bonus
        - time_penalty
        - velocity_penalty
        - oscillation_penalty
        - wall_penalty
        - robot_collision_penalty
    )

    # Final safety clamp + NaN scrub
    total_reward = torch.clamp(total_reward, -20.0, 120.0)
    total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=-10.0)

    # Logging
    def _m(x): return float(torch.nan_to_num(x.mean(), nan=0.0).item())
    env.reward_components["distance"].append(_m(distance_reward))
    env.reward_components["progress"].append(_m(progress_reward))
    env.reward_components["alignment"].append(_m(alignment_bonus))
    env.reward_components["velocity_penalty"].append(_m(velocity_penalty))
    env.reward_components["oscillation_penalty"].append(_m(oscillation_penalty))
    env.reward_components["wall_penalty"].append(_m(wall_penalty))
    env.reward_components["collision_penalty"].append(_m(robot_collision_penalty))

    return total_reward