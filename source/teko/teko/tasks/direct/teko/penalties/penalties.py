# SPDX-License-Identifier: BSD-3-Clause
#
# Penalty functions for the TEKO environment (v3.2)
# -------------------------------------------------
# Independent of env.arena_size; uses fixed 3m × 5m limits.


import torch

def compute_time_penalty(env):
    return torch.full((env.scene.cfg.num_envs,), 0.01, device=env.device)

def _safe_norm(x, dim=-1):
    # Replace NaN/Inf before norm; guard against uninitialized physics at step 0
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.norm(x, dim=dim)
    return torch.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)

def compute_velocity_penalty_when_close(env, surface_xy):
    lin_vel = _safe_norm(env.robot.data.root_lin_vel_w[:, :2], dim=-1)
    close = surface_xy < 0.2
    penalty = torch.where(close, lin_vel * 0.1, torch.zeros_like(lin_vel))
    return torch.nan_to_num(penalty, nan=0.0, posinf=5.0, neginf=0.0)

def compute_oscillation_penalty(env):
    cur = torch.nan_to_num(env.robot.data.root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
    if not hasattr(env, "prev_lin_vel"):
        env.prev_lin_vel = cur.clone()
        return torch.zeros(env.scene.cfg.num_envs, device=env.device)
    prev = torch.nan_to_num(env.prev_lin_vel, nan=0.0, posinf=0.0, neginf=0.0)
    delta_v = _safe_norm(cur - prev, dim=-1)
    env.prev_lin_vel = cur.clone()
    return torch.nan_to_num(delta_v * 0.05, nan=0.0, posinf=5.0, neginf=0.0)

def compute_wall_collision_penalty(env):
    pos_global = torch.nan_to_num(env.robot.data.root_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
    origins = env.scene.env_origins
    pos_local = pos_global - origins
    out_of_bounds = ((torch.abs(pos_local[:, 0]) > 1.4) | (torch.abs(pos_local[:, 1]) > 2.4))
    return torch.where(out_of_bounds,
                       torch.tensor(10.0, device=env.device),
                       torch.tensor(0.0, device=env.device))

def compute_robot_collision_penalty(env, surface_xy):
    too_close = (surface_xy < 0.05) & (surface_xy >= 0.03)
    return torch.where(too_close,
                       torch.tensor(5.0, device=env.device),
                       torch.tensor(0.0, device=env.device))