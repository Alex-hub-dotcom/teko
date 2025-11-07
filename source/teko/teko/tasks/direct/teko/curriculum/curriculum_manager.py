# SPDX-License-Identifier: BSD-3-Clause

#Curriculum manager for TEKO environment.


import torch
import numpy as np
from ..utils.geometry_utils import yaw_to_quat


def reset_environment_curriculum(env, env_ids):
    """Reset robot position and orientation according to current curriculum level."""
    if env.curriculum_level == 0:
        _reset_close(env, env_ids)
    elif env.curriculum_level == 1:
        _reset_medium(env, env_ids)
    else:
        _reset_hard(env, env_ids)


def _reset_close(env, env_ids):
    num_reset = len(env_ids)
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.3 + 0.5
    spawn_yaw = torch.ones(num_reset, device=env.device) * np.pi + \
                (torch.rand(num_reset, device=env.device) * 0.2 - 0.1)
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1] + (torch.rand(num_reset, device=env.device) * 0.1 - 0.05)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    env.robot.write_root_pose_to_sim(torch.cat([spawn_pos, spawn_quat], dim=1), env_ids=env_ids)


def _reset_medium(env, env_ids):
    num_reset = len(env_ids)
    spawn_distance = torch.rand(num_reset, device=env.device) * 0.8 + 0.8
    spawn_yaw = torch.rand(num_reset, device=env.device) * (np.pi / 4) - (np.pi / 8) + np.pi
    spawn_x = env.goal_positions[env_ids, 0] - spawn_distance
    spawn_y = env.goal_positions[env_ids, 1] + (torch.rand(num_reset, device=env.device) * 0.4 - 0.2)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    env.robot.write_root_pose_to_sim(torch.cat([spawn_pos, spawn_quat], dim=1), env_ids=env_ids)


def _reset_hard(env, env_ids):
    num_reset = len(env_ids)
    spawn_x = torch.rand(num_reset, device=env.device) * (env.arena_size - 0.5) - (env.arena_size / 2 - 0.25)
    spawn_y = torch.rand(num_reset, device=env.device) * (env.arena_size - 0.5) - (env.arena_size / 2 - 0.25)
    spawn_z = torch.ones(num_reset, device=env.device) * 0.40
    spawn_yaw = torch.rand(num_reset, device=env.device) * 2 * np.pi
    spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=1)
    spawn_quat = yaw_to_quat(spawn_yaw)
    env.robot.write_root_pose_to_sim(torch.cat([spawn_pos, spawn_quat], dim=1), env_ids=env_ids)


def set_curriculum_level(env, level: int):
    """Clamp and set curriculum level."""
    env.curriculum_level = max(0, min(2, level))
    print(f"[INFO] Curriculum level set to {env.curriculum_level}")
