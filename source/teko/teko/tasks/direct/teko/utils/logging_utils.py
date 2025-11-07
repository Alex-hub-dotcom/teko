# SPDX-License-Identifier: BSD-3-Clause

#Logging utilities for TEKO environment.


import numpy as np


def collect_episode_stats(env):
    """Compute and return rolling statistics over recent episodes."""
    if len(env.episode_rewards) == 0:
        return {}

    stats = {
        'mean_reward': np.mean(env.episode_rewards[-100:]),
        'mean_length': np.mean(env.episode_lengths[-100:]),
        'success_rate': np.mean(env.episode_successes[-100:]) if env.episode_successes else 0.0,
        'reward_components': {
            k: np.mean(v[-100:]) if v else 0.0
            for k, v in env.reward_components.items()
        },
    }
    return stats
