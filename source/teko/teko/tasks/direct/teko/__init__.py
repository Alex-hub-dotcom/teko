# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


gym.register(
    id="Template-Teko-Direct-v0",
    entry_point="teko.tasks.direct.teko.teko_env:TekoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "teko.tasks.direct.teko.teko_env_cfg:TekoEnvCfg",
        "skrl_cfg_entry_point": "teko.tasks.direct.teko.teko_brain:skrl_ppo_cfg.yaml",
    },
)
