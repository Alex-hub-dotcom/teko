"""
Policy Network for TEKO Vision-Based Docking (Final Version)
============================================================
- Compatible with Isaac Lab 0.47.1 and SKRL PPO
- Uses a visual encoder (SimpleCNN or MobileNetV3)
- Outputs Gaussian wheel velocity actions in [-1, 1]
"""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin
from source.teko.teko.tasks.direct.teko.agents.cnn_model import create_visual_encoder


class PolicyNetwork(GaussianMixin, Model):
    """Policy: RGB -> wheel velocities"""

    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)

        # ------------------------------------------------------------------
        # Visual encoder (choose "simple" for fast iteration or "mobilenet")
        # ------------------------------------------------------------------
        self.encoder = create_visual_encoder(
            architecture="simple",      # "simple" or "mobilenet"
            feature_dim=256,
            pretrained=False
        )

        # ------------------------------------------------------------------
        # Policy head: maps visual features -> continuous wheel velocities
        # ------------------------------------------------------------------
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh()  # Actions ∈ [-1, 1]
        )

        # Log std for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    # ----------------------------------------------------------------------
    # Forward computation
    # ----------------------------------------------------------------------
    def compute(self, inputs, role):
        """
        Args:
            inputs: dict provided by SKRL, containing "states"
                    (usually {"states": rgb_tensor})
        Returns:
            actions: tensor [B, num_actions]
            log_std: tensor for Gaussian variance
            {}
        """
        states = inputs["states"]

        # Extract RGB image tensor from nested dict
        if isinstance(states, dict):
            if "policy" in states:
                rgb = states["policy"]["rgb"]
            elif "rgb" in states:
                rgb = states["rgb"]
            else:
                raise ValueError(f"Cannot find RGB in states: {list(states.keys())}")
        else:
            rgb = states

        # Ensure correct tensor type and device
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.as_tensor(rgb, dtype=torch.float32, device=self.device)

        # Add batch dimension if single environment
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        # Expected shape: [B, 3, 480, 640]
        # CNN handles normalization internally
        features = self.encoder(rgb)

        # Produce mean actions (Gaussian policy)
        actions = self.policy(features)

        return actions, self.log_std, {}
