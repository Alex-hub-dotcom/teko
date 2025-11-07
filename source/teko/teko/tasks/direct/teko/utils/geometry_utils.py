# SPDX-License-Identifier: BSD-3-Clause

#Geometric utilities for TEKO environment.


import torch


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angle (radians) to quaternion [w, x, y, z]."""
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    z = torch.sin(half_yaw)
    return torch.stack([w, x, y, z], dim=1)
