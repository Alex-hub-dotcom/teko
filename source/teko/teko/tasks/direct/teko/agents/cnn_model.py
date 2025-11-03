"""
CNN Feature Extractor for TEKO Docking
--------------------------------------
Provides two encoders:
- DockingCNN: MobileNetV3-Small (pretrained, efficient)
- SimpleCNN: Lightweight CNN for faster training
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ================================================================
#  DockingCNN (MobileNetV3-Small backbone)
# ================================================================
class DockingCNN(nn.Module):
    """
    Visual feature extractor for robot docking.
    Uses pretrained MobileNetV3-Small for efficiency.
    """
    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        """
        Args:
            feature_dim: Output feature dimension
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)

        # Feature extractor (remove classifier)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        mobilenet_out_dim = 576  # MobileNetV3-small output channels

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(mobilenet_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from RGB images.
        Args:
            x: RGB tensor [B, 3, H, W] in [0, 1]
        Returns:
            features: [B, feature_dim]
        """
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


# ================================================================
#  SimpleCNN (Lightweight custom architecture)
# ================================================================
class SimpleCNN(nn.Module):
    """
    Lightweight CNN alternative if MobileNet is too heavy.
    Useful for faster prototyping or resource-constrained scenarios.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()

        # --- Convolutional feature extractor ---
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # --- Automatically determine flattened size ---
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 480, 640)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        # --- Fully connected projection ---
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )

        self.feature_dim = feature_dim

        # --- Initialize weights ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB images [B, 3, H, W], values in [0, 1]
        Returns:
            features: [B, feature_dim]
        """
        # Normalize inputs (same as MobileNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ================================================================
#  Factory Function
# ================================================================
def create_visual_encoder(
    architecture: str = "mobilenet",
    feature_dim: int = 256,
    pretrained: bool = True
) -> nn.Module:
    """
    Create a visual encoder.
    Args:
        architecture: "mobilenet" or "simple"
        feature_dim: output feature dimension
        pretrained: whether to use pretrained weights (for mobilenet)
    """
    if architecture == "mobilenet":
        return DockingCNN(feature_dim=feature_dim, pretrained=pretrained)
    elif architecture == "simple":
        return SimpleCNN(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ================================================================
#  Self-test (optional)
# ================================================================
if __name__ == "__main__":
    print("Testing CNN models...")

    test_input = torch.randn(4, 3, 480, 640)

    print("\n1. MobileNetV3 encoder:")
    m = DockingCNN(feature_dim=256, pretrained=False)
    out = m(test_input)
    print(f"   Input:  {test_input.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Params: {sum(p.numel() for p in m.parameters()):,}")

    print("\n2. SimpleCNN encoder:")
    s = SimpleCNN(feature_dim=256)
    out = s(test_input)
    print(f"   Input:  {test_input.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Params: {sum(p.numel() for p in s.parameters()):,}")

    print("\n✓ All tests passed!")
