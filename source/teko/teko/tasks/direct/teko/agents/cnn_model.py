"""
CNN Feature Extractor for TEKO Docking
Uses MobileNetV3 for efficient visual processing
"""

import torch
import torch.nn as nn
import torchvision.models as models


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
        
        # Load pretrained MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Extract feature layers (remove classifier)
        self.features = mobilenet.features  # Convolutional feature extractor
        self.avgpool = mobilenet.avgpool     # Global average pooling
        
        # Get output dimension from MobileNetV3
        # MobileNetV3-Small outputs 576 features
        mobilenet_out_dim = 576
        
        # Custom head for feature projection
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
            x: RGB images, shape (batch, 3, H, W), values in [0, 1]
        
        Returns:
            features: shape (batch, feature_dim)
        """
        # Normalize images (ImageNet stats)
        # MobileNet expects images normalized with these stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        # Extract convolutional features
        x = self.features(x)  # (batch, 576, H', W')
        
        # Global average pooling
        x = self.avgpool(x)   # (batch, 576, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 576)
        
        # Project to desired feature dimension
        features = self.projection(x)  # (batch, feature_dim)
        
        return features


class SimpleCNN(nn.Module):
    """
    Lightweight CNN alternative if MobileNet is too heavy.
    Useful for faster prototyping or resource-constrained scenarios.
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size for 640x480 input:
        # After Conv1 (k=8, s=4, p=2): (480-8+2*2)/4+1 = 119.5 -> 119, (640-8+2*2)/4+1 = 159.5 -> 159
        # After MaxPool1 (k=2, s=2): 59, 79
        # After Conv2 (k=4, s=2, p=1): (59-4+2*1)/2+1 = 29, (79-4+2*1)/2+1 = 39
        # After MaxPool2 (k=2, s=2): 14, 19
        # After Conv3 (k=3, s=1, p=1): 14, 19 (same)
        # After MaxPool3 (k=2, s=2): 7, 9
        # Total: 128 * 7 * 9 = 8064
        flattened_size = 128 * 7 * 9
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB images, shape (batch, 3, H, W), values in [0, 1]
        Returns:
            features: shape (batch, feature_dim)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        return features


# Factory function to easily switch between architectures
def create_visual_encoder(
    architecture: str = "mobilenet",
    feature_dim: int = 256,
    pretrained: bool = True
) -> nn.Module:
    """
    Create visual encoder.
    
    Args:
        architecture: "mobilenet" or "simple"
        feature_dim: Output feature dimension
        pretrained: Use pretrained weights (only for mobilenet)
    
    Returns:
        Visual encoder module
    """
    if architecture == "mobilenet":
        return DockingCNN(feature_dim=feature_dim, pretrained=pretrained)
    elif architecture == "simple":
        return SimpleCNN(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test the models
    print("Testing CNN models...")
    
    # Test input (4 envs, RGB 640x480)
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 480, 640)
    
    # Test MobileNetV3
    print("\n1. Testing MobileNetV3-based encoder:")
    mobilenet_encoder = DockingCNN(feature_dim=256, pretrained=False)
    mobilenet_output = mobilenet_encoder(test_input)
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {mobilenet_output.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in mobilenet_encoder.parameters()):,}")
    
    # Test SimpleCNN
    print("\n2. Testing Simple CNN encoder:")
    simple_encoder = SimpleCNN(feature_dim=256)
    simple_output = simple_encoder(test_input)
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {simple_output.shape}")
    print(f"   Parameters:   {sum(p.numel() for p in simple_encoder.parameters()):,}")
    
    print("\n✓ All tests passed!")