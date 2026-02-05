"""
Feature Extractor for Drifting Loss

This module implements the Feature Extractor (ResNet-style encoder) that extracts
multi-scale features for computing the drifting loss.

The paper states the method fails on ImageNet without a feature encoder.
The loss is computed on features f(x), not pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torchvision.models as models


class ResNetBlock(nn.Module):
    """Basic ResNet block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class FeatureExtractor(nn.Module):
    """
    Multi-scale Feature Extractor for computing drifting loss.
    
    Extracts features at multiple scales (stages) of a ResNet-style encoder
    to capture different granularity of the target distribution.
    
    The features are used to compute the drifting field V(x) in feature space,
    which is more meaningful than pixel space for complex images.
    
    Args:
        in_channels: Number of input channels (default: 4 for VAE latent)
        base_channels: Base number of channels (default: 64)
        num_blocks: Number of blocks per stage (default: [2, 2, 2, 2])
        pretrained_encoder: Optional pretrained encoder to use (e.g., 'resnet18')
        freeze_pretrained: Whether to freeze pretrained weights
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_blocks: List[int] = None,
        pretrained_encoder: Optional[str] = None,
        freeze_pretrained: bool = True,
    ):
        super().__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        
        self.in_channels = in_channels
        self.pretrained_encoder = pretrained_encoder
        
        if pretrained_encoder:
            self._build_pretrained(pretrained_encoder, freeze_pretrained)
        else:
            self._build_custom(in_channels, base_channels, num_blocks)
    
    def _build_pretrained(self, encoder_name: str, freeze: bool):
        """Build using a pretrained encoder."""
        if encoder_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif encoder_name == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif encoder_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Modify first conv if input channels != 3
        if self.in_channels != 3:
            self.input_conv = nn.Conv2d(
                self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.input_conv = resnet.conv1
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.stage1 = resnet.layer1  # Output: 64 channels
        self.stage2 = resnet.layer2  # Output: 128 channels
        self.stage3 = resnet.layer3  # Output: 256 channels
        self.stage4 = resnet.layer4  # Output: 512 channels
        
        # Feature dimensions at each stage
        if encoder_name in ['resnet18', 'resnet34']:
            self.feature_dims = [64, 128, 256, 512]
        else:  # resnet50, etc.
            self.feature_dims = [256, 512, 1024, 2048]
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            # Unfreeze input conv if modified
            if self.in_channels != 3:
                for param in self.input_conv.parameters():
                    param.requires_grad = True
    
    def _build_custom(self, in_channels: int, base_channels: int, num_blocks: List[int]):
        """Build a custom ResNet-style encoder."""
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        self.stage1 = self._make_stage(base_channels, channels[0], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], num_blocks[3], stride=2)
        
        self.feature_dims = channels
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a stage with multiple blocks."""
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors at 4 scales
        """
        # Initial processing
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features at each stage
        features = []
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        x = self.stage4(x)
        features.append(x)
        
        return features
    
    def extract_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and flatten to a single vector.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Flattened features of shape (B, D)
        """
        features = self.forward(x)
        
        # Global average pooling and concatenation
        pooled = []
        for feat in features:
            pooled.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        
        return torch.cat(pooled, dim=-1)


class LatentFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor for VAE latent space.
    
    Since the latent space is already a good representation,
    we use a simpler encoder that's faster to compute.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_stages: int = 4,
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        
        in_ch = in_channels
        out_ch = hidden_channels
        
        for i in range(num_stages):
            stage = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2 if i > 0 else 1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        
        # Cap channel expansion at 2^3=8x to prevent excessive memory usage
        # Feature dims: [hidden_channels, hidden_channels*2, hidden_channels*4, hidden_channels*8]
        MAX_CHANNEL_DOUBLING = 3
        self.feature_dims = [hidden_channels * (2 ** min(i, MAX_CHANNEL_DOUBLING)) for i in range(num_stages)]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
