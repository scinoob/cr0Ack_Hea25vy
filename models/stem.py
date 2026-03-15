"""
Stem Module: Initial feature extraction
输入: (3, H, W)
输出: (64, H/4, W/4)
"""
import torch
import torch.nn as nn


class StemModule(nn.Module):
    """
    Stem module for initial feature extraction.
    Uses two stride-2 conv layers to quickly reduce spatial dimensions.
    
    Tensor shape: (3, H, W) -> (64, H/4, W/4)
    """
    
    def __init__(self, in_channels=3, out_channels=64):
        super(StemModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output tensor (B, 64, H/4, W/4)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
