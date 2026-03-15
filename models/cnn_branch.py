"""
CNN Branch: Deep Separable Convolution for Local Detail Extraction
CNN分支：深度可分离卷积，负责局部细节提取
"""
import torch
import torch.nn as nn
from .dsam import DSAM


class CNNBlock(nn.Module):
    """
    CNN Block: Depthwise Separable Convolution Residual Block
    
    Formula:
    X_mid = ReLU(BN(Conv1x1(DepthwiseConv3x3(Conv1x1(X)))))
    X_out = ReLU(X + Conv1x1(X_mid))
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNNBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Point-wise conv (1x1) for dimension reduction
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Depth-wise conv (3x3)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        
        # Point-wise conv (1x1) for dimension expansion
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Final 1x1 conv for residual
        self.conv_residual = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C', H', W')
        """
        identity = x
        
        # X_mid = ReLU(BN(Conv1x1(DepthwiseConv3x3(Conv1x1(X)))))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn_dw(self.depthwise(out)))
        out = self.bn2(self.conv2(out))
        
        # Shortcut
        identity = self.shortcut(identity)
        
        # X_out = ReLU(X + Conv1x1(X_mid))
        out = self.relu(identity + self.conv_residual(out))
        
        return out


class CNNStage(nn.Module):
    """
    CNN Stage: Contains multiple CNN Blocks followed by DSAM
    """
    
    def __init__(self, in_channels, out_channels, num_blocks=2, stride=1):
        super(CNNStage, self).__init__()
        
        layers = []
        
        # First block with stride
        layers.append(CNNBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks with stride=1
        for _ in range(1, num_blocks):
            layers.append(CNNBlock(out_channels, out_channels, stride=1))
        
        self.blocks = nn.Sequential(*layers)
        
        # DSAM module
        self.dsam = DSAM(out_channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C', H', W')
        """
        x = self.blocks(x)
        x = self.dsam(x)
        return x


class CNNBranch(nn.Module):
    """
    CNN Branch: Multi-stage feature extraction
    
    Each stage consists of CNN Blocks + DSAM
    """
    
    def __init__(self, in_channels=64, channels=[64, 128, 256, 512], num_blocks=2):
        super(CNNBranch, self).__init__()
        
        self.stages = nn.ModuleList()
        
        current_channels = in_channels
        for i, out_channels in enumerate(channels):
            stride = 2  # All stages use stride=2 to match MiT branch spatial dimensions
            self.stages.append(
                CNNStage(current_channels, out_channels, num_blocks=num_blocks, stride=stride)
            )
            current_channels = out_channels
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            List of features from each stage
        """
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
