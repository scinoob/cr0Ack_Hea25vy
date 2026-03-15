"""
Directional Strip Attention Module (DSAM)
方向感知条带注意力模块

Reference: Strip Pooling (SPNet, NeurIPS 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):
    """
    Strip Pooling: Performs pooling along horizontal and vertical directions.
    """
    
    def __init__(self, channels):
        super(StripPooling, self).__init__()
        self.channels = channels
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            S_h: Horizontal strip pooling (B, C, H, 1)
            S_v: Vertical strip pooling (B, C, 1, W)
        """
        B, C, H, W = x.shape
        
        # Horizontal strip pooling: pool along W dimension
        # S_h shape: (B, C, H, 1)
        S_h = torch.mean(x, dim=3, keepdim=True)
        
        # Vertical strip pooling: pool along H dimension
        # S_v shape: (B, C, 1, W)
        S_v = torch.mean(x, dim=2, keepdim=True)
        
        return S_h, S_v


class DSAM(nn.Module):
    """
    Directional Strip Attention Module (DSAM)
    
    Captures long-range dependencies along horizontal and vertical directions,
    which is effective for elongated crack structures.
    
    Formula:
    1. Strip pooling: S_h = (1/W) * sum(X, dim=W), S_v = (1/H) * sum(X, dim=H)
    2. Feature expansion and fusion
    3. Generate attention weights: A_dsam = sigma(Conv(F_fuse))
    4. Feature recalibration: X_out = X + X * A_dsam
    """
    
    def __init__(self, channels, reduction=16):
        super(DSAM, self).__init__()
        
        self.channels = channels
        
        # Strip pooling
        self.strip_pool = StripPooling(channels)
        
        # 1x1 conv for transformation
        self.conv_transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv for attention generation
        self.conv_attention = nn.Sequential(
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            # 调整避免梯度损失
            # nn.Sigmoid()
        )
        
        # Learnable weights for fusion
        self.weight_h = nn.Parameter(torch.ones(1))
        self.weight_v = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Step 1: Strip pooling
        S_h, S_v = self.strip_pool(x)  # (B, C, H, 1), (B, C, 1, W)
        
        # Step 2: Expand and add
        # Expand S_h to (B, C, H, W)
        E_h = S_h.expand(B, C, H, W)
        # Expand S_v to (B, C, H, W)
        E_v = S_v.expand(B, C, H, W)
        
        # Fusion with learnable weights
        F_fuse = self.weight_h * E_h + self.weight_v * E_v
        
        # Step 3: Transform and generate attention
        Z = self.conv_transform(F_fuse)
        A_dsam = self.conv_attention(Z)
        
        # Step 4: Feature recalibration with residual connection
        X_out = x + x * A_dsam
        
        return X_out
