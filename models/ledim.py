"""
Local Entropy Guided Dynamic Interaction Module (LEDIM)
局部熵引导动态交互模块

Dynamically fuses CNN and MiT features based on local complexity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dsam import DSAM


class LocalVariance(nn.Module):
    """
    Calculate local variance as a measure of local complexity.
    Uses two kxk mean filters to approximate variance: E[X^2] - (E[X])^2
    """

    def __init__(self, kernel_size=7):
        super(LocalVariance, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Use average pooling as mean filter
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Local variance map (B, 1, H, W)
        """
        # Convert to single channel by averaging across channels
        x_gray = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # E[X]
        mean_x = self.avg_pool(x_gray)

        # E[X^2]
        mean_x2 = self.avg_pool(x_gray ** 2)

        # Var[X] = E[X^2] - (E[X])^2
        variance = mean_x2 - mean_x ** 2

        return variance


class LEDIM(nn.Module):
    """
    Local Entropy Guided Dynamic Interaction Module (LEDIM)
    
    Dynamically fuses CNN and MiT branch features based on local texture complexity.
    
    Formula:
    1. Complexity calculation: M = LocalVar(X_cnn)
    2. Dynamic weight generation: W_dyn = sigma(Conv3x3(M))
    3. Adaptive fusion: X_fuse = W_dyn * X_cnn + (1 - W_dyn) * X_mit
    4. Feature refinement: X_out = DSAM(X_fuse)
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()

        self.channels = channels

        # Local variance calculation
        self.local_var = LocalVariance(kernel_size=kernel_size)

        # Dynamic weight generation
        self.conv_weight = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # DSAM for feature refinement
        self.dsam = DSAM(channels)

        # 新增：可学习的门控参数，初始值设为 1.2，给予 CNN 特征初始的放大倾向
        self.cnn_gate = nn.Parameter(torch.tensor(1.2))

    def forward(self, x_cnn, x_mit):
        """
        Args:
            x_cnn: CNN branch feature (B, C, H, W)
            x_mit: MiT branch feature (B, C, H, W)
        Returns:
            Fused output feature (B, C, H, W)
        """
        # Step 1: Calculate local variance from CNN features
        M = self.local_var(x_cnn)  # (B, 1, H, W)

        # Step 2: Generate dynamic weight
        W_dyn = self.conv_weight(M)  # (B, 1, H, W)

        # Step 3: Adaptive fusion
        # X_fuse = W_dyn * X_cnn + (1 - W_dyn) * X_mit
        X_fuse = W_dyn * (x_cnn * self.cnn_gate) + (1 - W_dyn) * x_mit

        # Step 4: Feature refinement with DSAM
        X_out = self.dsam(X_fuse)

        return X_out
