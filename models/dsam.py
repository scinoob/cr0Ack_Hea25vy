"""
Directional Strip Attention Module (DSAM)
方向感知条带注意力模块

Reference: Strip Pooling (SPNet, NeurIPS 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mit_branch import ECAWithLN


class StripPooling(nn.Module):
    """
    Strip Pooling: Performs pooling along horizontal and vertical directions.
    """

    def __init__(self, channels):
        super().__init__()
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


class DSAM_conv(nn.Module):
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
        super().__init__()

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
            nn.GELU()
        )

        # Learnable weights for fusion
        self.weight_h = nn.Parameter(torch.ones(1))
        self.weight_v = nn.Parameter(torch.ones(1))

        # 新增弥补措施：LayerScale 缩放因子
        # 初始化为极小值 1e-4，防止未受 Sigmoid 限制的权重在训练初期导致特征值爆炸
        self.layer_scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-4)

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
        # X_out = x + x * A_dsam
        # 修改：引入 LayerScale 平滑注意力带来的剧烈数值波动
        X_out = x + self.layer_scale * (x * A_dsam)
        return X_out


class DSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.strip_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.strip_pool_v = nn.AdaptiveAvgPool2d((1, None))
        self.conv_fuse = nn.Conv2d(channels, channels, kernel_size=1)

        # 🔧 修改点：添加ECA增强通道信息交互
        self.eca = ECAWithLN(channels)  # ✅ 新增

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 条带池化
        s_h = self.strip_pool_h(x)
        s_v = self.strip_pool_v(x)

        # 扩展回原尺寸
        e_h = s_h.expand(-1, -1, -1, W)
        e_v = s_v.expand(-1, -1, H, -1)

        # 融合
        f_fuse = e_h + e_v
        z = self.conv_fuse(f_fuse)

        # 🔧 修改点：ECA增强后再sigmoid
        z = self.eca(z)  # ✅ 通道注意力增强
        a_dsam = self.sigmoid(z)

        # 特征重标定
        return x + x * a_dsam
