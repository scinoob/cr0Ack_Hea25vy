"""
Directional Strip Attention Module (DSAM)
方向感知条带注意力模块

Reference: Strip Pooling (SPNet, NeurIPS 2020)
"""
import torch
import torch.nn as nn

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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 捕捉空间上的最大响应和平均响应
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后计算空间权重
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return x * self.sigmoid(scale)


class CoordAtt(nn.Module):
    """
    坐标注意力 (Coordinate Attention)
    精准捕获裂缝的水平或垂直长距离连续方向特征
    """

    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        # 分别在水平和垂直方向做池化，保留方向和位置信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # 生成水平和垂直方向的注意力权重
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # [B, C, H, 1] 和 [B, C, 1, W]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 转置以便拼接

        # 沿空间维度拼接后进行特征变换
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离回水平和垂直方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 激活为 0~1 的权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 权重乘回原图，实现并行的坐标方向交叉注意力
        out = identity * a_w * a_h
        return out


# 原先的设计意图，通过双方向条形池化，计算权重，引导关注
class oldDSAM(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.strip_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.strip_pool_v = nn.AdaptiveAvgPool2d((1, None))
        self.conv_fuse = nn.Conv2d(channels, channels, kernel_size=1)

        # 🔧 修改点：添加ECA增强通道信息交互
        self.eca = ECAWithLN(channels)

        # self.spatial_attention = SpatialAttention(kernel_size=7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.shape

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
        # z = self.spatial_attention(z)
        a_dsam = self.sigmoid(z)

        # 特征重标定
        return x + x * a_dsam


class DSAM(nn.Module):
    """
    使用 CoordAtt 替换原本简单的通道/空间堆叠
    """

    def __init__(self, channels):
        super().__init__()
        # 局部细节卷积
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # 坐标注意力捕获长程连续性
        self.coord_att = CoordAtt(channels)

    def forward(self, x):
        out = self.local_conv(x)
        out = self.coord_att(out)
        return out + x  # 残差融合
