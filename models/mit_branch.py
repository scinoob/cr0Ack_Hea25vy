"""
MiT Branch: Mix Transformer for Global Semantic Modeling
MiT分支：基于SegFormer的Mix Transformer，负责全局语义建模

Reference: SegFormer (NeurIPS 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialReductionAttention(nn.Module):
    """
    Spatial Reduction Attention (SRA)
    Reduces the spatial dimensions of K and V to reduce computational cost.
    
    Formula:
    Q = X * W_q
    K = Reshape(X * W_k, ratio=r)
    V = Reshape(X * W_v, ratio=r)
    Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d)) * V
    """

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # Spatial reduction for K and V
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor (B, N, C) where N = H * W
            H, W: Spatial dimensions
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape

        # Q projection
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # K, V projection with spatial reduction
        if self.sr_ratio > 1:
            # Reshape to (B, C, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # Spatial reduction
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            k = self.k(x_).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class MixFFN(nn.Module):
    """
    Mix-FFN: Feed-forward network with depth-wise convolution
    
    Formula:
    X_mid = GELU(Conv1x1(X))
    X_out = Conv1x1(DepthwiseConv3x3(X_mid))
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)

        # Depth-wise convolution
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3,
            stride=1, padding=1, groups=hidden_features, bias=False
        )
        self.bn = nn.BatchNorm2d(hidden_features)

        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor (B, N, C) where N = H * W
            H, W: Spatial dimensions
        Returns:
            Output tensor (B, N, C)
        """
        x = self.fc1(x)
        B, N, C = x.shape

        # Reshape for depth-wise conv
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class SEChannelAttention(nn.Module):
    """
    Channel Attention (SE-style)
    
    Formula:
    A_c = sigma(MLP(GAP(X)))
    X_c = X * A_c
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, N, C)
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape

        # Global average pooling
        y = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        # MLP
        y = self.mlp(y)  # (B, 1, C)

        # Channel attention
        x = x * y

        return x


class CBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # 两层 1*1卷积代替全连接
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return F.sigmoid(out)


class ECAChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA)
    论文: Wang, Q., et al. "ECA-Net: Efficient Channel Attention for Deep CNNs." CVPR 2020

    核心优势:
    - 无降维操作，避免信息损失
    - 自适应卷积核大小，捕捉局部跨通道交互
    - 梯度传播路径短，缓解梯度消失

    适用场景: 替换SE、CBAM中的通道注意力模块
    """

    def __init__(self, channels, gamma=2, b=1):
        """
        Args:
            channels: 输入通道数
            gamma: 核大小计算参数（默认2）
            b: 核大小计算偏置（默认1）
        """
        super().__init__()

        # 自适应计算1D卷积核大小
        # 通道数越多，感受野越大
        t = int(abs((math.log2(channels) + b) / gamma))
        self.kernel_size = t if t % 2 else t + 1
        padding = self.kernel_size // 2

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1D卷积实现跨通道交互（无降维！）
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W) 或 (B, N, C)
        Returns:
            加权后的特征图
        """
        # 处理4D输入 (B, C, H, W) - CNN/DSAM/LEDIM场景
        if x.dim() == 4:
            B, C, H, W = x.shape

            # 全局平均池化: (B, C, 1, 1)
            y = self.avg_pool(x)

            # 变换为1D序列: (B, C, 1)
            y = y.squeeze(-1).transpose(-1, -2)

            # 1D卷积: (B, C, 1)
            y = self.conv(y)

            # 恢复形状: (B, C, 1, 1)
            y = y.transpose(-1, -2).unsqueeze(-1)

            # 通道权重: (B, C, 1, 1)
            attention = self.sigmoid(y)

            # 特征重标定
            return x * attention

        # 处理3D输入 (B, N, C) - MiT/Transformer场景
        elif x.dim() == 3:
            B, N, C = x.shape

            # 全局平均池化: (B, 1, C)
            y = x.mean(dim=1, keepdim=True)

            # 变换为1D序列: (B, C, 1)
            y = y.transpose(-1, -2)

            # 1D卷积: (B, C, 1)
            y = self.conv(y)

            # 恢复形状: (B, 1, C)
            y = y.transpose(-1, -2)

            # 通道权重: (B, 1, C)
            attention = self.sigmoid(y)

            # 特征重标定
            return x * attention

        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


# ============ 可选：带LayerNorm的ECA变体（梯度更稳定）============
class ECAWithLN(nn.Module):
    """ECA + LayerNorm 变体，适合深层网络"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        self.kernel_size = t if t % 2 else t + 1
        padding = self.kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size,
                              padding=padding, bias=False)
        self.ln = nn.LayerNorm(channels)  # 添加LayerNorm
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            # @kimi fix: 全局平均池化并调整维度为 (B, 1, C) 以匹配 Conv1d
            y = self.avg_pool(x).view(B, 1, C)  # (B, 1, C)
            # @kimi fix: 1D卷积: (B, 1, C) -> (B, 1, C)
            y = self.conv(y)  # (B, 1, C)
            # @kimi fix: LayerNorm 期望 (B, *, C)，输入已经是 (B, 1, C)
            y = self.ln(y)
            # @kimi fix: 恢复为 (B, C, 1, 1) 用于广播
            attention = self.sigmoid(y).view(B, C, 1, 1)
            return x * attention
        elif x.dim() == 3:
            B, N, C = x.shape
            # @kimi fix: 全局平均池化得到 (B, 1, C)
            y = x.mean(dim=1, keepdim=True)
            # @kimi fix: reshape为 (B, 1, C) 以匹配 Conv1d 的输入要求
            y = y.view(B, 1, C)
            # @kimi fix: 1D卷积: (B, 1, C) -> (B, 1, C)
            y = self.conv(y)
            # @kimi fix: LayerNorm 期望 (B, *, C)，输入已经是 (B, 1, C)
            y = self.ln(y)
            attention = self.sigmoid(y)
            return x * attention
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


class MiTBlock(nn.Module):
    """
    MiT Block: Mix Transformer Block
    
    Structure:
    - LayerNorm
    - Efficient Self-Attention (SRA)
    - Residual connection
    - LayerNorm
    - Mix-FFN
    - Residual connection
    - Channel Attention
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1, drop=0., drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Channel attention
        # self.channel_attn = ChannelAttention(dim)
        self.eca_channel_attn = ECAWithLN(dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input tensor (B, N, C) where N = H * W
            H, W: Spatial dimensions
        Returns:
            Output tensor (B, N, C)
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), H, W)

        # Mix-FFN with residual
        x = x + self.mlp(self.norm2(x), H, W)

        # Channel attention
        # 原来的se风格attention
        # x = self.channel_attn(x)
        x = self.eca_channel_attn(x)

        return x


class MiTStage(nn.Module):
    """
    MiT Stage: Contains patch embedding and multiple MiT Blocks
    """

    def __init__(self, in_channels, out_channels, depth, num_heads, mlp_ratio, sr_ratio, drop=0., drop_path=0.):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels)

        # MiT Blocks
        self.blocks = nn.ModuleList([
            MiTBlock(
                dim=out_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                sr_ratio=sr_ratio,
                drop=drop,
                drop_path=drop_path
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C', H', W')
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, C', H/2, W/2)
        B, C, H, W = x.shape

        # Reshape for transformer
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C')
        x = self.norm(x)

        # Apply blocks
        for block in self.blocks:
            x = block(x, H, W)

        # Reshape back
        x = x.transpose(1, 2).view(B, C, H, W)

        return x


class MiTBranch(nn.Module):
    """
    MiT Branch: Multi-stage Mix Transformer encoder
    """

    def __init__(self, in_channels=64, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1], depths=[2, 2, 2, 2],
                 drop_rate=0., drop_path_rate=0.1):
        super(MiTBranch, self).__init__()

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        current_channels = in_channels

        cur = 0
        for i, (embed_dim, num_head, mlp_ratio, sr_ratio, depth) in enumerate(
                zip(embed_dims, num_heads, mlp_ratios, sr_ratios, depths)
        ):
            self.stages.append(
                MiTStage(
                    in_channels=current_channels,
                    out_channels=embed_dim,
                    depth=depth,
                    num_heads=num_head,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur:cur + depth]
                )
            )
            current_channels = embed_dim
            cur += depth

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
