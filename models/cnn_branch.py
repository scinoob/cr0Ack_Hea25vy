"""
CNN Branch: Deep Separable Convolution for Local Detail Extraction
CNN分支：深度可分离卷积，负责局部细节提取
"""
import torch.nn as nn
from .dsam import DSAM


class CNNBlock(nn.Module):
    """
    CNN Block: Depthwise Separable Convolution with Dilation
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        # 动态计算 padding 以保持空间维度
        padding = dilation if stride == 1 else 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Point-wise conv (1x1) for dimension reduction
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Depth-wise conv (3x3)
        # 引入空洞卷积扩大感受野
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=False
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
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
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


# 定义一个辅助函数，快速生成 GroupNorm
def get_norm(channels, num_groups=8):
    # 确保 channels 能被 num_groups 整除
    groups = num_groups if channels % num_groups == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class StandardCNNBlock(nn.Module):
    """
    【改造核心】：抛弃深度可分离卷积，使用标准的高感受野残差块 (Standard ResBlock)
    真正赋予 CNN 分支捕捉极微弱跨通道梯度的能力。
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        # 动态计算 padding，确保维度对齐
        padding = dilation if stride == 1 else 1

        # 1. 密集的 3x3 标准卷积，替代 DWConv
        # 引入 padding_mode='reflect' 彻底消除边界伪影，强迫网络关注内部
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=padding, dilation=dilation,
                               padding_mode='reflect', bias=False)
        # BatchNorm 替换为 GroupNorm
        self.gn1 = get_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. 第二层密集的 3x3 标准卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation, dilation=dilation,
                               padding_mode='reflect', bias=False)

        self.gn2 = get_norm(out_channels)

        # 残差捷径 (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)
        return out


class CNNStage(nn.Module):
    """
    CNN Stage: Contains multiple CNN Blocks followed by DSAM
    """

    def __init__(self, in_channels, out_channels, num_blocks=2, stride=1, dilation=1):
        super().__init__()

        layers = []

        # First block with stride
        # 替换为标准卷积块
        layers.append(StandardCNNBlock(in_channels, out_channels, stride=stride,
                                       dilation=1 if stride > 1 else dilation))

        # Remaining blocks with stride=1
        for _ in range(1, num_blocks):
            # layers.append(CNNBlock(out_channels, out_channels, stride=1, dilation=dilation))
            # 使用标准卷积而非深度可分离卷积
            layers.append(StandardCNNBlock(out_channels, out_channels, stride=1, dilation=dilation))

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

    def __init__(self, in_channels=64, channels=[64, 128, 256],
                 num_blocks=2, dilations=[1, 1, 2]):
        super().__init__()

        self.stages = nn.ModuleList()
        self.high_res_injector = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),

            get_norm(in_channels), # 更换gn
            nn.ReLU(inplace=True)
        )

        current_channels = in_channels
        for i, out_channels in enumerate(channels):
            # All stages use stride=2 to match MiT branch spatial dimensions
            self.stages.append(
                CNNStage(current_channels, out_channels, num_blocks=num_blocks,
                         stride=2, dilation=dilations[i])
            )
            current_channels = out_channels

    def forward(self, x_half, x_stem):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            List of features from each stage
        """
        high_res_feat = self.high_res_injector(x_half)
        x = x_stem + high_res_feat
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
