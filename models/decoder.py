"""
Decoder Module with Sub-pixel Convolution
解码器模块：使用亚像素卷积替代插值上采样

Reference: Real-time Single Image and Video Super-resolution (CVPR 2016)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsam import DSAM


class SubPixelConv(nn.Module):
    """
    Sub-pixel Convolution for upsampling
    
    Formula:
    X_up = PixelShuffle(Conv1x1(X_low))
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor

        # 1x1 conv to expand channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=1,
            bias=False
        )

        # Pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C', H*scale, W*scale)
        """
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder Block: Bilinear interpolation upsampling + Concatenation + DSAM
    解码器块：双线性插值上采样 + 跳跃连接 + DSAM
    
    Modified by @kimi: Replace Sub-pixel convolution with bilinear interpolation
    
    Formula:
    X_up = F.interpolate(X_low, mode='bilinear')  # @kimi: changed from PixelShuffle
    X_cat = [X_up; X_skip]
    X_out = DSAM(ReLU(BN(Conv3x3(X_cat))))
    """

    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()

        # @kimi: Old code - Sub-pixel convolution for upsampling
        # self.upsample = SubPixelConv(in_channels, out_channels, scale_factor)

        # @kimi: New code - Bilinear interpolation upsampling with channel adjustment
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

        # 3x3 conv for feature refinement after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # DSAM module
        self.dsam = DSAM(out_channels)

        self.scale_factor = scale_factor

    def forward(self, x, skip):
        """
        Args:
            x: Low-resolution feature (B, C, H, W)
            skip: Skip connection feature (B, C_skip, H*scale, W*scale)
        Returns:
            Output tensor (B, C_out, H*scale, W*scale)
        """
        # @kimi: Old code - Sub-pixel convolution upsampling
        # x = self.upsample(x)

        # @kimi: New code - Bilinear interpolation upsampling
        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Feature refinement
        x = self.conv(x)

        # DSAM
        x = self.dsam(x)

        return x


class SegmentationHead(nn.Module):
    """
    Segmentation Head: Final prediction layer
    
    Modified by @kimi: Replace sub-pixel convolution with bilinear interpolation
    Uses bilinear interpolation to upsample to original resolution.
    优化：修复单通道 BN 导致的灰度问题，并引入二阶段上采样融合 1/2 尺度特征
    """

    def __init__(self, in_channels, skip_channels, out_channels=1):
        super().__init__()

        # self.scale_factor = scale_factor
        # 第一阶段：将 1/4 尺度的特征 (128x128) 上采样到 1/2 尺度 (256x256)
        self.up_to_half = nn.Sequential(
            nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )

        # 新增：用于融合跳跃连接特征的 3x3 卷积
        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )

        # 融合 1/2 尺度的跳跃连接特征
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )

        # @kimi: Old code - Sub-pixel convolution
        # self.upsample = SubPixelConv(in_channels, out_channels, scale_factor)

        # @kimi: New code - Bilinear interpolation upsampling with channel adjustment
        # self.upsample = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        # 第二阶段：输出层，绝对不能用 BN 和 ReLU！直接用带 bias 的卷积
        self.upsample = nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=True)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_feat=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, 1, H*scale, W*scale)
        """
        # x: (B, C, 128, 128) -> up_x: (B, skip_channels, 256, 256)
        x = self.up_to_half(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # 融合 Stem 传来的 1/2 尺度特征
        if skip_feat is not None:
            x = torch.cat([x, skip_feat], dim=1)
            x = self.fuse_conv(x)

        # 最后的输出卷积
        x = self.upsample(x)

        # 最终上采样到原图尺度 (512x512)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.sigmoid(x)
        return x


class BoundaryHead(nn.Module):
    """
    Boundary Head: Auxiliary boundary prediction
    
    Predicts crack boundaries at 1/4 resolution.
    """

    def __init__(self, in_channels, out_channels=1):
        super(BoundaryHead, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, 1, H, W)
        """
        return self.conv(x)


class AuxiliaryHead(nn.Module):
    """
    Auxiliary Head: Deep supervision at stage 4
    
    Provides additional supervision signal to accelerate gradient backpropagation.
    """

    def __init__(self, in_channels, out_channels=1):
        super(AuxiliaryHead, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, 1, H, W)
        """
        return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder: Multi-stage decoder with skip connections
    
    Structure:
    - Decoder 4: F4 + F3 -> 256 x 16 x 16
    - Decoder 3: + F2 -> 128 x 32 x 32
    - Decoder 2: + F1 -> 64 x 64 x 64
    - Decoder 1: + Stem -> 32 x 128 x 128 (new stage for 64x total downsampling)
    - Seg Head: -> 1 x 512 x 512
    """

    def __init__(self, encoder_channels=[64, 128, 256, 512], decoder_channels=[256, 128, 64, 32]):
        super().__init__()

        # Decoder blocks
        self.decoder4 = DecoderBlock(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=decoder_channels[0]
        )

        self.decoder3 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[1],
            out_channels=decoder_channels[1]
        )

        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[2]
        )

        # Decoder 1: additional stage to handle 64x downsampling
        self.decoder1 = DecoderBlock(
            in_channels=decoder_channels[2],
            skip_channels=decoder_channels[2],  # Will be replaced by stem features
            out_channels=decoder_channels[3]
        )

        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[3],
            skip_channels=32,
            out_channels=1,
        )

        # Boundary head (at 1/4 resolution = 128x128)
        self.boundary_head = BoundaryHead(decoder_channels[3])

        # Auxiliary head (at stage 4)
        self.aux_head = AuxiliaryHead(encoder_channels[3])

    def forward(self, features, stem_features=None, shallow_cnn_feat=None):
        """
        Args:
            features: List of encoder features [F1, F2, F3, F4]
                F1: (B, 64, 64, 64)
                F2: (B, 128, 32, 32)
                F3: (B, 256, 16, 16)
                F4: (B, 512, 8, 8)
            stem_features: Stem output (B, 64, 128, 128), used as skip for decoder1
        Returns:
            main_out: Main segmentation output (B, 1, 512, 512)
            boundary_out: Boundary prediction (B, 1, 128, 128)
            aux_out: Auxiliary prediction (B, 1, 8, 8)
        """
        F1, F2, F3, F4 = features

        # Auxiliary output at stage 4
        aux_out = self.aux_head(F4)

        # Decoder 4: F4 + F3 -> 256 x 16 x 16
        d4 = self.decoder4(F4, F3)

        # Decoder 3: d4 + F2 -> 128 x 32 x 32
        d3 = self.decoder3(d4, F2)

        # Decoder 2: d3 + F1 -> 64 x 64 x 64
        d2 = self.decoder2(d3, F1)

        # Decoder 1: d2 + stem_features -> 32 x 128 x 128
        if stem_features is not None:
            d1 = self.decoder1(d2, stem_features)
        else:
            # If no stem features provided, use d2 as skip (will still work but less accurate)
            d1 = self.decoder1(d2, d2)

        # Boundary prediction at 1/4 resolution (128x128)
        boundary_out = self.boundary_head(d1)
        # 修改：将浅层 CNN 特征传入 seg_head
        # Main segmentation output: 32 x 128 x 128 -> 1 x 512 x 512
        main_out = self.seg_head(d1, shallow_cnn_feat)

        # main_out = self.seg_head(d1)

        return main_out, boundary_out, aux_out
