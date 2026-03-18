# ================= models/decoder.py =================
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_branch import get_norm
from .dsam import DSAM
from .mit_branch import ECAChannelAttention


class ASPP(nn.Module):
    """空洞空间金字塔池化，捕获多尺度低频上下文"""

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super().__init__()
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                         nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(True))

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.global_pool(x), size=x.size()[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat((x1, x2, x3, x4, x5), dim=1))


class DecoderBlock(nn.Module):
    """
    Decoder Block: Bilinear interpolation upsampling + Concatenation + DSAM
    """

    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dsam = DSAM(out_channels)
        self.scale_factor = scale_factor

    def forward(self, x, skip):
        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        if skip is not None:
            # 安全检查：确保空间维度对齐
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        x = self.dsam(x)
        return x


class SegmentationHead(nn.Module):
    """
    Segmentation Head: Final prediction layer with shallow feature fusion
    """

    def __init__(self, in_channels, skip_channels, out_channels=1, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

        # 这里严格保证 Conv2d 和 BatchNorm2d 使用相同的 in_channels (32)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=3, padding=1, bias=False),
            get_norm(in_channels),
            nn.ReLU(inplace=True)
        )

        # 在最后输出前，加入 ECA 通道去噪
        self.eca_channel_att = ECAChannelAttention(in_channels)

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_feat=None):
        if skip_feat is not None:
            if skip_feat.shape[2:] != x.shape[2:]:
                skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
            x = self.fuse_conv(x)

        # 让eca压制无用的噪声通道
        x = self.eca_channel_att(x)

        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.sigmoid(x)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(BoundaryHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            get_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(AuxiliaryHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            get_norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """
    3-Stage Decoder: Multi-stage decoder with skip connections
    """

    def __init__(self, encoder_channels=[64, 128, 256], decoder_channels=[128, 64, 32]):
        super().__init__()

        # 【新增】：在 F3 接入处增加 ASPP，处理最深层的多尺度上下文
        # self.aspp = ASPP(encoder_channels[2], encoder_channels[2])

        # d3: F3(256) + F2(128) -> 128
        self.decoder3 = DecoderBlock(
            in_channels=encoder_channels[2],
            skip_channels=encoder_channels[1],
            out_channels=decoder_channels[0]
        )

        # d2: d3(128) + F1(64) -> 64
        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[1]
        )

        # d1: d2(64) + Stem(64) -> 32
        self.decoder1 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[2]
        )

        # SegHead: d1(32) + shallow_cnn(64) -> 1
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[2],
            skip_channels=encoder_channels[0],
            out_channels=1,
            scale_factor=4
        )

        self.boundary_head = BoundaryHead(decoder_channels[2])
        self.aux_head = AuxiliaryHead(encoder_channels[2])

    def forward(self,
                features,
                stem_features=None,
                shallow_cnn_feat=None,
                deep_cnn_feat=None,
                ):
        # 接收的是 3 个阶段的特征
        F1, F2, F3 = features

        # 对最深层特征 (F3) 进行 ASPP 多尺度上下文增强
        # F3 = self.aspp(F3)

        # 如果传入了深层 CNN 特征，辅损 (Aux Loss) 就直接算在 CNN 头上
        # 逼迫 CNN 必须独立学出裂缝，打通 CNN_stage3 的死水梯度！
        if deep_cnn_feat is not None:
            aux_out = self.aux_head(deep_cnn_feat)
        else:
            aux_out = self.aux_head(F3)
        # aux_out = self.aux_head(F3)

        d3 = self.decoder3(F3, F2)
        d2 = self.decoder2(d3, F1)

        if stem_features is not None:
            d1 = self.decoder1(d2, stem_features)
        else:
            d1 = self.decoder1(d2, d2)

        boundary_out = self.boundary_head(d1)
        main_out = self.seg_head(d1, shallow_cnn_feat)

        return main_out, boundary_out, aux_out
