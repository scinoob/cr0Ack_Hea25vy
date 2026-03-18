"""
Dual-Branch Crack Segmentation Network
基于方向感知与动态融合的双分支路面裂缝分割网络

Architecture:
- Stem Module
- MiT Branch (Global semantic modeling)
- CNN Branch (Local detail extraction)
- LEDIM (Dynamic feature fusion)
- Decoder (Sub-pixel convolution)
"""
import torch
import torch.nn as nn
from .stem import StemModule
from .mit_branch import MiTBranch
from .cnn_branch import CNNBranch
from .ledim import LEDIM
from .decoder import Decoder


class CrackSegmentationNetwork(nn.Module):
    """
    Dual-Branch Crack Segmentation Network
    
    Input: RGB image (B, 3, H, W)
    Output: Crack probability map (B, 1, H, W)
    
    Architecture:
    1. Stem: (3, H, W) -> (64, H/4, W/4)
    2. Stage 1-4: MiT + CNN + LEDIM fusion (each stage does 2x downsampling)
    3. Decoder: Multi-scale feature fusion with 4 decoder stages
    4. Output: Main (512x512) + Boundary (128x128) + Auxiliary (8x8) predictions
    """

    def __init__(self,
                 stem_in_channels=3,
                 stem_out_channels=64,
                 # 缩减为3阶段
                 mit_embed_dims=[64, 128, 256],
                 mit_num_heads=[1, 2, 4],
                 mit_mlp_ratios=[4, 4, 4],
                 mit_sr_ratios=[8, 4, 2],
                 mit_depths=[2, 2, 2],
                 mit_drop_rate=0.,
                 mit_drop_path_rate=0.1,
                 # CNN通道对应缩减，并加入空洞率配置
                 cnn_channels=[64, 128, 256],
                 # 空洞扩张率
                 # 对于 1 像素宽的极细目标，绝对不能在最浅层（Stage 1）使用空洞卷积，必须保持密集的局部感受野。
                 # 即使在深层使用，也需要结合标准卷积来填补空隙（这被称为 HDC 混合空洞卷积）
                 cnn_dilations=[1, 1, 1],
                 decoder_channels=[128, 64, 32]):
        super().__init__()

        # Stem module
        self.stem = StemModule(stem_in_channels, stem_out_channels)

        # MiT Branch
        self.mit_branch = MiTBranch(
            in_channels=stem_out_channels,
            embed_dims=mit_embed_dims,
            num_heads=mit_num_heads,
            mlp_ratios=mit_mlp_ratios,
            sr_ratios=mit_sr_ratios,
            depths=mit_depths,
            drop_rate=mit_drop_rate,
            drop_path_rate=mit_drop_path_rate
        )

        # CNN Branch
        self.cnn_branch = CNNBranch(
            in_channels=stem_out_channels,
            channels=cnn_channels,
            num_blocks=2,
            dilations=cnn_dilations  # 传入空洞率
        )

        # LEDIM fusion modules for each stage
        self.ledim_modules = nn.ModuleList([
            LEDIM(channels=mit_embed_dims[i])
            for i in range(len(mit_embed_dims))
        ])

        # Decoder
        self.decoder = Decoder(
            encoder_channels=mit_embed_dims,
            decoder_channels=decoder_channels
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            main_out: Main segmentation output (B, 1, H, W)
            boundary_out: Boundary prediction (B, 1, H/4, W/4)
            aux_out: Auxiliary prediction (B, 1, H/16, W/16)
        """
        # 兼容 stem 模块返回元组 (x_half, x_quarter) 或 单一张量 (x_quarter) 的情况
        stem_out = self.stem(x)
        if isinstance(stem_out, tuple):
            x_half, x_stem = stem_out  # 解包出 1/4 分辨率的特征给主干网络
        else:
            x_stem = stem_out  # (B, 64, H/4, W/4)
            x_half = None  # 防止出错

        # Get features from both branches
        mit_features = self.mit_branch(x_stem)  # List of [F1, F2, F3]
        cnn_features = self.cnn_branch(x_half, x_stem)  # List of [F1, F2, F3]

        # LEDIM fusion for each stage
        fused_features = []
        for i in range(len(mit_features)):
            fused = self.ledim_modules[i](cnn_features[i], mit_features[i])
            fused_features.append(fused)

        # 修改：额外传入 cnn_features[0] 作为最浅层的跳跃连接特征
        # 传入的第三个参数改为 cnn_features[0]，用 CNN 提取的最浅层高频边缘指导最后的分割预测
        main_out, boundary_out, aux_out = self.decoder(
            fused_features, x_stem,
            cnn_features[0], cnn_features[2]  # 新增第四个参数
        )

        return main_out, boundary_out, aux_out


def build_model(config):
    """
    Build the crack segmentation network from config
    
    Args:
        config: Model configuration
    Returns:
        model: CrackSegmentationNetwork instance
    """
    model = CrackSegmentationNetwork(
        stem_in_channels=config.stem_in_channels,
        stem_out_channels=config.stem_out_channels,
        mit_embed_dims=config.mit_embed_dims,
        mit_num_heads=config.mit_num_heads,
        mit_mlp_ratios=config.mit_mlp_ratios,
        mit_sr_ratios=config.mit_sr_ratios,
        mit_depths=config.mit_depths,
        mit_drop_rate=config.mit_drop_rate,
        mit_drop_path_rate=config.mit_drop_path_rate,
        cnn_channels=config.cnn_channels,
        # cnn_dilations=config.cnn_dilations, # 实际上config中没有，改为使用默认CrackSegmentation默认配置
        decoder_channels=config.decoder_channels
    )
    # 模型扩容，提高精度
    # model = CrackSegmentationNetwork(
    #     stem_in_channels=3,
    #     stem_out_channels=128,  # 【修改】：64 -> 128，极大增强初始低级特征捕捉能力

    #     mit_embed_dims=[128, 256, 512],  # 【修改】：[64, 128, 256] -> [128, 256, 512]
    #     mit_num_heads=[1, 2, 4],
    #     mit_mlp_ratios=[4, 4, 4],
    #     mit_sr_ratios=[8, 4, 2],
    #     mit_depths=[3, 4, 6],  # 【修改】：稍微增加深度 [2, 2, 2] -> [3, 4, 6]
    #     mit_drop_rate=0.,
    #     mit_drop_path_rate=0.1,

    #     cnn_channels=[128, 256, 512],  # 【修改】：[64, 128, 256] -> [128, 256, 512]
    #     # cnn_dilations=config.cnn_dilations, # 实际上config中没有，改为使用默认CrackSegmentation默认配置
    #     # 解码器同步拓宽
    #     decoder_channels=[256, 128, 64],
    # )

    # 改善梯度，使用凯明初始化
    model.apply(init_weights)
    return model


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # Test the model
    model = CrackSegmentationNetwork()
    x = torch.randn(2, 3, 512, 512)
    main_out, boundary_out, aux_out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Main output shape: {main_out.shape}")
    print(f"Boundary output shape: {boundary_out.shape}")
    print(f"Auxiliary output shape: {aux_out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
