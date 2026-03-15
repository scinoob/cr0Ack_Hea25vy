"""
Dual-Branch Crack Segmentation Network
基于方向感知与动态融合的双分支路面裂缝分割网络

This package implements a dual-branch crack segmentation network with:
- MiT Branch: Mix Transformer for global semantic modeling
- CNN Branch: Deep separable convolution for local detail extraction
- DSAM: Directional Strip Attention Module
- LEDIM: Local Entropy Guided Dynamic Interaction Module
- Sub-pixel convolution decoder
"""

__version__ = '1.0.0'
__author__ = 'Crack Segmentation Team'
