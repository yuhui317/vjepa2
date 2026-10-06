# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """
    【2D 图像分块与嵌入 (Image to Patch Embedding)】

    原理：
    把输入的 2D 图片切分成非重叠的小方块 (Patches)，并将每个 Patch 线性投影映射为 D 维向量。
    代码实现技巧：使用 kernel_size=patch_size, stride=patch_size 的 2D 卷积一步完成切块与线性投影。

    维度变换流程：
    输入:  [B, C, H, W]  (Batch, 通道数3, 高度, 宽度)
    卷积:  [B, D, H/P, W/P] (P 为 patch_size, D 为 embed_dim)
    展平:  [B, D, N_patches] (其中 N_patches = (H/P) * (W/P))
    转置:  [B, N_patches, D] (输出符合 Transformer 输入格式的 Token 序列)
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 卷积核大小与步长均为 patch_size，相当于对图像无重叠网格切分，并线性映射到 embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # self.proj(x) -> [B, D, H/P, W/P]
        # .flatten(2)  -> [B, D, (H/P * W/P)]
        # .transpose(1, 2) -> [B, N_patches, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    【3D 视频时空管道分块与嵌入 (Spatio-Temporal Tubelet Embedding)】

    这是 V-JEPA 2 处理视频的核心第一步！
    与处理静态图像的 ViT 不同，视频具有时间维度 T。
    V-JEPA 2 将连续多帧（例如 tubelet_size=2 帧）与空间方块（例如 patch_size=16x16 像素）
    打包为一个“3D 时空管道块 (Spatiotemporal Tubelet)”。

    维度变换流程：
    输入:  [B, C, T, H, W]  (Batch, 通道3, 帧数T, 高H, 宽W)
    3D卷积: [B, D, T/tubelet_size, H/patch_size, W/patch_size]
    展平:  [B, D, N_patches] (其中 N_patches = (T/tubelet) * (H/patch) * (W/patch))
    转置:  [B, N_patches, D] (输出总共 N_patches 个时空隐向量)
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # 3D 卷积：时间步长为 tubelet_size，空间步长为 patch_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # 1. 3D 卷积映射: [B, C, T, H, W] -> [B, D, T', H', W']
        # 2. flatten(2): 将 (T', H', W') 展平为时空 Patch 总数 N
        # 3. transpose(1, 2): 调换通道与 Token 维度，得到 [B, N_patches, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
