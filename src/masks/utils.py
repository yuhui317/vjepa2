# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x, masks, concat=True):
    """
    【应用掩码提取指定位置的 Patches】

    功能：
    根据掩码索引列表 (masks)，从全部 Token 特征 x 中选出保留的 Token。
    这是 JEPA 和 MAE 实现高计算效率的关键：
    - 对于上下文编码器 (Context Encoder)，只输入可见的非遮挡 patches；
    - 对于目标特征 (Target)，只提取被遮挡区域对应的特征用于计算损失。

    参数说明:
    :param x: 形状为 [B, N, D] 的张量 (Batch_size, 总Token数, 特征维度)
    :param masks: 形状为 [B, K] 的张量列表，每个张量包含 K 个需要选取的 Token 索引 (0 <= index < N)
    :param concat: 是否在 Batch 维度拼接多个掩码对应的输出，默认为 True

    返回:
    - 如果 concat=True，返回 [len(masks) * B, K, D]
    - 如果 concat=False，返回包含 len(masks) 个 [B, K, D] 张量的列表
    """
    all_x = []
    for m in masks:
        # m 的形状是 [B, K]，将其扩展并复制到特征维度 D，形状变为 [B, K, D]
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        # torch.gather: 沿着维度 1 (Token 序列维度) 按照 mask_keep 收集特征向量
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    # 沿 Batch 维度拼接所有掩码采样结果
    return torch.cat(all_x, dim=0)
