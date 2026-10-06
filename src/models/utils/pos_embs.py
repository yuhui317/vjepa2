# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    【3D 时空正弦-余弦位置编码 (3D Spatiotemporal Sin-Cos Positional Embedding)】

    为什么需要 3D 位置编码？
    Transformer 自注意力机制天生具有“排列不变性 (Permutation Invariance)”。
    对于视频序列，模型本身分不清哪个 Patch 是前一帧、哪个是后一帧，哪个在画面左上角、哪个在右下角。
    因此必须给每一个 Tubelet Token 注入唯一的 3D 坐标 $(t, h, w)$ 编码！

    算法原理：
    1. 构建三维网格：
       - 时间深度：grid_depth = T // tubelet_size
       - 空间高宽：grid_size = H // patch_size
    2. 维度分配：
       - 默认非均匀策略 (uniform_power=False)：
         - 时间维度 (d): 分配 embed_dim // 2 的特征维度（时间运动信息给予更高比重）
         - 高度维度 (h): 分配 embed_dim // 4
         - 宽度维度 (w): 分配 embed_dim // 4
       - 均匀策略 (uniform_power=True)：
         - 时间、高度、宽度各占约 1/3 维度
    3. 分别计算 1D 正弦余弦编码后沿最后一维拼接：
       pos_embed = Concat([emb_d, emb_h, emb_w], dim=-1)

    :param embed_dim: 隐特征维度 D (例如 1024)
    :param grid_size: 空间网格边长 (例如 224/16 = 14)
    :param grid_depth: 时间网格深度 (例如 16帧/2 = 8)
    :return: [grid_depth * grid_size * grid_size, embed_dim] 的位置矩阵 (例如 [1568, 1024])
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    【2D 空间正弦-余弦位置编码 (2D Spatial Sin-Cos Positional Embedding)】
    用于单帧静态图像或空间网格编码。
    高 (H) 和宽 (W) 各平分一半的维度 (embed_dim // 2)。
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    【1D 正弦-余弦位置编码 (1D Sin-Cos Positional Embedding)】
    基础 1 维序列位置编码。
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    【基于连续频率的正弦-余弦编码数学实现 (Vaswani et al., 2017)】

    公式：
    PE(pos, 2i)   = sin(pos / (10000^(2i / D)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i / D)))

    通过不同波长（从 2π 到 10000 * 2π）的正弦与余弦函数交织，
    使模型既能感知临近 Patch 的高频相对位置，又能感知远距离 Patch 的全局绝对坐标。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # 计算各个通道的角频率 (D/2,)

    pos = pos.reshape(-1)  # 展平成一维坐标向量 (M,)
    out = np.einsum("m,d->md", pos, omega)  # 外积计算相位矩阵: pos * omega -> (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # 拼接正弦与余弦分量 -> (M, D)
    return emb
