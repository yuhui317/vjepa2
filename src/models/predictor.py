# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """
    【JEPA 预测器 (Vision Transformer Predictor)】

    核心作用：
    这是 V-JEPA 架构的核心组件之一。
    JEPA 不像 MAE 那样去在像素空间做微观重建（比如预测 RGB 像素值），
    而是在“隐空间 (Latent/Representation Space)”中进行宏观语义预测。

    预测器接收：
    1. 上下文 Tokens (Context Tokens): 由 Encoder 从未被遮挡的视频块中提取的特征向量；
    2. 目标可学习 Mask Tokens (Target Mask Tokens): 带有目标时空位置编码的占位符向量；

    预测器输出：
    预测被遮挡区域在目标编码器 (Target Encoder) 中的高层隐特征表示。
    由于只预测隐空间特征，模型被迫学习视频中的物体动力学、连续运动规律和高层物理语义，
    而不会把容量浪费在高频噪声、纹理细节或光影渐变上。
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        chop_last_n_tokens=0,
        use_rope=False,
        **kwargs
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.chop_last_n_tokens = chop_last_n_tokens

        # 将 Encoder 的输出特征维度 (embed_dim) 映射到预测器内部维度 (predictor_embed_dim)
        # 注意：预测器通常比编码器更窄更浅（如 6 层 vs 编码器 24 层），降低计算量
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # 可学习的掩码占位符 (Mask Tokens)
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
            )

        # 确定图像/视频网格尺寸与位置编码
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减规则

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
            )
        else:
            self.num_patches = num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # 位置编码 (Position Embedding)
        self.uniform_power = uniform_power

        self.predictor_pos_embed = None
        if not use_rope:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
            )

        # 堆叠 Transformer 注意力层 (Attention Blocks)
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if out_embed_dim is None:
            teacher_embed_dim = kwargs.get("teacher_embed_dim", None)
            if teacher_embed_dim is not None:
                out_embed_dim = teacher_embed_dim
            else:
                out_embed_dim = embed_dim

        # 归一化层与输出投影层：将预测器内部维度映射回目标编码器的特征维度 (out_embed_dim)
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_embed_dim, bias=True)

        # ------ 初始化权重
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # 正弦余弦 3D 位置编码
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_height // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        """
        【预测器前向传播】

        工作流程：
        1. 投影上下文 Tokens 到 predictor 维度；
        2. 给上下文 Tokens 加上各自对应的 3D 时空位置编码；
        3. 构造目标预测 Tokens (可学习的 mask_token + 目标区域的 3D 时空位置编码)；
        4. 拼接上下文 Tokens 与目标预测 Tokens；
        5. 按原始 Patch 空间/时间索引重新排序（保持时空相对顺序）；
        6. 送入 Transformer 注意力块进行多层交互预测；
        7. 抽取对应于目标区域的输出，投影回目标维度进行特征回归！

        :param x: 上下文特征 Tokens，形状 [B, N_ctxt, D]
        :param masks_x: 上下文 Patches 在原始视频中的索引列表
        :param masks_y: 待预测的目标 Patches 在原始视频中的索引列表
        """
        assert (masks_x is not None) and (masks_y is not None), "Cannot run predictor without mask indices"
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]

        # 批大小 Batch Size
        B = len(x) // len(masks_x)

        # 步骤 1: 映射上下文 Tokens 到预测器维度
        x = self.predictor_embed(x)
        if has_cls:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        _, N_ctxt, D = x.shape

        # 步骤 2: 为上下文 Tokens 加上对应的 3D 时空位置编码
        if not self.use_rope:
            x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embed, masks_x)

        # 步骤 3: 构造目标 Tokens (可学习占位符向量 + 目标时空位置编码)
        # “告诉我这里是什么位置，我来预测这里应该有什么特征”
        mask_index = mask_index % self.num_mask_tokens
        pred_tokens = self.mask_tokens[mask_index]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)
        # -- 加上目标区域专属的时空位置编码
        if not self.use_rope:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks_y)
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            pred_tokens += pos_embs

        # 步骤 4: 拼接上下文 Tokens 与待预测的目标 Tokens
        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # 记录所有 Tokens (上下文 + 目标) 在原视频中的时空绝对位置
        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)

        # 步骤 5: 按照时空网格自然顺序 (argsort) 进行排序
        # 这保证了序列里的时空拓扑关系一致，便于自注意力机制捕捉局部临近关系
        argsort = torch.argsort(masks, dim=1)  # [B, N]
        masks = torch.stack([masks[i, row] for i, row in enumerate(argsort)], dim=0)
        x = torch.stack([x[i, row, :] for i, row in enumerate(argsort)], dim=0)

        # 预留裁剪 Tokens 的选项
        if self.chop_last_n_tokens > 0:
            x = x[:, : -self.chop_last_n_tokens]
            masks = masks[:, : -self.chop_last_n_tokens]

        if has_cls:
            x = torch.cat([x_cls, x], dim=1)

        # 步骤 6: 通过多层 Transformer 注意力块进行深度特征交互
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, None, use_reentrant=False)
            else:
                x = blk(x, mask=masks, attn_mask=None)
        x = self.predictor_norm(x)

        if has_cls:
            x = x[:, 1:, :]

        # 步骤 7: 还原排序，并只剥离出属于“目标位置 (Target)”的预测特征
        # 因为在 JEPA 中，我们只对预测出来的目标特征计算 Loss，不对上下文 Tokens 计算 Loss
        if not self.return_all_tokens:
            reverse_argsort = torch.argsort(argsort, dim=1)  # [B, N]
            x = torch.stack([x[i, row, :] for i, row in enumerate(reverse_argsort)], dim=0)
            x = x[:, N_ctxt:]

        # 步骤 8: 线性投影到与 Target Encoder 输出完全一致的隐空间维度
        x = self.predictor_proj(x)

        return x


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
