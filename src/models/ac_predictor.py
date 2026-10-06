# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """
    【动作条件预测器 (Action-Conditioned Vision Transformer Predictor / V-JEPA 2-AC)】

    具身智能与世界模型 (World Model) 的核心灵魂：
    在普通的 V-JEPA 中，预测器只根据无动作的上下文预测被遮挡的视频块。
    而在机器人/具身智能任务中，环境的未来演变是由“机器人的控制动作 (Action)”所驱动的！

    V-JEPA 2-AC (Action-Conditioned) 的设计：
    1. 冻结预训练好的视觉编码器 (Frozen ViT Encoder)；
    2. 将机械臂当前的本体感知状态 (State，如 7 自由度关节角/末端位姿) 和执行的动作 (Action) 投影为 Token；
    3. 将动作 Token、状态 Token 与当前帧的视觉 Patches 进行交织拼接；
    4. 采用时间因果注意力掩码 (Temporal Causal Attention Mask)，确保预测 t+1 步时只能看到 <=t 步的历史视觉与动作；
    5. 在隐空间预测下一时刻 $s_{t+1}$ 的视觉表征！

    这构成了可用于模型预测控制 (MPC) 和策略评估的完全可导的“隐空间神经物理模拟器”！
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        **kwargs
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics

        # 线性映射层：将视觉特征、机器人动作 (Action)、机器人状态 (State) 统一映射到预测器隐空间维度
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)

        # 视频与图像网格尺寸
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
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # 随机深度衰减率

        # 位置编码参数
        self.uniform_power = uniform_power

        # 堆叠的注意力块 (ACBlock: 支持因果掩码与 RoPE 旋转位置编码)
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
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

        # 输出层归一化与投影回原特征空间
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ 初始化权重
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # 构建因果注意力掩码 (Causal Attention Mask)
        # 保证时间维度上的单向因果性：当前时刻的状态和动作只能影响未来，不能倒流到过去
        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=3 if use_extrinsics else 2
            )
        self.attn_mask = attn_mask

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

    def forward(self, x, actions, states, extrinsics=None):
        """
        【动作条件前向推演 (Action-Conditioned Forward Rollout)】

        参数:
        :param x: 视觉上下文 Tokens，形状 [B, N_ctxt, D]
        :param actions: 机器人执行的控制动作向量，如末端位移或关节速度 [B, T, action_dim]
        :param states: 机器人本体感知状态向量，如位姿/角度 [B, T, state_dim]
        :param extrinsics: 相机外参（可选）

        返回:
        :return x: 在施加给定动作序列后，预测的未来视觉隐表征序列
        """
        # 步骤 1: 视觉特征映射到预测器维度
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # 步骤 2: 将动作 (Action) 和状态 (State) 编码为 Token，并在每个时间步交织拼接
        s = self.state_encoder(states).unsqueeze(2)  # [B, T, 1, D]
        a = self.action_encoder(actions).unsqueeze(2)  # [B, T, 1, D]
        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]

        # 每个时间步包含：[动作 a_t, 状态 s_t, 图像 Patches_t]
        if self.use_extrinsics:
            e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
            x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
        else:
            x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]

        cond_tokens = 3 if self.use_extrinsics else 2
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # 步骤 3: 带有因果掩码的 Transformer 深度推演
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=cond_tokens,
                )

        # 步骤 4: 剥离条件 Tokens (去掉 a_t 和 s_t)，只保留未来视频 Patches 的预测表征
        x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
        x = x[:, :, cond_tokens:, :].flatten(1, 2)

        # 步骤 5: 归一化并投影回原视觉特征维度
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
