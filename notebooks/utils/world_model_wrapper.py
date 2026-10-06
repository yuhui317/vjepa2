# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn.functional as F

from .mpc_utils import cem, compute_new_pose


class WorldModel(object):
    """
    【基于 V-JEPA 2-AC 的世界模型规划器封装 (World Model Planner)】

    这是具身智能中“世界模型驱动的规划与决策 (Model-Based Planning)”的标准范式：

    核心思想：
    1. 视觉编码器 (Encoder): 把当前环境 RGB 图片编码为紧凑的隐状态 $s_t \in \mathbb{R}^{N \times D}$；
       同时把目标图像 (Goal Image) 编码为目标隐状态 $g \in \mathbb{R}^{N \times D}$。
    2. 动作条件世界模型 (AC Predictor): 模拟真实世界物理交互：
       给定当前状态 $s_t$ 与假想的候选动作 $a_t$，在隐空间“畅想”出下一个状态 $\hat{s}_{t+1} = f(s_t, a_t)$。
    3. 模型预测控制 (MPC / CEM):
       用交叉熵采样法 (Cross-Entropy Method, CEM) 在世界模型中快速展开 (Rollout) 上百条候选动作轨迹，
       挑选出能让预测隐状态 $\hat{s}_{t+H}$ 与目标状态 $g$ 距离最近的那条动作序列，将第一步动作发送给真实机械臂执行！
    """

    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        transform,
        mpc_args={
            "rollout": 2,          # 前向预测的时间步长 (Horizon)
            "samples": 400,        # 每次迭代采样的候选动作序列数量
            "topk": 10,            # 选取表现最好的前 K 个动作轨迹用来更新高斯分布
            "cem_steps": 10,       # CEM 优化迭代轮数
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args

    def encode(self, image):
        """
        【将单帧观测图片映射为隐空间状态向量】
        输入: 原始图片 (H, W, 3)
        输出: 归一化后的时空特征向量 (B, Tokens_per_frame, D)
        """
        clip = np.expand_dims(image, axis=0)
        clip = self.transform(clip)[None, :]
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def infer_next_action(self, rep, pose, goal_rep, close_gripper=None):
        """
        【通过世界模型与 MPC 规划下一拍的最优控制动作】

        参数:
        :param rep: 当前时刻的视觉隐状态 $s_t$
        :param pose: 机械臂当前的本体位姿 $p_t$
        :param goal_rep: 目标任务的视觉隐状态 $g$
        :param close_gripper: 夹爪开关控制标志（可选）

        返回:
        :return mpc_action: 规划出的下一步最优动作 (如机械臂末端 dx, dy, dz, 旋转角速度)
        """

        # 内部模拟器单步函数：接收当前隐状态与候选动作，通过世界模型前向推演下一步隐状态
        def step_predictor(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            # 调用 V-JEPA 2-AC 预测未来特征
            next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            # 计算机械臂末端的新位姿 (运动学积分)
            next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
            return next_rep, next_pose

        # 运行 CEM (Cross-Entropy Method) 采样与梯度无关优化，选出最优动作
        mpc_action = cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            close_gripper=close_gripper,
            **self.mpc_args,
        )[0]

        return mpc_action
