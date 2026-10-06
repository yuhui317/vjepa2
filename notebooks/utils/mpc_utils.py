# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def cem(
    context_frame,
    context_pose,
    goal_frame,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_std=0.95,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    axis={},
    objective=l1,
    close_gripper=None,
):
    """
    【基于世界模型 (World Model) 与交叉熵方法 (CEM) 的模型预测控制 (MPC)】

    核心思想：
    在具身智能中，有了隐空间世界模型后，机械臂不需要真的去物理世界试错，
    而是在“脑海中的虚拟物理引擎 (Latent World Model)”里进行前向推演：
    1. 机械臂当前处于 context_frame，目标是达到 goal_frame；
    2. 随机生成 samples 条不同的候选动作轨迹 (Action Trajectories)；
    3. 利用世界模型向前推演 rollout 步，预测出每一条轨迹最终产生的未来视觉表征；
    4. 计算推演出的未来表征与目标 goal_frame 的隐空间 L1 距离（目标代价）；
    5. 挑选出效果最好的前 topk 个精英动作，更新动作高斯分布的均值 (mean) 与方差 (std)；
    6. 迭代优化 cem_steps 次后，收敛出的均值分布即为最优执行动作！

    参数解析：
    :param context_frame: 当前观察帧的隐特征 [B=1, T=1, HW, D] (例如 1x1x196x1024)
    :param context_pose: 机械臂当前本体位姿 (7 维: 3 维平移 + 3 维旋转/四元数 + 1 维夹爪)
    :param goal_frame: 目标期望达到的视觉帧特征 [B=1, T=1, HW, D]
    :param world_model: 具身世界模型包装器 f(frames, actions, poses) -> next_frame
    :param rollout: 向未来预测推演的时间步长 (Horizon)
    :param cem_steps: 交叉熵迭代优化轮数 (例如 100 轮)
    :param samples: 每轮采样的动作候选数量 (例如 100 条并行轨迹)
    :param topk: 每轮挑选出的精英轨迹数量 (例如选前 10 条最优轨迹更新分布)
    :return: [B=1, rollout, 7] 规划出的最优动作控制序列
    """
    context_frame = context_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame = goal_frame.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.cat(
        [
            torch.zeros((rollout, 3), device=context_frame.device),
            torch.zeros((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame.device) * maxnorm,
            torch.ones((rollout, 1), device=context_frame.device),
        ],
        dim=-1,
    )

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    def sample_action_traj():
        """从当前高斯分布采样多条动作序列，并用世界模型在隐空间前向推演未来状态"""
        action_traj, frame_traj, pose_traj = None, context_frame, context_pose

        for h in range(rollout):
            # 1. 采样新的动作候选 (平移 + 旋转 + 夹爪)
            action_samples = torch.randn(samples, mean.size(1), device=mean.device) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]
            action_samples = torch.cat(
                [
                    action_samples[:, :3],
                    torch.zeros((len(action_samples), 3), device=mean.device),
                    action_samples[:, -1:],
                ],
                dim=-1,
            )[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None else action_samples
            )

            # 2. 核心：调用隐空间世界模型向前推演一步未来的视觉特征 next_frame 与位姿 next_pose
            next_frame, next_pose = world_model(frame_traj, action_traj, pose_traj)
            frame_traj = torch.cat([frame_traj, next_frame], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj

    def select_topk_action_traj(final_state, goal_state, actions):
        """在隐空间计算推演出的最终状态与目标状态之间的 L1 距离，挑选出最接近目标的 top-k 条最优动作"""
        sims = objective(final_state.flatten(1), goal_state.flatten(1))
        # 寻找距离最小 (即最相似) 的 top-k 个样本索引
        indices = sims.topk(topk, largest=False).indices
        selected_actions = actions[indices]
        return selected_actions

    for step in tqdm(range(cem_steps), disable=True):
        action_traj, frame_traj = sample_action_traj()
        selected_actions = select_topk_action_traj(
            final_state=frame_traj[:, -1], goal_state=goal_frame, actions=action_traj
        )
        mean_selected_actions = selected_actions.mean(dim=0)
        std_selected_actions = selected_actions.std(dim=0)

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        logger.info(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

    new_action = torch.cat(
        [
            mean[..., :3],
            torch.zeros((rollout, 3), device=mean.device),
            round_small_elements(mean[..., -1:], 0.25),
        ],
        dim=-1,
    )[None, :]

    return new_action


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)
