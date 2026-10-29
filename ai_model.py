"""TicTacToePM Actor-Critic CNN。

输入：当前玩家视角的 (batch, 5, 9, 9) 状态。
输出：
- policy logits：81 个位置的动作分数；
- value：当前玩家最终结果的估计，范围约为 [-1, 1]。
"""
from __future__ import annotations

import torch
from torch import nn


class ActorCriticCNN(nn.Module):
    def __init__(self, in_channels: int = 5, action_size: int = 81):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.trunk(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


def mask_illegal_actions(logits, legal_mask):
    """把非法动作设为极小值，使其采样概率为零。"""
    return logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)
