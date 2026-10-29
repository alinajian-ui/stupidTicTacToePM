"""TicTacToePM 的强化学习环境适配层。

本模块复用 game_engine.GameEngine，不复制或改写游戏规则。
状态采用 5 个 9x9 通道，动作采用 0..80 的单一编号。
"""
from __future__ import annotations

import numpy as np

from game_engine import GameEngine


class TicTacToeEnv:
    """为自我对弈训练提供 reset/step/state/mask 接口。"""

    ACTION_SIZE = GameEngine.BOARD_SIZE * GameEngine.BOARD_SIZE
    OBS_CHANNELS = 5

    def __init__(self):
        self.game = GameEngine()

    def reset(self):
        """开始新对局并返回初始状态。"""
        self.game.reset()
        return self.get_state()

    def get_state(self):
        """返回 float32 状态张量，形状为 (5, 9, 9)。

        通道 0：当前玩家自己的棋子。
        通道 1：对手棋子。
        通道 2：中格状态映射到对应的 3x3 小格区域，当前玩家赢为 +1，
                对手赢为 -1，平中格为 0.5。
        通道 3：当前被强制进入的中格掩码；无强制中格时全 0。
        通道 4：全部为 +1 或 -1，标记当前玩家，方便网络区分视角。
        """
        player = self.game.current_player
        state = np.zeros(
            (self.OBS_CHANNELS, GameEngine.BOARD_SIZE, GameEngine.BOARD_SIZE),
            dtype=np.float32,
        )

        for row in range(9):
            for col in range(9):
                value = self.game.board[row][col]
                if value == player:
                    state[0, row, col] = 1.0
                elif value == -player:
                    state[1, row, col] = 1.0

        for mini_row in range(3):
            for mini_col in range(3):
                status = self.game.mini_status[mini_row][mini_col]
                if status == player:
                    encoded = 1.0
                elif status == -player:
                    encoded = -1.0
                elif status == GameEngine.MINI_DRAW:
                    encoded = 0.5
                else:
                    encoded = 0.0

                state[
                    2,
                    mini_row * 3:(mini_row + 1) * 3,
                    mini_col * 3:(mini_col + 1) * 3,
                ] = encoded

        if self.game.forced_board is not None:
            mini_row, mini_col = self.game.forced_board
            state[
                3,
                mini_row * 3:(mini_row + 1) * 3,
                mini_col * 3:(mini_col + 1) * 3,
            ] = 1.0

        state[4, :, :] = float(player)
        return state

    def legal_action_mask(self):
        """返回长度为 81 的 bool 数组；True 表示合法落子。"""
        mask = np.zeros(self.ACTION_SIZE, dtype=np.bool_)
        for row, col in self.game.available_moves():
            mask[row * 9 + col] = True
        return mask

    def step(self, action):
        """执行 0..80 的动作，返回 (state, reward, done, info)。

        reward 是相对于“刚刚落子的一方”：
        赢得第五个中格并取得整局胜利时 +1，平局或尚未结束时 0。
        非法动作抛出 ValueError，避免训练时静默地产生错误样本。
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < 81:
            raise ValueError("action 必须是 0 到 80 的整数")

        legal_mask = self.legal_action_mask()
        if not legal_mask[int(action)]:
            raise ValueError(f"非法动作：{action}")

        player_who_moved = self.game.current_player
        row, col = divmod(int(action), 9)
        if not self.game.move(row, col):
            raise RuntimeError("合法动作被 GameEngine 拒绝，规则状态可能不一致")

        done = self.game.winner != GameEngine.EMPTY or self.game.draw
        reward = 0.0
        if self.game.winner == player_who_moved:
            reward = 1.0

        info = {
            "winner": self.game.winner,
            "draw": self.game.draw,
            "player_who_moved": player_who_moved,
        }
        return self.get_state(), reward, done, info
