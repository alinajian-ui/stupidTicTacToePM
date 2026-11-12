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
    DEFAULT_CENTER_CELL_REWARD = 0.02
    DEFAULT_MINI_BOARD_REWARD = 0.15
    # 【新增】阻止对方占领中格的奖励，与中心格、中格奖励同时生效。
    DEFAULT_BLOCK_REWARD = 0.2

    def __init__(
        self,
        center_cell_reward=DEFAULT_CENTER_CELL_REWARD,
        mini_board_reward=DEFAULT_MINI_BOARD_REWARD,
        block_reward=DEFAULT_BLOCK_REWARD,  # 【新增】阻止奖励
    ):
        if center_cell_reward < 0 or mini_board_reward < 0 or block_reward < 0:
            raise ValueError("中间格奖励、中格胜利奖励和阻止奖励不能为负数")

        self.game = GameEngine()
        # 两种奖励都显著小于整局获胜的 +1，只用于给稀疏终局奖励补充方向，
        # 避免模型为了局部奖励而忽视最终胜负。训练脚本可通过命令行覆盖。
        self.center_cell_reward = float(center_cell_reward)
        self.mini_board_reward = float(mini_board_reward)
        # 【新增】阻止对方占领中格时给本次落子的奖励。
        self.block_reward = float(block_reward)

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

    # 【新增】一个中格内的 8 条三连线，用 0..8 表示中格里的小格位置。
    _MINI_LINES = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    def _blocks_opponent(self, row, col, player):
        """【新增】判断在 (row, col) 落子是否挡住了对手占领该中格。

        必须在落子之前调用。“挡住”的意思是：该中格还没有分出结果，
        并且对手在某条经过 (row, col) 的三连线上已经有两个棋子，
        只差这一格就能赢下该中格。
        """
        mini_row = row // GameEngine.MINI_SIZE
        mini_col = col // GameEngine.MINI_SIZE
        # 中格已经有结果时，不存在“阻止占领”的说法。
        if self.game.mini_status[mini_row][mini_col] != GameEngine.EMPTY:
            return False

        # 把该中格的 9 个小格按 0..8 的顺序取出来。
        start_row = mini_row * GameEngine.MINI_SIZE
        start_col = mini_col * GameEngine.MINI_SIZE
        cells = [
            self.game.board[start_row + r][start_col + c]
            for r in range(GameEngine.MINI_SIZE)
            for c in range(GameEngine.MINI_SIZE)
        ]
        index = (row - start_row) * GameEngine.MINI_SIZE + (col - start_col)
        opponent = -player

        # 只检查经过本次落子位置的三连线：另外两格都是对手棋子即为阻止成功。
        for line in self._MINI_LINES:
            if index in line and all(
                cells[i] == opponent for i in line if i != index
            ):
                return True
        return False

    def step(self, action):
        """执行 0..80 的动作，返回 (state, reward, done, info)。

        reward 是相对于“刚刚落子的一方”：
        落在任一中格的中心小格时获得中心奖励，首次赢得一个中格时获得
        中格奖励，挡住对手即将连成的三连时获得阻止奖励（新增），
        赢得第五个中格并取得整局胜利时再加 +1。
        非法动作抛出 ValueError，避免训练时静默地产生错误样本。
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < 81:
            raise ValueError("action 必须是 0 到 80 的整数")

        legal_mask = self.legal_action_mask()
        if not legal_mask[int(action)]:
            raise ValueError(f"非法动作：{action}")

        player_who_moved = self.game.current_player
        row, col = divmod(int(action), 9)
        mini_row = row // GameEngine.MINI_SIZE
        mini_col = col // GameEngine.MINI_SIZE
        mini_status_before = self.game.mini_status[mini_row][mini_col]
        # 【新增】必须在落子前判断，因为落子后棋盘已经改变。
        blocked_mini_board = self._blocks_opponent(row, col, player_who_moved)

        if not self.game.move(row, col):
            raise RuntimeError("合法动作被 GameEngine 拒绝，规则状态可能不一致")

        done = self.game.winner != GameEngine.EMPTY or self.game.draw
        # 每个中格只有一个中心小格；由于重复落子会被规则引擎拒绝，该奖励
        # 在一局中最多触发九次，不会对同一位置重复发放。
        claimed_center_cell = (
            row % GameEngine.MINI_SIZE == 1
            and col % GameEngine.MINI_SIZE == 1
        )
        center_reward = (
            self.center_cell_reward if claimed_center_cell else 0.0
        )

        # 只有本次落子把尚未结束的中格变为当前玩家所有时才发放奖励。
        # 在已经获胜的中格中继续落子不会重复获得中格奖励。
        won_mini_board = (
            mini_status_before == GameEngine.EMPTY
            and self.game.mini_status[mini_row][mini_col] == player_who_moved
        )
        mini_reward = self.mini_board_reward if won_mini_board else 0.0
        # 【新增】挡住对手占领中格时发放阻止奖励。同一步若挡住了多条线，
        # 也只发放一次。
        block_reward = self.block_reward if blocked_mini_board else 0.0
        shaping_reward = center_reward + mini_reward + block_reward  # 【修改】加入阻止奖励

        terminal_reward = 0.0
        if self.game.winner == player_who_moved:
            terminal_reward = 1.0
        reward = shaping_reward + terminal_reward

        info = {
            "winner": self.game.winner,
            "draw": self.game.draw,
            "player_who_moved": player_who_moved,
            "claimed_center_cell": claimed_center_cell,
            "won_mini_board": won_mini_board,
            "center_reward": center_reward,
            "mini_board_reward": mini_reward,
            "blocked_mini_board": blocked_mini_board,  # 【新增】
            "block_reward": block_reward,  # 【新增】
            "shaping_reward": shaping_reward,
            "terminal_reward": terminal_reward,
        }
        return self.get_state(), reward, done, info
