"""AI 环境适配层的基本测试。运行：python -m unittest -v test_ai_environment.py"""
import unittest

import numpy as np

from ai_environment import TicTacToeEnv
from game_engine import GameEngine


class TicTacToeEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_initial_state_shape_and_dtype(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (5, 9, 9))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(int(self.env.legal_action_mask().sum()), 81)

    def test_action_mask_respects_forced_board(self):
        self.env.reset()
        _, _, done, _ = self.env.step(1 * 9 + 2)
        self.assertFalse(done)
        mask = self.env.legal_action_mask().reshape(9, 9)
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue(mask[3:6, 6:9].all())
        self.assertFalse(mask[0:3, 0:3].any())

    def test_step_rejects_illegal_action(self):
        self.env.reset()
        self.env.step(0)
        with self.assertRaises(ValueError):
            self.env.step(0)

    def test_state_is_current_player_relative(self):
        self.env.reset()
        self.env.step(0)  # O 在 (0, 0) 落子，轮到 X
        state = self.env.get_state()
        self.assertEqual(state[0, 0, 0], 0.0)  # X 的己方棋子通道
        self.assertEqual(state[1, 0, 0], 1.0)  # O 在对手通道中
        self.assertEqual(state[4, 0, 0], -1.0)

    def test_step_rewards_fifth_mini_board_win(self):
        # 直接构造 O 已赢四个中格的合法测试局面；最后一步赢得第五格后，
        # 环境必须把新多数制终局转换为 done=True 和获胜奖励 +1。
        self.env.game.mini_status = [
            [GameEngine.O_PLAYER, GameEngine.O_PLAYER, GameEngine.EMPTY],
            [GameEngine.O_PLAYER, GameEngine.EMPTY, GameEngine.EMPTY],
            [GameEngine.EMPTY, GameEngine.EMPTY, GameEngine.O_PLAYER],
        ]
        self.env.game.board[3][6] = GameEngine.O_PLAYER
        self.env.game.board[3][7] = GameEngine.O_PLAYER
        self.env.game.forced_board = (1, 2)

        _, reward, done, info = self.env.step(3 * 9 + 8)

        self.assertTrue(done)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["winner"], GameEngine.O_PLAYER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
