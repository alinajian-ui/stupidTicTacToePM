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
        # 最后一手同时赢得一个中格和整局，因此奖励为 0.15 + 1.0。
        self.assertAlmostEqual(reward, 1.15)
        self.assertEqual(info["winner"], GameEngine.O_PLAYER)
        self.assertTrue(info["won_mini_board"])
        self.assertAlmostEqual(info["mini_board_reward"], 0.15)
        self.assertAlmostEqual(info["terminal_reward"], 1.0)

    def test_center_cell_and_mini_board_rewards_can_stack(self):
        # O 已占据左上中格的一条对角线两端，在中心落子会同时占领该中格
        # 的中心小格并赢得中格，因此应获得两种塑形奖励之和。
        self.env.game.board[0][0] = GameEngine.O_PLAYER
        self.env.game.board[2][2] = GameEngine.O_PLAYER

        _, reward, done, info = self.env.step(1 * 9 + 1)

        self.assertFalse(done)
        self.assertTrue(info["claimed_center_cell"])
        self.assertTrue(info["won_mini_board"])
        self.assertAlmostEqual(info["center_reward"], 0.02)
        self.assertAlmostEqual(info["mini_board_reward"], 0.15)
        self.assertAlmostEqual(info["shaping_reward"], 0.17)
        self.assertAlmostEqual(reward, 0.17)

    def test_reward_values_are_configurable(self):
        env = TicTacToeEnv(center_cell_reward=0.05, mini_board_reward=0.25)
        env.game.board[0][0] = GameEngine.O_PLAYER
        env.game.board[2][2] = GameEngine.O_PLAYER

        _, reward, _, info = env.step(1 * 9 + 1)

        self.assertAlmostEqual(info["center_reward"], 0.05)
        self.assertAlmostEqual(info["mini_board_reward"], 0.25)
        self.assertAlmostEqual(reward, 0.30)

    def test_negative_shaping_reward_is_rejected(self):
        with self.assertRaises(ValueError):
            TicTacToeEnv(center_cell_reward=-0.01)
        with self.assertRaises(ValueError):
            TicTacToeEnv(block_reward=-0.01)

    # 【新增】阻止奖励测试
    def test_block_reward_when_stopping_opponent_line(self):
        # X 已在左上中格占据 (0,0) 和 (0,1)，O 落在 (0,2) 挡住这条线。
        env = TicTacToeEnv(block_reward=0.2)
        env.game.board[0][0] = GameEngine.X_PLAYER
        env.game.board[0][1] = GameEngine.X_PLAYER

        _, reward, _, info = env.step(0 * 9 + 2)

        self.assertTrue(info["blocked_mini_board"])
        self.assertAlmostEqual(info["block_reward"], 0.2)
        self.assertAlmostEqual(reward, 0.2)

    def test_no_block_reward_without_threat(self):
        # X 只有一个棋子，不构成威胁，不应获得阻止奖励。
        env = TicTacToeEnv(block_reward=0.2)
        env.game.board[0][0] = GameEngine.X_PLAYER

        _, reward, _, info = env.step(0 * 9 + 2)

        self.assertFalse(info["blocked_mini_board"])
        self.assertAlmostEqual(reward, 0.0)

    def test_no_block_reward_in_finished_mini_board(self):
        # 左上中格已被 X 赢下，再挡住其中的两连不算阻止。
        env = TicTacToeEnv(block_reward=0.2)
        env.game.mini_status[0][0] = GameEngine.X_PLAYER
        env.game.board[0][0] = GameEngine.X_PLAYER
        env.game.board[0][1] = GameEngine.X_PLAYER

        _, _, _, info = env.step(0 * 9 + 2)

        self.assertFalse(info["blocked_mini_board"])

    def test_block_reward_is_on_by_default(self):
        # 环境默认就发放 0.2 阻止奖励，与原有奖励同时生效。
        self.env.game.board[0][0] = GameEngine.X_PLAYER
        self.env.game.board[0][1] = GameEngine.X_PLAYER

        _, reward, _, info = self.env.step(0 * 9 + 2)

        self.assertTrue(info["blocked_mini_board"])
        self.assertAlmostEqual(info["block_reward"], 0.2)
        self.assertAlmostEqual(reward, 0.2)

    def test_block_and_center_rewards_can_stack(self):
        # X 占据 (0,1) 和 (2,1)，O 落在中心 (1,1)：既占中心又挡住竖线。
        self.env.game.board[0][1] = GameEngine.X_PLAYER
        self.env.game.board[2][1] = GameEngine.X_PLAYER

        _, reward, _, info = self.env.step(1 * 9 + 1)

        self.assertTrue(info["claimed_center_cell"])
        self.assertTrue(info["blocked_mini_board"])
        self.assertAlmostEqual(reward, 0.02 + 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
