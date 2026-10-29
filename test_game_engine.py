"""GameEngine 的回归测试，可直接运行，也可由 Visual Studio/unittest 发现。"""

import unittest

from game_engine import GameEngine


class GameEngineTests(unittest.TestCase):
    def setUp(self):
        self.game = GameEngine()

    def test_initial_state_and_basic_move(self):
        self.assertEqual(self.game.current_player, GameEngine.O_PLAYER)
        self.assertIsNone(self.game.forced_board)
        self.assertTrue(self.game.move(0, 0))
        self.assertEqual(self.game.board[0][0], GameEngine.O_PLAYER)
        self.assertEqual(self.game.current_player, GameEngine.X_PLAYER)
        self.assertEqual(self.game.forced_board, (0, 0))

    def test_invalid_move_does_not_change_state(self):
        self.game.move(0, 0)
        history_length = len(self.game.history)

        # 重复落子、越界以及不在强制中格内的点击都必须被拒绝。
        for row, col in ((0, 0), (-1, 0), (9, 0), (4, 4)):
            self.assertFalse(self.game.move(row, col))

        self.assertEqual(len(self.game.history), history_length)
        self.assertEqual(self.game.current_player, GameEngine.X_PLAYER)

    def test_forced_board_allows_only_corresponding_middle_board(self):
        self.game.move(1, 2)

        # 小格 (1, 2) 把下一位玩家送到中格 (1, 2)。
        self.assertTrue(self.game.can_move(3, 6))
        self.assertTrue(self.game.can_move(5, 8))
        self.assertFalse(self.game.can_move(2, 8))

    def test_mini_board_win_is_detected(self):
        self.game.board[0][0] = GameEngine.O_PLAYER
        self.game.board[0][1] = GameEngine.O_PLAYER

        self.assertTrue(self.game.move(0, 2))
        self.assertEqual(self.game.mini_status[0][0], GameEngine.O_PLAYER)

    def test_finished_mini_board_remains_playable_and_unchanged(self):
        self.game.mini_status[0][0] = GameEngine.O_PLAYER
        self.game.forced_board = (0, 0)
        self.game.current_player = GameEngine.X_PLAYER

        self.assertTrue(self.game.move(1, 1))
        self.assertEqual(self.game.board[1][1], GameEngine.X_PLAYER)
        self.assertEqual(self.game.mini_status[0][0], GameEngine.O_PLAYER)

    def test_full_target_board_releases_forced_board(self):
        # 填满右下中格，只留下当前落子 (0, 0)；该落子的局部坐标会把
        # 下一位玩家送往右下中格，因其无空位，引擎应自动解除限制。
        for row in range(6, 9):
            for col in range(6, 9):
                self.game.board[row][col] = (
                    GameEngine.O_PLAYER
                    if (row + col) % 2 == 0
                    else GameEngine.X_PLAYER
                )

        self.assertTrue(self.game.move(2, 2))
        self.assertIsNone(self.game.forced_board)
        self.assertTrue(self.game.can_move(4, 4))

    def test_three_connected_mini_boards_do_not_end_game(self):
        # 旧规则会在最上方三个中格连线时结束；新规则必须继续到五格。
        self.game.mini_status[0][0] = GameEngine.O_PLAYER
        self.game.mini_status[0][1] = GameEngine.O_PLAYER
        self.game.board[0][6] = GameEngine.O_PLAYER
        self.game.board[0][7] = GameEngine.O_PLAYER
        self.game.forced_board = (0, 2)

        self.assertTrue(self.game.move(0, 8))
        self.assertEqual(self.game.mini_status[0][2], GameEngine.O_PLAYER)
        self.assertEqual(self.game.won_mini_boards(GameEngine.O_PLAYER), 3)
        self.assertEqual(self.game.winner, GameEngine.EMPTY)
        self.assertNotEqual(self.game.available_moves(), [])

    def test_fifth_mini_board_wins_without_a_line(self):
        # 四个 O 中格刻意不构成三连，第五个也不补成三连，以证明胜负只
        # 取决于中格总数，而不是中格在 3×3 布局中的位置。
        self.game.mini_status = [
            [GameEngine.O_PLAYER, GameEngine.O_PLAYER, GameEngine.EMPTY],
            [GameEngine.O_PLAYER, GameEngine.EMPTY, GameEngine.EMPTY],
            [GameEngine.EMPTY, GameEngine.EMPTY, GameEngine.O_PLAYER],
        ]
        self.game.board[3][6] = GameEngine.O_PLAYER
        self.game.board[3][7] = GameEngine.O_PLAYER
        self.game.forced_board = (1, 2)

        self.assertTrue(self.game.move(3, 8))
        self.assertEqual(self.game.mini_status[1][2], GameEngine.O_PLAYER)
        self.assertEqual(self.game.won_mini_boards(GameEngine.O_PLAYER), 5)
        self.assertEqual(self.game.winner, GameEngine.O_PLAYER)
        self.assertEqual(self.game.available_moves(), [])

    def test_draw_requires_all_small_cells_to_be_full(self):
        # 九个中格都有结果并不代表立即平局，因为已获胜中格中的空位
        # 仍可下；这里双方各有四格，中央中格为和棋，均未达到五格。
        self.game.mini_status = [
            [1, -1, 1],
            [-1, GameEngine.MINI_DRAW, 1],
            [-1, 1, -1],
        ]
        self.assertTrue(self.game.can_move(0, 0))
        self.assertFalse(self.game.draw)

        # 构造只剩一个空格、且双方都只有四个中格的局面。
        self.game.board = [
            [
                GameEngine.O_PLAYER
                if (row + col) % 2 == 0
                else GameEngine.X_PLAYER
                for col in range(9)
            ]
            for row in range(9)
        ]
        self.game.board[8][8] = GameEngine.EMPTY
        self.game.current_player = GameEngine.X_PLAYER

        self.assertTrue(self.game.move(8, 8))
        self.assertTrue(self.game.draw)
        self.assertEqual(self.game.winner, GameEngine.EMPTY)

    def test_undo_restores_complete_state(self):
        self.game.move(0, 0)
        self.game.move(1, 1)

        self.assertTrue(self.game.undo())
        self.assertEqual(self.game.board[1][1], GameEngine.EMPTY)
        self.assertEqual(self.game.current_player, GameEngine.X_PLAYER)
        self.assertEqual(self.game.forced_board, (0, 0))

        self.assertTrue(self.game.undo())
        self.assertEqual(self.game.board[0][0], GameEngine.EMPTY)
        self.assertEqual(self.game.current_player, GameEngine.O_PLAYER)
        self.assertIsNone(self.game.forced_board)
        self.assertFalse(self.game.undo())

    def test_has_three_supports_flat_and_two_dimensional_boards(self):
        flat_board = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        nested_board = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]

        self.assertTrue(GameEngine.has_three(flat_board, 1))
        self.assertTrue(GameEngine.has_three(nested_board, -1))
        self.assertFalse(GameEngine.has_three(nested_board, 1))

    def test_has_three_rejects_invalid_board_shape(self):
        with self.assertRaises(ValueError):
            GameEngine.has_three([1, 1, 1], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
