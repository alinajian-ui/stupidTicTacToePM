"""TicTacToePM 的纯规则引擎。

这个模块不依赖 PySide6，界面、测试以及后续的训练程序都可以复用同一套规则。
"""


class GameEngine:
    """维护棋盘状态，并负责判断落子、胜负、平局和悔棋。"""

    EMPTY = 0
    O_PLAYER = 1
    X_PLAYER = -1
    MINI_DRAW = 2

    BOARD_SIZE = 9
    MINI_SIZE = 3
    # 整局不再根据中格是否连成一线判胜。九个中格中，任一方先赢得
    # 五个就已经取得不可逆的多数，因此可以立即结束，不必填满剩余小格。
    WINNING_MINI_BOARDS = 5

    def __init__(self):
        self.reset()

    def reset(self):
        """恢复到 O 方先手的新对局。"""

        # 9×9 小格棋盘：0 表示空，1 表示 O，-1 表示 X。
        self.board = [
            [self.EMPTY for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]

        # 3×3 中格状态：0 未决，1 O 胜，-1 X 胜，2 表示填满且无人获胜。
        # 中格一旦分出胜负，之后即使继续在其空格落子，结果也不会被覆盖。
        self.mini_status = [
            [self.EMPTY for _ in range(self.MINI_SIZE)]
            for _ in range(self.MINI_SIZE)
        ]

        self.current_player = self.O_PLAYER
        self.forced_board = None
        self.winner = self.EMPTY
        self.draw = False

        # 每次成功落子前保存完整快照，使一次悔棋能精确恢复全部规则状态。
        self.history = []

    @staticmethod
    def has_three(values, player):
        """判断单个中格中是否存在指定玩家的三连。"""

        # 将两种输入统一摊平成九元素，避免两套三连规则产生差异。
        if len(values) == 3 and all(isinstance(row, (list, tuple)) for row in values):
            flat_values = [value for row in values for value in row]
        else:
            flat_values = list(values)

        if len(flat_values) != 9:
            raise ValueError("三连判断需要九个格子的状态")

        winning_lines = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        )

        return any(
            all(flat_values[index] == player for index in line)
            for line in winning_lines
        )

    def won_mini_boards(self, player):
        """返回指定玩家已经赢得的中格数量。"""

        return sum(
            status == player
            for status_row in self.mini_status
            for status in status_row
        )

    def mini_board_has_space(self, board_row, board_col):
        """返回指定中格内是否至少还有一个可落子的小格。"""

        start_row = board_row * self.MINI_SIZE
        start_col = board_col * self.MINI_SIZE
        return any(
            self.board[row][col] == self.EMPTY
            for row in range(start_row, start_row + self.MINI_SIZE)
            for col in range(start_col, start_col + self.MINI_SIZE)
        )

    def can_move(self, row, col):
        """判断坐标是否是当前局面中的合法落子。"""

        if self.winner != self.EMPTY or self.draw:
            return False

        if not (
            0 <= row < self.BOARD_SIZE
            and 0 <= col < self.BOARD_SIZE
        ):
            return False

        # 已有棋子的格子绝不能重复占用。
        if self.board[row][col] != self.EMPTY:
            return False

        if self.forced_board is None:
            return True

        forced_row, forced_col = self.forced_board
        return (
            row // self.MINI_SIZE == forced_row
            and col // self.MINI_SIZE == forced_col
        )

    def available_moves(self):
        """列出当前所有合法落子，供界面提示和后续训练代码复用。"""

        if self.winner != self.EMPTY or self.draw:
            return []

        if self.forced_board is None:
            rows = range(self.BOARD_SIZE)
            cols = range(self.BOARD_SIZE)
        else:
            board_row, board_col = self.forced_board
            rows = range(
                board_row * self.MINI_SIZE,
                (board_row + 1) * self.MINI_SIZE,
            )
            cols = range(
                board_col * self.MINI_SIZE,
                (board_col + 1) * self.MINI_SIZE,
            )

        return [
            (row, col)
            for row in rows
            for col in cols
            if self.board[row][col] == self.EMPTY
        ]

    def move(self, row, col):
        """执行一次落子；成功返回 True，非法落子不改变状态并返回 False。"""

        if not self.can_move(row, col):
            return False

        # 快照中的二维列表必须逐行复制，否则后续落子会同时改动历史记录。
        self.history.append((
            [line[:] for line in self.board],
            [line[:] for line in self.mini_status],
            self.current_player,
            self.forced_board,
            self.winner,
            self.draw,
        ))

        player = self.current_player
        self.board[row][col] = player

        board_row = row // self.MINI_SIZE
        board_col = col // self.MINI_SIZE

        # 只在中格尚未分出结果时更新它。根据项目规则，已获胜中格中的空位
        # 仍可继续落子，但原有 O 胜或 X 胜标记必须保持不变。
        if self.mini_status[board_row][board_col] == self.EMPTY:
            local_values = [
                self.board[local_row][local_col]
                for local_row in range(
                    board_row * self.MINI_SIZE,
                    (board_row + 1) * self.MINI_SIZE,
                )
                for local_col in range(
                    board_col * self.MINI_SIZE,
                    (board_col + 1) * self.MINI_SIZE,
                )
            ]

            if self.has_three(local_values, player):
                self.mini_status[board_row][board_col] = player
            elif all(value != self.EMPTY for value in local_values):
                self.mini_status[board_row][board_col] = self.MINI_DRAW

        # 整局采用中格多数制，而不是中格连线制。九个中格中先取得五个
        # 的玩家必然比对手多，所以第五个中格确定后立即结束对局。
        if self.won_mini_boards(player) >= self.WINNING_MINI_BOARDS:
            self.winner = player

        # 中格可能以和棋结束，也允许在已获胜中格的空位继续落子；因此
        # 双方都有可能无法取得五个中格。只有 81 个小格全部填满且仍无人
        # 达到五个中格时，整局才判为平局。
        if (
            self.winner == self.EMPTY
            and all(
                value != self.EMPTY
                for board_line in self.board
                for value in board_line
            )
        ):
            self.draw = True

        target_board = (row % self.MINI_SIZE, col % self.MINI_SIZE)

        # 一般情况下，对手必须进入落子对应的中格。若该中格九个位置已经
        # 全满，继续强制会造成无合法步骤的死局，因此此时允许对手任选空格。
        if self.mini_board_has_space(*target_board):
            self.forced_board = target_board
        else:
            self.forced_board = None

        self.current_player *= -1
        return True

    def undo(self):
        """撤销最近一次成功落子；无历史时返回 False。"""

        if not self.history:
            return False

        (
            self.board,
            self.mini_status,
            self.current_player,
            self.forced_board,
            self.winner,
            self.draw,
        ) = self.history.pop()
        return True
