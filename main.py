
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)

from game_engine import GameEngine


class BoardWidget(QWidget):
    # 成功落子后通知主窗口更新“悔棋”等控件。使用信号而不是让棋盘组件
    # 直接访问主窗口，可以保持绘图/点击处理与窗口控制逻辑相互独立。
    move_made = Signal()

    def __init__(
        self,
        game,
        status_label,
        interaction_allowed=None,
        status_updater=None,
        after_move=None,
    ):
        super().__init__()

        self.game = game
        self.status_label = status_label
        # 下面三个回调均为可选项。原双人窗口不传回调，因此行为完全保持
        # 不变；人机窗口则用它们限制只能在人类回合点击，并在落子后触发 AI。
        self.interaction_allowed = interaction_allowed
        self.status_updater = status_updater
        self.after_move = after_move

        self.setMinimumSize(600, 600)

    def get_geometry(self):
        """计算正方形棋盘在当前组件中的尺寸和左上角坐标。"""

        size = min(self.width(), self.height()) - 40
        cell = size / 9

        left = (self.width() - size) / 2
        top = (self.height() - size) / 2

        return size, cell, left, top

    def cell_at_position(self, x, y):
        """把组件坐标转换为棋盘格坐标；棋盘外返回 None。"""

        size, cell, left, top = self.get_geometry()

        # 必须在转成 int 之前检查边界。Python 的 int(-0.5) 会得到 0，
        # 若直接换算，紧贴棋盘左侧或上方的点击会被误当成第一个小格。
        if not (
            left <= x < left + size
            and top <= y < top + size
        ):
            return None

        row = int((y - top) / cell)
        col = int((x - left) / cell)
        return row, col

    def refresh(self):
        self.update()

        # 人机窗口需要显示“你的回合/模型思考中”等专用信息。若提供了
        # 状态更新回调，就把文字交给所属窗口生成，同时继续复用全部绘图。
        if self.status_updater is not None:
            self.status_updater()
            return

        # 始终显示双方赢得的中格数，让玩家能直接看到“五格多数制”的
        # 进度，而不再误以为需要让中格标记连成一线。
        o_score = self.game.won_mini_boards(GameEngine.O_PLAYER)
        x_score = self.game.won_mini_boards(GameEngine.X_PLAYER)
        score_text = f"中格 O {o_score} : {x_score} X"

        if self.game.winner == 1:
            self.status_label.setText(f"游戏结束：O 获胜！ | {score_text}")
        elif self.game.winner == -1:
            self.status_label.setText(f"游戏结束：X 获胜！ | {score_text}")
        elif self.game.draw:
            self.status_label.setText(f"游戏结束：平局！ | {score_text}")
        else:
            player = "O" if self.game.current_player == 1 else "X"
            target = self.game.forced_board

            if target is None:
                self.status_label.setText(
                    f"当前玩家：{player} | {score_text} | 可以选择任意空格"
                )
            else:
                r, c = target
                self.status_label.setText(
                    f"当前玩家：{player} | {score_text} | "
                    f"请在第 {r + 1} 行、第 {c + 1} 列中格落子"
                )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)

        size, cell, left, top = self.get_geometry()

        # 用柔和底色高亮当前允许进入的中格，使连续点击时无需自行计算
        # “上一步的小格位置对应哪一个中格”。无强制中格时不做高亮。
        target = self.game.forced_board

        if (
            target is not None
            and self.game.winner == 0
            and not self.game.draw
        ):
            br, bc = target

            painter.fillRect(
                int(left + bc * 3 * cell),
                int(top + br * 3 * cell),
                int(cell * 3),
                int(cell * 3),
                QColor(218, 238, 255),
            )

        # 绘制网格
        for i in range(10):
            width = 3 if i % 3 == 0 else 1

            painter.setPen(
                QPen(
                    Qt.GlobalColor.black
                    if i % 3 == 0
                    else Qt.GlobalColor.gray,
                    width,
                )
            )

            x = left + i * cell
            y = top + i * cell

            painter.drawLine(
                int(x), int(top),
                int(x), int(top + size)
            )
            painter.drawLine(
                int(left), int(y),
                int(left + size), int(y)
            )

        # 绘制棋子
        painter.setFont(QFont("Arial", max(10, int(cell * 0.55))))

        for r in range(9):
            for c in range(9):
                value = self.game.board[r][c]

                if value == 0:
                    continue

                symbol = "O" if value == 1 else "X"

                painter.setPen(
                    QPen(
                        Qt.GlobalColor.blue
                        if value == 1
                        else Qt.GlobalColor.red,
                        2,
                    )
                )

                painter.drawText(
                    int(left + c * cell),
                    int(top + r * cell),
                    int(cell),
                    int(cell),
                    Qt.AlignmentFlag.AlignCenter,
                    symbol,
                )

        # 标记已经结束的中格
        painter.setFont(QFont("Microsoft YaHei", max(9, int(cell * 0.35))))

        for br in range(3):
            for bc in range(3):
                result = self.game.mini_status[br][bc]

                if result == 0:
                    continue

                if result == 1:
                    text = "O胜"
                    color = Qt.GlobalColor.blue
                elif result == -1:
                    text = "X胜"
                    color = Qt.GlobalColor.red
                else:
                    text = "和棋"
                    color = Qt.GlobalColor.darkGray

                painter.setPen(QPen(color, 2))

                rect_x = left + bc * 3 * cell
                rect_y = top + br * 3 * cell

                painter.drawText(
                    int(rect_x),
                    int(rect_y),
                    int(cell * 3),
                    int(cell * 3),
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if (
            self.interaction_allowed is not None
            and not self.interaction_allowed()
        ):
            return

        cell_position = self.cell_at_position(
            event.position().x(),
            event.position().y(),
        )
        if cell_position is None:
            return

        row, col = cell_position
        history_depth = len(self.game.history)
        if self.game.move(row, col):
            self.refresh()
            self.move_made.emit()
            if self.after_move is not None:
                self.after_move(history_depth)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TicTacToePM")
        self.resize(850, 900)

        self.game = GameEngine()

        self.status_label = QLabel()
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setStyleSheet(
            "font-size: 18px; font-weight: 600; padding: 8px;"
        )

        self.board = BoardWidget(
            self.game,
            self.status_label
        )

        self.restart_button = QPushButton("重新开始")
        self.undo_button = QPushButton("悔棋")
        self.ai_button = QPushButton("与训练模型对弈")
        self.ai_window = None

        self.restart_button.clicked.connect(self.restart_game)
        self.undo_button.clicked.connect(self.undo_move)
        self.ai_button.clicked.connect(self.open_ai_window)
        self.board.move_made.connect(self.update_controls)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.restart_button)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.ai_button)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.board, 1)
        layout.addLayout(button_layout)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.board.refresh()
        self.update_controls()

    def update_controls(self):
        """根据当前历史记录启用或禁用悔棋按钮。"""

        # 非法点击不会写入 history，因此也不会错误地启用悔棋。
        self.undo_button.setEnabled(bool(self.game.history))

    def restart_game(self):
        self.game.reset()
        self.board.refresh()
        self.update_controls()

    def undo_move(self):
        if self.game.undo():
            self.board.refresh()
            self.update_controls()

    def open_ai_window(self):
        """打开独立人机窗口，原双人棋局与训练入口均不受影响。"""

        if self.ai_window is not None:
            self.ai_window.show()
            self.ai_window.raise_()
            self.ai_window.activateWindow()
            return

        try:
            # 延迟导入保证即使训练环境或模型文件损坏，原双人窗口仍然可以
            # 正常启动；具体错误会在点击人机按钮时以对话框显示。
            from human_vs_ai import HumanVsAIWindow

            self.ai_window = HumanVsAIWindow(BoardWidget)
            self.ai_window.setAttribute(
                Qt.WidgetAttribute.WA_DeleteOnClose,
                True,
            )
            self.ai_window.destroyed.connect(self._clear_ai_window)
            self.ai_window.show()
        except Exception as error:
            self.ai_window = None
            QMessageBox.critical(
                self,
                "无法启动人机对弈",
                str(error),
            )

    def _clear_ai_window(self):
        """在人机窗口销毁后清除引用，允许再次一键打开。"""

        self.ai_window = None


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
