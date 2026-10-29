"""使用 checkpoints_majority 模型进行本地人机对弈的 PySide6 窗口。"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from game_engine import GameEngine


PROJECT_DIR = Path(__file__).resolve().parent


class ModelBridge(QObject):
    """管理常驻模型进程，避免每个 AI 回合都重新加载检查点。"""

    ready = Signal(object)
    action_ready = Signal(int, float)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_DIR))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)

        self.stdout_buffer = ""
        self.stderr_buffer = ""
        self.is_ready = False
        self.pending_request = False
        self.stopping = False
        self.failure_reported = False
        self.request_id = 0

    @staticmethod
    def training_python():
        """返回包含 Torch 的项目训练解释器。"""

        candidates = (
            PROJECT_DIR / ".venv" / "Scripts" / "python.exe",
            PROJECT_DIR / ".venv" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "找不到训练环境解释器：请确认项目内 .venv 已创建并安装 PyTorch。"
        )

    @staticmethod
    def checkpoint_path():
        """优先使用最佳模型，若不存在则回退到最新模型。"""

        checkpoint_dir = PROJECT_DIR / "checkpoints_majority"
        for filename in ("policy_best.pt", "policy_latest.pt"):
            candidate = checkpoint_dir / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "checkpoints_majority 中没有 policy_best.pt 或 policy_latest.pt。"
        )

    def start(self):
        python_path = self.training_python()
        checkpoint_path = self.checkpoint_path()
        service_path = PROJECT_DIR / "model_service.py"
        self.process.setProgram(str(python_path))
        self.process.setArguments([
            "-u",
            str(service_path),
            "--checkpoint",
            str(checkpoint_path),
        ])
        self.process.start()

    def request_action(self, game):
        """把当前局面发送给模型；同一时间只允许一个推理请求。"""

        if not self.is_ready or self.pending_request:
            return False

        legal_actions = [
            row * GameEngine.BOARD_SIZE + col
            for row, col in game.available_moves()
        ]
        if not legal_actions:
            return False

        self.request_id += 1
        payload = {
            "id": self.request_id,
            "board": game.board,
            "mini_status": game.mini_status,
            "current_player": game.current_player,
            "forced_board": game.forced_board,
            "legal_actions": legal_actions,
        }
        message = json.dumps(payload, ensure_ascii=False) + "\n"
        self.pending_request = True
        self.process.write(message.encode("utf-8"))
        return True

    def _read_stdout(self):
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        self.stdout_buffer += data

        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._report_failure(f"模型服务返回了无效数据：{line}")
                continue

            message_type = message.get("type")
            if message_type == "ready":
                self.is_ready = True
                self.ready.emit(message)
            elif message_type == "action":
                self.pending_request = False
                self.action_ready.emit(
                    int(message["action"]),
                    float(message.get("value", 0.0)),
                )
            elif message_type == "error":
                self.pending_request = False
                self._report_failure(message.get("message", "模型推理失败"))

    def _read_stderr(self):
        data = bytes(self.process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )
        # 只保留最近的错误文本，防止第三方库长期输出导致内存增长。
        self.stderr_buffer = (self.stderr_buffer + data)[-6000:]

    def _process_error(self, _error):
        if self.stopping:
            return
        self._report_failure(self.process.errorString())

    def _process_finished(self, exit_code, _exit_status):
        if self.stopping:
            return
        detail = self.stderr_buffer.strip()
        message = f"模型服务意外退出（代码 {exit_code}）。"
        if detail:
            message += f"\n{detail}"
        self._report_failure(message)

    def _report_failure(self, message):
        if self.failure_reported:
            return
        self.failure_reported = True
        self.is_ready = False
        self.pending_request = False
        self.failed.emit(message)

    def stop(self):
        """关闭窗口时只终止本窗口创建的推理进程。"""

        self.stopping = True
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1000)


class HumanVsAIWindow(QMainWindow):
    """复用原棋盘绘制的人类对本地 Actor-Critic 模型窗口。"""

    def __init__(self, board_widget_class):
        super().__init__()
        self.setWindowTitle("TicTacToePM - 与训练模型对弈")
        self.resize(900, 960)

        self.game = GameEngine()
        self.human_player = GameEngine.O_PLAYER
        self.ai_player = GameEngine.X_PLAYER
        self.model_ready = False
        self.model_thinking = False
        self.model_error = None
        self.human_turn_starts = []

        self.status_label = QLabel("正在加载 checkpoints_majority 模型……")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-size: 17px; font-weight: 600; padding: 8px;"
        )
        self.model_label = QLabel("模型进程正在启动")
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_label.setWordWrap(True)

        self.side_combo = QComboBox()
        self.side_combo.addItem("玩家执 O（先手）", GameEngine.O_PLAYER)
        self.side_combo.addItem("玩家执 X（后手）", GameEngine.X_PLAYER)
        self.restart_button = QPushButton("按所选方重新开始")
        self.undo_button = QPushButton("撤销上一回合")

        # 传入回调后，原 BoardWidget 仍负责完全相同的棋盘绘制和坐标转换，
        # 人机窗口只接管回合权限、状态文字以及成功落子后的 AI 调度。
        self.board = board_widget_class(
            self.game,
            self.status_label,
            interaction_allowed=self.can_human_move,
            status_updater=self.update_status,
            after_move=self.after_human_move,
        )

        self.restart_button.clicked.connect(self.start_new_game)
        self.undo_button.clicked.connect(self.undo_round)

        controls = QHBoxLayout()
        controls.addWidget(self.side_combo)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.undo_button)

        layout = QVBoxLayout()
        layout.addWidget(self.model_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.board, 1)
        layout.addLayout(controls)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.bridge = ModelBridge(self)
        self.bridge.ready.connect(self.model_loaded)
        self.bridge.action_ready.connect(self.apply_ai_action)
        self.bridge.failed.connect(self.model_failed)
        self.update_controls()
        self.bridge.start()

    def model_loaded(self, metadata):
        self.model_ready = True
        checkpoint = Path(metadata["checkpoint"]).name
        episode = metadata.get("episode")
        score = metadata.get("best_eval_score")
        details = [f"已加载 {checkpoint}"]
        if episode is not None:
            details.append(f"训练局记录 {episode}")
        if score is not None:
            details.append(f"评估分数 {float(score):.3f}")
        self.model_label.setText(" | ".join(details))
        self.start_new_game()

    def model_failed(self, message):
        self.model_ready = False
        self.model_thinking = False
        self.model_error = message
        self.model_label.setText("模型加载或推理失败")
        self.model_label.setStyleSheet("color: #b00020;")
        self.update_status()
        self.update_controls()

    def start_new_game(self):
        """按下拉框选择重置棋局；玩家执 X 时由模型自动先手。"""

        if self.model_thinking:
            return
        self.game.reset()
        self.human_player = int(self.side_combo.currentData())
        self.ai_player = -self.human_player
        self.human_turn_starts.clear()
        self.board.refresh()
        self.update_controls()

        if self.model_ready and self.game.current_player == self.ai_player:
            QTimer.singleShot(0, self.request_ai_action)

    def can_human_move(self):
        return (
            self.model_ready
            and not self.model_thinking
            and self.game.winner == GameEngine.EMPTY
            and not self.game.draw
            and self.game.current_player == self.human_player
        )

    def after_human_move(self, history_depth):
        """记录本回合起点，使一次撤销同时恢复人类和随后 AI 的落子。"""

        self.human_turn_starts.append(history_depth)
        self.update_controls()
        if self.game.winner == GameEngine.EMPTY and not self.game.draw:
            QTimer.singleShot(80, self.request_ai_action)

    def request_ai_action(self):
        if (
            not self.model_ready
            or self.model_thinking
            or self.game.current_player != self.ai_player
            or self.game.winner != GameEngine.EMPTY
            or self.game.draw
        ):
            return

        self.model_thinking = True
        self.update_status()
        self.update_controls()
        if not self.bridge.request_action(self.game):
            self.model_failed("无法向模型发送当前局面。")

    def apply_ai_action(self, action, _value):
        self.model_thinking = False
        row, col = divmod(action, GameEngine.BOARD_SIZE)
        if (
            self.game.current_player != self.ai_player
            or not self.game.can_move(row, col)
            or not self.game.move(row, col)
        ):
            self.model_failed(f"模型返回非法落子：({row + 1}, {col + 1})")
            return

        self.board.refresh()
        self.update_controls()

    def undo_round(self):
        """撤销人类最近一手以及模型的回应，回到同一玩家重新选择。"""

        if self.model_thinking or not self.human_turn_starts:
            return
        target_depth = self.human_turn_starts.pop()
        while len(self.game.history) > target_depth:
            self.game.undo()
        self.board.refresh()
        self.update_controls()

    def update_status(self):
        o_score = self.game.won_mini_boards(GameEngine.O_PLAYER)
        x_score = self.game.won_mini_boards(GameEngine.X_PLAYER)
        score_text = f"中格 O {o_score} : {x_score} X"

        if self.model_error:
            self.status_label.setText(self.model_error)
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: 600; padding: 8px; color: #b00020;"
            )
            return

        self.status_label.setStyleSheet(
            "font-size: 17px; font-weight: 600; padding: 8px;"
        )
        if not self.model_ready:
            self.status_label.setText("正在加载 checkpoints_majority 模型……")
        elif self.game.winner == self.human_player:
            self.status_label.setText(f"游戏结束：你获胜！ | {score_text}")
        elif self.game.winner == self.ai_player:
            self.status_label.setText(f"游戏结束：模型获胜 | {score_text}")
        elif self.game.draw:
            self.status_label.setText(f"游戏结束：平局 | {score_text}")
        elif self.model_thinking or self.game.current_player == self.ai_player:
            ai_symbol = "O" if self.ai_player == GameEngine.O_PLAYER else "X"
            self.status_label.setText(f"模型思考中（{ai_symbol}）…… | {score_text}")
        else:
            human_symbol = "O" if self.human_player == GameEngine.O_PLAYER else "X"
            if self.game.forced_board is None:
                target_text = "可选择任意空格"
            else:
                row, col = self.game.forced_board
                target_text = f"请进入第 {row + 1} 行、第 {col + 1} 列中格"
            self.status_label.setText(
                f"你的回合（{human_symbol}） | {score_text} | {target_text}"
            )

    def update_controls(self):
        self.undo_button.setEnabled(
            bool(self.human_turn_starts) and not self.model_thinking
        )
        self.restart_button.setEnabled(self.model_ready and not self.model_thinking)
        self.side_combo.setEnabled(not self.model_thinking)

    def closeEvent(self, event):
        self.bridge.stop()
        super().closeEvent(event)
