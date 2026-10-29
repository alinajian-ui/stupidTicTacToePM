"""独立启动人机对弈窗口；也可在原 main.py 窗口中点击按钮进入。"""

import sys

from PySide6.QtWidgets import QApplication

from human_vs_ai import HumanVsAIWindow
from main import BoardWidget


def main():
    app = QApplication(sys.argv)
    window = HumanVsAIWindow(BoardWidget)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
