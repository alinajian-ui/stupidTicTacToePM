"""在训练虚拟环境中常驻运行的模型推理服务。

GUI 环境只需要 PySide6；本服务由项目内用于训练的 `.venv` 启动，负责加载
PyTorch 模型，并通过标准输入/输出的逐行 JSON 与 GUI 交换局面和落子。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def encode_state(torch, payload, device):
    """按照 ai_environment.py 相同的五通道格式编码一个 GUI 局面。"""

    board = payload["board"]
    mini_status = payload["mini_status"]
    player = int(payload["current_player"])
    state = torch.zeros((1, 5, 9, 9), dtype=torch.float32, device=device)

    for row in range(9):
        for col in range(9):
            value = int(board[row][col])
            if value == player:
                state[0, 0, row, col] = 1.0
            elif value == -player:
                state[0, 1, row, col] = 1.0

    for mini_row in range(3):
        for mini_col in range(3):
            status = int(mini_status[mini_row][mini_col])
            if status == player:
                encoded = 1.0
            elif status == -player:
                encoded = -1.0
            elif status == 2:
                encoded = 0.5
            else:
                encoded = 0.0
            state[
                0,
                2,
                mini_row * 3:(mini_row + 1) * 3,
                mini_col * 3:(mini_col + 1) * 3,
            ] = encoded

    forced_board = payload.get("forced_board")
    if forced_board is not None:
        mini_row, mini_col = map(int, forced_board)
        state[
            0,
            3,
            mini_row * 3:(mini_row + 1) * 3,
            mini_col * 3:(mini_col + 1) * 3,
        ] = 1.0

    state[0, 4, :, :] = float(player)
    return state


def emit(message):
    """输出一条完整 JSON 并立即刷新，避免 GUI 等待缓冲区。"""

    print(json.dumps(message, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="TicTacToePM 模型推理服务")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="测试用：处理指定次数后退出；0 表示持续运行",
    )
    args = parser.parse_args()

    # 延迟导入使缺少训练依赖时的错误由服务进程返回，不会破坏原 GUI。
    import torch

    from ai_model import ActorCriticCNN, mask_illegal_actions

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型文件：{checkpoint_path}")

    # GUI 推理固定使用 CPU：棋盘模型较小，CPU 足以实时响应，并可避免 GUI
    # 因 CUDA 驱动、显存占用或训练进程并行运行而启动失败。
    device = torch.device("cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = ActorCriticCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    emit({
        "type": "ready",
        "checkpoint": str(checkpoint_path),
        "episode": checkpoint.get("episode"),
        "best_eval_score": checkpoint.get("best_eval_score"),
    })

    handled = 0
    for line in sys.stdin:
        if not line.strip():
            continue

        request_id = None
        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            legal_actions = [int(action) for action in payload["legal_actions"]]
            if not legal_actions:
                raise ValueError("当前局面没有合法动作")

            state = encode_state(torch, payload, device)
            legal_mask = torch.zeros((1, 81), dtype=torch.bool, device=device)
            legal_mask[0, legal_actions] = True

            with torch.no_grad():
                logits, value = model(state)
                logits = mask_illegal_actions(logits, legal_mask)
                action = int(torch.argmax(logits, dim=1).item())

            if action not in legal_actions:
                raise RuntimeError("模型返回了非法动作")

            emit({
                "type": "action",
                "id": request_id,
                "action": action,
                "value": float(value.item()),
            })
        except Exception as error:
            emit({
                "type": "error",
                "id": request_id,
                "message": f"{type(error).__name__}: {error}",
            })

        handled += 1
        if args.max_requests and handled >= args.max_requests:
            break


if __name__ == "__main__":
    main()
