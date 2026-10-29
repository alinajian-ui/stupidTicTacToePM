"""评估 Actor-Critic 模型对随机合法策略的表现。

每个模型分别作为 O（先手）和 X（后手）测试。
随机对手只会从 GameEngine 提供的合法动作中选择。
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from ai_environment import TicTacToeEnv
from ai_model import ActorCriticCNN, mask_illegal_actions


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ActorCriticCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def model_action(model, state, legal_mask, device, greedy):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(state_t)
        logits = mask_illegal_actions(logits, mask_t)
        if greedy:
            action = int(torch.argmax(logits, dim=1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
    return action


def play_game(model, model_player, device, greedy):
    env = TicTacToeEnv()
    env.reset()
    for _ in range(81):
        legal = env.legal_action_mask()
        if not legal.any():
            break
        current = env.game.current_player
        if current == model_player:
            action = model_action(model, env.get_state(), legal, device, greedy)
        else:
            action = random.choice(env.game.available_moves())
            action = action[0] * 9 + action[1]
        _, _, done, _ = env.step(action)
        if done:
            break
    return env.game.winner, env.game.draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_ac/policy_latest.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--greedy", action="store_true",
                        help="模型总是选择最高分动作；不加此参数则按策略采样")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games 必须至少为 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("评估设备：", torch.cuda.get_device_name(device))
    model = load_model(args.checkpoint, device)

    results = {
        "O": {"win": 0, "draw": 0, "loss": 0},
        "X": {"win": 0, "draw": 0, "loss": 0},
    }
    # 平分对局数，尽可能平衡先后手；奇数时多出的一局分配给 O。
    counts = {"O": (args.games + 1) // 2, "X": args.games // 2}
    for side, count in counts.items():
        model_player = 1 if side == "O" else -1
        for _ in range(count):
            winner, draw = play_game(model, model_player, device, args.greedy)
            if draw or winner == 0:
                results[side]["draw"] += 1
            elif winner == model_player:
                results[side]["win"] += 1
            else:
                results[side]["loss"] += 1

    for side in ("O", "X"):
        r = results[side]
        total = sum(r.values())
        print(
            f"模型执 {side}：胜 {r['win']}，平 {r['draw']}，负 {r['loss']}，"
            f"胜率 {100*r['win']/max(total,1):.1f}%（{total} 局）"
        )
    total_w = results["O"]["win"] + results["X"]["win"]
    total_d = results["O"]["draw"] + results["X"]["draw"]
    total_l = results["O"]["loss"] + results["X"]["loss"]
    print(f"合计：胜 {total_w}，平 {total_d}，负 {total_l}，共 {args.games} 局")
    print("说明：这是对随机合法策略的基准测试，不等同于与强 AI 对战的棋力评估。")


if __name__ == "__main__":
    main()
