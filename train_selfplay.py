"""Actor-Critic 自我对弈训练：持续更新 latest，只在评估分数提高时更新 best。

用法：
  python train_selfplay.py --episodes 500 --resume --device cuda
  python train_selfplay.py --episodes 500 --resume --device cuda --eval-games 40

检查点：
  checkpoints_ac/policy_latest.pt  每次保存最新训练状态，会被覆盖
  checkpoints_ac/policy_best.pt    仅在随机对手评估分数提升时更新
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from ai_environment import TicTacToeEnv
from ai_model import ActorCriticCNN, mask_illegal_actions


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请检查当前项目 .venv 的 PyTorch 安装。")
    return device


def select_action(model, state, legal_mask, device, temperature, greedy=False):
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=device).unsqueeze(0)
    logits, value = model(state_t)
    logits = mask_illegal_actions(logits, mask_t) / temperature
    dist = Categorical(logits=logits)
    action = torch.argmax(logits, dim=1) if greedy else dist.sample()
    return int(action.item()), dist.log_prob(action).squeeze(0), value.squeeze(0)


def evaluate_vs_random(model, device, games=40, seed=123):
    """模型与随机合法策略对战，平衡 O/X 座位；得分 = (胜 + 0.5*平)/局数。"""
    rng = random.Random(seed)
    model.eval()
    results = {"win": 0, "draw": 0, "loss": 0}
    side_counts = [("O", (games + 1) // 2), ("X", games // 2)]
    with torch.no_grad():
        for side, count in side_counts:
            model_player = 1 if side == "O" else -1
            for _ in range(count):
                env = TicTacToeEnv()
                env.reset()
                for _ply in range(81):
                    legal = env.legal_action_mask()
                    if not legal.any():
                        break
                    current = env.game.current_player
                    if current == model_player:
                        action, _, _ = select_action(
                            model, env.get_state(), legal, device, 1.0, greedy=True
                        )
                    else:
                        row, col = rng.choice(env.game.available_moves())
                        action = row * 9 + col
                    _, _, done, _ = env.step(action)
                    if done:
                        break
                winner, draw = env.game.winner, env.game.draw
                if draw or winner == 0:
                    results["draw"] += 1
                elif winner == model_player:
                    results["win"] += 1
                else:
                    results["loss"] += 1
    model.train()
    total = sum(results.values())
    score = (results["win"] + 0.5 * results["draw"]) / max(total, 1)
    return score, results


def save_checkpoint(path, model, optimizer, episode, totals, best_score, device):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "episode": int(episode),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "totals": {k: int(v) for k, v in totals.items()},
        "best_eval_score": float(best_score),
        "algorithm": "actor_critic_monte_carlo",
        "device": str(device),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Actor-Critic 自我对弈 + 最佳模型筛选")
    parser.add_argument("--episodes", type=int, default=500, help="本次新增训练局数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints_ac")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="每多少局评估一次并决定是否更新 best")
    parser.add_argument("--eval-games", type=int, default=40,
                        help="每次评估的对局数，尽量平衡 O/X")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(args.episodes, args.save_every, args.eval_every, args.eval_games, args.log_every) < 1:
        parser.error("episodes/save-every/eval-every/eval-games/log-every 都必须大于 0")
    if args.lr <= 0 or args.temperature <= 0:
        parser.error("lr 和 temperature 必须大于 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    print(f"训练设备：{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    model = ActorCriticCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_episode = 1
    totals = {"o_wins": 0, "x_wins": 0, "draws": 0}
    best_score = -1.0
    latest_path = os.path.join(args.checkpoint_dir, "policy_latest.pt")
    best_path = os.path.join(args.checkpoint_dir, "policy_best.pt")

    if args.resume:
        if not os.path.isfile(latest_path):
            raise FileNotFoundError(f"找不到续训检查点：{latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_episode = int(ckpt.get("episode", 0)) + 1
        totals.update(ckpt.get("totals", {}))
        best_score = float(ckpt.get("best_eval_score", -1.0))
        # 若 latest 来自旧版脚本而没有 best_score，尝试从 best 检查点读取。
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
            best_score = max(best_score, float(best_ckpt.get("best_eval_score", -1.0)))
        print(f"从第 {start_episode} 局继续；当前记录的最佳评估分数 {best_score:.3f}")

    started = time.time()
    model.train()
    for episode in range(start_episode, start_episode + args.episodes):
        env = TicTacToeEnv()
        env.reset()
        trajectory = []
        for _ in range(81):
            state = env.get_state()
            legal = env.legal_action_mask()
            if not legal.any():
                break
            player = env.game.current_player
            action, log_prob, value = select_action(
                model, state, legal, device, args.temperature
            )
            trajectory.append((log_prob, value, player))
            _, _, done, _ = env.step(action)
            if done:
                break

        winner, draw = env.game.winner, env.game.draw
        if draw:
            totals["draws"] += 1
        elif winner == 1:
            totals["o_wins"] += 1
        elif winner == -1:
            totals["x_wins"] += 1

        if trajectory:
            log_probs = torch.stack([t[0] for t in trajectory])
            values = torch.stack([t[1] for t in trajectory])
            targets = torch.tensor([
                0.0 if draw or winner == 0 else (1.0 if player == winner else -1.0)
                for _, _, player in trajectory
            ], dtype=torch.float32, device=device)
            advantages = targets - values.detach()
            policy_loss = -(log_probs * advantages).mean()
            value_loss = nn.functional.smooth_l1_loss(values, targets)
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        completed = episode - start_episode + 1
        if completed % args.log_every == 0 or completed == 1:
            print(
                f"训练局 {episode} | O胜 {totals['o_wins']} | X胜 {totals['x_wins']} "
                f"| 平 {totals['draws']} | 用时 {time.time()-started:.1f}s"
            )

        should_evaluate = completed % args.eval_every == 0 or completed == args.episodes
        if completed % args.save_every == 0 or should_evaluate:
            save_checkpoint(latest_path, model, optimizer, episode, totals, best_score, device)
            print(f"最新模型已保存：{latest_path}")

        if should_evaluate:
            score, result = evaluate_vs_random(
                model, device, games=args.eval_games, seed=10000 + episode
            )
            print(
                f"评估（{args.eval_games} 局）得分 {score:.3f} | "
                f"胜 {result['win']} 平 {result['draw']} 负 {result['loss']}"
            )
            if score > best_score:
                best_score = score
                save_checkpoint(best_path, model, optimizer, episode, totals, best_score, device)
                print(f"刷新最佳模型！新分数 {best_score:.3f}，保存至：{best_path}")
            else:
                print(f"未超过历史最佳分数 {best_score:.3f}，保留原最佳模型。")
            # 将最新检查点里的最佳分数也写入，供下次续训比较。
            save_checkpoint(latest_path, model, optimizer, episode, totals, best_score, device)

    print("本轮训练结束。latest 是最新模型，best 是随机对手评估分数最高的模型。")


if __name__ == "__main__":
    main()
