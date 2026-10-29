"""评估最佳模型；如无 best 模型则评估 latest。"""
import os
import torch
from ai_environment import TicTacToeEnv
from ai_model import ActorCriticCNN, mask_illegal_actions


def main():
    best_path = os.path.join("checkpoints_ac", "policy_best.pt")
    latest_path = os.path.join("checkpoints_ac", "policy_latest.pt")
    path = best_path if os.path.isfile(best_path) else latest_path
    if not os.path.isfile(path):
        raise FileNotFoundError("找不到模型，请先运行 train_selfplay.py。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ActorCriticCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    env = TicTacToeEnv()
    state = env.reset()
    legal = env.legal_action_mask()
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.as_tensor(legal, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value = model(s)
        logits = mask_illegal_actions(logits, m)
        action = int(torch.argmax(logits, dim=1).item())
    row, col = divmod(action, 9)
    print(f"加载模型：{path}")
    print(f"训练局数记录：{ckpt.get('episode', '未知')}")
    print(f"最佳评估分数记录：{ckpt.get('best_eval_score', '未知')}")
    print(f"价值估计：{value.item():.3f}")
    print(f"推荐落子：第 {row+1} 行、第 {col+1} 列；是否合法：{bool(legal[action])}")


if __name__ == "__main__":
    main()
