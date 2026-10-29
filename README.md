# TicTacToePM

## 怎么玩

棋盘由 `9 × 9` 个小格组成，并划分为九个 `3 × 3` 中格。O 先手；一次落子在
当前中格中的局部行、列位置，会指定对手下一步必须进入的中格。例如在某个中格的
右下小格落子，对手下一步就必须进入右下中格；如果该中格已经填满，对手可选择任意
空格。玩家在一个中格内横、竖或斜向连成三个小格，即赢得该中格；已经分出胜负的
中格仍可继续落子，但归属不会改变。整局不检查中格连线，O 或 X 先累计赢得 5 个
中格即获胜；若 81 个小格填满后双方仍都未达到 5 个中格，则为平局。

## 各文件作用

| 文件或目录 | 作用 |
| --- | --- |
| `main.py` | 原本的本地双人图形界面，同时提供“与训练模型对弈”入口。 |
| `game_engine.py` | 唯一规则引擎，处理合法落子、中格归属、五格胜利、平局和悔棋。 |
| `human_vs_ai.py` | 人机对弈窗口，处理先后手选择、整回合撤销和模型进程通信。 |
| `model_service.py` | 在训练虚拟环境中常驻加载 PyTorch 模型，并为 GUI 返回贪心合法动作。 |
| `play_vs_ai.py` | 跳过双人主窗口，直接启动人机对弈窗口。 |
| `ai_environment.py` | 将 `GameEngine` 包装成强化学习需要的状态、动作掩码和奖励接口。 |
| `ai_model.py` | Actor-Critic CNN；Actor 输出 81 个动作分数，Critic 估计局面价值。 |
| `train_selfplay.py` | Actor-Critic 自我对弈训练入口，保存最新模型和最佳评估模型。 |
| `evaluate_vs_random.py` | 让指定模型分别执 O/X 与随机合法策略对战。 |
| `evaluate_model.py` | 旧 `checkpoints_ac` 模型的单局推荐动作检查工具。 |
| `test_game_engine.py` | 规则引擎回归测试。 |
| `test_ai_environment.py` | AI 状态、动作掩码、奖励和终局测试。 |
| `checkpoints_majority/` | 当前五格多数制模型目录，包含 `policy_latest.pt` 和 `policy_best.pt`。 |
| `checkpoints/` | 早期 REINFORCE 模型目录，仅保留作历史记录。 |
| `checkpoints_ac/` | 旧胜利规则下的 Actor-Critic 模型目录，仅保留作历史记录。 |
| `TicTacToePM.pyproj` | Visual Studio Python 项目配置，默认启动文件为 `main.py`。 |

## 环境说明

项目当前保留两个虚拟环境，分别服务于 GUI 和训练：

- 上一级 `..\.venv`：安装 PySide6，用于运行图形界面。
- 项目目录内 `.\.venv`：安装 PyTorch、CUDA 和 NumPy，用于训练、评估及模型推理。

PowerShell 执行策略阻止激活脚本时，可先在当前终端执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

检查训练环境中的 PyTorch、CUDA 和显卡：

```powershell
& .\.venv\Scripts\Activate.ps1
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 启动游戏

### 双人对弈与一键人机入口

在项目目录中使用 GUI 环境运行：

```powershell
& ..\.venv\Scripts\Activate.ps1
python main.py
```

主窗口保留原双人对弈、重新开始和悔棋功能。点击“与训练模型对弈”，即可打开
独立人机窗口。

### 直接启动人机窗口

```powershell
& ..\.venv\Scripts\Activate.ps1
python play_vs_ai.py
```

人机窗口会：

1. 优先加载 `checkpoints_majority/policy_best.pt`；若不存在则加载
   `checkpoints_majority/policy_latest.pt`。
2. 允许玩家选择执 O 先手或执 X 后手；玩家执 X 时模型自动先行。
3. 让模型从当前合法动作中选择策略分数最高的位置。
4. 通过“撤销上一回合”同时撤销玩家落子和模型回应。
5. 在关闭窗口时自动结束该窗口创建的模型服务进程。

GUI 不直接导入 PyTorch。`model_service.py` 使用项目内训练 `.venv` 作为常驻后台
进程，模型只加载一次，再通过逐行 JSON 与人机窗口交换局面和动作。因此原 GUI
环境、训练环境以及 `train_selfplay.py` 入口都不需要合并或改动。

## 当前规则细节

- `O_PLAYER = 1`，`X_PLAYER = -1`，空格为 `0`，O 先手。
- 单个中格内部仍以三个小格连线决定归属。
- 中格一旦归属 O 或 X，之后在该中格继续落子不会覆盖结果。
- 三个中格横向、纵向或斜向连线不会结束整局。
- 任一方赢得至少 5 个中格时立即获胜。
- 填满但无人形成三连的中格记为和棋中格，不属于任何一方。
- 只有 81 个小格全部填满且双方均未取得 5 个中格时，整局才是平局。
- 被指定的下一中格若已经填满，下一位玩家可以选择任意空格。

## AI 状态与动作

每个状态是形状为 `(5, 9, 9)` 的 `float32` 数组：

1. 当前玩家自己的棋子；
2. 对手棋子；
3. 中格结果扩展到对应的 `3 × 3` 小格区域；
4. 当前强制进入的中格；
5. 当前玩家标记。

动作编号为 `action = row * 9 + col`，范围是 `0..80`。训练和推理都会使用
`GameEngine.available_moves()` 生成合法动作掩码，非法位置不会被采样。

当前网络为共享的 Actor-Critic CNN：

- Actor 输出 81 个动作的 logits。
- Critic 从当前玩家视角预测终局结果，输出范围约为 `[-1, 1]`。
- 每局结束后，获胜方动作的目标为 `+1`，失败方为 `-1`，平局为 `0`。
- 策略损失使用终局目标减去价值预测作为优势，价值损失拟合终局目标。

## 训练当前五格多数制模型

旧 `checkpoints` 和 `checkpoints_ac` 模型学习的是旧规则或旧网络结构，不能用其
历史分数与当前五格多数制模型直接比较。当前训练统一使用
`checkpoints_majority`。

### 首次小规模验证

首次训练不要添加 `--resume`：

```powershell
& .\.venv\Scripts\Activate.ps1
python train_selfplay.py --episodes 20 --save-every 10 --eval-every 20 --eval-games 20 --log-every 1 --device cuda --checkpoint-dir checkpoints_majority
```

### 正式训练

```powershell
python train_selfplay.py --episodes 500 --device cuda --checkpoint-dir checkpoints_majority
```

### 继续训练

```powershell
python train_selfplay.py --episodes 500 --resume --device cuda --checkpoint-dir checkpoints_majority --eval-every 100 --eval-games 40
```

`--episodes` 表示本次新增的训练局数，不是最终累计局数。默认每 25 局保存一次；
训练会更新：

- `policy_latest.pt`：最新训练状态，用于 `--resume`，保存时会覆盖。
- `policy_best.pt`：仅当随机合法策略评估分数严格高于历史最佳时覆盖。

评估分数计算为：

```text
(胜场 + 0.5 × 平局) / 总局数
```

评估时模型尽量均分 O/X 座位，并采用贪心落子。

## 模型评估

使用当前多数制最佳模型对战随机合法策略：

```powershell
& .\.venv\Scripts\Activate.ps1
python evaluate_vs_random.py --checkpoint checkpoints_majority/policy_best.pt --games 100 --greedy
```

- `--greedy` 表示每回合选择 logits 最高的合法动作。
- 不加 `--greedy` 时按策略概率采样。
- 结果会分别统计模型执 O 和执 X 时的胜、平、负。

随机合法策略只是初级基准。少量评估会有明显波动，`policy_best.pt` 只表示在当前
随机基准和有限样本下的历史最高分，并不等于绝对最强模型。更可靠的比较需要固定
随机种子、更多对局、历史模型对手池以及先后手分开统计。

## 测试与检查

规则和 AI 环境测试应在项目内训练环境执行：

```powershell
& .\.venv\Scripts\Activate.ps1
python -m unittest -v test_game_engine.py test_ai_environment.py
```

GUI 文件语法检查可使用上一级 GUI 环境：

```powershell
& ..\.venv\Scripts\Activate.ps1
python -m py_compile game_engine.py main.py human_vs_ai.py play_vs_ai.py model_service.py
```

## 主要改进记录

### 2026-10-22：规则引擎与双人窗口

- 修复 `has_three` 静态方法参数错误，避免首次落子抛出 `TypeError`。
- 完善强制中格、已结束中格继续落子、目标中格填满后解除限制等规则。
- 增加 `available_moves()`、非法落子保护、完整状态悔棋和回归测试。
- 修复棋盘外负坐标被误判为第一格的问题。
- 增加强制中格高亮、当前玩家提示、重新开始和悔棋按钮状态。

### AI 管线演进

- 建立五通道强化学习环境和 81 动作合法掩码。
- 早期使用 REINFORCE；因终局奖励稀疏、没有单独价值估计，升级为 Actor-Critic。
- 增加随机合法策略评估，以及 `policy_latest.pt` / `policy_best.pt` 分离保存机制。
- 当前实现是 Monte Carlo Actor-Critic，不是 AlphaZero，也没有 MCTS、经验回放或
  对手池。

### 2026-10-29：五格多数制

- 整局胜利从“三个中格连线”改为“累计赢得至少 5 个中格”。
- 图形界面显示双方已经赢得的中格数量。
- 规则引擎、AI 终局奖励和测试同步更新。
- 新规则模型使用独立的 `checkpoints_majority`，避免与旧规则模型混用。

### 2026-10-29：本地模型人机对弈

- 原双人界面和训练入口保持不变。
- 增加一键打开的人机窗口、先后手选择和整回合撤销。
- 增加常驻模型服务，在分离的 GUI/训练虚拟环境之间完成本地推理。
- 默认优先加载多数制最佳模型，缺失时回退到最新模型。

---

Codex 协作
