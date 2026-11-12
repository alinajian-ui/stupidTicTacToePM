# TicTacToePM

## 神奇声明

模型很会提供情绪价值。肯定有什么地方出问题了。
在调整权限后显著改善了训练稳定性和评估分数，但仍然不保证模型是最优的。请不要把它当作最终解决方案。

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
| `train_selfplay.py` | Actor-Critic 自我对弈训练入口，保存最新模型和最佳评估模型；默认保存到 `checkpoint_prevent`。 |
| `evaluate_vs_random.py` | 让指定模型分别执 O/X 与随机合法策略对战。 |
| `evaluate_model.py` | 加载 `checkpoints_award` 最佳/最新模型并检查推荐动作。 |
| `test_game_engine.py` | 规则引擎回归测试。 |
| `test_ai_environment.py` | AI 状态、动作掩码、奖励和终局测试。 |
| `checkpoint_prevent/` | 新模型目录：原有奖励 + 阻止对方占领中格奖励，训练后包含 latest 和 best。 |
| `checkpoints_award/` | 上一版奖励模型（中心格 + 中格奖励，无阻止奖励）；评估脚本目前仍默认读取这里。 |
| `checkpoints_majority/` | 加入局部奖励前的五格多数制模型，仅保留作历史记录。 |
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

1. 加载 `checkpoint_prevent/policy_best.pt`（含阻止奖励的新模型）；该文件不存在时
   会提示先训练。
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
- 落在任一中格中心小格默认奖励 `+0.02`，首次赢得一个中格默认奖励 `+0.15`，
  挡住对方即将赢下的中格默认奖励 `+0.2`（新增）。
- 策略损失使用“终局目标减去价值预测，再加当前动作塑形奖励”作为优势；价值损失
  仍只拟合终局目标。

## 训练当前五格多数制模型

旧 `checkpoints` 和 `checkpoints_ac` 模型学习的是旧规则或旧网络结构，
`checkpoints_majority` 则没有中心小格/中格胜利奖励；这些历史分数都不应与当前
奖励塑形模型直接比较。当前训练默认使用 `checkpoint_prevent`（含阻止奖励，见下文新模型章节）。

### 首次小规模验证

首次训练不要添加 `--resume`：

```powershell
& .\.venv\Scripts\Activate.ps1
python train_selfplay.py --episodes 20 --save-every 10 --eval-every 20 --eval-games 20 --log-every 1 --device cuda
```

### 正式训练

```powershell
python train_selfplay.py --episodes 500 --device cuda
```

### 继续训练

```powershell
python train_selfplay.py --episodes 500 --resume --device cuda --eval-every 100 --eval-games 40
```

`--episodes` 表示本次新增的训练局数，不是最终累计局数。默认每 25 局保存一次；
训练会更新：

- `policy_latest.pt`：最新训练状态，用于 `--resume`，保存时会覆盖。
- `policy_best.pt`：仅当随机合法策略评估分数严格高于历史最佳时覆盖。

奖励默认值可通过命令行调整，例如：

```powershell
```

- `--center-reward`：占领任一中格中心小格的奖励，默认 `0.1`。
- `--mini-board-reward`：本次落子首次赢得一个中格的奖励，默认 `0.2`。
- 三种奖励都会记录在新检查点的 `reward_shaping` 字段中。续训时如果命令行配置与
  检查点记录不同，训练脚本会显示提示，但允许按新配置继续训练。
- `--block-reward`：挡住对方即将赢下的中格的奖励，默认 `0.2`（新增）。
- `--checkpoint-dir` 默认已经是 `checkpoint_prevent`；普通训练和续训无需再指定。

评估分数计算为：

```text
(胜场 + 0.5 × 平局) / 总局数
```

评估时模型尽量均分 O/X 座位，并采用贪心落子。

## 2026-11-05 新模型：阻止对方占领中格奖励（checkpoint_prevent）

在上一版模型中，AI 只因“自己占中心格”和“自己赢中格”获得奖励，没有任何奖励
告诉它要去防守，因此表现不佳。新模型保留原来全部奖励，并增加一项防守奖励。

### 改动内容

- **阻止奖励 `+0.2`**：某个中格还没有分出结果，对方在一条线上已经有两个棋子、
  只差一格就能赢下这个中格；此时我方下在这一格，就算“阻止成功”，这一步得到
  `+0.2`。同一步挡住多条线也只奖励一次。
- **原有奖励不变**：中心小格 `+0.1`、首次赢得中格 `+0.2`、整局获胜 `+1`
  仍然有效。同一步满足多个条件时，奖励相加。
- **新模型目录**：训练默认保存到 `checkpoint_prevent/`，不会覆盖
  `checkpoints_award/` 等旧模型。
- **训练日志**：每行新增“阻止奖励 N 次”，显示这一局里一共挡住了几次。
- **检查点记录**：`reward_shaping` 字段中新增 `block_reward`。
- **人机对弈**：`play_vs_ai.py` 和 `main.py` 打开的人机窗口改为使用
  `checkpoint_prevent/policy_best.pt`（在 `human_vs_ai.py` 中修改）。

改动的文件：

| 文件 | 改动 |
| --- | --- |
| `ai_environment.py` | 新增 `DEFAULT_BLOCK_REWARD = 0.2`、`_blocks_opponent()` 判断函数，`step()` 中加入阻止奖励，`info` 新增 `blocked_mini_board` 和 `block_reward`。 |
| `train_selfplay.py` | 新增 `--block-reward` 参数（默认 `0.2`），`--checkpoint-dir` 默认改为 `checkpoint_prevent`，日志显示阻止次数。 |
| `test_ai_environment.py` | 新增阻止奖励的测试：能挡住时得分、没有威胁时不得分、已结束中格不得分、与中心奖励叠加。 |

所有新增代码旁都有 `【新增】` 或 `【修改】` 注释，方便查找。

### 第一次训练（PowerShell）

第一次训练不要加 `--resume`：

```powershell
cd 项目目录\TicTacToePM
& .\.venv\Scripts\Activate.ps1
python train_selfplay.py --episodes 500 --device cuda --eval-every 100 --eval-games 40
```

训练完成后，`checkpoint_prevent` 中会出现 `policy_latest.pt` 和 `policy_best.pt`。

### 继续训练（resume，PowerShell）

在上一次结果的基础上继续训练，加上 `--resume`：

```powershell
cd 项目目录\TicTacToePM
& .\.venv\Scripts\Activate.ps1
python train_selfplay.py --episodes 500 --resume --device cuda --eval-every 100 --eval-games 40
```

`--episodes 500` 表示“这次再多训练 500 局”。没有显卡时把 `--device cuda` 改成
`--device cpu`。

### 评估新模型

```powershell
python evaluate_vs_random.py --checkpoint checkpoint_prevent/policy_best.pt --games 100 --greedy
```

### 如果还想继续训练旧模型 checkpoints_award

旧模型没有阻止奖励，续训时需要关掉它并指定旧目录：

```powershell
python train_selfplay.py --episodes 500 --resume --device cuda --block-reward 0 --checkpoint-dir checkpoints_award
```

## 模型评估

使用当前多数制最佳模型对战随机合法策略：

```powershell
& .\.venv\Scripts\Activate.ps1
python evaluate_vs_random.py --games 100 --greedy
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

## 当前训练策略总结

当前训练采用共享网络的 Actor-Critic 自我对弈：O 和 X 使用同一个
`ActorCriticCNN`，棋盘始终转换为当前行动玩家的视角，因此网络不需要为先手和
后手分别训练两套参数。

### 状态与动作

- 输入是 `(5, 9, 9)` 状态，包含己方棋子、对手棋子、中格归属、强制中格和
  当前玩家标记。
- Actor 为 81 个小格输出 logits；执行动作前使用 `GameEngine.available_moves()`
  生成掩码，把非法动作屏蔽掉。
- 训练时按带温度参数的策略概率采样动作，以保留探索；人机对弈和基准评估时使用
  贪心动作，即选择合法位置中 logits 最大的一项。

### 自我对弈与奖励

- 每局由当前模型同时扮演 O 和 X，最多进行 81 步。
- 游戏规则由 `GameEngine` 统一提供：赢得第五个中格时终局，棋盘填满且无人达到
  五格时为平局。
- 每次占领任一中格的中心小格，当前动作默认获得 `+0.02`；每次首次赢得一个中格，
  当前动作默认获得 `+0.15`。同一步同时满足两个条件时奖励可叠加。
- 整局结束后仍使用 Monte Carlo 终局目标：获胜方每步为 `+1`，失败方每步为
  `-1`，平局为 `0`；当前没有折扣累计。
- 局部奖励给关键动作更及时的正向反馈，终局奖励继续决定整局策略方向。

### Actor-Critic 更新

- Critic 预测当前玩家最终结果，输出约束在 `[-1, 1]`。
- 优势值为“终局目标减去 Critic 预测，再加该动作的中心/中格塑形奖励”；策略损失
  推动 Actor 增加正优势动作的概率，降低负优势动作的概率。
- Critic 使用 Smooth L1 损失拟合终局目标，总损失为策略损失加 `0.5 ×` 价值损失。
- 塑形奖励不加入 Critic 目标，避免叠加奖励超过 `value_head` 的 `[-1, 1]` 输出范围。
- 默认优化器为 Adam，学习率 `1e-4`；每次更新前将梯度范数裁剪到 `1.0`，降低训练
  过程中的梯度波动。
- 当前实现每完成一局就更新一次网络，没有经验回放、批量历史样本或目标网络。

### 检查点与模型筛选

- `policy_latest.pt` 保存最新模型、优化器、累计局数和统计数据，供 `--resume` 续训。
- 每隔 `--eval-every` 局，模型会分别执 O/X 与随机合法策略进行贪心对战。
- 评估分数为 `(胜场 + 0.5 × 平局) / 总局数`；只有严格超过历史最佳分数时才覆盖
  `policy_best.pt`。
- 奖励塑形版本默认使用独立的 `checkpoints_award`，避免与旧规则、旧网络结构或
  尚未加入局部奖励的 `checkpoints_majority` 检查点混用。

### 当前局限

- 对手始终是同一个正在更新的共享网络，自我对弈分布可能不稳定，也可能遗忘旧策略。
- 当前塑形奖励中心小格、赢得中格和阻止对手占领中格，但还没有奖励长期局面优势。
- “最佳模型”只以随机合法策略为基准，评估局数较少时容易受随机波动影响。
- 当前没有 MCTS、经验回放、历史模型对手池、固定测试局面或 Elo 评分；后续提升重点
  可放在对手池、批量轨迹、奖励塑形和更稳定的评估体系上。

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
- 增加中心小格 `+0.02` 和赢得中格 `+0.15` 的可配置 Actor 塑形奖励。
- 增加随机合法策略评估，以及 `policy_latest.pt` / `policy_best.pt` 分离保存机制。
- 当前实现是 Monte Carlo Actor-Critic，不是 AlphaZero，也没有 MCTS、经验回放或
  对手池。

### 2026-10-29：五格多数制

- 整局胜利从“三个中格连线”改为“累计赢得至少 5 个中格”。
- 图形界面显示双方已经赢得的中格数量。
- 规则引擎、AI 终局奖励和测试同步更新。
- 五格多数制初版模型使用独立的 `checkpoints_majority`，避免与旧规则模型混用。

### 2026-10-29：奖励塑形与独立检查点补丁

- 占领任一中格中心小格时，默认给予当前动作 `+0.02` Actor 奖励。
- 首次赢得一个中格时，默认给予当前动作 `+0.15` Actor 奖励。
- 新奖励版本统一保存到 `checkpoints_award/policy_latest.pt` 和
  `checkpoints_award/policy_best.pt`，不覆盖此前模型。
- 模型评估和人机对弈窗口均改为读取 `checkpoints_award`。

### 2026-10-29：本地模型人机对弈

- 原双人界面和训练入口保持不变。
- 增加一键打开的人机窗口、先后手选择和整回合撤销。
- 增加常驻模型服务，在分离的 GUI/训练虚拟环境之间完成本地推理。
- 默认优先加载多数制最佳模型，缺失时回退到最新模型。

### 2026-11-05: 新增阻止对方占领中格奖励，改良奖励权重。

- 训练默认保存到 `checkpoint_prevent`，不覆盖`checkpoints_award`。人机窗口改为加载 `checkpoint_prevent/policy_best.pt`。
- 改进了奖励权重，由`+0.02`、`+0.15`、`+0.2` 改为 `+0.1`、`+0.2`、`+0.2`，并可通过命令行参数调整。
---

Codex 协作

