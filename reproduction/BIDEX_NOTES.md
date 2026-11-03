# Bi-DexHands：实验性重建接口

**不是 DIMA 作者发布的 Bi-DexHands 代码，尚未在 Isaac Gym 上运行验证。**
作者公开仓库未提供这部分入口。因此这些文件是明确标注的工程重建，不能用来宣称已经精准复现论文曲线。

## 运行

在 Linux + NVIDIA GPU 的独立环境中先安装 Bi-DexHands 所要求的 Isaac Gym Preview 4、PyTorch 和对应 CUDA，再安装本仓库模型依赖。Isaac Gym 需自行从 NVIDIA 获得；不要将其替换为 Isaac Sim，二者 API 不同。先依照 Bi-DexHands README 跑通原仓库的 MAPPO 任务，再测试本入口。旧版 Isaac Gym 与新版模型依赖的组合仍需在目标机器上验证。

```bash
# 在 reproduction 目录执行；路径是示例，请改为实际克隆目录
python train_bidex.py --bidex-root /path/to/DexterousHands \
  --task ShadowHandPen --seed 1 --steps 1000 \
  --run-dir runs/bidex_pen_smoke

# 冒烟测试通过之后，完整预算：
python train_bidex.py --bidex-root /path/to/DexterousHands \
  --task ShadowHandPen --seed 1 --steps 300000 \
  --run-dir runs/bidex_pen_seed1
```

四个任务：`ShadowHandPen`、`ShadowHandDoorOpenOutward`、`ShadowHandDoorOpenInward`、`ShadowHandBottleCap`。各任务使用多个独立 seed；论文主结果与消融使用的 seed 数应以主复现说明中的证据表为准。`--render` 开启窗口。高维状态模型训练很慢，1000 步仅检验真实环境采集，尚未到继承的 5000 步 warm-up 门槛，不代表训练链路已验证。

`training.csv` 保存训练回报和计数，`evaluations.csv` 保存每 5000 个训练模拟器调用后的 10 回合确定性评估，`ckpt` 保存模型。评估周期/回合数是明确的重建选择。评估与训练共享物理环境、使用新的完整回合，但不把评估数据加入训练；它们共享 RNG 流，因此更改评估频率会改变之后训练采样序列。

## 接口与边界

- 强制一个物理环境、两个完整手智能体。观测 `[1,2,O]` 保持局部信息；全局状态使用上游返回的 `[1,2,S]`，两行相同，不拼接局部观测充当状态。动作以两个 `[1,A]` 张量传给上游，保存实际裁剪后的动作。
- 终止时先复制 terminal 状态，随后才 reset；不把 reset 后的状态当作上一回合下一状态。预算截断仅设置 `last`，不伪造 `done`。
- 上游 `reset()` 本身执行一次模拟动作，且不主动设置 `reset_buf`。本适配器先设置该标志，再调用 reset，并单独计数。`--steps 300000` 限制**训练模拟器调用总数（包括 reset 开销）**，训练数据数略少于 300000；日志同时报告 `policy_transitions`。评估调用另计。这个保守计数口径与论文未充分披露的 reset 计数规则可能不同。
- episode 完整采集后交给 DIMA learner；`N_SAMPLES=500` 是更新触发下限，长回合可能超过 500。为了严格每 500 步训练且保持上下文，需要进一步改写作者按回合回放的数据接口；此处没有悄悄截断回合冒充严格间隔。
- 采用 FSQ、γ=0.95、熵系数 0.01、4 次策略学习、二分类终止、state-only critic；其余共用连续控制配置。默认 Gaussian policy、单环境采集和预热长度为重建选择，不能视为作者 BiDex 配置。`--policy beta` 可用于敏感性对照。
- `ENV_TYPE=MAMUJOCO` 仅复用作者 learner 的连续动作分支，物理环境是 Bi-DexHands。保存文件包含这种实现映射。

## 已核查的一手接口

2026-10-30 核查下列 upstream main 源码（应在正式运行时保存实际 checkout commit）：

- [MultiVecTaskPython](https://github.com/PKU-MARL/DexterousHands/blob/main/bidexhands/tasks/hand_base/multi_vec_task.py)：step/reset 的张量布局、reset 的模拟调用。
- [parse_task](https://github.com/PKU-MARL/DexterousHands/blob/main/bidexhands/utils/parse_task.py)：MultiAgent task 构建接口。
- [config](https://github.com/PKU-MARL/DexterousHands/blob/main/bidexhands/utils/config.py)：get_args/load_cfg/parse_sim_params。
- [process_marl](https://github.com/PKU-MARL/DexterousHands/blob/main/bidexhands/utils/process_marl.py)：handAgentIndex 的左右手划分。

本地契约测试：`python -m unittest discover -s tests -p test_bidex_adapter.py`。它验证动作布局、计数、终止数据复制和错误状态拒绝；不测试物理仿真、数值收敛或论文分数。
