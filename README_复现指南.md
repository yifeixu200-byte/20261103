# DIMA 代码复现与两个模拟实验

## 当前交付状态

依据用户提供的 32 页论文《Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective》（NeurIPS 2025），阅读全文、核对附录 B–G、表 4/9/10，并检查论文链接的官方实现。

**这是固定版本官方源码 + 明确标注的论文对齐修改 + 实验操作方案。尚未跑完模拟训练，不能宣称已经复现论文回报或逐位等价于作者实验。** 公开源码与论文有实质差异，Bi-DexHands 接口也未包含在作者发布版本中。

目录：

| 路径 | 用途 |
|---|---|
| `official/` | 作者原始 Git 仓库，保持源码不改 |
| `reproduction/` | 可审查的本地适配、paper/official 配置、测试和启动脚本 |
| `reproduction/REPRODUCTION_CHANGES.md` | 每项论文/代码差异及修复 |
| `实验方案与原理.md` | 方法解释、完整实验矩阵、附加分析协议 |
| `paper_sources.json` | PDF SHA256、原始链接及所在页、官方提交号 |
| `paper_extracted.txt` | 全文提取，复杂公式应以原 PDF 为准 |
| `smoke_result_cpu.json`、`smoke_result_cuda.json` | 合成输入模型检查记录，不是实验结果 |
| `preflight_windows.json` | 本机环境检查 |
| `VERIFICATION.md` | 验证范围和剩余限制 |

官方：[breez3young/DIMA](https://github.com/breez3young/DIMA)，固定提交 `3dcacaa80162cf6822bf5972b4e3ad4cb2e6ceb0`。`paper` 是本地按明确论文描述对齐的配置名，不表示作者验证过该配置。

## 先选择运行环境

当前电脑为 Windows、系统 Python 3.8.5、PyTorch 2.4.1+cu121、RTX 3050 Laptop 约 4GB 显存。没有可用的 WSL Linux 环境，也未安装 MuJoCo/Isaac Gym。已经可以跑小规模核心模型测试；完整实验建议用 Linux NVIDIA GPU 机器。论文硬件为 RTX 4090、24 核 CPU、256GB 内存；Ant 单个种子报告约 1 天 19 小时，这不是本机耗时承诺。

MAMuJoCo 使用旧 `gym==0.21.0` / `mujoco-py` / MuJoCo 2.1 路径。不要直接用 Gymnasium、MuJoCo 3 或 `HalfCheetah-v5` 替换，会改变基准。Bi-DexHands 使用旧 Isaac Gym，不能把 Isaac Lab 或 Isaac Sim 当作无差别替换。

可在 Linux 服务器复制整个文件夹；若自行安装好 WSL，Windows 目录对应 `/mnt/d/Files/pythonexcel/DIMA`。大规模训练建议复制到 Linux 本地磁盘，避免跨文件系统开销。WSL 运行 Bi-DexHands 本次未验证。

## 实验一：MAMuJoCo

### 安装（Linux，独立环境）

下列为整理后的兼容性安装方案，**尚未在本机完整安装验证**。作者的 `official/environment.yml` 同时包含 Linux 固定 build、CUDA wheel 后缀和大量无关环境包，不能直接在 Windows 上照搬。

```bash
conda create -n dima-repro python=3.10 -y
conda activate dima-repro
cd /path/to/DIMA/reproduction
python -m pip install 'pip==23.0.1' 'setuptools==65.5.0' 'wheel==0.38.4'
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.23.5
python -m pip install --no-build-isolation gym==0.21.0
python -m pip install -r requirements-mamujoco.txt
```

兼容性环境使用 NumPy 1.23.5；作者导出文件为 2.1.3。这是为了旧 Gym/mujoco-py 的兼容性选择，须写入实验记录，不能称为完全相同的软件环境。

从 [MuJoCo 2.1.0 官方发布](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)取得 Linux 包，解压到 `~/.mujoco/mujoco210`。可能还需要 Linux 系统的编译工具、`libosmesa6-dev`、`libgl1-mesa-dev`、`libglfw3` 和 `patchelf`。参考 [mujoco-py 官方说明](https://github.com/openai/mujoco-py)。示例环境变量：

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python preflight.py
python smoke_test.py --device cuda
python -c "from configs.EnvConfigs import MAMujocoConfig; e=MAMujocoConfig('HalfCheetah-v2',123,'2x3').create_env(); print(e.n_agents,e.n_obs,e.state_dim,e.n_actions); print(e.reset()[0]); e.close()"
```

EGL 是否可用取决于 NVIDIA 驱动/系统 OpenGL 配置；先通过上述环境 reset，再启动长训练。若仅调试模型，无需安装模拟器。

### 先跑单任务

```bash
python train.py --profile paper --env mamujoco \
  --env_name HalfCheetah-v2 --agent_conf 2x3 \
  --n_workers 1 --policy_class gaussian --seed 1 \
  --steps 1000000 --sample_temp 20 --mode disabled \
  --state_decoder_type 1 --use_tensorboard
```

`--seed 1` 在作者入口中映射为 `23+100*1=123`，1/2/3/4 对应 123/223/323/423。论文未给出具体四个种子值，这组是我们选取的可重复实例。`--steps 5000` 只能检查环境采集和最初世界模型训练；作者直到第 10 次训练周期才开始策略想象更新，不能据此判断学习成功。

若要对比官方默认超参数，在 `reproduction/` 使用 `--profile official`；若需要一字不改的源码，则进入 `official/` 按其 README 和 train.sh 执行（没有 `--profile` 参数），也可能遇到它的原始依赖问题。

### 七个任务、四个种子

```bash
python run_mamujoco_matrix.py              # 只打印 28 条命令
python run_mamujoco_matrix.py --execute    # 顺序运行 28 个长实验
```

训练输出位于 `MMDD_results/mamujoco/<任务-划分>/runN/`；权重在 `ckpt/`，本地新增 `evaluations.csv` 逐次记录评估。TensorBoard 位于 `tb_logs/`。不用 W&B 账号也能运行（`--mode disabled`）。记录实际完成步数，因为原始采集器按 episode 边界停止，可能超过名义预算。

### 单独测试已训练 Actor

```bash
python evaluate_mamujoco.py \
  --checkpoint MMDD_results/mamujoco/HalfCheetah-v2-2x3/run1/ckpt/model_final.pth \
  --env_name HalfCheetah-v2 --agent_conf 2x3 --episodes 10 \
  --output actor_evaluation --save-trajectories
```

该入口只使用 Actor 和真实环境观测，不调用扩散世界模型；每个 episode 使用不同固定环境种子。轨迹保存为 NPZ，可用于后续预测误差分析。它已做源码检查，尚未经过真实 MuJoCo 验证。仅加载自己产生或可信来源的 PyTorch checkpoint。

将**同一个任务、同一个 profile、四个独立训练种子**的 CSV 传给：

```bash
python summarize_evaluations.py /path/seed1/evaluations.csv /path/seed2/evaluations.csv /path/seed3/evaluations.csv /path/seed4/evaluations.csv
```

输出最终均值与跨种子样本标准差（ddof=1，明确约定）。10 次 episode 先求单种子均值，再在 4 个训练种子间统计；不要把 40 个 episode 当 40 个独立训练重复。

## 实验二：Bi-DexHands

作者公开 DIMA 仓库未提供此环境的入口、配置或权重，因此不能把一个自写接口称为作者原始代码。本地 `train_bidex.py` / `bidex_adapter.py` 是依据 [Bi-DexHands 官方环境](https://github.com/PKU-MARL/DexterousHands)补写的实验性适配，必须先完成真模拟器联调。详细安装、四个任务命令及状态/重置的限制见 `reproduction/BIDEX_NOTES.md`。

基本顺序：在兼容 Linux 环境按 Bi-DexHands 文档安装 NVIDIA Isaac Gym，先运行其官方示例；再检查本地适配的 `--help` 和 `--env-smoke`，确认全局状态、两个局部观测、动作维度、共享奖励与终止；最后执行 DIMA 的 300000 步训练与 10 episode 评估。**运行 Bi-DexHands 自带 MAPPO 只能证明模拟器可运行，不能证明 DIMA 已复现。**

四个任务为 `ShadowHandPen`、`ShadowHandDoorOpenOutward`、`ShadowHandDoorOpenInward`、`ShadowHandBottleCap`。`gamma=0.95`、熵系数 0.01、想象长度 15；不要沿用 MAMuJoCo 的 0.99/0.001。必须将真实模拟器并行环境数乘进 transition 计数，智能体数不乘；重置时隐含的模拟器步数单独记录。

## 精准复现仍需核实的条件

代码结构和测试通过不等于论文数值复现。还需：完整仿真依赖安装、单任务闭环、至少四种子实验、同预算评估，并解决公开代码与附录的冲突。我们没有作者原始训练日志/权重、Bi-DexHands 实验配置、精确种子及所有未报告超参数。建议向作者确认这些细节，尤其 FSQ/VQ、critic 输入与损失、offset noise、Bi-DexHands observation/state 配置。不要通过反复挑种子来匹配表格。

本次使用 ARS-Codex 的 `experiment-agent` / reproducibility 流程记录来源、参数、测试与限制。验证状态为 `CANNOT_VERIFY`（论文统计结果尚未重新运行），不是 `REPRODUCIBLE`。
