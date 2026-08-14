# 项目启动与实验使用说明（Quick Start）

> 本说明覆盖整个 LIBERO + SmolVLA 基准实验的完整流程：环境搭建、PC-Local 基线、
> PC 模拟 + Jetson 远程推理，以及结果解读。
>
> 当前唯一受支持的 benchmark 入口是 **WSL/Linux 中的官方 `lerobot-eval`**。
> 旧的自研 YAML runner 已退役，仅保留历史证据（`outputs/`）。

## 1. 项目概览

目标：在 LIBERO Spatial 上评估 `HuggingFaceVLA/smolvla_libero` 策略，对比两种部署方式：

| 部署方式 | 模拟器位置 | 策略推理位置 | 当前基线 |
|---|---|---|---|
| PC-Local | WSL Ubuntu | WSL 本地 GPU | 8/10 (80%) |
| PC 模拟 + Jetson 推理 | WSL Ubuntu | Jetson Orin Nano（HTTP） | 9/10 (90%) |

架构分工：

- **WSL Ubuntu**：拥有 LIBERO 模拟器、MuJoCo、rollout 执行、成功判定、视频录制和指标聚合。
- **Jetson Orin Nano**：只做 SmolVLA 模型推理，通过 HTTP 提供服务。
- `remote_jetson`：LeRobot 的薄策略插件（`plugins/lerobot_policy_remote_jetson/`），
  只负责把观察发到 Jetson 并取回动作，不实现第二套 benchmark 循环。

固定实验身份（所有脚本共用的不可变参数）：

```text
CHECKPOINT     = HuggingFaceVLA/smolvla_libero
MODEL_REVISION = 6721902bc4d61e50a3bfdb11dfb4cb626f05d102
PRECISION      = fp16
SUITE          = libero_spatial（10 个任务，每任务 1 集）
EPISODE_LENGTH = 280 步（官方上限，含提前成功终止）
MUJOCO_GL      = egl
```

## 2. 前置条件

### 硬件

- 一台带 NVIDIA GPU 的 PC（跑 WSL Ubuntu，负责模拟 + PC-Local 推理）
- 一台 Jetson Orin Nano（`10.42.0.2`，负责远程推理），PC 与 Jetson 需在同一网络

### 软件

- WSL2 Ubuntu（Python 3.12）
- Jetson：Docker + NVIDIA Container Toolkit（`--runtime nvidia`）
- 官方数据集：LIBERO Spatial（由 `lerobot-eval` 通过 `lerobot/libero-assets` 自动加载，
  首次运行需要网络，之后缓存在 `HF_HOME`）

## 3. 一次性环境搭建

### 3.1 WSL Ubuntu（模拟与评估侧）

```bash
# 1. 创建 venv
python3.12 -m venv ~/vla/lerobot-libero
source ~/vla/lerobot-libero/bin/activate

# 2. 安装 LeRobot v0.6.1（libero + smolvla + evaluation 全套）
git clone --branch v0.6.1 --depth 1 https://github.com/huggingface/lerobot.git ~/vla/lerobot
cd ~/vla/lerobot
python -m pip install -e '.[libero,smolvla,evaluation]'

# 3. 约定缓存目录（模型 + 数据集都放这里，方便离线复用）
export HF_HOME=~/vla/hf-cache

# 4. 克隆/同步本项目代码
#    Windows 侧代码位于 D:\Bristol_IOT_with_AI\Capstone Project\.worktrees\libero-yaml-cli-v1\
#    WSL 中通过 /mnt/d/... 直接访问，无需复制
```

验证安装（可选）：

```bash
python -c "import lerobot; print(lerobot.__version__)"          # 0.6.1
python -c "import libero; print(libero.__file__)"                # 能 import 即可
```

### 3.2 Jetson（推理侧）

```bash
# 1. 把项目代码放到 Jetson 家目录（与 WSL 使用同一份代码）
mkdir -p ~/vla/project
#   将 LIBERO_Benchmark_Platform 内容同步到 ~/vla/project

# 2. 构建 Docker 镜像（首次，约需要几分钟）
cd ~/vla/project
docker build -f docker/jetson/Dockerfile -t libero-smolvla:jetson-0.1 .

# 3. 准备模型缓存目录（离线模式使用）
mkdir -p ~/vla/hf-cache ~/vla/outputs
```

> 首次需要在线下载模型时，用 `bootstrap` 模式启动服务（见 4.2），
> 模型会缓存到 `~/vla/hf-cache`，之后可用 `offline` 模式离线运行。

## 4. 标准实验流程

### 4.1 实验 A：PC-Local 基线

在 **WSL Ubuntu** 终端：

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="/mnt/d/Bristol_IOT_with_AI/Capstone Project/.worktrees/libero-yaml-cli-v1/Final_Project/LIBERO_Benchmark_Platform"
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102
cd "$PROJECT_ROOT"

# 每任务 1 集（默认）；如需更多可改为 N_EPISODES=5 bash ...
bash scripts/wsl/run_official_pc_local_eval.sh
```

结果写入：`~/vla/results/libero_spatial_pc_local_<UTC时间戳>/`

### 4.2 实验 B：PC 模拟 + Jetson 远程推理

需要 **两个终端**：

**终端 1 —— Jetson（启动策略服务）：**

```bash
cd ~/vla/project
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102

# 首次/需要联网下载模型时用 bootstrap；模型已缓存后可用 offline
./scripts/jetson/start_smolvla_libero_service.sh offline
```

保持该终端打开，看到下面这行即服务就绪：

```text
policy service listening on http://0.0.0.0:8081
```

**终端 2 —— WSL（预检 + 评估）：**

```bash
source ~/vla/lerobot-libero/bin/activate
export PROJECT_ROOT="/mnt/d/Bristol_IOT_with_AI/Capstone Project/.worktrees/libero-yaml-cli-v1/Final_Project/LIBERO_Benchmark_Platform"
cd "$PROJECT_ROOT"

# 首次需要安装 remote_jetson 策略插件
bash scripts/wsl/install_remote_jetson_policy.sh

export JETSON_ENDPOINT=http://10.42.0.2:8081
export MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102

# 预检：确认服务在线、checkpoint/revision/precision 一致
bash scripts/wsl/run_jetson_remote_preflight.sh

# 正式评估（每任务 1 集；可 N_EPISODES=5 增加重复）
N_EPISODES=1 bash scripts/wsl/run_official_jetson_remote_eval.sh
```

结果写入：`~/vla/results/libero_spatial_jetson_remote_<UTC时间戳>/`

> 注意：预检通过后再跑正式评估。Jetson 单步推理约 2.4s，
> 10 任务 × 1 集约需 2 小时，请预留时间。

## 5. 结果解读

每次运行在输出目录下生成：

```text
<output_dir>/
  eval_info.json                  # 汇总指标 + 每任务成功率
  videos/libero_spatial_<task_id>/eval_episode_0.mp4   # 每任务视频
  remote_transport.jsonl          # 仅 Jetson 模式：每步 HTTP round-trip 延迟
```

`eval_info.json` 关键字段：

- `overall.pc_success`：总成功率（如 80.0 = 8/10）
- `per_task[].metrics.successes`：每个任务是否成功
- `per_group.libero_spatial.n_episodes`：评估集数

当前已确认基线（2026-08-13，证据在 `evidence/latest/`）：

- PC-Local：**8/10**，失败任务 task 2、7
- Jetson 远程：**9/10**，失败任务 task 7

## 6. 环境变量速查

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_REVISION` | （必须设置） | 模型 commit，统一 `6721902b...` |
| `CHECKPOINT` | `HuggingFaceVLA/smolvla_libero` | 模型名 |
| `HF_HOME` | `~/vla/hf-cache` | 模型/数据集缓存 |
| `JETSON_ENDPOINT` | （必须设置） | Jetson 服务地址 |
| `N_EPISODES` | `1` | 每任务 episode 数 |
| `OUTPUT_ROOT` | `~/vla/results` | 结果根目录 |
| `MUJOCO_GL` | `egl` | MuJoCo 渲染后端 |
| `JETSON_IMAGE` | `libero-smolvla:jetson-0.1` | Jetson Docker 镜像名 |

## 7. 常见问题

**Q: `run_jetson_remote_preflight.sh` 报 unavailable / mismatch？**
检查 Jetson 服务终端是否仍开着、`JETSON_ENDPOINT` 是否可达，
以及 `MODEL_REVISION`/`CHECKPOINT` 是否与启动服务时一致。

**Q: 首次运行报模型或数据集下载失败？**
确保 WSL 与 Jetson 都能访问 Hugging Face（或已提前缓存到 `HF_HOME`）。
Jetson 离线运行时模型必须已存在于 `~/vla/hf-cache`。

**Q: 想跑更多集数、得到更稳的结果？**
增大 `N_EPISODES` 即可对每个任务重复评估，例如 `N_EPISODES=5 bash scripts/wsl/run_official_pc_local_eval.sh`。
注意 LIBERO 的初始状态由官方环境决定，脚本不做额外采样。

## 8. 相关文档

- `docs/PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md` —— PC-Local 官方协议
- `docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md` —— Jetson 远程推理官方流程
- `evidence/latest/CURRENT_PROGRESS.md` —— 当前实验基线证据
