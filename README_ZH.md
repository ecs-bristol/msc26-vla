# LIBERO 基准平台

> 当前唯一支持的模型基准入口是 WSL/Linux 中的官方 `lerobot-eval`。
> PC-local 与 PC 仿真 + Jetson 远程推理分别见
> `docs/PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md` 和
> `docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md`。下方自研 YAML runner 说明仅用于
> 历史回放诊断，不再用于生成模型 benchmark 结果。

本平台使用 YAML 配置运行可复现的 LIBERO 评测，并保存逐步动作、成功率、延迟和运行清单。`oracle_or_scripted` 是官方示范回放基线，不是零动作策略。

## YZ：LIBERO Object 任务感知动作执行

YZ 的工作是在不重新训练、也不修改 `HuggingFaceVLA/smolvla_libero`
模型结构的前提下，比较部署阶段的动作执行策略：

- **Native**：每执行一个动作后重新规划；
- **Smooth**：连续执行十个预测动作后重新规划；
- **Task-aware router**：在 rollout 开始前，根据去重后的历史任务结果选择
  Native 或 Smooth。构造历史数据时排除当前评估 seed，并使用 Laplace
  smoothing 估计成功率，从而控制数据泄漏。

在 seed 44、十个 LIBERO Object 任务、每个任务三次 rollout 的 matched-seed
实验中，结果如下：

| 策略 | 成功次数 | 平均每次 rollout 时间 |
|---|---:|---:|
| Native | 23/30（76.67%） | 153.73 秒 |
| Smooth | 24/30（80.00%） | 36.77 秒 |
| Task-aware router | 25/30（83.33%） | 119.00 秒 |

Router 的成功率点估计最高，并且比 Native 快 22.6%；但三种策略的 Wilson
95% 置信区间重叠，因此不能声称 Router 在统计上显著优于固定策略。该结果
验证的是无数据泄漏的任务级选择流程能够正常工作。失败恢复和基于
Grounding DINO Tiny 的 RGB 目标验证属于探索性原型，不计入正式性能结论。

## 获取官方数据集

在 LIBERO 仓库根目录运行：

```bash
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface
python benchmark_scripts/download_libero_datasets.py --datasets libero_object --use-huggingface
python benchmark_scripts/download_libero_datasets.py --datasets libero_goal --use-huggingface
```

将下载后的套件目录复制、软链接或 junction 到本项目的 `datasets/` 下，最终目录必须为：

```text
datasets/
  libero_spatial/
  libero_object/
  libero_goal/
```

Linux/WSL 示例：

```bash
ln -s /absolute/path/to/libero_spatial datasets/libero_spatial
ln -s /absolute/path/to/libero_object datasets/libero_object
ln -s /absolute/path/to/libero_goal datasets/libero_goal
```

Windows PowerShell junction 示例：

```powershell
New-Item -ItemType Junction -Path datasets\libero_spatial -Target C:\datasets\libero_spatial
New-Item -ItemType Junction -Path datasets\libero_object -Target C:\datasets\libero_object
New-Item -ItemType Junction -Path datasets\libero_goal -Target C:\datasets\libero_goal
```

## 示范回放规则

每个 episode 严格使用配置指定的任务和初始状态：

```text
<dataset_directory>/<task_name>_demo.hdf5
  data/demo_<initial_state_id>/states[0]
  data/demo_<initial_state_id>/actions
```

平台不会随机选择 demo，也不会由策略修改环境状态。预检会逐一打开所有选中的 HDF5，确认 `states` 和 `actions` 存在，确认 `actions` 的形状为 `(N, 7)`、所有值有限且位于 `[-1, 1]`，并确认 `LiberoBackend` 使用的首状态与同一 demo 的 `states[0]` 完全一致。

回放策略每次预测只输出下一条官方动作。若动作耗尽时环境仍未成功，episode 以 `reference_actions_exhausted` 失败保留为诊断证据，不会补零动作。真实回放未达到 LIBERO success 时，应将该 trial 视为 `reference_replay_failed`，集成门禁不能通过。`zero_policy` 仅用于管线 smoke test，任何零动作结果都不能标记为 oracle。

## 验证与运行

安装项目基础依赖后，可先验证配置：

```bash
python -m libero_platform validate configs/experiments/libero_spatial_oracle.yaml
```

旧 YAML 配置现在只保留验证和历史诊断用途，不再提供模型 benchmark
运行入口。正式 PC-local 与 Jetson 远程实验统一使用上方链接文档中的
官方 `lerobot-eval` 脚本；旧结果继续保留在 `outputs/libero_runs/` 作为溯源证据。

