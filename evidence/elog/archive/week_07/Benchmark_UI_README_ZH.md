# VLA 可视化实验控制台操作说明

本目录提供一个面向操作员的三阶段视觉实验控制台，用于在 PC 上配置、运行和复查 VLA/VLM 评估或轻量训练实验。它的目标不是绕过命令行能力，而是把可重复的实验规格、运行状态、结果完整性和视觉证据放在同一个入口中。

## 启动

在 PowerShell 中启动后端：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Benchmark_UI"
python server.py --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

常用启动参数：

- `--host`：监听地址。日常本机使用建议保持 `127.0.0.1`。
- `--port`：监听端口。默认工作流使用 `8765`，端口被占用时可换成其他未使用端口。

常用环境变量和运行时覆盖：

- `PYTHONUTF8=1`：建议在 Windows 终端中设置，避免中文日志或路径输出乱码。
- `MUJOCO_GL`：会被仿真脚本继承；未设置时仿真公共模块默认使用 Windows 的 `wgl`。
- `HF_HOME`、`TRANSFORMERS_CACHE`、`HF_HUB_OFFLINE`：会被模型执行进程继承，可用于控制 Hugging Face 缓存或离线加载行为。
- 执行器 Python 解释器不是通过环境变量选择的。控制台会按白名单使用项目内解释器：`Local_VLA_Benchmark_Framework/.venv/Scripts/python.exe` 和 `Robosuite_MuJoCo_Sim/.sim_venv/Scripts/python.exe`。如果需要换运行时，请更新对应虚拟环境，而不是从界面传入任意命令。

## 三阶段工作流

### 1. Setup

Setup 阶段用于生成可审计的实验规格。

你可以选择：

- 实验模式：`Evaluation` 或 `Training`。
- 部署配置：当前可用主线是 `PC Local`；Jetson 相关配置会显示为不可用，直到硬件和运行时被接入并验证。
- 环境：`Robosuite Pick Place` 或 `Offline Image Tasks`。
- 任务、trial 数、warmup、录帧选项、模型单选/批量选择。
- 训练模式下的 trainer 和 dataset。

建议先点击 `Validate configuration`。验证成功后，控制台会展示 executor、模型数量、trial 数、Viewer 可用性和输出根目录。只有验证通过后，`Enter Run & Observe` 才会进入 Run 阶段并启动白名单执行器。

### 2. Run

Run 阶段用于观察当前运行，而不是修改已经启动的实验规格。

页面会显示：

- run id、状态、阶段和耗时。
- 当前模型、任务、trial、step。
- 实时进度、执行器上报的指标、artifact 数量。
- 最新 recorded frame 预览。
- 日志尾部。
- 对可渲染 Robosuite 运行提供 MuJoCo Viewer 控制。

`Stop run` 会请求停止当前仍在运行的任务。终态包括 `completed`、`failed` 和 `stopped`。运行到终态后，控制台会加载 Results 阶段可读的结果索引。

### 3. Results

Results 阶段用于复查证据，不用于美化或补造结果。

页面会检查：

- `metadata.json`、`summary.csv`、`trials.csv` 等必须 artifact。
- `result_integrity` 是否为 `complete`、`partial` 或 `unavailable`。
- 批量评估的模型对比表。
- 单个 trial 的指令、目标、动作、成功/失败、延迟。
- recorded frames 回放。

训练运行完成后，如果输出了可识别的策略 artifact，Results 阶段可以把该策略带回 Setup，配置一次新的 validation rollout。只有后续评估 rollout 的结果，才能作为机器人任务成功率或泛化能力证据。

## 状态目录和结果目录

控制台自己的状态根目录：

```text
Final_Project/Benchmark_UI/state/
```

每次实验运行的审计入口位于 `Final_Project/Benchmark_UI/state/runs/<run_id>/`。

关键子目录：

- `state/tasks/`：通过界面保存的任务版本。
- `state/runs/<run_id>/spec.json`：启动时冻结的实验规格。
- `state/runs/<run_id>/manifest.json`：运行状态、进度、artifact 索引和 Viewer 状态。
- `state/runs/<run_id>/events.jsonl`：运行事件。
- `state/runs/<run_id>/job.log`：执行器标准输出和错误输出。
- `state/runs/<run_id>/viewer-control.json`、`viewer-status.json`：原生 Viewer 桥接文件，仅可渲染运行会使用。

执行器的原始结果仍写在各自项目目录下，并由控制台索引：

- 离线 VLA/VLM 评估：`Final_Project/Local_VLA_Benchmark_Framework/results/`
- Robosuite 评估、训练和数据集：`Final_Project/Robosuite_MuJoCo_Sim/outputs/`
- 旧 PC benchmark matrix：`Final_Project/Robosuite_MuJoCo_Sim/outputs/pc_benchmark_matrix/`

操作员复查时优先看 `state/runs/<run_id>/manifest.json` 里的 artifact 索引，再打开被索引的原始结果文件。

## MuJoCo Viewer 与 recorded frames

MuJoCo Viewer 和 recorded frames 是两类不同证据：

- `MuJoCo Viewer` 是原生交互窗口，只适用于正在运行、可渲染、带 Viewer 桥接文件的 Robosuite rollout。离线图像任务和训练任务没有原生 Viewer。
- `Recorded frames` 是执行器保存下来的帧图像，可在 Results 阶段复查。它们适合报告和复盘，也适合在运行结束后查看。

如果 Run 阶段显示 `Viewer unavailable for this run`，这不是错误本身，通常表示该 run 类型不支持原生 Viewer、run 已经结束、或者执行器没有轮询 Viewer 桥接文件。此时应使用 Results 阶段的 recorded frames 作为视觉证据。

## Jetson 不可用状态

当前 catalog 中会保留 Jetson 相关部署档位，但默认不可用：

- `Jetson Local`：Jetson 未连接。
- `Jetson Quantized`：Jetson runtime 尚未检查。
- `Remote Model + Jetson Client`：Jetson client 未连接。

这些状态是故意保守的。文档、截图和汇报中不要把不可用档位描述成已经完成上板验证。只有当硬件、依赖、模型权重和端到端运行证据全部存在后，才能把 Jetson 结果作为真实部署结果。

## 训练指标诚实规则

训练指标只说明训练过程本身，例如 loss、验证集误差或 checkpoint 产出。它们不能直接等同于机器人任务成功率。

汇报时请遵守：

- 训练 run 的 loss 或 validation metric 只能称为训练/验证指标。
- 机器人成功率、目标匹配率、失败类型和延迟必须来自 evaluation rollout。
- 如果策略训练后没有再跑 validation rollout，应写明“仅完成训练，尚未完成闭环评估”。
- 如果结果完整性不是 `complete`，不要把表格中的数字当作最终结论。

## 静态检查

文档和静态控制台结构可以用下面两个命令快速检查：

```powershell
python verify_dashboard_static_mvp.py
python verify_mujoco_viewer_debug_static.py
```

它们只做便宜、确定性的文本检查，不会启动模型、仿真器或 Jetson 硬件。
