# Jetson VLA 项目完整实施工作流

日期：2026-06-26

## 1. 项目最终形态

本项目建议实现为一个分层 benchmark 系统，而不是单次 demo。

```text
任务指令 + 图像/相机观测
        |
        v
输入数据层：静态图片 / robosuite 图像 / Jetson 摄像头图像
        |
        v
策略层：rule-based / VLM text-action / OpenVLA / SmolVLA 或轻量 VLA
        |
        v
动作适配层：文本动作或连续 action -> 机械臂/仿真器可执行命令
        |
        v
执行层：离线 replay / robosuite closed loop / Jetson + 真实或远程 policy
        |
        v
日志层：latency, memory, power, success, failure type, action stability
```

核心研究问题：

> 在 Jetson Orin Nano Super 这类资源受限平台上，VLA/VLA-style policy 以什么方式部署最可行，并且在延迟、内存、控制频率、任务成功率和安全性之间有什么 trade-off？

## 2. 推荐实现路线

整个项目分 6 个阶段推进。每个阶段都应该产出可提交的 evidence：代码、日志、截图、表格或分析图。

### 阶段 A：固定任务集和输入格式

目标：先把 benchmark 的输入固定下来，避免后面每次实验都改 prompt 和图片。

已有基础：

- `Final_Project/Local_VLA_Benchmark_Framework/data/tasks.csv`
- `Final_Project/Local_VLA_Benchmark_Framework/data/sample_images/`

需要完成：

1. 设计 8-12 个任务，覆盖 pick、move、avoid、sort、near/far、left/right。
2. 每个任务固定字段：
   - `task_id`
   - `image`
   - `instruction`
   - `expected_target`
   - `expected_action`
   - `success_criteria`
   - `failure_notes`
3. 同一任务后续在 laptop、Jetson、仿真器里重复使用。

最低交付：

- 一个稳定的 `tasks.csv`
- 每个任务至少一张输入图片
- 一页任务说明图或表格

### 阶段 B：离线 VLM/VLA-like benchmark

目标：先在 laptop/RTX 环境跑通模型推理和日志记录，为 Jetson 对比建立 reference baseline。

已有基础：

- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/benchmark.py`
- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/run_inference.py`
- `Final_Project/Local_VLA_Benchmark_Framework/results/`

推荐模型：

1. `Qwen/Qwen2-VL-2B-Instruct` 或当前已跑通的 Qwen baseline
2. `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
3. OpenVLA probe 作为 stress test

运行方式：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Local_VLA_Benchmark_Framework"
.\scripts\run_pre_jetson_workflow.cmd --list
.\scripts\run_pre_jetson_workflow.cmd --experiment offline_vlm_smoke --dry-run
.\scripts\run_pre_jetson_workflow.cmd --experiment offline_vlm_smoke
```

统一入口说明见：

```text
Final_Project/Local_VLA_Benchmark_Framework/PRE_JETSON_WORKFLOW_ZH.md
```

记录指标：

- 是否成功运行
- 单次推理延迟
- 平均延迟
- p95 延迟
- peak GPU memory
- 输出动作是否符合 action schema
- 输出是否识别正确目标

最低交付：

- 至少两个模型的 benchmark CSV/JSONL
- 一张模型对比表
- 一段关于失败样例的分析

### 阶段 C：OpenVLA / VLA action probe

目标：测试真正 VLA action 输出，而不是只让 VLM 输出文本动作。

已有基础：

- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/openvla_probe.py`
- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/openvla_server.py`
- `Final_Project/Local_VLA_Benchmark_Framework/OPENVLA_PROBE_ZH.md`

推荐步骤：

1. 先运行 `--preflight`，检查依赖和 GPU。
2. 尝试 4-bit OpenVLA 加载。
3. 对同一批任务图片输出连续 action vector。
4. 不急着接真实机械臂，先记录 action 数值、延迟、内存和是否 OOM。

运行方式：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Local_VLA_Benchmark_Framework"
.\scripts\run_openvla_probe.ps1 -Instruction "pick up the red block"
```

如果本地跑不动：

- 记录 OOM 或依赖失败原因。
- 把它作为 negative result。
- 改成远程 OpenVLA service，再让 Jetson 或仿真器通过 HTTP 调用。

最低交付：

- OpenVLA 可行性表：load / infer / OOM / latency / memory
- 典型 action 输出样例
- 本地 vs 远程部署建议

### 阶段 D：robosuite 闭环仿真

目标：把模型输出接进一个可执行环境，形成闭环 benchmark。

已有基础：

- `Final_Project/Robosuite_MuJoCo_Sim/src/robot_sim/qwen_closed_loop_demo.py`
- `Final_Project/Robosuite_MuJoCo_Sim/src/robot_sim/openvla_visual_demo.py`
- `Final_Project/Robosuite_MuJoCo_Sim/src/robot_sim/text_action_adapter.py`
- `Final_Project/Robosuite_MuJoCo_Sim/outputs/`

推荐先做三条线：

1. `scripted baseline`：固定动作序列，例如 move_down -> grasp -> move_up。
2. `VLM text-action baseline`：Qwen/VLM 输出 move_forward、move_left、grasp 等动作 token。
3. `OpenVLA visual baseline`：OpenVLA server 输出连续 action，再由 adapter 映射到仿真控制。

运行方式：

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Robosuite_MuJoCo_Sim"
.\scripts\check_sim_env.ps1
.\scripts\run_qwen_closed_loop.ps1
.\scripts\run_openvla_visual.ps1
```

记录指标：

- task success
- steps to completion
- closed-loop frequency
- model inference latency
- simulator step latency
- invalid action count
- failure type

最低交付：

- 每个 baseline 至少 5-10 次 trial
- 输出 JSONL 日志
- 成功/失败截图
- failure taxonomy

### 阶段 E：Jetson 部署与边缘系统 benchmark

目标：把 laptop 上可复现的推理流程迁移到 Jetson Orin Nano Super，并测资源瓶颈。

Jetson 侧建议结构：

```text
jetson_runtime/
  scripts/
    check_jetson_env.sh
    run_local_policy.sh
    start_policy_server.sh
  src/
    camera_capture.py
    policy_client.py
    resource_logger.py
    benchmark_runner.py
  results/
    jetson_local/
    jetson_remote/
```

Jetson 第一阶段不必追求完整 OpenVLA-7B 本地实时运行。推荐优先比较：

1. Jetson 只做 camera + logging + remote OpenVLA/VLM call。
2. Jetson 本地跑轻量 VLM/VLA-like 模型。
3. Jetson 本地跑量化模型，如果依赖可行。
4. OpenVLA-7B 本地加载作为 stress test。

Jetson 必测指标：

- install success / load success / OOM
- model loading time
- inference latency
- p95 latency
- end-to-end round trip latency
- CPU usage
- GPU usage
- RAM usage
- swap usage
- power mode
- temperature

最低交付：

- Jetson 环境记录
- 至少一个本地轻量模型 benchmark
- local vs remote 对比
- OOM 或不兼容问题记录

### 阶段 F：优化实验与最终分析

目标：把项目从“系统跑通”提升到“有研究分析”。

推荐优化变量：

1. Precision：FP32 / FP16 / INT8 / 4-bit
2. Image size：224 / 336 / 448
3. Runtime：local / remote / edge-cloud split
4. Power mode：Jetson default / max performance
5. Control frequency：1 Hz / 2 Hz / 5 Hz
6. Model choice：Qwen/VLM / SmolVLA / OpenVLA / ACT-style baseline

每个优化实验都要同时看两类指标：

- 系统收益：latency、memory、power、control frequency
- 任务代价：success rate、action validity、failure type、recovery rate

最终分析结构：

1. 哪些模型能在 Jetson 上运行？
2. 哪些模型只能远程推理？
3. 瓶颈是 model loading、visual encoding、action generation、network，还是 simulator/robot control？
4. 量化是否真的改善端到端控制，而不仅是降低显存？
5. 低成本机器人/仿真任务中，失败主要来自 perception、action mapping 还是 actuator/control？

## 3. Baseline 设计

建议最终至少保留 4 个 baseline：

| Baseline | 输入 | 输出 | 作用 |
|---|---|---|---|
| Rule-based scripted | 任务 ID / object pose | 固定动作序列 | 最低控制下限 |
| VLM text-action | RGB image + instruction | 离散动作 token | 轻量可解释 baseline |
| Remote OpenVLA | RGB image + instruction | 连续 action vector | 大模型性能上限 |
| Local lightweight policy | RGB image + instruction | 文本动作或 action | Jetson 可部署方向 |

如果时间允许，再加入：

- ACT-style imitation baseline
- SmolVLA local/remote baseline
- Quantised VLA baseline

## 4. 结果文件组织

建议统一成下面的结构：

```text
Final_Project/
  Implementation_Workflow/
    IMPLEMENTATION_WORKFLOW_ZH.md
    EXPERIMENT_MATRIX.csv
    RESULT_LOG_SCHEMA.csv
    WEEKLY_DELIVERABLES.csv
  Local_VLA_Benchmark_Framework/
    results/
      offline_vlm/
      openvla_probe/
  Robosuite_MuJoCo_Sim/
    outputs/
      scripted_baseline/
      qwen_closed_loop/
      openvla_visual/
  Jetson_Results/
    env_checks/
    local_inference/
    remote_inference/
    optimisation/
```

## 5. 每次实验的固定流程

每次跑实验都按同一个 checklist：

1. 写清楚 experiment ID。
2. 固定 git/code version 或记录当天日期。
3. 记录硬件：laptop GPU / Jetson power mode / RAM / JetPack version。
4. 记录模型：model id、precision、quantisation、input size。
5. 运行 3 次 warm-up。
6. 正式运行 N 次 trial。
7. 保存 JSONL 原始日志。
8. 汇总 CSV。
9. 标注失败类型。
10. 写 3-5 句 experiment note。

## 6. 风险控制

| 风险 | 处理方式 | 是否影响毕业交付 |
|---|---|---|
| OpenVLA-7B 本地 OOM | 作为 negative result，改远程 OpenVLA | 不影响 |
| Jetson CUDA/依赖不兼容 | 先跑轻量模型和远程 policy | 不影响 |
| 真实机械臂到货晚 | 用 robosuite closed-loop 完成核心评估 | 不影响 |
| action space 不匹配 | 加 action adapter，只做 constrained tabletop task | 可控 |
| VLA 输出不稳定 | 加 failure taxonomy 和 safety wrapper | 反而是分析点 |
| 时间不足 | 保留 offline + sim + Jetson resource benchmark | 可交付 |

## 7. 最小可交付版本

如果时间紧，最低版本这样收敛：

1. 离线 benchmark：Qwen/VLM + OpenVLA probe。
2. robosuite closed-loop：scripted baseline + VLM text-action baseline。
3. Jetson benchmark：一个轻量模型或远程 OpenVLA client。
4. 优化实验：至少做 input size 或 precision 对 latency/memory 的影响。
5. 报告：重点写 deployment feasibility、bottleneck、failure analysis。

这已经足够形成 MSc capstone 的完整研究闭环。

## 8. 强版本目标

如果进展顺利，强版本可以做到：

1. Jetson 本地轻量 VLA/VLM policy。
2. Jetson remote OpenVLA client。
3. robosuite simulator 作为 host，Jetson 作为 policy server。
4. local vs remote vs quantised 的系统对比。
5. stereo/depth safety wrapper 或 workspace safety filter。
6. 真实低成本机械臂上的 3-5 个 tabletop trial。

## 9. 报告章节映射

| 报告章节 | 对应工作流产物 |
|---|---|
| Introduction | 研究问题、Jetson/VLA 动机 |
| Literature Review | Top 10 papers, VLA/edge/low-cost robot 分类 |
| Methodology | 本文档的系统架构、baseline、实验矩阵 |
| Implementation | Local benchmark、OpenVLA server、robosuite、Jetson setup |
| Experiments | EXPERIMENT_MATRIX.csv 和实际日志 |
| Results | benchmark summary CSV、图表、失败样例 |
| Discussion | 瓶颈、negative results、trade-off |
| Conclusion | Jetson VLA 可行性结论和未来工作 |
