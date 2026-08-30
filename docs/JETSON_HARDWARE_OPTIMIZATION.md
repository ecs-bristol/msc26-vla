# Jetson 硬件优化实验方案：TensorRT + CUDA Graphs

> 状态：方法已选定，待 Jetson 板子到手后执行（2026-08-18）
> 目标：在已完成的实验 A（`num_steps` 扫描）和实验 B（权重量化）基础上，补充两项硬件侧优化，形成完整消融。
> 前置结果：PC 端已完成 fp16 / int8 / mixed 量化的成功率与延迟基准。

## 0. 选型结论

| 方法 | 结论 | 理由 |
| --- | --- | --- |
| TensorRT FP16/INT8 | **选做** | NVIDIA Jetson 官方硬件加速路径，可同时测延迟、显存、吞吐，并和实验 B 的 torchao INT8 对照 |
| CUDA Graphs | **选做** | 实现成本低，直接量化 kernel launch / Python eager 开销，适合与 `num_steps`、延迟消融并列 |
| 功耗模式（nvpmodel/tegrastats） | **选做（导师补充）** | 官方功耗控制与采集工具，量化功耗-延迟-成功率权衡，构成能效贡献线 |
| DLA | 评估后排除 | DLA 支持算子有限，SmolVLA 的动态 shape、attention 与自定义 expert 模块大概率不兼容 |
| GGUF / LiteVLA-Edge | 备选 | 若 TensorRT 对完整 SmolVLA 转不出来，再退回 GGUF Q4_K_M 路线 |

## 1. 目标与假设

- H1：TensorRT FP16 相比 PyTorch fp16，在相同输入下降低推理延迟并减少显存占用。
- H2：TensorRT INT8 相比 FP16 进一步降低延迟/显存，同时 LIBERO spatial 成功率下降幅度可接受（建议阈值 ≤10pp）。
- H3：CUDA Graphs 能减少 kernel launch / CPU-GPU 同步开销；若 eager 开销占比低，则量化“这部分收益有限”的结论。
- H4：硬件优化与实验 A/B 正交，可组合出“低 `num_steps` + INT8 + TensorRT/CUDA Graph”的最优部署配置。
- H5：不同 `nvpmodel` 功耗档位下，成功率不显著下降，但延迟/功耗/温度呈明显权衡，存在能效最优档位。

## 2. 固定实验协议

- 板卡：同现有远程评测 Jetson（`10.42.0.2`，`msc26vla`），先跑 `jetson_release` 确认型号/JetPack。
- 模型：`HuggingFaceVLA/smolvla_libero`
- revision：`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- 套件：`libero_spatial`（10 任务）
- 采样：先 `N_EPISODES=1` 冒烟，最终配置用 `N_EPISODES=5`
- 固定项：`episode_length=280`、`n_action_steps=1`、seed 与 PC 实验一致
- 对比配置：优先复测 `num_steps ∈ {10, 2}`，以及实验 B 中表现最好的 `int8_groupwise`（language/backbone scope）
- 指标：
  - 成功率：`eval_info.json` 的 `overall.pc_success` + per-task
  - 延迟：单次策略推理 mean / p95（microbenchmark）
  - 资源：权重内存、峰值显存、RAM
  - 能效：`tegrastats` / `jtop` 采集 GPU 利用率、功耗、温度

## 3. 方法一：TensorRT FP16/INT8

### 3.1 可行性优先策略

SmolVLA 含视觉塔、connector、text transformer、action expert 与 flow-matching 采样循环，**完整 ONNX 导出很可能失败**。因此不要强行整模型导出，按以下顺序验证：

1. 先跑可支持性子图探针，确认哪些模块能编译。
2. 优先用 Torch-TensorRT（`torch_tensorrt`）编译 **VLM backbone 子模块**，必要时回退到 ONNX + `trtexec`。
3. 对不支持的算子记录原因，报告中说明“完整策略未转换，采用子模块加速 + 成功率不回归”的证据链。

### 3.2 子实验

- TRT-A：FP16 编译 VLM backbone，microbenchmark 对比 PyTorch fp16。
- TRT-B：INT8 PTQ 校准 VLM backbone（用 LIBERO 观测样本做校准集），microbenchmark + 成功率冒烟。
- TRT-C（可选）：若子模块可插回策略，构造 hybrid policy 跑完整 `libero_spatial`；否则只报子模块级加速。

### 3.3 判定标准

- 通过：TRT FP16/INT8 在目标子模块上延迟显著下降，且 INT8 成功率掉点 ≤10pp。
- 风险：若核心算子不支持 INT8 或导出的子图不覆盖关键路径，则退化为“可编译子图加速”的结果，并启用 GGUF/LiteVLA-Edge 备选路线。

## 4. 方法二：CUDA Graphs

### 4.1 可行性边界

`select_action` 含 Python 控制流、随机噪声和多次去噪迭代，不适合直接整段 capture。目标改为：

- capture 单次 `vlm_with_expert.forward`（固定 batch、序列长度、图像尺寸、state 维度）。
- 用静态输入/输出 buffer 回放，对比 eager 与 graph replay 的 wall time。
- 用 `torch.profiler` 记录 kernel time vs CPU launch/sync 时间，量化 CUDA Graphs 的实际收益。

### 4.2 判定标准

- 若 wall time 下降明显：说明 CPU launch/sync 是瓶颈，CUDA Graphs 值得保留。
- 若下降很小：如实报告“该模型在 Jetson 上主要受计算/显存带宽限制，而非 launch 开销”，仍可作为硬件消融的一个数据点。

## 5. 功耗模式实验（nvpmodel / tegrastats）

### 5.1 实验内容

- 档位：`nvpmodel -q` 列出的可用档位，优先测 `MAXN SUPER`、`25W`、`15W`、`10W`（以板卡实际支持为准）。
- 固定：使用软件侧最优配置（低 `num_steps` + int8/mixed 精度）和同 seed，保证与延迟实验可比。
- 采集：推理期间用 `tegrastats` 记录 GPU/CPU 利用率、内存、温度与功耗；每次切换功耗档后重启或按官方要求生效。
- 可重复性：同一档位跑 2-3 次取中位数；必要时用 `jetson_clocks` 固定频率，排除 DVFS 抖动。

### 5.2 判定标准

- 通过：绘制“功耗-延迟-成功率”的权衡曲线，能找到一个成功率可接受前提下的能效最优档位。
- 若某档位成功率明显下降或系统不稳定，如实记录，并在报告中作为部署约束说明。

## 6. 实验矩阵

| 变体 | num_steps | 精度/引擎 | 成功率 | 单步延迟 | 显存 | 功耗 |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch fp16 基线 | 10 | fp16 eager | ? | ? | ? | ? |
| PyTorch fp16 | 2 | fp16 eager | ? | ? | ? | ? |
| PyTorch int8 | 2 | int8_groupwise | ? | ? | ? | ? |
| TensorRT FP16 | 2 | TRT FP16 | ? | ? | ? | ? |
| TensorRT INT8 | 2 | TRT INT8 | ? | ? | ? | ? |
| 最优配置 + CUDA Graph | 2 | 组合 | ? | ? | ? | ? |
| 最优配置 × 各功耗档 | 2 | 组合 | ? | ? | ? | 按档位记录 |

## 7. 数据与产出

数据统一放在 `evidence/latest/jetson/hardware/`：

- `tensorrt_bench.csv`
- `cudagraph_bench.csv`
- `power_mode_bench.csv`（含各档位功耗/温度/延迟/成功率）
- `eval_info_*.json`
- `power_tegrastats.log`
- `README.md`（记录板卡型号、JetPack、TensorRT 版本、成功/失败的子图范围）

图件建议：

- PyTorch vs TensorRT FP16/INT8 的延迟/显存柱状图
- CUDA Graphs eager vs replay 的延迟对比
- 软件优化 × 硬件优化的综合消融图
- 功耗-延迟-成功率权衡曲线（`nvpmodel` 各档位）

## 8. 排期（拿到板子后）

| 天 | 内容 |
| --- | --- |
| D1 | 确认 Jetson 环境与 TensorRT/Torch-TensorRT 版本，跑子模块可支持性探针 |
| D2 | TensorRT FP16/INT8 microbenchmark + INT8 校准 |
| D3 | CUDA Graphs capture/replay microbenchmark |
| D4 | 对 top 配置跑 `libero_spatial` 完整评测（N_EPISODES=5） |
| D5 | 功耗档位扫描（`nvpmodel` + `tegrastats`），汇总数据、出图 |

## 9. 文献支撑

- NVIDIA TensorRT 官方文档：Jetson/Orin 部署与 INT8 PTQ 官方路径。
- [GR00T-N1.6-bridge-INT8-Edge](https://huggingface.co/nvidia/GR00T-N1.6-bridge-INT8-Edge)：TensorRT INT8 VLA 部署示例。
- Seeed Studio Jetson GR00T TensorRT 教程：Jetson 上 VLA + TensorRT 的实操参考。
- Opara et al., “CUDA Graphs” 性能优化研究（IEEE 2024）：CUDA Graphs 减少 launch overhead 的文献支撑。
- LiteVLA-Edge（arXiv:2603.03380）：Jetson 上量化 VLA 的备选路线。
- SQIL（arXiv:2505.15304，ICCV 2025）：量化 + Jetson 实测加速/能效的支撑。
- “Deep Learning Workload Mapping Optimization on Jetson Platforms”（ACM TACO 2025）：DLA/硬件分区优化的排除依据。
- NVIDIA `nvpmodel` / `tegrastats` 官方文档：功耗模式控制与资源/温度/功耗采集。
