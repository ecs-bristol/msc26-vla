# Jetson 硬件优化实验汇总

> 板卡：Jetson Orin Nano，L4T R36.4.4，TensorRT 10.11.0，ONNX 1.17.0
> 模型：`HuggingFaceVLA/smolvla_libero`
> 目的：为报告整理硬件优化中成功、部分成功和失败的方法，并记录原因。

## 1. 成功且可直接写报告的方法

### 1.1 TensorRT connector（已验证，组件级 + 端到端 hybrid）

- 通过 `torch.onnx.export` 导出 connector。
- 用 `trtexec` 构建 FP16 engine。
- 实际接入使用的 1024-token engine：
  - GPU latency：mean 0.819 ms，p95 1.116 ms。
  - Throughput：1995.87 qps。
- 已接入完整 SmolVLA rollout，见最终消融。

### 1.2 TensorRT vision encoder（组件级）

- 通过 `torch.onnx.export(dynamo=True)` 成功导出。
- FP16 权重 engine（早期 256 输入探针）：
  - GPU latency：mean 15.37 ms，p95 16.89 ms。
  - Throughput：64.98 qps。
- 512 输入 INT8 engine（未校准探针）：
  - GPU latency：mean 83.85 ms，p95 100.34 ms。
  - Throughput：11.59 qps。

### 1.3 CUDA Graphs on TensorRT vision engine

- eager：mean 15.44 ms，p95 17.31 ms，enqueue 1.27 ms。
- CUDA Graph：mean 14.50 ms，p95 16.09 ms，enqueue 0.0145 ms。
- 结论：enqueue 开销下降约 87 倍，平均延迟降约 6.1%。

### 1.4 SDPA attention kernel

- text_model eager：mean 126.85 ms，p95 127.47 ms。
- text_model sdpa：mean 98.85 ms，p95 99.52 ms。
- 端到端 SDPA 与默认基线基本一致，说明默认已启用 SDPA。
- 报告结论：SDPA 已默认启用，禁用会显著变慢；不构成新增加速。

### 1.5 最终端到端消融

| 配置 | 成功率 | 每集时间 | 平均推理 |
| --- | --- | --- | --- |
| software FP16 `(2,20,20)` | 72.0% | 36.2s | 825.1ms |
| software + language INT8 | 80.0% | 33.3s | 881.8ms |
| software + language INT8 + TensorRT connector | 76.0% | 35.0s | 854.2ms |

connector hybrid 接入后成功率无显著退化，证明组件级 TensorRT 可回接到
完整 rollout；但 connector 计算量小，端到端收益有限。

## 2. 尝试但失败/受限的方法

### 2.1 完整 SmolVLA 直接 TensorRT

- 状态：未实现。
- 原因：SmolVLA 含 vision + connector + text + action expert + flow matching，
  动态控制流和 Transformers mask 使完整 ONNX 导出困难；更现实的路线是拆模块。

### 2.2 Torch-TensorRT

- 状态：不可用。
- 原因：容器 PyTorch 25.06 缺少 `torch._C._distributed_c10d`，import
  Torch-TensorRT 即崩溃。

### 2.3 FP16 vision hybrid（完整 rollout）

- 状态：失败。
- 原因：加载完整 PyTorch SmolVLA 后，再反序列化 343MB FP16 vision engine
  时 GPU/CMA 分配失败，报 `NvMapMemAllocInternalTagged error 12`。

### 2.4 INT8 vision hybrid

- 状态：链路能跑，但闭环成功率不可用。
- 原因：
  - 未校准 INT8 engine 成功率为 0。
  - 64/200/400 张校准后，冒烟成功率分别 10%/40%/50%，正式 5 集仅 16%。
  - 说明当前 PTQ 校准质量不足，需要更强的 QAT/校准流程。

### 2.5 DLA

- 状态：不可用。
- 原因：`trtexec --useDLACore=0` 返回 `No DLA core detected`；该 Orin Nano
  在当前 TensorRT runtime 下没有可用 DLA core。

### 2.6 PyTorch 完整策略 CUDA Graphs

- 状态：capture 失败。
- 原因：Transformers 在 capture 时动态创建 mask 和 CPU 标量
  `torch.tensor(0.0)`，CUDA Graphs 不允许；完整 `select_action` 还含
  flow-matching Python 循环，不适合整段 capture。

### 2.7 功耗模式

- 状态：未开展。
- 原因：导师认为单纯功耗档位技术含量不足，未纳入实验。

## 3. 建议的报告叙事

1. 软件层优化与量化层优化是主要、可端到端验证的贡献。
2. 硬件层采用组件级 TensorRT：connector 可端到端接入，vision 提供
   standalone engine 数据；CUDA Graphs 在 TensorRT engine 上显示 launch
   overhead 降低。
3. 完整 SmolVLA 的端到端硬件加速受 Orin Nano 显存/CMA、INT8 校准质量和
   Transformers 动态控制流限制；这些限制本身就是部署研究的有效结论。
