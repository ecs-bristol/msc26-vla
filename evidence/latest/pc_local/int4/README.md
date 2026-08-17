# Experiment B: VLM Weight Quantization (int4 negative, int8 positive)

> 记录时间：2026-08-17 ｜ 状态：PC-local 主数据完成

## Provenance

- Harness：官方 `lerobot-eval` + `lerobot_policy_smolvla_int4` 插件（`--policy.type=smolvla_int4`）。
- 模型：`HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。
- 套件：`libero_spatial`（10 任务）；`episode_length=280`；`batch_size=1`。
- 量化实现：自包含 per-group 权重量化（`int4_linear.py` / `int8_linear.py`）。
  - `int4_groupwise`：symmetric absmax，group=128，packed uint8。
  - `int8_groupwise`：symmetric absmax，group=128，int8 + fp32 scale。
  - scope 默认 `language`：只量化文本 transformer；视觉编码器、connector、动作专家保持原精度。
- fp16 基线来自实验 A 官方 `--policy.path` 5 集跑（72.0%）。

## Results（成功率，5 集/任务 = 50 集）

| 配置 | num_steps | 成功率 |
| --- | --- | --- |
| fp16 基线 | 10 | 72.0% |
| fp16 基线 | 2 | 72.0% |
| int8 语言-only | 10 | 78.0% |
| int8 语言-only | 2 | 80.0% |

单集冒烟（1 集/任务）：int4 全背干 30%、int4 语言-only 10%、int8 语言-only 100%。

## 延迟 / 内存（policy-only microbenchmark，n_action_steps=1）

| 精度 | num_steps | mean ms | p95 ms | 权重内存 MB | 峰值显存 MB |
| --- | --- | --- | --- | --- | --- |
| fp16 | 10 | 400.4 | 612.1 | 1217.9 | 1280.8 |
| int8 | 10 | 389.5 | 404.3 | 929.4 | 992.9 |
| fp16 | 2 | 144.6 | 189.4 | 1217.9 | 1280.8 |
| int8 | 2 | 133.3 | 149.2 | 929.4 | 992.9 |

## 结论

1. 均匀 4-bit PTQ 对 SmolVLA 不成立：全背干 30%、语言-only 10%（单集冒烟）。
2. int8 语言-only 量化保住并略提升成功率（78%/80% vs fp16 72%），同时权重内存 −23.7%、峰值显存 −22.5%、推理延迟略降。
3. num_steps=2 与 int8 正交叠加，形成“少步数 × 8-bit 量化”的组合最优。

## 数据文件

- `quant_summary.csv`：成功率汇总（含负结果）。
- `quant_bench.csv`：延迟/显存微基准。
- `figures/`：图（成功率对比、内存对比、延迟对比）。
