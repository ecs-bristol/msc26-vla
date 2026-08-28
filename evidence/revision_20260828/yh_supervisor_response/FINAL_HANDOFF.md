# Final handoff: YH supervisor-response evidence

> 生成时间：2026-08-28
> 说明：本文件只提供实验与代码证据，供报告作者更新 Table 9/10/11 与相关 figures。

## 1. 代码修改

- `plugins/lerobot_policy_smolvla_int4/src/lerobot_policy_smolvla_int4/int4_linear.py`
  - INT4 clamp 从 `[-8,7]` 改为 `[-7,7]`。
- `plugins/lerobot_policy_smolvla_int4/tests/test_int4_linear.py`
  - 增加对称 7 范围、pack/unpack round-trip 测试。
- `scripts/wsl/bench_smolvla_latency.py`
  - 增加 `torch.cuda.synchronize()`，保存逐次 latency JSONL，输出 median/std/min/max/chunk_size。
- `scripts/wsl/int4_consistency_diagnostic.py`
  - 统计旧实现下 mixed 与 full-backbone INT4 的 `q == -8` 数量。
- `scripts/jetson/bench_tensorrt_connector.py`
  - 统一单 stream CUDA Event latency + synchronized wall throughput 协议。
- `scripts/jetson/export_smolvla_connector_onnx.py`
  - 支持 `--image-size 256/512` 与 `--output`。

## 2. INT4 consistency

- old_range `[-8,7]`，new_range `[-7,7]`。
- mixed total int4 values：96,731,136；`q == -8` count：0。
- full-backbone INT4 total：427,032,576；`q == -8` count：0。
- 结论：旧 mixed 实际没有 `-8`，不需要因 clamp 修正重跑 mixed closed-loop。
- 数据：`01_int4_consistency/int4_consistency.json`。

## 3. PC policy-only latency

固定 `(N,E,H)=(2,1,50)`，warmup=5，iters=50，seed=1000，CUDA 同步计时。

| config | mean | median | p95 | std | param MB | peak MB |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 | 185.43 | 174.93 | 258.31 | 36.19 | 1217.9 | 1280.76 |
| language_int8 | 164.09 | 163.06 | 177.15 | 6.72 | 929.4 | 992.86 |
| backbone_int8 | 173.91 | 170.56 | 197.35 | 14.14 | 835.9 | 962.84 |
| mixed | 196.79 | 191.92 | 223.12 | 13.30 | 787.5 | 960.08 |

- 数据：`02_pc_policy_latency_sync/pc_policy_latency_sync.csv`。

## 4. TensorRT connector

| engine | input | mean | p95 | throughput qps | definition |
| --- | --- | --- | --- | --- | --- |
| 256 | 1x256x768 | 0.858 | 0.981 | 1038.76 | sequential wall qps |
| 1024 | 1x1024x768 | 1.102 | 1.376 | 833.80 | sequential wall qps |

- 说明：latency 为 CUDA Event GPU execution，throughput 为 synchronized wall
  time 内完成的 sequential queries，两者定义不同，不能混为一行。
- 数据：`04_jetson_connector_benchmark/connector_benchmark.csv`。

## 5. Mixed Jetson closed-loop

- 原始 run 已找回：`libero_spatial_jetson_remote_na20_20260825T172401Z`。
- 配置：mixed V4/C4/T8，`(2,20,20)`，50 episodes。
- 成功率：33/50 = 66.0%，Wilson 95% CI [52.2%, 77.6%]。
- inference mean/p95：1012.5 / 1023.6 ms。
- round-trip mean/p95：1045.9 / 1056.0 ms。
- 数据：`05_jetson_mixed_closed_loop/`。

## 6. 未完成或缺失

- PC/Jetson 最小环境版本尚未单独记录；Jetson 已知 L4T R36.4.4、
  TensorRT 10.11.0、PyTorch 2.8.0a0+5228986。
- episode-time p95 未在原始 eval_info 中记录，需要额外仪表或标记为 NA。
- 完整 SmolVLA TensorRT、DLA、Torch-TensorRT 等失败/受限项见
  `../jetson_hardware/HARDWARE_OPTIMIZATION_SUMMARY.md`。
