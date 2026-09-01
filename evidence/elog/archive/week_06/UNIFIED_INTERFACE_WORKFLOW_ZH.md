# 统一模型接口与部署 Benchmark 工作流

日期：2026-07-06

## 目标

本工作流用于在 Jetson 到货前先固定实验接口、结果 schema 和证据输出格式。后续更换小模型、大模型、本地推理、远程推理或 Jetson edge-client 模式时，只需要新增 adapter 或改配置，不需要重写 benchmark 主循环。

## 核心入口

```powershell
cd "D:\Bristol_IOT_with_AI\Capstone Project\Final_Project\Local_VLA_Benchmark_Framework"
python -m src.vla_bench.unified_runner --experiment jetson_readiness_interface_smoke --dry-run
```

真正生成结果：

```powershell
python -m src.vla_bench.unified_runner --experiment jetson_readiness_interface_smoke --models local_rule_baseline,mock_remote_policy --tasks desk_cup_pick --repeats 1
```

输出目录：

```text
results/unified/<experiment>_<timestamp>_<run_id>/
  metadata.json
  trials.jsonl
  trials.csv
  summary.csv
  failures.csv
```

## 当前可用 adapter

```text
scripted_adapter
mock_adapter
vlm_text_adapter
openvla_adapter
remote_http_adapter
```

建议顺序：

1. 先跑 `scripted_adapter` 和 `mock_adapter`，验证任务表、schema、CSV 和 Evidence 链路。
2. 再跑 `vlm_text_adapter`，验证 PC 本地 VLM baseline。
3. Jetson 到货后先跑 `scripted_adapter` 和 `mock_adapter`，确认 Jetson Python 环境与文件路径。
4. 小模型优先尝试 `jetson_local` 或 `jetson_quantized`。
5. 大模型如果本地加载失败或 OOM，记录失败行，再转为 `jetson_remote_client` 或 `remote_server`。

## 论文/报告优先结果表

优先使用：

```text
summary.csv
```

主表字段建议：

```text
model_key
adapter
deployment_mode
device_profile
load_success
oom
latency_mean_ms
latency_p95_ms
peak_memory_max_mb
success_rate
recommendation
```

失败分析使用：

```text
failures.csv
```

重点记录：

```text
failure_type
error
raw_output
notes
```

## Jetson 到货前完成标准

```text
python -m unittest discover -s tests -v
python -m src.vla_bench.unified_runner --experiment jetson_readiness_interface_smoke --dry-run
python -m src.vla_bench.unified_runner --experiment jetson_readiness_interface_smoke --models local_rule_baseline,mock_remote_policy --tasks desk_cup_pick --repeats 1
```

这三步通过后，项目具备统一接口、统一输出和 Evidence 可追踪结果。
