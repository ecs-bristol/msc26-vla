# TensorRT connector latency/throughput

- 统一协议：单 stream、100 warmup、1000 timed iterations、CUDA Events
  记录 GPU execution latency；throughput 为同步 wall time 内完成的
  sequential queries。
- 256-token engine：
  - latency mean 0.858 ms，p95 0.981 ms。
  - throughput 1038.76 qps。
- 1024-token engine：
  - latency mean 1.102 ms，p95 1.376 ms。
  - throughput 833.80 qps。

## latency 与 throughput 差异说明

- CUDA Event latency 仅包含 GPU kernel execution，不包含 host launch、
  H2D/D2H 和 Python 调度。
- throughput 使用同步 wall time，因此包含 host 开销。
- 因此 `1000 / latency_mean` 会略高于 throughput，这是协议定义不同造成的，
  不是数据错误。
- 报告如需同时展示两者，应分别标注 `gpu_latency_ms` 与
  `sequential_wall_throughput_qps`。
