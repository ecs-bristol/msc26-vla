# Jetson Hardware Acceleration Evidence

> 板卡：Jetson Orin Nano（L4T R36.4.4）｜TensorRT 10.11.0｜ONNX 1.17.0

## TensorRT connector（已验证）

- ONNX 文件：`smolvla_connector.onnx`
- 输入：`(1, 256, 768)`，输出：`(1, 16, 960)`
- 精度：FP16
- Engine 大小：22.5 MiB
- GPU latency：mean 0.443 ms，p95 0.598 ms
- Throughput：3031 qps

## TensorRT vision encoder（已验证）

- ONNX 文件：`smolvla_vision.onnx`
- 输入：`(1, 3, 256, 256)`，输出：`(1, 256, 768)`
- 精度：FP16 权重
- Engine 大小：330.9 MiB
- GPU latency：mean 15.37 ms，p95 16.89 ms
- Throughput：64.98 qps

## TensorRT vision encoder（INT8）

- ONNX 文件：`smolvla_vision.onnx`
- 精度：INT8
- Engine 大小：87.6 MiB
- GPU latency：mean 13.61 ms，p95 14.86 ms
- Throughput：73.48 qps

## CUDA Graphs on vision engine

| 模式 | mean ms | p95 ms | enqueue ms | qps |
| --- | --- | --- | --- | --- |
| eager | 15.44 | 17.31 | 1.27 | 64.65 |
| CUDA Graph | 14.50 | 16.09 | 0.0145 | 68.90 |

CUDA Graphs 将 enqueue 开销降低约 87 倍，平均延迟降低约 6.1%，
p95 降低约 7.1%。

## Attention kernel（eager vs SDPA）

| scope | implementation | mean ms | p95 ms | peak MB |
| --- | --- | --- | --- | --- |
| text_model | eager | 126.8 | 127.5 | 1252.5 |
| text_model | sdpa | 98.8 | 99.5 | 1252.4 |

端到端 `sdpa` 成功率 72.0%、每集 36.4s，与默认 FP16 基线基本一致，
说明 SmolVLA 默认已启用 SDPA；关闭 SDPA 会显著变慢。

## Jetson policy-only quantisation benchmark

> 固定 `num_steps=2`、`n_action_steps=1`，5 次迭代。

| 配置 | mean ms | p95 ms | peak GPU MB | param MB |
| --- | --- | --- | --- | --- |
| FP16 | 640.6 | 642.9 | 1279.7 | 1217.9 |
| language INT8 | 704.7 | 710.0 | 991.8 | 929.4 |
| backbone INT8 | 883.5 | 893.3 | 961.8 | 835.9 |
| mixed 4/8 | 745.7 | 748.1 | 959.0 | 787.5 |

原始数据：`jetson_quant_bench.csv`。

该结果证明 SmolVLA 的 connector 子模块可以完成
`PyTorch -> ONNX -> TensorRT -> engine` 的板端加速链路。

## 已知限制

- Torch-TensorRT 在当前 NVIDIA PyTorch 25.06 镜像中 import 失败
  （缺少 `torch._C._distributed_c10d`），因此采用 ONNX + `trtexec` 路线。
- DLA：`trtexec --useDLACore=0` 返回 `No DLA core detected`，该 Orin Nano
  在当前 TensorRT runtime 下无可用 DLA core，DLA 卸载不可用。
- 完整 PyTorch SmolVLA 的 CUDA Graphs capture 受 Transformers 动态 mask
  创建限制；CUDA Graphs 数据以 TensorRT vision engine 的对比为准。
- TensorRT INT8 vision hybrid 在当前 PTQ 校准下正式闭环成功率仅 16%，
  不适合作为端到端部署配置；INT8 vision 结果仅保留为组件级 microbenchmark。
