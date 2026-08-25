# Jetson Hardware Acceleration Evidence

> 板卡：Jetson Orin Nano（L4T R36.4.4）｜TensorRT 10.11.0｜ONNX 1.17.0

## TensorRT connector（已验证）

- ONNX 文件：`smolvla_connector.onnx`
- 输入：`(1, 256, 768)`，输出：`(1, 16, 960)`
- 精度：FP16
- Engine 大小：22.5 MiB
- GPU latency：mean 0.443 ms，p95 0.598 ms
- Throughput：3031 qps

该结果证明 SmolVLA 的 connector 子模块可以完成
`PyTorch -> ONNX -> TensorRT -> engine` 的板端加速链路。

## 已知限制

- vision transformer 的 TorchScript ONNX tracing 在 Transformers 动态
  attention mask 处失败；后续可尝试 `dynamo=True` 或记录为不支持路径。
- Torch-TensorRT 在当前 NVIDIA PyTorch 25.06 镜像中 import 失败
  （缺少 `torch._C._distributed_c10d`），因此采用 ONNX + `trtexec` 路线。
