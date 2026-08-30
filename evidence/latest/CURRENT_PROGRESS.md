# Current Experiment Progress

> 最后更新：2026-08-25
> 模型：`HuggingFaceVLA/smolvla_libero`
> revision：`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
> 套件：`libero_spatial`，正式配置每任务 5 集（共 50 集）

## PC-Local Software and Quantisation

已完成：

- `num_steps` 扫描：10 / 5 / 2，成功率均为 72.0%。
- `n_action_steps` 扫描：1 / 5 / 10 / 20 / 50，最优 `n_action_steps=20`。
- `chunk_size` 配对：`cs20_na20` 为时间-成功率折中候选。
- 量化：FP16、language INT8、backbone INT8、mixed 4/8。

关键结果：

- 默认 `(10,1,50)`：72.0%，90.8s/集。
- 软件候选 `(2,20,20)`：78.0%，28.5s/集。
- language INT8 `(2,1,50)`：80.0%，延迟 133.3ms。
- backbone INT8 `(2,1,50)`：78.0%，延迟 142.9ms。

详细数据：`pc_local/num_steps/`、`pc_local/int4/`、`pc_local/action_chunk/`。

## Jetson Remote Multi-Step and Quantisation

板卡：Jetson Orin Nano（L4T R36.4.4）

正式结果（`(2,20,20)`，每任务 5 集）：

| 配置 | 成功率 | 每集时间 | 平均推理 | 平均 round-trip |
| --- | --- | --- | --- | --- |
| FP16 | 72.0% | 36.2s | 825.1ms | 858.6ms |
| language INT8 | 80.0% | 33.3s | 881.8ms | 915.7ms |
| backbone INT8 | 82.0% | 33.5s | 938.1ms | 972.2ms |

多步动作返回已在服务端和客户端实现；客户端动作队列按 `n_action_steps`
逐条弹出，服务端 `predict_action_chunk` 返回完整动作块。

详细数据：`jetson_remote_multi/`。

## Jetson Hardware Acceleration

TensorRT：

- 可用版本：TensorRT 10.11.0，ONNX 1.17.0。
- Torch-TensorRT 当前镜像不可用，采用 ONNX + `trtexec`。
- connector 子模块已成功导出并构建 FP16 engine：
  - GPU latency mean 0.443 ms，p95 0.598 ms。
  - Throughput 3031 qps。
- vision transformer 的 TorchScript tracing 尚被动态 attention mask 阻断。

详细数据：`jetson_hardware/`。

## 待完成

- vision/text 子模块 TensorRT 导出（`dynamo=True` 或记录限制）。
- CUDA Graphs eager vs replay。
- `nvpmodel` 功耗档位扫描。
- 最终端到端消融：baseline → software → software+quant → full hardware stack。
