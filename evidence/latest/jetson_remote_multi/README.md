# Jetson Remote Multi-Step and Quantisation Summary

> 板卡：Jetson Orin Nano（L4T R36.4.4）｜项目：LIBERO Spatial
> 模型：`HuggingFaceVLA/smolvla_libero`
> revision：`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
> 固定：`num_steps=2`、`n_action_steps=20`、`chunk_size=20`、`N_EPISODES=5`

## 正式结果

| 配置 | 成功率 | 每集时间 | 平均推理 | 平均 round-trip |
| --- | --- | --- | --- | --- |
| FP16 | 72.0% | 36.2s | 825.1ms | 858.6ms |
| language-only INT8 | 80.0% | 33.3s | 881.8ms | 915.7ms |
| backbone INT8 | 82.0% | 33.5s | 938.1ms | 972.2ms |
| mixed 4/8 | 66.0% | 38.9s | 1012.5ms | 1045.9ms |

## Final ablation（full-stack connector hybrid）

| 配置 | 成功率 | 每集时间 | 平均推理 |
| --- | --- | --- | --- |
| software FP16 | 72.0% | 36.2s | 825.1ms |
| software + language INT8 | 80.0% | 33.3s | 881.8ms |
| software + language INT8 + TensorRT connector | 76.0% | 35.0s | 854.2ms |

connector TensorRT 接入后成功率和延迟未出现显著退化，但该子模块计算量小，
对端到端性能贡献有限；它主要用于证明 hybrid 执行链路可行。

## 说明

- 成功率差异在 Wilson 95% 置信区间内重叠，报告应表述为“未观察到成功率损失”，
  不做单一最优排名。
- INT8 配置在 Jetson 上每集时间与 FP16 相当，略快；推理延迟略高，符合
  custom layer 反量化开销的预期。
- `predict_count` 显示多步动作队列生效：每次服务端推理返回 20 步，客户端逐
  步弹出执行。
- 原始 WSL 结果目录：`~/vla/results/libero_spatial_jetson_remote_na20_*`。
