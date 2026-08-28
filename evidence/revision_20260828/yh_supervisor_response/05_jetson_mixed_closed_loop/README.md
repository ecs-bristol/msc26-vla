# Mixed 4/8-bit Jetson closed-loop

- 原始 run：`libero_spatial_jetson_remote_na20_20260825T172401Z`
- 配置：mixed V4/C4/T8，action expert/embeddings/output heads FP16
- \((N,E,H)=(2,20,20)\)
- 成功率：33/50 = 66.0%，Wilson 95% CI [52.2%, 77.6%]
- 平均 episode time：38.9s（p95 episode time 未在原始 eval_info 中记录）
- inference mean/p95：1012.5 / 1023.6 ms
- round-trip mean/p95：1045.9 / 1056.0 ms
- 原始文件：`raw/mixed/eval_info.json`、`raw/mixed/remote_transport.jsonl`
- `stored_parameter_mb` 与 `peak_allocated_mb` 来自 policy-only benchmark，非 closed-loop peak。
