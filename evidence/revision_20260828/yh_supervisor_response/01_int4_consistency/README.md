# INT4 consistency

- 修改：`int4_linear.py` 将 INT4 clamp 从 `[-8,7]` 改为 `[-7,7]`。
- 诊断结果：mixed 和 full-backbone INT4 的 `q == -8` 数量均为 0。
- 因此旧数据无需因 clamp 修改重跑 mixed closed-loop。
- 单元测试 4 项全部通过。
