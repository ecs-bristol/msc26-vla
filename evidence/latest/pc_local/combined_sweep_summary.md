# 联合扫描汇总：num_steps / n_action_steps / chunk_size

> 环境：PC 本地（WSL，MuJoCo EGL）｜套件：`libero_spatial`
> 模型：`HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
> 固定项：`episode_length=280`、`eval.batch_size=1`、`max_parallel_tasks=1`
> 正式数据均为每任务 5 集（共 50 集/配置）；冒烟为每任务 1 集。

## 1. 三个独立变量

| 变量 | 含义 | 扫描范围 |
| --- | --- | --- |
| `num_steps` | flow-matching 去噪迭代次数 | 10 / 5 / 2 |
| `n_action_steps` | 每次推理后执行多少步再重规划 | 1 / 5 / 10 / 20 / 50 |
| `chunk_size` | 模型一次预测多少步动作 | 50 / 20 / 10 / 1 |

## 2. 汇总表（正式 5 集）

| 实验 | num_steps | n_action_steps | chunk_size | 成功率 | 每集时间 | 单步推理 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 | 10 | 1 | 50 | 72.0% | 90.8s | 400.4ms |
| A | 5 | 1 | 50 | 72.0% | 63.1s | 249.3ms |
| A | 2 | 1 | 50 | 72.0% | 46.8s | 144.6ms |
| C | 2 | 5 | 50 | 58.0% | 36.9s | — |
| C | 2 | 10 | 50 | 66.0% | 32.5s | — |
| C | 2 | 20 | 50 | 76.0% | 29.6s | — |
| C | 2 | 50 | 50 | 56.0% | 34.4s | — |
| D | 2 | 20 | 20 | 78.0% | 28.5s | — |
| D | 2 | 10 | 10 | 74.0% | 31.0s | — |
| D | 2 | 1 | 1 | 80.0% | 43.5s | — |

## 3. 量化 × num_steps 追加

| 配置 | num_steps | 成功率 | 单步推理 |
| --- | --- | --- | --- |
| int8 language-only | 10 | 78.0% | 389.5ms |
| int8 language-only | 2 | 80.0% | 133.3ms |
| int8 backbone | 10 | 78.0% | 445.1ms |
| int8 backbone | 2 | 78.0% | 142.9ms |
| mixed（vision/connector 4bit，text 8bit） | 2 | 76.0% | 179.2ms |

> uniform int4 只有 1 集冒烟：backbone 30%、language 10%，属于失败路线，未入正式表。

## 4. 观察到的规律

1. `num_steps` 是最强的时间杠杆：10 → 2 时成功率保持不变（72%），每集时间从
   90.8s 降到 46.8s，单步推理从 400ms 降到 145ms。这直接支持“去噪主导延迟”。
2. `n_action_steps` 存在最优区间：20 步是拐点（76%、29.6s），50 步反而更慢且
   成功率更低（56%、34.4s），说明开环执行过长后误差累积。
3. `chunk_size` 可以缩小而不损失成功率：`cs20_na20`（78%）不劣于
   `cs50_na20`（76%），时间还略短；甚至 `cs1_na1` 也能稳定工作（80%）。
4. 把三者组合后，最优配置是 `num_steps=2 + n_action_steps=20 + chunk_size=20`：
   78.0% 成功率、28.5s/集；相比默认 `10 + 1 + 50`（72.0%、90.8s），时间缩短
   约 3.2 倍且成功率不降。

## 5. 结论

- **时间优化**：`num_steps` 和 `n_action_steps` 是主要杠杆，`chunk_size` 是次要微调。
- **成功率稳定性**：`num_steps=2`、`chunk_size` 缩短、`int8` 量化都没有显著伤害
  成功率；uniform `int4` 会严重掉点。
- **推荐部署配置**：`num_steps=2`、`n_action_steps=20`、`chunk_size=20`、
  `int8` language/backbone（若部署硬件支持）。
- **注意**：各档仅 50 集，成功率差异多数在抽样噪声内；报告应强调“成功率不劣化 +
  时间显著下降”，而不是追求几个百分点的成功率排名。

## 6. 关联数据文件

- `combined_sweep_summary.csv`：本表机器可读版。
- `num_steps/`：实验 A 详细结果。
- `action_chunk/`：实验 C / D 详细结果与图。
- `int4/`：实验 B 量化结果与延迟基准。
