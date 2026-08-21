# SmolVLA 自适应重规划频率：实施与评测规划书

## 1. 目标与结论口径

在不改变 SmolVLA checkpoint、任务定义和 LIBERO 官方评测语义的前提下，按当前状态自适应地决定连续执行多少个动作后再调用模型重规划。

项目的主要目标是降低端到端 rollout 时间；成功率采用非劣性目标，而不是预设会提升：相对固定 `n_action_steps=1` 基线，成功率下降不得超过 5 个百分点。只有达到该门槛，才将吞吐提升作为有效收益报告。

本项目不把 `num_steps`（flow-matching 去噪步数）和 `n_action_steps`（动作重规划间隔）混为同一变量。正式对照固定 `num_steps=2`，因为已有 Spatial 50 集结果表明它与 `num_steps=10` 的成功率均为 72%，但每集时长由 90.8 s 降至 46.8 s。

## 2. 关键事实与实现原则

SmolVLA 每次推理生成一个 action chunk；`n_action_steps` 决定内部队列一次允许消费的动作数量。`n_action_steps=1` 对应每步重新生成，值大于 1 对应短时开环执行。因此，运行时直接修改该配置字段不是可靠的自适应方案：已经生成的队列不会因此安全地更新。

首版实现使用完整 action chunk 加项目自有的动作缓存：每次重规划生成一个 chunk，控制器只释放其中 `h_t` 个动作；当缓存耗尽或触发硬重规划时重新生成。令 `h_t in {1, 2, 3, 4, 5}`。这让重规划频率由明确、可记录的决策控制，而非依赖 LeRobot 私有队列状态。

## 3. 不可变实验条件

- Checkpoint：`HuggingFaceVLA/smolvla_libero`，固定 immutable revision。
- Harness：官方 `lerobot-eval`；不得用旧 custom YAML runner 产出可报告结果。
- 环境：`libero_spatial`，10 个任务，280 steps 上限，batch size 1，max parallel tasks 1。
- 推理：FP16，`num_steps=2`，相同 GPU、MuJoCo EGL、HF cache 和依赖版本。
- 评测：固定 task × initial-state/seed 配对；每个策略条件执行同一对照集。

所有运行命令必须显式传入并记录 `--policy.n_action_steps`。第一步先从 `eval_info.json` 和运行日志核验当前正式基线实际解析到的值，不能以已废弃 YAML 的默认值代替证据。

## 4. 分阶段工作计划

### 阶段 0：冻结可复现基线

**目的**：确定所有后续实验真正比较的对象。

1. 固定 checkpoint revision、LeRobot v0.6.1、驱动/CUDA、GPU 型号和评测脚本提交版本。
2. 修改正式启动脚本，使 `POLICY_N_ACTION_STEPS` 为显式环境变量，并将其写入运行日志和结果元数据。
3. 运行 `n=1, num_steps=2` 的 10 任务 × 5 episode 基线（50 集）；核验 action queue 的模型调用频率和成功率。
4. 在同一套 seed 上补跑静态 `n=2,3,5` 筛选条件。

**验收标准**：每个输出目录可追溯完整命令、resolved config、revision 与结果；`n=1` 的表现与已有 72% / 46.8 s 结果在随机波动范围内一致。

**停止条件**：若 baseline 无法复现，先排查依赖、revision、seed 或环境差异；不进入控制器开发。

### 阶段 1：建立动作缓存与可观测性

**目的**：在不改变环境 step 语义的前提下，让每个执行动作可追溯到其生成时刻。

1. 为策略包装层增加 `predict_action_chunk` 调用和 episode-local action buffer。
2. 每个 step 记录：`chunk_id`、`chunk_offset`、`model_invoked`、`chosen_horizon`、`replan_reason`、缓存剩余量、动作裁剪标志和推理耗时。
3. 强制安全不变量：episode reset、模型错误、无效动作、硬重规划均清空缓存；动作仍须执行现有验证、缩放与 clip。
4. 编写单元测试，覆盖缓存顺序、恰好消费 `h_t` 个动作、reset 清空、硬触发提前丢弃剩余 chunk、异常不会复用旧动作。

**验收标准**：固定 `h=1` 时与旧路径逐动作输出相同（给定相同 deterministic sampling noise）；所有 action 可由日志唯一定位到 chunk。

### 阶段 2：最小、可解释的自适应控制器

**目的**：用低成本、可验证的物理信号选择短期开环长度。

控制器只使用已观测量，不调用额外模型，不把未经校准的生成模型随机性当作置信度。初始规则如下：

| 状态/事件 | 决策 `h_t` | 原因 |
| --- | ---: | --- |
| 新 episode、缓存耗尽 | 重新规划 | 没有可用动作 |
| gripper 指令发生翻转或 gripper state 突变 | 1 | 抓取/释放是高风险离散事件 |
| 末端位置或姿态增量超过预设阈值 | 1 | 可能接触、偏差或快速运动 |
| 图像变化分数超过阈值 | 1 | 场景状态发生未预期变化 |
| 动作被裁剪、NaN/异常、环境早停 | 1 并丢弃缓存 | 安全恢复 |
| 连续稳定、无上述事件 | 3–5 | 节省推理调用 |

为避免频繁来回切换，使用 2-step hysteresis：只有连续两步稳定才从低 horizon 升高；任何硬触发立即降到 1。首版不应超过 `h_max=5`。

**验收标准**：在录制的 observation trace 和模拟 smoke test 上，所有 hard trigger 的下一动作来自新 chunk；稳定段的模型调用数下降；没有越界 horizon 或跨 episode 缓存。

### 阶段 3：消融筛选

**目的**：确认 adaptive 的收益来自频率策略，而不是 `num_steps`、seed 或代码路径变化。

在完全相同的 seed/task 对上评测：

1. 固定 `n=1`（强反应基线）。
2. 固定 `n=2`、`n=3`、`n=5`（速度–成功率前沿）。
3. adaptive，无 hard trigger（验证触发器本身贡献）。
4. adaptive，完整规则。

每个条件先 10 任务 × 5 episode（50 集）。报告每任务成功数、平均/分位 episode time、模型调用数、平均 horizon、horizon 分布、硬触发频率及失败发生的阶段。

**晋级条件**：完整 adaptive 相对 `n=1` 的成功率不低于 −5 pp，且 episode time 至少下降 25%；若静态 `n=2` 已占优，则 adaptive 必须相对它显示更好的成功率–速度前沿，否则不增加系统复杂度。

### 阶段 4：正式确认与统计

**目的**：将筛选结果与可报告结论分开。

1. 固定阶段 3 选出的规则、阈值和代码 revision，不再调参。
2. 对 `n=1` 与最终 adaptive 运行至少 10 任务 × 10 episode（每条件 100 集）；任务和 seed 一一配对。
3. 主要检验：成功率差 `adaptive - n=1` 的 one-sided 95% confidence bound 大于 −5 pp；使用按 task/seed 配对的 bootstrap 或 McNemar 分析，并报告效应量与置信区间。
4. 次要指标：episode-time median/P95、模型调用数、平均 horizon、tail latency、无效动作率及每任务成功率。

**报告规则**：仅在主要非劣性成立后才报告“成功率保持、速度提升”。若不成立，如实报告为“延迟收益伴随成功率损失”，保留完整负结果。

## 5. 预期效果与量化假设

当前证据是 `num_steps=2, n_action_steps=1` 时 Spatial 的 36/50 成功（72%）和 46.8 s/episode。policy-only microbenchmark 为约 145 ms/action。若平均有效 horizon 为 `H`，粗略时间模型为：

`T(H) ≈ 6.3 s + 40.5 s / H + trigger_overhead`

其中 6.3 s 是由已有 46.8 s episode time 减去 `280 × 145 ms` 得到的近似非模型开销；它只用于设定实验预期，不作为结果结论。若 `H=3–4`，无额外开销的估计为 16–20 s；考虑触发、后处理和实际 step 数，正式目标应保守设为 19–30 s/episode，即较基线快 35–60%。

模型调用数的理论下降为约 `1 - 1/H`：平均 horizon 3–4 对应约 67–75% 更少的模型前向调用。成功率不能从现有 `num_steps` 实验外推；合理目标是保持在约 67–72% 以上，而非预设提升。

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
| --- | --- | --- |
| 接触/抓取阶段开环过长 | 成功率下降 | hard trigger、`h_max=5`、失败阶段分析 |
| 图像变化阈值对任务不稳健 | 误触发或漏触发 | 每任务报告触发率；阈值仅在筛选集调一次 |
| 动态改配置导致 queue 混乱 | 动作来源不可追溯 | 自有 buffer；不在 rollout 期间修改 config |
| 速度收益被环境或 I/O 掩盖 | 结论失真 | 分离模型、环境和端到端耗时；关闭视频录制 |
| 50 集筛选过拟合 | 假阳性 | 100 集冻结确认，种子配对，保留负结果 |

## 7. 最终交付物

- 实施代码及 unit/integration tests。
- 可复现启动脚本，显式记录 `num_steps`、`n_action_steps` 和 adaptive 配置。
- 50 集筛选数据、100 集冻结确认数据及原始 `eval_info.json`。
- 一张 success–latency Pareto 图、一张 horizon/trigger 分布图、逐任务结果表和统计检验说明。
- 一份结论：是否满足成功率非劣性、实际提速幅度、哪些任务/阶段从自适应中受益或受损。
