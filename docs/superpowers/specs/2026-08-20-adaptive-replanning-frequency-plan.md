# SmolVLA 自适应重规划频率：最终冻结实施与评测协议

## 状态与边界

本文件是实施 adaptive controller 前冻结的实验协议。除下文明确允许的开发/校准
工作外，不得根据 pilot 或正式结果继续改变 controller、trigger 阈值、候选 horizon、
deadline 或统计方法。

不可变边界：

- checkpoint：`HuggingFaceVLA/smolvla_libero` 的冻结 immutable revision；
- harness：官方 `lerobot-eval`；
- 环境：`libero_spatial` 的 10 个任务、280 environment-step cap；
- batch size=1、`max_parallel_tasks=1`、FP16、`num_steps=2`；
- adaptive 不读取 reward、success 或 privileged state；
- 不在 rollout 中修改 LeRobot 内部 `n_action_steps`；adaptive 使用项目自有
  action buffer。

`chunk_size=20` 是独立消融因素，不能直接并入主 adaptive 比较。主比较保持冻结
checkpoint 的 `chunk_size=50`。

## 1. 研究问题、成功锚点与指标

目标是在不降低控制质量到不可接受水平的前提下，减少端到端 simulation rollout
中的模型调用与时间。

主质量锚点为：

```text
Adaptive - Static H=1 >= -5 percentage points
```

该界限针对 `Success@280 steps`，它是主指标。`Static H=20` 是关键吞吐/Pareto
对照，但不能替代 H=1 作为质量非劣性锚点。正式报告始终同时给出：

1. Adaptive vs Static H=1；
2. Adaptive vs Static H=20；
3. Static H=20 vs Static H=1。

`Success@deadline` 是部署次指标，不替代 `Success@280`。它严格定义为
simulation end-to-end deadline：从 `reset` 后初始 observation 已就绪开始，到成功、
失败终止或 deadline 截止为止；必须记录 deadline 前实际执行的
`executed_env_steps`。该指标不能在论文中直接表述为真实机械臂物理动作完成速度。

每个 episode 必须记录：

- `success`、`termination_reason`、`executed_env_steps`、`success_step`；
- `wall_time_to_terminal_s`、`wall_time_to_success_s`；
- `model_invocations`、同步测得的 `model_inference_time_s`；
- planned 与 actually executed 的 `effective_horizon`；
- chunk/buffer provenance、refill、trigger reason、clip/NaN 行为。

失败 episode 不得删除。成功 episode 的完成时间中位数可作为条件描述指标，但不能
取代全样本时间或成功率比较。`eval_ep_s` 仅保留为官方 evaluator 的聚合吞吐审计，
不是逐 episode 完成时间或主速度证据。

deadline 必须在正式数据收集前、独立于本项目既有 sweep 的结果冻结。若没有外部
SLA，不得从 pilot 或既有 `eval_ep_s` 结果中选择 30 s、45 s 或其他阈值。

## 2. 配对与可复现性

正式配对单位为：

```text
(task_id, seed[, initial_state_id])
```

只有官方环境实际暴露 `initial_state_id` 时才记录该字段。若未暴露，则记录 seed
派生的初始状态 provenance（包括 reset 语义、seed、LeRobot/LIBERO 版本）；不得
虚构 state ID 或将 seed 等同于未验证的 state ID。

每次正式条件运行前生成、验证并复制相同的 paired manifest。manifest 和
provenance 至少包含 checkpoint/revision、Git SHA、完整命令、resolved config、
Python/LeRobot/Torch/CUDA、硬件、视频策略、deadline、manifest 路径与 SHA-256。
保留每个条件原始 `eval_info.json`、stdout/stderr 与逐 episode telemetry。条件顺序
应随机化或 block-counterbalance，以限制时间漂移和 GPU 热状态影响。

## 3. 固定执行方式与对照组

adaptive controller 的候选执行 horizon 为：

```text
H in {1, 2, 5, 10, 20}
```

主对照组为：

1. Static H=1，`chunk_size=50`；
2. Static H=20，`chunk_size=50`；
3. Adaptive project-owned action buffer，候选集合如上。

Static H=50 只可作为单独的小规模上限诊断，用于检查长开环错误累积；它不进入
adaptive 候选集合，也不占主要正式实验预算。

hard trigger 必须强制下一 horizon 为 H=1、丢弃剩余缓存并记录原因。它只能使用
已观测的非特权输入、模型输出与 buffer 状态；不得读取 reward、success、任务完成
标记或其他 privileged state。

## 4. 实施前 action-safety gate

在定义任何 gripper trigger、非法动作策略或项目级 clip 策略前，必须完成并保存
版本绑定的 action-contract 证据。该 gate 至少确认：

- 官方 `lerobot-eval` 到 `env.step()` 的逐层 action 映射；
- 实际环境的 action shape、每维低/高值、分量顺序、gripper 编码与开闭极性；
- checkpoint postprocessor 输出与环境动作 contract 的关系；
- 哪一层负责 NaN/Inf、维度、范围检查与 clipping，或明确记录其不存在。

不得从“7D action”、旧 custom YAML runner、`validate_action` 或历史结果推断这些
事实。特别地，若正式 LeRobot 路径没有在进入环境前执行 finite/range validation 或
clip，必须记录为“无官方保护”；下游 simulator/controller 的数值截断不能被表述为
正式 evaluator 的安全验证。

只有该 contract 被源码证据或另行批准的最小 environment probe 实证后，才可冻结：

- gripper 变化 trigger 的维度、阈值和极性；
- NaN/Inf、shape/range 失败时的 episode-local buffer 清理与终止策略；
- 是否允许、在哪一层允许 clip，以及 clip 是否构成独立实验条件。

若静态源码仍不能确认某项，不得猜测或执行 probe；应列出缺失项并等待单独授权。

## 5. 实施前 wrapper parity gate

在实现或评测 adaptive 前，先验证 Fixed-H=20 的项目自有 action buffer 与原生
LeRobot `n_action_steps=20` 路径的 parity。使用相同 checkpoint、task、seed、
初始状态以及冻结的 observation trace，比对：

- postprocessed action 序列；
- 模型调用次数；
- executed environment steps；
- episode result 与 termination reason。

比较须预先规定浮点容差与随机性控制方式。若不一致，先解释并修复 processor、
reset、action slicing、buffer 或环境时序差异；不得将差异直接归因于 adaptive
horizon。该 gate 通过前，不进入 adaptive controller 实现或 rollout。

## 6. 分阶段协议

### 阶段 A：开发/校准 pilot（不进入正式统计）

pilot 包含主条件 Static H=1、Static H=20、Adaptive 各 10 个独立且严格配对的
单位，即相同的 10 个 `(task_id, seed[, initial_state_id])` 单位由三个条件复用。
它可用于：

- telemetry 完整性、manifest replay、wrapper parity 和时钟边界验证；
- 在开发/校准集内校准 trigger 阈值；
- 检查禁用信息未被 controller 使用。

pilot 数据绝不合入 50/100 episode 正式统计。pilot 完成后，冻结 controller 与
trigger 阈值，并生成使用新的、独立 paired seeds/initial-state provenance 的正式
manifest。Static H=50 若需运行，只能作为与此 pilot 分离的小规模诊断。

### 阶段 B：正式样本量决定与 50-episode confirmation

pilot 后，依据 paired discordant-pair rate 对预注册的 `-5pp` 非劣性界限进行
power analysis，决定正式样本量。100 个 paired episodes 不是自动充分的
non-inferiority 证明；若所需样本数大于 100，100 集只能写为 confirmation，不能
写为正式 non-inferiority claim。

若运行 50-episode 阶段，则使用正式 manifest 中每任务 5 个独立配对单位，且不再
调 controller 或阈值。报告配对效应、任务分层结果、discordant pairs 与置信区间；
该阶段不把 pilot 数据并入。

### 阶段 C：100-episode 冻结确认

若 power analysis 支持，使用正式 manifest 中每任务 10 个独立配对单位完成冻结
确认。主要统计针对 Adaptive vs Static H=1 的 `Success@280` 非劣性；同时报告
Adaptive vs H=20 与 H=20 vs H=1。`Success@deadline`、全样本 terminal time、
model invocations、inference time、effective-horizon 分布均为预注册次指标。

如果统计功效不足，则如实将该阶段报告为 confirmation，并保留完整原始数据和不确定
性；不得把“100”本身写成非劣性证明。

## 7. Chunk-size 独立消融

只有主 horizon 比较和 parity gate 完成后，才可进行 `chunk_size` 消融。它必须以
同一正式 manifest、相同 timing 规则的配对设计分离 chunk 生成长度与执行 horizon，
例如：

```text
chunk_size in {50, 20} x Static H in {1, 10, 20}
```

不得把同时改变 `chunk_size` 和 H 所得到的差异归因于任一单独因素。只有在该消融
证明 processor/action 语义与结果均可接受后，才可考虑将 `chunk_size=20` 用于后续
adaptive 实验。

## 8. 停止条件与报告规则

在以下任一情况，不继续 adaptive 方法：

- paired manifest、状态 provenance 或 telemetry 不能证明严格可配对比较；
- action-safety gate 未通过，或 gripper/非法动作/clip 语义未被实证；
- wrapper parity gate 未通过；
- controller 使用禁用信息，或运行中修改 LeRobot 内部 `n_action_steps`；
- Adaptive 相对 Static H=1 无法满足预注册质量界限；
- Adaptive 相对 Static H=20 没有预注册的全样本吞吐、deadline、调用数或推理时间收益；
- 所谓优势只出现在成功样本的条件时间，而全样本指标不支持。

仅当所需统计功效成立、质量非劣性成立且预注册次指标显示预期收益时，才可报告
“成功率保持且获得端到端效率收益”。否则报告完整负结果与不确定性。
