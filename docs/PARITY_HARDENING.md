# Paired-pilot parity hardening

日期：2026-08-26

分支：`parity-hardening`

基线：`baseline-parity-audit` at `6539536fb30cbd34dcadfca059bdfa7c275bfb4f`

## 结论与边界

本分支只修改代码、测试、审计材料并生成 dry-run。没有运行 Adaptive、H5/H10/H20/
H25/H30/H50 rollout，也没有运行新的 50 集或 300 集 rollout。既有结果未被修改、
覆盖或删除。

paired-pilot 的方法语义现为：

| 条件 | 执行动作 | range telemetry | safety-triggered discard/replan |
| --- | --- | --- | --- |
| `Static-H1-original` | 官方 native、no clip | 记录 | 无 |
| `Static-H5/H10/H20/H50` | 官方 native、no clip | 记录 | 无 |
| `Adaptive-H20→H1` | 官方 native、no clip | 记录 | 越界时 detection-only discard，并强制下一次 H1 |

`range_violation` 检测与 `clip_actions` 已成为两个独立开关。当前六个主条件均显式
`clip_actions=false`；Adaptive 仍能以越界检测触发 replanning，但不修改送入环境的
当前动作。若以后引入 clipping，必须作为所有对照条件共同的 action transform，且
继续保留 `Static-H1-original` 这个 official-native reference。

## pytest 回归对比

两个指定 SHA 都在独立 detached worktree、同一个 WSL venv、相同 `PYTHONPATH` 和
完全相同的全仓 pytest 参数下运行。原始 JUnit SHA256、命令模板及工作树路径记录在
`evidence/parity_hardening/pytest_commands.json`；精确 node ID 集合记录在
`evidence/parity_hardening/pytest_06d_vs_653.json`。

| Git SHA | passed | skipped | failed | errors | 时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `06d8531c70b68009dbb51cca3326a0fb8de51d96` | 204 | 1 | 56 | 34 | 296.08 s |
| `6539536fb30cbd34dcadfca059bdfa7c275bfb4f` | 210 | 1 | 56 | 34 | 297.28 s |

- 新增 failure/error node ID：0
- 消失 failure/error node ID：0
- 不变 failure/error node ID：90
- node-ID comparison JSON SHA256：
  `37e47d8544e4008c6d5ba81bc0eb079e20336237df29c67d695bd9fab8387c72`

这些 90 个历史 failure/error 属于仓库早期 catalog/spec/recorder 测试与当前 checkout
不一致的既有状态；`baseline-parity-audit` 没有增加新的失败。parity-hardening 新增的
action、配置、导出与 cleanup 定向测试结果为 `65 passed in 210.84s`。

当前 parity-hardening 工作树随后运行同一全仓 suite，结果为
`217 passed, 1 skipped, 56 failed, 34 errors in 267.95s`。与 `6539536` 比较仍是：
新增失败 0、消失失败 0、不变失败/错误 90；七个新增通过项来自本分支新增/扩展测试。
当前分支精确 node-ID comparison 位于
`evidence/parity_hardening/pytest_653_vs_hardening.json`，其 SHA256 为
`9e90eca8127a756aefe704cd13a8d5f4d84fe92d2d75c22bf3141d56a2199aab`。

## EGL/CUDA 正常退出

旧的 10 集 paired H1 已在结果全部原子落盘并打印最终计数后卡于 interpreter
shutdown。cleanup 路径此前只关闭环境，仍可能保留 policy、CUDA tensor、MuJoCo
model/data、processor 和 runtime 引用。本分支增加了幂等、显式的释放顺序：

1. episode 关闭官方环境并清空 MuJoCo/processor/observation 引用；
2. policy reset 后清空 policy 引用，执行 GC、CUDA synchronize 与 empty-cache；
3. backend 清空官方 runtime 引用并 GC。

真实离线 preflight 只执行一次官方 reset（含 10 个 settle no-op），不选择 policy
action，也不进入 rollout loop。外层使用 600 秒硬超时；进程在 `214.59 s` 自行退出，
`exit_code=0`、`timed_out=false`。因此“结果打印后解释器不退出”的症状在新的显式
cleanup 路径上没有复现。由于没有做逐引用 A/B teardown，本审计将根因准确限定为
“缺少确定的 EGL/CUDA/MuJoCo 生命周期收尾”，不声称已证明其中某一个单独引用是
唯一原因。命令与结果见 `evidence/parity_hardening/shutdown_preflight.json`。

## chunk_size=50 审计

frozen checkpoint 的 `config.json` 明确记录 `chunk_size=50`。LeRobot v0.6.1 中
`action_in_proj` 与 `action_out_proj` 是逐 token 的线性层，其权重矩阵维度由
`max_action_dim` 和 hidden size 决定，并不直接包含 50；因此 50 不是这两个权重矩阵
的 shape constraint。

但是 `chunk_size` 直接决定采样 noise shape、suffix attention mask 长度以及输出 suffix
切片，并且 checkpoint 是按 50-step action sequence 训练和保存的。没有固定 observation/
noise 的 chunk-size 25/30 数值 parity 或任务效果证据，所以“把模型 chunk 改为 25/30
是安全的”尚未得到证明。本分支继续强制模型 `chunk_size=50`，配置中的 25 或 30 会被
拒绝。

H25/H30 只作为 execution horizon：模型仍生成 `[1,50,7]`，buffer 分别执行前 25/
30 个动作后重规划。这不改变 checkpoint 结构或模型 chunk，且已加入构造/边界测试；
它们没有加入当前六条件 manifest，也没有启动 rollout。

## 小型审计导出

`evidence/parity_hardening/baseline_parity_export/` 包含：

- 官方 `eval_info.json` 的原样副本；
- paired `summary.csv` 的原样副本；
- 10 个已完成 `Static-H1-original` episode 的关键字段 JSON；
- `parity_report.json` 的原样副本；
- 来源路径、源 episode SHA256、文件尺寸与 SHA256 manifest；
- 可直接使用 `sha256sum -c SHA256SUMS` 验证的数据文件清单。

导出 manifest SHA256 为
`2f8c59c6c6f48ec61f9b860187d78e9144316a46deea5b5afce59ef1db9d5233`。
视频、模型、tensor dump 和大型运行目录均未提交。

## dry-run

新 dry-run 位于：

`/home/xinrui_shen/vla/runs/parity-hardening/pilot-dry-run-native-actions`

结果为 50 个严格 pairing key、六条件各 50 条、总计 300 条 planned episode；manifest
SHA256 为 `934a4887e2ddea7703d43db727b3c66416652b3cd3806e77ed0a526bca1d44f2`。
未传 `--execute`，没有加载模型或调用 rollout `env.step`。

## 样本规模表述修正

冻结实验的 `3/50` 与修复后 parity gate 的 `10/10` 使用不同 episode 数和不同
initial-state 构成，不能表述为同一评估集上的“从 3/50 恢复到 10/10”。准确结论是：
冻结 50-episode sample set 为 `3/50`，而修复后的独立 10 tasks × benchmark state 0
gate 为 `10/10`；官方 H1 量级已在这个小规模 gate 上恢复，50 集结论尚待后续单独
决定并运行。
