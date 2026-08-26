# SmolVLA 官方 H1 基线 parity 审计

审计日期：2026-08-26  
审计分支：`baseline-parity-audit`  
被审计 paired-pilot 版本：`e870c38eb025b942dca13e04de73c6cd595c4821`  
冻结旧结果：`/home/xinrui_shen/vla/runs/pilot-final-preflight-deterministic-20260825`

## 结论

**BASELINE_PARITY_PASS**

本机官方 LeRobot v0.6.1 H1 路径与修复后的 paired-pilot H1 都取得了
`10/10`。模型、checkpoint、本地 snapshot、CUDA/MuJoCo 安装不是此前
`0%–8%` 成功率的解释。此前 paired-pilot 的 H1 不是官方 H1：它使用了手写
LIBERO backend，并且额外执行了 safety clipping。

已证明存在的 parity 破坏为：

1. 旧 backend 输出 `256×256`，而官方环境输出 `360×360`。
2. 旧 backend 只翻转图像高度；官方 `LiberoProcessorStep` 同时翻转 H、W，
   即旋转 180 度。同一 raw image 的最大绝对差为 agentview `0.937255`、wrist
   `0.909804`。
3. 旧 backend reset 后 `settle_steps=0`；官方执行 10 个 no-op steps。
4. 旧 quaternion→axis-angle 实现会归一化 quaternion 并强制 `w>=0`；官方实现
   不做这两步。同一 raw state 的最大绝对差为 `6.280733`，实际旧/官方 reset
   state 的最大绝对差为 `6.269057`。
5. 旧 `Static-H1` 开启自定义 `clip[-1,1]`。固定 observation/noise 的官方
   postprocessed chunk 有 15 个越界值，全部来自 action dimension 6（gripper），
   原始最小值 `-1.060972`，最大超界幅度 `0.060972`。因此旧 Static-H1 的实际
   action 不可能与 native H1 相同。

上述差异是共同确认的根因集合：修复全部 parity 破坏后，paired H1 从冻结实验的
`3/50`（6%）恢复到 `10/10`。本审计没有运行逐因素 rollout ablation，因此没有
声称图像方向、settle、state 或 clipping 中某一个单项独自解释全部成功率差异，
也没有给这些因素分摊因果权重。

## 安全边界

- 未运行 Adaptive、H5、H10、H20、H25、H30、H50 或 chunk-size sweep。
- 未运行新的 50 集或 300 集 rollout。
- 修复后仅运行 `Static-H1-original` 的 10 tasks × 1 benchmark state。
- dry-run manifest 仍含 300 个计划项，但 execute provenance 将执行条件严格限制为
  `Static-H1-original`，实际 `executed_episodes=10`。
- 未下载模型或依赖。模型和 VLM 均从指定 frozen snapshot 读取，
  `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`local_files_only=True`。
- 未修改、覆盖或删除冻结旧结果。其 `summary.csv` 仍为 96,582 bytes，mtime
  `2026-08-25 23:13:13.433646347 +0100`。
- 新结果只写入 `/home/xinrui_shen/vla/runs/baseline-parity-audit`；没有写入仓库的
  `runs/`。
- paired executor 已打印 `{"executed_episodes": 10, "skipped_episodes": 0}` 且全部
  episode JSON/summary 已原子落盘后，Python 进程仍停留在 EGL/CUDA interpreter
  shutdown；等待 10 秒后人工终止了这个 post-rollout 残留进程。终止发生在最终结果
  输出之后，没有增加、重跑或截断 episode。

## A/B/C 实现审计

| 项目 | A. 官方 LeRobot v0.6.1 | B. `smolvla-benchmark` | C. 旧 paired pilot (`e870c38`) | 修复后 paired H1 |
| --- | --- | --- | --- | --- |
| evaluator/backend | `lerobot-eval` + `LiberoEnv` | 调用同一官方 CLI | 手写 `LiberoBackend` | 官方 `LiberoEnv` adapter |
| env image | 360×360 | 360×360 | 256×256 | 360×360 |
| image orientation | H+W flip | H+W flip | H-only flip | 官方 `LiberoProcessorStep` H+W flip |
| reset settle | 10 no-op | 10 no-op | 0 | 10 no-op |
| state | 官方 pos3 + quat→axis-angle3 + gripper2 | 官方 processor | 手写不同 quaternion 公式 | 官方 processor，8 维 |
| H1 action | native、无额外 clip | native、无额外 clip | wrapper + safety clip | wrapper、safety disabled；逐 action parity |
| pre/postprocessor | policy pre/post 各一次 | 同 A | wrapper 内各一次 | wrapper 内各一次 |
| `use_amp` | `False` | 官方配置 | `precision="fp16"` 标签未触发 autocast | 与旧 wrapper 相同；固定 noise parity=0 |
| control | relative，20 Hz，action `(7,)` | 同 A | relative/default 20 Hz，action `(7,)` | relative，20 Hz，action `(7,)` |
| camera mapping | agentview→`image`，wrist→`image2` | 同 A | 手工映射相同语义 | 直接复用官方 mapping |
| task/state | 官方 task language，benchmark init index | 同 A | task language 相同，benchmark index | 官方 task language/index |

`B` 的脚本 `scripts/wsl/run_official_pc_local_eval.sh` 确实调用官方
`lerobot-eval`，但原脚本没有强制 offline/local snapshot、没有显式 seed，也没有
完整 provenance。本分支只为审计增强了这些约束，没有切换或改写同学分支。

## Observation parity gate

固定条件：`task_id=0`、`initial_state_id=0`、`environment_seed=1000`。

| tensor | 官方 shape/dtype | 修复后 shape/dtype | 官方 SHA256 | 修复后 SHA256 | max abs diff |
| --- | --- | --- | --- | --- | --- |
| agentview | `[1,3,360,360]` float32 | `[1,3,360,360]` float32 | `2d174104...9f996` | `2d174104...9f996` | 0 |
| wrist | `[1,3,360,360]` float32 | `[1,3,360,360]` float32 | `fcb361a4...a623` | `fcb361a4...a623` | 0 |
| state | `[1,8]` float32 | `[1,8]` float32 | `0bb290ca...6eda` | `0bb290ca...6eda` | 0 |
| task string | 相同 | 相同 | — | — | exact |

完整 tensor、min/max、SHA256 和差异记录位于（不提交 Git）：

`/home/xinrui_shen/vla/runs/baseline-parity-audit/parity-gate-task0-state0-seed1000-repaired/parity_report.json`

## Action parity gate

使用相同官方 observation 与固定 deterministic noise：

- 官方 base `predict_action_chunk` 经 base pre/postprocessor；
- wrapper 内部 `_PostprocessedChunkPredictor.predict_action_chunk`；
- 输出 shape 均为 `[1,50,7]` float32；
- 两者 SHA256 均为
  `d5a52758f88004809e7ef4018fce920a5ddfdf09a295c17278fc93d6baa00a1f`；
- `max_abs_difference=0.0`；
- base preprocessor 与 postprocessor 每条路径恰好各执行一次；
- wrapper 不调用 base `select_action`，只调用 base `predict_action_chunk`。

因此 wrapper 本身在 safety clipping 之前没有引入 action chunk 数值差异。真正的
`Static-H1-original` 使用 `fixed_h=1`、`safety_enabled=False`、
`replan_after_safety_violation=False`，越界 native gripper 值原样送入官方环境。

## 运行身份与命令

共同冻结项：

- SmolVLA revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- SmolVLM2 revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- LeRobot `0.6.1`，Python `3.12.3`，PyTorch `2.11.0`
- Transformers `5.5.4`，MuJoCo `3.8.1`，robosuite `1.4.0`
- `num_steps=2`，`n_action_steps/fixed_h=1`，`chunk_size=50`
- `episode_length=280`，`batch_size=1`，suite `libero_spatial`
- environment seed `1000`，benchmark initial state 0

官方运行的 resolved command：

```bash
lerobot-eval \
  --policy.path=/home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --policy.pretrained_revision=6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --policy.vlm_model_name=/home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --policy.n_action_steps=1 --policy.num_steps=2 --policy.chunk_size=50 \
  --env.type=libero --env.task=libero_spatial --env.episode_length=280 \
  --eval.n_episodes=1 --eval.batch_size=1 --env.max_parallel_tasks=1 \
  --seed=1000 \
  --output_dir=/home/xinrui_shen/vla/runs/baseline-parity-audit/official-h1-ns2-seed1000-final
```

修复后 paired dry-run 与唯一 execute 选择：

```bash
python scripts/analysis/libero_spatial_paired_pilot.py --dry-run \
  --output-dir /home/xinrui_shen/vla/runs/baseline-parity-audit/paired-h1-original-fixed-20260826 \
  --base-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --vlm-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467

python scripts/analysis/libero_spatial_paired_pilot.py --execute \
  --output-dir /home/xinrui_shen/vla/runs/baseline-parity-audit/paired-h1-original-fixed-20260826 \
  --base-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --vlm-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --device cuda --episodes-per-task 1 --condition Static-H1-original
```

execute provenance 记录 Git SHA
`58e3fe86da9eada60d78829c9208982154f68e74`、manifest SHA256
`934a4887e2ddea7703d43db727b3c66416652b3cd3806e77ed0a526bca1d44f2`，以及每任务
environment/inference seed。所有任务 environment seed 均为 1000；paired 路径继续遵守
既有要求，为每个 `(task_id, seed, initial_state_id)` 派生稳定且跨条件相同的 inference
seed。

## 10 集结果

| 路径 | success | task 0..9 | 平均 model invocations | 平均 inference s | 平均 wall s | action parity |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 同学官方 H1 参考 | 8/10 | S S F S S S S F S S | 未记录 | 未记录 | 78.429 | 官方路径参考；未在其机器重放 tensor |
| 本机官方 `lerobot-eval` H1 | 10/10 | S S S S S S S S S S | 101.5（由 H1 terminal progress 重建） | 未单独记录 | 84.997 | reference |
| 修复后 paired `Static-H1-original` | 10/10 | S S S S S S S S S S | 101.7 | 54.321 | 103.378 | PASS，chunk max diff 0 |

同学的 10 集 `libero_spatial_eval_info.json` 没有完整记录 `num_steps` provenance；更严格
匹配本审计 `num_steps=2, n_action_steps=1, chunk_size=50` 的同学 50 集确认结果为
`36/50`（72%，平均 46.8 s/episode），各任务成功数为
`3,4,3,5,3,5,4,2,4,3`。这两组同学证据都说明官方 H1 不处于 0%–8% 区间。

修复后逐任务成功 step：

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| success step | 74 | 105 | 99 | 88 | 123 | 89 | 104 | 121 | 90 | 124 |

修复后平均 range violations 为 18.0/episode，但 `range_clips=0`、
`buffer_discards=0`、`mean_actual_horizon=1.0`，证明 diagnostic telemetry 仍能看见
越界，而 `Static-H1-original` 不再修改官方 action。

## 修改文件

- `src/libero_platform/backends/libero_backend.py`
- `src/libero_platform/backends/__init__.py`
- `scripts/analysis/libero_spatial_paired_pilot.py`
- `configs/evaluation/libero_spatial_paired_pilot.yaml`
- `scripts/analysis/smolvla_baseline_parity_gate.py`
- `scripts/analysis/prepare_offline_hf_cache_view.py`
- `scripts/analysis/capture_official_eval_provenance.py`
- `scripts/wsl/run_official_pc_local_eval.sh`
- `tests/test_official_lerobot_libero_backend.py`
- `tests/test_libero_spatial_paired_pilot.py`
- `tests/test_fixed_h_action_buffer.py`
- `tests/test_prepare_offline_hf_cache_view.py`
- `plugins/lerobot_policy_smolvla_adaptive/tests/test_plugin.py`
- `docs/BASELINE_PARITY_AUDIT.md`

运行结果、tensor、视频、模型 cache 均未加入 Git。

## 测试

受影响测试集：`55 passed in 193.20s`。

覆盖：H/W 图像方向、官方 10 settle steps、360×360 环境输入、8 维 state parity、
固定 noise 首 chunk parity、base pre/postprocessor 单次执行、Static-H1-original 不被
Adaptive safety 修改、manifest/execute guard、condition-only execution、resume、
per-episode telemetry、deterministic seeding 和 offline cache/provenance。

全仓 pytest 另行收集到 `210 passed, 1 skipped, 56 failed, 34 errors`。这些失败来自
本分支开始前已存在的仓库测试/fixture 不一致，例如仓库缺少旧测试期望的
`configs/policy_catalog.yaml`、`VALID` fixture 缺少当前必填 `resolved_revision`，以及
viewer/旧 CLI 测试；本次受影响测试集没有回归。`ruff` 未运行，因为冻结 venv 未安装
ruff，且按要求没有下载新依赖；`git diff --check` 通过。

## 仍未排除或不可直接比较的差异

1. 官方 evaluator 在进程级设置 seed 1000；paired evaluator 按既有 deterministic
   protocol 为每个 pairing key 重设 inference seed。因此两个 10 集运行不要求 action
   trace 相同。固定 noise chunk parity 已把模型/wrapper 数值实现差异排除。
2. 官方 `eval_info.json` 不单独记录 model inference time；其 `eval_ep_s` 与 paired
   `wall_time_to_terminal_s` 的 instrumentation 边界不同，只能作量级比较。
3. 同学 10 集参考缺少完整 `num_steps` provenance；同学的 50 集 ns2 结果提供了严格
   配置匹配的成功率量级，但不是同一组 10 个 trajectory。
4. 本审计没有逐因素 rollout ablation，不能从 10 集结果量化图像方向、settle、state
   与 clipping 各自的独立效应。
5. paired executor 的 episode 执行与持久化已完成，但 EGL/CUDA shutdown 残留仍需
   单独排查；它不属于 observation/action parity，也没有影响本次 10 条 terminal 记录。

以上差异不阻止 baseline parity gate：同一 observation/noise 的 tensor/action 已逐值
一致，且修复后的 paired H1 与官方 H1 均为 `10/10`。
