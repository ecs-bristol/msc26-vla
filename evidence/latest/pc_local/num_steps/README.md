# Experiment A: Flow-Matching Step Sweep (num_steps) — PC Local

> 记录时间：2026-08-16 ｜ 状态：实验 A 主数据已完成（spatial），待补 object

## Provenance（复现条件）

- Harness：官方 `lerobot-eval`（lerobot v0.6.1，`run_official_pc_local_eval.sh`）。
- 模型：`HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。
- 套件：`libero_spatial`（10 任务）；`episode_length=280`；`batch_size=1`；`max_parallel_tasks=1`。
- 环境：WSL + MuJoCo EGL，`HF_HOME=~/vla/hf-cache`。
- 改动方式：`--policy.num_steps=<n>`（已确认覆盖生效：ns10 每集 90.8s vs ns2 46.8s）。
- 数据文件：`num_steps_summary.csv`（整体指标）、`num_steps_per_task.csv`（每任务成功数）。
- 原始输出：`~/vla/results/libero_spatial_pc_local_ns*_<stamp>/eval_info.json`。

## 结果摘要

| num_steps | 样本量（集/任务） | 成功率 | 每集用时 |
| --- | --- | --- | --- |
| 10（基线） | 5 | 72%（36/50） | 90.8s |
| 5 | 5 | 72%（36/50） | 63.1s（−30%） |
| 2 | 5 | 72%（36/50） | 46.8s（−48%） |

单集筛选（各 1 集/任务）：ns10=80%、ns8=80%、ns5=100%、ns3=100%、ns2=100%。筛选的高分是单集噪声，报告以 5 集档为准。

## 结论（供报告）

1. `num_steps` 10→2，整体成功率持平（72%），失败分布弥散、无系统性掉点。
2. 单集耗时下降 48%（90.8s→46.8s），与 SnapFlow「去噪占端到端延迟约 80%」的动机一致。
3. 评测高度可控：三档同为 36/50，同一批 seed 下结果高度相关 → 去噪冗余的强证据。

## 图（figures/）

- `fig_success_vs_numsteps.png`：成功率 vs 步数（5 集档 + 每任务误差棒，附筛选档背景）。
- `fig_episode_time_vs_numsteps.png`：每集耗时 vs 步数（标注相对基线降幅）。
- `fig_tradeoff.png`：成功率-耗时权衡散点。
- `fig_per_task_heatmap.png`：10 任务 × 3 配置成功率热图。

## 复现命令

```bash
SUITE=libero_spatial N_EPISODES=1 MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  bash scripts/wsl/run_numsteps_sweep.sh                      # 5 档筛选
for n in 10 5 2; do
  NUM_STEPS=$n SUITE=libero_spatial N_EPISODES=5 EPISODE_LENGTH=280 \
    MODEL_REVISION=6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
    bash scripts/wsl/run_official_pc_local_eval.sh            # 3 档 5 集确认
done
python scripts/analysis/collect_numsteps_results.py           # 重新生成本目录 CSV
python scripts/analysis/plot_numsteps_sweep.py                # 重新生成图
```
