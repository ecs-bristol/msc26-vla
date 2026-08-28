# Final VLA Experiment Report

## Scope and freeze

This report consolidates the final SmolVLA LIBERO Spatial results at repository
commit `a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`. It is a read-only synthesis of
completed experiments. No model, environment, rollout, policy rule, threshold,
configuration, or source result was rerun or modified to produce it.

All retained experiments use the parity-corrected official LIBERO backend and
processor, native unclipped actions, `chunk_size=50`, `num_steps=2`, an episode
cap of 280, local frozen model snapshots, and offline inference. Execution
horizon changes how many actions from each generated 50-action chunk are used;
it does not change the checkpoint's generated chunk size.

## Statistical design: two independent stages

### Stage A — parity-corrected development cohort

This cohort contains 50 fixed pairing keys (10 tasks × 5 benchmark initial
states), manifest SHA-256
`934a4887e2ddea7703d43db727b3c66416652b3cd3806e77ed0a526bca1d44f2`,
and result Git SHA `b5cbf91a7c1a47ac48b7d30ed318ecd2ea252d1a`.
It compares Static-H1, Static-H10, Static-H20, and Adaptive-v1-H20→H1.

This stage is development evidence. Adaptive-v1 outcomes informed diagnosis and
the later success-blind v2a preregistration process; they are not confirmatory.

### Stage B — untouched held-out confirmatory cohort

This cohort contains a new disjoint set of 50 inference seeds on the same 10 × 5
task/state structure, manifest SHA-256
`2156db1ca5fe906e3f1a7ebb76d6ff62f1521456a6e1550c4aab2737e65fc357`,
and Git SHA `a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`.
It compares only Static-H20 and frozen Adaptive-v2a-H20→H1. Its four 25-key
blocks balance condition order and run serially.

The two stages are never pooled. No success flip, confidence interval, p-value,
or resource difference crosses the cohort boundary.

## Results

### Stage A: development

| Condition | Success@280 | Mean calls | Mean inference (s) | Mean wall (s) | Aggregate chunk utilization |
|---|---:|---:|---:|---:|---:|
| Static-H1 | 33/50 (66%) | 162.48 | 137.651 | 248.815 | 2.00% |
| Static-H10 | 30/50 (60%) | 17.48 | 9.841 | 61.400 | 19.68% |
| Static-H20 | 32/50 (64%) | 8.66 | 5.741 | 54.569 | 38.50% |
| Adaptive-v1-H20→H1 | 31/50 (62%) | 9.08 | 5.942 | 55.430 | 37.57% |

Static-H20 versus Static-H1 produced 26 both-success, 6 H20-only, 7 H1-only,
and 11 both-fail pairs. The paired success difference was −2 percentage points
(task-cluster bootstrap 95% CI −16 to +14 points; exact McNemar p=1.0). Thus the
observed success rates were close, but the interval is too wide for a formal
non-inferiority claim.

The compute reduction is clear within this cohort. Relative to Static-H1,
Static-H20 reduced mean model calls from 162.48 to 8.66 (94.7%), mean inference
time from 137.65 s to 5.74 s (95.8%), and mean wall time from 248.82 s to 54.57 s
(78.1%). The paired H20−H1 confidence intervals exclude zero for calls
([−175.86, −132.46]), inference time ([−164.13, −104.83] s), and wall time
([−230.44, −159.33] s).

Static-H20 also used fewer calls than Static-H10: paired mean difference −8.82,
95% CI [−10.64, −7.00]. Its success was 32/50 versus 30/50, but the success
difference CI [−12, +20] percentage points remains wide.

Adaptive-v1 was descriptively dominated by Static-H20. Its success set was a
strict subset: 31 both-success, zero Adaptive-only, one Static-only, and 18
both-fail. Adaptive-v1 made 0.42 more calls per episode on average (95% CI
+0.08 to +0.90) and took 0.201 s more inference time (95% CI +0.013 to +0.460).
Across seven triggers in six episodes there were no rescues, one loss, and six
no-change trigger events.

### Stage B: held-out confirmatory

| Condition | Success@280 | Mean calls | Mean inference (s) | Mean wall (s) | Aggregate chunk utilization |
|---|---:|---:|---:|---:|---:|
| Static-H20 | 33/50 (66%) | 8.32 | 5.558 | 50.872 | 38.80% |
| Adaptive-v2a-H20→H1 | 33/50 (66%) | 8.36 | 5.676 | 53.558 | 38.61% |

All 50 paired outcomes were identical: 33 both-success and 17 both-fail, with
zero Adaptive-only and zero Static-only episodes. The paired success difference
was exactly 0 (task-cluster bootstrap 95% CI [0, 0]); exact McNemar p=1.0.

Adaptive-v2a nevertheless increased mean model calls by 0.04 (95% CI [0,
+0.12]) and mean wall time by 2.685 s (95% CI [+1.113, +4.537]). Its inference
time difference was +0.118 s (95% CI [−0.053, +0.288]). The only held-out trigger
completed the full pre-execution H20→H1→H20 chain, but both policies failed that
pairing: one no-change, zero rescues, and zero losses.

Block A, where Static ran first, gave 17/25 successes for both policies. Block B,
where Adaptive ran first, gave 16/25 for both. The within-block success difference
was zero in both orders, so the outcome equality is not explained by the balanced
execution order.

The preregistered decision gates required at least one Adaptive-only rescue,
more rescues than losses, and complete trigger evidence. Trigger evidence was
complete, but both outcome gates failed. The unique frozen decision is:

> No demonstrated value for the added calls.

## Interpretation

1. **Static-H20 preserves a similar observed success level to the official H1
   execution baseline while sharply reducing computation.** This is strong
   compute evidence but only descriptive success evidence; the study does not
   establish formal non-inferiority.
2. **Adaptive-v1 is descriptively dominated by Static-H20** in the development
   cohort: one lost success, no rescue, and more calls.
3. **Adaptive-v2a produces exactly the same held-out success outcomes as
   Static-H20 while adding calls and wall time.** Its preregistered value gate
   fails.
4. **Range-violation-triggered replanning shows no practical value in these
   experiments.** This supports freezing v1/v2a as negative results. Because
   v2a triggered only once, it does not prove that every possible adaptive signal
   is intrinsically ineffective.

## Excluded parity-error results

The earliest custom-pipeline results reporting approximately 0%–8% success are
excluded from final performance reporting. They were produced before official
observation/action parity was restored and are retained only as an evaluation
pipeline error-diagnosis case:

`/home/xinrui_shen/vla/runs/pilot-final-preflight-deterministic-20260825`
(Git `e870c38eb025b942dca13e04de73c6cd595c4821`).

They are not pooled, plotted, averaged, or compared with either valid cohort.

## Limitations

- There are only 10 independent task clusters, with five initial states per task.
- Each condition has 50 episodes. The cluster-bootstrap intervals for success
  differences are correspondingly wide.
- Similar observed success does not establish formal non-inferiority.
- Adaptive-v2a's held-out trigger count is one; rare-trigger utility remains
  weakly identified even though the frozen value decision is negative.
- Wall time depends on hardware and system state. The confirmatory block schedule
  balanced order, but timing is still less portable than model-call counts.
- The conclusions apply to this checkpoint, LIBERO Spatial protocol, 280-step
  cap, generated chunk size 50, and execution horizons tested here.

## Paper-ready conclusion / 可用于论文的结论

### English

> On a parity-corrected 50-episode development cohort, Static-H20 retained a
> similar observed success rate to native H1 (64% versus 66%; paired difference
> −2 percentage points, task-cluster bootstrap 95% CI −16 to +14) while reducing
> mean model invocations by 94.7%. Adaptive-v1 was descriptively dominated by
> Static-H20. In a separate untouched 50-pair confirmatory cohort, Static-H20 and
> preregistered Adaptive-v2a produced identical outcomes (33/50 each; no paired
> flips), whereas Adaptive-v2a increased model calls and wall time. Under the
> preregistered decision rule, range-violation-triggered replanning showed no
> demonstrated value for its added computation. These results do not constitute
> a formal non-inferiority claim and are limited by 10 task clusters and 50
> episodes per condition.

### 中文

> 在完成评测 parity 修复后的 50 集开发队列中，Static-H20 与原生 H1 的
> 观测成功率接近（64% 对 66%；配对差 −2 个百分点，按 task 聚类 bootstrap
> 95% CI 为 −16 至 +14），同时将平均模型调用减少 94.7%。Adaptive-v1 在
> 描述性结果上被 Static-H20 支配。在另一套完全独立、未观察的 50 对正式
> held-out 队列中，Static-H20 与预注册 Adaptive-v2a 的 50 个配对结果完全
> 相同（均为 33/50，零 success flip），但 Adaptive-v2a 增加了模型调用和
> wall time。按照预注册决策规则，基于动作范围越界的自适应重规划没有显示出
> 足以补偿额外计算的实际价值。由于仅有 10 个 task cluster、每条件 50 集，
> 本结果不构成正式非劣效性结论。

## Reproducibility pointers

- Machine-readable condition and paired statistics:
  `analysis/final_vla_statistics.json`
- Flat final result table: `analysis/final_vla_results.csv`
- Source/result hashes and exclusions: `analysis/final_reproducibility_manifest.json`
- Figure data: `analysis/figures/final_vla_{success,model_calls,wall_time}.csv`
- Plotting script: `scripts/analysis/plot_final_vla_results.py`
- Rendered figures: `analysis/figures/generated/`

No model cache, video, or run directory is included in Git.
