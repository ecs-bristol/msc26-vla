# Final VLA Results Tables

This file freezes the reportable LIBERO Spatial results at repository commit
`a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`.

**Statistical firewall:** Cohort A and Cohort B use different inference-seed namespaces and are never pooled or treated as one paired sample. All paired differences below are computed only within the named cohort.

## A. Parity-corrected development cohort

Git SHA: `b5cbf91a7c1a47ac48b7d30ed318ecd2ea252d1a`  
Manifest SHA-256: `934a4887e2ddea7703d43db727b3c66416652b3cd3806e77ed0a526bca1d44f2`

| Condition | Success@280 | Wilson 95% CI | Calls mean / median | Inference mean / median (s) | Wall mean / median (s) | Chunk utilization, mean / aggregate | Trigger / rescue / loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static-H1 | 33/50 (66%) | 52.2%–77.6% | 162.48 / 122 | 137.651 / 90.231 | 248.815 / 190.928 | 2.00% / 2.00% | 0 / 0 / 0 |
| Static-H10 | 30/50 (60%) | 46.2%–72.4% | 17.48 / 12.5 | 9.841 / 7.345 | 61.400 / 47.711 | 19.44% / 19.68% | 0 / 0 / 0 |
| Static-H20 | 32/50 (64%) | 50.1%–75.9% | 8.66 / 6 | 5.741 / 4.366 | 54.569 / 42.519 | 37.72% / 38.50% | 0 / 0 / 0 |
| Adaptive-v1-H20→H1 | 31/50 (62%) | 48.2%–74.1% | 9.08 / 7 | 5.942 / 4.639 | 55.430 / 43.617 | 37.04% / 37.57% | 7 / 0 / 1 |

The seven Adaptive-v1 triggers occurred in six episodes. At event level there were zero rescues, one loss, and six no-change triggers. At paired-outcome level Adaptive-v1's success set was a strict subset of Static-H20's success set.

### Development paired success statistics

`first_only` and `second_only` follow the comparison direction in the first column.

| Comparison (first − second) | Both success | First only | Second only | Both fail | Success difference | Task-cluster bootstrap 95% CI | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static-H10 − Static-H1 | 27 | 3 | 6 | 14 | −6 pp | [−16, +4] pp | 0.5078 |
| Static-H20 − Static-H1 | 26 | 6 | 7 | 11 | −2 pp | [−16, +14] pp | 1.0000 |
| Static-H20 − Static-H10 | 25 | 7 | 5 | 13 | +4 pp | [−12, +20] pp | 0.7744 |
| Adaptive-v1 − Static-H20 | 31 | 0 | 1 | 18 | −2 pp | [−6, 0] pp | 1.0000 |

### Development paired resource differences

| Comparison (first − second) | Calls mean difference (95% CI) | Inference-time mean difference (95% CI), s | Wall-time mean difference (95% CI), s |
|---|---:|---:|---:|
| Static-H20 − Static-H1 | −153.82 [−175.86, −132.46] | −131.910 [−164.128, −104.827] | −194.246 [−230.439, −159.335] |
| Static-H20 − Static-H10 | −8.82 [−10.64, −7.00] | −4.100 [−5.217, −2.977] | −6.831 [−14.551, +1.138] |
| Adaptive-v1 − Static-H20 | +0.42 [+0.08, +0.90] | +0.201 [+0.013, +0.460] | +0.861 [−0.771, +3.004] |

## B. Untouched held-out confirmatory cohort

Git SHA: `a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`  
Manifest SHA-256: `2156db1ca5fe906e3f1a7ebb76d6ff62f1521456a6e1550c4aab2737e65fc357`

| Condition | Success@280 | Wilson 95% CI | Calls mean / median | Inference mean / median (s) | Wall mean / median (s) | Chunk utilization, mean / aggregate | Trigger / rescue / loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static-H20 | 33/50 (66%) | 52.2%–77.6% | 8.32 / 6 | 5.558 / 4.054 | 50.872 / 41.226 | 38.17% / 38.80% | 0 / 0 / 0 |
| Adaptive-v2a-H20→H1 | 33/50 (66%) | 52.2%–77.6% | 8.36 / 6 | 5.676 / 4.083 | 53.558 / 43.928 | 38.07% / 38.61% | 1 / 0 / 0 |

### Confirmatory paired statistics

| Both success | Adaptive only | Static only | Both fail | Success difference | Task-cluster bootstrap 95% CI | McNemar exact p |
|---:|---:|---:|---:|---:|---:|---:|
| 33 | 0 | 0 | 17 | 0 pp | [0, 0] pp | 1.0000 |

| Paired metric, Adaptive-v2a − Static-H20 | Mean difference | Task-cluster bootstrap 95% CI |
|---|---:|---:|
| Model calls | +0.04 | [0.00, +0.12] |
| Inference time | +0.118 s | [−0.053, +0.288] s |
| Wall time | +2.685 s | [+1.113, +4.537] s |

One held-out trigger completed the full pre-execution `H20 → H1 → H20` state chain. It was a no-change event: Static-H20 and Adaptive-v2a both failed that pairing key. The frozen preregistered decision is **“no demonstrated value for the added calls.”**

## Excluded diagnostic cohort

The earlier `0%–8%` runs under
`/home/xinrui_shen/vla/runs/pilot-final-preflight-deterministic-20260825`
(Git `e870c38eb025b942dca13e04de73c6cd595c4821`) predate the parity correction.
They are excluded from every performance table and paired statistic. They may be cited only as an evaluation-pipeline error-diagnosis case.

## Statistical limits

- Ten task clusters and 50 episodes per condition yield wide success-effect confidence intervals.
- Static-H20's near-H1 success rate is descriptive; this experiment does not establish formal non-inferiority.
- Adaptive-v2a triggered once in the held-out cohort, so the study provides little information about rare-trigger behavior beyond the frozen negative decision.
- Wall time is hardware- and system-sensitive, although the confirmatory execution order was block-balanced.
