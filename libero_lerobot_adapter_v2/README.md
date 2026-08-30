# LIBERO–LeRobot evaluation adapter

This directory contains the code and result artefacts used for the LIBERO Object
evaluation reported in Figure 5 of the project report. The evaluation compares
the checkpoint-native action execution with a smoother ten-step action chunk.

## Figure 5 protocol

- Benchmark: `libero_object`
- Tasks: all 10 tasks in the suite
- Rollouts: 3 per task and strategy (30 Native + 30 Smooth)
- Seed: 42
- Episode horizon: 500 control steps
- Batch size: 1
- Reset policy: hard reset between tasks
- Native strategy: checkpoint defaults
- Smooth strategy: `num_steps=10`, `n_action_steps=10`

The archived machine-readable results are in
`results/figure5_seed42/`. `suite_summary.json` contains the aggregate and
per-task results, while `task_summary.csv` provides the values used to plot the
figure. The reported totals are 22/30 (73.33%) for Native and 24/30 (80.00%)
for Smooth.

## Environment

The scripts use the existing LeRobot virtual environment and a locally installed
LIBERO benchmark. In WSL:

```bash
cd ~/lerobot
source .venv/bin/activate
cd /mnt/c/Users/13636/Documents/Codex/2026-08-15/sh/outputs/libero_lerobot_adapter_v2
python3 interactive_grasp.py --list
```

## Reproduce the Figure 5 comparison

The command below runs both strategies with the fixed ordering used for the
archived comparison:

```bash
python3 interactive_grasp.py \
  --all-tasks \
  --attempts 3 \
  --batch-size 1 \
  --episode-length 500 \
  --strategy-order fixed \
  --seed 42 \
  --run
```

Results are written under `outputs/interactive_eval/<timestamp>_all_tasks/`.
Each run contains a `suite_summary.json`, a `task_summary.csv`, and per-strategy
rollout outputs. Omitting `--run` performs a dry run and prints the commands
without launching the simulator.

To run the newer matched-seed Native/Smooth/Router comparison, use:

```bash
bash run_matched_seed44.sh
```

Set `RUN_HYBRID=1` to add the separate recovery experiment. The router excludes
the current evaluation seed from historical evidence to avoid evaluation
leakage.

### Completed matched-seed results

The completed seed-44 comparison used the same ten tasks, three rollouts per
task, batch size one, hard resets, and a 500-step episode horizon for all three
strategies:

| Strategy | Successes | Success rate | Wilson 95% CI | Total time |
| --- | ---: | ---: | ---: | ---: |
| Native | 23/30 | 76.67% | 59.1-88.2% | 4611.85 s |
| Smooth | 24/30 | 80.00% | 62.7-90.5% | 1103.15 s |
| Task-aware router | 25/30 | 83.33% | 66.4-92.7% | 3569.93 s |

The router excluded seed 44 from its historical evidence and ignored repeated
same-seed summaries before calculating its Laplace-smoothed task scores. The
confidence intervals overlap, so the router result is reported as the highest
observed point estimate rather than statistically significant superiority.

Machine-readable aggregate and per-task results are stored under
`results/seed44_matched/{native,smooth,router}/`. The report figure is archived
as `paper_figures/vla-router-seed44.pdf`.

## Tests and figure generation

```bash
python3 -m unittest -v
python3 paper_figures/make_pc_action_chunk_figure.py
```

The test suite covers command construction, strategy selection, result parsing,
confidence intervals, history deduplication, and seed exclusion. The plotting
script reads the archived summary and produces the Figure 5 PDF/PNG under
`paper_figures/`.

## Main files

- `interactive_grasp.py`: suite runner, strategy definitions, summaries, and routing
- `libero_pipeline.py`: LeRobot/LIBERO evaluation pipeline
- `test_pipeline.py`: regression tests
- `run_matched_seed44.sh`: matched-seed evaluation entry point
- `paper_figures/make_pc_action_chunk_figure.py`: Figure 5 generation
- `paper_figures/vla-router-seed44.pdf`: matched-seed result figure
- `results/figure5_seed42/`: archived aggregate and per-task results
- `results/seed44_matched/`: matched Native, Smooth, and Router results


## Seed-45 recovery and RGB verification evidence

The exploratory seed-45 hybrid run first achieved 28/30 successes with task-aware routing. Alphabet soup and butter each contained one failed smooth rollout and triggered one native fallback attempt. Butter recovered and alphabet soup did not, giving 1/2 recovery attempts, 29/32 successes when initial and fallback attempts are pooled, and 530.32 s of additional recovery time. Because the fallback attempts used disjoint initial conditions and both tasks had already succeeded in 2/3 initial rollouts, this is trigger-coverage evidence rather than a causal recovery claim. Machine-readable results are in results/seed45_recovery/.

The archived RGB pilot uses four independently annotated target boxes from two classes and IoU >= 0.50 matching. It produced TP=0, FP=3 and FN=4, so precision, recall and F1 were all zero. This deliberately negative audit supports excluding the detector from formal policy execution and does not estimate benchmark-wide accuracy. Reproduce it with:

    python3 evaluate_rgb_verification.py

Annotations and metrics are stored in rgb_annotations_pilot.json and results/rgb_verification_pilot/.
