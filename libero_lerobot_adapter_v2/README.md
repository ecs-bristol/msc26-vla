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
- `results/figure5_seed42/`: archived aggregate and per-task results

