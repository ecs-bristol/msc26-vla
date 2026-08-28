# XS adaptive figure bundle

`xs_adaptive_mechanism.pdf` is a vector method diagram. The quantitative
trade-off is available as vector PDF and 300 dpi PNG. Its plotted rows are
saved in `xs_success_compute_source.csv`; `xs_figure_provenance.json` records
the two frozen quantitative inputs and their SHA-256 hashes.

## Reproduce the figures

From the repository root, using Python 3 with Pillow and ReportLab installed:

```bash
python scripts/analysis/plot_xs_adaptive_results.py --root .
```

The script validates the selected CSV rows against the frozen statistics JSON
before writing either quantitative output.

## Real LIBERO evidence-strip audit

The frozen held-out run contains one verified full trigger chain in
`task_00_seed_1001_state_1.json`: task ID 0, initial-state ID 1, environment
seed 1001, inference seed 980056247616686888, trigger at environment step 222,
condition `Adaptive-v2a-H20→H1`, Git SHA
`a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`. The realized call window is
`2, 1, 20`; clipping is disabled and the current native action is unchanged.

No video, saved frame, image array, or replay cache exists in the formal run
tree, so no screenshot strip is generated. The following is the only bounded
replay command supplied. It executes exactly the verified pairing key and one
adaptive condition into a new output directory. It deliberately refuses to run
unless `REPLAY_ROOT` is a clean worktree at the frozen SHA. It has not been run
by Codex.

```bash
REPLAY_ROOT=/mnt/d/VLA/msc26-vla-a9afdc0 && \
test "$(git -C "$REPLAY_ROOT" rev-parse HEAD)" = a9afdc0b4feee120f5c3c71f22d84c691ed85ba6 && \
test -z "$(git -C "$REPLAY_ROOT" status --porcelain)" && \
test ! -e /home/xinrui_shen/vla/runs/xs-figure-replay/task0-state1-seed1001-a9afdc0 && \
cd "$REPLAY_ROOT" && \
python3 scripts/analysis/libero_spatial_paired_pilot.py --dry-run \
  --config configs/evaluation/libero_spatial_adaptive_v2_formal_heldout.yaml \
  --output-dir /home/xinrui_shen/vla/runs/xs-figure-replay/task0-state1-seed1001-a9afdc0 \
  --base-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --vlm-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 && \
python3 scripts/analysis/libero_spatial_paired_pilot.py --execute \
  --config configs/evaluation/libero_spatial_adaptive_v2_formal_heldout.yaml \
  --output-dir /home/xinrui_shen/vla/runs/xs-figure-replay/task0-state1-seed1001-a9afdc0 \
  --base-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --vlm-snapshot-path /home/xinrui_shen/vla/hf-cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --device cuda --task-id 0 --condition 'Adaptive-v2a-H20→H1' \
  --pairing-key-file /mnt/d/VLA/msc26-vla/figures/xs_adaptive/xs_verified_trigger_pairing_key.json
```

Any frames recorded in a user-authorized extension of this replay must be
captioned as **non-formal mechanism coverage**, never as success-rate evidence.
