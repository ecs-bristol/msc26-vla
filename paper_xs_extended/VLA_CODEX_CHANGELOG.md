# VLA Codex changelog

## 2026-08-28 — XS adaptive figures and manuscript integration

### Scope and provenance

- Added two high-value figures; no third figure was created because the saved
  real rollout artifacts contain no frames or videos.
- Quantitative data were read only from:
  - `analysis/final_vla_results.csv`
  - `analysis/final_vla_statistics.json`
- The plotting script cross-checks every selected CSV value against the frozen
  JSON, requires strict cohort separation, and rejects a missing early-parity
  exclusion before drawing.
- Figure-level input/output SHA-256 hashes are recorded in
  `figures/xs_adaptive/xs_figure_provenance.json`.

### New artifacts

- `scripts/analysis/plot_xs_adaptive_results.py`
- `figures/xs_adaptive/xs_adaptive_mechanism.pdf` — vector PDF.
- `figures/xs_adaptive/xs_success_compute_tradeoff.pdf` — vector PDF.
- `figures/xs_adaptive/xs_success_compute_tradeoff.png` — 300 dpi PNG.
- `figures/xs_adaptive/xs_success_compute_source.csv` — exact plotted rows.
- `figures/xs_adaptive/xs_figure_provenance.json` — input/output hashes and
  plotting constraints.
- `figures/xs_adaptive/README.md` — real-evidence audit and the sole bounded
  replay command (not executed).
- `figures/xs_adaptive/xs_verified_trigger_pairing_key.json` — one-key,
  outcome-labelled non-formal replay selector for task 0 / state 1 / seed 1001.
- The manuscript bundle mirrors these figure artifacts under
  `vla/figures/xs_adaptive/` so Overleaf can use its existing `vla/figures/`
  path convention.

### Manuscript changes

- Added the review-mode `codexadded` environment and
  `[Codex-added figure for review]` caption tag in `main.tex`.
- In review mode, the added figure labels and full captions are explicitly red;
  switching to `\codexreviewfalse` removes both the red markup and review tag.
- Added the XS adaptive method subsection, method-figure reference and caption.
- Added the XS success--compute results subsection, trade-off reference and
  caption.
- All new figure environments, captions and body references are enclosed by
  `\begin{codexadded}` / `\end{codexadded}`.
- Captions explicitly distinguish descriptive development evidence, paired
  within-cohort comparisons, and untouched held-out confirmation. No
  cross-cohort statistic is reported.

### Real LIBERO evidence-strip audit

- Read-only inspection covered the checked-in `runs/` tree and the WSL frozen
  run directory
  `/home/xinrui_shen/vla/runs/adaptive-v2-prereg/adaptive-v2a-formal-heldout-100-a9afdc0-20260828`.
- One complete real telemetry chain exists in
  `episodes/adaptive-v2a-h20-to-h1/task_00_seed_1001_state_1.json`:
  task ID 0, initial-state ID 1, environment seed 1001, inference seed
  980056247616686888, trigger at environment step 222, condition
  `Adaptive-v2a-H20→H1`, formal held-out source run, Git SHA
  `a9afdc0b4feee120f5c3c71f22d84c691ed85ba6`, realized call window
  `[2, 1, 20]`.
- No MP4/AVI/MOV/MKV/WebM, PNG/JPEG, NPY or NPZ artifact exists in that run
  tree. Consequently no screenshot strip was generated and no rollout was
  started by Codex.
- Any user-authorized replay after inspecting the outcome is non-formal
  mechanism coverage and must not be presented as success-rate evidence.

### Existing teammate-figure review

- All ten PDFs under `vla/figures/` were rendered and visually checked against
  their nearby manuscript captions/tables. Seven are vector-only. The three
  PDFs containing raster images resolve to approximately 300 dpi at their PDF
  page size; no low-resolution replacement is required.
- No existing figure was deleted or replaced.
- `pc_action_chunk_results.pdf` is not referenced by `vla/mainbody.tex` and
  duplicates the aggregate Native--Smooth comparison already presented by
  `vla_action_chunk_results.pdf`. **Original-author confirmation requested:**
  decide whether the orphan duplicate should remain in the submission bundle.
- Existing Fig. `vla_runtime_parameters.pdf` defines `H` as prediction horizon
  and `E` as execution horizon, while the frozen adaptive condition names use
  `H1/H20` for execution. A red manuscript review note now disambiguates the
  condition labels as `E=1/E=20` with fixed prediction chunk `C=50`.
  **Original-author confirmation requested:** retain this convention or rename
  the conditions consistently throughout the paper and frozen result tables.
- No existing figure was found to pool the development and held-out adaptive
  cohorts. No silent numerical or caption change was made.

### Verification

- Figure PDFs were rasterized for visual QA; labels, colours, full axes and
  panel separation were checked at manuscript-scale aspect ratios.
- The quantitative PDF contains no raster image XObjects; the mechanism PDF is
  also vector-only.
- A 25-page VLA-only integration build completed successfully with all new
  references resolved. Visual inspection placed the mechanism figure on page
  12 and the success--compute figure on page 17; both fit the text width, have
  legible captions, and show no clipping. The final added text introduced no
  overfull-box warning.
- A complete top-level build remains blocked by pre-existing teammate LaTeX
  errors: undefined `\order` in `yolo/mainbody.tex` and malformed nested
  `\label{\label{...}}` commands in `agent/mainbody.tex`. Those files were not
  edited. **Original-author confirmation requested:** repair those unrelated
  sources before the final Overleaf compilation.
- The local QA wrapper used Tectonic with BibTeX because the portable engine did
  not provide Biber; the deliverable retains the paper's original Biber setup.
