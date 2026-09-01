# Experiment Control Dashboard Design

Date: 2026-07-11

## 1. Purpose

This document defines a complete visual experiment control dashboard for the Jetson VLA/VLM benchmark project.

The dashboard should let the project operator configure, launch, monitor, compare, and archive a full experiment from one local web interface. It should connect directly to the existing benchmark framework instead of becoming a separate presentation-only page.

The primary design goal is:

> Turn the current project from a collection of scripts and result folders into a controlled experimental cockpit that can produce thesis-ready evidence.

## 2. Current Project Context

The project already has several usable pieces:

- `Final_Project/Benchmark_UI/server.py`
  - Local HTTP server.
  - Existing `/api/config`, `/api/results`, `/api/start`, `/api/jobs`, `/api/job`, and `/api/file` endpoints.
  - Whitelisted job execution for pre-Jetson, scripted robosuite, VLM pick, PC matrix, and Tiny BC workflows.

- `Final_Project/Benchmark_UI/static/core.html`
  - Existing dashboard shell.
  - Side controls for PC matrix and Tiny BC demo.
  - Result table, trial selector, frame viewer, and gallery.

- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/unified_runner.py`
  - Unified adapter runner for model/policy experiments.
  - Supports dry-run and real run modes.
  - Produces structured output through `unified_results.py`.

- `Final_Project/Local_VLA_Benchmark_Framework/src/vla_bench/unified_schema.py`
  - Unified trial schema.
  - Deployment modes include `pc_local`, `jetson_local`, `jetson_quantized`, `jetson_remote_client`, `remote_server`, and `mock`.

- `Final_Project/Evidence/evidence_index.csv`
  - Evidence registry for selected outputs.
  - Recent selected results include unified interface smoke evidence.

The new dashboard should build on this foundation.

## 3. Dashboard Product Definition

### 3.1 Name

Working name:

**VLA Experiment Control Center**

### 3.2 Intended User

Primary user:

- The project author running experiments before and after Jetson hardware arrival.

Secondary users:

- Supervisor or examiner reviewing experimental progress.
- Future project maintainer reproducing selected benchmark runs.

### 3.3 Core Promise

The control panel should answer five questions at all times:

1. What experiment am I about to run?
2. What models, tasks, deployment modes, precision settings, and trial counts are included?
3. Is the experiment currently running, failed, paused, or completed?
4. What did it produce?
5. Is the output selected as Evidence for the report/thesis?

## 4. Contribution Alignment

The dashboard is not just a convenience feature. It supports the academic contribution structure:

| Academic contribution | Dashboard support |
| --- | --- |
| Unified Benchmark Protocol | Experiment builder uses one shared schema and runner family. |
| Deployment-Aware Evaluation | Deployment mode is a first-class control and result column. |
| Optimization / Compression Study | Precision, quantization, image size, and remote/local mode are visible variables. |
| Reusable Edge Deployment System | Jetson readiness and remote-client modes are controlled from the same interface. |
| Evidence-Based Evaluation | Selected outputs can be copied into Evidence and indexed from the UI. |

## 5. Scope

### 5.1 In Scope

The first complete version should support:

- Experiment planning from the UI.
- Dry-run preview before launching expensive runs.
- Launching whitelisted experiment types.
- Live job status and log tail.
- Result discovery across unified, pre-Jetson, PC matrix, robosuite, and selected Evidence outputs.
- Summary tables and trial-level inspection.
- Failure taxonomy visualization.
- Evidence selection workflow.
- Jetson readiness checklist.
- Exportable result tables for thesis writing.

### 5.2 Out of Scope

The first complete version should not include:

- Arbitrary shell command execution.
- Cloud authentication management.
- Multi-user permissions.
- Remote browser-based control of Jetson over the internet.
- Editing model source code from the dashboard.
- Training large models through the UI.

## 6. Information Architecture

The dashboard should have six main areas.

### 6.1 Left Navigation

Persistent vertical navigation:

1. Overview
2. Experiment Builder
3. Live Run
4. Results
5. Evidence
6. Jetson Readiness
7. Settings

### 6.2 Overview

Purpose:

- Give a fast status view of the whole experimental programme.

Content:

- Current project stage:
  - Protocol validated
  - PC model-scale experiments pending/running/completed
  - Optimization experiments pending/running/completed
  - Jetson local pending/running/completed
  - Jetson remote pending/running/completed

- Key cards:
  - Total selected Evidence runs
  - Latest run status
  - Best current local candidate
  - Latest failure type
  - Jetson readiness state

- Experiment matrix coverage:
  - Rows: A1, A2, B1, C1, C2, C3, D1, D2, E1, E2
  - Columns: planned, runnable, completed, selected evidence

### 6.3 Experiment Builder

Purpose:

- Configure an experiment without editing YAML or remembering commands.

Controls:

- Experiment type:
  - Unified interface smoke
  - Unified VLM PC smoke
  - PC model-scale benchmark
  - Robosuite scripted baseline
  - Robosuite VLM target pick
  - Tiny BC closed-loop
  - Jetson local benchmark
  - Jetson remote-client benchmark
  - Optimization / compression ablation

- Models:
  - Multi-select from `configs/models.yaml`.
  - Show `model_id`, adapter, supported deployment modes, dtype, quantization, notes.

- Tasks:
  - Multi-select from `data/tasks.csv`.
  - Show image thumbnail, prompt, expected target, expected action.

- Deployment:
  - `pc_local`
  - `remote_server`
  - `jetson_local`
  - `jetson_quantized`
  - `jetson_remote_client`
  - `mock`

- Runtime controls:
  - repeats
  - warmup
  - max new tokens
  - local files only
  - save frame sequence
  - frame stride
  - max steps

- Optimization controls:
  - precision: `FP32`, `FP16`, `BF16`, `INT8`, `4-bit`
  - quantization method: `none`, `bitsandbytes-8bit`, `bitsandbytes-4bit`, `tensorrt`, `onnx`
  - image size: `224`, `336`, `448`

Required actions:

- Dry run
- Start run
- Save preset

### 6.4 Live Run

Purpose:

- Show what is happening while an experiment is running.

Content:

- Active job card:
  - job id
  - experiment type
  - status
  - command preview
  - working directory
  - created time
  - elapsed time
  - return code

- Progress panel:
  - model currently running
  - task currently running
  - trial index
  - completed / total
  - estimated remaining time if available

- Log viewer:
  - log tail
  - full log download/open
  - error highlight

- Safe controls:
  - refresh
  - stop job
  - mark failed

The stop button should terminate only known child processes stored in the job registry. It must not run arbitrary shell commands.

### 6.5 Results

Purpose:

- Compare experiment outputs across runs.

Views:

- Run overview table
- Model comparison table
- Task comparison table
- Failure analysis table
- Trial detail drawer
- Frame/video viewer for robosuite outputs

Core columns:

- run id
- experiment
- model key
- model id
- adapter
- deployment mode
- runtime precision
- quantization
- task id
- trials
- success rate
- action valid rate
- auto-score pass rate
- latency mean
- latency p95
- peak memory
- load failures
- OOM count
- failure type
- recommendation

Visualizations:

- Success rate by model
- Latency by model
- Failure type distribution
- Deployment mode comparison
- Jetson candidate ranking

### 6.6 Evidence

Purpose:

- Turn raw experiment results into thesis-ready selected evidence.

Features:

- Show current `evidence_index.csv`.
- Show selected result folders.
- Button: "Select this run as Evidence".
- Button: "Copy summary / failures / metadata".
- Button: "Create evidence index rows".
- Evidence preview:
  - source path
  - selected path
  - related research question
  - selected/candidate status
  - notes

Evidence IDs:

- Next ID should be calculated from the existing index.
- The system should avoid overwriting existing selected results unless user confirms.

### 6.7 Jetson Readiness

Purpose:

- Make hardware deployment preparation visible.

Checklist:

- Jetson connected
- Python environment checked
- CUDA available
- camera/static image input available
- local small model tested
- quantized mode tested
- remote-client mode tested
- output copied back to Evidence

Status sources:

- Manual checkboxes stored locally.
- Optional environment check output from `env_check.py`.
- Latest Jetson experiment result directories.

## 7. User Flow

### 7.1 Standard PC Model-Scale Flow

1. Open dashboard.
2. Go to Experiment Builder.
3. Select "Unified VLM PC smoke" or "PC model-scale benchmark".
4. Select models:
   - `smolvlm2_500m`
   - `qwen2_vl_2b`
   - `paligemma2_3b`
   - `qwen2_5_vl_3b`
   - `openvla_7b_probe`
5. Select tasks.
6. Set repeats.
7. Click Dry run.
8. Review generated command and expected output directory.
9. Click Start run.
10. Monitor Live Run.
11. Inspect Results.
12. Select successful or informative failed outputs as Evidence.

### 7.2 Jetson Arrival Flow

1. Go to Jetson Readiness.
2. Run environment check.
3. Confirm device profile.
4. Run unified interface smoke on Jetson.
5. Run small VLM local test.
6. Run quantized test.
7. Run remote-client test.
8. Compare local vs remote results.
9. Register selected results as Evidence.

### 7.3 Optimization Flow

1. Choose best local candidate model.
2. Select optimization ablation.
3. Choose precision / quantization / image size variants.
4. Dry-run matrix.
5. Start run.
6. Compare latency, memory, output drift, and action validity.
7. Register summary and failure table as Evidence.

## 8. Backend Design

### 8.1 Keep Existing Backend Style

The current project uses a lightweight `ThreadingHTTPServer` in `server.py`. The next version should keep this style unless the UI becomes too complex.

Reasons:

- No extra web framework dependency.
- Easy to run on Windows.
- Good enough for local single-user experiment control.
- Already implemented and tested by the project.

### 8.2 New Endpoints

Existing endpoints should remain:

- `GET /api/config`
- `GET /api/results`
- `GET /api/jobs`
- `GET /api/job?id=...`
- `GET /api/file?path=...`
- `POST /api/start`

Add:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/experiments` | GET | List experiment presets from YAML and known scripts. |
| `/api/dry-run` | POST | Return command, models, tasks, output root, and estimated trial count. |
| `/api/start` | POST | Extend existing endpoint to support `unified_runner` jobs. |
| `/api/stop` | POST | Stop a running whitelisted job. |
| `/api/results/unified` | GET | List unified result directories and parsed outputs. |
| `/api/evidence` | GET | Parse `evidence_index.csv` and selected result folders. |
| `/api/evidence/select` | POST | Copy selected files and append evidence rows. |
| `/api/readiness` | GET | Return Jetson readiness checklist and latest check outputs. |
| `/api/readiness` | POST | Update manual readiness checklist. |

### 8.3 Job Kinds

Extend `create_job(payload)` to support these job kinds:

- `unified_runner`
- `pre_jetson`
- `pc_matrix`
- `scripted_pick`
- `vlm_pick`
- `tiny_bc_closed_loop`
- `jetson_env_check`
- `evidence_select`

Each job kind must have:

- explicit whitelist validation
- fixed runner path
- bounded numeric controls
- safe working directory
- log file
- public job state

### 8.4 Unified Runner Command Mapping

Payload:

```json
{
  "kind": "unified_runner",
  "experiment": "jetson_readiness_interface_smoke",
  "models": ["local_rule_baseline", "mock_remote_policy"],
  "tasks": ["desk_cup_pick", "blue_can_pick"],
  "repeats": 1,
  "warmup": 0,
  "max_new_tokens": 64,
  "device_profile": "windows_pc_pre_jetson",
  "dry_run": false
}
```

Command:

```powershell
.\.venv\Scripts\python.exe -m src.vla_bench.unified_runner `
  --experiment jetson_readiness_interface_smoke `
  --models local_rule_baseline,mock_remote_policy `
  --tasks desk_cup_pick,blue_can_pick `
  --repeats 1 `
  --warmup 0 `
  --max-new-tokens 64 `
  --device-profile windows_pc_pre_jetson
```

### 8.5 Result Parsing

Add `list_unified_runs()` in `server.py`.

It should read:

```text
Final_Project/Local_VLA_Benchmark_Framework/results/unified/<run_id>/
  metadata.json
  summary.csv
  failures.csv
  trials.csv
  trials.jsonl
```

Return:

```json
{
  "id": "jetson_readiness_interface_smoke_20260707_162504_c6fa96f7",
  "kind": "unified",
  "path": "...",
  "metadata": {},
  "summary": [],
  "failures": [],
  "trials": []
}
```

## 9. Frontend Design

### 9.1 Interaction Style

This is an operational research dashboard, not a marketing page.

Visual style:

- dense but readable
- restrained colours
- clear status indicators
- compact controls
- tables optimized for comparison
- no decorative hero section

### 9.2 First View Layout

The first screen should immediately show the actual experiment control surface.

Layout:

```text
+--------------------------------------------------------------------------------+
| Top bar: Project status | Latest run | Evidence count | Jetson readiness        |
+----------------------+---------------------------------------------------------+
| Left navigation      | Main workspace                                           |
|                      |                                                         |
| Overview             | Current selected section                                 |
| Experiment Builder   |                                                         |
| Live Run             |                                                         |
| Results              |                                                         |
| Evidence             |                                                         |
| Jetson Readiness     |                                                         |
| Settings             |                                                         |
+----------------------+---------------------------------------------------------+
```

### 9.3 Experiment Builder Layout

```text
+----------------------------+------------------------------+-------------------+
| Experiment preset          | Models                       | Tasks             |
| Deployment mode            | Runtime/optimization         | Trial settings    |
+----------------------------+------------------------------+-------------------+
| Dry-run preview: generated command, output directory, trial count              |
+--------------------------------------------------------------------------------+
| Start run | Save preset | Reset                                                   |
+--------------------------------------------------------------------------------+
```

### 9.4 Live Run Layout

```text
+----------------------------+-----------------------------------------------+
| Active job summary         | Progress                                       |
| status / elapsed / command | model / task / trial / completed count        |
+----------------------------+-----------------------------------------------+
| Log tail                                                                   |
+----------------------------------------------------------------------------+
| Stop | Refresh | Open output directory                                     |
+----------------------------------------------------------------------------+
```

### 9.5 Results Layout

Tabs:

- Runs
- Model comparison
- Tasks
- Failures
- Trials
- Frames

The model comparison tab should be the default because it directly supports the report.

### 9.6 Evidence Layout

Two-column layout:

- Left: available runs
- Right: evidence preview and selection form

Fields:

- evidence type
- title
- related research question
- status: selected / candidate
- notes

Actions:

- copy selected files
- append index rows
- open selected evidence folder

## 10. Data Model

### 10.1 Dashboard Run Object

```json
{
  "id": "string",
  "kind": "unified | pc_matrix | pre_jetson | vlm_pick | scripted | tiny_bc",
  "path": "string",
  "created_at": "string",
  "metadata": {},
  "summary": [],
  "trials": [],
  "failures": [],
  "evidence_status": "none | candidate | selected"
}
```

### 10.2 Dashboard Job Object

```json
{
  "id": "string",
  "kind": "string",
  "status": "queued | running | completed | failed | stopped",
  "created_at": "string",
  "started_at": "string",
  "finished_at": "string",
  "cwd": "string",
  "cmd": [],
  "log_path": "string",
  "return_code": 0,
  "log_tail": "string",
  "output_dir": "string"
}
```

### 10.3 Evidence Selection Object

```json
{
  "source_run_id": "string",
  "source_paths": {
    "summary": "string",
    "failures": "string",
    "metadata": "string",
    "trials": "string"
  },
  "selected_folder": "Final_Project/Evidence/selected_results/<slug>",
  "related_question": "string",
  "notes": "string",
  "status": "selected"
}
```

## 11. Safety and Reproducibility

### 11.1 Safety Rules

- All launched commands must be constructed from whitelisted job kinds.
- The frontend must never send a raw command string for execution.
- Paths returned by `/api/file` must stay inside the workspace.
- Numeric fields must have conservative bounds.
- Stop action must only target tracked job child processes.
- Evidence selection must not overwrite existing selected folders unless explicitly confirmed.

### 11.2 Reproducibility Rules

Every run should capture:

- git commit
- experiment name
- model list
- task list
- deployment mode
- runtime precision
- quantization
- repeats
- warmup
- max new tokens
- device profile
- created timestamp
- command
- output directory

## 12. Implementation Phases

### Phase 1: Unified Dashboard Backbone

Goal:

- Make the current UI understand unified runner results and launch unified experiments.

Tasks:

- Add `list_unified_runs()` to `server.py`.
- Add unified runs to `/api/results`.
- Add `kind == "unified_runner"` to `create_job()`.
- Add dry-run endpoint for unified runner.
- Add frontend experiment selector for unified smoke and VLM PC smoke.
- Display unified summary/failure/trial tables.

Validation:

- UI can dry-run `jetson_readiness_interface_smoke`.
- UI can start `jetson_readiness_interface_smoke`.
- UI displays the generated `summary.csv` and `failures.csv`.

### Phase 2: Experiment Builder

Goal:

- Replace hard-coded PC matrix controls with a general experiment builder.

Tasks:

- Render models from `models.yaml`.
- Render tasks from `tasks.csv`.
- Render deployment mode options from `unified_schema.py` or a mirrored API list.
- Add repeats, warmup, max tokens, local-files-only, precision, quantization, and image-size controls.
- Show generated command in dry-run preview.

Validation:

- Invalid model/task combinations are rejected.
- Dry-run preview matches actual runner payload.
- No arbitrary command execution is possible.

### Phase 3: Live Run Monitor

Goal:

- Make long experiments observable.

Tasks:

- Add job elapsed time.
- Add output directory once detected.
- Add stop endpoint.
- Add log severity highlighting.
- Poll active job until terminal state.

Validation:

- Running job updates status.
- Completed job appears in Results without manual refresh.
- Failed job shows return code and log tail.

### Phase 4: Evidence Pipeline

Goal:

- Select report-ready outputs from the UI.

Tasks:

- Parse `evidence_index.csv`.
- Add `/api/evidence`.
- Add `/api/evidence/select`.
- Copy summary/failures/metadata/trials to `Evidence/selected_results/<slug>/`.
- Append evidence rows with next available EV id.

Validation:

- Selecting a run creates selected result files.
- `evidence_index.csv` contains the new rows.
- Re-selecting an existing slug asks for confirmation.

### Phase 5: Jetson Readiness and Optimization Panels

Goal:

- Support hardware arrival and optimization experiments.

Tasks:

- Add Jetson readiness checklist.
- Add environment check launcher.
- Add deployment-mode comparison view.
- Add optimization matrix builder for precision, quantization, and image size.

Validation:

- Jetson checklist persists.
- Optimization result table can compare at least two precision/image-size settings.
- Deployment mode view compares local vs remote-client runs.

## 13. Acceptance Criteria

The dashboard design is implemented when:

1. A user can run a unified smoke experiment from the UI.
2. A user can run or dry-run a model-scale experiment from the UI.
3. A user can see job logs while the experiment is running.
4. A user can inspect summary, failures, metadata, and trials after completion.
5. A user can compare models by success rate, latency, OOM count, load failures, and recommendation.
6. A user can select a run as Evidence without manually copying files.
7. A user can see which experiment matrix cells are planned, runnable, completed, and selected as Evidence.
8. The backend never executes arbitrary user-provided shell commands.

## 14. Recommended First Implementation Slice

The first implementation should be deliberately narrow:

**Build unified runner support into the current dashboard.**

Minimum useful slice:

- Add unified result discovery.
- Add unified experiment dry-run.
- Add unified experiment start.
- Add result display for unified summary/failures/trials.
- Add "select as Evidence" for unified runs.

This slice directly supports the next project phase: model-scale evaluation before Jetson hardware arrives.

## 15. Open Decisions

These decisions should be confirmed before implementation:

1. Should the dashboard remain framework-free static HTML/JS, or move to a small frontend framework later?
2. Should Evidence selection append directly to `evidence_index.csv`, or create a pending review file first?
3. Should long-running model jobs allow stop/terminate in the first implementation slice?
4. Should Jetson remote control be local-network only for this project?
5. Should optimization experiments be represented as separate presets or as a matrix builder mode?

Recommended defaults:

- Keep framework-free static HTML/JS for now.
- Append directly to `evidence_index.csv`, but require confirmation on overwrite.
- Add stop/terminate in Phase 3, not Phase 1.
- Treat Jetson as local-network only.
- Represent optimization as matrix builder mode.
