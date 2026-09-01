# VLA E-Logbook Evidence Appendix

This folder is the GitHub evidence appendix for the VLA-direction part of the EEMEM0017 weekly e-logbook. It is not the project README or a standalone technical report.

Use this index when following a `GitHub evidence` link from the e-logbook: each Week heading mirrors the logbook's chronological Week 1–13 structure and points to the supporting plans, code snapshots, run records and result artefacts for that entry.

## Provenance note

- Files under `archive/week_01` to `archive/week_08` are preserved copies of the original local project artefacts assembled into this branch at project close. Their Git commit date is the archive date, not the original creation date. Original timing is supported by filenames, embedded metadata, result records and the main project Git history where available.
- Files already under `evidence/latest` were retained in their formal evaluation structure. Raw CSV/JSON/JSONL files remain separate from later report figures.
- The five files under `system_design` are retrospective architecture diagrams prepared at project close from the contemporaneous plans, code, Git history and experiment records. They are not claimed as figures drawn during the earlier weeks.

See [EVIDENCE_MANIFEST.csv](EVIDENCE_MANIFEST.csv) for the machine-readable mapping.

## Week 1

Project framing, related work and the decision to build a reproducible VLA deployment benchmark.

- [Related-work and baseline notes](archive/week_01/OpenVLA_Jetson_Related_Work_and_Baselines.md)
- [Initial VLA deployment benchmark project summary](archive/week_01/Jetson_VLA_Deployment_Benchmark_Project_Summary_ZH.md)

## Week 2

Initial project plan, task schedule and deployment-route concept.

- [Project-plan summary](archive/week_02/project_summary.csv)
- [Gantt task data](archive/week_02/gantt_tasks.csv)
- [Gantt chart](archive/week_02/gantt_chart.png)
- [Retrospective initial-deployment diagram](system_design/D-W02-01_initial_deployment.png)

## Week 3

First local benchmark framework and traceable JSONL/CSV result output.

- [Benchmark framework notes](archive/week_03/local_benchmark_README_ZH.md)
- [Benchmark summary](archive/week_03/benchmark_summary_20260619_115936.csv)
- [Trial-level benchmark results](archive/week_03/benchmark_results_20260619_115936.jsonl)
- [Evidence index snapshot](archive/week_03/evidence_index.csv)
- [Benchmark runner snapshot](archive/week_03/benchmark.py)
- [Inference entry-point snapshot](archive/week_03/run_inference.py)

## Week 4

Experiment matrix, result schema and weekly evidence-delivery structure.

- [Literature shortlist](archive/week_04/Jetson_VLA_Top10_Papers_2026-06-25.md)
- [Implementation workflow](archive/week_04/IMPLEMENTATION_WORKFLOW_ZH.md)
- [Experiment matrix](archive/week_04/EXPERIMENT_MATRIX.csv)
- [Result-log schema](archive/week_04/RESULT_LOG_SCHEMA.csv)
- [Weekly deliverables](archive/week_04/WEEKLY_DELIVERABLES.csv)

## Week 5

Pre-Jetson simulation validation using Qwen2-VL perception and deterministic Cartesian control.

- [Offline VLM smoke summary](archive/week_05/offline_vlm_smoke_summary.csv)
- [10-trial robosuite summary](archive/week_05/vlm_target_pick_summary.csv)
- [Run metadata](archive/week_05/vlm_target_pick_metadata.json)
- [Archived red-block final frame](archive/week_05/red_block_final.jpg)
- [Simulation benchmark implementation](archive/week_05/vlm_target_pick_benchmark.py)
- [Deterministic Cartesian controller](archive/week_05/cartesian_pick_policy.py)
- [Retrospective perception-to-control diagram](system_design/D-W05-01_perception_control_prototype.png)

## Week 6

Unified model adapters, normalised result schema and failure taxonomy.

- [Unified-interface workflow](archive/week_06/UNIFIED_INTERFACE_WORKFLOW_ZH.md)
- [Selected summary](archive/week_06/summary.csv)
- [Run metadata](archive/week_06/metadata.json)
- [Trial records](archive/week_06/trials.csv)
- [Failure records](archive/week_06/failures.csv)
- [Adapter snapshot](archive/week_06/adapters.py)
- [Schema snapshot](archive/week_06/unified_schema.py)
- [Runner snapshot](archive/week_06/unified_runner.py)
- [Dashboard design specification](archive/week_06/experiment_control_dashboard_design.md)

## Week 7

Persistent experiment console, preflight hardening, telemetry and result inspection.

- [Visual experiment console design](archive/week_07/visual_experiment_control_console_design.md)
- [Benchmark UI notes](archive/week_07/Benchmark_UI_README_ZH.md)
- [Server snapshot](archive/week_07/server.py)
- [Orchestrator snapshot](archive/week_07/orchestrator.py)
- [Result handling snapshot](archive/week_07/results.py)
- [Console schema snapshot](archive/week_07/schema.py)

## Week 8

Transition from console-led work to versioned YAML/CLI execution.

- [Pre-Jetson workflow specification](archive/week_08/PRE_JETSON_WORKFLOW_ZH.md)
- [Versioned YAML configuration](archive/week_08/pre_jetson_workflow.yaml)
- [PowerShell execution entry point](archive/week_08/run_pre_jetson_workflow.ps1)
- [Command-shell execution entry point](archive/week_08/run_pre_jetson_workflow.cmd)
- [Retrospective reproducible-platform diagram](system_design/D-W08-01_reproducible_platform.png)

## Week 9

LIBERO benchmark-platform scaffold, immutable configuration and deterministic execution.

- [Deployment profiles](../../configs/deployment_profiles.yaml)
- [LIBERO suites](../../configs/libero_suites.yaml)
- [Official evaluation design](../../docs/superpowers/specs/2026-08-13-official-lerobot-eval-jetson-remote-policy-design.md)
- [Repository commit history](https://github.com/ecs-bristol/msc26-vla/commits/codex/elog-evidence)

## Week 10

Split PC/Jetson architecture with a remote SmolVLA policy service.

- [Official remote-policy guide](../../docs/OFFICIAL_LEROBOT_JETSON_REMOTE.md)
- [PC/Jetson network guide](../../docs/JETSON_PC_NETWORK.md)
- [Remote evaluation entry point](../../scripts/wsl/run_official_jetson_remote_eval.sh)
- [Remote-policy installation script](../../scripts/wsl/install_remote_jetson_policy.sh)
- [Remote evaluation record](../latest/jetson_remote/eval_info.json)
- [Remote transport log](../latest/jetson_remote/remote_transport.jsonl)
- [Retrospective split-architecture diagram](system_design/D-W10-01_split_pc_jetson.png)

## Week 11

Formal `num_steps` selection and reproducible PC evidence.

- [Num-steps evidence notes](../latest/pc_local/num_steps/README.md)
- [Num-steps summary](../latest/pc_local/num_steps/num_steps_summary.csv)
- [Per-task results](../latest/pc_local/num_steps/num_steps_per_task.csv)
- [Jetson remote evaluation record](../latest/jetson_remote/eval_info.json)

## Week 12

Quantisation and action-chunk experiments with report-ready CSV evidence.

- [INT4 evidence notes](../latest/pc_local/int4/README.md)
- [Quantisation benchmark](../latest/pc_local/int4/quant_bench.csv)
- [Quantisation summary](../latest/pc_local/int4/quant_summary.csv)
- [Action-chunk evidence notes](../latest/pc_local/action_chunk/README.md)
- [Action-chunk summary](../latest/pc_local/action_chunk/action_chunk_summary.csv)
- [Chunk-size summary](../latest/pc_local/action_chunk/chunk_size_summary.csv)
- [Combined sweep summary](../latest/pc_local/combined_sweep_summary.csv)

## Week 13

Final closed-loop Jetson comparison, hardware-component benchmarks and report evidence.

- [Final Jetson ablation summary](../latest/jetson_remote_multi/final_ablation_summary.csv)
- [Jetson software quantisation summary](../latest/jetson_remote_multi/jetson_software_quant_summary.csv)
- [Run manifest](../latest/jetson_remote_multi/run_manifest.csv)
- [Raw FP16 evaluation record](../latest/jetson_remote_multi/raw/fp16/eval_info.json)
- [Raw backbone-INT8 evaluation record](../latest/jetson_remote_multi/raw/backbone_int8/eval_info.json)
- [Hardware optimisation summary](../latest/jetson_hardware/HARDWARE_OPTIMIZATION_SUMMARY.md)
- [Jetson quantisation benchmark](../latest/jetson_hardware/jetson_quant_bench.csv)
- [TensorRT connector benchmark](../latest/jetson_hardware/tensorrt_connector.csv)
- [TensorRT hybrid results](../latest/jetson_hardware/tensorrt_hybrid_results.csv)
- [TensorRT vision results](../latest/jetson_hardware/tensorrt_vision.csv)
- [CUDA Graph vision results](../latest/jetson_hardware/cudagraph_vision.csv)
- [Retrospective final-system diagram](system_design/D-W13-01_final_optimised_system.png)
