# Official LeRobot Eval With Jetson Remote Policy Design

## Status

Approved. This design replaces the project-owned LIBERO rollout runner for all new benchmark evidence.

## Objective

Run the official LeRobot `lerobot-eval` LIBERO protocol in WSL while executing SmolVLA inference on the Jetson Orin Nano over the existing direct Ethernet connection.

The official evaluator is the only authority for environment creation, task ordering, episode length, early success termination, reward and success calculation, video recording, and aggregate metrics. The Jetson is an inference appliance only.

## Architecture

```text
Official lerobot-eval in WSL
  -> official LIBERO environment and env processors
  -> installed LeRobot policy plugin: remote_jetson
  -> HTTP /reset and /predict over 10.42.0.0/24
  -> Jetson SmolVLA service pinned to one checkpoint revision
  -> 7-D action tensor
  -> official LIBERO env postprocessor and environment step
  -> official LeRobot metrics, videos, and logs
```

The plugin is packaged as an independently installable distribution named `lerobot_policy_remote_jetson`. LeRobot 0.6.1 discovers distributions with the `lerobot_policy_` prefix and imports their matching Python module.

## Component Ownership

### Official LeRobot evaluator

- Creates the `libero_spatial` tasks and official benchmark initial states.
- Adds the language instruction to each observation.
- Applies the official LIBERO environment preprocessor and postprocessor.
- Runs one policy action per environment step.
- Stops an episode on success or at 280 steps.
- Writes official videos, logs, and aggregate success metrics.

### WSL remote policy plugin

- Registers `remote_jetson` as a LeRobot policy type.
- Accepts the observation produced by the official evaluator.
- Supports batch size 1 for the first reportable implementation.
- Serializes the two RGB cameras, 8-D robot state, and task instruction without normalizing them a second time.
- Calls Jetson `/reset` once per official episode and `/predict` once per policy action.
- Validates that the response is a finite 7-D action and returns a CPU Torch tensor.
- Writes transport-only JSONL telemetry for latency and request failures.
- Never falls back to PC-local inference.

### Jetson service

- Loads `HuggingFaceVLA/smolvla_libero` at revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
- Applies the checkpoint's official SmolVLA preprocessor and postprocessor.
- Runs FP16 inference on CUDA.
- Exposes `/health`, `/metadata`, `/reset`, and `/predict`.
- Reports the exact checkpoint, revision, precision, and service schema in `/health`.

## Evaluation Contract

The first paired experiment is fixed to:

- suite: `libero_spatial`
- tasks: 0 through 9
- episodes per task: 1
- maximum steps: 280
- batch size: 1
- maximum parallel tasks: 1
- model: `HuggingFaceVLA/smolvla_libero`
- revision: `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- SmolVLA simulation settings: `n_action_steps=1`, `num_steps=10`
- Jetson precision: FP16

The PC-local official result and Jetson-remote official result must use the same model revision, LIBERO assets, LeRobot version, hf-libero version, MuJoCo 3.3.2, task set, episode count, and step limit.

## Processor Contract

The official evaluator first converts raw environment observations to LeRobot keys and applies the LIBERO environment processor. The remote plugin then performs only transport conversion:

- `observation.images.image` -> agent-view PNG
- `observation.images.image2` -> wrist-view PNG
- `observation.state` -> eight float values
- `task` -> one instruction string

The Jetson service remains responsible for checkpoint normalization, image resize/padding, state adaptation, SmolVLA inference, and action unnormalization. The WSL plugin must not run SmolVLA normalization or unnormalization.

## Evidence

Official benchmark evidence is the `lerobot-eval` output directory, including its aggregate metrics, per-task results, logs, and videos. Remote transport telemetry is supplementary and includes request latency, server inference latency, endpoint, checkpoint revision, and error category.

The project-owned `outputs/libero_runs` format is legacy evidence only and must not be combined with official `lerobot-eval` results.

## Migration Policy

- New reportable LIBERO runs must use `lerobot-eval`.
- Existing custom runner source and historical outputs remain temporarily for provenance and regression tests.
- Custom 600-step Jetson smoke, pilot, and formal configurations are retired from the active workflow.
- Active documentation and scripts expose one 280-step official remote experiment.
- Deleting legacy source is deferred until the official remote plugin passes discovery, health, one-task smoke, and ten-task paired verification.

## Failure Behavior

The run stops with an explicit error when the endpoint is unavailable, model identity differs, response schema is invalid, action values are non-finite, or batch size exceeds one. No silent fallback, clipping, model substitution, or revision drift is allowed.

## Acceptance Criteria

1. `lerobot-eval --policy.type=remote_jetson --help` discovers the installed plugin in the WSL environment.
2. Preflight verifies the Jetson health response and exact model revision.
3. A one-task official smoke run completes and writes official LeRobot evidence.
4. A ten-task 280-step remote run completes using the same task protocol as the 8/10 PC-local reference.
5. The active workflow contains no command that invokes `python -m libero_platform run` for benchmark evaluation.
