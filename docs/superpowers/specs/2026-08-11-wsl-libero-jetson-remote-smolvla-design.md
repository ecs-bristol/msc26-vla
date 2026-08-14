# WSL LIBERO With Jetson Remote SmolVLA Design

## Status

Approved approach: PC simulation with Jetson remote inference.

## Objective

Measure the deployment trade-off of the checkpoint `HuggingFaceVLA/smolvla_libero` by keeping LIBERO simulation on the PC and moving only policy inference to a Jetson Orin Nano 8GB.

The PC-local reference is the successful official WSL evaluation executed with LeRobot 0.6.1, HF-LIBERO 0.1.4, MuJoCo 3.3.2, EGL rendering, and CUDA. Its initial LIBERO Spatial smoke result is 8 successes from 10 tasks with one episode per task.

## Scope

The system has three clearly separated roles.

- The WSL PC owns the official LIBERO task environment, reset state, success evaluation, video recording, seed selection, and result files.
- The Jetson owns the SmolVLA checkpoint, GPU inference, and server-side inference telemetry.
- A direct Ethernet connection carries only policy requests and responses over HTTP.

The implementation must not reuse old Windows Python environments, Windows Hugging Face caches, or historical custom-runner success rates as the PC-local reference.

## Architecture

```text
WSL LIBERO evaluator
  observation (images, 8-D state, instruction)
        |
        | HTTP POST /predict over 10.42.0.0/24
        v
Jetson SmolVLA policy service
  HuggingFaceVLA/smolvla_libero -> 7-D action
        |
        | action plus service latency and model identity
        v
WSL LIBERO evaluator
  action validation -> environment step -> result and video
```

The service endpoint is `http://10.42.0.2:8081`. The WSL PC is the client. The service binds to `0.0.0.0` on the Jetson, but the direct-link network is the intended access boundary.

## Model Identity

Both reference and remote experiments use `HuggingFaceVLA/smolvla_libero`.

Before the first remote run, the WSL model cache revision is recorded and the identical revision is requested on Jetson. The Jetson health response must expose the model ID, revision, precision, device, and schema version. A mismatch is a preflight failure, not a benchmark result.

## Protocol Contract

The request contains the instruction, policy observation images, and 8-D robot state produced by the WSL LIBERO environment. The response contains one finite 7-D action plus server compute latency.

The WSL client validates response schema, action rank, finite values, and action bounds before applying an action. It records client round-trip latency separately from Jetson compute latency. No local policy fallback is permitted when the Jetson service is unavailable.

## Evaluation Protocol

All remote measurements use the LIBERO Spatial suite, the same model checkpoint, the same simulator version, and the same WSL execution environment as the PC-local reference.

The active evidence plan has one reportable deployment condition:

1. Preflight: service health, model revision identity, one synthetic policy request, and a response-validation check.
2. Remote protocol-alignment run: 1 episode for each of the 10 Spatial tasks, a 280-step cap, saved videos, and per-step latency. This is paired with the WSL-native 8/10 reference, while remaining explicitly a PC-simulator/Jetson-inference condition.

Pilot and formal remote matrices are out of scope until this paired run has been interpreted.

PC-local and Jetson-remote are reported as separate conditions. The PC-local official result is an inference-capability reference. The remote result measures the system and deployment condition; it is not claimed to be an unmodified native `lerobot-eval` policy type.

## Evidence Recorded

For every run, write a machine-readable manifest with:

- checkpoint ID and resolved revision;
- LeRobot, HF-LIBERO, MuJoCo, CUDA, and Python versions;
- execution role, host, direct-link endpoint, Jetson device profile, and precision;
- task ID, initial state ID, seed, episode success, steps, and termination reason;
- action-validity count and failure reason;
- mean and p95 client round-trip latency;
- mean and p95 Jetson service latency;
- Jetson peak memory, power, and temperature where available;
- video paths and logs.

## Failure Handling

Preflight fails immediately for unavailable service, incorrect model ID/revision, incompatible schema, malformed action, or unavailable CUDA on Jetson. Runtime network loss and invalid actions are recorded as explicit failures. They must never silently route inference back to the PC.

If a remote run produces lower task success than PC-local, record the comparison as an observed deployment result. Do not tune action clipping or task state selection during the formal comparison without creating a new named experiment condition.

## Acceptance Criteria

- WSL reaches the Jetson `/health` endpoint over the direct link.
- Jetson health identifies `HuggingFaceVLA/smolvla_libero` and the pinned revision.
- A synthetic request returns one finite 7-D action within the declared action bounds.
- A remote Spatial smoke run completes all 10 tasks and produces one result row per task.
- The result manifest distinguishes local policy latency, network round-trip latency, and Jetson service latency.
- Existing PC-local official output remains unchanged and is retained as reference evidence.

## Out of Scope

- Training or fine-tuning SmolVLA.
- Jetson-side MuJoCo/LIBERO simulation.
- A dashboard or MCP integration.
- Merging historical Windows results into the Linux baseline.
