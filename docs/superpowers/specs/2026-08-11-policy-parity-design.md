# PC-Jetson SmolVLA Action-Parity Design

## Status

Approved approach: one fixed-observation local-versus-remote action comparison.

## Objective

Determine whether the WSL PC-local SmolVLA adapter and the Jetson HTTP SmolVLA service return materially different actions when they receive the same deterministic LIBERO observation.

This diagnostic is required before interpreting the observed difference between the official WSL `lerobot-eval` Spatial reference (8/10) and the PC-simulation/Jetson-inference rollout (0/10).

## Scope

The command creates exactly one LIBERO observation using a supplied suite, task ID, initial-state ID, and seed. It does not step the environment and does not change any experiment YAML file.

The command loads the WSL-local `SmolVLAPolicyAdapter`, probes the Jetson `/health` endpoint, creates one `RemoteHTTPPolicyAdapter`, resets both adapters with the same seed, and requests one action from each adapter using the same `PolicyRequest`.

## Command Contract

New command:

```text
python -m libero_platform policy-parity \
  --suite libero_spatial \
  --task-id 0 \
  --initial-state-id 0 \
  --seed 42 \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --precision fp16 \
  --endpoint http://10.42.0.2:8081
```

Defaults use the same values as the paired deployment condition: `libero_spatial`, task 0, initial-state 0, seed 42, `HuggingFaceVLA/smolvla_libero`, FP16, and `http://10.42.0.2:8081`. The caller explicitly supplies a revision for reportable evidence.

## Evidence

The command writes one directory beneath `outputs/policy_parity/` containing:

- `summary.json`: fixed input identity, local and remote model identities, local and remote actions, per-dimension deltas, mean absolute error, maximum absolute error, action-validity status, and latencies;
- `actions.csv`: one 7-dimensional paired row that can be opened directly in a spreadsheet.

The observation is represented by instruction, state shape, image shapes, and SHA-256 hashes of contiguous image and state bytes. Raw camera images and state values are not written into the parity artifact.

## Pass/Fail Interpretation

- The command fails before prediction if the remote health contract is unavailable, reports a different checkpoint/revision/precision, or either adapter returns an invalid action.
- A parity report is `aligned` when maximum absolute action delta is at most `1e-4`; otherwise it is `diverged`.
- `aligned` means the next diagnostic target is runner-level episode reset, control timing, or environment configuration.
- `diverged` means the next diagnostic target is request serialization/deserialization, state/image preprocessing, random sampling state, or model service configuration.

## Constraints

- Use `PolicyRequest`, `SmolVLAPolicyAdapter`, and `RemoteHTTPPolicyAdapter`; do not make a parallel inference protocol.
- Use the existing LIBERO backend and its `benchmark` initial-state source.
- Do not replace, retune, or rerun the active 280-step remote experiment as part of this diagnostic.
- Do not silently fall back from remote inference to local inference.
