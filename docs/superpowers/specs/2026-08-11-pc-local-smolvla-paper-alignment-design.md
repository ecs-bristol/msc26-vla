# PC-Local SmolVLA Paper Alignment Design

## Goal

Create one canonical PC-local configuration that evaluates the released SmolVLA LIBERO checkpoint on all ten LIBERO Spatial tasks and ten benchmark initial states per task.

## Scope

The canonical configuration is the existing `libero_spatial_pc_local_smolvla_official_alignment.yaml` file. It replaces its earlier two-task exploratory envelope rather than adding a competing experiment file.

## Protocol

- Suite: `libero_spatial`.
- Tasks: IDs 0 through 9.
- Initial states: benchmark IDs 0 through 9.
- Episodes per initial state: 1, giving 100 formal trials.
- Episode horizon: 600 steps, matching LIBERO's bundled evaluation default.
- Settle steps: 0. No reset perturbation is introduced.
- Failure handling: continue so every scheduled trial produces evidence.
- Viewer, frame, and video capture: disabled for the formal run to avoid timing and storage distortion. Step-level result evidence remains enabled.

## SmolVLA Settings

- Checkpoint: the catalog default `lerobot/smolvla_libero`.
- Deployment: `pc_local`.
- Precision: `fp16`.
- Quantization: none.
- Action control: identity with unit translation and rotation scales.
- Execution action steps: 1. The simulator re-observes and requests a new action after every executed action.
- Flow matching integration steps: 10.

The execution action count is distinct from the checkpoint action chunk size. The model retains its checkpoint chunk configuration; only one action is consumed per observation in simulation.

## Architecture

Add a strict SmolVLA inference-settings object to the experiment policy schema. The catalog passes it into the local SmolVLA adapter, and the adapter passes it to the LeRobot runtime. The runtime overrides only `n_action_steps` and `num_steps` after loading checkpoint configuration. The resolved experiment metadata then carries the requested settings as evidence.

## Error Handling

The runtime rejects unsupported values before model loading. Existing action validation and raw/transformed action recording remain unchanged. The formal run continues after policy or episode failures so the final artifacts represent all 100 scheduled trials.

## Verification

Tests must prove that the canonical YAML resolves to exactly 100 trials, uses benchmark initial states and the official 600-step envelope, and passes `n_action_steps=1` and `num_steps=10` to the SmolVLA runtime. Existing configuration validation and focused policy tests must remain green.

