# Adaptive-v2a formal held-out paired evaluation preregistration

## Freeze and coverage gate

The formal preparation starts from `adaptive-v2-prereg` commit
`7f8ac408a482843b7938e31bdf62b28f2af69d51`. The development coverage artifact
reports `V2A_TRIGGER_COVERAGE_PASS`, six completed development episodes, one real
trigger, one complete H20 → pre-execution trigger → H1 → full H20 cooldown/H20
recovery chain, and `success_fields_used=false`.

Coverage outcomes do not authorize any trigger change. Adaptive-v2a severity
thresholds, persistence, cooldown and state transitions remain frozen exactly as
they were at `a40dc27c3d78dd9c8f647db560ad4cb58510eb61`.

## Code-diff audit

The `a40dc27..7f8ac40` diff changes only coverage selection, execution filtering,
validation, documentation, and seed/config artifacts. It does not change:

- `src/libero_platform/policies/adaptive_v2_trigger.py` (Git blob
  `4f3096cb45cab4419990b0c918554049c1ff5fb8` at both commits);
- `fixed_h_action_buffer.py`;
- the plugin configuration or modeling policy path;
- evaluator `_adaptive_v2_trigger_event` (source SHA-256
  `28a62b8adcb1dee8836f566e5767592001ce9f3ba02e94c446204a66b9ef44b7`);
- evaluator `_run_episode`, including native action delivery to `env.step`
  (source SHA-256
  `0227a7aa9c05d91e8146724d39c7ada8516ea3716c8e95c74455037a54fd0c54`).

A targeted test repeats this audit against the frozen base. Any later mismatch is
a preflight failure.

## Formal conditions and pairing

Exactly two conditions are allowed:

1. `Static-H20`
2. `Adaptive-v2a-H20→H1`

Both use the same frozen SmolVLA and SmolVLM2 revisions, local-only snapshots,
`chunk_size=50`, `num_steps=2`, H20 default/max execution horizon, official
backend/preprocessing, benchmark initial states, episode cap 280, batch size 1,
native actions and `clip_actions=false`. Both enable identical range detection.
The only condition-level difference is `adaptive_v2_trigger=false` versus `true`.

For each of 50 pairing keys, both conditions receive identical task ID,
initial-state ID, environment seed and inference seed. The new inference namespace
is `adaptive-v2-confirmatory-v1|libero_spatial`. Its explicit 50-seed manifest is
disjoint from all legacy H1/H10/H20/Adaptive-v1 seeds, the complete prior v2a
mechanism-smoke namespace (including its observed 10), and the coverage-dev seeds.

## Deterministic blocks and serial order

For each pairing key compute:

`sha256("adaptive-v2-confirmatory-block-v1"|task_id|seed|initial_state_id)`

Sort all 50 keys lexicographically by digest. The first 25 are Block A and the
last 25 are Block B. The four phases are immutable and strictly serial:

1. Block A — Static-H20
2. Block A — Adaptive-v2a
3. Block B — Adaptive-v2a
4. Block B — Static-H20

No later phase starts until the current phase has 25 completed immutable episode
JSON files. Resume skips completed episodes and prints phase, block, condition,
completed and remaining counts. A nonblocking file lock forbids parallel launchers.

## Outcome lock and frozen analysis

Before all 100 episode records pass pairing, protocol, Git SHA, seed, action and
accounting checks, the analysis completeness gate wraps records in an outcome
firewall. Access to `success_at_280`, `success_step`, or `termination_reason`
raises `AnalysisLockedError`. No interim success-based decision is permitted.

After the gate unlocks, the immutable analysis reports:

- success flip table (`both_success`, `adaptive_only`, `static_only`, `both_fail`);
- Adaptive-minus-Static success difference and 95% task-cluster bootstrap CI;
- two-sided exact McNemar p-value;
- paired model-call, inference-time and wall-time differences with the same
  task-cluster bootstrap CI;
- total triggers and a complete per-trigger casebook, marking each event as
  `rescue`, `loss`, or `no_flip` from its paired outcomes.

Bootstrap uses 100,000 replicates, task ID as the cluster, percentile 95% bounds,
and RNG seed 20260828.

## Decision rule

The existing `ADAPTIVE_V2_PREREG.md` gates remain unchanged. Because Adaptive-v2a
cannot use fewer calls than Static-H20, the extra calls have demonstrated value
only if there is at least one Adaptive-only rescue, Adaptive-only rescues outnumber
Static-only losses, and every claimed rescue has complete valid trigger and seed
evidence. Otherwise the conclusion is “no demonstrated value for the added
calls.” Results cannot modify this rule or any v2a mechanism parameter.

This preparation performs tests and dry-run materialization only. It does not load
a model, construct LIBERO, or call `env.step`.
