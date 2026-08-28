# Adaptive v2a preregistration

## Scope and frozen evidence

This preregistration is based on commit `b5cbf91a7c1a47ac48b7d30ed318ecd2ea252d1a` and the following read-only evidence:

- `four_condition_paired_analysis.json`, SHA256 `958f14c368e58ecb3a147cf2bdd791a64f01e7b06ae9180b53df318429ca17d2`;
- `adaptive_trigger_casebook.csv`, SHA256 `2358fea575a710cb9e33a2de168cfaeb3d291f0b059a1a7031095321ff5928bc`.

The frozen descriptive result is Static-H20 `32/50` with `8.66` model calls per episode versus Adaptive-v1 `31/50` with `9.08` calls. Seven v1 triggers produced zero rescues and one loss. Adaptive-v1 is frozen and is neither changed nor run by this work.

## Experimental isolation

Adaptive-v2a changes only the trigger mechanism. The checkpoint, revisions, official backend and processor, native action handling, `chunk_size=50`, `num_steps=2`, `max_steps=280`, batch size one, default/max execution horizon H20, and one-call fallback H1 are frozen. Actions are not clipped. H25/H30 and variable model chunks are outside scope.

The primary paired control is Static-H20. The 300-row dry-run retains the five historical static conditions plus Adaptive-v2a only to preserve the established six-condition manifest shape; no v1 condition appears.

## Trigger v2a

LIBERO action dimensions 0–6 have declared bounds `[-1, 1]`, with span 2. Before returning each native action to the evaluator, v2a computes for each dimension:

`excess = max(abs(raw_value) - 1, 0)`

`severity = excess / 2`

Dimension 6 (gripper) is recorded but excluded from trigger state. The deterministic, calibration-free rule for dimensions 0–5 is:

1. Immediate path: trigger if severity is at least `0.05`, corresponding to raw excess `0.10` or 5% of the declared action span.
2. Persistence path: severity must be at least `0.01` on the same dimension for two consecutive pre-action checks, corresponding to raw excess `0.02` or 1% of span. A single mild crossing cannot trigger.
3. Any sub-threshold step resets that dimension’s persistence count.
4. Multi-dimensional actions are assessed independently; satisfying either path on any non-gripper dimension produces one trigger event for that action.

These constants are round fractions of the declared action span. They were not fitted to v1 outcomes or selected by inspecting success labels. No calibration dataset is used in v2a.

## State machine

The state machine is deterministic:

`MONITORING_H20 → FALLBACK_H1_PENDING → FALLBACK_H1 → COOLDOWN_H20_PENDING → COOLDOWN_H20 → MONITORING_H20`

- A qualifying pre-action trigger executes the current native action unchanged, discards the unexecuted active-horizon tail, and forces the next model call to H1.
- The H1 call cannot trigger. After its one action, the next model call returns to H20.
- That complete 20-action H20 call is cooldown: violations are recorded, but cannot trigger or build persistence.
- After all 20 cooldown actions, monitoring resumes. Episode reset always returns to `MONITORING_H20`.
- If the episode terminates during fallback or cooldown, the next episode starts from a clean monitoring state.

## Required per-trigger artifact

Every trigger is appended to `adaptive_v2_trigger_events` in its episode JSON before `env.step`; summary.csv retains the aggregate trigger count. Each event contains:

- task ID, initial-state ID, environment seed and inference seed;
- model-call index and environment step;
- every triggering dimension with raw value, directional bound, excess, severity and persistence count;
- trigger-tail discard count and total buffer entries cleared;
- horizon before trigger, forced horizon after trigger, and recovery horizon;
- state before/after, immediate/persistent path, and fixed cooldown length;
- the literal ordering marker `trigger_evaluated_before_env_step`.

Aggregate-only evidence is a protocol failure. The mechanism smoke cannot be marked complete if an observed trigger lacks any required field.

## Outcome-label firewall

The trigger module accepts only a raw action. It receives no reward, terminal signal, task outcome, or evaluator result. A targeted AST test rejects outcome-field names, attributes, subscripts and mapping lookups in the trigger implementation. Evaluator outcome handling remains downstream of trigger evaluation and cannot feed the trigger state.

No implementation or analysis routine may use an outcome field to set severity, persistence, cooldown, dimension selection or any trigger parameter. Changing this firewall invalidates the preregistration.

## Held-out paired inference seeds

The 50 held-out inference seeds use:

`sha256("adaptive-v2-heldout-v1|libero_spatial" | task_id | environment_seed | initial_state_id)[:8] & ((1<<63)-1)`

Environment seeds remain 1000–1004 and benchmark initial-state IDs remain 0–4. Static-H20 and Adaptive-v2a receive exactly the same held-out inference seed for a pairing key. A preflight test requires the 50 held-out seeds to be unique and completely disjoint from the 50 legacy seeds derived under namespace `libero_spatial`. The materialized `paired_manifest.json` records all 50 seeds explicitly.

## Stages, sample size and decisions

### Mechanism smoke

Run 10 episodes: one benchmark state-0 episode for each task, Adaptive-v2a only. This stage is not used for a success-rate claim or parameter selection. It passes only if:

- all 10 planned keys complete exactly once;
- native actions remain unclipped;
- every trigger has the complete event schema;
- observed transitions obey trigger → H1 → one full H20 cooldown → monitoring whenever the episode is long enough;
- gripper-only and single mild crossings never trigger;
- generated/executed/unused action accounting remains conserved.

Any mechanism failure stops the study. No trigger parameter may be changed using smoke outcomes.

### Held-out paired pilot

Only after the mechanism smoke passes unchanged, run Static-H20 and Adaptive-v2a on the 50 held-out pairs (100 episodes total). Fifty pairs are the fixed pilot sample, not a claim of prospective power for small effects. Report the paired flip table, task-cluster bootstrap 95% CI, exact McNemar p, model calls, inference time and wall time.

Because Adaptive-v2a cannot use fewer model calls than Static-H20 by construction, extra calls have empirical value only if all of the following pre-registered descriptive gates pass:

1. At least one pairing is an Adaptive-only success (a directly verifiable rescue).
2. Adaptive-only successes outnumber Static-H20-only successes.
3. Every claimed rescue has valid complete trigger evidence and deterministic seed provenance.

If any gate fails, the conclusion is “no demonstrated value for the added calls.” Formal superiority additionally requires a positive task-cluster bootstrap lower bound and a pre-specified exact paired-test criterion; this 50-pair pilot is not automatically promoted to such a claim.

If the pilot passes the descriptive gates, confirm once on a second untouched namespace before changing thresholds. If it fails, v2a is frozen as a negative result; its outcome labels may not be recycled to tune v2b.

## Commands and safety

Dry-run is the evaluator default. Real environment execution requires the explicit `--execute` flag. The external mechanism-smoke launcher asserts the branch/commit, clean worktree, frozen artifacts, offline snapshots, single selected condition, held-out seed non-overlap, resume boundaries and a nonblocking file lock. It writes only under `/home/xinrui_shen/vla/runs/adaptive-v2-prereg/`.

No smoke or pilot rollout is launched as part of this preregistration.
