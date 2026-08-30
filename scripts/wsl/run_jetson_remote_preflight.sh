#!/usr/bin/env bash
set -Eeuo pipefail

JETSON_ENDPOINT="${JETSON_ENDPOINT:?JETSON_ENDPOINT must be set}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"
CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"

python - "$JETSON_ENDPOINT" "$CHECKPOINT" "$MODEL_REVISION" "$N_ACTION_STEPS" <<'PY'
import sys

import requests
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

endpoint, checkpoint, revision, n_action_steps = sys.argv[1:]
register_third_party_plugins()
assert PreTrainedConfig.get_choice_class("remote_jetson").__name__ == "RemoteJetsonConfig"
assert get_policy_class("remote_jetson").__name__ == "RemoteJetsonPolicy"

try:
    response = requests.get(f"{endpoint.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
except requests.RequestException as error:
    raise SystemExit(f"Jetson service is unavailable at {endpoint}: {error}") from error
health = response.json()
if health.get("status") != "ok":
    raise SystemExit(f"Jetson service is not ready: {health}")
policy = health.get("policy") or {}
if policy.get("checkpoint") != checkpoint:
    raise SystemExit(f"checkpoint mismatch: expected {checkpoint}, got {policy.get('checkpoint')}")
if policy.get("revision") != revision:
    raise SystemExit(f"revision mismatch: expected {revision}, got {policy.get('revision')}")
if policy.get("precision") != "fp16":
    raise SystemExit(f"precision mismatch: expected fp16, got {policy.get('precision')}")
if policy.get("n_action_steps") != int(n_action_steps):
    raise SystemExit(
        f"n_action_steps mismatch: expected {n_action_steps}, got {policy.get('n_action_steps')}"
    )
print(f"official lerobot-eval remote preflight: ok ({checkpoint}@{revision})")
PY
