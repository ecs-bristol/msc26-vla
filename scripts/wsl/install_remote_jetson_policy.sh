#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLUGIN_DIR="$PROJECT_DIR/plugins/lerobot_policy_remote_jetson"

python -m pip install --no-deps --editable "$PLUGIN_DIR"
python - <<'PY'
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
assert PreTrainedConfig.get_choice_class("remote_jetson").__name__ == "RemoteJetsonConfig"
assert get_policy_class("remote_jetson").__name__ == "RemoteJetsonPolicy"
print("remote_jetson policy plugin: installed and discoverable")
PY
