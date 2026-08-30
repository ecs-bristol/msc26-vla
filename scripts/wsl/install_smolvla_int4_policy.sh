#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLUGIN_DIR="$PROJECT_DIR/plugins/lerobot_policy_smolvla_int4"

python -m pip install --no-deps --editable "$PLUGIN_DIR"
python - <<'PY'
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
assert PreTrainedConfig.get_choice_class("smolvla_int4").__name__ == "SmolVLAInt4Config"
assert get_policy_class("smolvla_int4").__name__ == "SmolVLAInt4Policy"
print("smolvla_int4 policy plugin: installed and discoverable")
PY
