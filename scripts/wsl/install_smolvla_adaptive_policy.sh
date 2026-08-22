#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLUGIN_DIR="$PROJECT_DIR/plugins/lerobot_policy_smolvla_adaptive"

# The third-party plugin imports the project-owned action buffer rather than
# copying it, so both distributions are intentionally installed editable.
python -m pip install --no-deps --editable "$PROJECT_DIR"
python -m pip install --no-deps --editable "$PLUGIN_DIR"
python - <<'PY'
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
config = PreTrainedConfig.get_choice_class("smolvla_adaptive")()
assert config.fixed_h == 20
assert config.num_steps == 2
assert get_policy_class("smolvla_adaptive").__name__ == "SmolVLAAdaptivePolicy"
print("smolvla_adaptive policy plugin: installed and discoverable")
PY
