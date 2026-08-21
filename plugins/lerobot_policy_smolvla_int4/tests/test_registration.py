from pathlib import Path
import tomllib

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

import lerobot_policy_smolvla_int4  # noqa: F401


def test_smolvla_int4_policy_is_discoverable_by_lerobot_factory():
    config_class = PreTrainedConfig.get_choice_class("smolvla_int4")

    assert config_class.__name__ == "SmolVLAInt4Config"
    assert get_policy_class("smolvla_int4").__name__ == "SmolVLAInt4Policy"


def test_distribution_name_matches_lerobot_plugin_discovery_prefix():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["name"].startswith("lerobot_policy_")


def test_config_defaults_match_reportable_pc_local_workflow():
    config_class = PreTrainedConfig.get_choice_class("smolvla_int4")
    config = config_class()

    assert config.checkpoint == "HuggingFaceVLA/smolvla_libero"
    assert config.revision == "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    assert config.quant_method == "int4_groupwise"
    assert config.num_steps is None
    assert config.n_action_steps is None


def test_config_rejects_unsupported_values():
    config_class = PreTrainedConfig.get_choice_class("smolvla_int4")

    with pytest.raises(ValueError, match="quant_method"):
        config_class(quant_method="fp8")
    with pytest.raises(ValueError, match="num_steps"):
        config_class(num_steps=0)
    with pytest.raises(ValueError, match="n_action_steps"):
        config_class(n_action_steps=0)
