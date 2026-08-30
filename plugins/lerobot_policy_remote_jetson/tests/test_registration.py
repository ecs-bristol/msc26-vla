from pathlib import Path
import tomllib

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

import lerobot_policy_remote_jetson  # noqa: F401


def test_remote_jetson_policy_is_discoverable_by_lerobot_factory():
    config_class = PreTrainedConfig.get_choice_class("remote_jetson")

    assert config_class.__name__ == "RemoteJetsonConfig"
    assert get_policy_class("remote_jetson").__name__ == "RemoteJetsonPolicy"


def test_distribution_name_matches_lerobot_plugin_discovery_prefix():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["name"].startswith("lerobot_policy_")


def test_config_defaults_match_reportable_jetson_workflow():
    config_class = PreTrainedConfig.get_choice_class("remote_jetson")
    config = config_class()

    assert config.checkpoint == "HuggingFaceVLA/smolvla_libero"
    assert config.revision == "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    assert config.precision == "fp16"
    assert config.device == "cpu"


def test_config_rejects_unreported_precision():
    config_class = PreTrainedConfig.get_choice_class("remote_jetson")

    with pytest.raises(ValueError, match="requires fp16"):
        config_class(precision="fp32")
