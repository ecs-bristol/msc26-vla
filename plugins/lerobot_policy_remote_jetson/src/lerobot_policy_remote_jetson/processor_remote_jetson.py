from typing import Any

from lerobot.processor import make_policy_processor_pipelines

from .configuration_remote_jetson import RemoteJetsonConfig


def make_remote_jetson_pre_post_processors(
    config: RemoteJetsonConfig,
    dataset_stats: dict[str, Any] | None = None,
):
    del config, dataset_stats
    return make_policy_processor_pipelines(input_steps=[], output_steps=[])
