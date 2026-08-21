from __future__ import annotations

from typing import Any

from lerobot.policies.factory import make_pre_post_processors

from .configuration_smolvla_int4 import SmolVLAInt4Config


def make_smolvla_int4_pre_post_processors(
    config: SmolVLAInt4Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
):
    """Load the checkpoint's official SmolVLA processor pipelines.

    The official `--policy.path` flow loads the same preprocessor/postprocessor
    JSON files from the checkpoint. We mirror it (including the device and
    rename overrides the official evaluator applies) so int4 runs are directly
    comparable with the fp16 baseline.
    """
    return make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=config.checkpoint,
        pretrained_revision=config.revision,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": config.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
