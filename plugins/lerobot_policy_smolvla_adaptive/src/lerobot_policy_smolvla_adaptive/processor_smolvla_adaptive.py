"""External processor factory for the self-contained adaptive wrapper.

The wrapper applies the frozen base SmolVLA processors internally immediately
around ``base.predict_action_chunk``.  The evaluator-facing pipelines must be
identity pipelines so those processors are not applied a second time.
"""

from typing import Any

from lerobot.processor import make_policy_processor_pipelines

from .configuration_smolvla_adaptive import SmolVLAAdaptiveConfig


def make_smolvla_adaptive_pre_post_processors(
    config: SmolVLAAdaptiveConfig,
    dataset_stats: dict[str, Any] | None = None,
):
    del config, dataset_stats
    return make_policy_processor_pipelines(input_steps=[], output_steps=[])
