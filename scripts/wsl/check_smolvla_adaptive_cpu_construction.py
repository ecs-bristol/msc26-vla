"""Explicit opt-in CPU-only construction gate for the adaptive LeRobot plugin.

This script intentionally never invokes ``select_action``,
``predict_action_chunk``, ``forward``, or an environment API. It is not a
pytest test: loading the frozen policy weights is expensive and must remain an
operator-selected check.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path


SMOLVLA_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def _snapshot(path: str, revision: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.name != revision or resolved.parent.name != "snapshots":
        raise ValueError(f"{label} must be the exact snapshots/{revision} directory")
    return str(resolved)


def _cuda_memory(torch) -> dict[str, int | bool]:
    available = bool(torch.cuda.is_available())
    if not available:
        return {"available": False, "allocated_bytes": 0, "reserved_bytes": 0}
    return {
        "available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-weight-load",
        action="store_true",
        help="Required explicit opt-in; this loads the local frozen checkpoint weights on CPU.",
    )
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--vlm-snapshot", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if not args.allow_weight_load:
        parser.error("--allow-weight-load is required; this guard prevents accidental 6.7 GiB loading")
    if args.trust_remote_code:
        parser.error("trust_remote_code must remain False")

    base_snapshot = _snapshot(args.base_snapshot, SMOLVLA_REVISION, "base_snapshot")
    vlm_snapshot = _snapshot(args.vlm_snapshot, SMOLVLM2_REVISION, "vlm_snapshot")
    cache_dir = str(Path(args.cache_dir).expanduser().resolve(strict=True))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.utils.import_utils import register_third_party_plugins

    cuda_before = _cuda_memory(torch)
    register_third_party_plugins()
    config = make_policy_config(
        "smolvla_adaptive",
        base_snapshot_path=base_snapshot,
        base_cache_dir=cache_dir,
        vlm_snapshot_path=vlm_snapshot,
        device="cpu",
    )
    policy_class = get_policy_class("smolvla_adaptive")
    if PreTrainedConfig.get_choice_class("smolvla_adaptive") is not type(config):
        raise RuntimeError("smolvla_adaptive factory did not resolve its registered config")

    # This constructor internally loads base policy + processors. Do not call
    # reset afterwards: SmolVLAPolicy construction owns its in-memory queue setup.
    print(
        json.dumps(
            {
                "event": "construction_start",
                "base_snapshot": base_snapshot,
                "vlm_snapshot": vlm_snapshot,
                "device": "cpu",
                "trust_remote_code": False,
                "offline": True,
                "cuda_before": cuda_before,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    policy = policy_class(config)
    base = policy._base_policy
    parameter_devices = {parameter.device.type for parameter in base.parameters()}
    if parameter_devices != {"cpu"}:
        raise RuntimeError(f"base model parameters are not CPU-only: {sorted(parameter_devices)}")
    if base.config.n_action_steps != 1:
        raise RuntimeError("base n_action_steps changed; the wrapper must not mutate it")
    if base.config.num_steps != 2:
        raise RuntimeError("base num_steps does not match the frozen protocol")
    if Path(base.config.vlm_model_name).resolve() != Path(vlm_snapshot):
        raise RuntimeError("nested SmolVLM2 did not resolve to the frozen snapshot")

    report = {
        "plugin_policy_type": config.type,
        "policy_class": policy_class.__name__,
        "base_policy_class": type(base).__name__,
        "base_checkpoint": config.base_checkpoint,
        "base_revision": config.base_revision,
        "base_snapshot": base_snapshot,
        "vlm_checkpoint": config.vlm_checkpoint,
        "vlm_revision": config.vlm_revision,
        "vlm_snapshot": vlm_snapshot,
        "trust_remote_code": False,
        "offline": True,
        "base_chunk_size": int(base.config.chunk_size),
        "base_n_action_steps_unchanged": int(base.config.n_action_steps),
        "base_num_steps": int(base.config.num_steps),
        "parameter_devices": sorted(parameter_devices),
        "parameter_count": int(sum(parameter.numel() for parameter in base.parameters())),
        "preprocessor_type": type(policy._base_preprocessor).__name__,
        "postprocessor_type": type(policy._base_postprocessor).__name__,
        "cuda_before": cuda_before,
        "cuda_after_construction": _cuda_memory(torch),
    }
    print(json.dumps(report, sort_keys=True))

    del base
    del policy
    gc.collect()
    release_report = {"cuda_after_release": _cuda_memory(torch), "exit": "released"}
    print(json.dumps(release_report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
