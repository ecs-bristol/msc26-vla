"""Load and close the real paired EGL/CUDA runtime without executing actions.

This is a preflight, not a rollout: it performs one official environment
reset (including LeRobot's ten settle no-ops), loads the frozen local policy,
and then explicitly closes both sides.  A caller should wrap it in a bounded
subprocess and require exit code zero to verify interpreter shutdown.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_SRC = PROJECT_ROOT / "plugins" / "lerobot_policy_smolvla_adaptive" / "src"
for path in (PROJECT_ROOT / "src", PLUGIN_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _pilot_module():
    source = PROJECT_ROOT / "scripts" / "analysis" / "libero_spatial_paired_pilot.py"
    spec = importlib.util.spec_from_file_location("paired_shutdown_preflight", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load paired-pilot module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-snapshot-path", type=Path, required=True)
    parser.add_argument("--vlm-snapshot-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise SystemExit("shutdown preflight requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")

    module = _pilot_module()
    config_path = PROJECT_ROOT / "configs" / "evaluation" / "libero_spatial_paired_pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["base_snapshot_path"] = module._snapshot_path(
        str(args.base_snapshot_path), module.SMOLVLA_REVISION, "base_snapshot_path"
    )
    config["model"]["vlm_snapshot_path"] = module._snapshot_path(
        str(args.vlm_snapshot_path), module.SMOLVLM2_REVISION, "vlm_snapshot_path"
    )
    condition = config["conditions"][0]
    backend = module._local_backend_factory()()
    policy = module._local_policy_factory(device=args.device)(condition, config)
    episode = None
    try:
        episode = backend.open_episode("libero_spatial", 0, 0, 280, 1000)
        observation = episode.reset()
        if observation.proprioception.shape != (8,):
            raise RuntimeError("official reset did not produce an 8D state")
    finally:
        if episode is not None:
            episode.close()
        policy.close()
        backend.close()
        del episode, policy, backend
        gc.collect()

    print(
        json.dumps(
            {
                "status": "clean_shutdown_ready",
                "environment_steps_executed": 0,
                "policy_actions_selected": 0,
                "reset_settle_steps": 10,
                "device": args.device,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
