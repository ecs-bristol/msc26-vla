"""Capture official-vs-paired SmolVLA observation and action parity evidence.

This diagnostic performs resets only (including the official ten settle no-ops)
and one fixed-noise chunk inference per path.  It never starts an episode
rollout and writes all tensor payloads outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_SRC = PROJECT_ROOT / "plugins" / "lerobot_policy_smolvla_adaptive" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))


def _array(value: Any) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    return np.ascontiguousarray(cpu.numpy() if hasattr(cpu, "numpy") else cpu)


def _fingerprint(value: Any) -> dict[str, Any]:
    array = _array(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _comparison(left: Any, right: Any) -> dict[str, Any]:
    left_array = _array(left)
    right_array = _array(right)
    if left_array.shape != right_array.shape:
        return {
            "shape_equal": False,
            "max_abs_difference": None,
            "reason": f"shape mismatch: {left_array.shape} != {right_array.shape}",
        }
    return {
        "shape_equal": True,
        "max_abs_difference": float(
            np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
        ),
    }


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _custom_policy_batch(observation: Any) -> dict[str, Any]:
    images = observation.images
    return {
        "observation.images.image": torch.from_numpy(
            np.ascontiguousarray(np.asarray(images["agentview"], dtype=np.uint8).transpose(2, 0, 1))
        )
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div(255.0),
        "observation.images.image2": torch.from_numpy(
            np.ascontiguousarray(np.asarray(images["wrist"], dtype=np.uint8).transpose(2, 0, 1))
        )
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div(255.0),
        "observation.state": torch.from_numpy(
            np.ascontiguousarray(np.asarray(observation.proprioception, dtype=np.float32))
        ).unsqueeze(0),
        "task": [observation.instruction],
    }


def _official_observation(task_id: int, initial_state_id: int, seed: int) -> tuple[Any, dict[str, Any]]:
    from lerobot.envs import preprocess_observation
    from lerobot.envs.libero import LiberoEnv
    from lerobot.processor.env_processor import LiberoProcessorStep
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name="libero_spatial",
        episode_length=280,
        obs_type="pixels_agent_pos",
        observation_width=360,
        observation_height=360,
        init_states=True,
        episode_index=initial_state_id,
        n_envs=1,
        num_steps_wait=10,
        control_freq=20,
        control_mode="relative",
        hard_reset=True,
    )
    try:
        formatted, _ = env.reset(seed=seed)
        # Vector evaluation supplies a leading environment dimension.
        batched = {
            "pixels": {name: value[None, ...] for name, value in formatted["pixels"].items()},
            "robot_state": {
                group: {name: value[None, ...] for name, value in entries.items()}
                for group, entries in formatted["robot_state"].items()
            },
        }
        processed = preprocess_observation(batched)
        processed["task"] = [env.task_description]
        processed = LiberoProcessorStep().observation(processed)
        return formatted, processed
    finally:
        env.close()


def _custom_observation(task_id: int, initial_state_id: int, seed: int) -> tuple[Any, dict[str, Any]]:
    from libero_platform.backends.libero_backend import LiberoBackend

    backend = LiberoBackend(
        camera_size=256,
        settle_steps=0,
        initial_state_source="benchmark",
    )
    episode = backend.open_episode(
        "libero_spatial", task_id, initial_state_id, max_steps=280, seed=seed
    )
    try:
        observation = episode.reset()
        return observation, dict(episode._last_raw_observation)
    finally:
        episode.close()


def _repaired_observation(task_id: int, initial_state_id: int, seed: int) -> Any:
    from libero_platform.backends.libero_backend import OfficialLeRobotLiberoBackend

    backend = OfficialLeRobotLiberoBackend()
    episode = backend.open_episode(
        "libero_spatial", task_id, initial_state_id, max_steps=280, seed=seed
    )
    try:
        return _custom_policy_batch(episode.reset())
    finally:
        episode.close()


def _same_raw_custom_state(official_formatted: dict[str, Any]) -> np.ndarray:
    from libero_platform.backends.libero_backend import _quaternion_to_axis_angle

    robot = official_formatted["robot_state"]
    return np.concatenate(
        (
            np.asarray(robot["eef"]["pos"], dtype=np.float32),
            _quaternion_to_axis_angle(robot["eef"]["quat"]),
            np.asarray(robot["gripper"]["qpos"], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)


def _range_diagnostics(chunk: Any) -> list[dict[str, Any]]:
    array = _array(chunk)
    rows: list[dict[str, Any]] = []
    for action_index, dimension in np.argwhere((array[0] < -1.0) | (array[0] > 1.0)):
        raw = float(array[0, action_index, dimension])
        rows.append(
            {
                "action_index": int(action_index),
                "dimension": int(dimension),
                "raw_value": raw,
                "excess_beyond_unit_range": max(raw - 1.0, -1.0 - raw),
                "is_gripper": int(dimension) == 6,
            }
        )
    return rows


def _action_parity(
    *, official_batch: dict[str, Any], base_snapshot: Path, vlm_snapshot: Path, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from lerobot_policy_smolvla_adaptive.configuration_smolvla_adaptive import (
        SmolVLAAdaptiveConfig,
    )
    from lerobot_policy_smolvla_adaptive.modeling_smolvla_adaptive import (
        SmolVLAAdaptivePolicy,
    )

    config = SmolVLAAdaptiveConfig(
        base_snapshot_path=str(base_snapshot),
        vlm_snapshot_path=str(vlm_snapshot),
        fixed_h=1,
        safety_enabled=False,
        replan_after_safety_violation=False,
        num_steps=2,
        chunk_size=50,
        precision="fp16",
        local_files_only=True,
        device="cuda",
    )
    policy = SmolVLAAdaptivePolicy(config)
    policy.to("cuda")
    policy.eval()
    base = policy._base_policy

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    fixed_noise = torch.randn((1, 50, 32), generator=generator, device="cuda")
    base.model.sample_noise = lambda shape, device: fixed_noise.to(device).clone()

    try:
        base.reset()
        native_input = policy._base_preprocessor(_clone_batch(official_batch))
        native_raw = base.predict_action_chunk(native_input)
        native = policy._base_postprocessor(native_raw)

        base.reset()
        wrapped = policy._chunk_predictor.predict_action_chunk(_clone_batch(official_batch))
        result = {
            "fixed_noise_sha256": _fingerprint(fixed_noise)["sha256"],
            "native": _fingerprint(native),
            "wrapper_internal": _fingerprint(wrapped),
            "comparison": _comparison(native, wrapped),
            "range_violations": _range_diagnostics(native),
        }
        return _array(native), _array(wrapped), result
    finally:
        del policy
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-snapshot-path", type=Path, required=True)
    parser.add_argument("--vlm-snapshot-path", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite parity evidence: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    official_formatted, official = _official_observation(
        args.task_id, args.initial_state_id, args.seed
    )
    custom_observation, custom_raw = _custom_observation(
        args.task_id, args.initial_state_id, args.seed
    )
    custom = _custom_policy_batch(custom_observation)
    repaired = _repaired_observation(args.task_id, args.initial_state_id, args.seed)

    same_raw_custom_agent = torch.from_numpy(
        np.ascontiguousarray(official_formatted["pixels"]["image"][::-1].transpose(2, 0, 1))
    ).unsqueeze(0).float().div(255.0)
    same_raw_custom_wrist = torch.from_numpy(
        np.ascontiguousarray(official_formatted["pixels"]["image2"][::-1].transpose(2, 0, 1))
    ).unsqueeze(0).float().div(255.0)
    same_raw_custom_state = torch.from_numpy(_same_raw_custom_state(official_formatted)).unsqueeze(0)

    tensors = {
        "official_agentview": official["observation.images.image"],
        "official_wrist": official["observation.images.image2"],
        "official_state": official["observation.state"],
        "custom_agentview": custom["observation.images.image"],
        "custom_wrist": custom["observation.images.image2"],
        "custom_state": custom["observation.state"],
        "repaired_agentview": repaired["observation.images.image"],
        "repaired_wrist": repaired["observation.images.image2"],
        "repaired_state": repaired["observation.state"],
        "same_raw_custom_agentview": same_raw_custom_agent,
        "same_raw_custom_wrist": same_raw_custom_wrist,
        "same_raw_custom_state": same_raw_custom_state,
    }
    for name, tensor in tensors.items():
        np.save(args.output_dir / f"{name}.npy", _array(tensor), allow_pickle=False)

    native, wrapped, action = _action_parity(
        official_batch=official,
        base_snapshot=args.base_snapshot_path.resolve(strict=True),
        vlm_snapshot=args.vlm_snapshot_path.resolve(strict=True),
        seed=args.seed,
    )
    np.save(args.output_dir / "native_action_chunk.npy", native, allow_pickle=False)
    np.save(args.output_dir / "wrapper_action_chunk.npy", wrapped, allow_pickle=False)

    report = {
        "identity": {
            "suite": "libero_spatial",
            "task_id": args.task_id,
            "initial_state_id": args.initial_state_id,
            "environment_seed": args.seed,
            "official_settle_steps": 10,
            "custom_settle_steps": 0,
            "official_camera_size": [360, 360],
            "custom_camera_size": [256, 256],
            "repaired_settle_steps": 10,
            "repaired_camera_size": [360, 360],
        },
        "official": {name.removeprefix("official_"): _fingerprint(value) for name, value in tensors.items() if name.startswith("official_")},
        "custom": {name.removeprefix("custom_"): _fingerprint(value) for name, value in tensors.items() if name.startswith("custom_")},
        "repaired": {name.removeprefix("repaired_"): _fingerprint(value) for name, value in tensors.items() if name.startswith("repaired_")},
        "task": {
            "official": official["task"][0],
            "custom": custom["task"][0],
            "equal": official["task"] == custom["task"],
        },
        "actual_path_comparison": {
            "agentview": _comparison(official["observation.images.image"], custom["observation.images.image"]),
            "wrist": _comparison(official["observation.images.image2"], custom["observation.images.image2"]),
            "state": _comparison(official["observation.state"], custom["observation.state"]),
        },
        "repaired_path_comparison": {
            "agentview": _comparison(official["observation.images.image"], repaired["observation.images.image"]),
            "wrist": _comparison(official["observation.images.image2"], repaired["observation.images.image2"]),
            "state": _comparison(official["observation.state"], repaired["observation.state"]),
        },
        "same_raw_processor_comparison": {
            "agentview": _comparison(official["observation.images.image"], same_raw_custom_agent),
            "wrist": _comparison(official["observation.images.image2"], same_raw_custom_wrist),
            "state": _comparison(official["observation.state"], same_raw_custom_state),
        },
        "custom_reset_raw": {
            "agentview": _fingerprint(custom_raw["agentview_image"]),
            "wrist": _fingerprint(custom_raw["robot0_eye_in_hand_image"]),
        },
        "action_parity": action,
    }
    (args.output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
