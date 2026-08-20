from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image


CAMERA_KEYS = {
    "observation.images.image": "agentview",
    "observation.images.image2": "wrist",
}


def _single_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 0 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got shape {tuple(tensor.shape)}")
    return tensor[0].detach().cpu()


def _encode_image(value: Any, name: str) -> str:
    tensor = _single_tensor(value, name)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be CHW or HWC, got shape {tuple(tensor.shape)}")
    array = tensor.numpy()
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.asarray(np.rint(array), dtype=np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _instruction(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    raise ValueError("task must contain exactly one instruction")


def observation_to_request(
    observation: Mapping[str, Any],
    *,
    run_id: str,
    episode_id: int,
    step_id: int,
    previous_action: np.ndarray,
) -> dict[str, Any]:
    missing = [key for key in (*CAMERA_KEYS, "observation.state", "task") if key not in observation]
    if missing:
        raise ValueError(f"official LeRobot observation is missing: {', '.join(missing)}")

    state = _single_tensor(observation["observation.state"], "observation.state")
    if state.ndim != 1 or state.shape[0] != 8:
        raise ValueError(f"observation.state must have shape (1, 8), got {(1, *state.shape)}")

    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "step_id": step_id,
        "instruction": _instruction(observation["task"]),
        "images": {wire: _encode_image(observation[key], key) for key, wire in CAMERA_KEYS.items()},
        "proprioception": state.to(torch.float32).numpy().tolist(),
        "previous_action": np.asarray(previous_action, dtype=np.float32).tolist(),
    }


def decode_action_response(payload: Mapping[str, Any]) -> np.ndarray:
    if "action" not in payload:
        raise ValueError("remote response is missing action")
    action = np.asarray(payload["action"], dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"remote action must contain exactly 7 values, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("remote action must contain only finite values")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(
            "remote action must stay in [-1, 1]; "
            f"observed min={float(action.min()):.6g}, max={float(action.max()):.6g}"
        )
    return action
