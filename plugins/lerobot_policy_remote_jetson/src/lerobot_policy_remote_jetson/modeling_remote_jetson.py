from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_remote_jetson import RemoteJetsonConfig
from .transport import decode_action_chunk_response, observation_to_request


class RemoteJetsonPolicy(PreTrainedPolicy):
    config_class = RemoteJetsonConfig
    name = "remote_jetson"

    def __init__(
        self,
        config: RemoteJetsonConfig,
        *,
        session: requests.Session | None = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._session = session or requests.Session()
        self._run_id = f"official-lerobot-eval-{uuid.uuid4().hex[:12]}"
        self._episode_id = -1
        self._step_id = 0
        self._previous_action = np.zeros(7, dtype=np.float32)
        self._action_queue: list[np.ndarray] = []

    def get_optim_params(self):
        raise RuntimeError("remote_jetson is an inference-only policy")

    def reset(self) -> None:
        self._episode_id += 1
        self._step_id = 0
        self._previous_action.fill(0.0)
        self._action_queue.clear()
        payload = {
            "suite": self.config.suite,
            "task_id": 0,
            "task_name": "official_lerobot_eval",
            "initial_state_id": self._episode_id,
            "seed": self.config.seed + self._episode_id,
        }
        started = time.perf_counter()
        response = self._session.post(
            f"{self.config.endpoint}/reset",
            json=payload,
            timeout=self.config.reset_timeout_s,
        )
        response.raise_for_status()
        self._write_telemetry("reset", started, payload, response.json())

    @torch.inference_mode()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._action_queue:
            action_chunk = self._predict_action_chunk(batch)
            self._action_queue = [step.copy() for step in action_chunk]
        action = self._action_queue.pop(0)
        return torch.from_numpy(action).unsqueeze(0)

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        action_chunk = self._predict_action_chunk(batch)
        return torch.from_numpy(action_chunk).unsqueeze(0)

    def _predict_action_chunk(
        self, batch: dict[str, torch.Tensor]
    ) -> np.ndarray:
        request = observation_to_request(
            batch,
            run_id=self._run_id,
            episode_id=self._episode_id,
            step_id=self._step_id,
            previous_action=self._previous_action,
        )
        started = time.perf_counter()
        response = self._session.post(
            f"{self.config.endpoint}/predict",
            json=request,
            timeout=self.config.request_timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        action_chunk = decode_action_chunk_response(payload)
        self._previous_action = action_chunk[-1].copy()
        self._step_id += self.config.n_action_steps
        self._write_telemetry("predict", started, request, payload)
        return action_chunk

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        raise RuntimeError("remote_jetson is an inference-only policy")

    def _write_telemetry(
        self,
        event: str,
        started: float,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        if not self.config.telemetry_path:
            return
        path = Path(self.config.telemetry_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "run_id": self._run_id,
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "round_trip_ms": (time.perf_counter() - started) * 1000.0,
            "inference_ms": response.get("inference_ms"),
            "service_latency_ms": response.get("service_latency_ms"),
            "checkpoint": self.config.checkpoint,
            "revision": self.config.revision,
            "instruction": request.get("instruction"),
            "server_metadata": response.get("metadata", {}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
