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
from .transport import decode_action_response, observation_to_request


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

    def get_optim_params(self):
        raise RuntimeError("remote_jetson is an inference-only policy")

    def reset(self) -> None:
        self._episode_id += 1
        self._step_id = 0
        self._previous_action.fill(0.0)
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
        action = decode_action_response(payload)
        self._previous_action = action.copy()
        self._step_id += 1
        self._write_telemetry("predict", started, request, payload)
        return torch.from_numpy(action).unsqueeze(0)

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.select_action(batch).unsqueeze(1)

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
            "checkpoint": self.config.checkpoint,
            "revision": self.config.revision,
            "instruction": request.get("instruction"),
            "server_metadata": response.get("metadata", {}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
