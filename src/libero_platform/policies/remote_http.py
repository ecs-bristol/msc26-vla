from __future__ import annotations

import base64
import math
from io import BytesIO
from time import perf_counter
from typing import Mapping

import numpy as np
import requests
from PIL import Image

from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    PolicyResponse,
    validate_action,
)

_PREDICT_TIMEOUT_SECONDS = 10.0
_FAILURE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class RemotePolicyUnavailable(RuntimeError):
    """Raised when a remote policy cannot provide a ready health contract."""


def probe_remote_policy(endpoint: str, timeout_s: float = 2.0) -> dict[str, object]:
    """Return a ready remote policy identity after a bounded health probe."""
    if timeout_s <= 0:
        raise ValueError("remote policy probe timeout must be positive")
    health_url = f"{endpoint.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RemotePolicyUnavailable(
            f"remote policy service unavailable at {health_url}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RemotePolicyUnavailable("remote health response is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RemotePolicyUnavailable("remote health response must be a mapping")
    if payload.get("schema_version") != 1:
        raise RemotePolicyUnavailable("remote health response has unsupported schema_version")
    if payload.get("status") != "ok":
        raise RemotePolicyUnavailable("remote health response is not ready")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise RemotePolicyUnavailable("remote health response is missing policy identity")
    if policy.get("ready") is not True:
        raise RemotePolicyUnavailable("remote policy is not ready")
    return dict(policy)


class RemoteHTTPPolicyAdapter(PolicyAdapter):
    def __init__(
        self, model_key: str, endpoint: str, session: requests.Session
    ) -> None:
        self._model_key = model_key
        self._endpoint = endpoint.rstrip("/")
        self._session = session

    def begin_episode(self, context: EpisodeContext) -> None:
        try:
            response = self._session.post(
                f"{self._endpoint}/reset",
                json=_serialize_episode_context(context),
                timeout=_PREDICT_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise RemotePolicyUnavailable(
                f"remote policy reset unavailable at {self._endpoint}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise RemotePolicyUnavailable(
                f"remote policy reset returned HTTP {response.status_code}: {response.text}"
            )

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        started_at = perf_counter()
        try:
            response = self._session.post(
                f"{self._endpoint}/predict",
                json=_serialize_request(request),
                timeout=_PREDICT_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return self._failure("remote_unavailable", str(exc), started_at)

        if response.status_code != 200:
            return self._failure(
                "remote_unavailable",
                f"remote policy returned HTTP {response.status_code}: {response.text}",
                started_at,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return self._failure("remote_unavailable", f"invalid remote JSON: {exc}", started_at)
        if not isinstance(payload, Mapping):
            return self._failure("invalid_action", "remote response must be a mapping", started_at)

        try:
            action = validate_action(payload.get("action"))
        except (TypeError, ValueError) as exc:
            return self._failure("invalid_action", str(exc), started_at)

        metadata = _response_metadata(payload.get("metadata"))
        service_latency = payload.get("service_latency_ms")
        if (
            isinstance(service_latency, (int, float))
            and not isinstance(service_latency, bool)
            and math.isfinite(float(service_latency))
        ):
            metadata["service_latency_ms"] = float(service_latency)
        metadata["remote_round_trip_ms"] = _elapsed_ms(started_at)

        return PolicyResponse(
            action=action,
            inference_ms=float(payload.get("inference_ms", _elapsed_ms(started_at))),
            model_key=str(payload.get("model_key", self._model_key)),
            device=str(payload.get("device", "remote")),
            failure_type=str(payload.get("failure_type", "")),
            error=str(payload.get("error", "")),
            metadata=metadata,
        )

    def _failure(
        self, failure_type: str, error: str, started_at: float
    ) -> PolicyResponse:
        return PolicyResponse(
            action=_FAILURE_ACTION.copy(),
            inference_ms=_elapsed_ms(started_at),
            model_key=self._model_key,
            device="remote",
            failure_type=failure_type,
            error=error,
            metadata={"remote_round_trip_ms": max(_elapsed_ms(started_at), 0.0)},
        )


def _serialize_request(request: PolicyRequest) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "episode_id": request.episode_id,
        "step_id": request.step_id,
        "instruction": request.instruction,
        "images": {
            name: _encode_png(image) for name, image in request.images.items()
        },
        "proprioception": request.proprioception.tolist(),
        "previous_action": (
            None if request.previous_action is None else request.previous_action.tolist()
        ),
    }


def _serialize_episode_context(context: EpisodeContext) -> dict[str, object]:
    return {
        "suite": context.suite,
        "task_id": context.task_id,
        "task_name": context.task_name,
        "initial_state_id": context.initial_state_id,
        "seed": context.seed,
    }


def _encode_png(image: np.ndarray) -> str:
    encoded = BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def _response_metadata(value: object) -> dict[str, str | int | float | None]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (str, int, float)) or item is None
    }


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000.0
