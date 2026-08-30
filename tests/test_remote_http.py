from __future__ import annotations

import base64
import math
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import requests

from libero_platform.policies.base import EpisodeContext, PolicyRequest
from libero_platform.policies.remote_http import (
    RemoteHTTPPolicyAdapter,
    RemotePolicyUnavailable,
    probe_remote_policy,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class HealthResponse(FakeResponse):
    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def request() -> PolicyRequest:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[1, 2] = [255, 3, 7]
    return PolicyRequest(
        run_id="run_1",
        episode_id=2,
        step_id=3,
        instruction="move the object",
        images={"agentview": image},
        proprioception=np.arange(8, dtype=np.float32),
        previous_action=np.full(7, -0.5, dtype=np.float32),
    )


def test_remote_http_serializes_predict_request_with_base64_png_images() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "action": [0, 0, 0, 0, 0, 0, -1],
                "inference_ms": 12.5,
                "model_key": "remote-model",
                "device": "jetson",
                "metadata": {"temperature_c": 48.0},
            },
        )
    )
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test/", session)

    response = adapter.predict(request())

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "http://policy.test/predict"
    assert call["timeout"] == 10.0
    payload = call["json"]
    assert payload["run_id"] == "run_1"
    assert payload["episode_id"] == 2
    assert payload["step_id"] == 3
    assert payload["instruction"] == "move the object"
    assert payload["proprioception"] == [float(value) for value in range(8)]
    assert payload["previous_action"] == [-0.5] * 7
    encoded_image = payload["images"]["agentview"]
    with Image.open(BytesIO(base64.b64decode(encoded_image))) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (3, 2)
        assert decoded.getpixel((2, 1)) == (255, 3, 7)
    assert response.action.dtype == np.float32
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert response.inference_ms == 12.5
    assert response.model_key == "remote-model"
    assert response.device == "jetson"
    assert response.metadata["temperature_c"] == 48.0
    assert math.isfinite(response.metadata["remote_round_trip_ms"])


def test_remote_http_resets_remote_episode_with_the_full_seed_context() -> None:
    session = FakeSession(FakeResponse(200, {"status": "ok"}))
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test/", session)

    adapter.begin_episode(
        EpisodeContext(
            suite="libero_spatial",
            task_id=3,
            task_name="pick_the_black_bowl",
            initial_state_id=45,
            seed=42,
        )
    )

    assert session.calls == [
        {
            "url": "http://policy.test/reset",
            "json": {
                "suite": "libero_spatial",
                "task_id": 3,
                "task_name": "pick_the_black_bowl",
                "initial_state_id": 45,
                "seed": 42,
            },
            "timeout": 10.0,
        }
    ]


def test_remote_http_records_service_and_round_trip_latency() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                "inference_ms": 12.5,
                "service_latency_ms": 14.0,
                "metadata": {"device": "Orin"},
            },
        )
    )
    response = RemoteHTTPPolicyAdapter(
        "remote_http_policy", "http://policy.test", session
    ).predict(request())

    assert response.metadata["service_latency_ms"] == 14.0
    assert math.isfinite(response.metadata["remote_round_trip_ms"])


@pytest.mark.parametrize(
    "service_latency_ms",
    [float("nan"), float("inf"), "14.0", True],
    ids=["nan", "infinity", "string", "bool"],
)
def test_remote_http_ignores_invalid_service_latency_values(
    service_latency_ms: object,
) -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                "service_latency_ms": service_latency_ms,
            },
        )
    )

    response = RemoteHTTPPolicyAdapter(
        "remote_http_policy", "http://policy.test", session
    ).predict(request())

    assert "service_latency_ms" not in response.metadata
    assert math.isfinite(response.metadata["remote_round_trip_ms"])


def test_remote_http_classifies_non_success_status_as_remote_unavailable() -> None:
    session = FakeSession(FakeResponse(503, {"error": "offline"}))
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test", session)

    response = adapter.predict(request())

    assert response.failure_type == "remote_unavailable"
    assert "503" in response.error
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_remote_http_classifies_malformed_remote_action_as_invalid_action() -> None:
    session = FakeSession(FakeResponse(200, {"action": [0, 0, 0, 0, 0, 0]}))
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test", session)

    response = adapter.predict(request())

    assert response.failure_type == "invalid_action"
    assert "exactly 7" in response.error
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_remote_http_classifies_oversized_integer_action_as_invalid_action() -> None:
    session = FakeSession(
        FakeResponse(200, {"action": [0, 0, 0, 0, 0, 0, 10**4000]})
    )
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test", session)

    response = adapter.predict(request())

    assert response.failure_type == "invalid_action"
    assert "action" in response.error
    assert response.action.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_remote_http_network_failure_does_not_fall_back_to_a_local_policy() -> None:
    session = FakeSession(requests.ConnectionError("connection refused"))
    adapter = RemoteHTTPPolicyAdapter("remote_http_policy", "http://policy.test", session)

    response = adapter.predict(request())

    assert response.failure_type == "remote_unavailable"
    assert "connection refused" in response.error
    assert len(session.calls) == 1
    assert "service_latency_ms" not in response.metadata
    assert math.isfinite(response.metadata["remote_round_trip_ms"])
    assert response.metadata["remote_round_trip_ms"] >= 0.0


def test_probe_rejects_not_ready_server(monkeypatch) -> None:
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda url, timeout: HealthResponse(
            200,
            {
                "schema_version": 1,
                "status": "ok",
                "policy": {"ready": False},
            },
        ),
    )

    with pytest.raises(RemotePolicyUnavailable, match="not ready"):
        probe_remote_policy("http://jetson:8081")


def test_probe_returns_ready_policy_identity(monkeypatch) -> None:
    observed = {}

    def get(url: str, timeout: float) -> HealthResponse:
        observed.update(url=url, timeout=timeout)
        return HealthResponse(
            200,
            {
                "schema_version": 1,
                "status": "ok",
                "policy": {
                    "checkpoint": "HuggingFaceVLA/smolvla_libero",
                    "revision": "abc123",
                    "precision": "fp16",
                    "ready": True,
                },
            },
        )

    monkeypatch.setattr("libero_platform.policies.remote_http.requests.get", get)

    assert probe_remote_policy("http://10.42.0.2:8081/", timeout_s=1.5) == {
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "revision": "abc123",
        "precision": "fp16",
        "ready": True,
    }
    assert observed == {"url": "http://10.42.0.2:8081/health", "timeout": 1.5}


def test_probe_classifies_timeout_as_service_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout("timed out")),
    )

    with pytest.raises(RemotePolicyUnavailable, match="timed out"):
        probe_remote_policy("http://jetson:8081", timeout_s=0.5)


def test_probe_rejects_malformed_json(monkeypatch) -> None:
    class MalformedResponse(HealthResponse):
        def json(self) -> object:
            raise ValueError("invalid JSON")

    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: MalformedResponse(200, None),
    )

    with pytest.raises(RemotePolicyUnavailable, match="valid JSON"):
        probe_remote_policy("http://jetson:8081")


def test_probe_rejects_non_mapping_health_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: HealthResponse(200, ["not", "a", "mapping"]),
    )

    with pytest.raises(RemotePolicyUnavailable, match="must be a mapping"):
        probe_remote_policy("http://jetson:8081")
