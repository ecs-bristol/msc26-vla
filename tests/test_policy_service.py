from __future__ import annotations

import base64
import threading
import time
from io import BytesIO
from typing import Iterator

import numpy as np
import pytest
import requests
from PIL import Image

from libero_platform.deployment.policy_service import create_policy_server
from libero_platform.policies.base import EpisodeContext, PolicyAdapter, PolicyRequest
from libero_platform.policies.zero_policy import ZeroPolicyAdapter


@pytest.fixture
def service_url() -> Iterator[str]:
    server = create_policy_server(ZeroPolicyAdapter("zero_policy"), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def encoded_request() -> dict[str, object]:
    image = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return {
        "run_id": "run_1",
        "episode_id": 2,
        "step_id": 3,
        "instruction": "move the object",
        "images": {"agentview": base64.b64encode(encoded.getvalue()).decode("ascii")},
        "proprioception": [float(value) for value in range(8)],
        "previous_action": [-0.5] * 7,
    }


def test_health_metadata_and_predict(service_url: str, encoded_request: dict[str, object]) -> None:
    health = requests.get(f"{service_url}/health", timeout=2).json()
    assert health == {
        "schema_version": 1,
        "status": "ok",
        "policy": {
            "model_key": "zero_policy",
            "checkpoint": None,
            "revision": None,
            "precision": None,
            "device": "unavailable",
            "ready": False,
        },
    }

    metadata = requests.get(f"{service_url}/metadata", timeout=2).json()
    assert metadata["schema_version"] == 1
    assert metadata["model_key"] == "zero_policy"
    assert metadata["policy"] == health["policy"]

    response = requests.post(f"{service_url}/predict", json=encoded_request, timeout=2).json()
    assert response["schema_version"] == 1
    assert response["action"] == [0, 0, 0, 0, 0, 0, -1]


def test_reset_forwards_episode_context_to_the_policy() -> None:
    received: list[EpisodeContext] = []

    class ResetTrackingPolicy(ZeroPolicyAdapter):
        def begin_episode(self, context: EpisodeContext) -> None:
            received.append(context)

    server = create_policy_server(ResetTrackingPolicy("zero_policy"), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        response = requests.post(
            f"http://{host}:{port}/reset",
            json={
                "suite": "libero_spatial",
                "task_id": 3,
                "task_name": "pick_the_black_bowl",
                "initial_state_id": 45,
                "seed": 42,
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json() == {"schema_version": 1, "status": "ok"}
    assert received == [
        EpisodeContext(
            suite="libero_spatial",
            task_id=3,
            task_name="pick_the_black_bowl",
            initial_state_id=45,
            seed=42,
        )
    ]


def test_health_and_metadata_expose_loaded_smolvla_identity() -> None:
    from libero_platform.policies.smolvla_policy import SmolVLAPolicyAdapter

    class Runtime:
        device = "cuda"

        def load(self) -> None:
            return None

        def predict(self, batch: dict[str, object]) -> object:
            del batch
            return np.zeros(7, dtype=np.float32)

        def reset(self, seed: int) -> None:
            del seed

    adapter = SmolVLAPolicyAdapter(
        model_key="smolvla_libero",
        checkpoint="HuggingFaceVLA/smolvla_libero",
        revision="0123456789abcdef",
        precision="fp16",
        runtime=Runtime(),
    )
    adapter.load()
    server = create_policy_server(adapter, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    expected_policy = {
        "model_key": "smolvla_libero",
        "checkpoint": "HuggingFaceVLA/smolvla_libero",
        "revision": "0123456789abcdef",
        "precision": "fp16",
        "device": "cuda",
        "ready": True,
    }
    try:
        health = requests.get(f"http://{host}:{port}/health", timeout=2).json()
        metadata = requests.get(f"http://{host}:{port}/metadata", timeout=2).json()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health == {"schema_version": 1, "status": "ok", "policy": expected_policy}
    assert metadata["policy"] == expected_policy


def test_predict_requests_are_serialized(
    encoded_request: dict[str, object]
) -> None:
    active = 0
    maximum = 0

    class SlowPolicy(ZeroPolicyAdapter):
        def predict(self, request: PolicyRequest):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                time.sleep(0.1)
                return super().predict(request)
            finally:
                active -= 1

    server = create_policy_server(SlowPolicy("slow_policy"), "127.0.0.1", 0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address[:2]
    start_gate = threading.Barrier(3)
    responses: list[requests.Response] = []

    def post_prediction() -> None:
        start_gate.wait(timeout=2)
        responses.append(
            requests.post(
                f"http://{host}:{port}/predict", json=encoded_request, timeout=2
            )
        )

    request_threads = [threading.Thread(target=post_prediction) for _ in range(2)]
    for request_thread in request_threads:
        request_thread.start()
    try:
        start_gate.wait(timeout=2)
        for request_thread in request_threads:
            request_thread.join(timeout=3)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert [response.status_code for response in responses] == [200, 200]
    assert maximum == 1


def test_predict_rejects_request_larger_than_eight_mebibytes(service_url: str) -> None:
    response = requests.post(
        f"{service_url}/predict",
        data=b"x" * (8 * 1024 * 1024 + 1),
        headers={"Content-Type": "application/json"},
        timeout=5,
    )

    assert response.status_code == 413
    assert response.json()["schema_version"] == 1


def test_predict_rejects_malformed_images(service_url: str, encoded_request: dict[str, object]) -> None:
    encoded_request["images"] = {"agentview": "not-a-png"}

    response = requests.post(f"{service_url}/predict", json=encoded_request, timeout=2)

    assert response.status_code == 400
    assert response.json()["schema_version"] == 1


def test_predict_returns_internal_error_when_adapter_raises(
    encoded_request: dict[str, object]
) -> None:
    class RaisingPolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest):
            del request
            raise RuntimeError("model failure")

    server = create_policy_server(RaisingPolicy(), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        response = requests.post(
            f"http://{host}:{port}/predict", json=encoded_request, timeout=2
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 500
    assert response.json() == {"error": "prediction failed", "schema_version": 1}


def test_predict_ignores_arbitrary_module_and_shell_fields(
    service_url: str, encoded_request: dict[str, object]
) -> None:
    encoded_request["module"] = "os"
    encoded_request["command"] = "echo should-not-run"

    response = requests.post(f"{service_url}/predict", json=encoded_request, timeout=2)

    assert response.status_code == 200
    assert response.json()["action"] == [0, 0, 0, 0, 0, 0, -1]


def test_serve_policy_cli_constructs_only_the_zero_policy() -> None:
    from libero_platform.cli import main

    received: list[tuple[PolicyAdapter, str, int]] = []
    exit_code = main(
        ["serve-policy", "--policy", "zero_policy", "--host", "127.0.0.1", "--port", "8082"],
        serve_callable=lambda policy, host, port: received.append((policy, host, port)),
    )

    assert exit_code == 0
    assert len(received) == 1
    adapter, host, port = received[0]
    assert isinstance(adapter, ZeroPolicyAdapter)
    assert host == "127.0.0.1"
    assert port == 8082


def test_oversized_declared_request_returns_prompt_413_and_server_uses_daemon_threads() -> None:
    import socket
    import time

    server = create_policy_server(ZeroPolicyAdapter("zero_policy"), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = socket.create_connection(server.server_address, timeout=1)
    try:
        client.settimeout(1)
        client.sendall(
            b"POST /predict HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Length: 8388609\r\n"
            b"Content-Type: application/json\r\n\r\n"
        )
        started_at = time.monotonic()
        response = client.recv(1024)
        assert time.monotonic() - started_at < 0.5
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert server.daemon_threads is True
    assert b" 413 " in response


def test_predict_returns_500_for_non_json_serializable_adapter_response(
    encoded_request: dict[str, object]
) -> None:
    from libero_platform.policies.base import PolicyResponse

    class InvalidResponsePolicy(PolicyAdapter):
        def predict(self, request: PolicyRequest) -> PolicyResponse:
            del request
            return PolicyResponse(
                action=np.zeros(7, dtype=np.float32),
                inference_ms=0.0,
                model_key="invalid",
                device="cpu",
                metadata={"unsupported": object()},
            )

    server = create_policy_server(InvalidResponsePolicy(), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        response = requests.post(
            f"http://{host}:{port}/predict", json=encoded_request, timeout=2
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 500
    assert response.json() == {"error": "prediction failed", "schema_version": 1}


def test_predict_rejects_jpeg_and_excessive_decoded_pixels(
    service_url: str, encoded_request: dict[str, object]
) -> None:
    jpeg = BytesIO()
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(jpeg, format="JPEG")
    encoded_request["images"] = {
        "agentview": base64.b64encode(jpeg.getvalue()).decode("ascii")
    }

    jpeg_response = requests.post(f"{service_url}/predict", json=encoded_request, timeout=2)

    huge = BytesIO()
    Image.fromarray(np.zeros((2049, 2048, 3), dtype=np.uint8)).save(huge, format="PNG")
    encoded_request["images"] = {
        "agentview": base64.b64encode(huge.getvalue()).decode("ascii")
    }
    pixels_response = requests.post(
        f"{service_url}/predict", json=encoded_request, timeout=2
    )

    assert jpeg_response.status_code == 400
    assert pixels_response.status_code == 400


def test_serve_policy_cli_loads_and_closes_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from libero_platform import cli
    from libero_platform.policies import zero_policy

    calls: list[str] = []

    class RecordingAdapter(PolicyAdapter):
        def load(self) -> None:
            calls.append("load")

        def predict(self, request: PolicyRequest):
            raise AssertionError("serve-policy must not predict during startup")

        def close(self) -> None:
            calls.append("close")

    class RecordingServer:
        server_port = 8082

        def serve_forever(self) -> None:
            calls.append("serve")

        def server_close(self) -> None:
            calls.append("server_close")

    monkeypatch.setattr(zero_policy, "ZeroPolicyAdapter", lambda key: RecordingAdapter())
    from libero_platform.deployment import policy_service

    monkeypatch.setattr(policy_service, "create_policy_server", lambda *args: RecordingServer())

    assert cli.main(["serve-policy", "--policy", "zero_policy", "--port", "8082"]) == 0
    assert calls == ["load", "serve", "server_close", "close"]


def test_predict_rejects_aggregate_decoded_pixel_limit(
    service_url: str, encoded_request: dict[str, object]
) -> None:
    image = BytesIO()
    Image.fromarray(np.zeros((2048, 2048, 3), dtype=np.uint8)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    encoded_request["images"] = {
        "first": encoded,
        "second": encoded,
        "third": encoded,
    }

    response = requests.post(f"{service_url}/predict", json=encoded_request, timeout=2)

    assert response.status_code == 400


def test_serve_policy_cli_classifies_load_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from libero_platform import cli
    from libero_platform.policies import zero_policy

    class FailingLoadAdapter(PolicyAdapter):
        def load(self) -> None:
            raise RuntimeError("secret model path")

        def predict(self, request: PolicyRequest):
            raise AssertionError("load failure must prevent serving")

    monkeypatch.setattr(zero_policy, "ZeroPolicyAdapter", lambda key: FailingLoadAdapter())

    assert cli.main(["serve-policy", "--policy", "zero_policy"]) == 4
    output = capsys.readouterr()
    assert "Failure: policy_service_startup_failed" in output.err
    assert "secret model path" not in output.out + output.err
