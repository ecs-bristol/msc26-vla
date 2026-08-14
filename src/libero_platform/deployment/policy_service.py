from __future__ import annotations

import base64
import binascii
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from math import isfinite
from threading import Lock
from time import monotonic, perf_counter
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from libero_platform.deployment.device_profile import collect_device_profile
from libero_platform.policies.base import (
    EpisodeContext,
    PolicyAdapter,
    PolicyRequest,
    PolicyResponse,
    validate_action,
)

_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_PIXELS = 4 * 1024 * 1024
_MAX_TOTAL_IMAGE_PIXELS = 8 * 1024 * 1024
_REQUEST_SOCKET_TIMEOUT_SECONDS = 5.0
_SCHEMA_VERSION = 1


class PolicyThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        request.settimeout(_REQUEST_SOCKET_TIMEOUT_SECONDS)
        super().process_request_thread(request, client_address)


def create_policy_server(
    adapter: PolicyAdapter, host: str, port: int
) -> PolicyThreadingHTTPServer:
    """Create a policy server bound to one already-constructed adapter."""
    return PolicyThreadingHTTPServer((host, port), _handler_factory(adapter))


def _handler_factory(adapter: PolicyAdapter) -> type[BaseHTTPRequestHandler]:
    prediction_lock = Lock()

    class PolicyServiceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "policy": adapter.identity()},
                )
                return
            if self.path == "/metadata":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "model_key": _model_key(adapter),
                        "policy": adapter.identity(),
                        "device_profile": collect_device_profile(),
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/reset":
                self._reset_episode()
                return
            if self.path != "/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                request = _decode_request(self._read_json_body())
            except RequestTooLargeError:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request too large"},
                    connection_close=True,
                )
                self._drain_rejected_request()
                self.close_connection = True
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            started_at = perf_counter()
            try:
                with prediction_lock:
                    response = adapter.predict(request)
                payload = _predict_response_payload(response, started_at)
                self._send_json(HTTPStatus.OK, payload)
            except Exception:
                try:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": "prediction failed"},
                    )
                except (OSError, TypeError, ValueError):
                    self.close_connection = True

        def _reset_episode(self) -> None:
            try:
                context = _decode_episode_context(self._read_json_body())
            except RequestTooLargeError:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "request too large"},
                    connection_close=True,
                )
                self._drain_rejected_request()
                self.close_connection = True
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                with prediction_lock:
                    adapter.begin_episode(context)
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            except Exception:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "episode reset failed"}
                )

        def _read_json_body(self) -> Mapping[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("Content-Length is required")
            try:
                length = int(content_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0:
                raise ValueError("invalid Content-Length")
            if length > _MAX_REQUEST_BYTES:
                raise RequestTooLargeError
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
                raise ValueError("request body must be JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError("request body must be a mapping")
            return payload

        def _drain_rejected_request(self) -> None:
            original_timeout = self.connection.gettimeout()
            self.connection.settimeout(0.1)
            deadline = monotonic() + 0.5
            try:
                while monotonic() < deadline:
                    chunk = self.rfile.read1(64 * 1024)
                    if not chunk:
                        return
            except OSError:
                return
            finally:
                self.connection.settimeout(original_timeout)

        def _send_json(
            self,
            status: HTTPStatus,
            payload: Mapping[str, object],
            *,
            connection_close: bool = False,
        ) -> None:
            body = json.dumps(
                {"schema_version": _SCHEMA_VERSION, **payload},
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if connection_close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return PolicyServiceHandler


class RequestTooLargeError(ValueError):
    pass


def _predict_response_payload(
    response: PolicyResponse, started_at: float
) -> dict[str, object]:
    action = validate_action(response.action)
    inference_ms = float(response.inference_ms)
    service_latency_ms = (perf_counter() - started_at) * 1000.0
    if not isfinite(inference_ms) or not isfinite(service_latency_ms):
        raise ValueError("policy response latency must be finite")
    return {
        "action": action.tolist(),
        "inference_ms": inference_ms,
        "model_key": response.model_key,
        "device": response.device,
        "failure_type": response.failure_type,
        "error": response.error,
        "metadata": response.metadata,
        "service_latency_ms": service_latency_ms,
    }


def _decode_request(payload: Mapping[str, Any]) -> PolicyRequest:
    previous = payload.get("previous_action")
    return PolicyRequest(
        run_id=_required_string(payload, "run_id"),
        episode_id=_non_negative_int(payload, "episode_id"),
        step_id=_non_negative_int(payload, "step_id"),
        instruction=_required_string(payload, "instruction"),
        images=_decode_images(payload.get("images")),
        proprioception=_finite_vector(payload.get("proprioception"), "proprioception"),
        previous_action=None if previous is None else validate_action(previous),
    )


def _decode_episode_context(payload: Mapping[str, Any]) -> EpisodeContext:
    return EpisodeContext(
        suite=_required_string(payload, "suite"),
        task_id=_non_negative_int(payload, "task_id"),
        task_name=_required_string(payload, "task_name"),
        initial_state_id=_non_negative_int(payload, "initial_state_id"),
        seed=_non_negative_int(payload, "seed"),
    )


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _non_negative_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_vector(value: object, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain numeric float32 values") from exc
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a finite one-dimensional vector")
    return vector


def _decode_images(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("images must be a non-empty mapping")
    images: dict[str, np.ndarray] = {}
    total_pixels = 0
    for name, encoded in value.items():
        if not isinstance(name, str) or not name or not isinstance(encoded, str):
            raise ValueError("images must map non-empty names to base64 PNG strings")
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(raw)) as image:
                if image.format != "PNG":
                    raise ValueError("images must contain PNG data")
                width, height = image.size
                pixels = width * height
                if pixels > _MAX_IMAGE_PIXELS:
                    raise ValueError("image exceeds decoded pixel limit")
                total_pixels += pixels
                if total_pixels > _MAX_TOTAL_IMAGE_PIXELS:
                    raise ValueError("images exceed aggregate decoded pixel limit")
                image.load()
                images[name] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError("images must contain valid PNG data") from exc
    return images


def _model_key(adapter: PolicyAdapter) -> str:
    model_key = getattr(adapter, "_model_key", None)
    return model_key if isinstance(model_key, str) and model_key else type(adapter).__name__
