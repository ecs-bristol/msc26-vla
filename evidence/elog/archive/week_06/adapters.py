from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from .action_schema import build_action_prompt
from .unified_schema import AdapterRequest, AdapterResponse, parse_action_tokens


class BaseAdapter(ABC):
    adapter_name = "base_adapter"

    def __init__(self, *, model_key: str, model_cfg: dict[str, Any]) -> None:
        self.model_key = model_key
        self.model_cfg = model_cfg
        self.model_id = str(model_cfg.get("model_id", model_key))
        self.model_load_ms = 0.0

    def load(self, *, local_files_only: bool = False) -> None:
        self.model_load_ms = 0.0

    @abstractmethod
    def predict(self, request: AdapterRequest) -> AdapterResponse:
        raise NotImplementedError


class ScriptedAdapter(BaseAdapter):
    adapter_name = "scripted_adapter"

    def predict(self, request: AdapterRequest) -> AdapterResponse:
        start = time.perf_counter()
        expected_target = request.expected_target.strip() or "object"
        expected_action = request.expected_action.strip()
        action_plan = ["move_forward"]
        if expected_action and expected_action not in action_plan:
            action_plan.append(expected_action)
        action_text = " ".join(action_plan)
        raw_output = (
            f"target: {expected_target}; "
            f"actions: {action_text}; "
            "reason: local scripted adapter for offline workflow validation."
        )
        return AdapterResponse(
            model_key=request.model_key,
            adapter=self.adapter_name,
            deployment_mode=request.deployment_mode,
            task_id=request.task_id,
            target=expected_target,
            action_text=action_text,
            raw_output=raw_output,
            parsed_actions=parse_action_tokens(raw_output),
            latency_ms=(time.perf_counter() - start) * 1000,
            model_load_ms=self.model_load_ms,
            success=True,
        )


class MockAdapter(BaseAdapter):
    adapter_name = "mock_adapter"

    def predict(self, request: AdapterRequest) -> AdapterResponse:
        start = time.perf_counter()
        target = request.expected_target.strip() or "object"
        expected_action = request.expected_action.strip() or "stop"
        action_text = f"move_forward {expected_action}"
        raw_output = (
            f"target: {target}; "
            f"actions: {action_text}; "
            "reason: mock adapter response for interface and UI validation."
        )
        return AdapterResponse(
            model_key=request.model_key,
            adapter=self.adapter_name,
            deployment_mode=request.deployment_mode,
            task_id=request.task_id,
            target=target,
            action_text=action_text,
            raw_output=raw_output,
            parsed_actions=parse_action_tokens(raw_output),
            latency_ms=(time.perf_counter() - start) * 1000,
            model_load_ms=self.model_load_ms,
            success=True,
        )


class VLMTextAdapter(BaseAdapter):
    adapter_name = "vlm_text_adapter"

    def __init__(self, *, model_key: str, model_cfg: dict[str, Any]) -> None:
        super().__init__(model_key=model_key, model_cfg=model_cfg)
        self.processor: Any = None
        self.model: Any = None

    def load(self, *, local_files_only: bool = False) -> None:
        from .run_inference import load_model

        start = time.perf_counter()
        self.processor, self.model = load_model(self.model_cfg, local_files_only=local_files_only)
        self.model_load_ms = (time.perf_counter() - start) * 1000

    def predict(self, request: AdapterRequest) -> AdapterResponse:
        if self.processor is None or self.model is None:
            return AdapterResponse(
                model_key=request.model_key,
                adapter=self.adapter_name,
                deployment_mode=request.deployment_mode,
                task_id=request.task_id,
                load_success=False,
                success=False,
                failure_type="load_error",
                error="VLMTextAdapter.load() must be called before predict().",
            )

        import torch

        from .pre_jetson_runner import cuda_peak_gb, cuda_reset_peak, cuda_sync
        from .run_inference import build_processor_inputs, move_inputs_to_device

        image = Image.open(request.image_path).convert("RGB")
        prompt = build_action_prompt(request.instruction)
        max_new_tokens = int(request.runtime_config.get("max_new_tokens", 80))

        cuda_reset_peak()
        cuda_sync()
        start = time.perf_counter()
        inputs = build_processor_inputs(self.processor, image, request.image_path, prompt)
        inputs = move_inputs_to_device(inputs, self.model)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        cuda_sync()
        latency_ms = (time.perf_counter() - start) * 1000

        input_ids = inputs.get("input_ids")
        if input_ids is not None and generated.shape[-1] > input_ids.shape[-1]:
            decoded_tokens = generated[:, input_ids.shape[-1] :]
        else:
            decoded_tokens = generated
        output_text = self.processor.batch_decode(decoded_tokens, skip_special_tokens=True)[0].strip()
        peak_memory_gb = cuda_peak_gb()
        return AdapterResponse(
            model_key=request.model_key,
            adapter=self.adapter_name,
            deployment_mode=request.deployment_mode,
            task_id=request.task_id,
            target=request.expected_target,
            action_text=output_text,
            raw_output=output_text,
            parsed_actions=parse_action_tokens(output_text),
            latency_ms=latency_ms,
            model_load_ms=self.model_load_ms,
            peak_memory_mb=peak_memory_gb * 1024 if peak_memory_gb is not None else None,
            success=True,
        )


class OpenVLAAdapter(BaseAdapter):
    adapter_name = "openvla_adapter"

    def predict(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(
            model_key=request.model_key,
            adapter=self.adapter_name,
            deployment_mode=request.deployment_mode,
            task_id=request.task_id,
            load_success=False,
            success=False,
            failure_type="adapter_unavailable",
            error="OpenVLA action adapter is reserved for the later model-specific probe path.",
        )


class RemoteHTTPAdapter(BaseAdapter):
    adapter_name = "remote_http_adapter"

    def predict(self, request: AdapterRequest) -> AdapterResponse:
        endpoint = str(request.runtime_config.get("endpoint", self.model_cfg.get("endpoint", ""))).strip()
        timeout_sec = float(request.runtime_config.get("timeout_sec", self.model_cfg.get("timeout_sec", 30)))
        if not endpoint:
            return AdapterResponse(
                model_key=request.model_key,
                adapter=self.adapter_name,
                deployment_mode=request.deployment_mode,
                task_id=request.task_id,
                success=False,
                failure_type="config_error",
                error="remote_http_adapter requires an endpoint in runtime_config or models.yaml.",
            )

        start = time.perf_counter()
        payload = {
            "task_id": request.task_id,
            "instruction": request.instruction,
            "expected_target": request.expected_target,
            "expected_action": request.expected_action,
            "image_b64": _read_image_b64(request.image_path),
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return AdapterResponse(
                model_key=request.model_key,
                adapter=self.adapter_name,
                deployment_mode=request.deployment_mode,
                task_id=request.task_id,
                success=False,
                failure_type="remote_unavailable",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        raw_output = str(data.get("raw_output", data.get("action_text", "")))
        action_text = str(data.get("action_text", raw_output))
        target = str(data.get("target", request.expected_target))
        return AdapterResponse(
            model_key=request.model_key,
            adapter=self.adapter_name,
            deployment_mode=request.deployment_mode,
            task_id=request.task_id,
            target=target,
            action_text=action_text,
            raw_output=raw_output,
            parsed_actions=parse_action_tokens(raw_output or action_text),
            confidence=data.get("confidence"),
            latency_ms=(time.perf_counter() - start) * 1000,
            model_load_ms=self.model_load_ms,
            success=True,
        )


def _read_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def create_adapter(model_key: str, model_cfg: dict[str, Any]) -> BaseAdapter:
    adapter_name = str(model_cfg.get("adapter", "")).strip()
    if not adapter_name:
        model_type = model_cfg.get("model_type", "")
        if model_type == "rule_baseline":
            adapter_name = "scripted_adapter"
        elif model_type == "openvla":
            adapter_name = "openvla_adapter"
        else:
            adapter_name = "vlm_text_adapter"

    adapter_classes = {
        "scripted_adapter": ScriptedAdapter,
        "mock_adapter": MockAdapter,
        "vlm_text_adapter": VLMTextAdapter,
        "openvla_adapter": OpenVLAAdapter,
        "remote_http_adapter": RemoteHTTPAdapter,
    }
    try:
        adapter_class = adapter_classes[adapter_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported adapter: {adapter_name}") from exc
    return adapter_class(model_key=model_key, model_cfg={**model_cfg, "adapter": adapter_name})
