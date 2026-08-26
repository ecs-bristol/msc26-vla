from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


def _torch_dtype(trt_dtype: object) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    return mapping[trt_dtype]


class TensorRTVisionEncoder(nn.Module):
    """Run the exported SmolVLA vision encoder through a TensorRT engine.

    The wrapper keeps the same forward contract as the Transformers
    ``SmolVLMVisionTransformer``: it accepts ``pixel_values`` and an optional
    ``patch_attention_mask`` (ignored because the mask is baked into the
    exported ONNX graph) and returns a ``BaseModelOutput`` with
    ``last_hidden_state``.
    """

    def __init__(self, engine_path: str | Path, device: str = "cuda") -> None:
        super().__init__()
        import tensorrt as trt

        self._trt = trt
        self.device = torch.device(device)
        engine_path = Path(engine_path).expanduser()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        self.context = self.engine.create_execution_context()

        self._input_name = "pixel_values"
        self._output_name = "vision_features"
        self._allocate_bindings()

    @property
    def dtype(self) -> torch.dtype:
        return self._input_tensor.dtype

    def _allocate_bindings(self) -> None:
        self._input_tensor = self._make_tensor(self._input_name)
        self._output_tensor = self._make_tensor(self._output_name)
        self.context.set_tensor_address(
            self._input_name, self._input_tensor.data_ptr()
        )
        self.context.set_tensor_address(
            self._output_name, self._output_tensor.data_ptr()
        )

    def _make_tensor(self, name: str) -> torch.Tensor:
        shape = tuple(self.engine.get_tensor_shape(name))
        dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
        return torch.empty(shape, dtype=dtype, device=self.device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        del patch_attention_mask, kwargs
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=self._input_tensor.dtype,
            copy=False,
        ).contiguous()
        self._input_tensor.copy_(pixel_values)
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT vision encoder execution failed")
        return BaseModelOutput(last_hidden_state=self._output_tensor)


class TensorRTConnector(nn.Module):
    """Run the exported SmolVLA connector through a TensorRT engine."""

    def __init__(self, engine_path: str | Path, device: str = "cuda") -> None:
        super().__init__()
        import tensorrt as trt

        self._trt = trt
        self.device = torch.device(device)
        engine_path = Path(engine_path).expanduser()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"failed to load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self._input_name = "image_hidden_states"
        self._output_name = "connector_features"
        self._input_tensor = self._make_tensor(self._input_name)
        self._output_tensor = self._make_tensor(self._output_name)
        self.context.set_tensor_address(
            self._input_name, self._input_tensor.data_ptr()
        )
        self.context.set_tensor_address(
            self._output_name, self._output_tensor.data_ptr()
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._input_tensor.dtype

    def _make_tensor(self, name: str) -> torch.Tensor:
        shape = tuple(self.engine.get_tensor_shape(name))
        dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
        return torch.empty(shape, dtype=dtype, device=self.device)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = image_hidden_states.to(
            device=self.device,
            dtype=self._input_tensor.dtype,
            copy=False,
        ).contiguous()
        self._input_tensor.copy_(image_hidden_states)
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT connector execution failed")
        return self._output_tensor
