from __future__ import annotations

import torch
from torch import nn


class Int4WeightOnlyLinear(nn.Module):
    """Symmetric per-group 4-bit weight-only linear layer.

    Weights are stored as packed uint8 (two signed int4 values per byte) plus a
    float32 per-group scale. The forward dequantizes to the input dtype, so the
    layer works under fp32 and fp16/autocast without any external kernels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._weight_dtype_probe = torch.empty(0, dtype=torch.float32)
        n_groups = (in_features + group_size - 1) // group_size
        padded_in = n_groups * group_size
        self.padded_in = padded_in
        self.weight_packed = nn.Parameter(
            torch.zeros(out_features, padded_in // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
    ) -> "Int4WeightOnlyLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
        )
        original_dtype = linear.weight.dtype
        module._weight_dtype_probe = torch.empty(0, dtype=original_dtype)
        weight = linear.weight.detach().float()
        out_features, in_features = weight.shape
        padding = module.padded_in - in_features
        if padding:
            weight = torch.nn.functional.pad(weight, (0, padding))

        weight = weight.view(out_features, -1, group_size)
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 7.0
        quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
        quantized_unsigned = (quantized + 8).to(torch.uint8).view(out_features, -1)

        even = quantized_unsigned[:, 0::2]
        odd = quantized_unsigned[:, 1::2]
        module.weight_packed.data = (even | (odd << 4)).to(torch.uint8)
        module.weight_scale.data = scale.squeeze(-1)
        if linear.bias is not None:
            module.bias.data = linear.bias.detach().float()
        return module

    @property
    def weight(self) -> torch.Tensor:
        """Expose a dtype probe so transformers dtype checks keep working.

        SmolVLA's custom attention layers read `linear.weight.dtype` to decide
        how to cast activations. The real packed int4 weights live in
        `weight_packed` / `weight_scale`; this probe only reports the compute
        dtype without materializing a full weight tensor.
        """
        return self._weight_dtype_probe

    def _dequantized_weight(self) -> torch.Tensor:
        low = self.weight_packed & 0x0F
        high = (self.weight_packed >> 4) & 0x0F
        quantized_unsigned = torch.stack((low, high), dim=-1).view(
            self.out_features, self.padded_in
        )
        signed = quantized_unsigned.to(torch.int16) - 8
        weight = signed.view(self.out_features, -1, self.group_size).to(torch.float32)
        weight = weight * self.weight_scale.unsqueeze(-1)
        return weight.view(self.out_features, self.padded_in)[:, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight().to(x.dtype)
        if self.bias is not None:
            return torch.nn.functional.linear(x, weight, self.bias.to(x.dtype))
        return torch.nn.functional.linear(x, weight)
