from __future__ import annotations

import torch
from torch import nn


class Int8WeightOnlyLinear(nn.Module):
    """Symmetric per-group 8-bit weight-only linear layer.

    Weights are stored as int8 plus a float32 per-group scale and dequantized
    to the input dtype on the forward pass. This is the low-risk quantization
    baseline that keeps success while cutting weight memory by ~4x vs fp32.
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
        n_groups = (in_features + group_size - 1) // group_size
        padded_in = n_groups * group_size
        self.padded_in = padded_in
        self.weight_int8 = nn.Parameter(
            torch.zeros(out_features, padded_in, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(out_features, n_groups, dtype=torch.float32),
            requires_grad=False,
        )
        self._weight_dtype_probe = torch.empty(0, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
    ) -> "Int8WeightOnlyLinear":
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
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
        quantized = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
        module.weight_int8.data = quantized.view(out_features, module.padded_in)
        module.weight_scale.data = scale.squeeze(-1)
        if linear.bias is not None:
            module.bias.data = linear.bias.detach().float()
        return module

    @property
    def weight(self) -> torch.Tensor:
        """Dtype probe so SmolVLA attention dtype checks keep working."""
        return self._weight_dtype_probe

    def _dequantized_weight(self) -> torch.Tensor:
        weight = self.weight_int8.to(torch.float32).view(
            self.out_features, -1, self.group_size
        )
        weight = weight * self.weight_scale.unsqueeze(-1)
        return weight.view(self.out_features, self.padded_in)[:, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight().to(x.dtype)
        if self.bias is not None:
            return torch.nn.functional.linear(x, weight, self.bias.to(x.dtype))
        return torch.nn.functional.linear(x, weight)
