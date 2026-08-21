import torch

from lerobot_policy_smolvla_int4.int4_linear import Int4WeightOnlyLinear


def test_int4_linear_output_stays_close_to_fp16():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 64)
    quantized = Int4WeightOnlyLinear.from_float(linear, group_size=128)
    x = torch.randn(4, 256)

    expected = linear(x)
    actual = quantized(x)

    relative_error = (actual - expected).abs().mean() / expected.abs().mean()
    assert relative_error < 0.2


def test_int4_linear_reduces_weight_memory():
    linear = torch.nn.Linear(768, 768, bias=True)
    quantized = Int4WeightOnlyLinear.from_float(linear, group_size=128)

    fp_bytes = sum(p.numel() * p.element_size() for p in linear.parameters())
    int4_bytes = sum(p.numel() * p.element_size() for p in quantized.parameters())

    assert int4_bytes < fp_bytes / 3
