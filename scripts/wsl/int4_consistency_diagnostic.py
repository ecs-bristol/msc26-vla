from __future__ import annotations

import json

import torch

from lerobot_policy_smolvla_int4.configuration_smolvla_int4 import (
    SmolVLAInt4Config,
)
from lerobot_policy_smolvla_int4.int4_linear import Int4WeightOnlyLinear
from lerobot_policy_smolvla_int4.modeling_smolvla_int4 import SmolVLAInt4Policy


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _count(config: SmolVLAInt4Config):
    policy = SmolVLAInt4Policy(config)
    total = 0
    minus8 = 0
    modules = {}
    for name, module in policy.named_modules():
        if not isinstance(module, Int4WeightOnlyLinear):
            continue
        low = module.weight_packed & 0x0F
        high = (module.weight_packed >> 4) & 0x0F
        quantized_unsigned = torch.stack((low, high), dim=-1).view(
            module.out_features, module.padded_in
        )
        signed = quantized_unsigned.to(torch.int16) - 8
        count = int((signed == -8).sum().item())
        total += signed.numel()
        minus8 += count
        if count:
            modules[name] = count
    return total, minus8, modules


def main():
    results = {}
    for label, quant_method, quant_scope in (
        ("mixed", "mixed", "language"),
        ("full_backbone_int4", "int4_groupwise", "backbone"),
    ):
        config = SmolVLAInt4Config(
            checkpoint=CHECKPOINT,
            revision=REVISION,
            quant_method=quant_method,
            quant_scope=quant_scope,
            vision_bits=4,
            connector_bits=4,
            text_bits=8 if quant_method == "mixed" else 4,
            num_steps=2,
            n_action_steps=1,
            device="cuda",
        )
        total, minus8, modules = _count(config)
        results[label] = {
            "total_int4_values": total,
            "old_minus8_count": minus8,
            "modules_with_minus8": modules,
        }
    output = {
        "checkpoint": CHECKPOINT,
        "revision": REVISION,
        "old_range": [-8, 7],
        "new_range": [-7, 7],
        "mixed": results["mixed"],
        "full_backbone_int4": results["full_backbone_int4"],
        "mixed_closed_loop_rerun_required": (
            results["mixed"]["old_minus8_count"] > 0
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
