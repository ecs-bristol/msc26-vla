from __future__ import annotations

import sys
from pathlib import Path

import torch

from libero_platform.policies.smolvla_policy import (
    LeRobotSmolVLARuntime,
    SmolVLAInferenceSpec,
)


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
OUTPUT_PATH = Path("/workspace/outputs/smolvla_vision.onnx")


class VisionWrapper(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        patch_attention_mask = torch.ones(
            (1, 32, 32),
            dtype=torch.bool,
            device=pixel_values.device,
        )
        output = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )
        return output.last_hidden_state


def main() -> int:
    runtime = LeRobotSmolVLARuntime(
        checkpoint=CHECKPOINT,
        precision="fp16",
        smolvla_inference=SmolVLAInferenceSpec(
            n_action_steps=20,
            num_steps=2,
            chunk_size=20,
        ),
        revision=REVISION,
    )
    runtime.load()

    policy = runtime._policy
    vlm = policy.model.vlm_with_expert.vlm
    vision_model = vlm.model.vision_model

    vision_model.config._attn_implementation = "eager"
    vision_model.config.attn_implementation = "eager"
    vision_model.eval()
    vision_model.to("cpu")
    vision_model.to(torch.float16)

    pixel_values = torch.zeros((1, 3, 512, 512), dtype=torch.float16)
    wrapper = VisionWrapper(vision_model)
    wrapper.eval()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (pixel_values,),
        str(OUTPUT_PATH),
        input_names=["pixel_values"],
        output_names=["vision_features"],
        opset_version=17,
        dynamo=True,
    )
    print(f"exported {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
