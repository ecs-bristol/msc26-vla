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
OUTPUT_PATH = Path("/workspace/outputs/smolvla_connector.onnx")


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
    connector = vlm.model.connector

    vision_model.config._attn_implementation = "eager"
    vision_model.config.attn_implementation = "eager"
    vision_model.eval()
    vision_model.to("cpu")
    vision_model.to(torch.float32)

    pixel_values = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        reference = vision_model(pixel_values)
    image_hidden_states = reference.last_hidden_state

    connector.eval()
    connector.to("cpu")
    connector.to(torch.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        connector,
        (image_hidden_states,),
        str(OUTPUT_PATH),
        input_names=["image_hidden_states"],
        output_names=["connector_features"],
        opset_version=17,
    )
    print(f"exported {OUTPUT_PATH}")
    print("input_shape", tuple(image_hidden_states.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
