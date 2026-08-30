from __future__ import annotations

import inspect
import sys

from libero_platform.policies.smolvla_policy import (
    LeRobotSmolVLARuntime,
    SmolVLAInferenceSpec,
)


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"


def _describe(name: str, obj: object) -> None:
    print(f"{name}: {type(obj).__module__}.{type(obj).__name__}")
    forward = getattr(obj, "forward", None)
    if forward is not None:
        try:
            print(f"  forward{inspect.signature(forward)}")
        except (TypeError, ValueError):
            print("  forward: <signature unavailable>")


def main() -> int:
    inference = SmolVLAInferenceSpec(
        n_action_steps=20,
        num_steps=2,
        chunk_size=20,
    )
    runtime = LeRobotSmolVLARuntime(
        checkpoint=CHECKPOINT,
        precision="fp16",
        smolvla_inference=inference,
        revision=REVISION,
    )
    runtime.load()
    policy = runtime._policy

    model = getattr(policy, "model", None)
    vlm_with_expert = getattr(model, "vlm_with_expert", None)
    vlm = getattr(vlm_with_expert, "vlm", None)
    vlm_model = getattr(vlm, "model", None)

    print("policy.model:", type(model).__name__ if model is not None else None)
    print(
        "vlm_with_expert:",
        type(vlm_with_expert).__name__ if vlm_with_expert is not None else None,
    )
    print("vlm:", type(vlm).__name__ if vlm is not None else None)

    for name in ("vision_model", "connector", "text_model"):
        part = getattr(vlm_model, name, None) if vlm_model is not None else None
        _describe(name, part)

    print("--- forward signatures ---")
    _describe("vlm_with_expert", vlm_with_expert)
    _describe("vlm", vlm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
