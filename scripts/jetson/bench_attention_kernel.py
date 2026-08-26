from __future__ import annotations

import statistics
import sys

import torch

from libero_platform.policies.smolvla_policy import (
    LeRobotSmolVLARuntime,
    SmolVLAInferenceSpec,
)


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SEQ_LENGTH = 48
WARMUP = 10
ITERS = 30


def _measure(model, input_ids, attention_mask, mode):
    model.config._attn_implementation = mode
    model.config.attn_implementation = mode

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(ITERS):
            start.record()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    latencies.sort()
    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 3),
        "peak_allocated_mb": round(peak_mb, 1),
    }


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

    text_model = runtime._policy.model.vlm_with_expert.vlm.model.text_model
    text_model.eval()
    text_model.to("cuda")
    input_ids = torch.randint(
        0, 1000, (1, SEQ_LENGTH), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones(
        (1, SEQ_LENGTH), dtype=torch.long, device="cuda"
    )

    results = {
        "eager": _measure(text_model, input_ids, attention_mask, "eager"),
        "sdpa": _measure(text_model, input_ids, attention_mask, "sdpa"),
    }
    print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
