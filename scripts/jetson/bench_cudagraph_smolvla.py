from __future__ import annotations

import statistics
import sys
import time

import torch

from libero_platform.policies.smolvla_policy import (
    LeRobotSmolVLARuntime,
    SmolVLAInferenceSpec,
)


CHECKPOINT = "HuggingFaceVLA/smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SEQ_LENGTH = 48
WARMUP = 10
ITERS = 50


def _timed(model, input_ids, attention_mask, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        for _ in range(iters):
            start.record()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    return latencies


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
    text_model = policy.model.vlm_with_expert.vlm.model.text_model
    text_model.eval()
    text_model.to("cuda")
    text_model.config._attn_implementation = "eager"
    text_model.config.attn_implementation = "eager"

    input_ids = torch.randint(
        0, 1000, (1, SEQ_LENGTH), dtype=torch.long, device="cuda"
    )
    attention_mask = torch.ones(
        (1, SEQ_LENGTH), dtype=torch.long, device="cuda"
    )

    eager = _timed(text_model, input_ids, attention_mask, ITERS)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            _ = text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    replay = []
    with torch.no_grad():
        for _ in range(WARMUP):
            graph.replay()
        for _ in range(ITERS):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            replay.append(start.elapsed_time(end))

    eager.sort()
    replay.sort()
    result = {
        "module": "text_model",
        "eager_mean_ms": round(statistics.mean(eager), 3),
        "eager_p95_ms": round(eager[int(len(eager) * 0.95) - 1], 3),
        "graph_mean_ms": round(statistics.mean(replay), 3),
        "graph_p95_ms": round(replay[int(len(replay) * 0.95) - 1], 3),
    }
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
