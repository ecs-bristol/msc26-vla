#!/usr/bin/env python
"""Microbenchmark for SmolVLA per-step inference latency and memory.

Measures policy-only latency (`select_action` with n_action_steps=1, i.e. a full
flow-matching denoising pass per call), peak CUDA allocation, and parameter
memory for fp16 and 4-bit variants. Inputs are a fixed synthetic LIBERO batch,
so the benchmark isolates model compute (same convention as the platform's
`inference_ms`).

Usage:
  python scripts/wsl/bench_smolvla_latency.py --quant-method none --num-steps 10
  python scripts/wsl/bench_smolvla_latency.py --quant-method torchao_int4 --num-steps 10
  python scripts/wsl/bench_smolvla_latency.py --quant-method torchao_int4 --probe
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from lerobot_policy_smolvla_int4.configuration_smolvla_int4 import SmolVLAInt4Config
from lerobot_policy_smolvla_int4.modeling_smolvla_int4 import SmolVLAInt4Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parser.add_argument("--revision", default="6721902bc4d61e50a3bfdb11dfb4cb626f05d102")
    parser.add_argument(
        "--quant-method",
        choices=("none", "int4_groupwise", "int8_groupwise", "bnb_nf4", "mixed"),
        default="none",
    )
    parser.add_argument(
        "--quant-scope",
        choices=("language", "backbone"),
        default="language",
    )
    parser.add_argument("--vision-bits", type=int, default=None, choices=(4, 8, 16))
    parser.add_argument("--connector-bits", type=int, default=None, choices=(4, 8, 16))
    parser.add_argument("--text-bits", type=int, default=None, choices=(4, 8, 16))
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="load the policy and run one forward pass, then exit (phase-0 check)",
    )
    return parser.parse_args()


def _observation_batch(device: str, token_len: int = 48) -> dict[str, torch.Tensor]:
    return {
        "observation.images.image": torch.zeros(1, 3, 256, 256, device=device),
        "observation.images.image2": torch.zeros(1, 3, 256, 256, device=device),
        "observation.state": torch.zeros(1, 8, device=device),
        "observation.language.tokens": torch.randint(0, 1000, (1, token_len), device=device),
        "observation.language.attention_mask": torch.ones(
            1, token_len, dtype=torch.bool, device=device
        ),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; timings on CPU are not meaningful", flush=True)

    config_kwargs = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "quant_method": args.quant_method,
        "quant_scope": args.quant_scope,
        "num_steps": args.num_steps,
        "n_action_steps": args.n_action_steps,
        "device": device,
    }
    if args.vision_bits is not None:
        config_kwargs["vision_bits"] = args.vision_bits
    if args.connector_bits is not None:
        config_kwargs["connector_bits"] = args.connector_bits
    if args.text_bits is not None:
        config_kwargs["text_bits"] = args.text_bits
    config = SmolVLAInt4Config(**config_kwargs)
    print(
        f"Loading {config.checkpoint} rev={config.revision} "
        f"quant={args.quant_method} num_steps={args.num_steps} ...",
        flush=True,
    )
    policy = SmolVLAInt4Policy(config)
    policy.to(device)
    policy.eval()

    param_bytes = sum(p.numel() * p.element_size() for p in policy.parameters())
    print(f"Loaded on {device}; use_amp={policy.config.use_amp}", flush=True)
    print(f"Parameter memory estimate: {param_bytes / 1e6:.1f} MB", flush=True)

    batch = _observation_batch(device)
    with torch.inference_mode():
        policy.select_action(batch)

    if args.probe:
        print("PROBE OK: policy loaded and one forward pass completed", flush=True)
        return

    torch.manual_seed(args.seed)
    for _ in range(args.warmup):
        with torch.inference_mode():
            policy.select_action(batch)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    latencies_ms: list[float] = []
    for _ in range(args.iters):
        start = time.perf_counter()
        with torch.inference_mode():
            policy.select_action(batch)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    result = {
        "quant_method": args.quant_method,
        "quant_scope": args.quant_scope,
        "num_steps": args.num_steps,
        "n_action_steps": args.n_action_steps,
        "device": device,
        "iters": args.iters,
        "mean_ms": round(statistics.mean(latencies_ms), 2),
        "p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95) - 1], 2),
        "peak_allocated_mb": peak_mb,
        "param_bytes_mb": round(param_bytes / 1e6, 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
