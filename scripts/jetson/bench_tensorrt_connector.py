from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import tensorrt as trt
import torch


def _torch_dtype(trt_dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    return mapping[trt_dtype]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--input-name", default="image_hidden_states")
    parser.add_argument("--output-name", default="connector_features")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with args.engine.open("rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise SystemExit(f"failed to load {args.engine}")
    context = engine.create_execution_context()

    def make_tensor(name):
        shape = tuple(engine.get_tensor_shape(name))
        dtype = _torch_dtype(engine.get_tensor_dtype(name))
        return torch.empty(shape, dtype=dtype, device="cuda")

    input_tensor = make_tensor(args.input_name)
    output_tensor = make_tensor(args.output_name)
    context.set_tensor_address(args.input_name, input_tensor.data_ptr())
    context.set_tensor_address(args.output_name, output_tensor.data_ptr())

    stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def run_once():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    latencies_ms = []
    wall_start = time.perf_counter()
    for _ in range(args.iters):
        start_event.record(stream)
        run_once()
        end_event.record(stream)
        torch.cuda.synchronize()
        latencies_ms.append(start_event.elapsed_time(end_event))
    wall_elapsed = time.perf_counter() - wall_start

    latencies_sorted = sorted(latencies_ms)
    summary = {
        "engine": str(args.engine),
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output_tensor.shape),
        "engine_size_mib": round(args.engine.stat().st_size / (1024 * 1024), 2),
        "streams": 1,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": round(statistics.mean(latencies_ms), 4),
        "latency_median_ms": round(statistics.median(latencies_ms), 4),
        "latency_p95_ms": round(latencies_sorted[int(len(latencies_sorted) * 0.95) - 1], 4),
        "latency_std_ms": round(statistics.pstdev(latencies_ms), 4),
        "throughput_qps": round(args.iters / wall_elapsed, 2),
        "throughput_definition": "sequential_queries_over_synchronized_wall_time",
        "output_finite": bool(torch.isfinite(output_tensor).all().item()),
    }
    print(json.dumps(summary, indent=2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.engine.stem}_latency.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for value in latencies_ms:
            handle.write(json.dumps({"latency_ms": value}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
