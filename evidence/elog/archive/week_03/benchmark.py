from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .run_inference import ROOT, run_once


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-key", default="smolvlm2_500m")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = results_dir / f"benchmark_results_{timestamp}.jsonl"
    csv_path = results_dir / f"benchmark_summary_{timestamp}.csv"

    rows = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for image_path in tqdm(images, desc="Benchmark"):
            try:
                result = run_once(
                    image_path=image_path,
                    prompt=args.prompt,
                    model_key=args.model_key,
                    max_new_tokens=args.max_new_tokens,
                )
                result["success"] = True
            except Exception as exc:  # pragma: no cover - experiment logging
                result = {
                    "model_key": args.model_key,
                    "image_path": str(image_path),
                    "prompt": args.prompt,
                    "success": False,
                    "error": str(exc),
                }
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            rows.append(result)

    fieldnames = [
        "success",
        "model_key",
        "model_id",
        "image_path",
        "prompt",
        "latency_sec",
        "peak_gpu_memory_gb",
        "cuda_available",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()

