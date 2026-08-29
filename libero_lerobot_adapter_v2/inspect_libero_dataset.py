"""Validate one LeRobot/LIBERO sample without optimization-sensitive asserts."""
from __future__ import annotations

import argparse
import json
from typing import Any

REQUIRED_DIMS = {
    "observation.state": 8,
    "observation.images.image": None,
    "observation.images.image2": None,
    "action": 7,
    "task": None,
}


def describe(value: Any) -> dict[str, Any]:
    return {"type": type(value).__name__, "shape": list(value.shape) if hasattr(value, "shape") else None}


def validate_sample(sample: Any) -> dict[str, dict[str, Any]]:
    if not hasattr(sample, "__contains__"):
        raise TypeError("数据样本必须是映射类型")
    missing = [key for key in REQUIRED_DIMS if key not in sample]
    if missing:
        raise ValueError(f"数据缺少 LIBERO 必需字段：{missing}")
    report = {key: describe(sample[key]) for key in REQUIRED_DIMS}
    for key, expected in REQUIRED_DIMS.items():
        if expected is None:
            continue
        shape = report[key]["shape"]
        if not shape or shape[-1] != expected:
            raise ValueError(f"{key} 末维应为 {expected}，实际为 {shape}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 LeRobot LIBERO 数据契约")
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.index < 0:
        parser.error("index 不能为负数")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=args.repo_id, revision=args.revision, video_backend=args.video_backend)
    if args.index >= len(dataset):
        parser.error(f"index 超出范围；数据集共有 {len(dataset)} 条")
    print(json.dumps(validate_sample(dataset[args.index]), ensure_ascii=False, indent=2))
    print("检查通过：双摄像头 + 8 维状态 + 7 维连续动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
