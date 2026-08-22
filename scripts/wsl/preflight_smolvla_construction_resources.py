"""Read checkpoint metadata and construction headroom without loading weights."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


GIB = 1024**3


def _snapshot(path: str, revision: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.name != revision or resolved.parent.name != "snapshots":
        raise ValueError(f"{label} must be snapshots/{revision}")
    return resolved


def _weight_bytes(snapshot: Path) -> dict[str, object]:
    files = sorted(snapshot.rglob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights under {snapshot}")
    return {
        "files": [str(path) for path in files],
        "weight_bytes": sum(path.stat().st_size for path in files),
    }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        match = re.match(r"(MemAvailable|MemTotal|SwapFree|SwapTotal):\s+(\d+) kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def _gpu() -> dict[str, int | str] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    name, total_mib, free_mib = [part.strip() for part in output.splitlines()[0].split(",")]
    return {
        "name": name,
        "total_bytes": int(total_mib) * 1024**2,
        "free_bytes": int(free_mib) * 1024**2,
    }


def _gib(value: int) -> float:
    return round(value / GIB, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--vlm-snapshot", required=True)
    parser.add_argument("--vlm-revision", required=True)
    args = parser.parse_args()

    base = _weight_bytes(_snapshot(args.base_snapshot, args.base_revision, "base_snapshot"))
    vlm = _weight_bytes(_snapshot(args.vlm_snapshot, args.vlm_revision, "vlm_snapshot"))
    combined = int(base["weight_bytes"]) + int(vlm["weight_bytes"])
    memory = _meminfo()
    gpu = _gpu()

    # Conservative planning budgets, not measured allocator peaks: the current
    # loader materializes nested VLM weights and then the outer policy state.
    cpu_peak = 2 * combined + 2 * GIB
    gpu_peak = int(1.25 * combined) + 1 * GIB
    report: dict[str, object] = {
        "base": base,
        "vlm": vlm,
        "combined_weight_bytes": combined,
        "combined_weight_gib": _gib(combined),
        "wsl_memory": memory,
        "wsl_available_plus_swap_bytes": memory["MemAvailable"] + memory["SwapFree"],
        "cpu_construction_conservative_peak_bytes": cpu_peak,
        "cpu_construction_conservative_peak_gib": _gib(cpu_peak),
        "cpu_headroom_after_conservative_peak_bytes": memory["MemAvailable"] + memory["SwapFree"] - cpu_peak,
        "gpu": gpu,
        "gpu_construction_conservative_peak_bytes": gpu_peak,
        "gpu_construction_conservative_peak_gib": _gib(gpu_peak),
        "gpu_headroom_after_conservative_peak_bytes": (
            int(gpu["free_bytes"]) - gpu_peak if gpu else None
        ),
        "estimation_note": "CPU=2x combined safetensors bytes + 2 GiB; GPU=1.25x combined bytes + 1 GiB. These are conservative planning estimates, not measured peaks.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
