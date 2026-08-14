from __future__ import annotations

import platform
from pathlib import Path

import psutil


def collect_device_profile() -> dict[str, str | float | None]:
    """Return portable host metadata without inventing unavailable device metrics."""
    return {
        "device_model": _device_model(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "jetpack_version": None,
        "cuda_version": None,
        "runtime_version": None,
        "power_mode": None,
        "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "temperature_peak_c": None,
        "power_average_w": None,
        "power_peak_w": None,
        "metric_unavailable_reason": "Jetson metrics unavailable on this host",
    }


def _device_model() -> str | None:
    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(encoding="utf-8").rstrip("\x00").strip()
    except OSError:
        return None
    return model or None
