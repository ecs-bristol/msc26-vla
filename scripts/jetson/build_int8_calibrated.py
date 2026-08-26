from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


ONNX_PATH = Path("/workspace/outputs/smolvla_vision.onnx")
CALIB_DIR = Path("/workspace/outputs/vision_calib")
ENGINE_PATH = Path("/workspace/outputs/smolvla_vision_int8_calib.engine")
CALIB_CACHE = ENGINE_PATH.with_suffix(".calib")
BATCH_SIZE = 1
WORKSPACE_MIB = 128


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, files: list[Path], batch_size: int = BATCH_SIZE) -> None:
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.index = 0
        self.device_input = None

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.index >= len(self.files):
            return None
        tensor = torch.load(self.files[self.index], map_location="cpu")
        array = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
        if self.device_input is None or self.device_input.shape != tuple(array.shape):
            self.device_input = torch.empty(
                tuple(array.shape), dtype=torch.float32, device="cuda"
            )
        self.device_input.copy_(torch.from_numpy(array))
        self.index += 1
        return [self.device_input.data_ptr()]

    def read_calibration_cache(self) -> memoryview | None:
        if CALIB_CACHE.exists():
            return CALIB_CACHE.read_bytes()
        return None

    def write_calibration_cache(self, cache: memoryview) -> None:
        CALIB_CACHE.write_bytes(bytes(cache))


def main() -> int:
    os.chdir(ONNX_PATH.parent)
    files = sorted(glob.glob(str(CALIB_DIR / "*.pt")))[:400]
    if not files:
        raise SystemExit(f"no calibration tensors in {CALIB_DIR}")
    print(f"calibration tensors: {len(files)}", flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with ONNX_PATH.open("rb") as handle:
        if not parser.parse(handle.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            return 2

    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, WORKSPACE_MIB * 1024 * 1024
    )
    config.int8_calibrator = TorchEntropyCalibrator([Path(p) for p in files])

    print("building calibrated INT8 engine...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("engine build failed", file=sys.stderr)
        return 3
    data = bytes(serialized)
    ENGINE_PATH.write_bytes(data)
    print(f"wrote {ENGINE_PATH} ({len(data) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
