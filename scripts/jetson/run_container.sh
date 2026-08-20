#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${JETSON_IMAGE:-libero-smolvla:jetson-0.1}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/vla/project}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/vla/hf-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/vla/outputs}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
  printf 'Project directory is missing pyproject.toml: %s\n' "$PROJECT_DIR" >&2
  exit 2
fi

mkdir -p "$HF_CACHE_DIR" "$OUTPUT_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  printf 'Jetson image is unavailable: %s\nBuild it from %s/docker/jetson/Dockerfile first.\n' "$IMAGE" "$PROJECT_DIR" >&2
  exit 3
fi

if [[ "$#" -eq 0 ]]; then
  set -- bash
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

exec docker run --rm --init \
  "${TTY_ARGS[@]}" \
  --runtime nvidia \
  --network host \
  --ipc host \
  --shm-size=1g \
  -e "CHECKPOINT=${CHECKPOINT:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
  -e "MODEL_REVISION=${MODEL_REVISION:-}" \
  -e PYTHONPATH=/workspace/project/src \
  -e "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE" \
  -v "$PROJECT_DIR:/workspace/project:ro" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$OUTPUT_DIR:/workspace/outputs" \
  -w /workspace/project \
  "$IMAGE" "$@"
