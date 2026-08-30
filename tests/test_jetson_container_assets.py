from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_jetson_container_assets_encode_the_reproducible_runtime_contract() -> None:
    dockerfile = (PROJECT_ROOT / "docker" / "jetson" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    launcher = (PROJECT_ROOT / "scripts" / "jetson" / "run_container.sh").read_text(
        encoding="utf-8"
    )

    for required in (
        "FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu",
        '"transformers>=5.4,<5.6"',
        '"accelerate>=1.14,<2"',
        '"num2words>=0.5.14,<0.6"',
        'python3 -m pip install --no-cache-dir --no-deps "lerobot==0.6.1"',
        'assert ".nv25." in torch.__version__',
        'assert torchvision.__version__.startswith("0.22.0a0+")',
        "numpy>=2.0,<2.3",
        "--force-reinstall",
        "numpy>=1.26,<2",
        "ENV PYTHONPATH=/workspace/project/src",
        "pytest>=8.3,<9",
    ):
        assert required in dockerfile

    assert '"lerobot[smolvla]>=0.6.1,<0.7.0"' not in dockerfile
    assert 'assert ".nv25.6" in torch.__version__' not in dockerfile

    for required in (
        "--runtime nvidia",
        "--network host",
        "--ipc host",
        "--shm-size=1g",
        'HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"',
        'TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"',
        '"HF_HUB_OFFLINE=$HF_HUB_OFFLINE"',
        '"TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"',
        "if [[ -t 0 && -t 1 ]]; then",
        "TTY_ARGS=(-it)",
        '"${TTY_ARGS[@]}"',
        '"$PROJECT_DIR:/workspace/project:ro"',
        '"$HF_CACHE_DIR:/root/.cache/huggingface"',
        '"$OUTPUT_DIR:/workspace/outputs"',
    ):
        assert required in launcher


def test_jetson_container_launcher_uses_linux_line_endings() -> None:
    launcher = PROJECT_ROOT / "scripts" / "jetson" / "run_container.sh"
    attributes_path = PROJECT_ROOT / ".gitattributes"

    assert attributes_path.exists()
    assert "*.sh text eol=lf" in attributes_path.read_text(encoding="utf-8")
    assert b"\r\n" not in launcher.read_bytes()


def test_jetson_service_launcher_requires_a_revision_and_supports_modes() -> None:
    launcher = (
        PROJECT_ROOT / "scripts" / "jetson" / "start_smolvla_libero_service.sh"
    )

    assert launcher.exists()
    content = launcher.read_text(encoding="utf-8")

    for required in (
        'MODE="${1:-}"',
        'CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"',
        'MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"',
        "export CHECKPOINT MODEL_REVISION",
        'bootstrap)',
        'offline)',
        'HF_HUB_OFFLINE=1',
        'TRANSFORMERS_OFFLINE=1',
        'serve-policy',
        '--policy smolvla_libero',
        '"$CHECKPOINT"',
        '"$MODEL_REVISION"',
        '--precision fp16',
        '--host 0.0.0.0',
        '--port 8081',
        'exec "$(dirname "$0")/run_container.sh"',
    ):
        assert required in content


def test_jetson_service_launcher_uses_linux_line_endings() -> None:
    launcher = (
        PROJECT_ROOT / "scripts" / "jetson" / "start_smolvla_libero_service.sh"
    )

    assert b"\r\n" not in launcher.read_bytes()
