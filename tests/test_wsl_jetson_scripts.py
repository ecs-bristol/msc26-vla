from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wsl_preflight_uses_official_plugin_and_health_identity() -> None:
    launcher = PROJECT_ROOT / "scripts" / "wsl" / "run_jetson_remote_preflight.sh"

    assert launcher.exists()
    content = launcher.read_text(encoding="utf-8")

    for required in (
        'JETSON_ENDPOINT="${JETSON_ENDPOINT:?JETSON_ENDPOINT must be set}"',
        'MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION must be set to a concrete revision}"',
        'CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"',
        "register_third_party_plugins()",
        'get_policy_class("remote_jetson")',
        '/health',
        'policy.get("checkpoint")',
        'policy.get("revision")',
        'policy.get("precision")',
        'except requests.RequestException as error',
    ):
        assert required in content

    assert "libero_platform" not in content
    assert "scp " not in content


def test_official_eval_launcher_owns_environment_rollout_and_scoring() -> None:
    launcher = PROJECT_ROOT / "scripts" / "wsl" / "run_official_jetson_remote_eval.sh"
    content = launcher.read_text(encoding="utf-8")

    for required in (
        "exec lerobot-eval",
        "--policy.type=remote_jetson",
        "--policy.precision=fp16",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.episode_length=280",
        "--eval.n_episodes=\"$N_EPISODES\"",
        "--eval.batch_size=1",
        "--env.max_parallel_tasks=1",
        "--output_dir=\"$OUTPUT_DIR\"",
    ):
        assert required in content

    assert "python -m libero_platform run" not in content


def test_official_pc_local_launcher_uses_lerobot_eval() -> None:
    launcher = PROJECT_ROOT / "scripts" / "wsl" / "run_official_pc_local_eval.sh"
    content = launcher.read_text(encoding="utf-8")

    for required in (
        "lerobot-eval",
        '"--policy.path=$CHECKPOINT"',
        '"--policy.pretrained_revision=$MODEL_REVISION"',
        '"--policy.num_steps=$NUM_STEPS"',
        "--env.type=libero",
        '"--env.task=$SUITE"',
        '"--env.episode_length=$EPISODE_LENGTH"',
        '"--eval.n_episodes=$N_EPISODES"',
        "--eval.batch_size=1",
        "--env.max_parallel_tasks=1",
        '"--seed=$EVAL_SEED"',
        '"--output_dir=$OUTPUT_DIR"',
        "capture_official_eval_provenance.py",
        "paired_seed_manifest.json",
        "resolved_config.log",
    ):
        assert required in content

    assert "libero_platform run" not in content


def test_plugin_install_launcher_uses_editable_no_dependency_install() -> None:
    launcher = PROJECT_ROOT / "scripts" / "wsl" / "install_remote_jetson_policy.sh"
    content = launcher.read_text(encoding="utf-8")

    assert "python -m pip install --no-deps --editable" in content
    assert "register_third_party_plugins()" in content
    assert 'get_policy_class("remote_jetson")' in content


def test_wsl_preflight_launcher_uses_linux_line_endings() -> None:
    launchers = (
        PROJECT_ROOT / "scripts" / "wsl" / "install_remote_jetson_policy.sh",
        PROJECT_ROOT / "scripts" / "wsl" / "run_jetson_remote_preflight.sh",
        PROJECT_ROOT / "scripts" / "wsl" / "run_official_jetson_remote_eval.sh",
        PROJECT_ROOT / "scripts" / "wsl" / "run_official_pc_local_eval.sh",
    )
    assert all(b"\r\n" not in launcher.read_bytes() for launcher in launchers)


def test_active_readmes_do_not_run_the_legacy_benchmark_runner() -> None:
    active_docs = (
        PROJECT_ROOT / "README_ZH.md",
        PROJECT_ROOT / "docs" / "PC_LOCAL_SMOLVLA_PAPER_ALIGNMENT.md",
        PROJECT_ROOT / "docs" / "OFFICIAL_LEROBOT_JETSON_REMOTE.md",
    )

    for document in active_docs:
        content = document.read_text(encoding="utf-8")
        assert "python -m libero_platform run" not in content
