from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from libero_platform.preflight import ValidationReport
from libero_platform.cli import build_parser, main
from libero_platform.validator import validate_config as real_validate_config


def test_cli_validate_returns_two_for_bad_yaml(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: nope\n", encoding="utf-8")

    assert main(["validate", str(path)]) == 2
    assert "config_error" in capsys.readouterr().err


def test_cli_validate_uses_libero_resolvers_for_runtime_preflight(
    monkeypatch, catalog_root: Path, capsys
) -> None:
    baseline = real_validate_config(
        catalog_root / "experiments" / "smoke_fake.yaml",
        catalog_root,
        runtime_checks=False,
    )
    assert baseline.resolved_spec is not None
    resolved = baseline.resolved_spec.model_copy(
        update={
            "benchmark": baseline.resolved_spec.benchmark.model_copy(
                update={"backend": "libero"}
            )
        }
    )
    calls = []

    def fake_validate_config(path, config_root, **kwargs):
        del path, config_root
        calls.append(kwargs)
        return ValidationReport(ok=True, resolved_spec=resolved)

    class Backend:
        def list_tasks(self, suite):
            assert suite == "libero_spatial"
            return [{"task_id": 0, "task_name": "task_zero"}]

        def _read_initial_state(self, task_name, initial_state_id):
            assert task_name == "task_zero"
            assert initial_state_id == 0
            return object()

    monkeypatch.setattr("libero_platform.cli.validate_config", fake_validate_config)
    monkeypatch.setattr("libero_platform.cli._build_backend", lambda spec: Backend())

    assert main(["validate", str(catalog_root / "experiments" / "smoke_fake.yaml")], config_root=catalog_root) == 0
    assert calls[0] == {"runtime_checks": False}
    assert calls[1]["check_network"] is True
    assert calls[1]["task_name_resolver"]("libero_spatial", 0) == "task_zero"
    assert calls[1]["initial_state_resolver"]("libero_spatial", 0, 0) is not None
    assert "valid: fake_smoke" in capsys.readouterr().out


def test_cli_run_returns_two_for_invalid_config_without_echoing_values(
    tmp_path: Path, catalog_root: Path, capsys
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: secret-value\n", encoding="utf-8")

    assert main(["run", str(path)], config_root=catalog_root) == 2
    output = capsys.readouterr()
    assert "config_error" in output.err
    assert "secret-value" not in output.err


def test_cli_list_policies_is_read_only(capsys) -> None:
    assert main(["list", "policies"]) == 0
    assert "smolvla_libero" in capsys.readouterr().out


def test_cli_list_hides_catalog_path_details(tmp_path: Path, capsys) -> None:
    secret_root = tmp_path / "catalog-secret-path"

    assert main(["list", "suites"], config_root=secret_root) == 3
    output = capsys.readouterr()
    assert "catalog: unavailable" in output.err
    assert "catalog-secret-path" not in output.err


def test_cli_lists_tasks_for_selected_suite(capsys) -> None:
    assert main(["list", "tasks", "--suite", "libero_spatial"]) == 0
    assert "0" in capsys.readouterr().out


def test_cli_run_executes_fake_config(
    tmp_path: Path, catalog_root: Path, capsys
) -> None:
    exit_code = main(
        ["run", str(catalog_root / "experiments" / "smoke_fake.yaml")],
        config_root=catalog_root,
        output_root=tmp_path / "outputs",
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Episode: 1 / 1" in output


def test_cli_run_records_the_checked_out_git_commit(
    tmp_path: Path, catalog_root: Path
) -> None:
    output_root = tmp_path / "outputs"
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=catalog_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    exit_code = main(
        ["run", str(catalog_root / "experiments" / "smoke_fake.yaml")],
        config_root=catalog_root,
        output_root=output_root,
    )

    assert exit_code == 0
    run_dir = next(path for path in output_root.iterdir() if path.is_dir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"] == expected_commit


def test_cli_run_maps_stopped_outcome_to_five(
    tmp_path: Path, catalog_root: Path, capsys
) -> None:
    class StoppedOutcome:
        status = "stopped"
        result_integrity = "partial"

    def stopped_run(spec, dependencies):
        del spec, dependencies
        return StoppedOutcome()

    exit_code = main(
        ["run", str(catalog_root / "experiments" / "smoke_fake.yaml")],
        config_root=catalog_root,
        output_root=tmp_path / "outputs",
        run_callable=stopped_run,
    )

    assert exit_code == 5
    assert "Status: stopped" in capsys.readouterr().out




def test_cli_routes_catalog_adapters_to_controlled_unavailable_outcomes(
    tmp_path: Path, catalog_root: Path, capsys
) -> None:
    for policy_key, precision, quantization in (
        ("oracle_or_scripted", "none", "none"),
        ("smolvla_libero", "fp16", "none"),
    ):
        path = _write_config(
            tmp_path,
            catalog_root,
            policy_key=policy_key,
            precision=precision,
            quantization=quantization,
        )

        assert main(["run", str(path)], config_root=catalog_root) == 4
        output = capsys.readouterr()
        assert "Failure: policy_adapter_unavailable" in output.err
        assert "unsupported policy adapter" not in output.err


def test_cli_run_hides_exception_text_from_terminal_output(
    tmp_path: Path, catalog_root: Path, capsys
) -> None:
    secret = "super-secret-token-value"

    def failing_run(spec, dependencies):
        del spec, dependencies
        raise RuntimeError(secret)

    exit_code = main(
        ["run", str(catalog_root / "experiments" / "smoke_fake.yaml")],
        config_root=catalog_root,
        output_root=tmp_path / "outputs",
        run_callable=failing_run,
    )

    assert exit_code == 4
    output = capsys.readouterr()
    assert "Status: failed" in output.out
    assert "Failure: execution_failed" in output.err
    assert "log:" in output.err
    assert secret not in output.out + output.err


def test_cli_uses_keyword_only_config_root(catalog_root: Path, capsys) -> None:
    assert main(["list", "suites"], config_root=catalog_root) == 0
    assert "libero_goal" in capsys.readouterr().out


def test_serve_policy_accepts_smolvla_runtime_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve-policy",
            "--policy",
            "smolvla_libero",
            "--checkpoint",
            "lerobot/smolvla_libero",
            "--precision",
            "fp16",
            "--revision",
            "0123456789abcdef",
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
        ]
    )

    assert args.policy == "smolvla_libero"
    assert args.checkpoint == "lerobot/smolvla_libero"
    assert args.precision == "fp16"
    assert args.revision == "0123456789abcdef"


def test_policy_parity_requires_an_explicit_model_revision() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["policy-parity"])


def test_cli_routes_policy_parity(monkeypatch) -> None:
    recorded = {}

    def fake_policy_parity(args):
        recorded["args"] = args
        return 0

    monkeypatch.setattr("libero_platform.cli._policy_parity", fake_policy_parity)

    assert (
        main(
            [
                "policy-parity",
                "--revision",
                "0123456789abcdef",
                "--endpoint",
                "http://10.42.0.2:8081",
            ]
        )
        == 0
    )
    assert recorded["args"].revision == "0123456789abcdef"
    assert recorded["args"].endpoint == "http://10.42.0.2:8081"


def test_policy_parity_accepts_repeat_check_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "policy-parity",
            "--revision",
            "0123456789abcdef",
            "--repeat-check",
        ]
    )

    assert args.repeat_check is True


def test_serve_policy_builds_loaded_smolvla_adapter(monkeypatch) -> None:
    recorded = {}

    class Adapter:
        def load(self):
            recorded["loaded"] = True

        def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(
        "libero_platform.policies.smolvla_policy.SmolVLAPolicyAdapter",
        lambda **kwargs: Adapter(),
    )

    assert (
        main(
            ["serve-policy", "--policy", "smolvla_libero", "--host", "127.0.0.1"],
            serve_callable=lambda adapter, host, port: recorded.update(
                adapter=adapter, host=host, port=port
            ),
        )
        == 0
    )
    assert recorded["loaded"] is True
    assert recorded["closed"] is True
    assert recorded["host"] == "127.0.0.1"


def test_serve_policy_passes_revision_to_smolvla_adapter(monkeypatch) -> None:
    from libero_platform import cli

    recorded = {}

    class Adapter:
        pass

    monkeypatch.setattr(
        "libero_platform.policies.smolvla_policy.SmolVLAPolicyAdapter",
        lambda **kwargs: recorded.setdefault("kwargs", kwargs) or Adapter(),
    )

    cli._build_service_policy(
        "smolvla_libero",
        "HuggingFaceVLA/smolvla_libero",
        "fp16",
        "0123456789abcdef",
    )

    assert recorded["kwargs"]["revision"] == "0123456789abcdef"


def test_build_policy_passes_smolvla_controls_to_adapter(monkeypatch) -> None:
    from libero_platform import cli

    recorded = {}
    action_control = object()
    smolvla_inference = object()

    class Adapter:
        pass

    monkeypatch.setattr(
        "libero_platform.policies.smolvla_policy.SmolVLAPolicyAdapter",
        lambda **kwargs: recorded.setdefault("kwargs", kwargs) or Adapter(),
    )

    spec = type(
        "Spec",
        (),
        {
            "policy_adapter": "smolvla",
            "policy": type(
                "Policy",
                (),
                {
                    "key": "smolvla_libero",
                    "precision": "fp16",
                    "action_control": action_control,
                    "smolvla_inference": smolvla_inference,
                },
            )(),
            "resolved_checkpoint": "lerobot/smolvla_libero",
            "resolved_revision": "pinned-model-revision",
        },
    )()

    cli._build_policy(spec)

    assert recorded["kwargs"]["action_control"] is action_control
    assert recorded["kwargs"]["smolvla_inference"] is smolvla_inference
    assert recorded["kwargs"]["revision"] == "pinned-model-revision"


def _write_config(
    tmp_path: Path,
    catalog_root: Path,
    *,
    backend: str = "fake",
    policy_key: str = "zero_policy",
    precision: str = "none",
    quantization: str = "none",
) -> Path:
    payload = yaml.safe_load(
        (catalog_root / "experiments" / "smoke_fake.yaml").read_text(encoding="utf-8")
    )
    payload["benchmark"]["backend"] = backend
    payload["policy"].update(
        key=policy_key,
        precision=precision,
        quantization=quantization,
    )
    path = tmp_path / f"{backend}_{policy_key}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path
