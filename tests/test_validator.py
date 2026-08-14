from __future__ import annotations

from pathlib import Path

import h5py
import requests
import yaml

from libero_platform.preflight import validate_demo_hdf5
from libero_platform.validator import validate_config
from tests.test_spec import VALID


def test_validate_returns_resolved_spec(catalog_root: Path) -> None:
    report = validate_config(
        catalog_root / "experiments" / "smoke_fake.yaml", catalog_root
    )

    assert report.ok is True
    assert report.resolved_spec is not None
    assert report.resolved_spec.policy_adapter == "zero"


def test_validate_reports_schema_error_for_invalid_yaml(tmp_path: Path, catalog_root: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: nope\n", encoding="utf-8")

    report = validate_config(path, catalog_root)

    assert report.ok is False
    assert report.resolved_spec is None
    assert report.issues[0].failure_type == "config_error"


def test_validate_reports_catalog_error_for_unknown_policy(
    tmp_path: Path, catalog_root: Path
) -> None:
    path = tmp_path / "unknown-policy.yaml"
    payload = {**VALID, "policy": {**VALID["policy"], "key": "unknown"}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = validate_config(path, catalog_root)

    assert report.ok is False
    assert report.issues[0].failure_type == "preflight_error"


def test_fake_backend_does_not_require_libero(
    catalog_root: Path, monkeypatch
) -> None:
    def unavailable(_: str) -> None:
        raise AssertionError("fake validation must not check for LIBERO")

    monkeypatch.setattr("libero_platform.validator.importlib.util.find_spec", unavailable)

    report = validate_config(
        catalog_root / "experiments" / "smoke_fake.yaml", catalog_root
    )

    assert report.ok is True


def test_http_endpoint_must_be_http_url(tmp_path: Path, catalog_root: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    for source in catalog_root.glob("*.yaml"):
        (config_root / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / "remote.yaml"
    payload = {
        **VALID,
        "policy": {
            "key": "remote_http_policy",
            "model_key": "smolvla_libero",
            "checkpoint": "catalog:default",
            "precision": "fp16",
            "quantization": "none",
        },
        "deployment": {
            "mode": "remote_server",
            "profile": "remote_server_default",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    deployments = yaml.safe_load(
        (config_root / "deployment_profiles.yaml").read_text(encoding="utf-8")
    )
    deployments["deployments"][-1]["endpoint"] = "not-a-url"
    (config_root / "deployment_profiles.yaml").write_text(
        yaml.safe_dump(deployments, sort_keys=False), encoding="utf-8"
    )

    report = validate_config(path, config_root)

    assert report.ok is False
    assert report.issues[0].failure_type == "preflight_error"


def test_libero_dataset_requires_expected_task_filename(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    for source in catalog_root.glob("*.yaml"):
        (config_root / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    dataset_directory = tmp_path / "datasets" / "libero_spatial"
    dataset_directory.mkdir(parents=True)
    with h5py.File(dataset_directory / "wrong_name_demo.hdf5", "w") as dataset:
        dataset.attrs["task_name"] = "wrong_name"
    path = tmp_path / "libero.yaml"
    payload = {**VALID, "benchmark": {**VALID["benchmark"], "backend": "libero"}}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "libero_platform.validator.importlib.util.find_spec", lambda _: object()
    )

    report = validate_config(
        path,
        config_root,
        task_name_resolver=lambda _suite, _task_id: "expected_task",
    )

    assert report.ok is False
    assert report.issues[0].failure_type == "preflight_error"


def test_network_preflight_classifies_unavailable_service(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = tmp_path / "remote.yaml"
    payload = {
        **VALID,
        "policy": {
            "key": "remote_http_policy",
            "model_key": "smolvla_libero",
            "checkpoint": "catalog:default",
            "precision": "fp16",
            "quantization": "none",
        },
        "deployment": {
            "mode": "remote_server",
            "profile": "remote_server_default",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    def unavailable(*_args, **_kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get", unavailable
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is False
    assert report.issues[0].failure_type == "service_unavailable"
    assert "connection refused" in report.issues[0].message


def test_network_preflight_rejects_wrong_checkpoint(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path)
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _HealthResponse(
            {
                "schema_version": 1,
                "status": "ok",
                "policy": {
                    "checkpoint": "wrong/checkpoint",
                    "revision": "expected-revision",
                    "precision": "fp16",
                    "ready": True,
                },
            }
        ),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is False
    assert report.issues[0].failure_type == "policy_identity_mismatch"
    assert "checkpoint mismatch" in report.issues[0].message
    assert "expected 'HuggingFaceVLA/smolvla_libero'" in report.issues[0].message
    assert "observed 'wrong/checkpoint'" in report.issues[0].message


def test_network_preflight_rejects_wrong_model_key(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path)
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _HealthResponse(
            {
                "schema_version": 1,
                "status": "ok",
                "policy": {
                    "model_key": "different_model",
                    "checkpoint": "HuggingFaceVLA/smolvla_libero",
                    "precision": "fp16",
                    "ready": True,
                },
            }
        ),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is False
    assert report.issues[0].failure_type == "policy_identity_mismatch"
    assert "model_key mismatch" in report.issues[0].message
    assert "expected 'smolvla_libero'" in report.issues[0].message
    assert "observed 'different_model'" in report.issues[0].message


def test_network_preflight_rejects_revision_and_precision_mismatches(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path, revision="expected-revision")
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _HealthResponse(
            {
                "schema_version": 1,
                "status": "ok",
                "policy": {
                    "checkpoint": "HuggingFaceVLA/smolvla_libero",
                    "revision": "other-revision",
                    "precision": "bf16",
                    "ready": True,
                },
            }
        ),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is False
    assert report.issues[0].failure_type == "policy_identity_mismatch"
    assert "revision mismatch" in report.issues[0].message
    assert "precision mismatch" in report.issues[0].message


def test_network_preflight_rejects_loopback_for_jetson_remote_client(
    tmp_path: Path, catalog_root: Path
) -> None:
    path = _remote_policy_config(tmp_path, profile="jetson_remote_client_default")

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is False
    assert report.issues[0].failure_type == "preflight_error"
    assert "non-loopback" in report.issues[0].message


def test_network_preflight_allows_explicit_loopback_test_and_probes_service(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path, profile="jetson_remote_client_default")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["deployment"]["allow_loopback_endpoint"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _ready_health(),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is True


def test_network_preflight_accepts_matching_remote_identity(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path, revision="expected-revision")
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _ready_health(revision="expected-revision"),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is True
    assert report.resolved_spec is not None
    assert report.resolved_spec.policy.revision == "expected-revision"
    assert report.resolved_spec.resolved_revision == "expected-revision"


def test_validate_expands_revision_environment_placeholder_before_health_check(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path, revision="${MODEL_REVISION}")
    monkeypatch.setenv("MODEL_REVISION", "pinned-commit")
    monkeypatch.setattr(
        "libero_platform.policies.remote_http.requests.get",
        lambda *_args, **_kwargs: _ready_health(revision="pinned-commit"),
    )

    report = validate_config(path, catalog_root, check_network=True)

    assert report.ok is True
    assert report.resolved_spec is not None
    assert report.resolved_spec.policy.revision == "pinned-commit"
    assert report.resolved_spec.resolved_revision == "pinned-commit"


def test_validate_reports_missing_revision_environment_variable(
    tmp_path: Path, catalog_root: Path, monkeypatch
) -> None:
    path = _remote_policy_config(tmp_path, revision="${MODEL_REVISION}")
    monkeypatch.delenv("MODEL_REVISION", raising=False)

    report = validate_config(path, catalog_root)

    assert report.ok is False
    assert report.issues[0].failure_type == "config_error"
    assert "MODEL_REVISION" in report.issues[0].message
    assert "not set" in report.issues[0].message


class _HealthResponse:
    status_code = 200

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def _ready_health(*, revision: str | None = None) -> _HealthResponse:
    return _HealthResponse(
        {
            "schema_version": 1,
            "status": "ok",
            "policy": {
                "model_key": "smolvla_libero",
                "checkpoint": "HuggingFaceVLA/smolvla_libero",
                "revision": revision,
                "precision": "fp16",
                "ready": True,
            },
        }
    )


def _remote_policy_config(
    tmp_path: Path,
    *,
    profile: str = "jetson_remote_client_direct",
    revision: str | None = None,
) -> Path:
    payload = {
        **VALID,
        "policy": {
            "key": "remote_http_policy",
            "model_key": "smolvla_libero",
            "checkpoint": "HuggingFaceVLA/smolvla_libero",
            "precision": "fp16",
            "quantization": "none",
        },
        "deployment": {"mode": "jetson_remote_client", "profile": profile},
    }
    if revision is not None:
        payload["policy"]["revision"] = revision
    path = tmp_path / "remote.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
