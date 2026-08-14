from __future__ import annotations

import importlib.util
import ipaddress
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import yaml
from pydantic import ValidationError

from .catalog import Catalog, CatalogError
from .policies.remote_http import RemotePolicyUnavailable, probe_remote_policy
from .preflight import ValidationIssue, ValidationReport, validate_demo_hdf5
from .spec import ResolvedExperimentSpec, load_experiment_spec

InitialStateResolver = Callable[[str, int, int], object]


class RemotePolicyIdentityMismatch(ValueError):
    """Raised when a ready remote service has a different policy identity."""


def validate_config(
    path: Path,
    config_root: Path,
    *,
    check_network: bool = False,
    task_name_resolver: Callable[[str, int], str] | None = None,
    initial_state_resolver: InitialStateResolver | None = None,
    runtime_checks: bool = True,
) -> ValidationReport:
    """Resolve an experiment YAML and run preflight checks without side effects."""
    try:
        source_path = path.resolve(strict=True)
        spec = load_experiment_spec(source_path)
    except (OSError, ValueError, yaml.YAMLError, ValidationError) as exc:
        return _failure("config", str(exc), "config_error")

    try:
        resolved = Catalog.load(config_root).resolve(spec, source_path)
        if runtime_checks:
            _validate_backend(
                resolved,
                task_name_resolver=task_name_resolver,
                initial_state_resolver=initial_state_resolver,
            )
            _validate_policy_dependency(resolved)
            _validate_endpoint(resolved, check_network=check_network)
    except RemotePolicyUnavailable as exc:
        return _failure("network", str(exc), "service_unavailable")
    except RemotePolicyIdentityMismatch as exc:
        return _failure("policy", str(exc), "policy_identity_mismatch")
    except (CatalogError, OSError, RuntimeError, ValueError) as exc:
        return _failure("preflight", str(exc), "preflight_error")

    return ValidationReport(ok=True, resolved_spec=resolved)


def _validate_backend(
    resolved: ResolvedExperimentSpec,
    *,
    task_name_resolver: Callable[[str, int], str] | None,
    initial_state_resolver: InitialStateResolver | None,
) -> None:
    if resolved.benchmark.backend == "fake":
        return
    if importlib.util.find_spec("libero") is None:
        raise RuntimeError("LIBERO dependency is not installed")
    _validate_libero_dataset(
        resolved,
        task_name_resolver=task_name_resolver,
        initial_state_resolver=initial_state_resolver,
    )


def _validate_policy_dependency(resolved: ResolvedExperimentSpec) -> None:
    if resolved.policy_adapter == "smolvla" and importlib.util.find_spec("lerobot") is None:
        raise RuntimeError("LeRobot dependency is not installed")


def _validate_libero_dataset(
    resolved: ResolvedExperimentSpec,
    *,
    task_name_resolver: Callable[[str, int], str] | None,
    initial_state_resolver: InitialStateResolver | None,
) -> None:
    if resolved.benchmark.initial_state_source == "benchmark":
        for task_id in resolved.benchmark.task_ids:
            _resolve_task_name(resolved.benchmark.suite, task_id, task_name_resolver)
            for initial_state_id in resolved.benchmark.initial_state_ids:
                _resolve_initial_state(
                    resolved.benchmark.suite,
                    task_id,
                    initial_state_id,
                    initial_state_resolver,
                )
        return

    dataset_directory = Path(resolved.dataset_directory)
    if not dataset_directory.is_dir():
        raise RuntimeError(f"LIBERO suite dataset directory is missing: {dataset_directory}")
    if not any(dataset_directory.iterdir()):
        raise RuntimeError(f"LIBERO suite dataset directory is empty: {dataset_directory}")

    for task_id in resolved.benchmark.task_ids:
        task_name = _resolve_task_name(
            resolved.benchmark.suite, task_id, task_name_resolver
        )
        demo_file = dataset_directory / f"{task_name}_demo.hdf5"
        if not demo_file.is_file():
            raise RuntimeError(
                f"LIBERO dataset is missing demonstration file: {demo_file.name}"
            )
        for initial_state_id in resolved.benchmark.initial_state_ids:
            backend_initial_state = _resolve_initial_state(
                resolved.benchmark.suite,
                task_id,
                initial_state_id,
                initial_state_resolver,
            )
            validate_demo_hdf5(
                demo_file,
                initial_state_id=initial_state_id,
                backend_initial_state=backend_initial_state,
            )


def _resolve_task_name(
    suite: str,
    task_id: int,
    task_name_resolver: Callable[[str, int], str] | None,
) -> str:
    if task_name_resolver is None:
        raise RuntimeError("LIBERO task name resolver is required for dataset preflight")
    try:
        task_name = task_name_resolver(suite, task_id)
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve LIBERO task name for {suite} task {task_id}"
        ) from exc
    if not isinstance(task_name, str) or not task_name or Path(task_name).name != task_name:
        raise RuntimeError(f"invalid LIBERO task name for {suite} task {task_id}")
    return task_name


def _resolve_initial_state(
    suite: str,
    task_id: int,
    initial_state_id: int,
    resolver: InitialStateResolver | None,
) -> object:
    if resolver is None:
        raise RuntimeError(
            "LIBERO backend initial state resolver is required for dataset preflight"
        )
    try:
        return resolver(suite, task_id, initial_state_id)
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve LIBERO initial state {initial_state_id} "
            f"for {suite} task {task_id}"
        ) from exc


def _validate_endpoint(
    resolved: ResolvedExperimentSpec, *, check_network: bool
) -> None:
    if resolved.policy_adapter != "remote_http":
        return
    endpoint = resolved.policy_endpoint
    parsed = urlparse(endpoint or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("remote HTTP policy endpoint must be an absolute HTTP(S) URL")
    if not resolved.policy.model_key:
        raise ValueError(
            "remote HTTP policy requires policy.model_key for health identity verification"
        )
    if (
        resolved.deployment.mode == "jetson_remote_client"
        and not resolved.deployment.allow_loopback_endpoint
        and _is_loopback_host(parsed.hostname)
    ):
        raise ValueError(
            "Jetson remote client endpoint must use a non-loopback host; "
            "set deployment.allow_loopback_endpoint only for an explicit local test"
        )
    if not check_network:
        return
    observed = probe_remote_policy(endpoint)
    mismatches = _policy_identity_mismatches(resolved, observed)
    if mismatches:
        raise RemotePolicyIdentityMismatch("; ".join(mismatches))


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _policy_identity_mismatches(
    resolved: ResolvedExperimentSpec, observed: dict[str, object]
) -> list[str]:
    expected = {
        "model_key": resolved.policy.model_key,
        "checkpoint": resolved.resolved_checkpoint,
        "precision": resolved.policy.precision,
    }
    if resolved.resolved_revision is not None:
        expected["revision"] = resolved.resolved_revision
    return [
        f"{field} mismatch: expected {value!r}, observed {observed.get(field)!r}"
        for field, value in expected.items()
        if observed.get(field) != value
    ]


def _failure(field: str, message: str, failure_type: str) -> ValidationReport:
    return ValidationReport(
        ok=False,
        resolved_spec=None,
        issues=(ValidationIssue(field=field, message=message, failure_type=failure_type),),
    )
