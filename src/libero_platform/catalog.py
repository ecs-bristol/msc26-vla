from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from libero_platform.spec import ExperimentSpec, ResolvedExperimentSpec


_ENVIRONMENT_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class CatalogError(ValueError):
    """Raised when catalog data or a catalog resolution is invalid."""


class Catalog:
    def __init__(
        self,
        project_root: Path,
        suites: dict[str, Mapping[str, Any]],
        policies: dict[str, Mapping[str, Any]],
        deployments: dict[str, Mapping[str, Any]],
    ) -> None:
        self._project_root = project_root
        self._suites = suites
        self._policies = policies
        self._deployments = deployments

    @classmethod
    def load(cls, config_root: Path) -> "Catalog":
        config_root = config_root.resolve(strict=True)
        if not config_root.is_dir():
            raise CatalogError(f"catalog root is not a directory: {config_root}")

        return cls(
            project_root=config_root.parent.resolve(),
            suites=_load_catalog(
                config_root / "libero_suites.yaml",
                "suites",
                {"key", "backend_key", "task_count", "dataset_directory"},
                "suite",
            ),
            policies=_load_catalog(
                config_root / "policy_catalog.yaml",
                "policies",
                {
                    "key",
                    "adapter",
                    "default_checkpoint",
                    "supported_deployments",
                    "supported_precisions",
                    "supported_quantizations",
                },
                "policy",
            ),
            deployments=_load_catalog(
                config_root / "deployment_profiles.yaml",
                "deployments",
                {"key", "mode", "endpoint"},
                "deployment",
            ),
        )

    def list_suites(self) -> tuple[Mapping[str, Any], ...]:
        return _read_only_rows(self._suites)

    def list_policies(self) -> tuple[Mapping[str, Any], ...]:
        return _read_only_rows(self._policies)

    def list_deployments(self) -> tuple[Mapping[str, Any], ...]:
        return _read_only_rows(self._deployments)

    def resolve(
        self, spec: ExperimentSpec, source_path: Path
    ) -> ResolvedExperimentSpec:
        suite = _catalog_row(self._suites, spec.benchmark.suite, "suite")
        policy = _catalog_row(self._policies, spec.policy.key, "policy")
        deployment = _catalog_row(self._deployments, spec.deployment.profile, "deployment")

        if deployment["mode"] != spec.deployment.mode:
            raise CatalogError(
                f"deployment profile {spec.deployment.profile!r} mode "
                f"{deployment['mode']!r} does not match {spec.deployment.mode!r}"
            )
        _require_supported(policy, "supported_deployments", spec.deployment.mode)
        _require_supported(policy, "supported_precisions", spec.policy.precision)
        _require_supported(policy, "supported_quantizations", spec.policy.quantization)

        dataset_directory = (self._project_root / suite["dataset_directory"]).resolve()
        try:
            dataset_directory.relative_to(self._project_root)
        except ValueError as exc:
            raise CatalogError(
                f"suite {spec.benchmark.suite!r} dataset_directory escapes package project root"
            ) from exc

        resolved_checkpoint = (
            policy["default_checkpoint"]
            if spec.policy.checkpoint == "catalog:default"
            else spec.policy.checkpoint
        )
        return ResolvedExperimentSpec.model_validate(
            {
                **spec.model_dump(mode="json"),
                "source_path": str(source_path),
                "dataset_directory": str(dataset_directory),
                "resolved_checkpoint": resolved_checkpoint,
                "resolved_revision": spec.policy.revision,
                "policy_adapter": policy["adapter"],
                "policy_endpoint": _resolve_endpoint_environment_placeholder(
                    deployment["endpoint"]
                ),
            }
        )


def _resolve_endpoint_environment_placeholder(endpoint: object) -> object:
    """Expand an exact environment placeholder used by a deployment endpoint."""
    if not isinstance(endpoint, str) or "${" not in endpoint:
        return endpoint
    match = _ENVIRONMENT_PLACEHOLDER.fullmatch(endpoint)
    if match is None:
        raise CatalogError(
            "deployment endpoint supports only an exact environment placeholder like "
            "'${JETSON_ENDPOINT}'"
        )
    variable_name = match.group(1)
    value = os.environ.get(variable_name)
    if not value:
        raise CatalogError(
            f"deployment endpoint requires environment variable {variable_name!r}, "
            "but it is not set"
        )
    return value


def _load_catalog(
    path: Path, root_key: str, allowed_fields: set[str], row_name: str
) -> dict[str, Mapping[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not load catalog {path.name}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {root_key}:
        raise CatalogError(f"catalog {path.name} must contain only {root_key!r}")
    rows = data[root_key]
    if not isinstance(rows, list):
        raise CatalogError(f"catalog {path.name} {root_key!r} must be a list")

    catalog: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != allowed_fields:
            raise CatalogError(f"catalog {path.name} contains unknown or incomplete {row_name} data")
        key = row["key"]
        if not isinstance(key, str) or not key:
            raise CatalogError(f"catalog {path.name} {row_name} key must be a non-empty string")
        if key in catalog:
            raise CatalogError(f"duplicate {row_name} key: {key}")
        _validate_row_types(path.name, row_name, row)
        catalog[key] = MappingProxyType(
            {field: _freeze_catalog_value(value) for field, value in row.items()}
        )
    return catalog


def _validate_row_types(filename: str, row_name: str, row: dict[str, Any]) -> None:
    for key, value in row.items():
        if key.startswith("supported_"):
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise CatalogError(f"catalog {filename} {row_name} {key} must be a list of strings")
        elif key == "task_count":
            if not isinstance(value, int) or isinstance(value, bool):
                raise CatalogError(f"catalog {filename} {row_name} task_count must be an integer")
        elif key == "endpoint":
            if value is not None and not isinstance(value, str):
                raise CatalogError(f"catalog {filename} deployment endpoint must be a string or null")
        elif not isinstance(value, str):
            raise CatalogError(f"catalog {filename} {row_name} {key} must be a string")


def _read_only_rows(
    rows: dict[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(row)) for row in rows.values())


def _freeze_catalog_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_catalog_value(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_catalog_value(item) for key, item in value.items()}
        )
    return value


def _catalog_row(
    rows: dict[str, Mapping[str, Any]], key: str, row_name: str
) -> Mapping[str, Any]:
    try:
        return rows[key]
    except KeyError as exc:
        raise CatalogError(f"unknown {row_name} key: {key}") from exc


def _require_supported(policy: Mapping[str, Any], field: str, value: str) -> None:
    if value not in policy[field]:
        label = field[len("supported_") :]
        if label.endswith("s"):
            label = label[:-1]
        raise CatalogError(f"policy {policy['key']!r} does not support {value} {label}")
