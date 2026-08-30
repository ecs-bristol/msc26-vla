from pathlib import Path

import pytest
import yaml

from libero_platform.catalog import Catalog, CatalogError
from libero_platform.spec import ExperimentSpec
from tests.test_spec import VALID


def test_resolves_policy_and_deployment(tmp_path: Path, catalog_root: Path) -> None:
    spec = ExperimentSpec.model_validate(VALID)

    resolved = Catalog.load(catalog_root).resolve(spec, tmp_path / "source.yaml")

    assert resolved.policy_adapter == "zero"
    assert resolved.resolved_checkpoint == "none"
    assert resolved.deployment.mode == "pc_local"
    assert resolved.policy_endpoint is None
    assert resolved.dataset_directory == str(
        (catalog_root.parent / "datasets" / "libero_spatial").resolve()
    )
    assert resolved.model_dump(exclude={
        "source_path",
        "dataset_directory",
        "resolved_checkpoint",
        "policy_adapter",
        "policy_endpoint",
        "device_metadata",
    }) == spec.model_dump()


def test_replaces_only_catalog_default_checkpoint(catalog_root: Path) -> None:
    custom_checkpoint = "local/checkpoints/model"
    spec = ExperimentSpec.model_validate(
        {**VALID, "policy": {**VALID["policy"], "checkpoint": custom_checkpoint}}
    )

    resolved = Catalog.load(catalog_root).resolve(spec, Path("x.yaml"))

    assert resolved.resolved_checkpoint == custom_checkpoint
    assert resolved.policy.checkpoint == custom_checkpoint


def test_rejects_incompatible_policy_deployment(catalog_root: Path) -> None:
    payload = {
        **VALID,
        "policy": {
            **VALID["policy"],
            "key": "smolvla_libero",
            "checkpoint": "catalog:default",
            "precision": "fp16",
        },
        "deployment": {"mode": "jetson_local", "profile": "jetson_default"},
    }

    with pytest.raises(CatalogError, match="does not support jetson_local"):
        Catalog.load(catalog_root).resolve(ExperimentSpec.model_validate(payload), Path("x.yaml"))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("suites", "libero_spatial"),
        ("policies", "missing_policy"),
        ("deployments", "missing_profile"),
    ],
)
def test_rejects_unknown_catalog_key(
    tmp_path: Path, catalog_root: Path, section: str, key: str
) -> None:
    for source in catalog_root.glob("*.yaml"):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {**VALID}
    if section == "suites":
        suites = yaml.safe_load((tmp_path / "libero_suites.yaml").read_text(encoding="utf-8"))
        suites["suites"] = [row for row in suites["suites"] if row["key"] != key]
        (tmp_path / "libero_suites.yaml").write_text(
            yaml.safe_dump(suites, sort_keys=False), encoding="utf-8"
        )
    elif section == "policies":
        payload["policy"] = {**VALID["policy"], "key": key}
    else:
        payload["deployment"] = {**VALID["deployment"], "profile": key}

    with pytest.raises(CatalogError, match=key):
        Catalog.load(tmp_path).resolve(ExperimentSpec.model_validate(payload), Path("x.yaml"))


def test_rejects_profile_mode_mismatch(catalog_root: Path) -> None:
    spec = ExperimentSpec.model_validate(
        {**VALID, "deployment": {"mode": "pc_local", "profile": "jetson_default"}}
    )

    with pytest.raises(CatalogError, match="does not match"):
        Catalog.load(catalog_root).resolve(spec, Path("x.yaml"))


@pytest.mark.parametrize(
    ("field", "value"), [("precision", "fp16"), ("quantization", "int8")]
)
def test_rejects_unsupported_policy_option(
    catalog_root: Path, field: str, value: str
) -> None:
    spec = ExperimentSpec.model_validate(
        {**VALID, "policy": {**VALID["policy"], field: value}}
    )

    with pytest.raises(CatalogError, match=f"does not support {value}"):
        Catalog.load(catalog_root).resolve(spec, Path("x.yaml"))


def test_rejects_non_list_catalog_data(tmp_path: Path) -> None:
    for filename, root_key in (
        ("libero_suites.yaml", "suites"),
        ("policy_catalog.yaml", "policies"),
        ("deployment_profiles.yaml", "deployments"),
    ):
        (tmp_path / filename).write_text(
            yaml.safe_dump({root_key: {"not": "a list"}}), encoding="utf-8"
        )

    with pytest.raises(CatalogError, match="must be a list"):
        Catalog.load(tmp_path)


def test_rejects_duplicate_catalog_key(tmp_path: Path, catalog_root: Path) -> None:
    for source in catalog_root.glob("*.yaml"):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    suites = yaml.safe_load((tmp_path / "libero_suites.yaml").read_text(encoding="utf-8"))
    suites["suites"].append(dict(suites["suites"][0]))
    (tmp_path / "libero_suites.yaml").write_text(
        yaml.safe_dump(suites, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(CatalogError, match="duplicate suite key"):
        Catalog.load(tmp_path)


def test_rejects_dataset_directory_outside_project_root(tmp_path: Path, catalog_root: Path) -> None:
    for source in catalog_root.glob("*.yaml"):
        (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    suites = yaml.safe_load((tmp_path / "libero_suites.yaml").read_text(encoding="utf-8"))
    suites["suites"][0]["dataset_directory"] = "../outside"
    (tmp_path / "libero_suites.yaml").write_text(
        yaml.safe_dump(suites, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(CatalogError, match="escapes package project root"):
        Catalog.load(tmp_path).resolve(ExperimentSpec.model_validate(VALID), Path("x.yaml"))


def test_list_methods_return_read_only_copies(catalog_root: Path) -> None:
    catalog = Catalog.load(catalog_root)

    suites = catalog.list_suites()
    policies = catalog.list_policies()
    deployments = catalog.list_deployments()

    assert isinstance(suites, tuple)
    assert isinstance(policies, tuple)
    assert isinstance(deployments, tuple)
    assert suites[0]["key"] == "libero_spatial"
    with pytest.raises(TypeError):
        suites[0]["key"] = "changed"


def test_exposed_policy_mutation_cannot_change_resolution(catalog_root: Path) -> None:
    catalog = Catalog.load(catalog_root)
    policy = next(row for row in catalog.list_policies() if row["key"] == "smolvla_libero")

    try:
        policy["supported_deployments"].append("jetson_local")
    except (AttributeError, TypeError):
        pass

    spec = ExperimentSpec.model_validate(
        {
            **VALID,
            "policy": {
                **VALID["policy"],
                "key": "smolvla_libero",
                "checkpoint": "catalog:default",
                "precision": "fp16",
            },
            "deployment": {"mode": "jetson_local", "profile": "jetson_default"},
        }
    )
    with pytest.raises(CatalogError, match="does not support jetson_local"):
        catalog.resolve(spec, Path("x.yaml"))
