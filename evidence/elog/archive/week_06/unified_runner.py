from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .adapters import create_adapter
from .pre_jetson_runner import Task, read_tasks, select_tasks
from .run_inference import ROOT
from .unified_results import get_git_commit, write_run_outputs
from .unified_schema import AdapterRequest, AdapterResponse, make_trial_record


DEFAULT_CONFIG = ROOT / "configs" / "pre_jetson_workflow.yaml"


def read_workflow_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_model_config(root: Path) -> dict[str, Any]:
    with (root / "configs" / "models.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_model_entries(
    *,
    experiment: dict[str, Any],
    model_cfg: dict[str, Any],
    models_override: str | None,
) -> list[dict[str, str]]:
    if models_override:
        experiment_entries = {
            entry["model_key"]: entry
            for entry in experiment.get("models", [])
        }
        entries = [
            {**experiment_entries.get(key.strip(), {}), "model_key": key.strip()}
            for key in models_override.split(",")
            if key.strip()
        ]
    elif "models" in experiment:
        entries = list(experiment["models"])
    else:
        entries = [{"model_key": key} for key in experiment.get("model_keys", [])]

    resolved = []
    for entry in entries:
        model_key = entry["model_key"]
        selected_model_cfg = model_cfg["models"][model_key]
        deployment_mode = entry.get("deployment_mode")
        if not deployment_mode:
            supported_modes = selected_model_cfg.get("supported_deployment_modes", ["pc_local"])
            deployment_mode = supported_modes[0]
        adapter = selected_model_cfg.get("adapter", infer_adapter(selected_model_cfg))
        resolved.append(
            {
                "model_key": model_key,
                "adapter": adapter,
                "deployment_mode": deployment_mode,
            }
        )
    return resolved


def infer_adapter(model_cfg: dict[str, Any]) -> str:
    model_type = model_cfg.get("model_type", "")
    if model_type == "rule_baseline":
        return "scripted_adapter"
    if model_type == "openvla":
        return "openvla_adapter"
    if model_type == "mock":
        return "mock_adapter"
    return "vlm_text_adapter"


def make_request(
    *,
    task: Task,
    image_path: Path,
    model_key: str,
    model_cfg: dict[str, Any],
    adapter: str,
    deployment_mode: str,
    experiment: dict[str, Any],
    max_new_tokens: int,
) -> AdapterRequest:
    runtime_config = {
        **model_cfg,
        "max_new_tokens": max_new_tokens,
        "local_files_only": bool(experiment.get("local_files_only", False)),
        "runtime_precision": model_cfg.get("dtype", ""),
        "quantization": model_cfg.get("quantization", ""),
    }
    return AdapterRequest(
        task_id=task.task_id,
        image_path=image_path,
        instruction=task.prompt,
        expected_target=task.expected_target,
        expected_action=task.expected_action,
        model_key=model_key,
        model_id=str(model_cfg.get("model_id", model_key)),
        adapter=adapter,
        deployment_mode=deployment_mode,
        runtime_config=runtime_config,
    )


def run_experiment(
    config_path: Path,
    experiment_name: str,
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    workflow_cfg = read_workflow_config(config_path)
    model_cfg = read_model_config(root)
    experiment = workflow_cfg["experiments"][experiment_name]
    if experiment.get("runner") != "unified_adapter_runner":
        raise ValueError(f"Unsupported runner for unified_runner: {experiment.get('runner')}")

    model_entries = resolve_model_entries(
        experiment=experiment,
        model_cfg=model_cfg,
        models_override=args.models,
    )
    task_ids: list[str] | str = args.tasks.split(",") if args.tasks else experiment["task_ids"]
    repeats = args.repeats if args.repeats is not None else int(experiment.get("repeats", 1))
    warmup = args.warmup if args.warmup is not None else int(experiment.get("warmup", 0))
    max_new_tokens = args.max_new_tokens or int(experiment.get("max_new_tokens", 80))
    device_profile = args.device_profile or str(experiment.get("device_profile", "unknown_device"))

    task_file = root / experiment["task_file"]
    image_dir = root / experiment["image_dir"]
    output_root = root / experiment["output_dir"]
    tasks = select_tasks(read_tasks(task_file), task_ids)

    dry_run_payload = {
        "experiment": experiment_name,
        "models": model_entries,
        "tasks": [task.task_id for task in tasks],
        "repeats": repeats,
        "warmup": warmup,
        "max_new_tokens": max_new_tokens,
        "device_profile": device_profile,
    }
    if args.dry_run:
        return dry_run_payload

    created_at = datetime.now().isoformat(timespec="seconds")
    run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    rows: list[dict[str, Any]] = []

    for model_entry in model_entries:
        model_key = model_entry["model_key"]
        selected_model_cfg = model_cfg["models"][model_key]
        adapter_name = model_entry["adapter"]
        deployment_mode = model_entry["deployment_mode"]
        adapter = create_adapter(model_key, {**selected_model_cfg, "adapter": adapter_name})
        load_success = True
        load_error = ""
        try:
            adapter.load(local_files_only=bool(experiment.get("local_files_only", False)))
        except RuntimeError as exc:
            load_success = False
            load_error = str(exc)
        except Exception as exc:
            load_success = False
            load_error = str(exc)

        warmup_tasks = tasks[:1] * warmup
        for warmup_task in warmup_tasks:
            warmup_image = image_dir / warmup_task.image
            request = make_request(
                task=warmup_task,
                image_path=warmup_image,
                model_key=model_key,
                model_cfg=selected_model_cfg,
                adapter=adapter_name,
                deployment_mode=deployment_mode,
                experiment=experiment,
                max_new_tokens=max_new_tokens,
            )
            if load_success:
                adapter.predict(request)

        for task in tasks:
            image_path = image_dir / task.image
            if not image_path.exists():
                raise FileNotFoundError(f"Task image not found: {image_path}")
            for repeat_idx in range(repeats):
                request = make_request(
                    task=task,
                    image_path=image_path,
                    model_key=model_key,
                    model_cfg=selected_model_cfg,
                    adapter=adapter_name,
                    deployment_mode=deployment_mode,
                    experiment=experiment,
                    max_new_tokens=max_new_tokens,
                )
                timestamp = datetime.now().isoformat(timespec="seconds")
                if not load_success:
                    response = AdapterResponse(
                        model_key=model_key,
                        adapter=adapter_name,
                        deployment_mode=deployment_mode,
                        task_id=task.task_id,
                        load_success=False,
                        success=False,
                        failure_type=classify_load_failure(load_error),
                        error=load_error,
                    )
                else:
                    start = time.perf_counter()
                    try:
                        response = adapter.predict(request)
                    except RuntimeError as exc:
                        response = AdapterResponse(
                            model_key=model_key,
                            adapter=adapter_name,
                            deployment_mode=deployment_mode,
                            task_id=task.task_id,
                            load_success=True,
                            success=False,
                            failure_type=classify_load_failure(str(exc)),
                            error=str(exc),
                            end_to_end_ms=(time.perf_counter() - start) * 1000,
                        )
                    except Exception as exc:
                        response = AdapterResponse(
                            model_key=model_key,
                            adapter=adapter_name,
                            deployment_mode=deployment_mode,
                            task_id=task.task_id,
                            load_success=True,
                            success=False,
                            failure_type="runtime_error",
                            error=str(exc),
                            end_to_end_ms=(time.perf_counter() - start) * 1000,
                        )
                rows.append(
                    make_trial_record(
                        run_id=run_id,
                        experiment=experiment_name,
                        timestamp=timestamp,
                        repeat_idx=repeat_idx,
                        request=request,
                        response=response,
                        device_profile=device_profile,
                        notes=task.notes,
                    )
                )

    metadata = {
        "run_id": run_id,
        "experiment": experiment_name,
        "description": experiment.get("description", ""),
        "git_commit": get_git_commit(root.parents[1] if root.name == "Local_VLA_Benchmark_Framework" else root),
        "config_path": str(config_path),
        "device_profile": device_profile,
        "models": model_entries,
        "task_ids": [task.task_id for task in tasks],
        "repeats": repeats,
        "warmup": warmup,
        "max_new_tokens": max_new_tokens,
        "local_files_only": bool(experiment.get("local_files_only", False)),
        "created_at": created_at,
    }
    return write_run_outputs(run_dir=run_dir, metadata=metadata, trial_rows=rows)


def classify_load_failure(error: str) -> str:
    lowered = error.lower()
    if "out of memory" in lowered or "oom" in lowered or "cuda memory" in lowered:
        return "oom"
    if "import" in lowered or "module" in lowered or "dependency" in lowered:
        return "dependency_error"
    return "load_error"


def parse_args() -> argparse.Namespace:
    cfg = read_workflow_config(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser(description="Run unified VLA/VLM adapter benchmark workflows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", default=cfg.get("default_experiment", "jetson_readiness_interface_smoke"))
    parser.add_argument("--list", action="store_true", help="List unified-adapter experiments.")
    parser.add_argument("--models", help="Comma-separated model keys, e.g. local_rule_baseline,mock_remote_policy")
    parser.add_argument("--tasks", help="Comma-separated task_id override, e.g. desk_cup_pick,blue_can_pick")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device-profile", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_workflow_config(args.config)
    if args.list:
        for name, experiment in cfg["experiments"].items():
            if experiment.get("runner") == "unified_adapter_runner":
                print(f"{name}: {experiment.get('description', '')}")
        return
    if args.experiment not in cfg["experiments"]:
        raise SystemExit(f"Unknown experiment: {args.experiment}")
    outputs = run_experiment(args.config, args.experiment, args)
    if outputs:
        print(json.dumps({key: _json_safe(value) for key, value in outputs.items()}, indent=2, ensure_ascii=False))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
