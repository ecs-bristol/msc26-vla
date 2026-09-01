from __future__ import annotations

from copy import deepcopy


class SpecValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]) -> None:
        super().__init__("ExperimentSpec validation failed")
        self.errors = errors


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row[key]): row for row in rows}


def _bounded_int(value: object, *, field: str, minimum: int, maximum: int, errors: list[dict[str, str]]) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append({"field": field, "message": "must be an integer"})
        return minimum
    if not minimum <= parsed <= maximum:
        errors.append({"field": field, "message": f"must be between {minimum} and {maximum}"})
    return parsed


def validate_experiment_spec(payload: dict, catalog: dict) -> dict:
    spec = deepcopy(payload)
    errors: list[dict[str, str]] = []
    if int(spec.get("schema_version", 0)) != 1:
        errors.append({"field": "schema_version", "message": "must equal 1"})
    mode = str(spec.get("mode", ""))
    if mode not in {"evaluation", "training"}:
        errors.append({"field": "mode", "message": "must be evaluation or training"})

    deployments = _index(catalog["deployments"], "key")
    deployment_key = str(spec.get("deployment_profile", ""))
    deployment = deployments.get(deployment_key)
    if deployment is None:
        errors.append({"field": "deployment_profile", "message": "unknown deployment profile"})
    elif not deployment.get("available", False):
        errors.append({"field": "deployment_profile", "message": str(deployment.get("reason", "deployment unavailable"))})

    environments = _index(catalog["environments"], "key")
    environment = spec.setdefault("environment", {})
    environment_definition = environments.get(str(environment.get("key", "")))
    if environment_definition is None:
        errors.append({"field": "environment.key", "message": "unknown environment"})
    environment["seed"] = _bounded_int(environment.get("seed", 0), field="environment.seed", minimum=0, maximum=2_147_483_647, errors=errors)
    environment["max_steps"] = _bounded_int(environment.get("max_steps", 500), field="environment.max_steps", minimum=50, maximum=5000, errors=errors)
    environment["camera"] = str(environment.get("camera", "frontview"))
    environment["robot"] = str(environment.get("robot", (environment_definition or {}).get("robot", "Panda")))
    environment["controller"] = str(environment.get("controller", "OSC_POSE"))
    environment["randomization"] = bool(environment.get("randomization", False))
    valid_objects = {str(item["name"]) for item in (environment_definition or {}).get("objects", [])}
    overrides = list(environment.get("object_overrides", []))
    if any(str(item.get("name", "")) not in valid_objects for item in overrides):
        errors.append({"field": "environment.object_overrides", "message": "contains an unknown scene object"})
    for index, item in enumerate(overrides):
        position = item.get("position")
        if not isinstance(position, list) or len(position) != 3:
            errors.append({"field": f"environment.object_overrides.{index}.position", "message": "must contain x, y, z"})
            continue
        try:
            item["position"] = [float(value) for value in position]
        except (TypeError, ValueError):
            errors.append({"field": f"environment.object_overrides.{index}.position", "message": "coordinates must be numeric"})
        if any(abs(value) > 2.0 for value in item.get("position", []) if isinstance(value, float)):
            errors.append({"field": f"environment.object_overrides.{index}.position", "message": "coordinates must stay within the scene bounds"})
    environment["object_overrides"] = overrides

    if mode == "evaluation":
        models = _index(catalog["models"], "key")
        selection = spec.get("model_selection") or {}
        selection_mode = str(selection.get("mode", "single"))
        model_keys = [str(value) for value in selection.get("model_keys", [])]
        if selection_mode not in {"single", "batch"}:
            errors.append({"field": "model_selection.mode", "message": "must be single or batch"})
        required_count = 2 if selection_mode == "batch" else 1
        if len(model_keys) < required_count or (selection_mode == "single" and len(model_keys) != 1):
            errors.append({"field": "model_selection.model_keys", "message": f"{selection_mode} mode requires {'at least two' if required_count == 2 else 'exactly one'} model"})
        elif any(key not in models for key in model_keys):
            errors.append({"field": "model_selection.model_keys", "message": "contains an unknown model"})
        elif deployment is not None:
            incompatible = [key for key in model_keys if deployment_key not in models[key].get("supported_deployment_modes", [])]
            if incompatible:
                errors.append({"field": "model_selection.model_keys", "message": f"deployment incompatible: {', '.join(incompatible)}"})
        selection["mode"] = selection_mode
        selection["model_keys"] = model_keys
        selection["runtime_precision"] = str(selection.get("runtime_precision", "none"))
        selection["quantization"] = str(selection.get("quantization", "none"))
        spec["model_selection"] = selection
    else:
        spec["model_selection"] = None

    tasks = {(str(row["task_id"]), int(row["version"])): row for row in catalog["tasks"]}
    task = spec.setdefault("task", {})
    task_key = (str(task.get("task_id", "")), int(task.get("version", 0) or 0))
    if task_key not in tasks:
        errors.append({"field": "task", "message": "unknown task version"})
    else:
        spec["task"] = deepcopy(tasks[task_key])

    execution = spec.setdefault("execution", {})
    execution["trials"] = _bounded_int(execution.get("trials", 1), field="execution.trials", minimum=1, maximum=100, errors=errors)
    execution["warmup"] = _bounded_int(execution.get("warmup", 0), field="execution.warmup", minimum=0, maximum=10, errors=errors)

    recording = spec.setdefault("recording", {})
    recording["core"] = True
    recording["save_frames"] = bool(recording.get("save_frames", False))
    recording["save_video"] = bool(recording.get("save_video", False))
    recording["frame_stride"] = _bounded_int(recording.get("frame_stride", 20), field="recording.frame_stride", minimum=1, maximum=200, errors=errors)

    if mode == "training":
        trainers = _index(catalog["trainers"], "key")
        datasets = _index(catalog["datasets"], "key")
        training = spec.get("training") or {}
        if str(training.get("trainer_key", "")) not in trainers:
            errors.append({"field": "training.trainer_key", "message": "unknown trainer"})
        if str(training.get("dataset_key", "")) not in datasets:
            errors.append({"field": "training.dataset_key", "message": "unknown dataset"})
        spec["training"] = training
    else:
        spec["training"] = None

    if errors:
        raise SpecValidationError(errors)
    return spec


def expand_model_keys(spec: dict) -> list[str]:
    selection = spec.get("model_selection")
    return list(selection["model_keys"]) if selection else []
