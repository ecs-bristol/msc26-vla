from __future__ import annotations

import ast
import csv
import json
import mimetypes
import re
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, unquote, urlparse

from console.catalog import build_catalog
from console.executors import ConsolePaths
from console.orchestrator import ExperimentOrchestrator
from console.preview import render_scene_preview
from console.recorder import RunRecorder
from console.results import inspect_result, normalize_batch
from console.schema import SpecValidationError
from console.task_store import TaskStore


UI_ROOT = Path(__file__).resolve().parent
FINAL_ROOT = UI_ROOT.parent
WORKSPACE_ROOT = FINAL_ROOT.parent

LOCAL_BENCH = FINAL_ROOT / "Local_VLA_Benchmark_Framework"
SIM_ROOT = FINAL_ROOT / "Robosuite_MuJoCo_Sim"
JOBS_ROOT = UI_ROOT / "jobs"
JOBS: dict[str, dict] = {}
CORE_MATRIX_MODELS = {"qwen2_vl_2b", "smolvlm2_500m"}
UNIFIED_EXPERIMENTS = {"jetson_readiness_interface_smoke", "unified_vlm_pc_smoke"}
DEVICE_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PREVIEW_ID_RE = re.compile(r"^[a-f0-9]{64}$")
UNSAFE_CONSOLE_PAYLOAD_KEYS = frozenset({"cmd", "cwd", "pid"})


@dataclass
class ConsoleServices:
    """Server-owned console dependencies; tests replace the singleton directly."""

    catalog: dict
    task_store: TaskStore
    recorder: RunRecorder
    orchestrator: ExperimentOrchestrator
    paths: ConsolePaths
    state_root: Path
    preview_root: Path


_console_services: ConsoleServices | None = None
_console_services_lock = RLock()


def get_console_services() -> ConsoleServices:
    """Return the console composition root without importing ML or simulator runtimes."""
    global _console_services
    with _console_services_lock:
        if _console_services is None:
            state_root = UI_ROOT / "state"
            paths = ConsolePaths(UI_ROOT, LOCAL_BENCH, SIM_ROOT)
            task_store = TaskStore(LOCAL_BENCH, state_root)
            catalog = build_catalog(LOCAL_BENCH, SIM_ROOT, state_root)
            recorder = RunRecorder(state_root)
            _console_services = ConsoleServices(
                catalog=catalog,
                task_store=task_store,
                recorder=recorder,
                orchestrator=ExperimentOrchestrator(recorder, catalog, paths),
                paths=paths,
                state_root=state_root,
                preview_root=state_root / "previews",
            )
        return _console_services


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def parse_models(path: Path) -> list[dict[str, str]]:
    """Parse the small local models.yaml without requiring PyYAML."""
    models: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_models = False
    for raw_line in read_text(path).splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line == "models:":
            in_models = True
            continue
        if not in_models:
            continue
        key_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if key_match:
            if current:
                models.append(current)
            current = {"key": key_match.group(1)}
            continue
        field_match = re.match(r"^    ([A-Za-z0-9_:-]+):\s*(.*)$", line)
        if field_match and current is not None:
            current[field_match.group(1)] = parse_scalar(field_match.group(2))
    if current:
        models.append(current)
    return models


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def coerce(value: str):
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value == "":
        return ""
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_trial_rows(path: Path) -> list[dict]:
    rows = []
    for row in read_csv(path):
        parsed = {key: coerce(value) for key, value in row.items()}
        for list_field in ("phase_changes", "frame_paths"):
            value = parsed.get(list_field)
            if not isinstance(value, str) or not value.startswith("["):
                continue
            try:
                parsed[list_field] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed[list_field] = []
        rows.append(parsed)
    return rows


def list_scripted_runs() -> list[dict]:
    root = SIM_ROOT / "outputs" / "scripted_pick_benchmark"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        summary = read_csv(run_dir / "summary.csv")
        trials = parse_trial_rows(run_dir / "trials.csv")
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "scripted_pick",
                "path": str(run_dir),
                "summary": summary,
                "trials": trials,
                "metadata": metadata,
            }
        )
    return runs


def list_vlm_pick_runs() -> list[dict]:
    root = SIM_ROOT / "outputs" / "vlm_target_pick_benchmark"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "vlm_pick",
                "path": str(run_dir),
                "summary": read_csv(run_dir / "summary.csv"),
                "trials": parse_trial_rows(run_dir / "trials.csv"),
                "metadata": metadata,
            }
        )
    return runs


def list_pre_jetson_runs() -> list[dict]:
    root = LOCAL_BENCH / "results" / "pre_jetson"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "pre_jetson",
                "path": str(run_dir),
                "summary": read_csv(run_dir / "summary.csv"),
                "trials": parse_trial_rows(run_dir / "trials.csv"),
                "metadata": metadata,
            }
        )
    return runs


def list_unified_runs() -> list[dict]:
    root = LOCAL_BENCH / "results" / "unified"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "unified",
                "path": str(run_dir),
                "metadata": metadata,
                "summary": read_csv(run_dir / "summary.csv"),
                "failures": read_csv(run_dir / "failures.csv"),
                "trials": parse_trial_rows(run_dir / "trials.csv"),
            }
        )
    return runs


def list_pc_matrix_runs() -> list[dict]:
    root = SIM_ROOT / "outputs" / "pc_benchmark_matrix"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "manifest.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "pc_matrix",
                "path": str(run_dir),
                "summary": read_csv(run_dir / "matrix_summary.csv"),
                "trials": parse_trial_rows(run_dir / "matrix_trials.csv"),
                "overview": read_csv(run_dir / "matrix_runs.csv"),
                "metadata": metadata,
            }
        )
    return runs


def list_tiny_bc_runs() -> list[dict]:
    root = SIM_ROOT / "outputs" / "tiny_bc_closed_loop"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "tiny_bc_closed_loop",
                "path": str(run_dir),
                "summary": read_csv(run_dir / "summary.csv"),
                "trials": parse_trial_rows(run_dir / "trials.csv"),
                "metadata": metadata,
            }
        )
    return runs


def list_visual_target_bc_runs() -> list[dict]:
    root = SIM_ROOT / "outputs" / "visual_target_bc_closed_loop"
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        metadata = {}
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        runs.append(
            {
                "id": run_dir.name,
                "kind": "visual_target_bc_closed_loop",
                "path": str(run_dir),
                "summary": read_csv(run_dir / "summary.csv"),
                "trials": parse_trial_rows(run_dir / "trials.csv"),
                "metadata": metadata,
            }
        )
    return runs


def list_trained_policies() -> list[dict]:
    root = SIM_ROOT / "outputs" / "trained_policies"
    if not root.exists():
        return []
    policies = []
    for policy_dir in sorted([path for path in root.iterdir() if path.is_dir()], reverse=True):
        policy_path = policy_dir / "policy.npz"
        if not policy_path.exists():
            continue
        metadata = {}
        metadata_path = policy_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(read_text(metadata_path))
            except json.JSONDecodeError:
                metadata = {}
        policies.append(
            {
                "id": policy_dir.name,
                "path": str(policy_path),
                "metadata": metadata,
            }
        )
    return policies


def load_scene() -> dict:
    path = SIM_ROOT / "configs" / "scene_config.json"
    if not path.exists():
        return {}
    return json.loads(read_text(path))


def load_config() -> dict:
    return {
        "models": parse_models(LOCAL_BENCH / "configs" / "models.yaml"),
        "tasks": read_csv(LOCAL_BENCH / "data" / "tasks.csv"),
        "scene": load_scene(),
        "trained_policies": list_trained_policies(),
        "paths": {
            "local_benchmark": str(LOCAL_BENCH),
            "simulator": str(SIM_ROOT),
        },
    }


def load_evidence() -> dict:
    rows = read_csv(FINAL_ROOT / "Evidence" / "evidence_index.csv")
    selected = [row for row in rows if row.get("status") == "selected"]
    source_paths = {
        row.get("source_path", ""): row
        for row in selected
        if row.get("source_path")
    }
    evidence_paths = {
        row.get("evidence_path", ""): row
        for row in selected
        if row.get("evidence_path")
    }
    return {
        "rows": rows,
        "selected_ids": [row.get("evidence_id", "") for row in selected if row.get("evidence_id")],
        "source_paths": source_paths,
        "evidence_paths": evidence_paths,
    }


def python_for(root: Path, venv_name: str) -> str:
    candidate = root / venv_name / "Scripts" / "python.exe"
    return str(candidate if candidate.exists() else Path(sys.executable))


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def poll_job(job: dict) -> dict:
    process: subprocess.Popen | None = job.get("_process")
    if process is None:
        return job
    return_code = process.poll()
    if return_code is None:
        job["status"] = "running"
    else:
        job["status"] = "completed" if return_code == 0 else "failed"
        job["return_code"] = return_code
        job["finished_at"] = job.get("finished_at") or time.strftime("%Y-%m-%d %H:%M:%S")
    job["log_tail"] = tail_text(Path(job["log_path"]))
    return job


def public_job(job: dict) -> dict:
    poll_job(job)
    return {key: value for key, value in job.items() if key != "_process"}


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def validate_subset(values: list[str], allowed: set[str], label: str) -> list[str]:
    cleaned = [value.strip() for value in values if value and value.strip()]
    missing = sorted(set(cleaned) - allowed)
    if missing:
        raise ValueError(f"Unknown {label}: {', '.join(missing)}")
    return cleaned


def validate_int_range(value, default: int, minimum: int, maximum: int, label: str) -> int:
    parsed = int(value if value not in (None, "") else default)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def validate_device_profile(value: str | None) -> str:
    profile = str(value or "windows_pc_pre_jetson").strip()
    if not DEVICE_PROFILE_RE.match(profile):
        raise ValueError("device_profile may only contain letters, numbers, '_' and '-'.")
    return profile


def build_unified_runner_command(payload: dict, *, dry_run: bool = False) -> tuple[list[str], Path]:
    config = load_config()
    experiment = str(payload.get("experiment", "jetson_readiness_interface_smoke")).strip()
    if experiment not in UNIFIED_EXPERIMENTS:
        raise ValueError(f"Unsupported unified experiment: {experiment}")

    model_keys = validate_subset(
        [str(item) for item in payload.get("model_keys", payload.get("models", []))],
        {item["key"] for item in config["models"]},
        "model",
    )
    task_ids = validate_subset(
        [str(item) for item in payload.get("task_ids", payload.get("tasks", []))],
        {item["task_id"] for item in config["tasks"]},
        "task",
    )
    if not model_keys:
        raise ValueError("Select at least one unified model.")
    if not task_ids:
        raise ValueError("Select at least one unified task.")

    repeats = validate_int_range(payload.get("repeats"), 1, 1, 20, "repeats")
    warmup = validate_int_range(payload.get("warmup"), 0, 0, 5, "warmup")
    max_new_tokens = validate_int_range(payload.get("max_new_tokens"), 64, 1, 512, "max_new_tokens")
    device_profile = validate_device_profile(payload.get("device_profile"))

    cmd = [
        python_for(LOCAL_BENCH, ".venv"),
        "-m",
        "src.vla_bench.unified_runner",
        "--experiment",
        experiment,
        "--models",
        ",".join(model_keys),
        "--tasks",
        ",".join(task_ids),
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--max-new-tokens",
        str(max_new_tokens),
        "--device-profile",
        device_profile,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd, LOCAL_BENCH


def build_mujoco_viewer_command(payload: dict) -> tuple[list[str], Path]:
    config = load_config()
    scene_targets = {item["name"] for item in config["scene"].get("objects", [])}
    target = str(payload.get("target", "blue_can")).strip()
    validate_subset([target], scene_targets, "target")
    max_steps = validate_int_range(payload.get("max_steps"), 300, 50, 1000, "max_steps")
    viewer_hold_sec = validate_int_range(payload.get("viewer_hold_sec"), 20, 0, 120, "viewer_hold_sec")

    cmd = [
        python_for(SIM_ROOT, ".sim_venv"),
        "-m",
        "src.robot_sim.scripted_pick_benchmark",
        "--targets",
        target,
        "--trials",
        "1",
        "--max-steps",
        str(max_steps),
        "--include-types",
        "all",
        "--save-final-images",
        "--save-frame-sequence",
        "--frame-stride",
        "5",
        "--viewer",
        "--viewer-hold-sec",
        str(viewer_hold_sec),
    ]
    return cmd, SIM_ROOT


def dry_run_job(payload: dict) -> dict:
    kind = str(payload.get("kind", ""))
    if kind != "unified_runner":
        raise ValueError(f"Unsupported dry-run kind: {kind}")
    cmd, cwd = build_unified_runner_command(payload, dry_run=True)
    process = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    parsed = {}
    if process.stdout.strip():
        try:
            parsed = json.loads(process.stdout)
        except json.JSONDecodeError:
            parsed = {}
    return {
        "kind": kind,
        "cmd": cmd,
        "cwd": str(cwd),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "dry_run": parsed,
    }


def create_job(payload: dict) -> dict:
    config = load_config()
    kind = str(payload.get("kind", ""))
    job_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"

    if kind == "unified_runner":
        cmd, cwd = build_unified_runner_command(payload)
    elif kind == "mujoco_viewer_debug":
        cmd, cwd = build_mujoco_viewer_command(payload)
    elif kind == "pre_jetson":
        model_keys = validate_subset(
            [str(payload.get("model_key", ""))],
            {item["key"] for item in config["models"]},
            "model",
        )
        task_ids = validate_subset(
            [str(item) for item in payload.get("task_ids", [])],
            {item["task_id"] for item in config["tasks"]},
            "task",
        )
        if not task_ids:
            raise ValueError("Select at least one task.")
        cmd = [
            python_for(LOCAL_BENCH, ".venv"),
            "-m",
            "src.vla_bench.pre_jetson_runner",
            "--experiment",
            "offline_vlm_smoke",
            "--models",
            ",".join(model_keys),
            "--tasks",
            ",".join(task_ids),
        ]
        cwd = LOCAL_BENCH
    elif kind == "scripted_pick":
        scene_types = {item["type"] for item in config["scene"].get("objects", [])}
        include_types = str(payload.get("include_types", "box,cylinder"))
        if include_types != "all":
            validate_subset(include_types.split(","), scene_types, "object type")
        trials = int(payload.get("trials", 5))
        max_steps = int(payload.get("max_steps", 500))
        save_frame_sequence = bool(payload.get("save_frame_sequence", True))
        frame_stride = int(payload.get("frame_stride", 10))
        if not 1 <= trials <= 100:
            raise ValueError("trials must be between 1 and 100.")
        if not 50 <= max_steps <= 5000:
            raise ValueError("max_steps must be between 50 and 5000.")
        if not 1 <= frame_stride <= 200:
            raise ValueError("frame_stride must be between 1 and 200.")
        cmd = [
            python_for(SIM_ROOT, ".sim_venv"),
            "-m",
            "src.robot_sim.scripted_pick_benchmark",
            "--targets",
            "all",
            "--trials",
            str(trials),
            "--max-steps",
            str(max_steps),
            "--save-final-images",
        ]
        if save_frame_sequence:
            cmd.extend(["--save-frame-sequence", "--frame-stride", str(frame_stride)])
        if include_types != "all":
            cmd.extend(["--include-types", include_types])
        cwd = SIM_ROOT
    elif kind == "vlm_pick":
        model_keys = validate_subset(
            [str(payload.get("model_key", ""))],
            {item["key"] for item in config["models"]},
            "model",
        )
        selected_model = next(item for item in config["models"] if item["key"] == model_keys[0])
        if selected_model.get("model_type") == "rule_baseline":
            raise ValueError("VLA robot benchmark requires a real visual model, not local_rule_baseline.")
        scene_types = {item["type"] for item in config["scene"].get("objects", [])}
        include_types = str(payload.get("include_types", "box,cylinder"))
        if include_types != "all":
            validate_subset(include_types.split(","), scene_types, "object type")
        trials = int(payload.get("trials", 1))
        max_steps = int(payload.get("max_steps", 500))
        save_frame_sequence = bool(payload.get("save_frame_sequence", True))
        frame_stride = int(payload.get("frame_stride", 10))
        port = 8100 + (int(uuid.uuid4().hex[:4], 16) % 800)
        if not 1 <= trials <= 20:
            raise ValueError("trials must be between 1 and 20 for VLM pick benchmark.")
        if not 50 <= max_steps <= 5000:
            raise ValueError("max_steps must be between 50 and 5000.")
        cmd = [
            str(Path(sys.executable)),
            str(UI_ROOT / "run_vlm_pick_job.py"),
            "--model-key",
            ",".join(model_keys),
            "--targets",
            "all",
            "--trials",
            str(trials),
            "--max-steps",
            str(max_steps),
            "--include-types",
            include_types,
            "--port",
            str(port),
            "--frame-stride",
            str(frame_stride),
        ]
        if save_frame_sequence:
            cmd.append("--save-frame-sequence")
        cwd = UI_ROOT
    elif kind == "pc_matrix":
        allowed_models = {item["key"] for item in config["models"]} & CORE_MATRIX_MODELS
        model_keys = validate_subset(
            [str(item) for item in payload.get("model_keys", [])],
            allowed_models,
            "model",
        )
        if not model_keys:
            raise ValueError("Select at least one VLA/VLM model for the matrix.")
        model_map = {item["key"]: item for item in config["models"]}
        rule_models = [key for key in model_keys if model_map[key].get("model_type") == "rule_baseline"]
        if rule_models:
            raise ValueError("PC matrix VLA runs require real visual models, not local_rule_baseline.")
        scene_types = {item["type"] for item in config["scene"].get("objects", [])}
        include_types = str(payload.get("include_types", "box,cylinder"))
        if include_types != "all":
            validate_subset(include_types.split(","), scene_types, "object type")
        targets = str(payload.get("targets", "all"))
        if targets != "all":
            scene_targets = {item["name"] for item in config["scene"].get("objects", [])}
            validate_subset(targets.split(","), scene_targets, "target")
        trials = int(payload.get("trials", 5))
        max_steps = int(payload.get("max_steps", 500))
        save_frame_sequence = bool(payload.get("save_frame_sequence", True))
        frame_stride = int(payload.get("frame_stride", 20))
        skip_scripted = bool(payload.get("skip_scripted", False))
        if not 1 <= trials <= 10:
            raise ValueError("trials must be between 1 and 10 for PC matrix benchmark.")
        if not 50 <= max_steps <= 5000:
            raise ValueError("max_steps must be between 50 and 5000.")
        if not 1 <= frame_stride <= 200:
            raise ValueError("frame_stride must be between 1 and 200.")
        cmd = [
            str(Path(sys.executable)),
            str(UI_ROOT / "run_pc_benchmark_matrix.py"),
            "--models",
            ",".join(model_keys),
            "--targets",
            targets,
            "--include-types",
            include_types,
            "--trials",
            str(trials),
            "--max-steps",
            str(max_steps),
            "--frame-stride",
            str(frame_stride),
        ]
        if save_frame_sequence:
            cmd.append("--save-frame-sequence")
        if skip_scripted:
            cmd.append("--skip-scripted")
        cwd = UI_ROOT
    elif kind == "tiny_bc_closed_loop":
        policies = {item["id"]: item for item in list_trained_policies()}
        policy_id = str(payload.get("policy_id", ""))
        if policy_id not in policies:
            raise ValueError(f"Unknown trained policy: {policy_id}")
        scene_targets = {item["name"] for item in config["scene"].get("objects", [])}
        targets = str(payload.get("targets", "blue_can,red_block"))
        if targets != "all":
            validate_subset(targets.split(","), scene_targets, "target")
        trials = int(payload.get("trials", 2))
        max_steps = int(payload.get("max_steps", 220))
        frame_stride = int(payload.get("frame_stride", 10))
        xy_jitter = float(payload.get("xy_jitter", 0.025))
        target_z_biases = str(payload.get("target_z_biases", "red_block:-0.01"))
        phase_gripper_guard = bool(payload.get("phase_gripper_guard", True))
        save_frame_sequence = bool(payload.get("save_frame_sequence", True))
        stop_on_success = bool(payload.get("stop_on_success", True))
        if not 1 <= trials <= 20:
            raise ValueError("trials must be between 1 and 20 for Tiny BC closed-loop evaluation.")
        if not 50 <= max_steps <= 1000:
            raise ValueError("max_steps must be between 50 and 1000.")
        if not 1 <= frame_stride <= 200:
            raise ValueError("frame_stride must be between 1 and 200.")
        cmd = [
            python_for(SIM_ROOT, ".sim_venv"),
            "-m",
            "src.robot_sim.eval_tiny_bc_closed_loop",
            "--policy",
            str(Path(policies[policy_id]["path"]).relative_to(SIM_ROOT)),
            "--targets",
            targets,
            "--trials-per-target",
            str(trials),
            "--xy-jitter",
            str(xy_jitter),
            "--frame-stride",
            str(frame_stride),
            "--max-steps",
            str(max_steps),
        ]
        if save_frame_sequence:
            cmd.append("--save-frame-sequence")
        if phase_gripper_guard:
            cmd.append("--phase-gripper-guard")
        if stop_on_success:
            cmd.append("--stop-on-success")
        if target_z_biases:
            cmd.extend(["--target-z-biases", target_z_biases])
        cwd = SIM_ROOT
    else:
        raise ValueError(f"Unsupported job kind: {kind}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    job = {
        "id": job_id,
        "kind": kind,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(cwd),
        "cmd": cmd,
        "log_path": str(log_path),
        "return_code": None,
        "debug_mode": "mujoco_viewer" if kind == "mujoco_viewer_debug" else "",
        "log_tail": "",
        "_process": process,
    }
    JOBS[job_id] = job
    return public_job(job)


def resolve_safe_path(value: str) -> Path:
    candidate = Path(unquote(value))
    if not candidate.is_absolute():
        candidate = FINAL_ROOT / candidate
    resolved = candidate.resolve()
    allowed_roots = [FINAL_ROOT.resolve(), WORKSPACE_ROOT.resolve()]
    if not any(str(resolved).lower().startswith(str(root).lower()) for root in allowed_roots):
        raise ValueError(f"Path outside workspace: {resolved}")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


class ConsoleBadRequest(ValueError):
    pass


class ConsoleConflict(ValueError):
    pass


def parse_console_json_body(handler: BaseHTTPRequestHandler) -> dict:
    payload = parse_json_body(handler)
    if not isinstance(payload, dict):
        raise ConsoleBadRequest("console payload must be an object")
    if _has_unsafe_console_key(payload):
        raise ConsoleBadRequest("console payload contains a forbidden process-control key")
    return payload


def _has_unsafe_console_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in UNSAFE_CONSOLE_PAYLOAD_KEYS or _has_unsafe_console_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_unsafe_console_key(item) for item in value)
    return False


def refresh_console_catalog(services: ConsoleServices) -> None:
    services.catalog["tasks"] = services.task_store.list_versions()
    services.orchestrator.catalog = deepcopy(services.catalog)


def validate_console_spec(services: ConsoleServices, payload: dict) -> dict:
    try:
        return services.orchestrator.validate(payload)
    except SpecValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise SpecValidationError([
            {"field": "spec", "message": str(exc) or "malformed experiment specification"}
        ]) from exc


def read_console_result(services: ConsoleServices, run_id: str) -> dict:
    manifest = services.orchestrator.get_run(run_id)
    if manifest.get("executor_key") != "batch":
        return inspect_result(manifest)
    children = []
    for child_id in manifest.get("child_run_ids", []):
        try:
            children.append(services.orchestrator.get_run(str(child_id)))
        except FileNotFoundError:
            continue
    return normalize_batch(manifest, children)


def resolve_console_preview(services: ConsoleServices, preview_id: str) -> Path:
    if PREVIEW_ID_RE.fullmatch(preview_id) is None:
        raise ConsoleBadRequest("preview id must be a SHA-256 hash")
    root = Path(services.preview_root).resolve()
    path = root / preview_id / "preview.png"
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError("preview not found") from None
    if root not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError("preview not found")
    return resolved


def resolve_console_artifact(services: ConsoleServices, run_id: str, artifact_id: str) -> Path:
    result = read_console_result(services, run_id)
    artifact = next(
        (
            item for item in result.get("artifacts", [])
            if item.get("artifact_id") == artifact_id and item.get("exists") is True
        ),
        None,
    )
    if artifact is None:
        raise FileNotFoundError("artifact not found")
    try:
        path = Path(str(artifact["path"])).resolve(strict=True)
    except (KeyError, OSError):
        raise FileNotFoundError("artifact not found") from None
    if not path.is_file():
        raise FileNotFoundError("artifact not found")
    return path


def console_environment_is_renderable(catalog: dict, spec: dict) -> bool:
    environment = spec.get("environment") if isinstance(spec.get("environment"), dict) else {}
    key = str(environment.get("key", ""))
    row = next((item for item in catalog.get("environments", []) if item.get("key") == key), None)
    return bool(row and row.get("viewer"))


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_file(UI_ROOT / "static" / "core.html")
            elif parsed.path.startswith("/static/"):
                self.send_file(UI_ROOT / parsed.path.lstrip("/"))
            elif parsed.path == "/api/console/catalog":
                services = get_console_services()
                self.send_json({"catalog": services.catalog})
            elif parsed.path == "/api/console/runs":
                services = get_console_services()
                self.send_json({"runs": services.orchestrator.list_runs()})
            elif parsed.path == "/api/console/run":
                services = get_console_services()
                run_id = parse_qs(parsed.query).get("id", [""])[0]
                if not run_id:
                    raise ConsoleBadRequest("missing run id")
                manifest = services.orchestrator.get_run(run_id)
                poll_run = getattr(services.orchestrator, "poll", None)
                if manifest.get("status") in {"queued", "running"} and callable(poll_run):
                    poll_run(run_id)
                detail_reader = getattr(services.orchestrator, "get_run_detail", services.orchestrator.get_run)
                self.send_json({"run": detail_reader(run_id)})
            elif parsed.path == "/api/console/results":
                services = get_console_services()
                run_id = parse_qs(parsed.query).get("id", [""])[0]
                if not run_id:
                    raise ConsoleBadRequest("missing run id")
                self.send_json({"result": read_console_result(services, run_id)})
            elif parsed.path == "/api/console/artifact":
                services = get_console_services()
                params = parse_qs(parsed.query)
                run_id = params.get("run_id", [""])[0]
                artifact_id = params.get("artifact_id", [""])[0]
                if not run_id or not artifact_id:
                    raise ConsoleBadRequest("missing artifact identifiers")
                self.send_file(resolve_console_artifact(services, run_id, artifact_id))
            elif parsed.path == "/api/console/preview":
                services = get_console_services()
                preview_id = parse_qs(parsed.query).get("id", [""])[0]
                self.send_file(resolve_console_preview(services, preview_id))
            elif parsed.path == "/api/config":
                self.send_json(load_config())
            elif parsed.path == "/api/results":
                self.send_json(
                    {
                        "scripted_runs": list_scripted_runs(),
                        "vlm_pick_runs": list_vlm_pick_runs(),
                        "pre_jetson_runs": list_pre_jetson_runs(),
                        "unified_runs": list_unified_runs(),
                        "pc_matrix_runs": list_pc_matrix_runs(),
                        "tiny_bc_runs": list_tiny_bc_runs(),
                        "visual_target_bc_runs": list_visual_target_bc_runs(),
                    }
                )
            elif parsed.path == "/api/results/unified":
                self.send_json({"unified_runs": list_unified_runs()})
            elif parsed.path == "/api/evidence":
                self.send_json(load_evidence())
            elif parsed.path == "/api/jobs":
                self.send_json({"jobs": [public_job(job) for job in JOBS.values()]})
            elif parsed.path == "/api/job":
                params = parse_qs(parsed.query)
                job_id = params.get("id", [""])[0]
                if job_id not in JOBS:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown job")
                    return
                self.send_json(public_job(JOBS[job_id]))
            elif parsed.path == "/api/file":
                params = parse_qs(parsed.query)
                if "path" not in params:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
                    return
                self.send_file(resolve_safe_path(params["path"][0]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError as error:
            if parsed.path.startswith("/api/console/"):
                self.send_json({"error": "not_found", "message": str(error)}, status=HTTPStatus.NOT_FOUND)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, str(error))
        except ValueError as error:
            if parsed.path.startswith("/api/console/"):
                self.send_json({"error": "bad_request", "message": str(error)}, status=HTTPStatus.BAD_REQUEST)
            else:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # Keep the dashboard available after a bad file.
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/start":
                self.send_json(create_job(parse_json_body(self)), status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/dry-run":
                self.send_json(dry_run_job(parse_json_body(self)))
            elif parsed.path == "/api/console/tasks":
                services = get_console_services()
                created_task = services.task_store.create_version(parse_console_json_body(self))
                refresh_console_catalog(services)
                self.send_json(
                    {"task": created_task},
                    status=HTTPStatus.CREATED,
                )
            elif parsed.path == "/api/console/validate":
                services = get_console_services()
                spec = validate_console_spec(services, parse_console_json_body(self))
                self.send_json({"spec": spec, "preflight": services.orchestrator.preflight(spec)})
            elif parsed.path == "/api/console/preview":
                services = get_console_services()
                spec = validate_console_spec(services, parse_console_json_body(self))
                if not console_environment_is_renderable(services.catalog, spec):
                    self.send_json(
                        {"error": "preview_unavailable", "message": "selected environment has no renderable scene"},
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                self.send_json(render_scene_preview(spec, services.catalog, services.paths, services.preview_root))
            elif parsed.path == "/api/console/runs":
                services = get_console_services()
                spec = validate_console_spec(services, parse_console_json_body(self))
                selection = spec.get("model_selection") or {}
                run = (
                    services.orchestrator.start_batch(spec)
                    if selection.get("mode") == "batch"
                    else services.orchestrator.start(spec)
                )
                self.send_json({"run": run}, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/console/stop":
                services = get_console_services()
                payload = parse_console_json_body(self)
                run_id = str(payload.get("run_id", ""))
                if not run_id:
                    raise ConsoleBadRequest("missing run id")
                self.send_json({"run": services.orchestrator.stop(run_id)}, status=HTTPStatus.ACCEPTED)
            elif parsed.path == "/api/console/viewer/open":
                services = get_console_services()
                payload = parse_console_json_body(self)
                run_id = str(payload.get("run_id", ""))
                if not run_id:
                    raise ConsoleBadRequest("missing run id")
                run = services.orchestrator.get_run(run_id)
                viewer = run.get("viewer")
                viewer_unavailable = isinstance(viewer, dict) and viewer.get("available") is False
                if run.get("status") in {"completed", "failed", "stopped"} or viewer_unavailable:
                    raise ConsoleConflict("viewer is unavailable for this run")
                self.send_json({"viewer": services.orchestrator.open_viewer(run_id)})
            elif parsed.path == "/api/console/viewer/close":
                services = get_console_services()
                payload = parse_console_json_body(self)
                run_id = str(payload.get("run_id", ""))
                if not run_id:
                    raise ConsoleBadRequest("missing run id")
                self.send_json({"viewer": services.orchestrator.close_viewer(run_id)})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except SpecValidationError as error:
            self.send_json(
                {"error": "validation_failed", "fields": error.errors},
                status=HTTPStatus.BAD_REQUEST,
            )
        except FileNotFoundError as error:
            if parsed.path.startswith("/api/console/"):
                self.send_json({"error": "not_found", "message": str(error)}, status=HTTPStatus.NOT_FOUND)
            else:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except ConsoleBadRequest as error:
            self.send_json({"error": "bad_request", "message": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except ConsoleConflict as error:
            self.send_json({"error": "conflict", "message": str(error)}, status=HTTPStatus.CONFLICT)
        except ValueError as error:
            if parsed.path == "/api/console/tasks":
                self.send_json({"error": "bad_request", "message": str(error)}, status=HTTPStatus.BAD_REQUEST)
            elif parsed.path.startswith("/api/console/"):
                message = str(error)
                status = HTTPStatus.NOT_FOUND if "unknown run" in message or "run not found" in message else HTTPStatus.CONFLICT
                self.send_json(
                    {"error": "not_found" if status == HTTPStatus.NOT_FOUND else "conflict", "message": message},
                    status=status,
                )
            else:
                self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_file(self, path: Path) -> None:
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        if path.suffix.lower() in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[Benchmark UI] {self.address_string()} - {fmt % args}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the local VLA benchmark dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    get_console_services()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"VLA Benchmark UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
