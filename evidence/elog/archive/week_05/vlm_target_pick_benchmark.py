from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cartesian_pick_policy import CartesianPickPolicy, PickControlConfig
from .common import OUTPUT_DIR, configure_windows_mujoco, get_camera_image, save_image
from .editable_tabletop_env import EditableTabletopEnv
from .passive_viewer_bridge import PassiveViewerBridge
from .scene_config import load_scene_config
from .target_resolver import resolve_target_from_instruction
from .vlm_client import request_vlm_target


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "scene_config.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a VLM-target-selection to Cartesian-pick robot execution loop."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/infer")
    parser.add_argument("--vla-model", default="qwen2_vl_2b")
    parser.add_argument("--targets", default="all")
    parser.add_argument("--include-types", default="box,cylinder")
    parser.add_argument("--exclude-types", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--success-lift-threshold", type=float, default=0.06)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--save-final-images", action="store_true")
    parser.add_argument("--save-frame-sequence", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--allow-instruction-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--viewer-control-path", type=Path)
    parser.add_argument("--viewer-status-path", type=Path)
    return parser.parse_args(argv)


def _split_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def select_targets(scene_config: dict[str, Any], targets_arg: str, include_types_arg: str, exclude_types_arg: str) -> list[str]:
    include_types = None if include_types_arg.strip().lower() == "all" else _split_csv(include_types_arg)
    exclude_types = _split_csv(exclude_types_arg)
    filtered_specs = []
    for spec in scene_config["objects"]:
        object_type = spec["type"]
        if include_types is not None and object_type not in include_types:
            continue
        if object_type in exclude_types:
            continue
        filtered_specs.append(spec)

    available = [spec["name"] for spec in filtered_specs]
    if targets_arg.strip().lower() == "all":
        return available
    requested = [item.strip() for item in targets_arg.split(",") if item.strip()]
    all_names = {spec["name"] for spec in scene_config["objects"]}
    missing = sorted(set(requested) - all_names)
    if missing:
        raise ValueError(f"Unknown target(s): {', '.join(missing)}. Available: {', '.join(sorted(all_names))}")
    filtered_out = sorted(set(requested) - set(available))
    if filtered_out:
        raise ValueError(
            "Requested target(s) were filtered out by --include-types/--exclude-types: "
            f"{', '.join(filtered_out)}"
        )
    return requested


def instruction_for_target(target_id: str, object_specs: list[dict[str, Any]]) -> str:
    spec = next(item for item in object_specs if item["name"] == target_id)
    name = spec.get("aliases", [target_id.replace("_", " ")])[0]
    return f"Pick up the {name}."


def make_control_config(scene_config: dict[str, Any]) -> PickControlConfig:
    values = scene_config.get("pick_control", {})
    return PickControlConfig(**{key: values[key] for key in PickControlConfig.__dataclass_fields__ if key in values})


def classify_failure(
    *,
    vlm_selected_target: str | None,
    requested_target: str,
    lifted_requested: bool,
    lifted_selected: bool,
    policy_complete: bool,
    steps: int,
    max_steps: int,
) -> str:
    if lifted_requested:
        return "none"
    if vlm_selected_target is None:
        return "vlm_no_target"
    if vlm_selected_target != requested_target:
        return "vlm_wrong_target"
    if lifted_selected:
        return "wrong_success_check"
    if not policy_complete and steps >= max_steps:
        return "timeout"
    if policy_complete:
        return "grasp_or_lift_failed"
    return "control_failed"


def run_trial(
    *,
    scene_config: dict[str, Any],
    requested_target: str,
    trial_index: int,
    run_dir: Path,
    endpoint: str,
    vla_model: str,
    max_steps: int,
    success_lift_threshold: float,
    camera_width: int,
    camera_height: int,
    save_final_images: bool,
    save_frame_sequence: bool,
    frame_stride: int,
    allow_instruction_fallback: bool,
    viewer_control_path: Path | None = None,
    viewer_status_path: Path | None = None,
) -> dict[str, Any]:
    configure_windows_mujoco()
    if (viewer_control_path is None) != (viewer_status_path is None):
        raise ValueError("--viewer-control-path and --viewer-status-path must be supplied together")
    env = EditableTabletopEnv(
        scene_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_width=camera_width,
        camera_height=camera_height,
        horizon=max_steps + 20,
    )
    observation = env.reset()
    viewer_bridge = None
    if viewer_control_path is not None and viewer_status_path is not None:
        viewer_bridge = PassiveViewerBridge(
            model=env.sim.model._model,
            data=env.sim.data._data,
            control_path=viewer_control_path,
            status_path=viewer_status_path,
        )
    start = time.perf_counter()
    try:
        object_specs = scene_config["objects"]
        instruction = instruction_for_target(requested_target, object_specs)
        candidate_targets = [spec["name"] for spec in object_specs]
        initial_state = env.scene_state()
        initial_requested_height = float(initial_state[requested_target][2])
        frame = get_camera_image(observation)
        initial_frame_path = ""
        if save_frame_sequence:
            initial_frame_path = str(run_dir / f"{requested_target}_trial_{trial_index:03d}_step_0000_initial.jpg")
            save_image(frame, Path(initial_frame_path))

        vlm_result = request_vlm_target(frame, instruction, candidate_targets, endpoint=endpoint)
        selected_target = vlm_result.get("target_id")
        target_source = "vlm"
        if selected_target is None and allow_instruction_fallback:
            selected_target = resolve_target_from_instruction(instruction, object_specs)
            target_source = "instruction_fallback"

        if selected_target is None:
            elapsed_sec = time.perf_counter() - start
            return {
                "vla_model": vla_model,
                "controller_model": "VLMTargetAdapter+CartesianPickPolicy",
                "requested_target": requested_target,
                "selected_target": "",
                "target_source": "none",
                "trial_index": trial_index,
                "instruction": instruction,
                "success": False,
                "target_match": False,
                "failure_type": "vlm_no_target",
                "steps": 0,
                "elapsed_sec": elapsed_sec,
                "vlm_latency_sec": vlm_result.get("latency_sec"),
                "vlm_raw_output": vlm_result.get("raw_output", ""),
                "phase_changes": [],
                "frame_paths": [{"step": 0, "phase": "initial", "path": initial_frame_path}] if initial_frame_path else [],
                "final_image_path": "",
            }

        selected_initial_height = float(initial_state[selected_target][2])
        requested_max_height = initial_requested_height
        selected_max_height = selected_initial_height
        policy = CartesianPickPolicy(np.asarray(initial_state[selected_target], dtype=np.float64), make_control_config(scene_config))
        previous_phase = policy.phase
        phase_changes = [{"step": 0, "phase": previous_phase}]
        frame_paths: list[dict[str, Any]] = []
        if initial_frame_path:
            frame_paths.append({"step": 0, "phase": "initial", "path": initial_frame_path})

        def save_frame(label: str, step_value: int) -> None:
            if not save_frame_sequence:
                return
            image = get_camera_image(observation)
            path = run_dir / f"{requested_target}_trial_{trial_index:03d}_step_{step_value:04d}_{label}.jpg"
            save_image(image, path)
            frame_paths.append({"step": step_value, "phase": label, "path": str(path)})

        steps_executed = 0
        for step in range(max_steps):
            action = policy.next_action(env, observation)
            observation, _, _, _ = env.step(action)
            if viewer_bridge is not None:
                viewer_bridge.poll()
                viewer_bridge.sync()
            steps_executed = step + 1
            current_state = env.scene_state()
            requested_max_height = max(requested_max_height, float(current_state[requested_target][2]))
            selected_max_height = max(selected_max_height, float(current_state[selected_target][2]))
            if policy.phase != previous_phase:
                previous_phase = policy.phase
                phase_changes.append({"step": steps_executed, "phase": previous_phase})
                save_frame(previous_phase, steps_executed)
            elif frame_stride > 0 and steps_executed % frame_stride == 0:
                save_frame(policy.phase, steps_executed)
            if policy.is_complete:
                break

        final_state = env.scene_state()
        requested_height_gain = float(final_state[requested_target][2]) - initial_requested_height
        requested_max_height_gain = requested_max_height - initial_requested_height
        selected_height_gain = float(final_state[selected_target][2]) - selected_initial_height
        selected_max_height_gain = selected_max_height - selected_initial_height
        lifted_requested = requested_max_height_gain >= success_lift_threshold
        lifted_selected = selected_max_height_gain >= success_lift_threshold
        elapsed_sec = time.perf_counter() - start

        image_path = ""
        if save_final_images:
            image_path = str(run_dir / f"{requested_target}_trial_{trial_index:03d}_final.jpg")
            save_image(get_camera_image(observation), Path(image_path))
        if save_frame_sequence:
            save_frame("final", steps_executed)

        return {
            "vla_model": vla_model,
            "controller_model": "VLMTargetAdapter+CartesianPickPolicy",
            "requested_target": requested_target,
            "selected_target": selected_target,
            "target_source": target_source,
            "trial_index": trial_index,
            "instruction": instruction,
            "success": lifted_requested,
            "target_match": selected_target == requested_target,
            "failure_type": classify_failure(
                vlm_selected_target=selected_target,
                requested_target=requested_target,
                lifted_requested=lifted_requested,
                lifted_selected=lifted_selected,
                policy_complete=policy.is_complete,
                steps=steps_executed,
                max_steps=max_steps,
            ),
            "steps": steps_executed,
            "elapsed_sec": elapsed_sec,
            "vlm_latency_sec": vlm_result.get("latency_sec"),
            "vlm_raw_output": vlm_result.get("raw_output", ""),
            "policy_complete": policy.is_complete,
            "requested_initial_height": initial_requested_height,
            "requested_final_height": float(final_state[requested_target][2]),
            "requested_height_gain": requested_height_gain,
            "requested_max_height_gain": requested_max_height_gain,
            "selected_initial_height": selected_initial_height,
            "selected_final_height": float(final_state[selected_target][2]),
            "selected_height_gain": selected_height_gain,
            "selected_max_height_gain": selected_max_height_gain,
            "success_lift_threshold": success_lift_threshold,
            "phase_changes": phase_changes,
            "frame_paths": frame_paths,
            "final_image_path": image_path,
        }
    finally:
        if viewer_bridge is not None:
            viewer_bridge.close()
        env.close()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["requested_target"], []).append(row)

    summary_rows = []
    for target_id, items in sorted(by_target.items()):
        successes = [bool(item["success"]) for item in items]
        target_matches = [bool(item.get("target_match")) for item in items]
        steps = [int(item["steps"]) for item in items]
        elapsed = [float(item["elapsed_sec"]) for item in items]
        vlm_latency = [float(item["vlm_latency_sec"]) for item in items if item.get("vlm_latency_sec") not in (None, "")]
        summary_rows.append(
            {
                "vla_model": str(items[0].get("vla_model", "")),
                "controller_model": str(items[0].get("controller_model", "")),
                "requested_target": target_id,
                "trials": len(items),
                "success_count": sum(successes),
                "task_success_rate": sum(successes) / len(successes) if successes else 0.0,
                "target_match_rate": sum(target_matches) / len(target_matches) if target_matches else 0.0,
                "mean_steps": sum(steps) / len(steps) if steps else None,
                "mean_elapsed_sec": sum(elapsed) / len(elapsed) if elapsed else None,
                "mean_vlm_latency_sec": sum(vlm_latency) / len(vlm_latency) if vlm_latency else None,
                "failure_types": "|".join(sorted({str(item["failure_type"]) for item in items})),
            }
        )

    if rows:
        successes = [bool(row["success"]) for row in rows]
        target_matches = [bool(row.get("target_match")) for row in rows]
        steps = [int(row["steps"]) for row in rows]
        elapsed = [float(row["elapsed_sec"]) for row in rows]
        vlm_latency = [float(row["vlm_latency_sec"]) for row in rows if row.get("vlm_latency_sec") not in (None, "")]
        summary_rows.append(
            {
                "vla_model": str(rows[0].get("vla_model", "")),
                "controller_model": str(rows[0].get("controller_model", "")),
                "requested_target": "ALL",
                "trials": len(rows),
                "success_count": sum(successes),
                "task_success_rate": sum(successes) / len(successes),
                "target_match_rate": sum(target_matches) / len(target_matches),
                "mean_steps": sum(steps) / len(steps),
                "mean_elapsed_sec": sum(elapsed) / len(elapsed),
                "mean_vlm_latency_sec": sum(vlm_latency) / len(vlm_latency) if vlm_latency else None,
                "failure_types": "|".join(sorted({str(row["failure_type"]) for row in rows})),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scene_config = load_scene_config(args.config)
    targets = select_targets(scene_config, args.targets, args.include_types, args.exclude_types)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": str(args.config),
                    "endpoint": args.endpoint,
                    "vla_model": args.vla_model,
                    "targets": targets,
                    "trials_per_target": args.trials,
                    "max_steps": args.max_steps,
                },
                indent=2,
            )
        )
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "vlm_target_pick_benchmark" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "trials.jsonl"
    trials_csv_path = run_dir / "trials.csv"
    summary_csv_path = run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"

    metadata = {
        "run_id": run_id,
        "config": str(args.config),
        "endpoint": args.endpoint,
        "vla_model": args.vla_model,
        "controller_model": "VLMTargetAdapter+CartesianPickPolicy",
        "targets": targets,
        "include_types": args.include_types,
        "exclude_types": args.exclude_types,
        "trials_per_target": args.trials,
        "max_steps": args.max_steps,
        "success_lift_threshold": args.success_lift_threshold,
        "save_frame_sequence": args.save_frame_sequence,
        "frame_stride": args.frame_stride,
        "allow_instruction_fallback": args.allow_instruction_fallback,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for requested_target in targets:
            for trial_index in range(args.trials):
                row = run_trial(
                    scene_config=scene_config,
                    requested_target=requested_target,
                    trial_index=trial_index,
                    run_dir=run_dir,
                    endpoint=args.endpoint,
                    vla_model=args.vla_model,
                    max_steps=args.max_steps,
                    success_lift_threshold=args.success_lift_threshold,
                    camera_width=args.camera_width,
                    camera_height=args.camera_height,
                    save_final_images=args.save_final_images,
                    save_frame_sequence=args.save_frame_sequence,
                    frame_stride=args.frame_stride,
                    allow_instruction_fallback=args.allow_instruction_fallback,
                    viewer_control_path=args.viewer_control_path,
                    viewer_status_path=args.viewer_status_path,
                )
                rows.append(row)
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_file.flush()
                print(
                    f"{requested_target} trial {trial_index + 1}/{args.trials}: "
                    f"selected={row['selected_target'] or 'none'} success={row['success']} "
                    f"steps={row['steps']} failure={row['failure_type']}"
                )

    summary_rows = summarize(rows)
    write_csv(trials_csv_path, rows)
    write_csv(summary_csv_path, summary_rows)
    print(json.dumps({"run_dir": str(run_dir), "summary_csv": str(summary_csv_path)}, indent=2))


if __name__ == "__main__":
    main()
