"""RGB open-vocabulary detection for a LIBERO initial scene."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_label(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def target_detected(target: str, detections: list[dict[str, Any]]) -> bool:
    wanted = normalize_label(target)
    return any(normalize_label(str(item.get("label", ""))) == wanted for item in detections)


def capture_libero_frame(task_id: int, output_file: Path, seed: int) -> None:
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id 超出范围：{task_id}")
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=360, camera_widths=360)
    try:
        env.seed(seed)
        observation = env.reset()
        states = suite.get_task_init_states(task_id)
        state_result = env.set_init_state(states[seed % len(states)])
        if isinstance(state_result, dict):
            observation = state_result
        if not isinstance(observation, dict) or "agentview_image" not in observation:
            observation = env.env._get_observations()
        image = np.asarray(observation["agentview_image"], dtype=np.uint8)
        image = image[::-1, ::-1]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_file)
    finally:
        env.close()


def detect_image(
    image_file: Path,
    labels: list[str],
    output_dir: Path,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    threshold: float = 0.20,
) -> list[dict[str, Any]]:
    try:
        import torch
        from PIL import Image, ImageDraw
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "视觉检测依赖缺失；请在 LeRobot 环境运行：uv pip install transformers pillow"
        ) from exc
    if not 0.0 < threshold < 1.0:
        raise ValueError("视觉检测阈值必须在 0 和 1 之间")
    if not labels:
        raise ValueError("候选标签不能为空")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_file).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    inputs = processor(images=image, text=[labels], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    detections: list[dict[str, Any]] = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        name = str(label)
        if name.isdigit() and int(name) < len(labels):
            name = labels[int(name)]
        detections.append({
            "label": name,
            "score": round(float(score), 4),
            "box": [round(float(value), 1) for value in box.tolist()],
        })

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 3, max(0, y1 - 14)), f"{item['label']} {item['score']:.2f}", fill="red")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated.save(output_dir / "detected_scene.png")
    (output_dir / "detections.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del model, inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detections


def detect_libero_scene(
    task_id: int,
    labels: list[str],
    output_dir: Path,
    seed: int,
    model_id: str,
    threshold: float,
) -> list[dict[str, Any]]:
    frame = output_dir / "initial_scene.png"
    capture_libero_frame(task_id, frame, seed)
    return detect_image(frame, labels, output_dir, model_id=model_id, threshold=threshold)
