from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from .action_schema import build_action_prompt


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model(model_cfg: dict[str, Any], *, local_files_only: bool = False):
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

    model_id = model_cfg["model_id"]
    dtype = resolve_dtype(str(model_cfg.get("dtype", "auto")))
    common_kwargs = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "local_files_only": local_files_only,
    }
    model_kwargs = {
        **common_kwargs,
        "device_map": model_cfg.get("device_map", "auto"),
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    processor = AutoProcessor.from_pretrained(model_id, **common_kwargs)
    model_type = model_cfg.get("model_type", "image_text_to_text")
    if model_type == "vision2seq":
        if AutoModelForVision2Seq is None:
            raise ImportError(
                "AutoModelForVision2Seq is unavailable. "
                "Please upgrade transformers: pip install -U transformers"
            )
        model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    elif model_type == "image_text_to_text":
        if AutoModelForImageTextToText is not None:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        elif AutoModelForMultimodalLM is not None:
            model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
        elif AutoModelForVision2Seq is not None:
            model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
        else:
            raise ImportError(
                "This transformers version does not provide a compatible multimodal model class. "
                "Please upgrade transformers: pip install -U transformers"
            )
    else:
        raise ValueError(
            f"Model type '{model_type}' needs a custom loader. "
            "Add it after confirming the model's official inference API."
        )
    model.eval()
    return processor, model


def build_processor_inputs(processor: Any, image: Image.Image, image_path: Path, prompt: str) -> dict[str, Any]:
    if getattr(processor, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            return processor(text=text, images=[image], return_tensors="pt")

    return processor(images=image, text=prompt, return_tensors="pt")


def move_inputs_to_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return inputs

    try:
        first_param = next(model.parameters())
        target_device = first_param.device
    except StopIteration:
        target_device = torch.device("cuda")

    return {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}


def run_once(
    image_path: Path,
    prompt: str,
    model_key: str,
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    cfg = load_yaml(ROOT / "configs" / "models.yaml")
    model_cfg = cfg["models"][model_key]

    image = Image.open(image_path).convert("RGB")
    action_prompt = build_action_prompt(prompt)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    processor, model = load_model(model_cfg)

    inputs = build_processor_inputs(processor, image, image_path, action_prompt)
    inputs = move_inputs_to_device(inputs, model)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    output_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "model_key": model_key,
        "model_id": model_cfg["model_id"],
        "image_path": str(image_path),
        "prompt": prompt,
        "output": output_text,
        "latency_sec": elapsed,
        "peak_gpu_memory_gb": peak_memory_gb,
        "cuda_available": torch.cuda.is_available(),
    }


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROOT / "configs" / "models.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-key", default=cfg.get("default_model", "smolvlm2_500m"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_once(args.image, args.prompt, args.model_key, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
