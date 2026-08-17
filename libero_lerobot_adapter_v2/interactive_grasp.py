"""随机列出 LIBERO 已知物品，并按中文或英文名称执行相应抓取任务。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ALIASES = {
    "牛奶": "milk", "黄油": "butter", "番茄酱": "tomato sauce",
    "西红柿酱": "tomato sauce", "沙拉酱": "salad dressing",
    "烧烤酱": "bbq sauce", "奶油奶酪": "cream cheese",
    "橙汁": "orange juice", "巧克力布丁": "chocolate pudding",
}


def infer_object(language: str) -> str:
    text = language.strip().lower().replace("_", " ")
    match = re.search(r"(?:pick up|grasp|take) (?:the )?(.+?)(?: and| then|$)", text)
    return match.group(1).strip() if match else text


def load_tasks():
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise RuntimeError("找不到 LIBERO；请先激活 ~/lerobot/.venv") from exc
    suite_type = benchmark.get_benchmark_dict().get("libero_object")
    if suite_type is None:
        raise RuntimeError("当前安装中没有 libero_object 任务集")
    suite = suite_type()
    return [
        (task_id, str(suite.get_task(task_id).language), infer_object(str(suite.get_task(task_id).language)))
        for task_id in range(len(suite.tasks))
    ]


def normalize(value: str) -> str:
    value = ALIASES.get(value.strip().lower(), value)
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def select_target(candidates, target: str):
    wanted = normalize(target)
    matches = [item for item in candidates if normalize(item[2]) == wanted]
    if len(matches) != 1:
        choices = "、".join(item[2] for item in candidates)
        raise ValueError(f"无法唯一匹配“{target}”；请选择：{choices}")
    return matches[0]


def build_candidates(tasks, count: int, seed: int | None, target: str | None):
    """Sample candidates while guaranteeing an explicitly requested target is present."""
    if count <= 0:
        raise ValueError("count 必须大于 0")
    rng = random.Random(seed)
    if not target:
        return rng.sample(tasks, min(count, len(tasks)))
    chosen = select_target(tasks, target)
    others = [item for item in tasks if item[0] != chosen[0]]
    candidates = [chosen, *rng.sample(others, min(count - 1, len(others)))]
    rng.shuffle(candidates)
    return candidates


def show(tasks) -> None:
    for index, (task_id, language, object_name) in enumerate(tasks, 1):
        print(f"{index}. {object_name}（任务 {task_id}）")
        print(f"   {language}")


def accuracy_strategies():
    """Ordered from the least intrusive policy behavior to fallback variants."""
    return [
        ("native", []),
        ("smooth", ["--policy-num-steps", "10", "--policy-n-action-steps", "10"]),
        ("responsive", ["--policy-num-steps", "10", "--policy-n-action-steps", "1"]),
    ]


def read_successes(output_dir: Path) -> list[bool]:
    info_file = output_dir / "eval_info.json"
    if not info_file.is_file():
        return []
    data = json.loads(info_file.read_text(encoding="utf-8"))
    tasks = data.get("per_task", [])
    if not tasks:
        return []
    return [bool(value) for value in tasks[0].get("metrics", {}).get("successes", [])]


def success_rate(successes: list[bool]) -> float:
    return round(100.0 * sum(successes) / len(successes), 2) if successes else 0.0


def append_history(history_file: Path, summary: dict) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "target", "task_id", "recognition_method", "mode", "seed",
        "best_strategy", "successes", "attempts", "success_rate", "run_directory",
    ]
    write_header = not history_file.exists()
    with history_file.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: summary[key] for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="随机物品识别与指定抓取")
    parser.add_argument("--count", type=int, default=4, help="随机候选物品数")
    parser.add_argument("--target", help="中文或英文目标名称")
    parser.add_argument("--auto-random", action="store_true",
                        help="自动随机选择目标，无需人工输入")
    parser.add_argument("--attempts", type=int, default=3, help="使用不同初始状态尝试的回合数")
    parser.add_argument("--episode-length", type=int, default=500, help="每回合最大控制步数")
    parser.add_argument("--seed", type=int, help="固定随机选择和评测结果")
    parser.add_argument("--output-dir", help="结果根目录")
    parser.add_argument("--mode", choices=("accurate", "fast"), default="accurate",
                        help="accurate 自动切换策略；fast 只用模型原生配置")
    parser.add_argument("--vision", action="store_true",
                        help="抓取前使用 Grounding DINO 从 RGB 画面确认目标")
    parser.add_argument("--vision-threshold", type=float, default=0.20)
    parser.add_argument("--vision-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--list", action="store_true", help="列出全部支持的物品")
    parser.add_argument("--run", action="store_true", help="真正执行，省略时只预览")
    args = parser.parse_args()
    try:
        if args.target and args.auto_random:
            raise ValueError("--target 与 --auto-random 不能同时使用")
        tasks = load_tasks()
        if args.list:
            show(tasks)
            return 0
        if args.count <= 0 or args.attempts <= 0 or args.episode_length <= 0:
            raise ValueError("count、attempts 和 episode-length 必须大于 0")
        run_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**31)
        candidates = build_candidates(tasks, args.count, run_seed, args.target)
        print("\n本轮识别到的可抓取物品：")
        show(candidates)
        if args.auto_random:
            chosen = random.Random(run_seed + 1).choice(candidates)
            selection_method = "automatic_random"
        else:
            target = args.target or input("\n请输入要抓取的物品名称：").strip()
            chosen = select_target(candidates, target)
            selection_method = "specified" if args.target else "interactive"
        print(f"\n识别结果：{chosen[2]}（任务元数据）")
        print(f"选择方式：{selection_method}\n机械臂任务：{chosen[1]}\n")

        import libero_pipeline
        base = Path(args.output_dir).expanduser() if args.output_dir else libero_pipeline.DEFAULT_OUTPUT / "interactive_eval"
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_task{chosen[0]}"
        destination = base / run_name
        strategies = accuracy_strategies() if args.mode == "accurate" else accuracy_strategies()[:1]
        detections = []
        recognition_method = "libero_task_metadata"
        if args.vision:
            from visual_detector import detect_libero_scene, target_detected
            labels = sorted({item[2] for item in tasks})
            print("正在从主摄像头 RGB 画面识别物品……")
            detections = detect_libero_scene(
                chosen[0], labels, destination / "vision", run_seed,
                args.vision_model, args.vision_threshold,
            )
            recognition_method = "grounding_dino_rgb"
            names = "、".join(f"{item['label']}({item['score']:.2f})" for item in detections) or "无"
            print(f"RGB 识别结果：{names}")
            if not target_detected(chosen[2], detections):
                raise RuntimeError(
                    f"RGB 检测未确认目标 {chosen[2]}，已停止抓取；请检查 vision/detected_scene.png"
                )
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "target": chosen[2], "task_id": chosen[0], "task_instruction": chosen[1],
            "recognition_method": recognition_method, "detections": detections,
            "selection_method": selection_method,
            "mode": args.mode, "seed": run_seed, "stages": [],
        }
        print(f"结果目录：{destination}\n")
        for stage_index, (stage_name, policy_options) in enumerate(strategies):
            stage_dir = destination / f"{stage_index + 1}_{stage_name}"
            command = [
                "eval", "--suites", "libero_object", "--task-ids", f"[{chosen[0]}]",
                "--episodes", str(args.attempts), "--batch-size", "1",
                "--episode-length", str(args.episode_length), "--output-dir", str(stage_dir),
                *policy_options,
            ]
            command.extend(("--seed", str(run_seed)))
            if args.run:
                command.append("--run")
            print(f"准确率策略 {stage_index + 1}/{len(strategies)}：{stage_name}")
            returncode = libero_pipeline.main(command)
            successes = read_successes(stage_dir) if args.run and returncode == 0 else []
            summary["stages"].append({
                "name": stage_name, "output_dir": str(stage_dir),
                "returncode": returncode, "successes": successes,
                "success_count": sum(successes), "attempts": len(successes),
                "success_rate": success_rate(successes),
            })
            if returncode != 0:
                return returncode
            if successes and all(successes):
                print(f"\n策略 {stage_name} 达到 {len(successes)}/{len(successes)}，停止后续尝试。")
                break
        if args.run:
            best = max(summary["stages"], key=lambda stage: (stage["success_rate"], -summary["stages"].index(stage)))
            summary["success"] = best["success_count"] > 0
            summary["best_strategy"] = best["name"]
            summary["successes"] = best["success_count"]
            summary["attempts"] = best["attempts"]
            summary["success_rate"] = best["success_rate"]
            summary["run_directory"] = str(destination)
            (destination / "accuracy_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            append_history(base / "grasp_history.csv", summary)
            print(f"\n最佳策略：{best['name']}，成功率 {best['success_rate']}%")
            print(f"历史汇总：{base / 'grasp_history.csv'}")
        return 0
    except (EOFError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
