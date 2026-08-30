"""随机列出 LIBERO 已知物品，并按中文或英文名称执行相应抓取任务。"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


def select_tasks_by_ids(tasks, task_ids: list[int]):
    """Resolve an explicit, ordered list of unique LIBERO task ids."""
    if not task_ids:
        raise ValueError("task-ids 不能为空")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task-ids 不能包含重复编号")
    index = {item[0]: item for item in tasks}
    missing = [task_id for task_id in task_ids if task_id not in index]
    if missing:
        raise ValueError(f"不存在的任务编号：{missing}；请先使用 --list 查看任务")
    return [index[task_id] for task_id in task_ids]


def accuracy_strategies():
    """Distinct strategies for the current SmolVLA checkpoint.

    The checkpoint-native behavior and n_action_steps=1 produced byte-identical
    rollout videos, so the redundant responsive variant is intentionally omitted.
    """
    return [
        ("native", []),
        ("smooth", ["--policy-num-steps", "10", "--policy-n-action-steps", "10"]),
    ]


def named_strategy(name: str):
    """Return an explicitly selected execution profile."""
    catalog = {
        "native": ("native", []),
        "balanced": ("balanced", ["--policy-num-steps", "10", "--policy-n-action-steps", "4"]),
        "smooth": ("smooth", ["--policy-num-steps", "10", "--policy-n-action-steps", "10"]),
    }
    return catalog[name]


def fallback_strategy(name: str) -> str:
    """Switch between the accurate native baseline and fast smooth execution."""
    return "smooth" if name == "native" else "native"


def adaptive_strategies(base_dir: Path, target: str):
    """Rank strategies using Laplace-smoothed success rates from prior runs."""
    defaults = accuracy_strategies()
    totals = {name: {"successes": 0, "attempts": 0} for name, _ in defaults}
    for summary_file in base_dir.glob("*/accuracy_summary.json"):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if normalize(str(data.get("target", ""))) != normalize(target):
            continue
        for stage in data.get("stages", []):
            name = stage.get("name")
            if name not in totals:
                continue
            values = [bool(value) for value in stage.get("successes", [])]
            totals[name]["successes"] += sum(values)
            totals[name]["attempts"] += len(values)
    default_index = {name: index for index, (name, _) in enumerate(defaults)}
    scores = {
        name: round((stats["successes"] + 1) / (stats["attempts"] + 2), 4)
        for name, stats in totals.items()
    }
    ordered = sorted(defaults, key=lambda item: (-scores[item[0]], default_index[item[0]]))
    return ordered, totals, scores


def read_successes(output_dir: Path) -> list[bool]:
    info_file = output_dir / "eval_info.json"
    if not info_file.is_file():
        return []
    data = json.loads(info_file.read_text(encoding="utf-8"))
    tasks = data.get("per_task", [])
    if not tasks:
        return []
    return [bool(value) for value in tasks[0].get("metrics", {}).get("successes", [])]


def read_task_successes(output_dir: Path) -> dict[int, list[bool]]:
    """Read every task result from a multi-task LeRobot evaluation."""
    info_file = output_dir / "eval_info.json"
    if not info_file.is_file():
        return {}
    data = json.loads(info_file.read_text(encoding="utf-8"))
    results: dict[int, list[bool]] = {}
    for task in data.get("per_task", []):
        task_id = task.get("task_id")
        if type(task_id) is not int:
            continue
        results[task_id] = [
            bool(value) for value in task.get("metrics", {}).get("successes", [])
        ]
    return results


def success_rate(successes: list[bool]) -> float:
    return round(100.0 * sum(successes) / len(successes), 2) if successes else 0.0


def wilson_interval(successes: int, attempts: int, z: float = 1.96) -> list[float]:
    """Return a percentage Wilson interval for a binomial success count."""
    if attempts <= 0 or not 0 <= successes <= attempts:
        return [0.0, 0.0]
    proportion = successes / attempts
    z_squared = z * z
    denominator = 1.0 + z_squared / attempts
    centre = (proportion + z_squared / (2.0 * attempts)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / attempts
            + z_squared / (4.0 * attempts * attempts)
        )
        / denominator
    )
    return [round(100.0 * max(0.0, centre - half_width), 1),
            round(100.0 * min(1.0, centre + half_width), 1)]


def effective_batch_size(requested: int | None, attempts: int, all_tasks: bool) -> int:
    """Use episode-level batching for suite runs while preserving single-task defaults."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("batch-size 必须大于 0")
        return min(requested, attempts)
    return attempts if all_tasks else 1


def read_duration(output_dir: Path) -> float:
    info_file = output_dir / "last_run.json"
    if not info_file.is_file():
        return 0.0
    try:
        return float(json.loads(info_file.read_text(encoding="utf-8")).get("duration_seconds", 0.0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0


def build_strategy_routes(base_dir: Path, tasks, exclude_seed: int | None = None) -> tuple[dict[int, str], dict[int, dict]]:
    """Choose a strategy from historical runs, deduplicated by suite, seed and task."""
    names = ("native", "balanced", "smooth")
    totals = {
        task_id: {name: {"successes": 0, "attempts": 0} for name in names}
        for task_id, _, _ in tasks
    }
    duplicate_counts = {task_id: 0 for task_id, _, _ in tasks}
    seen_records: set[tuple] = set()
    for summary_file in sorted(base_dir.glob("*/suite_summary.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if exclude_seed is not None and data.get("seed") == exclude_seed:
            continue
        for stage in data.get("stages", []):
            name = stage.get("name")
            if name not in names:
                continue
            for row in stage.get("tasks", []):
                task_id = row.get("task_id")
                if task_id not in totals:
                    continue
                seed = data.get("seed")
                source_identity = seed if seed is not None else str(summary_file.resolve())
                record_key = (data.get("suite"), source_identity, name, task_id)
                if record_key in seen_records:
                    duplicate_counts[task_id] += 1
                    continue
                seen_records.add(record_key)
                totals[task_id][name]["successes"] += int(row.get("successes", 0))
                totals[task_id][name]["attempts"] += int(row.get("attempts", 0))

    routes: dict[int, str] = {}
    evidence: dict[int, dict] = {}
    default_order = {name: index for index, name in enumerate(names)}
    for task_id, _, _ in tasks:
        observed = [name for name in names if totals[task_id][name]["attempts"] > 0]
        candidates = observed or ["native"]
        scores = {
            name: round(
                (totals[task_id][name]["successes"] + 1)
                / (totals[task_id][name]["attempts"] + 2), 4
            )
            for name in candidates
        }
        selected = sorted(candidates, key=lambda name: (-scores[name], default_order[name]))[0]
        routes[task_id] = selected
        evidence[task_id] = {
            "selected": selected, "scores": scores, "history": totals[task_id],
            "fallback": not observed,
            "duplicate_records_ignored": duplicate_counts[task_id],
            "excluded_evaluation_seed": exclude_seed,
        }
    return routes, evidence


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


def run_strategy_router(args, tasks, base: Path, destination: Path, run_seed: int,
                        recovery: bool = False) -> int:
    """Evaluate with per-task routing and optionally retry failures using the other strategy."""
    import libero_pipeline

    routes, evidence = build_strategy_routes(base, tasks, exclude_seed=run_seed)
    batch_size = effective_batch_size(args.batch_size, args.attempts, True)
    groups = {name: [] for name in ("native", "balanced", "smooth")}
    for task in tasks:
        groups[routes[task[0]]].append(task)
    groups = {name: values for name, values in groups.items() if values}
    print("任务感知策略路由：")
    for name, values in groups.items():
        print(f"  {name}: " + "、".join(item[2] for item in values))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": "libero_object", "task_count": len(tasks),
        "attempts_per_task": args.attempts, "batch_size": batch_size,
        "hard_reset": not args.soft_reset, "max_parallel_tasks": args.max_parallel_tasks,
        "mode": "hybrid_recovery" if recovery else "strategy_router", "seed": run_seed,
        "routing_decisions": evidence, "stages": [], "run_directory": str(destination),
    }
    initial_outcomes: dict[int, list[bool]] = {}
    for stage_index, (name, selected_tasks) in enumerate(groups.items(), 1):
        stage_dir = destination / f"{stage_index}_{name}"
        command = [
            "eval", "--suites", "libero_object",
            "--task-ids", json.dumps([item[0] for item in selected_tasks]),
            "--episodes", str(args.attempts), "--batch-size", str(batch_size),
            "--max-parallel-tasks", str(args.max_parallel_tasks),
            "--episode-length", str(args.episode_length), "--output-dir", str(stage_dir),
            "--seed", str(run_seed), *named_strategy(name)[1],
        ]
        if args.soft_reset:
            command.append("--no-hard-reset")
        if args.run:
            command.append("--run")
        print(f"\n路由组 {stage_index}/{len(groups)}：{name}，{len(selected_tasks)} 个任务")
        returncode = libero_pipeline.main(command)
        if returncode != 0:
            return returncode
        by_task = read_task_successes(stage_dir) if args.run else {}
        rows, all_values = [], []
        for task_id, instruction, target in selected_tasks:
            values = by_task.get(task_id, [])
            initial_outcomes[task_id] = values
            all_values.extend(values)
            rows.append({
                "task_id": task_id, "target": target, "instruction": instruction,
                "successes": sum(values), "attempts": len(values),
                "success_rate": success_rate(values),
                "wilson_95_ci": wilson_interval(sum(values), len(values)),
            })
        duration = read_duration(stage_dir) if args.run else 0.0
        summary["stages"].append({
            "name": name, "output_dir": str(stage_dir), "tasks": rows,
            "successes": sum(all_values), "attempts": len(all_values),
            "success_rate": success_rate(all_values), "duration_seconds": duration,
            "wilson_95_ci": wilson_interval(sum(all_values), len(all_values)),
            "seconds_per_episode": round(duration / len(all_values), 3) if all_values else 0.0,
        })

    recovery_outcomes: dict[int, list[bool]] = {}
    if recovery and args.run:
        failed_tasks = [task for task in tasks if not any(initial_outcomes.get(task[0], []))]
        recovery_groups = {"native": [], "smooth": []}
        for task in failed_tasks:
            fallback = fallback_strategy(routes[task[0]])
            recovery_groups[fallback].append(task)
        recovery_groups = {name: values for name, values in recovery_groups.items() if values}
        if failed_tasks:
            print(f"\n检测到 {len(failed_tasks)} 个首轮失败任务，切换备用策略重试一次。")
        for recovery_index, (name, selected_tasks) in enumerate(recovery_groups.items(), 1):
            stage_number = len(summary["stages"]) + 1
            stage_dir = destination / f"{stage_number}_recovery_{name}"
            command = [
                "eval", "--suites", "libero_object",
                "--task-ids", json.dumps([item[0] for item in selected_tasks]),
                "--episodes", "1", "--batch-size", "1",
                "--max-parallel-tasks", str(args.max_parallel_tasks),
                "--episode-length", str(args.episode_length), "--output-dir", str(stage_dir),
                "--seed", str(run_seed + 1000 + recovery_index), *named_strategy(name)[1], "--run",
            ]
            if args.soft_reset:
                command.append("--no-hard-reset")
            print(f"恢复组 {recovery_index}/{len(recovery_groups)}：{name}，{len(selected_tasks)} 个任务")
            returncode = libero_pipeline.main(command)
            if returncode != 0:
                return returncode
            by_task = read_task_successes(stage_dir)
            rows, all_values = [], []
            for task_id, instruction, target in selected_tasks:
                values = by_task.get(task_id, [])
                recovery_outcomes[task_id] = values
                all_values.extend(values)
                rows.append({
                    "task_id": task_id, "target": target, "instruction": instruction,
                    "successes": sum(values), "attempts": len(values),
                    "success_rate": success_rate(values),
                    "wilson_95_ci": wilson_interval(sum(values), len(values)),
                })
            duration = read_duration(stage_dir)
            summary["stages"].append({
                "name": f"recovery_{name}", "fallback_strategy": name,
                "output_dir": str(stage_dir), "tasks": rows,
                "successes": sum(all_values), "attempts": len(all_values),
                "success_rate": success_rate(all_values), "duration_seconds": duration,
                "wilson_95_ci": wilson_interval(sum(all_values), len(all_values)),
                "seconds_per_episode": round(duration / len(all_values), 3) if all_values else 0.0,
            })

    if args.run:
        total_successes = sum(stage["successes"] for stage in summary["stages"])
        total_attempts = sum(stage["attempts"] for stage in summary["stages"])
        completed_tasks = sum(
            any(initial_outcomes.get(task_id, [])) or any(recovery_outcomes.get(task_id, []))
            for task_id, _, _ in tasks
        )
        recovered_tasks = sum(any(values) for values in recovery_outcomes.values())
        summary.update({
            "best_strategy": "task_aware_hybrid" if recovery else "task_aware_router",
            "successes": total_successes,
            "attempts": total_attempts,
            "success_rate": round(100.0 * total_successes / total_attempts, 2) if total_attempts else 0.0,
            "wilson_95_ci": wilson_interval(total_successes, total_attempts),
            "duration_seconds": round(sum(stage["duration_seconds"] for stage in summary["stages"]), 3),
            "completed_tasks": completed_tasks, "task_count": len(tasks),
            "task_completion_rate": round(100.0 * completed_tasks / len(tasks), 2) if tasks else 0.0,
            "recovery_attempted_tasks": len(recovery_outcomes), "recovered_tasks": recovered_tasks,
            "recovery_success_rate": round(100.0 * recovered_tasks / len(recovery_outcomes), 2)
            if recovery_outcomes else 0.0,
        })
        destination.mkdir(parents=True, exist_ok=True)
        report_file = destination / "suite_summary.json"
        report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        csv_file = destination / "task_summary.csv"
        with csv_file.open("w", newline="", encoding="utf-8-sig") as stream:
            fields = ["strategy", "task_id", "target", "successes", "attempts", "success_rate"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for stage in summary["stages"]:
                for row in stage["tasks"]:
                    writer.writerow({"strategy": stage["name"], **{key: row[key] for key in fields[1:]}})
        print(f"\n任务完成率：{summary['task_completion_rate']}%；回合成功率：{summary['success_rate']}%")
        if recovery:
            print(f"失败恢复：{recovered_tasks}/{len(recovery_outcomes)} 个任务")
        print(f"JSON 汇总：{report_file}\nCSV 表格：{csv_file}")
    return 0


def run_all_tasks(args, tasks, base: Path, run_seed: int) -> int:
    """Evaluate the complete Object suite and produce paper-friendly aggregates."""
    if args.vision:
        raise ValueError(
            "--all-tasks 暂不与 --vision 同时使用；批量基准测试请移除 --vision，"
            "单物品视觉闭环仍可继续使用 --vision"
        )
    import libero_pipeline

    ids = [item[0] for item in tasks]
    suffix = "all_tasks" if len(tasks) == 10 and ids == list(range(10)) else "tasks_" + "-".join(map(str, ids))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{suffix}"
    destination = base / run_name
    if args.strategy in ("router", "hybrid"):
        return run_strategy_router(args, tasks, base, destination, run_seed,
                                   recovery=args.strategy == "hybrid")
    if args.strategy != "auto":
        strategies = [named_strategy(args.strategy)]
    else:
        strategies = accuracy_strategies()[:1] if args.mode == "fast" else accuracy_strategies()
    batch_size = effective_batch_size(args.batch_size, args.attempts, True)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": "libero_object", "task_count": len(tasks),
        "attempts_per_task": args.attempts, "batch_size": batch_size,
        "hard_reset": not args.soft_reset, "max_parallel_tasks": args.max_parallel_tasks,
        "mode": args.mode, "seed": run_seed,
        "strategy_order": [name for name, _ in strategies], "stages": [],
        "run_directory": str(destination),
    }
    print(f"\n批量评测：LIBERO Object 已选择 {len(tasks)} 个任务")
    print(f"每个任务 {args.attempts} 次，并行回合数 {batch_size}；策略顺序："
          + " → ".join(name for name, _ in strategies))
    print(f"结果目录：{destination}\n")

    task_ids = [task_id for task_id, _, _ in tasks]
    for stage_index, (stage_name, policy_options) in enumerate(strategies):
        stage_dir = destination / f"{stage_index + 1}_{stage_name}"
        command = [
            "eval", "--suites", "libero_object", "--task-ids", json.dumps(task_ids),
            "--episodes", str(args.attempts), "--batch-size", str(batch_size),
            "--max-parallel-tasks", str(args.max_parallel_tasks),
            "--episode-length", str(args.episode_length), "--output-dir", str(stage_dir),
            "--seed", str(run_seed), *policy_options,
        ]
        if args.soft_reset:
            command.append("--no-hard-reset")
        if args.run:
            command.append("--run")
        print(f"批量策略 {stage_index + 1}/{len(strategies)}：{stage_name}")
        returncode = libero_pipeline.main(command)
        if returncode != 0:
            return returncode
        by_task = read_task_successes(stage_dir) if args.run else {}
        task_rows = []
        all_values: list[bool] = []
        for task_id, instruction, object_name in tasks:
            values = by_task.get(task_id, [])
            all_values.extend(values)
            task_rows.append({
                "task_id": task_id, "target": object_name, "instruction": instruction,
                "successes": sum(values), "attempts": len(values),
                "success_rate": success_rate(values),
                "wilson_95_ci": wilson_interval(sum(values), len(values)),
            })
        duration = read_duration(stage_dir) if args.run else 0.0
        summary["stages"].append({
            "name": stage_name, "output_dir": str(stage_dir),
            "successes": sum(all_values), "attempts": len(all_values),
            "success_rate": success_rate(all_values), "duration_seconds": duration,
            "wilson_95_ci": wilson_interval(sum(all_values), len(all_values)),
            "seconds_per_episode": round(duration / len(all_values), 3) if all_values else 0.0,
            "tasks": task_rows,
        })
        if all_values and all(all_values):
            print(f"\n策略 {stage_name} 全部 {len(all_values)}/{len(all_values)} 成功，停止后续策略。")
            break

    if args.run:
        best = max(summary["stages"], key=lambda row: row["success_rate"])
        summary.update({
            "best_strategy": best["name"], "successes": best["successes"],
            "attempts": best["attempts"], "success_rate": best["success_rate"],
            "wilson_95_ci": best["wilson_95_ci"],
        })
        destination.mkdir(parents=True, exist_ok=True)
        report_file = destination / "suite_summary.json"
        report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        csv_file = destination / "task_summary.csv"
        with csv_file.open("w", newline="", encoding="utf-8-sig") as stream:
            fields = ["strategy", "task_id", "target", "successes", "attempts", "success_rate"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for stage in summary["stages"]:
                for row in stage["tasks"]:
                    writer.writerow({"strategy": stage["name"], **{key: row[key] for key in fields[1:]}})
        print(f"\n最佳策略：{best['name']}，总体成功率 {best['success_rate']}% ")
        print(f"JSON 汇总：{report_file}\nCSV 表格：{csv_file}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="随机物品识别与指定抓取")
    parser.add_argument("--count", type=int, default=4, help="随机候选物品数")
    parser.add_argument("--target", help="中文或英文目标名称")
    parser.add_argument("--auto-random", action="store_true",
                        help="自动随机选择目标，无需人工输入")
    parser.add_argument("--all-tasks", action="store_true",
                        help="批量评测 LIBERO Object 全部任务并生成总汇报")
    parser.add_argument("--sample-tasks", "--random-tasks", dest="sample_tasks", type=int,
                        help="从 Object 任务集中随机选择指定数量的任务")
    parser.add_argument("--task-ids", nargs="+", type=int,
                        help="由用户选择任务编号，例如 --task-ids 0 6 7")
    parser.add_argument("--attempts", type=int, default=3, help="使用不同初始状态尝试的回合数")
    parser.add_argument("--batch-size", type=int,
                        help="同时运行的回合数；全任务模式默认等于 attempts，单任务默认 1")
    parser.add_argument("--max-parallel-tasks", type=int, default=1,
                        help="同时运行的不同任务数；显存有限时保持 1")
    parser.add_argument("--soft-reset", action="store_true",
                        help="使用更快的环境软重置；正式复现实验建议不启用")
    parser.add_argument("--episode-length", type=int, default=500, help="每回合最大控制步数")
    parser.add_argument("--seed", type=int, help="固定随机选择和评测结果")
    parser.add_argument("--output-dir", help="结果根目录")
    parser.add_argument("--mode", choices=("accurate", "fast"), default="accurate",
                        help="accurate 自动切换策略；fast 只用模型原生配置")
    parser.add_argument("--strategy-order", choices=("adaptive", "fixed"), default="adaptive",
                        help="adaptive 根据历史成功率排序；fixed 使用固定顺序")
    parser.add_argument("--strategy", choices=("auto", "native", "balanced", "smooth", "router", "hybrid"), default="auto",
                        help="显式运行一种策略；auto 按 mode 和历史结果决定")
    parser.add_argument("--vision", action="store_true",
                        help="抓取前使用 Grounding DINO 从 RGB 画面确认目标")
    parser.add_argument("--vision-threshold", type=float, default=0.20)
    parser.add_argument("--vision-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--list", action="store_true", help="列出全部支持的物品")
    parser.add_argument("--run", action="store_true", help="真正执行，省略时只预览")
    args = parser.parse_args()
    try:
        selection_modes = (args.target, args.auto_random, args.all_tasks, args.sample_tasks, args.task_ids)
        if sum(bool(value) for value in selection_modes) > 1:
            raise ValueError("--target、--auto-random、--all-tasks、--random-tasks 与 --task-ids 只能选择一种")
        if args.strategy in ("router", "hybrid") and not (args.all_tasks or args.sample_tasks or args.task_ids):
            raise ValueError("--strategy router/hybrid 仅用于多任务模式")
        tasks = load_tasks()
        if args.list:
            show(tasks)
            return 0
        if args.count <= 0 or args.attempts <= 0 or args.episode_length <= 0 or args.max_parallel_tasks <= 0:
            raise ValueError("count、attempts、episode-length 和 max-parallel-tasks 必须大于 0")
        batch_size = effective_batch_size(args.batch_size, args.attempts, args.all_tasks)
        run_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**31)
        import libero_pipeline
        base = Path(args.output_dir).expanduser() if args.output_dir else libero_pipeline.DEFAULT_OUTPUT / "interactive_eval"
        if args.all_tasks:
            return run_all_tasks(args, tasks, base, run_seed)
        if args.sample_tasks is not None:
            if not 1 <= args.sample_tasks <= len(tasks):
                raise ValueError(f"random-tasks 必须在 1 到 {len(tasks)} 之间")
            selected = random.Random(run_seed).sample(tasks, args.sample_tasks)
            print(f"\n随机选择的 {len(selected)} 个任务（seed={run_seed}）：")
            show(selected)
            return run_all_tasks(args, selected, base, run_seed)
        if args.task_ids:
            selected = select_tasks_by_ids(tasks, args.task_ids)
            print("\n用户选择的任务：")
            show(selected)
            return run_all_tasks(args, selected, base, run_seed)
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

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_task{chosen[0]}"
        destination = base / run_name
        if args.strategy != "auto":
            strategies, strategy_history, strategy_scores = [named_strategy(args.strategy)], {}, {}
            strategy_order_source = "explicit"
        elif args.mode == "fast":
            strategies, strategy_history, strategy_scores = accuracy_strategies()[:1], {}, {}
            strategy_order_source = "fast_native_only"
        elif args.strategy_order == "adaptive":
            strategies, strategy_history, strategy_scores = adaptive_strategies(base, chosen[2])
            strategy_order_source = "historical_laplace_success_rate"
        else:
            strategies, strategy_history, strategy_scores = accuracy_strategies(), {}, {}
            strategy_order_source = "fixed"
        print("策略顺序：" + " → ".join(name for name, _ in strategies))
        detections = []
        recognition_method = "libero_task_metadata"
        if args.vision:
            from visual_detector import detect_image, detect_libero_scene, target_detected, target_prompts
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
                print(f"多物品检测未确认 {chosen[2]}，正在使用目标专用提示词复检……")
                retry = detect_image(
                    destination / "vision" / "initial_scene.png",
                    target_prompts(chosen[2]), destination / "vision" / "target_retry",
                    model_id=args.vision_model,
                    threshold=max(0.15, args.vision_threshold - 0.05),
                )
                if retry:
                    best_retry = max(retry, key=lambda item: item["score"])
                    best_retry = {**best_retry, "raw_label": best_retry["label"], "label": chosen[2],
                                  "detection_pass": "target_prompt_retry"}
                    detections.append(best_retry)
                    print(f"目标复检成功：{chosen[2]}({best_retry['score']:.2f})")
                else:
                    raise RuntimeError(
                        f"RGB 两轮检测均未确认目标 {chosen[2]}，已停止抓取；"
                        "请检查 vision/detected_scene.png 和 vision/target_retry/"
                    )
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "target": chosen[2], "task_id": chosen[0], "task_instruction": chosen[1],
            "recognition_method": recognition_method, "detections": detections,
            "selection_method": selection_method,
            "mode": args.mode, "seed": run_seed, "strategy_order_source": strategy_order_source,
            "strategy_history": strategy_history, "strategy_scores": strategy_scores,
            "strategy_order": [name for name, _ in strategies], "stages": [],
        }
        print(f"结果目录：{destination}\n")
        for stage_index, (stage_name, policy_options) in enumerate(strategies):
            stage_dir = destination / f"{stage_index + 1}_{stage_name}"
            command = [
                "eval", "--suites", "libero_object", "--task-ids", f"[{chosen[0]}]",
                "--episodes", str(args.attempts), "--batch-size", str(batch_size),
                "--max-parallel-tasks", str(args.max_parallel_tasks),
                "--episode-length", str(args.episode_length), "--output-dir", str(stage_dir),
                *policy_options,
            ]
            command.extend(("--seed", str(run_seed)))
            if args.soft_reset:
                command.append("--no-hard-reset")
            if args.run:
                command.append("--run")
            print(f"准确率策略 {stage_index + 1}/{len(strategies)}：{stage_name}")
            returncode = libero_pipeline.main(command)
            successes = read_successes(stage_dir) if args.run and returncode == 0 else []
            duration = read_duration(stage_dir) if args.run and returncode == 0 else 0.0
            summary["stages"].append({
                "name": stage_name, "output_dir": str(stage_dir),
                "returncode": returncode, "successes": successes,
                "success_count": sum(successes), "attempts": len(successes),
                "success_rate": success_rate(successes), "duration_seconds": duration,
                "seconds_per_episode": round(duration / len(successes), 3) if successes else 0.0,
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
