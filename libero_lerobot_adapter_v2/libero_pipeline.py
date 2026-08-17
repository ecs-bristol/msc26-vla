"""Validated, fail-safe launcher for LeRobot training and LIBERO evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
LOG = logging.getLogger("libero_pipeline")


@dataclass(frozen=True)
class RunResult:
    command: list[str]
    output_dir: str
    returncode: int
    duration_seconds: float
    executed: bool


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def safe_identifier(value: str) -> str:
    if not SAFE_ID.fullmatch(value) or ".." in value.split("/"):
        raise argparse.ArgumentTypeError("仓库/模型标识格式无效")
    return value


def task_ids(value: str) -> str:
    if value == "":
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("task-ids 必须是 JSON 数组，例如 [0,1]") from exc
    if not isinstance(parsed, list) or any(type(item) is not int or item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("task-ids 只能包含非负整数")
    return json.dumps(parsed, separators=(",", ":"))


def output_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path == Path(path.anchor):
        raise argparse.ArgumentTypeError("拒绝把文件系统根目录作为输出目录")
    return path


def system_report() -> dict[str, object]:
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "linux": platform.system() == "Linux",
        "commands": {name: shutil.which(name) for name in ("lerobot-train", "lerobot-eval")},
    }
    try:
        import torch
        cuda = torch.cuda.is_available()
        report["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda,
            "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        }
    except (ImportError, RuntimeError) as exc:
        report["torch"] = {"installed": False, "error": str(exc)}
    return report


def train_command(args: argparse.Namespace) -> list[str]:
    return [
        "lerobot-train", "--policy.type=smolvla", "--policy.load_vlm_weights=true",
        "--policy.push_to_hub=false", f"--dataset.repo_id={args.dataset}",
        f"--dataset.revision={args.revision}", f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={args.output_dir}", f"--steps={args.steps}",
        f"--batch_size={args.batch_size}", f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}", f"--log_freq={args.log_freq}",
    ]


def eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        "lerobot-eval", f"--output_dir={args.output_dir}", f"--policy.path={args.policy}",
        "--env.type=libero", f"--env.task={','.join(args.suites)}",
        f"--eval.batch_size={args.batch_size}", f"--eval.n_episodes={args.episodes}",
        f"--env.max_parallel_tasks={args.max_parallel_tasks}",
        f"--env.control_mode={args.control_mode}", "--env.init_states=true",
        f"--env.hard_reset={'true' if args.hard_reset else 'false'}",
    ]
    if args.episode_length is not None:
        command.append(f"--env.episode_length={args.episode_length}")
    if args.seed is not None:
        command.append(f"--seed={args.seed}")
    if args.policy_num_steps is not None:
        command.append(f"--policy.num_steps={args.policy_num_steps}")
    if args.policy_n_action_steps is not None:
        command.append(f"--policy.n_action_steps={args.policy_n_action_steps}")
    if args.task_ids:
        command.append(f"--env.task_ids={args.task_ids}")
    return command


def execute(command: Sequence[str], output_dir: Path, run: bool, render_backend: str) -> RunResult:
    printable = subprocess.list2cmdline(list(command)) if os.name == "nt" else " ".join(command)
    print(printable)
    if not run:
        print("\n预览模式：加入 --run 后才会真正执行。")
        return RunResult(list(command), str(output_dir), 0, 0.0, False)
    if platform.system() != "Linux":
        raise RuntimeError("LIBERO 仅支持 Linux；请在 WSL2 Ubuntu 或原生 Linux 中运行")
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"找不到 {command[0]}，请先安装 LeRobot LIBERO 依赖")

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = render_backend
    started = time.monotonic()
    process = subprocess.Popen(list(command), env=env, start_new_session=True)
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        LOG.warning("收到中断，正在停止子进程……")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
    duration = round(time.monotonic() - started, 3)
    result = RunResult(list(command), str(output_dir), returncode, duration, True)
    (output_dir / "last_run.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安全、可复现的 LeRobot / LIBERO 入口")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("check", help="检查系统、CUDA 与 LeRobot 命令")

    train = sub.add_parser("train", help="训练 SmolVLA")
    train.add_argument("--dataset", type=safe_identifier, default="lerobot/libero")
    train.add_argument("--revision", type=safe_identifier, default="v3.0")
    train.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    train.add_argument("--output-dir", type=output_path, default=DEFAULT_OUTPUT / "libero_smolvla")
    train.add_argument("--steps", type=positive_int, default=100_000)
    train.add_argument("--batch-size", type=positive_int, default=8)
    train.add_argument("--num-workers", type=int, choices=range(0, 65), default=min(8, os.cpu_count() or 4))
    train.add_argument("--save-freq", type=positive_int, default=10_000)
    train.add_argument("--log-freq", type=positive_int, default=100)
    train.add_argument("--render-backend", choices=("egl", "glfw", "osmesa"), default="egl")
    train.add_argument("--run", action="store_true")

    evaluate = sub.add_parser("eval", help="评测并生成 MP4")
    evaluate.add_argument("--policy", type=safe_identifier, default="HuggingFaceVLA/smolvla_libero")
    evaluate.add_argument("--suites", nargs="+", choices=SUITES, default=["libero_object"])
    evaluate.add_argument("--task-ids", type=task_ids, default="[0]")
    evaluate.add_argument("--episodes", type=positive_int, default=1)
    evaluate.add_argument("--batch-size", type=positive_int, default=1)
    evaluate.add_argument("--max-parallel-tasks", type=positive_int, default=1)
    evaluate.add_argument("--episode-length", type=positive_int, help="最大控制步数；可靠模式建议 500")
    evaluate.add_argument("--seed", type=int, help="随机种子，用于复现实验")
    evaluate.add_argument("--policy-num-steps", type=positive_int, help="策略去噪/生成步数")
    evaluate.add_argument("--policy-n-action-steps", type=positive_int, help="每次规划后执行的动作数")
    evaluate.add_argument("--control-mode", choices=("relative", "absolute"), default="relative")
    evaluate.add_argument("--hard-reset", action=argparse.BooleanOptionalAction, default=True)
    evaluate.add_argument("--output-dir", type=output_path, default=DEFAULT_OUTPUT / "eval")
    evaluate.add_argument("--render-backend", choices=("egl", "glfw", "osmesa"), default="egl")
    evaluate.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.mode == "check":
            report = system_report()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            commands = report["commands"]
            return 0 if report["linux"] and all(commands.values()) else 2
        command = train_command(args) if args.mode == "train" else eval_command(args)
        result = execute(command, args.output_dir, args.run, args.render_backend)
        if result.executed:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return result.returncode
    except (OSError, RuntimeError) as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
