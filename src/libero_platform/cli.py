from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .catalog import Catalog, CatalogError
from .policies.base import PolicyAdapter
from .recorder import RunRecorder
from .runner import RunnerDependencies, run_experiment
from .terminal_status import TerminalStatus
from .validator import validate_config


class ExecutionUnavailableError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="libero_platform")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate")
    validate.add_argument("yaml_path", type=Path)

    listing = subcommands.add_parser("list")
    list_commands = listing.add_subparsers(dest="list_command", required=True)
    list_commands.add_parser("suites")
    tasks = list_commands.add_parser("tasks")
    tasks.add_argument("--suite", required=True)
    list_commands.add_parser("policies")
    list_commands.add_parser("deployments")

    run = subcommands.add_parser("run")
    run.add_argument("yaml_path", type=Path)

    serve_policy = subcommands.add_parser("serve-policy")
    serve_policy.add_argument(
        "--policy", required=True, choices=("zero_policy", "smolvla_libero")
    )
    serve_policy.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    serve_policy.add_argument("--revision")
    serve_policy.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp16"
    )
    serve_policy.add_argument("--num-steps", type=int, default=10)
    serve_policy.add_argument("--n-action-steps", type=int, default=1)
    serve_policy.add_argument("--chunk-size", type=int, default=None)
    serve_policy.add_argument(
        "--quant-method",
        choices=("none", "int4_groupwise", "int8_groupwise", "bnb_nf4", "mixed"),
        default="none",
    )
    serve_policy.add_argument(
        "--quant-scope", choices=("language", "backbone"), default="language"
    )
    serve_policy.add_argument("--vision-bits", type=int, choices=(4, 8, 16), default=4)
    serve_policy.add_argument("--connector-bits", type=int, choices=(4, 8, 16), default=8)
    serve_policy.add_argument("--text-bits", type=int, choices=(4, 8, 16), default=8)
    serve_policy.add_argument("--tensorrt-vision-engine", default=None)
    serve_policy.add_argument("--tensorrt-connector-engine", default=None)
    serve_policy.add_argument("--host", default="127.0.0.1")
    serve_policy.add_argument("--port", default=8081, type=int)

    parity = subcommands.add_parser(
        "policy-parity",
        help="Compare one deterministic LIBERO observation locally and through a remote policy service.",
    )
    parity.add_argument("--suite", default="libero_spatial")
    parity.add_argument("--task-id", default=0, type=int)
    parity.add_argument("--initial-state-id", default=0, type=int)
    parity.add_argument("--seed", default=42, type=int)
    parity.add_argument("--checkpoint", default="HuggingFaceVLA/smolvla_libero")
    parity.add_argument("--revision", required=True)
    parity.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp16"
    )
    parity.add_argument("--endpoint", default="http://10.42.0.2:8081")
    parity.add_argument("--settle-steps", default=0, type=int)
    parity.add_argument("--threshold", default=1e-4, type=float)
    parity.add_argument(
        "--repeat-check",
        action="store_true",
        help="Repeat the identical request on each side and record same-device drift.",
    )
    parity.add_argument("--output-dir", type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    config_root: Path | None = None,
    output_root: Path | None = None,
    run_callable: Callable[..., object] | None = None,
    serve_callable: Callable[[PolicyAdapter, str, int], object] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    active_config_root = config_root or _default_config_root()

    if args.command == "validate":
        report = validate_config(
            args.yaml_path, active_config_root, runtime_checks=False
        )
        if not report.ok:
            return _report_validation_failure(report)

        assert report.resolved_spec is not None
        task_name_resolver = None
        initial_state_resolver = None
        if report.resolved_spec.benchmark.backend == "libero":
            try:
                backend = _build_backend(report.resolved_spec)
                task_name_resolver = _task_name_resolver(backend)
                initial_state_resolver = _initial_state_resolver(backend)
            except Exception:
                print("preflight: preflight_error", file=sys.stderr)
                return 3

        report = validate_config(
            args.yaml_path,
            active_config_root,
            check_network=True,
            task_name_resolver=task_name_resolver,
            initial_state_resolver=initial_state_resolver,
        )
        if report.ok:
            assert report.resolved_spec is not None
            print(f"valid: {report.resolved_spec.name}")
            return 0
        return _report_validation_failure(report)

    if args.command == "list":
        return _list_catalog(args, active_config_root)

    if args.command == "serve-policy":
        from .spec import SmolVLAInferenceSpec

        smolvla_inference = SmolVLAInferenceSpec(
            n_action_steps=args.n_action_steps,
            num_steps=args.num_steps,
            chunk_size=args.chunk_size,
        )
        quant_config = {
            "quant_method": args.quant_method,
            "quant_scope": args.quant_scope,
            "vision_bits": args.vision_bits,
            "connector_bits": args.connector_bits,
            "text_bits": args.text_bits,
            "tensorrt_vision_engine": args.tensorrt_vision_engine or None,
            "tensorrt_connector_engine": args.tensorrt_connector_engine or None,
        }
        return _serve_policy(
            args.policy,
            args.checkpoint,
            args.precision,
            args.revision,
            args.host,
            args.port,
            smolvla_inference,
            quant_config,
            serve_callable,
        )

    if args.command == "policy-parity":
        return _policy_parity(args)

    return _run(args.yaml_path, active_config_root, output_root, run_callable)


def _default_config_root() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "libero_runs"


def _default_policy_parity_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "policy_parity"


def _policy_parity(args: argparse.Namespace) -> int:
    """Compare a single deterministic LIBERO reset through local and remote policies."""

    import requests

    from .backends.libero_backend import LiberoBackend
    from .policies.base import EpisodeContext, PolicyRequest
    from .policies.remote_http import RemoteHTTPPolicyAdapter, probe_remote_policy
    from .policies.smolvla_policy import SmolVLAPolicyAdapter
    from .policy_parity import (
        PolicyParityError,
        build_policy_parity_summary,
        build_policy_repeatability_summary,
        write_policy_parity_artifacts,
    )

    if args.settle_steps < 0:
        print("policy-parity: settle_steps must be non-negative", file=sys.stderr)
        return 2
    if args.threshold < 0.0:
        print("policy-parity: threshold must be non-negative", file=sys.stderr)
        return 2

    local_policy = SmolVLAPolicyAdapter(
        model_key="smolvla_libero",
        checkpoint=args.checkpoint,
        precision=args.precision,
        revision=args.revision,
    )
    remote_policy = RemoteHTTPPolicyAdapter(
        "smolvla_libero", args.endpoint, requests.Session()
    )
    episode = None
    try:
        remote_identity = probe_remote_policy(args.endpoint)
        _require_remote_policy_identity(
            remote_identity,
            checkpoint=args.checkpoint,
            revision=args.revision,
            precision=args.precision,
        )

        backend = LiberoBackend(
            settle_steps=args.settle_steps,
            initial_state_source="benchmark",
        )
        task_name = _task_name_resolver(backend)(args.suite, args.task_id)
        episode = backend.open_episode(
            args.suite,
            args.task_id,
            args.initial_state_id,
            max_steps=1,
            seed=args.seed,
        )
        observation = episode.reset()
        context = EpisodeContext(
            suite=args.suite,
            task_id=args.task_id,
            task_name=task_name,
            initial_state_id=args.initial_state_id,
            seed=args.seed,
        )
        request = PolicyRequest(
            run_id="policy-parity",
            episode_id=0,
            step_id=0,
            instruction=observation.instruction,
            images=observation.images,
            proprioception=observation.proprioception,
            previous_action=None,
        )

        local_policy.load()
        local_policy.begin_episode(context)
        remote_policy.begin_episode(context)
        local_response = local_policy.predict(request)
        remote_response = remote_policy.predict(request)
        summary = build_policy_parity_summary(
            request=request,
            local_identity=local_policy.identity(),
            remote_identity=remote_identity,
            local_response=local_response,
            remote_response=remote_response,
            threshold=args.threshold,
        )
        if args.repeat_check:
            local_policy.begin_episode(context)
            repeated_local_response = local_policy.predict(request)
            remote_policy.begin_episode(context)
            repeated_remote_response = remote_policy.predict(request)
            summary["repeatability"] = {
                "enabled": True,
                "local": build_policy_repeatability_summary(
                    first_response=local_response,
                    repeated_response=repeated_local_response,
                ),
                "remote": build_policy_repeatability_summary(
                    first_response=remote_response,
                    repeated_response=repeated_remote_response,
                ),
            }
        summary["episode"] = {
            "suite": args.suite,
            "task_id": args.task_id,
            "task_name": task_name,
            "initial_state_id": args.initial_state_id,
            "seed": args.seed,
            "settle_steps": args.settle_steps,
            "reset": _reset_evidence(episode),
        }
        output_root = args.output_dir or _default_policy_parity_root()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        artifacts = write_policy_parity_artifacts(
            summary,
            output_root
            / f"{args.suite}_task{args.task_id}_state{args.initial_state_id}_{timestamp}",
        )
        message = (
            "policy-parity: "
            f"{summary['status']} (max_abs={summary['delta']['max_abs']:.6g})"
        )
        if args.repeat_check:
            repeatability = summary["repeatability"]
            message += (
                "; repeat_local_max="
                f"{repeatability['local']['max_abs']:.6g}"
                "; repeat_remote_max="
                f"{repeatability['remote']['max_abs']:.6g}"
            )
        print(f"{message}; evidence: {artifacts['summary']}")
        return 0
    except (PolicyParityError, ValueError, RuntimeError) as exc:
        print(f"policy-parity: {exc}", file=sys.stderr)
        return 3
    finally:
        if episode is not None:
            episode.close()
        local_policy.close()


def _require_remote_policy_identity(
    remote_identity: Mapping[str, object],
    *,
    checkpoint: str,
    revision: str,
    precision: str,
) -> None:
    expected = {
        "model_key": "smolvla_libero",
        "checkpoint": checkpoint,
        "revision": revision,
        "precision": precision,
    }
    for field, expected_value in expected.items():
        if remote_identity.get(field) != expected_value:
            raise RuntimeError(f"remote policy {field} does not match the requested value")


def _reset_evidence(episode: object) -> dict[str, object] | None:
    evidence = getattr(episode, "reset_evidence", None)
    if evidence is None:
        return None
    return {
        "seed": evidence.seed,
        "initial_state_source": evidence.initial_state_source,
        "settle_steps": evidence.settle_steps,
        "fingerprint": evidence.fingerprint,
    }


def _serve_policy(
    policy_key: str,
    checkpoint: str,
    precision: str,
    revision: str | None,
    host: str,
    port: int,
    smolvla_inference: object,
    quant_config: dict[str, object],
    serve_callable: Callable[[PolicyAdapter, str, int], object] | None,
) -> int:
    adapter = _build_service_policy(
        policy_key, checkpoint, precision, revision, smolvla_inference, quant_config
    )
    server = None
    try:
        adapter.load()
        if serve_callable is not None:
            serve_callable(adapter, host, port)
            return 0

        from .deployment.policy_service import create_policy_server

        server = create_policy_server(adapter, host, port)
        print(f"policy service listening on http://{host}:{server.server_port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return _report_unavailable(TerminalStatus(), "policy_service_startup_failed")
    finally:
        if server is not None:
            try:
                server.server_close()
            except Exception:
                pass
        try:
            adapter.close()
        except Exception:
            pass


def _build_service_policy(
    policy_key: str,
    checkpoint: str,
    precision: str,
    revision: str | None = None,
    smolvla_inference: object | None = None,
    quant_config: dict[str, object] | None = None,
) -> PolicyAdapter:
    if policy_key == "zero_policy":
        from .policies.zero_policy import ZeroPolicyAdapter

        return ZeroPolicyAdapter(policy_key)
    if policy_key == "smolvla_libero":
        from .policies.smolvla_policy import SmolVLAPolicyAdapter

        return SmolVLAPolicyAdapter(
            model_key=policy_key,
            checkpoint=checkpoint,
            precision=precision,
            revision=revision,
            smolvla_inference=smolvla_inference,
            **({} if quant_config is None else quant_config),
        )
    raise ValueError(f"unsupported service policy: {policy_key}")

def _run(
    yaml_path: Path,
    config_root: Path,
    output_root: Path | None,
    run_callable: Callable[..., object] | None,
) -> int:
    try:
        Catalog.load(config_root)
    except (CatalogError, OSError) as exc:
        del exc
        print("catalog: unavailable", file=sys.stderr)
        return 3

    report = validate_config(yaml_path, config_root, runtime_checks=False)
    if not report.ok:
        return _report_validation_failure(report)

    assert report.resolved_spec is not None
    spec = report.resolved_spec
    status = TerminalStatus()
    unavailable_category = _unavailable_category(spec)
    if unavailable_category is not None:
        return _report_unavailable(status, unavailable_category)

    try:
        backend = _build_backend(spec)
    except ExecutionUnavailableError as exc:
        return _report_unavailable(status, exc.category)
    except Exception:
        return _report_unavailable(status, "execution_failed")

    task_name_resolver = _task_name_resolver(backend) if spec.benchmark.backend == "libero" else None
    initial_state_resolver = (
        _initial_state_resolver(backend)
        if spec.benchmark.backend == 'libero'
        else None
    )
    report = validate_config(
        yaml_path,
        config_root,
        check_network=True,
        task_name_resolver=task_name_resolver,
        initial_state_resolver=initial_state_resolver,
    )
    if not report.ok:
        return _report_validation_failure(report)
    assert report.resolved_spec is not None
    spec = report.resolved_spec
    try:
        dependencies = RunnerDependencies(
            backend=backend,
            policy=_build_policy(spec),
            recorder=RunRecorder(output_root or _default_output_root()),
            source_path=Path(spec.source_path),
            git_commit=_current_git_commit(Path(spec.source_path)),
            event_handler=status.on_event,
        )
        outcome = (run_callable or run_experiment)(spec, dependencies)
    except ExecutionUnavailableError as exc:
        return _report_unavailable(status, exc.category)
    except Exception as exc:
        del exc
        return _report_unavailable(status, "execution_failed")

    outcome_status = getattr(outcome, "status", None)
    if outcome_status not in {"completed", "failed", "stopped"}:
        print("run: runner returned an invalid outcome status", file=sys.stderr)
        return 3
    if not status.terminal_seen:
        status.on_event(
            {
                "event": f"run_{outcome_status}",
                "failure_type": outcome_status,
                "log_path": "run.log",
            }
        )
    return {"completed": 0, "failed": 4, "stopped": 5}[outcome_status]


def _current_git_commit(source_path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        return None
    return commit


def _build_backend(spec):
    if spec.benchmark.backend == "fake":
        from .backends.fake_backend import FakeBackend

        return FakeBackend()
    if spec.benchmark.backend == "libero":
        from .backends.libero_backend import LiberoBackend

        return LiberoBackend(
            dataset_directory=Path(spec.dataset_directory),
            settle_steps=spec.benchmark.settle_steps,
            initial_state_source=spec.benchmark.initial_state_source,
        )
    raise ValueError(f"unsupported backend: {spec.benchmark.backend}")


def _build_policy(spec):
    if spec.policy_adapter == 'demo_replay':
        from .policies.demo_replay_policy import DemoReplayPolicyAdapter

        return DemoReplayPolicyAdapter(
            spec.policy.key, Path(spec.dataset_directory)
        )
    if spec.policy_adapter == "zero":
        from .policies.zero_policy import ZeroPolicyAdapter

        return ZeroPolicyAdapter(spec.policy.key)
    if spec.policy_adapter == "remote_http":
        import requests

        from .policies.remote_http import RemoteHTTPPolicyAdapter

        if spec.policy_endpoint is None:
            raise ValueError("remote HTTP policy requires an endpoint")
        return RemoteHTTPPolicyAdapter(spec.policy.key, spec.policy_endpoint, requests.Session())
    if spec.policy_adapter == "smolvla":
        from .policies.smolvla_policy import SmolVLAPolicyAdapter

        return SmolVLAPolicyAdapter(
            model_key=spec.policy.key,
            checkpoint=spec.resolved_checkpoint,
            precision=spec.policy.precision,
            revision=spec.resolved_revision,
            action_control=getattr(spec.policy, "action_control", None),
            smolvla_inference=getattr(spec.policy, "smolvla_inference", None),
        )
    raise ExecutionUnavailableError("policy_adapter_unavailable")


def _unavailable_category(spec) -> str | None:
    if (
        spec.policy_adapter in {"demo_replay", "smolvla"}
        and spec.benchmark.backend == 'libero'
    ):
        return None
    if spec.policy_adapter in {"demo_replay", "smolvla"}:
        return "policy_adapter_unavailable"
    return None


def _report_validation_failure(report) -> int:
    issue = report.issues[0]
    print(f"{issue.field}: {issue.failure_type}", file=sys.stderr)
    return 2 if issue.failure_type == "config_error" else 3


def _report_unavailable(status: TerminalStatus, category: str) -> int:
    status.on_event(
        {
            "event": "run_failed",
            "failure_type": category,
            "log_path": "unavailable",
        }
    )
    return 4


def _list_catalog(args: argparse.Namespace, config_root: Path) -> int:
    try:
        catalog = Catalog.load(config_root)
        if args.list_command == "suites":
            _print_keys(catalog.list_suites())
        elif args.list_command == "policies":
            _print_keys(catalog.list_policies())
        elif args.list_command == "deployments":
            _print_keys(catalog.list_deployments())
        else:
            suite = next(
                row for row in catalog.list_suites() if row["key"] == args.suite
            )
            print("\n".join(str(task_id) for task_id in range(suite["task_count"])))
    except (CatalogError, OSError, StopIteration) as exc:
        del exc
        print("catalog: unavailable", file=sys.stderr)
        return 3
    return 0


def _print_keys(rows: Sequence[Mapping[str, object]]) -> None:
    print("\n".join(str(row["key"]) for row in rows))


def _task_name_resolver(backend):
    def resolve(suite: str, task_id: int) -> str:
        for task in backend.list_tasks(suite):
            if task.get("task_id") == task_id:
                task_name = task.get("task_name")
                if isinstance(task_name, str):
                    return task_name
        raise ValueError(f"LIBERO task {task_id} is unavailable for suite {suite}")

    return resolve


def _initial_state_resolver(backend):
    task_name_resolver = _task_name_resolver(backend)

    def resolve(suite: str, task_id: int, initial_state_id: int):
        official_reader = getattr(backend, "resolve_initial_state", None)
        if callable(official_reader):
            return official_reader(suite, task_id, initial_state_id)
        reader = getattr(backend, '_read_initial_state', None)
        if not callable(reader):
            raise ValueError('LIBERO backend cannot resolve demonstration states')
        task_name = task_name_resolver(suite, task_id)
        return reader(task_name, initial_state_id)

    return resolve
