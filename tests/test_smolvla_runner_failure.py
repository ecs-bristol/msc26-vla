from __future__ import annotations

from pathlib import Path

from libero_platform.backends.fake_backend import FakeBackend
from libero_platform.policies.base import PolicyAdapter, PolicyRequest, PolicyResponse
from libero_platform.policies.smolvla_policy import SmolVLAPolicyRuntimeError
from libero_platform.recorder import RunRecorder
from libero_platform.runner import RunnerDependencies, run_experiment
from libero_platform.spec import ResolvedExperimentSpec


def test_runner_preserves_smolvla_load_failure_type_in_terminal_evidence(
    tmp_path: Path,
) -> None:
    class FailingSmolVLAPolicy(PolicyAdapter):
        def load(self) -> None:
            raise SmolVLAPolicyRuntimeError('model_load_error', 'checkpoint unavailable')

        def predict(self, request: PolicyRequest) -> PolicyResponse:
            raise AssertionError('predict should not be called')

    source_path = tmp_path / 'experiment.yaml'
    source_path.write_text('schema_version: 1\nname: smolvla_failure\n', encoding='utf-8')
    spec = ResolvedExperimentSpec.model_validate(
        {
            'schema_version': 1,
            'name': 'smolvla_failure',
            'benchmark': {
                'backend': 'fake',
                'suite': 'libero_spatial',
                'task_ids': [0],
                'initial_state_ids': [0],
                'max_steps': 1,
            },
            'policy': {
                'key': 'smolvla_libero',
                'checkpoint': 'lerobot/smolvla_libero',
                'precision': 'fp16',
                'quantization': 'none',
            },
            'deployment': {'mode': 'pc_local', 'profile': 'pc_default'},
            'execution': {
                'episodes_per_initial_state': 1,
                'warmup_episodes': 0,
                'seed': 42,
                'on_episode_failure': 'continue',
            },
            'viewer': {'enabled': False},
            'recording': {
                'save_frames': False,
                'save_video': False,
                'frame_stride': 20,
                'save_steps': True,
            },
            'source_path': str(source_path),
            'dataset_directory': 'datasets/libero',
            'resolved_checkpoint': 'lerobot/smolvla_libero',
            'policy_adapter': 'smolvla',
        }
    )
    events: list[dict[str, object]] = []
    recorder = RunRecorder(tmp_path / 'outputs')

    outcome = run_experiment(
        spec,
        RunnerDependencies(
            backend=FakeBackend(),
            policy=FailingSmolVLAPolicy(),
            recorder=recorder,
            source_path=source_path,
            event_handler=events.append,
        ),
    )

    assert (outcome.status, outcome.result_integrity) == ('failed', 'partial')
    assert recorder.read_manifest(outcome.run_id)['error'] == {
        'failure_type': 'model_load_error'
    }
    assert events[-1]['failure_type'] == 'model_load_error'
