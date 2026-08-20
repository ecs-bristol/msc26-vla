from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from libero_platform import cli
from libero_platform.policies.base import EpisodeContext
from libero_platform.preflight import validate_demo_hdf5
from tests.test_policy_contract import request


def write_demo(
    path: Path,
    *,
    demo_id: int,
    states: np.ndarray | None = None,
    actions: np.ndarray | None = None,
) -> None:
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as dataset:
        demo = dataset.create_group(f"data/demo_{demo_id}")
        if states is not None:
            demo.create_dataset("states", data=states)
        if actions is not None:
            demo.create_dataset("actions", data=actions)


def test_preflight_checks_every_selected_demo_dataset(tmp_path: Path) -> None:
    path = tmp_path / "task_demo.hdf5"
    write_demo(
        path,
        demo_id=0,
        states=np.zeros((1, 3), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
    )
    write_demo(
        path,
        demo_id=1,
        states=np.ones((1, 3), dtype=np.float32),
    )

    validate_demo_hdf5(
        path,
        initial_state_id=0,
        backend_initial_state=np.zeros(3, dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="data/demo_1/actions"):
        validate_demo_hdf5(
            path,
            initial_state_id=1,
            backend_initial_state=np.ones(3, dtype=np.float32),
        )


def test_preflight_rejects_state_not_supplied_by_selected_backend_demo(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task_demo.hdf5"
    write_demo(
        path,
        demo_id=0,
        states=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="backend initial state"):
        validate_demo_hdf5(
            path,
            initial_state_id=0,
            backend_initial_state=np.array([9.0, 9.0, 9.0], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((2, 6), dtype=np.float32), r"shape \(N, 7\)"),
        (
            np.array([[0, 0, 0, 0, 0, np.nan, -1]], dtype=np.float32),
            "finite",
        ),
        (
            np.array([[0, 0, 0, 0, 0, 0, -1.1]], dtype=np.float32),
            r"\[-1, 1\]",
        ),
    ],
)
def test_preflight_rejects_invalid_actions(
    tmp_path: Path, actions: np.ndarray, message: str
) -> None:
    path = tmp_path / "task_demo.hdf5"
    write_demo(
        path,
        demo_id=0,
        states=np.zeros((1, 3), dtype=np.float32),
        actions=actions,
    )

    with pytest.raises(RuntimeError, match=message):
        validate_demo_hdf5(
            path,
            initial_state_id=0,
            backend_initial_state=np.zeros(3, dtype=np.float32),
        )


def test_cli_routes_demo_replay_to_resolved_dataset(tmp_path: Path) -> None:
    path = tmp_path / "pick_the_block_demo.hdf5"
    actions = np.array([[0.5, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
    write_demo(
        path,
        demo_id=0,
        states=np.zeros((1, 3), dtype=np.float32),
        actions=actions,
    )
    spec = SimpleNamespace(
        policy_adapter="demo_replay",
        policy=SimpleNamespace(key="oracle_or_scripted"),
        benchmark=SimpleNamespace(backend="libero"),
        dataset_directory=str(tmp_path),
    )

    policy = cli._build_policy(spec)
    policy.begin_episode(
        EpisodeContext("libero_spatial", 0, "pick_the_block", 0, 42)
    )

    np.testing.assert_allclose(policy.predict(request()).action, actions[0])
    assert cli._unavailable_category(spec) is None


def test_cli_backend_state_resolver_uses_exact_task_and_demo() -> None:
    calls: list[tuple[str, int]] = []

    class Backend:
        def list_tasks(self, suite: str) -> list[dict[str, object]]:
            assert suite == "libero_spatial"
            return [{"task_id": 3, "task_name": "selected_task"}]

        def _read_initial_state(
            self, task_name: str, initial_state_id: int
        ) -> np.ndarray:
            calls.append((task_name, initial_state_id))
            return np.array([3.0, 7.0], dtype=np.float32)

    resolver = cli._initial_state_resolver(Backend())

    state = resolver("libero_spatial", 3, 7)

    np.testing.assert_array_equal(state, [3.0, 7.0])
    assert calls == [("selected_task", 7)]
