from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from libero_platform.policies.base import EpisodeContext
from libero_platform.policies.demo_replay_policy import (
    DemoReplayPolicyAdapter,
    ReferenceActionsExhaustedError,
)
from tests.test_policy_contract import request


def write_demo(
    path: Path,
    actions: np.ndarray,
    *,
    demo_id: int = 0,
    states: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as handle:
        demo = handle.create_group(f"data/demo_{demo_id}")
        demo.create_dataset(
            "states",
            data=(
                np.zeros((1, 16), dtype=np.float32)
                if states is None
                else states
            ),
        )
        demo.create_dataset("actions", data=actions)


def context(initial_state_id: int = 0) -> EpisodeContext:
    return EpisodeContext(
        suite="libero_spatial",
        task_id=0,
        task_name="pick_the_block",
        initial_state_id=initial_state_id,
        seed=42,
    )


def test_replays_exact_actions_for_selected_demo(tmp_path: Path) -> None:
    demo_path = tmp_path / "libero_spatial" / "pick_the_block_demo.hdf5"
    actions = np.array(
        [[0, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    write_demo(demo_path, actions)
    adapter = DemoReplayPolicyAdapter(
        model_key="oracle_or_scripted",
        dataset_directory=tmp_path / "libero_spatial",
    )

    adapter.begin_episode(context())

    first = adapter.predict(request())
    second = adapter.predict(request())
    np.testing.assert_allclose(first.action, actions[0])
    np.testing.assert_allclose(second.action, actions[1])
    assert first.model_key == "oracle_or_scripted"
    assert first.device == "cpu"


def test_selects_initial_state_demo_and_resets_cursor(tmp_path: Path) -> None:
    demo_path = tmp_path / "pick_the_block_demo.hdf5"
    write_demo(
        demo_path,
        np.array([[np.nan] * 7], dtype=np.float32),
        demo_id=0,
    )
    selected = np.array(
        [[0.25, 0, 0, 0, 0, 0, -1], [0.5, 0, 0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    write_demo(demo_path, selected, demo_id=1)
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)

    adapter.begin_episode(context(initial_state_id=1))
    np.testing.assert_allclose(adapter.predict(request()).action, selected[0])
    np.testing.assert_allclose(adapter.predict(request()).action, selected[1])
    adapter.begin_episode(context(initial_state_id=1))

    np.testing.assert_allclose(adapter.predict(request()).action, selected[0])


def test_exhausted_demo_raises_typed_failure(tmp_path: Path) -> None:
    demo_path = tmp_path / "pick_the_block_demo.hdf5"
    write_demo(
        demo_path,
        np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=np.float32),
    )
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)
    adapter.begin_episode(context())
    adapter.predict(request())

    with pytest.raises(
        ReferenceActionsExhaustedError, match="reference_actions_exhausted"
    ) as error:
        adapter.predict(request())

    assert error.value.failure_type == "reference_actions_exhausted"


def test_missing_task_file_is_rejected(tmp_path: Path) -> None:
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)

    with pytest.raises(RuntimeError, match="demonstration file"):
        adapter.begin_episode(context())


def test_missing_selected_demo_group_is_rejected(tmp_path: Path) -> None:
    demo_path = tmp_path / "pick_the_block_demo.hdf5"
    write_demo(
        demo_path,
        np.zeros((1, 7), dtype=np.float32),
        demo_id=1,
    )
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)

    with pytest.raises(RuntimeError, match="data/demo_0"):
        adapter.begin_episode(context())


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (np.zeros((2, 6), dtype=np.float32), r"shape \(N, 7\)"),
        (
            np.array([[0, 0, 0, 0, 0, np.nan, -1]], dtype=np.float32),
            "finite",
        ),
        (
            np.array([[0, 0, 0, 0, 0, 0, 1.01]], dtype=np.float32),
            r"\[-1, 1\]",
        ),
    ],
)
def test_invalid_selected_actions_are_rejected(
    tmp_path: Path, actions: np.ndarray, message: str
) -> None:
    demo_path = tmp_path / "pick_the_block_demo.hdf5"
    write_demo(demo_path, actions)
    adapter = DemoReplayPolicyAdapter("oracle_or_scripted", tmp_path)

    with pytest.raises(RuntimeError, match=message):
        adapter.begin_episode(context())
