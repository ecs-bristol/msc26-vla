from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .spec import ResolvedExperimentSpec


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    failure_type: str


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    resolved_spec: ResolvedExperimentSpec | None
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)


def validate_demo_actions(actions: object):
    import numpy as np

    try:
        value = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("LIBERO actions must contain numeric values") from exc
    if value.ndim != 2 or value.shape[1] != 7:
        raise RuntimeError("LIBERO actions must have shape (N, 7)")
    if value.shape[0] == 0:
        raise RuntimeError("LIBERO actions must contain at least one action")
    if not np.isfinite(value).all():
        raise RuntimeError("LIBERO actions must contain finite values")
    if (value < -1.0).any() or (value > 1.0).any():
        raise RuntimeError("LIBERO actions must stay in [-1, 1]")
    return value.copy()


def validate_demo_hdf5(
    path: Path,
    *,
    initial_state_id: int,
    backend_initial_state: object,
) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and numpy dependencies are required") from exc

    demo_file = Path(path)
    if not demo_file.is_file():
        raise RuntimeError(
            f"LIBERO dataset is missing demonstration file: {demo_file.name}"
        )
    group_path = f"data/demo_{initial_state_id}"
    states_path = f"{group_path}/states"
    actions_path = f"{group_path}/actions"
    try:
        with h5py.File(demo_file, "r") as dataset:
            if states_path not in dataset:
                raise RuntimeError(
                    f"LIBERO demonstration dataset is missing: {states_path}"
                )
            if actions_path not in dataset:
                raise RuntimeError(
                    f"LIBERO demonstration dataset is missing: {actions_path}"
                )
            states = dataset[states_path]
            if states.ndim < 1 or states.shape[0] == 0:
                raise RuntimeError(f"LIBERO states dataset is empty: {states_path}")
            first_state = np.asarray(states[0]).copy()
            actions = np.asarray(dataset[actions_path][...]).copy()
    except OSError as exc:
        raise RuntimeError(
            f"could not open LIBERO demonstration file: {demo_file.name}"
        ) from exc

    validate_demo_actions(actions)
    expected_state = np.asarray(backend_initial_state)
    if not np.array_equal(first_state, expected_state):
        raise RuntimeError(
            f"backend initial state does not match {states_path}[0]"
        )
