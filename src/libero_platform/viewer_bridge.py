from __future__ import annotations

from typing import Any, Callable


ViewerLauncher = Callable[[object, object], object]


def _launch_passive(model: object, data: object) -> object:
    import mujoco.viewer

    return mujoco.viewer.launch_passive(model, data)


class PassiveViewerBridge:
    """Keep a native MuJoCo passive viewer optional and non-fatal to rollouts."""

    def __init__(
        self,
        model: object,
        data: object,
        enabled: bool,
        *,
        launcher: ViewerLauncher | None = None,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._enabled = enabled
        self._launcher = launcher or _launch_passive
        self._on_warning = on_warning or (lambda _message: None)
        self._handle: Any | None = None
        self._closed_by_user = False

    def open(self) -> None:
        if not self._enabled or self._handle is not None:
            return
        try:
            self._handle = self._launcher(self._model, self._data)
            self._closed_by_user = False
        except Exception as exc:  # noqa: BLE001 - display failures must not stop rollouts.
            self._handle = None
            self._warn(f"MuJoCo Viewer open failed: {exc}")

    def sync(self) -> None:
        if self._handle is None:
            return
        try:
            if not self._is_running():
                self._handle = None
                if not self._closed_by_user:
                    self._closed_by_user = True
                    self._warn("MuJoCo Viewer was closed manually; continuing without a viewer")
                return
            self._handle.sync()
        except Exception as exc:  # noqa: BLE001 - viewer failures must not stop rollouts.
            self._warn(f"MuJoCo Viewer sync failed: {exc}")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception as exc:  # noqa: BLE001 - viewer failures must not stop rollouts.
            self._warn(f"MuJoCo Viewer close failed: {exc}")

    def reopen(self) -> None:
        self.close()
        self.open()

    def _is_running(self) -> bool:
        assert self._handle is not None
        is_running = getattr(self._handle, "is_running", None)
        if callable(is_running):
            return bool(is_running())
        return bool(getattr(self._handle, "running", True))

    def _warn(self, message: str) -> None:
        try:
            self._on_warning(message)
        except Exception:
            pass
