from __future__ import annotations

import builtins

import pytest

from libero_platform.viewer_bridge import PassiveViewerBridge


class FakeViewerHandle:
    def __init__(self) -> None:
        self.running = True
        self.sync_calls = 0
        self.close_calls = 0

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.sync_calls += 1

    def close(self) -> None:
        self.close_calls += 1
        self.running = False


def test_disabled_bridge_never_imports_mujoco(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mujoco.viewer":
            raise AssertionError("disabled Viewer must not import mujoco")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    bridge = PassiveViewerBridge(object(), object(), enabled=False)

    bridge.open()
    bridge.sync()
    bridge.close()


def test_open_sync_manual_close_and_explicit_reopen() -> None:
    handles: list[FakeViewerHandle] = []
    warnings: list[str] = []

    def launcher(model: object, data: object) -> FakeViewerHandle:
        assert (model, data) == ("model", "data")
        handle = FakeViewerHandle()
        handles.append(handle)
        return handle

    bridge = PassiveViewerBridge(
        "model", "data", enabled=True, launcher=launcher, on_warning=warnings.append
    )
    bridge.open()
    bridge.sync()
    handles[0].running = False
    bridge.sync()
    bridge.sync()
    bridge.reopen()
    bridge.close()
    bridge.close()

    assert len(handles) == 2
    assert handles[0].sync_calls == 1
    assert handles[0].close_calls == 0
    assert handles[1].close_calls == 1
    assert len(warnings) == 1
    assert "closed" in warnings[0].lower()


def test_viewer_exceptions_become_warnings() -> None:
    warnings: list[str] = []

    def unavailable_launcher(_model: object, _data: object) -> object:
        raise RuntimeError("no display")

    bridge = PassiveViewerBridge(
        object(), object(), enabled=True, launcher=unavailable_launcher, on_warning=warnings.append
    )
    bridge.open()
    bridge.sync()
    bridge.close()

    assert warnings == ["MuJoCo Viewer open failed: no display"]
