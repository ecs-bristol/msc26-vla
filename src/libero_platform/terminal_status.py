from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TextIO


class TerminalStatus:
    """Render runner lifecycle events without taking ownership of a run."""

    def __init__(
        self, *, output: TextIO | None = None, error: TextIO | None = None
    ) -> None:
        self._output = output or sys.stdout
        self._error = error or sys.stderr
        self._announced = False
        self._progress_active = False
        self.terminal_seen = False

    def on_event(self, event: Mapping[str, object]) -> None:
        event_name = event.get("event")
        if not isinstance(event_name, str):
            return

        if event_name == "run_started" and not self._announced:
            self._announced = True
            print(f"Run: {event.get('name', 'unknown')}", file=self._output)
            print(f"Task: {event.get('suite', 'unknown')}", file=self._output)
            print(f"Policy: {event.get('policy_key', 'unknown')}", file=self._output)
            return

        if event_name == "step_completed":
            print(
                "\rEpisode: "
                f"{event.get('episode', 0)} / {event.get('episode_total', 0)} "
                f"| Step: {event.get('step', 0)} / {event.get('max_steps', 0)}",
                end="",
                file=self._output,
                flush=True,
            )
            self._progress_active = True
            return

        if event_name == "warning":
            self._finish_progress()
            print(f"Warning: {event.get('message', 'unknown warning')}", file=self._error)
            return

        if event_name in {"run_completed", "run_failed", "run_stopped"}:
            self._finish_progress()
            status = event_name[4:]
            print(f"Status: {status}", file=self._output)
            if status != "completed":
                failure_type = event.get("failure_type", status)
                log_path = event.get("log_path", "run.log")
                print(
                    f"Failure: {failure_type}; log: {log_path}", file=self._error
                )
            self.terminal_seen = True

    def _finish_progress(self) -> None:
        if self._progress_active:
            print(file=self._output)
            self._progress_active = False
