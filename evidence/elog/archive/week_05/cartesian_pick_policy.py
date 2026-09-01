from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import get_zero_action


@dataclass
class PickControlConfig:
    hover_height: float = 0.12
    grasp_offset_z: float = 0.035
    lift_height: float = 0.18
    position_tolerance: float = 0.012
    close_steps: int = 40
    max_translation_per_step: float = 0.05


class CartesianPickPolicy:
    """A small state machine that sends Cartesian deltas to robosuite's OSC controller."""

    def __init__(self, target_position: np.ndarray, config: PickControlConfig) -> None:
        self.target_position = np.asarray(target_position, dtype=np.float64)
        self.config = config
        self.phase = "approach"
        self.close_step_count = 0

    @property
    def is_complete(self) -> bool:
        return self.phase == "complete"

    def next_action(self, env: Any, observation: dict[str, Any]) -> np.ndarray:
        eef_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        if self.phase == "approach":
            goal = self.target_position + np.array([0.0, 0.0, self.config.hover_height])
            action = self._move_towards(env, eef_position, goal, close_gripper=False)
            if np.linalg.norm(goal - eef_position) < self.config.position_tolerance:
                self.phase = "descend"
            return action

        if self.phase == "descend":
            goal = self.target_position + np.array([0.0, 0.0, self.config.grasp_offset_z])
            action = self._move_towards(env, eef_position, goal, close_gripper=False)
            if np.linalg.norm(goal - eef_position) < self.config.position_tolerance:
                self.phase = "close"
            return action

        if self.phase == "close":
            goal = self.target_position + np.array([0.0, 0.0, self.config.grasp_offset_z])
            action = self._move_towards(env, eef_position, goal, close_gripper=True)
            if np.linalg.norm(goal - eef_position) < self.config.position_tolerance:
                self.close_step_count += 1
                if self.close_step_count >= self.config.close_steps:
                    self.phase = "lift"
            return action

        if self.phase == "lift":
            goal = self.target_position + np.array([0.0, 0.0, self.config.lift_height])
            action = self._move_towards(env, eef_position, goal, close_gripper=True)
            if np.linalg.norm(goal - eef_position) < self.config.position_tolerance:
                self.phase = "complete"
            return action

        action = get_zero_action(env)
        action[-1] = 1.0
        return action

    def _move_towards(
        self,
        env: Any,
        current_position: np.ndarray,
        goal_position: np.ndarray,
        *,
        close_gripper: bool,
    ) -> np.ndarray:
        action = get_zero_action(env)
        delta = (goal_position - current_position) / self.config.max_translation_per_step
        action[:3] = np.clip(delta, -1.0, 1.0)
        action[-1] = 1.0 if close_gripper else -1.0
        low, high = env.action_spec
        return np.clip(action, low, high).astype(np.float32)
