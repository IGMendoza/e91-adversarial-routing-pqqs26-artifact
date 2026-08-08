"""Exact rolling-window reward extrema for online-learning telemetry."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Sequence


class RollingRewardMinimumTracker:
    """Track the minimum mean reward over declared sliding windows.

    Windows are evaluated at every turn and must lie entirely inside the
    declared evaluation interval. Ties retain the earliest window.
    """

    def __init__(
        self,
        window_turns: Sequence[int],
        *,
        evaluation_start_turn: int,
    ) -> None:
        windows = tuple(sorted({int(value) for value in window_turns}))
        if any(value < 1 for value in windows):
            raise ValueError("window_turns must contain positive integers")
        if int(evaluation_start_turn) < 1:
            raise ValueError("evaluation_start_turn must be positive")
        self.window_turns = windows
        self.evaluation_start_turn = int(evaluation_start_turn)
        self._queues = {
            window: deque() for window in self.window_turns
        }
        self._sums = {window: 0.0 for window in self.window_turns}
        self._minima: dict[int, dict[str, Any] | None] = {
            window: None for window in self.window_turns
        }
        self._last_turn = 0

    def update(self, turn: int, reward: float) -> None:
        step = int(turn)
        value = float(reward)
        if step != self._last_turn + 1:
            raise ValueError("rolling-window turns must be consecutive")
        if not math.isfinite(value):
            raise ValueError("rolling-window reward must be finite")
        self._last_turn = step
        if step < self.evaluation_start_turn:
            return

        for window in self.window_turns:
            queue = self._queues[window]
            queue.append(value)
            self._sums[window] += value
            if len(queue) > window:
                self._sums[window] -= float(queue.popleft())
            if len(queue) != window:
                continue
            mean_reward = self._sums[window] / window
            current = self._minima[window]
            if (
                current is None
                or mean_reward < float(current["mean_reward"])
            ):
                self._minima[window] = {
                    "window_turns": window,
                    "window_start_turn": step - window + 1,
                    "window_end_turn": step,
                    "mean_reward": mean_reward,
                }

    def diagnostics(self) -> list[dict[str, Any]]:
        rows = []
        for window in self.window_turns:
            row = self._minima[window]
            if row is None:
                raise RuntimeError(
                    f"no complete {window}-turn reward window was observed"
                )
            rows.append(dict(row))
        return rows
