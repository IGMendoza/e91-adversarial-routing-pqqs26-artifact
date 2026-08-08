"""Generic tabular Q-learning core.

Independent of quantum-network specifics: states are hashable tuples of
strings, actions are integer indices into a fixed action list. No SeQUeNCe or
topology imports belong here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

StateKey = tuple[str, ...]


class QLearningError(ValueError):
    """Invalid Q-learning configuration or usage."""


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float
    gamma: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise QLearningError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.gamma <= 1.0:
            raise QLearningError(f"gamma must be in [0, 1], got {self.gamma}")

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "gamma": self.gamma}


class QTable:
    """Dense-on-demand Q-table over (state tuple, action index)."""

    def __init__(self, num_actions: int):
        if num_actions < 1:
            raise QLearningError("num_actions must be >= 1")
        self.num_actions = num_actions
        self._q: dict[StateKey, list[float]] = {}

    def values(self, state: StateKey) -> list[float]:
        key = tuple(str(s) for s in state)
        if key not in self._q:
            self._q[key] = [0.0] * self.num_actions
        return self._q[key]

    def get(self, state: StateKey, action: int) -> float:
        return self.values(state)[action]

    def best_action(self, state: StateKey,
                    allowed_actions: Optional[Sequence[int]] = None) -> int:
        """Greedy action; deterministic tie-break to the lowest allowed index."""
        vals = self.values(state)
        allowed = _allowed_action_indices(self.num_actions, allowed_actions)
        return min(allowed, key=lambda index: (-vals[index], index))

    def update(self, state: StateKey, action: int, reward: float,
               next_state: Optional[StateKey], config: QLearningConfig) -> float:
        """One-step Q-learning update; returns the new Q(state, action)."""
        if not 0 <= action < self.num_actions:
            raise QLearningError(f"action {action} out of range")
        vals = self.values(state)
        target = reward
        if next_state is not None:
            target += config.gamma * max(self.values(next_state))
        vals[action] += config.alpha * (target - vals[action])
        return vals[action]

    def to_dict(self) -> dict:
        return {
            "num_actions": self.num_actions,
            "entries": [{"state": list(state), "q": list(vals)}
                        for state, vals in sorted(self._q.items())],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QTable":
        table = cls(data["num_actions"])
        for entry in data["entries"]:
            table._q[tuple(entry["state"])] = [float(v) for v in entry["q"]]
        return table

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "QTable":
        return cls.from_dict(json.loads(text))


def epsilon_greedy(table: QTable, state: StateKey, epsilon: float,
                   rng: np.random.Generator,
                   allowed_actions: Optional[Sequence[int]] = None) -> int:
    """Explicit-RNG epsilon-greedy action selection."""
    if not 0.0 <= epsilon <= 1.0:
        raise QLearningError(f"epsilon must be in [0, 1], got {epsilon}")
    if rng is None:
        raise QLearningError("epsilon_greedy requires an explicit rng")
    allowed = _allowed_action_indices(table.num_actions, allowed_actions)
    if rng.random() < epsilon:
        return int(allowed[int(rng.integers(len(allowed)))])
    return table.best_action(state, allowed)


def linear_epsilon_schedule(start: float, end: float, episodes: int):
    """Per-episode epsilon, linear from start to end inclusive."""
    if episodes < 1:
        raise QLearningError("episodes must be >= 1")
    if episodes == 1:
        return lambda episode: float(start)
    span = end - start
    return lambda episode: float(start + span * min(episode, episodes - 1) / (episodes - 1))


def _allowed_action_indices(num_actions: int,
                            allowed_actions: Optional[Sequence[int]]) -> tuple[int, ...]:
    if allowed_actions is None:
        return tuple(range(num_actions))
    allowed = tuple(int(action) for action in allowed_actions)
    if not allowed:
        raise QLearningError("allowed_actions must be non-empty")
    bad = [action for action in allowed if action < 0 or action >= num_actions]
    if bad:
        raise QLearningError(f"allowed action out of range: {bad}")
    return tuple(dict.fromkeys(allowed))
