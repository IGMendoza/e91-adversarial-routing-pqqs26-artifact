"""Exp3 and oracle policies for online SeQUeNCe runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sequence_game.game_theory import Exp3

DENIAL_OUTCOMES = frozenset({
    "chsh_abort",
    "qber_abort",
    "delivery_failure",
})


class Policy(Protocol):
    def sample(self) -> int: ...
    def update(self, reward: float) -> None: ...
    def current_strategy(self) -> np.ndarray: ...
    def empirical_strategy(self) -> np.ndarray: ...


@dataclass
class Exp3Policy:
    learner: Exp3

    def sample(self) -> int:
        return self.learner.sample()

    def update(self, reward: float) -> None:
        self.learner.update(_clip01(reward))

    def current_strategy(self) -> np.ndarray:
        return self.learner.policy()

    def empirical_strategy(self) -> np.ndarray:
        return self.learner.empirical_strategy()

    def schedule_at_turn(self, turn: int):
        return self.learner.schedule_at_turn(turn)

    @property
    def last_schedule(self):
        return self.learner.last_schedule


class OraclePolicy:
    def __init__(self, strategy: np.ndarray, rng: np.random.Generator):
        probs = np.asarray(strategy, dtype=float)
        if probs.ndim != 1 or probs.size < 1:
            raise ValueError("oracle strategy must be a non-empty vector")
        total = probs.sum()
        if total <= 0:
            probs = np.full(probs.size, 1.0 / probs.size)
        else:
            probs = probs / total
        self._strategy = probs
        self._rng = rng
        self._counts = np.zeros(probs.size, dtype=float)

    def sample(self) -> int:
        index = int(self._rng.choice(len(self._strategy), p=self._strategy))
        self._counts[index] += 1
        return index

    def update(self, reward: float) -> None:
        _ = reward

    def current_strategy(self) -> np.ndarray:
        return self._strategy.copy()

    def empirical_strategy(self) -> np.ndarray:
        total = self._counts.sum()
        if total <= 0:
            return self.current_strategy()
        return self._counts / total


class CautiousGreedyAlicePolicy:
    """Deterministic Alice control: best clean route, avoid public denials."""

    def __init__(self, route_scores: np.ndarray, *, avoidance_horizon: int):
        scores = np.asarray(route_scores, dtype=float)
        if scores.ndim != 1 or scores.size < 1:
            raise ValueError("cautious_greedy requires non-empty route_scores")
        if avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        self._scores = scores
        self._order = np.asarray(
            sorted(range(scores.size), key=lambda index: (-scores[index], index)),
            dtype=int,
        )
        self._avoidance_horizon = int(avoidance_horizon)
        self._avoided_until = np.full(scores.size, -1, dtype=int)
        self._counts = np.zeros(scores.size, dtype=float)
        self._turn = 0
        self._last_a: int | None = None
        self._last_fallback = False

    def sample(self) -> int:
        available = [
            int(index) for index in self._order
            if self._turn > self._avoided_until[int(index)]
        ]
        self._last_fallback = not available
        action = int((available or list(self._order))[0])
        self._last_a = action
        self._counts[action] += 1
        return action

    def update(self, reward: float) -> None:
        _ = reward

    def observe_public_outcome(self, public_outcome: str) -> None:
        if self._last_a is None:
            raise RuntimeError("observe_public_outcome() called before sample()")
        if (
                public_outcome in DENIAL_OUTCOMES
                and self._avoidance_horizon > 0
        ):
            self._avoided_until[self._last_a] = (
                self._turn + self._avoidance_horizon
            )
        self._turn += 1

    def current_strategy(self) -> np.ndarray:
        strategy = np.zeros_like(self._scores, dtype=float)
        available = [
            int(index) for index in self._order
            if self._turn > self._avoided_until[int(index)]
        ]
        strategy[int((available or list(self._order))[0])] = 1.0
        return strategy

    def empirical_strategy(self) -> np.ndarray:
        total = self._counts.sum()
        if total <= 0:
            return self.current_strategy()
        return self._counts / total

    @property
    def last_fallback(self) -> bool:
        return self._last_fallback


def make_policy(mode: str, num_actions: int, *, oracle_strategy: np.ndarray,
                gamma: float, rng: np.random.Generator,
                prior: np.ndarray | None = None,
                route_scores: np.ndarray | None = None,
                avoidance_horizon: int = 0,
                schedule_mode: str = "constant",
                eta_c: float = 1.0,
                t0: float = 10000.0,
                gamma_max: float = 0.20) -> Policy:
    if mode == "exp3_bandit":
        return Exp3Policy(Exp3(
            num_actions,
            gamma=gamma,
            rng=rng,
            schedule_mode=schedule_mode,
            eta_c=eta_c,
            t0=t0,
            gamma_max=gamma_max,
            prior=prior,
        ))
    if mode == "oracle":
        return OraclePolicy(oracle_strategy, rng)
    if mode == "cautious_greedy":
        if route_scores is None:
            raise ValueError("cautious_greedy requires route_scores")
        if len(route_scores) != num_actions:
            raise ValueError("route_scores length must match num_actions")
        return CautiousGreedyAlicePolicy(
            route_scores,
            avoidance_horizon=avoidance_horizon,
        )
    raise ValueError(f"unsupported policy mode {mode!r}")


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
