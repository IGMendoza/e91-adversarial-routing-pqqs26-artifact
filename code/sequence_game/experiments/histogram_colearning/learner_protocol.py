"""Player-neutral scalar-feedback learner contract and runtime adapter.

The histogram game intentionally gives Alice and Eve different action
semantics and different public rewards.  Their online learners nevertheless
share the same narrow protocol: select one action, observe one scalar reward,
and update from that reward only.  Keeping that protocol explicit prevents the
experiment runner from hard-wiring one player to a concrete algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


PlayerRole = Literal["alice", "eve"]


@runtime_checkable
class ScalarFeedbackLearner(Protocol):
    """Minimum online interface used by histogram co-learning.

    Implementations may maintain finite arms, topology features, or an
    implicit path polytope.  None of those details cross this boundary.
    """

    k: int

    def sample(self) -> int:
        """Select one action and retain the behavior state for its update."""

    def update(self, reward: float) -> None:
        """Update from the selected action's scalar reward only."""

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        """Return the current behavior policy over declared actions."""

    def empirical_strategy(self) -> np.ndarray:
        """Return normalized action frequencies over completed turns."""


@runtime_checkable
class ConditionableScalarFeedbackLearner(Protocol):
    """Learner that can condition its next update on an observed selected action.

    This is used by an external observer maintaining a hypothesis about an
    Alice learner. The action is not chosen for Alice; it is the action Alice
    actually selected. The method must register exactly the same behavior
    policy, estimator state, and action count as the learner's own sampler,
    conditional on that action having been drawn.
    """

    def select_observed(self, action: int) -> float:
        """Register an observed selected action and return its policy mass."""


@runtime_checkable
class BehaviorConditionableScalarFeedbackLearner(Protocol):
    """Learner supporting selection from an externally declared behavior."""

    def sample_from_behavior(self, behavior_policy: np.ndarray) -> int:
        """Sample and register an action using the declared behavior policy."""


@dataclass
class PlayerLearner:
    """Validated, timed binding of one learner to one player's action set."""

    player: PlayerRole
    action_ids: tuple[str, ...]
    learner: ScalarFeedbackLearner
    metadata: dict[str, Any]
    sample_calls: int = 0
    update_calls: int = 0
    sample_seconds: float = 0.0
    update_seconds: float = 0.0
    behavior_conditioned_sample_calls: int = 0
    _last_reward: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.player not in {"alice", "eve"}:
            raise ValueError(f"unsupported player role {self.player!r}")
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("learner action_ids must be non-empty and unique")
        if not isinstance(self.learner, ScalarFeedbackLearner):
            raise TypeError(
                "learner must implement sample, update, policy, and "
                "empirical_strategy"
            )
        if int(self.learner.k) != len(self.action_ids):
            raise ValueError(
                "learner action count does not match the declared action ids"
            )
        for key in ("mode", "algorithm", "feedback"):
            if not str(self.metadata.get(key) or "").strip():
                raise ValueError(f"learner metadata requires {key!r}")
        self._validated_distribution(self.learner.policy(turn=1), "initial policy")
        self.metadata = {
            **self.metadata,
            "player": self.player,
            "protocol": "selected_action_scalar_reward_v1",
            "action_count": len(self.action_ids),
            "opponent_action_disclosed": False,
            "counterfactual_payoff_cells_disclosed": False,
        }

    def sample(self) -> int:
        started = time.perf_counter()
        action = int(self.learner.sample())
        self.sample_seconds += time.perf_counter() - started
        self.sample_calls += 1
        if not 0 <= action < len(self.action_ids):
            raise IndexError(
                f"{self.player} learner selected undeclared action index {action}"
            )
        return action

    def update(self, reward: float) -> None:
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("learner reward must be finite")
        started = time.perf_counter()
        self.learner.update(value)
        self.update_seconds += time.perf_counter() - started
        self.update_calls += 1
        self._last_reward = value

    def sample_from_behavior(
        self,
        behavior_policy: np.ndarray,
    ) -> int:
        """Select from a controller policy while retaining a valid update."""

        if not isinstance(
            self.learner,
            BehaviorConditionableScalarFeedbackLearner,
        ):
            raise TypeError(
                f"{self.metadata['algorithm']} does not expose "
                "behavior-conditioned scalar-feedback updates"
            )
        behavior = self._validated_distribution(
            behavior_policy,
            "controller behavior policy",
        )
        started = time.perf_counter()
        action = int(self.learner.sample_from_behavior(behavior))
        self.sample_seconds += time.perf_counter() - started
        self.sample_calls += 1
        self.behavior_conditioned_sample_calls += 1
        if not 0 <= action < len(self.action_ids):
            raise IndexError(
                f"{self.player} learner selected undeclared action index "
                f"{action}"
            )
        return action

    def condition_on_observation(
        self,
        action: int,
        reward: float,
    ) -> float:
        """Advance one learner hypothesis from an observed action and reward."""

        if not isinstance(self.learner, ConditionableScalarFeedbackLearner):
            raise TypeError(
                f"{self.metadata['algorithm']} does not expose exact "
                "selected-action conditioning"
            )
        index = int(action)
        if not 0 <= index < len(self.action_ids):
            raise IndexError("observed action index is outside the learner")
        started = time.perf_counter()
        probability = float(self.learner.select_observed(index))
        self.sample_seconds += time.perf_counter() - started
        self.sample_calls += 1
        if (
            not math.isfinite(probability)
            or probability <= 0.0
            or probability > 1.0
        ):
            raise FloatingPointError(
                "conditioned action probability must lie in (0, 1]"
            )
        self.update(reward)
        return probability

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        return self._validated_distribution(
            self.learner.policy(turn=turn),
            "current policy",
        )

    def empirical_strategy(self) -> np.ndarray:
        return self._validated_distribution(
            self.learner.empirical_strategy(),
            "empirical strategy",
        )

    @property
    def last_schedule(self) -> Any | None:
        return getattr(self.learner, "last_schedule", None)

    def runtime_telemetry(self, *, turn: int) -> dict[str, Any]:
        policy = self.policy(turn=turn)
        empirical = self.empirical_strategy()
        policy_statistics = policy_distribution_statistics(policy)
        empirical_statistics = policy_distribution_statistics(empirical)
        return {
            "schema": "scalar_feedback_player_telemetry.v2",
            "player": self.player,
            "protocol": self.metadata["protocol"],
            "action_ids": list(self.action_ids),
            "action_count": len(self.action_ids),
            "sample_calls": int(self.sample_calls),
            "update_calls": int(self.update_calls),
            "behavior_conditioned_sample_calls": int(
                self.behavior_conditioned_sample_calls
            ),
            "sample_seconds": float(self.sample_seconds),
            "update_seconds": float(self.update_seconds),
            "mean_sample_microseconds": (
                1e6 * self.sample_seconds / self.sample_calls
                if self.sample_calls else None
            ),
            "mean_update_microseconds": (
                1e6 * self.update_seconds / self.update_calls
                if self.update_calls else None
            ),
            "last_reward": self._last_reward,
            "current_policy": policy.tolist(),
            "empirical_strategy": empirical.tolist(),
            "current_policy_entropy_nats": policy_statistics["entropy_nats"],
            "current_policy_minimum_action_probability": policy_statistics[
                "minimum_action_probability"
            ],
            "current_policy_uniform_coverage_mass": policy_statistics[
                "uniform_coverage_mass"
            ],
            "current_policy_nonmodal_probability_mass": policy_statistics[
                "nonmodal_probability_mass"
            ],
            "current_policy_effective_action_count": policy_statistics[
                "effective_action_count"
            ],
            "current_policy_effective_action_fraction": policy_statistics[
                "effective_action_fraction"
            ],
            "empirical_strategy_entropy_nats": empirical_statistics[
                "entropy_nats"
            ],
            "empirical_strategy_minimum_action_probability": (
                empirical_statistics["minimum_action_probability"]
            ),
            "empirical_strategy_uniform_coverage_mass": empirical_statistics[
                "uniform_coverage_mass"
            ],
            "empirical_strategy_nonmodal_probability_mass": (
                empirical_statistics["nonmodal_probability_mass"]
            ),
            "empirical_strategy_effective_action_count": empirical_statistics[
                "effective_action_count"
            ],
            "empirical_strategy_effective_action_fraction": (
                empirical_statistics["effective_action_fraction"]
            ),
        }

    def _validated_distribution(
        self,
        values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        distribution = np.asarray(values, dtype=float)
        if distribution.shape != (len(self.action_ids),):
            raise ValueError(
                f"{self.player} {name} has shape {distribution.shape}, expected "
                f"{(len(self.action_ids),)}"
            )
        if (
            not np.all(np.isfinite(distribution))
            or np.any(distribution < -1e-14)
            or not math.isclose(
                float(np.sum(distribution)),
                1.0,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ):
            raise FloatingPointError(f"invalid {self.player} {name}")
        return distribution


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=float)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def policy_distribution_statistics(
    distribution: np.ndarray,
) -> dict[str, float]:
    """Return exact concentration and all-action coverage statistics.

    For a distribution ``p`` over ``K`` actions, ``K * min_i p_i`` is the
    largest ``gamma`` for which ``p = (1-gamma)q + gamma u`` with a simplex
    distribution ``q`` and uniform distribution ``u``. It is a property of
    the realized policy, not necessarily an exploration coefficient selected
    by the learner.
    """

    values = np.asarray(distribution, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("policy distribution must be a non-empty vector")
    if (
        not np.all(np.isfinite(values))
        or np.any(values < -1e-14)
        or not math.isclose(
            float(np.sum(values)),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("policy distribution must lie on the probability simplex")
    clipped = np.maximum(values, 0.0)
    entropy = _entropy(clipped)
    action_count = int(clipped.size)
    minimum = float(np.min(clipped))
    effective_count = float(math.exp(entropy))
    return {
        "entropy_nats": entropy,
        "minimum_action_probability": minimum,
        "uniform_coverage_mass": float(min(1.0, action_count * minimum)),
        "nonmodal_probability_mass": float(1.0 - np.max(clipped)),
        "effective_action_count": effective_count,
        "effective_action_fraction": float(effective_count / action_count),
    }
