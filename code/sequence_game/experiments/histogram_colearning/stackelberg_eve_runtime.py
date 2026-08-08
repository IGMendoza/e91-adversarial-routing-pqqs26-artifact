"""Explore-then-commit Stackelberg Eve on compiled histogram outcome banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable

import numpy as np

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .learning import build_histogram_alice_learner
from .stackelberg_eve import (
    StackelbergCommitment,
    StackelbergOracleResult,
    evaluate_stackelberg_commitment,
    simultaneous_hoeffding_radii,
    solve_stackelberg_oracle,
)


STACKELBERG_LEARNER_RESULT_SCHEMA = "histogram_stackelberg_eve_run.v1"


@dataclass(frozen=True)
class StackelbergExploreCommitConfig:
    """Finite-horizon exploration and commitment contract."""

    minimum_exploration_turns: int
    maximum_exploration_turns: int
    decision_interval: int
    minimum_unique_draws_per_supported_cell: int
    confidence_delta: float = 0.05
    record_every: int = 1_000

    def __post_init__(self) -> None:
        if self.minimum_exploration_turns < 1:
            raise ValueError("minimum_exploration_turns must be positive")
        if self.maximum_exploration_turns < self.minimum_exploration_turns:
            raise ValueError(
                "maximum_exploration_turns must not precede minimum exploration"
            )
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        if self.minimum_unique_draws_per_supported_cell < 1:
            raise ValueError(
                "minimum_unique_draws_per_supported_cell must be positive"
            )
        if not math.isfinite(self.confidence_delta) or not (
            0.0 < self.confidence_delta < 1.0
        ):
            raise ValueError("confidence_delta must lie strictly between zero and one")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def validate_horizon(self, turns: int) -> None:
        if self.maximum_exploration_turns >= int(turns):
            raise ValueError(
                "maximum exploration must leave at least one commitment turn"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteActionRewardEstimator:
    """Unique-bank-draw estimator for bounded public Alice rewards."""

    def __init__(self, route_count: int, action_count: int, draws_per_cell: int):
        if min(route_count, action_count, draws_per_cell) < 1:
            raise ValueError("estimator dimensions must be positive")
        self.route_count = int(route_count)
        self.action_count = int(action_count)
        self.draws_per_cell = int(draws_per_cell)
        self._seen = np.zeros(
            (route_count, action_count, draws_per_cell),
            dtype=bool,
        )
        self._counts = np.zeros((route_count, action_count), dtype=np.int64)
        self._sums = np.zeros((route_count, action_count), dtype=float)

    def observe(
        self,
        route_index: int,
        action_index: int,
        draw_index: int,
        reward: float,
    ) -> bool:
        route = int(route_index)
        action = int(action_index)
        draw = int(draw_index)
        value = float(reward)
        if not 0 <= route < self.route_count:
            raise IndexError("route index is outside the estimator")
        if not 0 <= action < self.action_count:
            raise IndexError("action index is outside the estimator")
        if not 0 <= draw < self.draws_per_cell:
            raise IndexError("draw index is outside the estimator")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Alice public reward must lie in [0, 1]")
        if self._seen[route, action, draw]:
            return False
        self._seen[route, action, draw] = True
        self._counts[route, action] += 1
        self._sums[route, action] += value
        return True

    @property
    def counts(self) -> np.ndarray:
        return self._counts.copy()

    def means(self) -> np.ndarray:
        result = np.full(self._counts.shape, 0.5, dtype=float)
        observed = self._counts > 0
        result[observed] = self._sums[observed] / self._counts[observed]
        return result

    def radii(self, *, delta: float) -> np.ndarray:
        return simultaneous_hoeffding_radii(self._counts, delta=delta)

    def coverage_summary(self) -> dict[str, Any]:
        return {
            "unique_observations": int(np.sum(self._counts)),
            "observed_cell_count": int(np.sum(self._counts > 0)),
            "total_cell_count": int(self._counts.size),
            "fully_observed_cell_fraction": float(np.mean(self._counts > 0)),
            "minimum_unique_draws_per_cell": int(np.min(self._counts)),
            "median_unique_draws_per_cell": float(np.median(self._counts)),
            "maximum_unique_draws_per_cell": int(np.max(self._counts)),
            "draws_per_cell_available": self.draws_per_cell,
        }


def run_stackelberg_explore_then_commit(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    learner_config: StackelbergExploreCommitConfig,
    *,
    seed: int,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    feedback_oracle: StackelbergOracleResult | None = None,
    audit_oracle: StackelbergOracleResult | None = None,
) -> dict[str, Any]:
    """Learn a stationary Eve commitment from route-labelled public outcomes."""

    active_config = replace(config, seed=int(seed))
    learner_config.validate_horizon(active_config.turns)
    validate_bank_against_case(
        bank,
        case,
        active_config,
        selection_digest_blake2b_128=(
            str(selection_digest_blake2b_128)
            if selection_digest_blake2b_128 is not None
            else str(bank.metadata["selection_digest_blake2b_128"])
        ),
    )
    route_count = len(bank.route_ids)
    action_count = len(bank.action_ids)
    alice_implementation, alice_metadata = build_histogram_alice_learner(
        case=case,
        config=active_config,
        prior=np.full(route_count, 1.0 / route_count, dtype=float),
    )
    alice = PlayerLearner(
        player="alice",
        action_ids=bank.route_ids,
        learner=alice_implementation,
        metadata=alice_metadata,
    )
    stream = CommonCellOutcomeStream(
        route_count=route_count,
        action_count=action_count,
        draws_per_cell=bank.draws_per_cell,
        seed=int(seed),
    )
    estimator = RouteActionRewardEstimator(
        route_count,
        action_count,
        bank.draws_per_cell,
    )
    exploration_rng = np.random.default_rng([seed, 401])
    commitment_rng = np.random.default_rng([seed, 409])
    exploration_order = exploration_rng.permutation(action_count)
    current_commitment: StackelbergCommitment | None = None
    commitment_turn: int | None = None
    last_oracle = None
    trajectory: list[dict[str, Any]] = []
    joint_counts = np.zeros((route_count, action_count), dtype=np.int64)
    alice_reward_sum = 0.0
    eve_utility_sum = 0.0
    commit_eve_utility_sum = 0.0
    commit_turn_count = 0
    started = time.perf_counter()

    for turn_index in range(active_config.turns):
        step = turn_index + 1
        exploring = current_commitment is None
        if exploring:
            action_index = int(
                exploration_order[turn_index % exploration_order.size]
            )
        else:
            action_index = int(
                commitment_rng.choice(
                    action_count,
                    p=np.asarray(current_commitment.probabilities),
                )
            )
        route_index = alice.sample()
        draw_index = stream.next_draw(route_index, action_index)
        observation = bank.feedback.draw(route_index, action_index, draw_index)
        alice_reward = float(observation["alice_reward"])
        alice.update(alice_reward)
        eve_utility = (
            1.0 - alice_reward - float(bank.action_costs[action_index])
        )
        alice_reward_sum += alice_reward
        eve_utility_sum += eve_utility
        joint_counts[route_index, action_index] += 1
        if exploring:
            estimator.observe(
                route_index,
                action_index,
                draw_index,
                alice_reward,
            )
        else:
            commit_eve_utility_sum += eve_utility
            commit_turn_count += 1

        decision_due = (
            current_commitment is None
            and step >= learner_config.minimum_exploration_turns
            and (
                step % learner_config.decision_interval == 0
                or step == learner_config.maximum_exploration_turns
            )
        )
        if decision_due:
            means = estimator.means()
            radii = estimator.radii(delta=learner_config.confidence_delta)
            last_oracle = solve_stackelberg_oracle(
                means,
                bank.action_costs,
                alice_reward_radii=radii,
            )
            candidate = last_oracle.selected
            supported_actions = np.flatnonzero(
                np.asarray(candidate.probabilities) > 1e-12
            )
            support_counts = estimator.counts[:, supported_actions]
            support_coverage_ready = bool(
                support_counts.size
                and np.min(support_counts)
                >= learner_config.minimum_unique_draws_per_supported_cell
            )
            if (
                candidate.response_certified
                and support_coverage_ready
            ) or step >= learner_config.maximum_exploration_turns:
                current_commitment = candidate
                commitment_turn = step

        if progress_callback is not None and (
            step == 1
            or step == active_config.turns
            or step % max(50_000, 10 * learner_config.record_every) == 0
        ):
            progress_callback(
                step,
                "exploration" if current_commitment is None else "commitment",
            )
        if (
            step == 1
            or step == active_config.turns
            or step % learner_config.record_every == 0
        ):
            policy = alice.policy(turn=step + 1)
            coverage = estimator.coverage_summary()
            trajectory.append({
                "turn": step,
                "phase": (
                    "exploration"
                    if current_commitment is None
                    else "commitment"
                ),
                "mean_alice_reward": alice_reward_sum / step,
                "mean_eve_public_utility": eve_utility_sum / step,
                "alice_policy_effective_action_fraction": (
                    policy_distribution_statistics(policy)[
                        "effective_action_fraction"
                    ]
                ),
                "observed_route_action_cell_fraction": coverage[
                    "fully_observed_cell_fraction"
                ],
                "minimum_unique_draws_per_cell": coverage[
                    "minimum_unique_draws_per_cell"
                ],
                "commitment_support_size": (
                    current_commitment.support_size
                    if current_commitment is not None else None
                ),
                "response_certified": (
                    current_commitment.response_certified
                    if current_commitment is not None else None
                ),
            })

    if current_commitment is None or commitment_turn is None or last_oracle is None:
        raise AssertionError("Stackelberg learner did not enter its commitment phase")

    feedback_matrix = bank.feedback.alice_payoff_matrix()
    audit_matrix = bank.audit.alice_payoff_matrix()
    active_feedback_oracle = feedback_oracle or solve_stackelberg_oracle(
        feedback_matrix,
        bank.action_costs,
    )
    active_audit_oracle = audit_oracle or solve_stackelberg_oracle(
        audit_matrix,
        bank.action_costs,
    )
    learned_feedback_evaluation = evaluate_stackelberg_commitment(
        feedback_matrix,
        bank.action_costs,
        np.asarray(current_commitment.probabilities),
    )
    learned_audit_evaluation = evaluate_stackelberg_commitment(
        audit_matrix,
        bank.action_costs,
        np.asarray(current_commitment.probabilities),
    )
    audit_value = (
        active_audit_oracle.selected.planning_evaluation.optimistic_eve_value
    )
    audit_pessimistic_value = (
        active_audit_oracle.selected.planning_evaluation.pessimistic_eve_value
    )
    total_stackelberg_regret = (
        active_config.turns * audit_value - eve_utility_sum
    )
    total_pessimistic_stackelberg_regret = (
        active_config.turns * audit_pessimistic_value - eve_utility_sum
    )
    elapsed = time.perf_counter() - started
    return {
        "schema": STACKELBERG_LEARNER_RESULT_SCHEMA,
        "graph_id": case.graph_id,
        "family": case.family,
        "case_id": bank.case_id,
        "seed": int(seed),
        "turns": active_config.turns,
        "elapsed_seconds": elapsed,
        "turns_per_second": (
            active_config.turns / elapsed if elapsed > 0.0 else None
        ),
        "alice_learner": dict(alice.metadata),
        "learner_config": learner_config.to_dict(),
        "commitment_turn": commitment_turn,
        "exploration_turns": commitment_turn,
        "commitment_turns": commit_turn_count,
        "learned_commitment": current_commitment.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "learned_commitment_feedback_evaluation": (
            learned_feedback_evaluation.to_dict()
        ),
        "learned_commitment_audit_evaluation": (
            learned_audit_evaluation.to_dict()
        ),
        "feedback_stackelberg_oracle": active_feedback_oracle.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "audit_stackelberg_oracle": active_audit_oracle.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "estimation": {
            **estimator.coverage_summary(),
            "means": estimator.means().tolist(),
            "confidence_radii": estimator.radii(
                delta=learner_config.confidence_delta
            ).tolist(),
            "response_certified_at_commitment": (
                current_commitment.response_certified
            ),
        },
        "mean_alice_reward": alice_reward_sum / active_config.turns,
        "mean_eve_public_utility": eve_utility_sum / active_config.turns,
        "mean_commitment_phase_eve_public_utility": (
            commit_eve_utility_sum / commit_turn_count
            if commit_turn_count else None
        ),
        "audit_optimistic_stackelberg_value": audit_value,
        "audit_pessimistic_value_of_optimistic_commitment": (
            audit_pessimistic_value
        ),
        "learned_commitment_audit_optimistic_value": (
            learned_audit_evaluation.optimistic_eve_value
        ),
        "learned_commitment_audit_pessimistic_value": (
            learned_audit_evaluation.pessimistic_eve_value
        ),
        "audit_optimistic_value_loss": (
            audit_value - learned_audit_evaluation.optimistic_eve_value
        ),
        "total_optimistic_stackelberg_regret": total_stackelberg_regret,
        "total_pessimistic_stackelberg_regret": (
            total_pessimistic_stackelberg_regret
        ),
        "per_turn_optimistic_stackelberg_regret": (
            total_stackelberg_regret / active_config.turns
        ),
        "per_turn_pessimistic_stackelberg_regret": (
            total_pessimistic_stackelberg_regret / active_config.turns
        ),
        "joint_action_counts": joint_counts.tolist(),
        "trajectory": trajectory,
        "information_boundary": {
            "online_eve_observes_selected_route_identity": True,
            "online_eve_observes_alice_public_reward": True,
            "online_eve_observes_own_action_and_cost": True,
            "online_eve_observes_alice_policy_or_weights": False,
            "online_eve_observes_counterfactual_cells": False,
            "online_eve_uses_feedback_bank_matrix": False,
            "online_eve_uses_audit_split": False,
            "audit_and_feedback_oracles": "offline scoring and comparison only",
        },
    }
