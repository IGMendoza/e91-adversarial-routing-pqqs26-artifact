"""Pessimistic Stackelberg steering against a route-learning Alice.

The leader knows the declared feedback payoff surface supplied by the
histogram channel model, computes a physical Eve-action commitment that makes
one Alice route uniquely optimal by a requested margin, and then holds that
commitment fixed.  The disjoint audit surface is used only after play.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .learning import build_histogram_alice_learner
from .stackelberg_eve import (
    StackelbergOracleResult,
    evaluate_stackelberg_commitment,
    solve_stackelberg_oracle,
)


STACKELBERG_STEERING_RESULT_SCHEMA = "histogram_stackelberg_steering_run.v1"


@dataclass(frozen=True)
class StackelbergSteeringConfig:
    """One strict-response Stackelberg commitment experiment."""

    response_margin: float
    record_every: int = 1_000

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.response_margin)
            or self.response_margin < 0.0
            or self.response_margin > 1.0
        ):
            raise ValueError("response_margin must lie in [0, 1]")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feasible_stackelberg_response_margins(
    alice_payoff_matrix: np.ndarray,
    action_costs: np.ndarray,
    margins: Sequence[float],
) -> list[dict[str, Any]]:
    """Return the exact value/target frontier for declared response margins."""

    rows = []
    for value in margins:
        margin = float(value)
        try:
            oracle = solve_stackelberg_oracle(
                alice_payoff_matrix,
                action_costs,
                minimum_response_margin=margin,
            )
        except RuntimeError:
            rows.append({
                "response_margin": margin,
                "feasible": False,
                "commitment": None,
            })
            continue
        rows.append({
            "response_margin": margin,
            "feasible": True,
            "commitment": oracle.selected.to_dict(),
            "eve_value": (
                oracle.selected.planning_evaluation.optimistic_eve_value
            ),
            "intended_alice_route_index": (
                oracle.selected.intended_alice_route_index
            ),
            "support_size": oracle.selected.support_size,
        })
    return rows


def run_pessimistic_stackelberg_steering(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    steering_config: StackelbergSteeringConfig,
    *,
    seed: int,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
    feedback_unconstrained_oracle: StackelbergOracleResult | None = None,
    audit_unconstrained_oracle: StackelbergOracleResult | None = None,
) -> dict[str, Any]:
    """Commit to a strict Alice-response facet and measure finite-time steering."""

    active_config = replace(
        config,
        seed=int(seed),
        record_every=int(steering_config.record_every),
    )
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
    feedback_matrix = bank.feedback.alice_payoff_matrix()
    audit_matrix = bank.audit.alice_payoff_matrix()
    feedback_oracle = solve_stackelberg_oracle(
        feedback_matrix,
        bank.action_costs,
        minimum_response_margin=steering_config.response_margin,
    )
    commitment = feedback_oracle.selected
    unconstrained_feedback = (
        feedback_unconstrained_oracle
        or solve_stackelberg_oracle(feedback_matrix, bank.action_costs)
    )
    unconstrained_audit = (
        audit_unconstrained_oracle
        or solve_stackelberg_oracle(audit_matrix, bank.action_costs)
    )
    try:
        matched_margin_audit = solve_stackelberg_oracle(
            audit_matrix,
            bank.action_costs,
            minimum_response_margin=steering_config.response_margin,
        )
    except RuntimeError:
        matched_margin_audit = None

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
    rng = np.random.default_rng([int(seed), 503])
    probabilities = np.asarray(commitment.probabilities, dtype=float)
    target_route = int(commitment.intended_alice_route_index)
    route_counts = np.zeros(route_count, dtype=np.int64)
    action_counts = np.zeros(action_count, dtype=np.int64)
    joint_counts = np.zeros((route_count, action_count), dtype=np.int64)
    alice_reward_sum = 0.0
    eve_utility_sum = 0.0
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()

    for turn_index in range(active_config.turns):
        step = turn_index + 1
        action_index = int(rng.choice(action_count, p=probabilities))
        route_index = alice.sample()
        draw_index = stream.next_draw(route_index, action_index)
        observation = bank.feedback.draw(route_index, action_index, draw_index)
        alice_reward = float(observation["alice_reward"])
        alice.update(alice_reward)
        eve_utility = (
            1.0 - alice_reward - float(bank.action_costs[action_index])
        )
        route_counts[route_index] += 1
        action_counts[action_index] += 1
        joint_counts[route_index, action_index] += 1
        alice_reward_sum += alice_reward
        eve_utility_sum += eve_utility

        if progress_callback is not None and (
            step == 1
            or step == active_config.turns
            or step % 50_000 == 0
        ):
            progress_callback(step)
        if (
            step == 1
            or step == active_config.turns
            or step % steering_config.record_every == 0
        ):
            policy = alice.policy(turn=step + 1)
            empirical = route_counts / step
            policy_stats = policy_distribution_statistics(policy)
            trajectory.append({
                "turn": step,
                "mean_alice_reward": alice_reward_sum / step,
                "mean_eve_public_utility": eve_utility_sum / step,
                "intended_route_empirical_mass": float(
                    empirical[target_route]
                ),
                "intended_route_current_policy_mass": float(
                    policy[target_route]
                ),
                "alice_policy_effective_action_fraction": (
                    policy_stats["effective_action_fraction"]
                ),
                "alice_policy_uniform_coverage_mass": (
                    policy_stats["uniform_coverage_mass"]
                ),
            })

    elapsed = time.perf_counter() - started
    alice_strategy = route_counts / active_config.turns
    alice_current_policy = alice.policy(turn=active_config.turns + 1)
    feedback_evaluation = evaluate_stackelberg_commitment(
        feedback_matrix,
        bank.action_costs,
        probabilities,
    )
    audit_evaluation = evaluate_stackelberg_commitment(
        audit_matrix,
        bank.action_costs,
        probabilities,
    )
    audit_value = (
        unconstrained_audit.selected.planning_evaluation.optimistic_eve_value
    )
    matched_audit_value = (
        matched_margin_audit.selected.planning_evaluation.optimistic_eve_value
        if matched_margin_audit is not None else None
    )
    mean_eve_utility = eve_utility_sum / active_config.turns
    rational_audit_value = audit_evaluation.pessimistic_eve_value
    return {
        "schema": STACKELBERG_STEERING_RESULT_SCHEMA,
        "graph_id": case.graph_id,
        "family": case.family,
        "case_id": bank.case_id,
        "seed": int(seed),
        "turns": int(active_config.turns),
        "elapsed_seconds": elapsed,
        "turns_per_second": (
            active_config.turns / elapsed if elapsed > 0.0 else None
        ),
        "alice_learner": dict(alice.metadata),
        "steering_config": steering_config.to_dict(),
        "commitment": commitment.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "feedback_margin_oracle": feedback_oracle.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "feedback_unconstrained_oracle": unconstrained_feedback.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "audit_unconstrained_oracle": unconstrained_audit.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "audit_matched_margin_oracle": (
            matched_margin_audit.to_dict(
                action_ids=bank.action_ids,
                route_ids=bank.route_ids,
            )
            if matched_margin_audit is not None else None
        ),
        "commitment_feedback_evaluation": feedback_evaluation.to_dict(),
        "commitment_audit_evaluation": audit_evaluation.to_dict(),
        "intended_alice_route_index": target_route,
        "intended_alice_route_id": str(bank.route_ids[target_route]),
        "intended_route_empirical_mass": float(alice_strategy[target_route]),
        "intended_route_current_policy_mass": float(
            alice_current_policy[target_route]
        ),
        "intended_route_is_audit_best_response": bool(
            target_route in audit_evaluation.alice_best_response_indices
        ),
        "alice_empirical_strategy": alice_strategy.tolist(),
        "alice_current_policy": alice_current_policy.tolist(),
        "eve_empirical_strategy": (
            action_counts / active_config.turns
        ).tolist(),
        "mean_alice_reward": alice_reward_sum / active_config.turns,
        "mean_eve_public_utility": mean_eve_utility,
        "audit_unconstrained_stackelberg_value": audit_value,
        "audit_matched_margin_stackelberg_value": matched_audit_value,
        "commitment_audit_rational_follower_value": rational_audit_value,
        "pessimism_price_on_audit": (
            audit_value - matched_audit_value
            if matched_audit_value is not None else None
        ),
        "feedback_to_audit_commitment_value_loss": (
            matched_audit_value - rational_audit_value
            if matched_audit_value is not None else None
        ),
        "finite_follower_value_loss": (
            rational_audit_value - mean_eve_utility
        ),
        "per_turn_unconstrained_stackelberg_regret": (
            audit_value - mean_eve_utility
        ),
        "joint_action_counts": joint_counts.tolist(),
        "trajectory": trajectory,
        "information_boundary": {
            "online_eve_knows_declared_feedback_payoff_surface": True,
            "online_eve_uses_feedback_surface_to_compute_commitment": True,
            "online_eve_observes_alice_policy_or_weights": False,
            "online_eve_adapts_after_commitment": False,
            "online_eve_uses_audit_split": False,
            "audit_split": "held-out scoring only",
            "interventions": "declared physical histogram Eve actions only",
            "alice_feedback": "selected-route public scalar service reward",
        },
    }
