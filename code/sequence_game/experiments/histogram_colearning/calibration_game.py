"""Fast two-player calibration runs over histogram-derived outcome banks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Sequence

import numpy as np

from sequence_game.game_theory import Exp3PlusPlus, TsallisINF, solve_zero_sum_lp

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .config import EVE_LEARNER_FLAT_EXP3, HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner
from .learning import (
    build_histogram_alice_learner,
    build_histogram_eve_learner,
    family_balanced_action_prior,
)
from .reward_windows import RollingRewardMinimumTracker
from .strategic_metrics import EquilibriumSetProjector


CALIBRATION_RUN_SCHEMA = "histogram_two_player_bank_run.v1"
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class CalibrationGameOracle:
    """Scoring matrix and exact equilibrium computed once per payoff bank."""

    equivalent_matrix: np.ndarray
    value: float
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray
    projector: EquilibriumSetProjector


def build_calibration_game_oracle(
    bank: HistogramPayoffBank,
) -> CalibrationGameOracle:
    matrix = bank.equivalent_zero_sum_matrix()
    value, alice, eve = solve_zero_sum_lp(matrix)
    return CalibrationGameOracle(
        equivalent_matrix=matrix,
        value=float(value),
        alice_strategy=alice,
        eve_strategy=eve,
        projector=EquilibriumSetProjector(matrix, value),
    )


@dataclass
class CommonCellOutcomeStream:
    """Deterministic per-cell stream shared across matched learner settings."""

    route_count: int
    action_count: int
    draws_per_cell: int
    seed: int

    def __post_init__(self) -> None:
        if min(self.route_count, self.action_count, self.draws_per_cell) < 1:
            raise ValueError("outcome-stream dimensions must be positive")
        self._visits = np.zeros(
            (self.route_count, self.action_count), dtype=np.uint64
        )

    def next_draw(self, route_index: int, action_index: int) -> int:
        route = int(route_index)
        action = int(action_index)
        visit = int(self._visits[route, action])
        self._visits[route, action] += np.uint64(1)
        mixed = _splitmix64(
            int(self.seed)
            ^ ((route + 1) * 0xD2B74407B1CE6E93)
            ^ ((action + 1) * 0xCA5A826395121157)
            ^ (visit * 0x9E3779B97F4A7C15)
        )
        return int(mixed % self.draws_per_cell)

    @property
    def visits(self) -> np.ndarray:
        return self._visits.copy()


def run_bank_two_player_learning(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    turns: int | None = None,
    seed: int | None = None,
    record_every: int = 0,
    oracle: CalibrationGameOracle | None = None,
    include_equilibrium_distance: bool = True,
    selection_digest_blake2b_128: str | None = None,
    reward_window_turns: Sequence[int] = (),
    reward_window_start_turn: int = 1,
) -> dict[str, Any]:
    """Run both declared learners with selected-cell scalar rewards only."""

    active_turns = int(config.turns if turns is None else turns)
    active_seed = int(config.seed if seed is None else seed)
    if active_turns < 1:
        raise ValueError("turns must be positive")
    if record_every < 0:
        raise ValueError("record_every must be nonnegative")
    if not 1 <= int(reward_window_start_turn) <= active_turns:
        raise ValueError(
            "reward_window_start_turn must lie inside the run horizon"
        )
    if any(
        int(window) > active_turns - int(reward_window_start_turn) + 1
        for window in reward_window_turns
    ):
        raise ValueError(
            "reward windows must fit inside the declared evaluation interval"
        )
    active_config = replace(config, turns=active_turns, seed=active_seed)
    if selection_digest_blake2b_128 is None:
        validate_bank_against_case(bank, case, active_config)
    else:
        validate_bank_against_case(
            bank,
            case,
            active_config,
            selection_digest_blake2b_128=str(
                selection_digest_blake2b_128
            ),
        )
    active_oracle = oracle or build_calibration_game_oracle(bank)
    actions = case.actions(active_config.game_mode)
    alice_prior = np.full(len(case.routes), 1.0 / len(case.routes), dtype=float)
    eve_prior = (
        family_balanced_action_prior(actions)
        if active_config.eve_learner_mode == EVE_LEARNER_FLAT_EXP3
        else np.full(len(actions), 1.0 / len(actions), dtype=float)
    )
    alice_implementation, alice_metadata = build_histogram_alice_learner(
        case=case,
        config=active_config,
        prior=alice_prior,
    )
    eve_implementation, eve_metadata = build_histogram_eve_learner(
        case=case,
        actions=actions,
        config=active_config,
        prior=eve_prior,
    )
    alice = PlayerLearner(
        player="alice",
        action_ids=bank.route_ids,
        learner=alice_implementation,
        metadata=alice_metadata,
    )
    eve = PlayerLearner(
        player="eve",
        action_ids=bank.action_ids,
        learner=eve_implementation,
        metadata=eve_metadata,
    )
    stream = CommonCellOutcomeStream(
        route_count=len(bank.route_ids),
        action_count=len(bank.action_ids),
        draws_per_cell=bank.draws_per_cell,
        seed=active_seed,
    )
    joint_counts = np.zeros(
        (len(bank.route_ids), len(bank.action_ids)), dtype=np.int64
    )
    alice_reward_sum = 0.0
    evaluation_interval_alice_reward_sum = 0.0
    eve_reward_sum = 0.0
    corrected_reward_delta_sum = 0.0
    corrected_reward_absolute_delta_sum = 0.0
    bell_rate_sum = 0.0
    false_signal_count = 0
    classifiable_signal_count = 0
    target_active_count = 0
    information_observed_count = 0
    eve_observation_count_sum = 0
    outcome_counts = np.zeros(4, dtype=np.int64)
    signal_class_counts = np.zeros(6, dtype=np.int64)
    counterfactual_offset = np.zeros(len(bank.route_ids), dtype=float)
    state_replay = isinstance(alice.learner, (Exp3PlusPlus, TsallisINF))
    reward_windows = RollingRewardMinimumTracker(
        reward_window_turns,
        evaluation_start_turn=int(reward_window_start_turn),
    )
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for turn_index in range(active_turns):
        route_index = alice.sample()
        action_index = eve.sample()
        draw_index = stream.next_draw(route_index, action_index)
        observation = bank.feedback.draw(
            route_index, action_index, draw_index
        )
        alice_reward = observation["alice_reward"]
        corrected_reward = observation["corrected_alice_reward"]
        reward_delta = corrected_reward - alice_reward
        if state_replay:
            alice.learner.update_counterfactual(
                alice_reward,
                corrected_reward,
            )
        else:
            counterfactual_offset += (
                alice.learner.counterfactual_log_weight_correction(reward_delta)
            )
        alice.update(alice_reward)
        eve.update(observation["eve_bandit_reward"])
        joint_counts[route_index, action_index] += 1
        alice_reward_sum += alice_reward
        if turn_index + 1 >= int(reward_window_start_turn):
            evaluation_interval_alice_reward_sum += alice_reward
        reward_windows.update(turn_index + 1, alice_reward)
        eve_reward_sum += observation["eve_bandit_reward"]
        corrected_reward_delta_sum += reward_delta
        corrected_reward_absolute_delta_sum += abs(reward_delta)
        bell_rate_sum += observation["bell_qualified_rate_hz"]
        false_signal_count += int(observation["false_signal"])
        classifiable_signal_count += int(observation["classifiable_signal"])
        target_active_count += int(observation["target_active"])
        information_observed_count += int(observation["information_observed"])
        eve_observation_count_sum += int(observation["eve_observation_count"])
        outcome_counts[observation["public_outcome_code"]] += 1
        signal_class_counts[observation["signal_class_code"]] += 1
        step = turn_index + 1
        if record_every and (step == 1 or step == active_turns or step % record_every == 0):
            curve.append(
                _trajectory_row(
                    step=step,
                    joint_counts=joint_counts,
                    equivalent_matrix=active_oracle.equivalent_matrix,
                    mean_alice_reward=alice_reward_sum / step,
                    false_signal_rate=false_signal_count / step,
                )
            )
    elapsed = time.perf_counter() - started
    alice_strategy = alice.empirical_strategy()
    eve_strategy = eve.empirical_strategy()
    strategic = _strategic_diagnostics(
        oracle=active_oracle,
        joint_counts=joint_counts,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        include_equilibrium_distance=include_equilibrium_distance,
    )
    if state_replay:
        counterfactual_policy = alice.learner.counterfactual_policy(
            turn=active_turns + 1
        )
        counterfactual_method = (
            "fixed_behavior_selected_reward_state_replay"
        )
    else:
        counterfactual_policy = alice.learner.policy_with_log_weight_offset(
            counterfactual_offset,
            turn=active_turns + 1,
        )
        counterfactual_method = (
            "fixed_behavior_selected_reward_score_correction"
        )
    factual_policy = alice.policy(turn=active_turns + 1)
    return {
        "schema": CALIBRATION_RUN_SCHEMA,
        "case_id": bank.case_id,
        "bank_content_digest_blake2b_128": bank.content_digest(),
        "turns": active_turns,
        "seed": active_seed,
        "elapsed_seconds": elapsed,
        "turns_per_second": active_turns / elapsed if elapsed > 0.0 else None,
        "learner_outcome_source": "empirical_resampling_from_feedback_bank",
        "scoring_source": "reserved_equal_coverage_audit_bank",
        "physics_cache_compiled_before_learning": None,
        "config": active_config.to_dict(),
        "alice_learner": dict(alice.metadata),
        "eve_learner": dict(eve.metadata),
        "alice_strategy": alice_strategy.tolist(),
        "eve_strategy": eve_strategy.tolist(),
        "alice_current_policy": factual_policy.tolist(),
        "eve_current_policy": eve.policy(turn=active_turns + 1).tolist(),
        "joint_action_counts": joint_counts.tolist(),
        "mean_alice_reward": alice_reward_sum / active_turns,
        "mean_alice_reward_evaluation_interval": (
            evaluation_interval_alice_reward_sum
            / (active_turns - int(reward_window_start_turn) + 1)
        ),
        "mean_alice_reward_before_evaluation_interval": (
            (
                alice_reward_sum
                - evaluation_interval_alice_reward_sum
            )
            / (int(reward_window_start_turn) - 1)
            if int(reward_window_start_turn) > 1
            else None
        ),
        "alice_reward_window_minima": reward_windows.diagnostics(),
        "mean_eve_bandit_reward": eve_reward_sum / active_turns,
        "mean_bell_qualified_rate_hz": bell_rate_sum / active_turns,
        "target_active_rate_diagnostic": target_active_count / active_turns,
        "information_observation_rate": information_observed_count / active_turns,
        "mean_eve_observation_count": (
            eve_observation_count_sum / active_turns
        ),
        "outcome_counts": {
            label: int(outcome_counts[code])
            for label, code in {
                "delivery_failure": 0,
                "qber_abort": 1,
                "chsh_abort": 2,
                "accepted": 3,
            }.items()
        },
        "alice_false_signal_telemetry": {
            "false_signal_count": false_signal_count,
            "false_signal_rate": false_signal_count / active_turns,
            "classifiable_signal_count": classifiable_signal_count,
            "classifiable_signal_rate": classifiable_signal_count / active_turns,
            "classification_counts": {
                label: int(signal_class_counts[code])
                for label, code in {
                    "delivery_failure_not_quality_signal": 0,
                    "missing_latent_quality_reference": 1,
                    "true_accept": 2,
                    "false_accept": 3,
                    "false_abort": 4,
                    "true_abort": 5,
                }.items()
            },
            "corrected_reward_delta_mean": (
                corrected_reward_delta_sum / active_turns
            ),
            "corrected_reward_absolute_delta_mean": (
                corrected_reward_absolute_delta_sum / active_turns
            ),
            "policy_attribution": {
                "method": counterfactual_method,
                "behavior_actions_held_fixed": True,
                "counterfactual_rewards_disclosed_to_online_learner": False,
                "factual_current_policy": factual_policy.tolist(),
                "false_signal_corrected_current_policy": (
                    counterfactual_policy.tolist()
                ),
                "policy_total_variation": _total_variation(
                    factual_policy, counterfactual_policy
                ),
            },
        },
        "strategic": strategic,
        "trajectory": curve,
        "online_learning_telemetry": {
            "feedback_protocol": "selected_action_scalar_reward_v1",
            "alice": alice.runtime_telemetry(turn=active_turns + 1),
            "eve": eve.runtime_telemetry(turn=active_turns + 1),
        },
        "information_boundary": {
            "learner_outcome_source": "feedback_bank_only",
            "scoring_outcome_source": "disjoint_audit_bank_only",
            "opponent_action_disclosed": False,
            "unselected_cell_rewards_disclosed": False,
            "target_active_disclosed": False,
            "private_instrument_record_disclosed": False,
            "evaluation_partition_loaded": False,
        },
    }


def score_calibration_behavior(
    oracle: CalibrationGameOracle,
    *,
    joint_counts: np.ndarray,
    alice_strategy: np.ndarray,
    eve_strategy: np.ndarray,
    include_equilibrium_distance: bool = True,
) -> dict[str, Any]:
    """Score one realized behavior transcript on the reserved audit matrix."""

    return _strategic_diagnostics(
        oracle=oracle,
        joint_counts=np.asarray(joint_counts, dtype=np.int64),
        alice_strategy=np.asarray(alice_strategy, dtype=float),
        eve_strategy=np.asarray(eve_strategy, dtype=float),
        include_equilibrium_distance=include_equilibrium_distance,
    )


def _strategic_diagnostics(
    *,
    oracle: CalibrationGameOracle,
    joint_counts: np.ndarray,
    alice_strategy: np.ndarray,
    eve_strategy: np.ndarray,
    include_equilibrium_distance: bool,
) -> dict[str, Any]:
    turns = int(np.sum(joint_counts))
    if turns < 1:
        raise ValueError("joint_counts must contain at least one turn")
    frequency = np.asarray(joint_counts, dtype=float) / turns
    matrix = oracle.equivalent_matrix
    product_value = float(alice_strategy @ matrix @ eve_strategy)
    realized_value = float(np.sum(frequency * matrix))
    alice_best = float(np.max(matrix @ eve_strategy))
    eve_best = float(np.min(alice_strategy @ matrix))
    alice_external = alice_best - realized_value
    eve_external = realized_value - eve_best
    return {
        "audit_matrix_source": "disjoint_equal_coverage_scoring_bank",
        "oracle_value": oracle.value,
        "oracle_alice_strategy": oracle.alice_strategy.tolist(),
        "oracle_eve_strategy": oracle.eve_strategy.tolist(),
        "product_profile_equivalent_value": product_value,
        "audit_expected_realized_joint_equivalent_value": realized_value,
        "alice_external_regret_per_turn": float(alice_external),
        "eve_external_regret_per_turn": float(eve_external),
        "external_regret_sum_per_turn": float(alice_external + eve_external),
        "alice_best_response_gap": float(alice_best - product_value),
        "eve_best_response_gap": float(product_value - eve_best),
        "exploitability": float(alice_best - eve_best),
        "decomposition_identity_error": float(
            abs(alice_best - eve_best - alice_external - eve_external)
        ),
        "alice_equilibrium_set_total_variation": (
            oracle.projector.total_variation("alice", alice_strategy)
            if include_equilibrium_distance else None
        ),
        "eve_equilibrium_set_total_variation": (
            oracle.projector.total_variation("eve", eve_strategy)
            if include_equilibrium_distance else None
        ),
    }


def _trajectory_row(
    *,
    step: int,
    joint_counts: np.ndarray,
    equivalent_matrix: np.ndarray,
    mean_alice_reward: float,
    false_signal_rate: float,
) -> dict[str, Any]:
    counts = np.asarray(joint_counts, dtype=float)
    alice = counts.sum(axis=1) / step
    eve = counts.sum(axis=0) / step
    alice_best = float(np.max(equivalent_matrix @ eve))
    eve_best = float(np.min(alice @ equivalent_matrix))
    return {
        "turn": int(step),
        "mean_alice_reward": float(mean_alice_reward),
        "false_signal_rate": float(false_signal_rate),
        "exploitability": float(alice_best - eve_best),
        "alice_strategy": alice.tolist(),
        "eve_strategy": eve.tolist(),
    }


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or not np.all(np.isfinite(first - second)):
        raise ValueError("total-variation inputs are incompatible")
    return 0.5 * float(np.sum(np.abs(first - second)))
