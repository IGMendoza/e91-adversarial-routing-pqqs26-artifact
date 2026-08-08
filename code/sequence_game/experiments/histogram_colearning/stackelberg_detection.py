"""Public-outcome detection and opt-in mitigation for Stackelberg steering.

The detector is deliberately separated from simulator-private state.  It sees
only the selected route identity and Alice's public terminal outcome code.
Monitoring requests use a topology-only G-optimal route design and are kept
outside the service learner's sample/update sequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

from sequence_game.game_theory import linear_exploration_design

from .calibration_bank import (
    HistogramPayoffBank,
    PayoffSampleBlock,
    validate_bank_against_case,
)
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner
from .learning import build_histogram_alice_learner
from .reward_windows import RollingRewardMinimumTracker
from .stackelberg_eve import (
    StackelbergCommitment,
    evaluate_stackelberg_commitment,
    solve_stackelberg_oracle,
)
from .structure import route_resource_embedding


STACKELBERG_DETECTION_RESULT_SCHEMA = (
    "histogram_stackelberg_detection_mitigation_run.v5"
)
PUBLIC_OUTCOME_COUNT = 4
DETECTOR_QUANTUM_FOUR_OUTCOME = "quantum_four_outcome"
DETECTOR_CLASSICAL_BINARY_SERVICE = "classical_binary_service"
ACTIVE_DETECTOR_MODES = frozenset({
    DETECTOR_QUANTUM_FOUR_OUTCOME,
    DETECTOR_CLASSICAL_BINARY_SERVICE,
})


@dataclass(frozen=True)
class RouteOutcomeDetectorConfig:
    """Commissioning-relative route-conditioned categorical change detector."""

    horizon: int
    model_false_alarm_level: float = 0.01
    commissioning_pseudocount: float = 0.5
    change_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("detector horizon must be positive")
        if not 0.0 < self.model_false_alarm_level < 1.0:
            raise ValueError("model_false_alarm_level must lie in (0, 1)")
        if (
            not math.isfinite(self.commissioning_pseudocount)
            or self.commissioning_pseudocount <= 0.0
        ):
            raise ValueError("commissioning_pseudocount must be positive")
        if (
            not self.change_fractions
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in self.change_fractions
            )
            or len(set(self.change_fractions)) != len(self.change_fractions)
        ):
            raise ValueError(
                "change_fractions must be distinct finite values in (0, 1)"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteConditionedCategoricalEProcess:
    """Mixture likelihood-ratio process over routes and change times.

    The no-change model is the Dirichlet posterior predictive distribution
    initialized by commissioning counts and updated with subsequent public
    outcomes.  Each fixed alternative is a categorical shift from the initial
    commissioning predictive distribution toward one public outcome on one
    route.  A uniform prior mixes over routes, outcome directions, configured
    shift sizes, and every possible change time inside the declared horizon.
    Adaptive route selection is permitted because every likelihood ratio is
    conditioned on the route selected before its terminal outcome is observed.

    The alarm calibration is model-based under that posterior-predictive null;
    clean-control runs measure departures between it and the compiled channel.
    """

    def __init__(
        self,
        commissioning_counts: np.ndarray,
        config: RouteOutcomeDetectorConfig,
    ) -> None:
        counts = np.asarray(commissioning_counts, dtype=np.int64)
        if (
            counts.ndim != 2
            or counts.shape[0] < 1
            or counts.shape[1] < 2
            or np.any(counts < 0)
            or np.any(np.sum(counts, axis=1) < 1)
        ):
            raise ValueError(
                "commissioning_counts must be nonnegative route x outcome "
                "counts "
                "with at least one observation per route"
            )
        beta = float(config.commissioning_pseudocount)
        self.config = config
        self.commissioning_counts = counts.copy()
        self.outcome_count = int(counts.shape[1])
        self.null_probabilities = (
            counts.astype(float) + beta
        ) / (
            np.sum(counts, axis=1, keepdims=True)
            + self.outcome_count * beta
        )
        self._online_counts = np.zeros_like(counts)
        route_count = counts.shape[0]
        routes: list[int] = []
        alternatives: list[np.ndarray] = []
        labels: list[tuple[int, int, float]] = []
        for route_index in range(route_count):
            baseline = self.null_probabilities[route_index]
            for outcome_code in range(self.outcome_count):
                direction = np.zeros(self.outcome_count, dtype=float)
                direction[outcome_code] = 1.0
                for fraction in config.change_fractions:
                    routes.append(route_index)
                    alternatives.append(
                        (1.0 - fraction) * baseline + fraction * direction
                    )
                    labels.append((route_index, outcome_code, float(fraction)))
        self._alternative_routes = np.asarray(routes, dtype=np.int64)
        self._alternative_probabilities = np.asarray(
            alternatives, dtype=float
        )
        self._alternative_labels = tuple(labels)
        self._log_change_time_sums = np.full(
            len(labels), -np.inf, dtype=float
        )
        self.turn = 0
        self.alarm_turn: int | None = None
        self.log_e_value = 0.0
        self.maximum_log_e_value = 0.0
        self.leading_alternative: tuple[int, int, float] | None = None

    @property
    def alarmed(self) -> bool:
        return self.alarm_turn is not None

    @property
    def threshold_log_e_value(self) -> float:
        return -math.log(self.config.model_false_alarm_level)

    def update(self, route_index: int, outcome_code: int) -> dict[str, Any]:
        route = int(route_index)
        outcome = int(outcome_code)
        if self.turn >= self.config.horizon:
            raise RuntimeError("detector received more observations than its horizon")
        if not 0 <= route < self.null_probabilities.shape[0]:
            raise IndexError("route_index is outside the commissioning model")
        if not 0 <= outcome < self.outcome_count:
            raise ValueError("outcome code is outside the detector alphabet")

        self.turn += 1
        log_likelihood_ratio = np.zeros(
            self._alternative_routes.shape[0], dtype=float
        )
        selected = self._alternative_routes == route
        null_predictive = (
            self.commissioning_counts[route].astype(float)
            + self._online_counts[route]
            + self.config.commissioning_pseudocount
        ) / (
            float(np.sum(self.commissioning_counts[route]))
            + float(np.sum(self._online_counts[route]))
            + self.outcome_count
            * self.config.commissioning_pseudocount
        )
        log_likelihood_ratio[selected] = np.log(
            self._alternative_probabilities[selected, outcome]
            / null_predictive[outcome]
        )
        self._log_change_time_sums = (
            log_likelihood_ratio
            + np.logaddexp(0.0, self._log_change_time_sums)
        )
        self._online_counts[route, outcome] += 1

        remaining_prior_mass = (
            self.config.horizon - self.turn
        ) / self.config.horizon
        if remaining_prior_mass > 0.0:
            component_log_e = np.logaddexp(
                math.log(remaining_prior_mass),
                self._log_change_time_sums
                - math.log(self.config.horizon),
            )
        else:
            component_log_e = (
                self._log_change_time_sums
                - math.log(self.config.horizon)
            )
        maximum = float(np.max(component_log_e))
        self.log_e_value = (
            maximum
            + math.log(float(np.sum(np.exp(component_log_e - maximum))))
            - math.log(component_log_e.size)
        )
        self.maximum_log_e_value = max(
            self.maximum_log_e_value, self.log_e_value
        )
        leading_index = int(np.argmax(component_log_e))
        self.leading_alternative = self._alternative_labels[leading_index]
        newly_alarmed = False
        if (
            self.alarm_turn is None
            and self.log_e_value >= self.threshold_log_e_value
        ):
            self.alarm_turn = self.turn
            newly_alarmed = True
        return {
            "turn": self.turn,
            "log_e_value": float(self.log_e_value),
            "e_value": _finite_exponential(self.log_e_value),
            "threshold_log_e_value": self.threshold_log_e_value,
            "alarmed": self.alarmed,
            "newly_alarmed": newly_alarmed,
            "alarm_turn": self.alarm_turn,
            "leading_alternative": (
                {
                    "route_index": int(self.leading_alternative[0]),
                    "outcome_code": int(self.leading_alternative[1]),
                    "change_fraction": float(self.leading_alternative[2]),
                }
                if self.leading_alternative is not None
                else None
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema": "route_conditioned_categorical_e_process.v3",
            "config": self.config.to_dict(),
            "outcome_count": self.outcome_count,
            "turn": int(self.turn),
            "alarm_turn": self.alarm_turn,
            "alarmed": self.alarmed,
            "log_e_value": float(self.log_e_value),
            "e_value": _finite_exponential(self.log_e_value),
            "maximum_log_e_value": float(self.maximum_log_e_value),
            "null_probabilities": self.null_probabilities.tolist(),
            "online_outcome_counts": self._online_counts.tolist(),
            "commissioning_counts": self.commissioning_counts.tolist(),
            "alternative_count": int(self._alternative_routes.size),
            "leading_alternative": (
                {
                    "route_index": int(self.leading_alternative[0]),
                    "outcome_code": int(self.leading_alternative[1]),
                    "change_fraction": float(self.leading_alternative[2]),
                }
                if self.leading_alternative is not None
                else None
            ),
            "information_boundary": {
                "observed": [
                    "selected route identity",
                    "Alice public terminal outcome code",
                ],
                "excluded": [
                    "Eve action identity",
                    "target-active diagnostic",
                    "Eve private instrument record",
                    "corrected Alice reward",
                    "audit split",
                    "simulator hidden state",
                ],
            },
            "calibration_boundary": (
                "model-based under a Dirichlet posterior-predictive no-change "
                "distribution initialized by commissioning; clean controls "
                "measure remaining channel mismatch"
            ),
        }


@dataclass(frozen=True)
class DetectionMitigationCondition:
    """One declared monitoring and response controller."""

    name: str
    attack_enabled: bool
    pre_alarm_monitoring_fraction: float
    post_alarm_monitoring_fraction: float
    restart_service_learner_on_alarm: bool = False
    freeze_service_policy_on_alarm: bool = False
    active_detector: str = DETECTOR_QUANTUM_FOUR_OUTCOME

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("condition name must be non-empty")
        for field_name in (
            "pre_alarm_monitoring_fraction",
            "post_alarm_monitoring_fraction",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if self.active_detector not in ACTIVE_DETECTOR_MODES:
            raise ValueError(
                f"unsupported active_detector {self.active_detector!r}"
            )
        if (
            self.restart_service_learner_on_alarm
            and self.freeze_service_policy_on_alarm
        ):
            raise ValueError("restart and policy freeze are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StackelbergDetectionMitigationConfig:
    """Opt-in single-topology steering detection experiment."""

    attack_start_turn: int = 25_000
    response_margin: float = 0.01
    commissioning_draws_per_route: int = 32
    record_every: int = 500
    detector_false_alarm_level: float = 0.01
    detector_commissioning_pseudocount: float = 0.5
    detector_change_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)
    damage_window_turns: tuple[int, ...] = (1_000, 5_000, 10_000)

    def validate(self, *, turns: int, draws_per_cell: int) -> None:
        if not 1 <= self.attack_start_turn < turns:
            raise ValueError("attack_start_turn must lie inside the run horizon")
        if (
            not math.isfinite(self.response_margin)
            or not 0.0 <= self.response_margin <= 1.0
        ):
            raise ValueError("response_margin must lie in [0, 1]")
        if not 2 <= self.commissioning_draws_per_route <= draws_per_cell - 2:
            raise ValueError(
                "commissioning_draws_per_route must leave at least two "
                "online feedback draws"
            )
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if (
            not self.damage_window_turns
            or any(int(window) < 1 for window in self.damage_window_turns)
            or len(set(map(int, self.damage_window_turns)))
            != len(self.damage_window_turns)
            or max(map(int, self.damage_window_turns))
            > turns - self.attack_start_turn
        ):
            raise ValueError(
                "damage_window_turns must be distinct positive windows that "
                "fit after attack_start_turn"
            )
        RouteOutcomeDetectorConfig(
            horizon=turns,
            model_false_alarm_level=self.detector_false_alarm_level,
            commissioning_pseudocount=(
                self.detector_commissioning_pseudocount
            ),
            change_fractions=self.detector_change_fractions,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_DETECTION_MITIGATION_CONDITIONS = (
    DetectionMitigationCondition(
        name="clean_control",
        attack_enabled=False,
        pre_alarm_monitoring_fraction=0.0,
        post_alarm_monitoring_fraction=0.0,
    ),
    DetectionMitigationCondition(
        name="detect_only",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.0,
        post_alarm_monitoring_fraction=0.0,
    ),
    DetectionMitigationCondition(
        name="fixed_sentinel_05",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.05,
        post_alarm_monitoring_fraction=0.05,
    ),
    DetectionMitigationCondition(
        name="adaptive_sentinel_02_20",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.02,
        post_alarm_monitoring_fraction=0.20,
    ),
    DetectionMitigationCondition(
        name="adaptive_sentinel_02_20_restart",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.02,
        post_alarm_monitoring_fraction=0.20,
        restart_service_learner_on_alarm=True,
    ),
    DetectionMitigationCondition(
        name="quantum_detect_freeze_policy",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.0,
        post_alarm_monitoring_fraction=0.0,
        freeze_service_policy_on_alarm=True,
        active_detector=DETECTOR_QUANTUM_FOUR_OUTCOME,
    ),
    DetectionMitigationCondition(
        name="classical_binary_detect_freeze_policy",
        attack_enabled=True,
        pre_alarm_monitoring_fraction=0.0,
        post_alarm_monitoring_fraction=0.0,
        freeze_service_policy_on_alarm=True,
        active_detector=DETECTOR_CLASSICAL_BINARY_SERVICE,
    ),
)


def run_stackelberg_detection_mitigation(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    experiment: StackelbergDetectionMitigationConfig,
    condition: DetectionMitigationCondition,
    *,
    seed: int,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
    precomputed_commitment: StackelbergCommitment | None = None,
) -> dict[str, Any]:
    """Run one public-outcome detector and mitigation condition."""

    active_config = replace(
        config,
        seed=int(seed),
        record_every=int(experiment.record_every),
    )
    experiment.validate(
        turns=active_config.turns,
        draws_per_cell=bank.draws_per_cell,
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
    try:
        no_attack_index = bank.action_ids.index("no_attack")
    except ValueError as exc:
        raise ValueError("detection experiment requires a no_attack action") from exc

    commissioning_draws = int(experiment.commissioning_draws_per_route)
    commissioning_counts = commissioning_outcome_counts(
        bank.feedback,
        action_index=no_attack_index,
        stop=commissioning_draws,
    )
    online_feedback = slice_payoff_sample_block(
        bank.feedback,
        commissioning_draws,
        bank.draws_per_cell,
    )
    feedback_matrix = online_feedback.alice_payoff_matrix()
    commitment = precomputed_commitment or solve_stackelberg_oracle(
        feedback_matrix,
        bank.action_costs,
        minimum_response_margin=experiment.response_margin,
    ).selected
    if (
        len(commitment.probabilities) != len(bank.action_ids)
        or commitment.intended_alice_route_index >= len(bank.route_ids)
        or not math.isclose(
            commitment.required_response_margin,
            experiment.response_margin,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("precomputed commitment does not match this experiment")
    commitment_probabilities = np.asarray(
        commitment.probabilities, dtype=float
    )
    target_route = int(commitment.intended_alice_route_index)
    audit_evaluation = evaluate_stackelberg_commitment(
        bank.audit.alice_payoff_matrix(),
        bank.action_costs,
        commitment_probabilities,
    )

    quantum_detector = RouteConditionedCategoricalEProcess(
        commissioning_counts,
        RouteOutcomeDetectorConfig(
            horizon=active_config.turns,
            model_false_alarm_level=experiment.detector_false_alarm_level,
            commissioning_pseudocount=(
                experiment.detector_commissioning_pseudocount
            ),
            change_fractions=experiment.detector_change_fractions,
        ),
    )
    classical_binary_detector = RouteConditionedCategoricalEProcess(
        binary_service_outcome_counts(commissioning_counts),
        RouteOutcomeDetectorConfig(
            horizon=active_config.turns,
            model_false_alarm_level=experiment.detector_false_alarm_level,
            commissioning_pseudocount=(
                experiment.detector_commissioning_pseudocount
            ),
            change_fractions=experiment.detector_change_fractions,
        ),
    )
    sentinel_embedding = route_resource_embedding(case)
    sentinel_design = linear_exploration_design(
        sentinel_embedding.embedding,
        mode="g_optimal",
    )
    sentinel_probabilities = np.asarray(
        sentinel_design.probabilities, dtype=float
    )
    alice = _build_alice(case, active_config, bank.route_ids)
    stream = CommonCellOutcomeStream(
        route_count=len(bank.route_ids),
        action_count=len(bank.action_ids),
        draws_per_cell=online_feedback.draws_per_cell,
        seed=int(seed),
    )
    monitoring_rng = np.random.default_rng([int(seed), 701])
    sentinel_route_rng = np.random.default_rng([int(seed), 709])
    eve_rng = np.random.default_rng([int(seed), 719])
    frozen_policy_rng = np.random.default_rng([int(seed), 727])

    route_counts_all = np.zeros(len(bank.route_ids), dtype=np.int64)
    route_counts_service = np.zeros(len(bank.route_ids), dtype=np.int64)
    action_counts = np.zeros(len(bank.action_ids), dtype=np.int64)
    service_reward_sum = 0.0
    all_reward_sum = 0.0
    eve_utility_sum = 0.0
    attack_eve_utility_sum = 0.0
    attack_request_count = 0
    clean_window_reward_sum = 0.0
    attack_window_reward_sum = 0.0
    clean_window_service_reward_sum = 0.0
    attack_window_service_reward_sum = 0.0
    clean_window_service_count = 0
    attack_window_service_count = 0
    service_count = 0
    monitoring_count = 0
    restart_turns: list[int] = []
    service_policy_frozen_turn: int | None = None
    frozen_service_policy: np.ndarray | None = None
    alarm_snapshot: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    detector_transition: list[dict[str, Any]] = []
    application_reward_windows = RollingRewardMinimumTracker(
        experiment.damage_window_turns,
        evaluation_start_turn=experiment.attack_start_turn + 1,
    )
    started = time.perf_counter()

    for turn_index in range(active_config.turns):
        step = turn_index + 1
        post_onset_window = step > experiment.attack_start_turn
        active_detector = (
            quantum_detector
            if condition.active_detector == DETECTOR_QUANTUM_FOUR_OUTCOME
            else classical_binary_detector
        )
        monitoring_fraction = (
            condition.post_alarm_monitoring_fraction
            if active_detector.alarmed
            else condition.pre_alarm_monitoring_fraction
        )
        is_monitoring = bool(
            monitoring_fraction > 0.0
            and monitoring_rng.random() < monitoring_fraction
        )
        if is_monitoring:
            route_index = int(
                sentinel_route_rng.choice(
                    len(bank.route_ids),
                    p=sentinel_probabilities,
                )
            )
        elif frozen_service_policy is not None:
            route_index = int(
                frozen_policy_rng.choice(
                    len(bank.route_ids),
                    p=frozen_service_policy,
                )
            )
        else:
            route_index = alice.sample()

        attack_active = bool(
            condition.attack_enabled
            and step > experiment.attack_start_turn
        )
        if attack_active:
            action_index = int(
                eve_rng.choice(
                    len(bank.action_ids),
                    p=commitment_probabilities,
                )
            )
        else:
            action_index = no_attack_index
        draw_index = stream.next_draw(route_index, action_index)
        observation = online_feedback.draw(
            route_index, action_index, draw_index
        )
        reward = float(observation["alice_reward"])
        if is_monitoring:
            monitoring_count += 1
        else:
            if frozen_service_policy is None:
                alice.update(reward)
            route_counts_service[route_index] += 1
            service_reward_sum += reward
            service_count += 1
            if post_onset_window:
                attack_window_service_reward_sum += reward
                attack_window_service_count += 1
            else:
                clean_window_service_reward_sum += reward
                clean_window_service_count += 1
        route_counts_all[route_index] += 1
        action_counts[action_index] += 1
        all_reward_sum += reward
        application_reward_windows.update(
            step,
            0.0 if is_monitoring else reward,
        )
        if post_onset_window:
            attack_window_reward_sum += reward
        else:
            clean_window_reward_sum += reward
        eve_utility = (
            1.0 - reward - float(bank.action_costs[action_index])
        )
        eve_utility_sum += eve_utility
        if attack_active:
            attack_request_count += 1
            attack_eve_utility_sum += eve_utility

        quantum_detector_row = quantum_detector.update(
            route_index,
            int(observation["public_outcome_code"]),
        )
        classical_binary_detector_row = classical_binary_detector.update(
            route_index,
            int(observation["public_outcome_code"] == 3),
        )
        active_detector_row = (
            quantum_detector_row
            if condition.active_detector == DETECTOR_QUANTUM_FOUR_OUTCOME
            else classical_binary_detector_row
        )
        relative_to_attack = step - experiment.attack_start_turn
        if -50 <= relative_to_attack <= 250:
            detector_transition.append({
                "turn": step,
                "turn_relative_to_attack": relative_to_attack,
                "log_e_value": float(active_detector.log_e_value),
                "alarmed": active_detector.alarmed,
                "quantum_log_e_value": float(
                    quantum_detector.log_e_value
                ),
                "quantum_alarmed": quantum_detector.alarmed,
                "classical_binary_log_e_value": float(
                    classical_binary_detector.log_e_value
                ),
                "classical_binary_alarmed": (
                    classical_binary_detector.alarmed
                ),
                "monitoring_request": is_monitoring,
                "public_outcome_code": int(
                    observation["public_outcome_code"]
                ),
            })
        if active_detector_row["newly_alarmed"]:
            alarm_snapshot = _target_mass_snapshot(
                target_route=target_route,
                route_counts_all=route_counts_all,
                route_counts_service=route_counts_service,
                alice=alice,
                policy_turn=alice.sample_calls + 1,
            )
            if condition.restart_service_learner_on_alarm:
                alice = _build_alice(
                    case,
                    replace(active_config, seed=int(seed) + 1_000_003),
                    bank.route_ids,
                )
                restart_turns.append(step)
            elif condition.freeze_service_policy_on_alarm:
                frozen_service_policy = alice.policy(
                    turn=alice.sample_calls + 1
                ).copy()
                service_policy_frozen_turn = step

        if progress_callback is not None and (
            step == 1
            or step == active_config.turns
            or step % 50_000 == 0
        ):
            progress_callback(step)
        if (
            step == 1
            or step == active_config.turns
            or step % experiment.record_every == 0
        ):
            target_snapshot = _target_mass_snapshot(
                target_route=target_route,
                route_counts_all=route_counts_all,
                route_counts_service=route_counts_service,
                alice=alice,
                policy_turn=alice.sample_calls + 1,
            )
            trajectory.append({
                "turn": step,
                "attack_active": attack_active,
                "detector_log_e_value": float(active_detector.log_e_value),
                "detector_e_value": _finite_exponential(
                    active_detector.log_e_value
                ),
                "detector_alarmed": active_detector.alarmed,
                "quantum_detector_log_e_value": float(
                    quantum_detector.log_e_value
                ),
                "classical_binary_detector_log_e_value": float(
                    classical_binary_detector.log_e_value
                ),
                "monitoring_fraction_empirical": monitoring_count / step,
                "mean_service_reward": (
                    service_reward_sum / service_count
                    if service_count else None
                ),
                **target_snapshot,
            })

    elapsed = time.perf_counter() - started
    final_active_detector = (
        quantum_detector
        if condition.active_detector == DETECTOR_QUANTUM_FOUR_OUTCOME
        else classical_binary_detector
    )
    alarm_turn = final_active_detector.alarm_turn
    pre_attack_alarm = bool(
        alarm_turn is not None
        and (
            not condition.attack_enabled
            or alarm_turn <= experiment.attack_start_turn
        )
    )
    detected_after_attack = bool(
        condition.attack_enabled
        and alarm_turn is not None
        and alarm_turn > experiment.attack_start_turn
    )
    final_snapshot = _target_mass_snapshot(
        target_route=target_route,
        route_counts_all=route_counts_all,
        route_counts_service=route_counts_service,
        alice=alice,
        policy_turn=alice.sample_calls + 1,
    )
    return {
        "schema": STACKELBERG_DETECTION_RESULT_SCHEMA,
        "graph_id": case.graph_id,
        "family": case.family,
        "case_id": bank.case_id,
        "seed": int(seed),
        "turns": int(active_config.turns),
        "elapsed_seconds": float(elapsed),
        "turns_per_second": (
            active_config.turns / elapsed if elapsed > 0.0 else None
        ),
        "condition": condition.to_dict(),
        "experiment_config": experiment.to_dict(),
        "alice_learner": dict(alice.metadata),
        "commitment": commitment.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "commitment_audit_evaluation": audit_evaluation.to_dict(),
        "intended_alice_route_index": target_route,
        "intended_alice_route_id": str(bank.route_ids[target_route]),
        "detector": final_active_detector.diagnostics(),
        "quantum_detector": quantum_detector.diagnostics(),
        "classical_binary_detector": (
            classical_binary_detector.diagnostics()
        ),
        "observer_comparison": {
            "quantum_four_outcome_alarm_turn": (
                quantum_detector.alarm_turn
            ),
            "classical_binary_service_alarm_turn": (
                classical_binary_detector.alarm_turn
            ),
            "quantum_four_outcome_detection_delay_turns": (
                quantum_detector.alarm_turn
                - experiment.attack_start_turn
                if condition.attack_enabled
                and quantum_detector.alarm_turn is not None
                and quantum_detector.alarm_turn
                > experiment.attack_start_turn
                else None
            ),
            "classical_binary_service_detection_delay_turns": (
                classical_binary_detector.alarm_turn
                - experiment.attack_start_turn
                if condition.attack_enabled
                and classical_binary_detector.alarm_turn is not None
                and classical_binary_detector.alarm_turn
                > experiment.attack_start_turn
                else None
            ),
            "semantics": {
                "quantum_four_outcome": (
                    "route identity plus delivery failure, QBER abort, "
                    "CHSH abort, or accepted"
                ),
                "classical_binary_service": (
                    "matched observation control using route identity plus "
                    "binary accepted/rejected service only"
                ),
            },
        },
        "pre_attack_alarm": pre_attack_alarm,
        "detected_after_attack": detected_after_attack,
        "detection_delay_turns": (
            int(alarm_turn - experiment.attack_start_turn)
            if detected_after_attack and alarm_turn is not None
            else None
        ),
        "alarm_target_mass": alarm_snapshot,
        "final_target_mass": final_snapshot,
        "monitoring_request_count": int(monitoring_count),
        "monitoring_request_fraction": (
            monitoring_count / active_config.turns
        ),
        "service_request_count": int(service_count),
        "mean_service_reward": (
            service_reward_sum / service_count if service_count else None
        ),
        "mean_all_request_reward": all_reward_sum / active_config.turns,
        "mean_clean_window_all_request_reward": (
            clean_window_reward_sum / experiment.attack_start_turn
        ),
        "mean_attack_window_all_request_reward": (
            attack_window_reward_sum
            / (active_config.turns - experiment.attack_start_turn)
        ),
        "mean_clean_window_service_reward": (
            clean_window_service_reward_sum / clean_window_service_count
            if clean_window_service_count else None
        ),
        "mean_attack_window_service_reward": (
            attack_window_service_reward_sum / attack_window_service_count
            if attack_window_service_count else None
        ),
        "application_reward_window_minima": (
            application_reward_windows.diagnostics()
        ),
        "application_reward_window_semantics": (
            "minimum mean application reward over every sliding window wholly "
            "inside turns after attack_start_turn; monitoring requests count "
            "as zero application reward"
        ),
        "mean_eve_public_utility": eve_utility_sum / active_config.turns,
        "mean_attack_window_eve_public_utility": (
            attack_eve_utility_sum / attack_request_count
            if attack_request_count else None
        ),
        "learner_restart_turns": restart_turns,
        "service_policy_frozen_turn": service_policy_frozen_turn,
        "frozen_service_policy": (
            frozen_service_policy.tolist()
            if frozen_service_policy is not None
            else None
        ),
        "eve_empirical_strategy": (
            action_counts / active_config.turns
        ).tolist(),
        "sentinel_design": sentinel_design.to_dict(),
        "trajectory": trajectory,
        "detector_transition": detector_transition,
        "information_boundary": {
            "detector_observes": [
                "selected route identity",
                "Alice public terminal outcome code",
                "fixed clean commissioning counts",
            ],
            "mitigation_controller_observes": [
                "detector alarm state",
                "topology-only sentinel design",
                (
                    "Alice current service policy only when the declared "
                    "response freezes that policy"
                ),
            ],
            "alice_service_learner_observes": (
                "selected service-route scalar public reward only"
            ),
            "excluded_online_information": [
                "Eve action identity",
                "Eve commitment probabilities",
                "target-active diagnostic",
                "Eve private instrument record",
                "audit outcomes",
                "counterfactual route rewards",
            ],
            "monitoring_semantics": (
                "sentinel requests consume their own entanglement request and "
                "do not enter the service learner update sequence"
            ),
        },
    }


def commissioning_outcome_counts(
    block: PayoffSampleBlock,
    *,
    action_index: int,
    stop: int,
) -> np.ndarray:
    """Count public terminal outcomes in a clean commissioning slice."""

    if not 0 <= action_index < block.shape[1]:
        raise IndexError("action_index is outside the payoff block")
    if not 1 <= stop <= block.draws_per_cell:
        raise ValueError("commissioning stop is outside the payoff block")
    values = np.asarray(
        block.public_outcome_code[:, action_index, :stop], dtype=np.int64
    )
    if np.any(values < 0) or np.any(values >= PUBLIC_OUTCOME_COUNT):
        raise ValueError("payoff block contains an invalid public outcome code")
    counts = np.zeros(
        (block.shape[0], PUBLIC_OUTCOME_COUNT), dtype=np.int64
    )
    for outcome_code in range(PUBLIC_OUTCOME_COUNT):
        counts[:, outcome_code] = np.sum(
            values == outcome_code, axis=1
        )
    return counts


def binary_service_outcome_counts(
    four_outcome_counts: np.ndarray,
) -> np.ndarray:
    """Coarsen quantum terminal categories to rejected/accepted service."""

    counts = np.asarray(four_outcome_counts, dtype=np.int64)
    if (
        counts.ndim != 2
        or counts.shape[1] != PUBLIC_OUTCOME_COUNT
        or np.any(counts < 0)
    ):
        raise ValueError("four_outcome_counts must be nonnegative route x 4")
    return np.column_stack((
        np.sum(counts[:, :3], axis=1),
        counts[:, 3],
    ))


def slice_payoff_sample_block(
    block: PayoffSampleBlock,
    start: int,
    stop: int,
) -> PayoffSampleBlock:
    """Return a draw-axis view while preserving every transcript field."""

    if not 0 <= start < stop <= block.draws_per_cell:
        raise ValueError("invalid payoff-block draw slice")
    if stop - start < 2:
        raise ValueError("payoff-block slice must retain at least two draws")
    return PayoffSampleBlock(**{
        name: np.asarray(getattr(block, name))[:, :, start:stop]
        for name in (
            "alice_reward",
            "corrected_alice_reward",
            "eve_bandit_reward",
            "bell_qualified_rate_hz",
            "false_signal",
            "classifiable_signal",
            "target_active",
            "eve_observation_count",
            "information_observed",
            "public_outcome_code",
            "signal_class_code",
        )
    })


def _build_alice(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    route_ids: Sequence[str],
) -> PlayerLearner:
    route_count = len(route_ids)
    implementation, metadata = build_histogram_alice_learner(
        case=case,
        config=config,
        prior=np.full(route_count, 1.0 / route_count, dtype=float),
    )
    return PlayerLearner(
        player="alice",
        action_ids=tuple(map(str, route_ids)),
        learner=implementation,
        metadata=metadata,
    )


def _target_mass_snapshot(
    *,
    target_route: int,
    route_counts_all: np.ndarray,
    route_counts_service: np.ndarray,
    alice: PlayerLearner,
    policy_turn: int,
) -> dict[str, Any]:
    all_count = int(np.sum(route_counts_all))
    service_count = int(np.sum(route_counts_service))
    policy = alice.policy(turn=max(1, int(policy_turn)))
    return {
        "intended_route_all_request_mass": (
            float(route_counts_all[target_route] / all_count)
            if all_count else None
        ),
        "intended_route_service_request_mass": (
            float(route_counts_service[target_route] / service_count)
            if service_count else None
        ),
        "intended_route_current_service_policy_mass": float(
            policy[target_route]
        ),
    }


def _finite_exponential(value: float) -> float | None:
    if value > math.log(np.finfo(float).max):
        return None
    return float(math.exp(value))
