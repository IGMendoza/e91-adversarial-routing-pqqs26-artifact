"""Alarm-conditioned Alice responses on held-out endpoint evidence.

The response layer branches from one exact post-follow Alice state.  Detector
selection uses public endpoint evidence only.  Fixed response policies may use
the disjoint feedback bank as a commissioned threat model, but never the
realized Eve action, target, or audit payoff matrix.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.special import betaln

from .calibration_bank import (
    PUBLIC_OUTCOME_CODES,
    HistogramPayoffBank,
    validate_bank_against_case,
)
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    validate_endpoint_evidence_bank_against_case,
)
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .quantum_stackelberg_detection import (
    CHSH_SETTING_LABELS,
    CHSH_SAME_CHANNEL_PREFIX,
    KEY_ERROR_CHANNEL,
    PHYSICAL_DELIVERY_CHANNEL,
    PUBLIC_SERVICE_ACCEPTANCE_CHANNEL,
    MatchedClassicalQuantumEndpointDetectors,
    RouteConditionedBinomialEProcess,
)
from .stackelberg_bandit import RouteObservationDesign
from .stackelberg_controller import (
    ControllerPrimeExploitSchedule,
    passive_exploit_trigger,
)
from .stackelberg_controller_plans import ControllerPrimeExploitPlan


ENDPOINT_RESPONSE_RUN_SCHEMA = (
    "histogram_stackelberg_endpoint_response_run.v1"
)
DETECTOR_CLASSICAL_SERVICE = "classical_service"
DETECTOR_QUANTUM_ENDPOINT = "quantum_endpoint"
ACTIVE_RESPONSE_DETECTORS = frozenset({
    DETECTOR_CLASSICAL_SERVICE,
    DETECTOR_QUANTUM_ENDPOINT,
})

RESPONSE_NATIVE_CONTINUATION = "native_continuation"
RESPONSE_FREEZE_LIVE_POLICY = "freeze_live_policy"
RESPONSE_QUARANTINE_LEADING_ROUTE = "quarantine_leading_route"
RESPONSE_COMMISSIONED_ROBUST_POLICY = "commissioned_robust_policy"
RESPONSE_POSTERIOR_TRUST_REGION = "posterior_trust_region"
RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION = (
    "sequential_posterior_trust_region"
)
RESPONSE_UPDATE_NATIVE_ONLY = "native_policy_turns_only"
RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW = (
    "tsallis_behavior_importance_weighted"
)
RESPONSE_UPDATE_MODES = frozenset({
    RESPONSE_UPDATE_NATIVE_ONLY,
    RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW,
})
ALARM_RESPONSE_STRATEGIES = frozenset({
    RESPONSE_NATIVE_CONTINUATION,
    RESPONSE_FREEZE_LIVE_POLICY,
    RESPONSE_QUARANTINE_LEADING_ROUTE,
    RESPONSE_COMMISSIONED_ROBUST_POLICY,
    RESPONSE_POSTERIOR_TRUST_REGION,
    RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
})


@dataclass(frozen=True)
class AlarmResponseConfig:
    """Frozen response selection and reporting contract."""

    strategy: str
    active_detector: str
    switching_penalty: float = 0.05
    compatibility_effect_floor: float = 0.01
    topology_sharing_trust_region_tv: float = 0.25
    independent_arm_trust_region_tv: float = 0.10
    topology_sharing_arming_turns: int = 4
    independent_arm_arming_turns: int = 12
    topology_sharing_containment_ceiling: float = 0.50
    independent_arm_containment_ceiling: float = 0.25
    independent_arm_update_mode: str = RESPONSE_UPDATE_NATIVE_ONLY
    posterior_pseudocount: float = 0.5
    posterior_window_turns: int = 64
    posterior_refresh_turns: int = 8
    posterior_model_draw_start: int = 0
    posterior_model_draw_stop: int | None = None
    record_every: int = 100

    def __post_init__(self) -> None:
        if self.strategy not in ALARM_RESPONSE_STRATEGIES:
            raise ValueError(f"unsupported response strategy {self.strategy!r}")
        if self.active_detector not in ACTIVE_RESPONSE_DETECTORS:
            raise ValueError(
                f"unsupported active detector {self.active_detector!r}"
            )
        if (
            not math.isfinite(self.switching_penalty)
            or self.switching_penalty < 0.0
        ):
            raise ValueError(
                "switching_penalty must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.compatibility_effect_floor)
            or not 0.0 <= self.compatibility_effect_floor <= 1.0
        ):
            raise ValueError(
                "compatibility_effect_floor must lie in [0, 1]"
            )
        for name, value in (
            (
                "topology_sharing_trust_region_tv",
                self.topology_sharing_trust_region_tv,
            ),
            (
                "independent_arm_trust_region_tv",
                self.independent_arm_trust_region_tv,
            ),
            (
                "topology_sharing_containment_ceiling",
                self.topology_sharing_containment_ceiling,
            ),
            (
                "independent_arm_containment_ceiling",
                self.independent_arm_containment_ceiling,
            ),
        ):
            if (
                not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.independent_arm_update_mode not in RESPONSE_UPDATE_MODES:
            raise ValueError(
                "independent_arm_update_mode is not a declared response "
                "update mode"
            )
        for name, value in (
            (
                "topology_sharing_arming_turns",
                self.topology_sharing_arming_turns,
            ),
            (
                "independent_arm_arming_turns",
                self.independent_arm_arming_turns,
            ),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.posterior_pseudocount)
            or self.posterior_pseudocount <= 0.0
        ):
            raise ValueError(
                "posterior_pseudocount must be finite and positive"
            )
        if self.posterior_window_turns < 1:
            raise ValueError("posterior_window_turns must be positive")
        if self.posterior_refresh_turns < 1:
            raise ValueError("posterior_refresh_turns must be positive")
        if self.posterior_model_draw_start < 0:
            raise ValueError(
                "posterior_model_draw_start must be nonnegative"
            )
        if (
            self.posterior_model_draw_stop is not None
            and self.posterior_model_draw_stop
            <= self.posterior_model_draw_start
        ):
            raise ValueError(
                "posterior_model_draw_stop must exceed the start"
            )
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommissionedRobustRoutePolicy:
    """One feedback-selected route distribution and its declared threat set."""

    probabilities: tuple[float, ...]
    candidate_action_indices: tuple[int, ...]
    candidate_action_ids: tuple[str, ...]
    feedback_worst_case_reward: float
    feedback_objective: float
    total_variation_from_live_policy: float
    switching_penalty: float
    leading_channel: str
    leading_direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommissionedInterventionPosterior:
    """Feedback-bank posterior from one detector-leading evidence channel."""

    action_indices: tuple[int, ...]
    action_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    log_likelihoods: tuple[float, ...]
    prior_probabilities: tuple[float, ...]
    leading_route_index: int
    leading_channel: str
    leading_direction: str
    observed_successes: int
    observed_trials: int
    baseline_probability: float
    commissioned_probabilities: tuple[float, ...]
    compatibility_effect_floor: float
    posterior_entropy_nats: float
    posterior_effective_action_count: float
    model: str = "single_leading_channel_beta_binomial"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PosteriorTrustRegionRoutePolicy:
    """Posterior-weighted route policy inside a live-policy TV trust region."""

    probabilities: tuple[float, ...]
    posterior: CommissionedInterventionPosterior
    feedback_posterior_expected_reward: float
    feedback_worst_case_reward: float
    feedback_objective: float
    total_variation_from_live_policy: float
    maximum_total_variation: float
    switching_penalty: float
    feedback_draw_start: int
    feedback_draw_stop: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["posterior"] = self.posterior.to_dict()
        return payload


class SequentialInterventionPosterior:
    """Rolling one-channel filter over commissioned intervention emissions."""

    def __init__(
        self,
        evidence_bank: HistogramEndpointEvidenceBank,
        initial: CommissionedInterventionPosterior,
        *,
        window_turns: int,
        pseudocount: float,
        feedback_draw_start: int,
        feedback_draw_stop: int | None,
    ) -> None:
        self.evidence_bank = evidence_bank
        self.initial = initial
        self.window_turns = int(window_turns)
        self.pseudocount = float(pseudocount)
        if self.window_turns < 1:
            raise ValueError("sequential posterior window must be positive")
        if (
            not math.isfinite(self.pseudocount)
            or self.pseudocount <= 0.0
        ):
            raise ValueError(
                "sequential posterior pseudocount must be positive"
            )
        self.feedback_draw_start, self.feedback_draw_stop = _draw_bounds(
            evidence_bank.draws_per_cell,
            feedback_draw_start,
            feedback_draw_stop,
            label="endpoint feedback model",
        )
        self.action_indices = tuple(map(int, initial.action_indices))
        self.action_ids = tuple(initial.action_ids)
        self.channel = str(initial.leading_channel)
        self.direction = str(initial.leading_direction)
        self.leading_route_index = int(initial.leading_route_index)
        successes, trials = _feedback_channel_counts(
            evidence_bank,
            self.channel,
            draw_start=self.feedback_draw_start,
            draw_stop=self.feedback_draw_stop,
        )
        selected_successes = successes[:, self.action_indices].astype(float)
        selected_trials = trials[:, self.action_indices].astype(float)
        self._commissioned_probabilities = (
            selected_successes + self.pseudocount
        ) / (
            selected_trials + 2.0 * self.pseudocount
        )
        self._commissioned_probabilities = np.clip(
            self._commissioned_probabilities,
            1e-12,
            1.0 - 1e-12,
        )
        initial_probabilities = _probability_vector(
            initial.probabilities,
            expected_size=len(self.action_indices),
            label="initial intervention posterior",
        )
        self._base_log_prior = np.log(
            np.clip(initial_probabilities, 1e-300, 1.0)
        )
        self._observations: deque[tuple[int, int, int]] = deque(
            maxlen=self.window_turns
        )
        self.update_calls = 0
        self._posterior = initial_probabilities
        self._log_likelihoods = np.zeros(
            len(self.action_indices), dtype=float
        )

    def update(
        self,
        route_index: int,
        endpoint_evidence: Any,
    ) -> CommissionedInterventionPosterior:
        route = int(route_index)
        if not 0 <= route < len(self.evidence_bank.route_ids):
            raise IndexError("sequential posterior route is out of range")
        successes, trials = endpoint_evidence.binomial_counts(
            (self.channel,)
        )
        success_count = int(successes[0])
        trial_count = int(trials[0])
        if (
            success_count < 0
            or trial_count < success_count
        ):
            raise ValueError(
                "sequential posterior received invalid endpoint counts"
            )
        self._observations.append(
            (route, success_count, trial_count)
        )
        self.update_calls += 1
        self._recompute()
        return self.posterior()

    def posterior(self) -> CommissionedInterventionPosterior:
        leading_probabilities = self._commissioned_probabilities[
            self.leading_route_index
        ]
        successes = sum(row[1] for row in self._observations)
        trials = sum(row[2] for row in self._observations)
        positive = self._posterior[self._posterior > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        baseline_index = self.action_ids.index("no_attack")
        return CommissionedInterventionPosterior(
            action_indices=self.action_indices,
            action_ids=self.action_ids,
            probabilities=tuple(map(float, self._posterior)),
            log_likelihoods=tuple(map(float, self._log_likelihoods)),
            prior_probabilities=tuple(
                map(float, np.exp(self._base_log_prior))
            ),
            leading_route_index=self.leading_route_index,
            leading_channel=self.channel,
            leading_direction=self.direction,
            observed_successes=int(successes),
            observed_trials=int(trials),
            baseline_probability=float(
                leading_probabilities[baseline_index]
            ),
            commissioned_probabilities=tuple(
                map(float, leading_probabilities)
            ),
            compatibility_effect_floor=(
                self.initial.compatibility_effect_floor
            ),
            posterior_entropy_nats=entropy,
            posterior_effective_action_count=float(math.exp(entropy)),
            model="rolling_single_channel_commissioned_likelihood",
        )

    def diagnostics(self) -> dict[str, Any]:
        posterior = self.posterior()
        no_attack = self.action_ids.index("no_attack")
        route_trials = np.zeros(
            len(self.evidence_bank.route_ids), dtype=np.int64
        )
        for route, _successes, trials in self._observations:
            route_trials[route] += int(trials)
        return {
            "schema": "sequential_intervention_posterior.v1",
            "posterior": posterior.to_dict(),
            "window_turns": self.window_turns,
            "window_observation_count": len(self._observations),
            "update_calls": self.update_calls,
            "posterior_attack_mass": float(
                1.0 - self._posterior[no_attack]
            ),
            "feedback_model_draw_start": self.feedback_draw_start,
            "feedback_model_draw_stop": self.feedback_draw_stop,
            "route_trial_counts": route_trials.tolist(),
            "channel_dependence": (
                "one endpoint channel only; route-conditioned likelihoods "
                "are accumulated across observed turns"
            ),
            "forgetting": (
                "fixed alarm posterior prior plus a rolling post-alarm "
                "observation window"
            ),
        }

    def _recompute(self) -> None:
        route_count = len(self.evidence_bank.route_ids)
        successes = np.zeros(route_count, dtype=np.int64)
        trials = np.zeros(route_count, dtype=np.int64)
        for route, success_count, trial_count in self._observations:
            successes[route] += int(success_count)
            trials[route] += int(trial_count)
        log_likelihoods = np.zeros(
            len(self.action_indices), dtype=float
        )
        for route in np.flatnonzero(trials > 0):
            probability = self._commissioned_probabilities[route]
            success_count = int(successes[route])
            trial_count = int(trials[route])
            failure_count = trial_count - success_count
            log_likelihoods += (
                success_count * np.log(probability)
                + failure_count * np.log1p(-probability)
            )
        log_weights = self._base_log_prior + log_likelihoods
        maximum = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum)
        self._posterior = weights / np.sum(weights)
        self._log_likelihoods = log_likelihoods


def solve_commissioned_robust_route_policy(
    payoff_bank: HistogramPayoffBank,
    current_policy: Sequence[float] | np.ndarray,
    candidate_action_indices: Sequence[int],
    *,
    switching_penalty: float,
    leading_channel: str,
    leading_direction: str,
) -> CommissionedRobustRoutePolicy:
    """Maximize worst-case feedback reward minus a live-policy switch cost."""

    current = _probability_vector(
        current_policy,
        expected_size=len(payoff_bank.route_ids),
        label="current_policy",
    )
    candidates = tuple(sorted(set(map(int, candidate_action_indices))))
    if (
        not candidates
        or candidates[0] < 0
        or candidates[-1] >= len(payoff_bank.action_ids)
    ):
        raise ValueError("candidate action indices are empty or out of range")
    penalty = float(switching_penalty)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("switching_penalty must be finite and nonnegative")

    payoff = payoff_bank.feedback.alice_payoff_matrix()[:, candidates]
    route_count = payoff.shape[0]
    action_count = payoff.shape[1]
    # Variables are route mass x, guaranteed reward v, and |x-current| d.
    objective = np.concatenate([
        np.zeros(route_count),
        [-1.0],
        np.full(route_count, 0.5 * penalty),
    ])
    worst_case_constraints = np.column_stack([
        -payoff.T,
        np.ones(action_count),
        np.zeros((action_count, route_count)),
    ])
    positive_deviation = np.column_stack([
        np.eye(route_count),
        np.zeros(route_count),
        -np.eye(route_count),
    ])
    negative_deviation = np.column_stack([
        -np.eye(route_count),
        np.zeros(route_count),
        -np.eye(route_count),
    ])
    result = linprog(
        objective,
        A_ub=np.vstack([
            worst_case_constraints,
            positive_deviation,
            negative_deviation,
        ]),
        b_ub=np.concatenate([
            np.zeros(action_count),
            current,
            -current,
        ]),
        A_eq=np.concatenate([
            np.ones(route_count),
            [0.0],
            np.zeros(route_count),
        ])[None, :],
        b_eq=np.asarray([1.0]),
        bounds=(
            [(0.0, 1.0)] * route_count
            + [(None, None)]
            + [(0.0, 1.0)] * route_count
        ),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"commissioned robust response LP failed: {result.message}"
        )
    policy = _probability_vector(
        result.x[:route_count],
        expected_size=route_count,
        label="robust policy",
    )
    worst_case = float(np.min(policy @ payoff))
    total_variation = 0.5 * float(np.sum(np.abs(policy - current)))
    feedback_objective = worst_case - penalty * total_variation
    return CommissionedRobustRoutePolicy(
        probabilities=tuple(map(float, policy)),
        candidate_action_indices=candidates,
        candidate_action_ids=tuple(
            payoff_bank.action_ids[index] for index in candidates
        ),
        feedback_worst_case_reward=worst_case,
        feedback_objective=feedback_objective,
        total_variation_from_live_policy=total_variation,
        switching_penalty=penalty,
        leading_channel=str(leading_channel),
        leading_direction=str(leading_direction),
    )


def commissioned_intervention_posterior(
    evidence_bank: HistogramEndpointEvidenceBank,
    detector_diagnostics: Mapping[str, Any],
    leading: Mapping[str, Any],
    *,
    effect_floor: float,
    pseudocount: float = 0.5,
    feedback_draw_start: int = 0,
    feedback_draw_stop: int | None = None,
) -> CommissionedInterventionPosterior:
    """Infer compatible interventions from one public detector channel.

    Only the detector-leading route/channel counts and the disjoint feedback
    bank enter this posterior. Using one channel avoids an unsupported
    conditional-independence product across delivery, QBER, and CHSH fields.
    """

    route = int(leading["route_index"])
    channel = str(leading["channel"])
    direction = str(leading["direction"])
    if direction not in {"increase", "decrease"}:
        raise ValueError("detector direction must be increase or decrease")
    if not 0 <= route < len(evidence_bank.route_ids):
        raise IndexError("detector-leading route is outside the evidence bank")
    floor = float(effect_floor)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("effect_floor must lie in [0, 1]")
    pseudo = float(pseudocount)
    if not math.isfinite(pseudo) or pseudo <= 0.0:
        raise ValueError("pseudocount must be finite and positive")

    labels = tuple(map(str, detector_diagnostics["channel_labels"]))
    try:
        channel_index = labels.index(channel)
    except ValueError as exc:
        raise ValueError(
            f"detector diagnostics do not contain channel {channel!r}"
        ) from exc
    online_successes = np.asarray(
        detector_diagnostics["online_successes"],
        dtype=np.int64,
    )
    online_trials = np.asarray(
        detector_diagnostics["online_trials"],
        dtype=np.int64,
    )
    expected_shape = (len(evidence_bank.route_ids), len(labels))
    if (
        online_successes.shape != expected_shape
        or online_trials.shape != expected_shape
        or np.any(online_successes < 0)
        or np.any(online_trials < online_successes)
    ):
        raise ValueError("detector online count arrays are invalid")
    observed_successes = int(online_successes[route, channel_index])
    observed_trials = int(online_trials[route, channel_index])
    if observed_trials <= 0:
        raise ValueError(
            "detector-leading route/channel has no online trials"
        )

    commissioned_successes, commissioned_trials = _feedback_channel_counts(
        evidence_bank,
        channel,
        draw_start=feedback_draw_start,
        draw_stop=feedback_draw_stop,
    )
    action_successes = commissioned_successes[route].astype(float)
    action_trials = commissioned_trials[route].astype(float)
    posterior_means = (
        action_successes + pseudo
    ) / (action_trials + 2.0 * pseudo)
    no_attack = evidence_bank.action_ids.index("no_attack")
    baseline = float(posterior_means[no_attack])
    signed_effect = (
        posterior_means - baseline
        if direction == "increase"
        else baseline - posterior_means
    )
    compatible = set(map(int, np.flatnonzero(
        (signed_effect >= floor) & (action_trials > 0.0)
    )))
    compatible.add(no_attack)
    candidates = tuple(sorted(compatible))
    if action_trials[no_attack] <= 0.0:
        raise ValueError(
            "no_attack commissioning has no leading-channel trials"
        )

    successes = action_successes[list(candidates)]
    trials = action_trials[list(candidates)]
    alpha = successes + pseudo
    beta = trials - successes + pseudo
    failures = observed_trials - observed_successes
    log_likelihoods = (
        betaln(alpha + observed_successes, beta + failures)
        - betaln(alpha, beta)
    )
    prior = np.full(len(candidates), 1.0 / len(candidates), dtype=float)
    log_weights = log_likelihoods + np.log(prior)
    maximum = float(np.max(log_weights))
    weights = np.exp(log_weights - maximum)
    probabilities = weights / np.sum(weights)
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return CommissionedInterventionPosterior(
        action_indices=candidates,
        action_ids=tuple(
            evidence_bank.action_ids[index] for index in candidates
        ),
        probabilities=tuple(map(float, probabilities)),
        log_likelihoods=tuple(map(float, log_likelihoods)),
        prior_probabilities=tuple(map(float, prior)),
        leading_route_index=route,
        leading_channel=channel,
        leading_direction=direction,
        observed_successes=observed_successes,
        observed_trials=observed_trials,
        baseline_probability=baseline,
        commissioned_probabilities=tuple(
            map(float, posterior_means[list(candidates)])
        ),
        compatibility_effect_floor=floor,
        posterior_entropy_nats=entropy,
        posterior_effective_action_count=float(math.exp(entropy)),
    )


def solve_posterior_trust_region_route_policy(
    payoff_bank: HistogramPayoffBank,
    current_policy: Sequence[float] | np.ndarray,
    posterior: CommissionedInterventionPosterior,
    *,
    switching_penalty: float,
    maximum_total_variation: float,
    feedback_draw_start: int = 0,
    feedback_draw_stop: int | None = None,
) -> PosteriorTrustRegionRoutePolicy:
    """Optimize commissioned posterior reward inside a live-policy TV ball."""

    current = _probability_vector(
        current_policy,
        expected_size=len(payoff_bank.route_ids),
        label="current_policy",
    )
    candidates = tuple(map(int, posterior.action_indices))
    weights = _probability_vector(
        posterior.probabilities,
        expected_size=len(candidates),
        label="intervention posterior",
    )
    penalty = float(switching_penalty)
    radius = float(maximum_total_variation)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("switching_penalty must be finite and nonnegative")
    if not math.isfinite(radius) or not 0.0 <= radius <= 1.0:
        raise ValueError("maximum_total_variation must lie in [0, 1]")

    start, stop = _draw_bounds(
        payoff_bank.draws_per_cell,
        feedback_draw_start,
        feedback_draw_stop,
        label="payoff feedback model",
    )
    payoff = np.mean(
        payoff_bank.feedback.alice_reward[:, candidates, start:stop],
        axis=2,
    )
    posterior_route_reward = payoff @ weights
    policy = _solve_linear_tv_trust_region(
        current,
        posterior_route_reward,
        switching_penalty=penalty,
        maximum_total_variation=radius,
    )
    action_values = policy @ payoff
    expected_reward = float(action_values @ weights)
    worst_case = float(np.min(action_values))
    total_variation = 0.5 * float(np.sum(np.abs(policy - current)))
    return PosteriorTrustRegionRoutePolicy(
        probabilities=tuple(map(float, policy)),
        posterior=posterior,
        feedback_posterior_expected_reward=expected_reward,
        feedback_worst_case_reward=worst_case,
        feedback_objective=(
            expected_reward - penalty * total_variation
        ),
        total_variation_from_live_policy=total_variation,
        maximum_total_variation=radius,
        switching_penalty=penalty,
        feedback_draw_start=start,
        feedback_draw_stop=stop,
    )


def _solve_linear_tv_trust_region(
    current_policy: np.ndarray,
    route_rewards: np.ndarray,
    *,
    switching_penalty: float,
    maximum_total_variation: float,
) -> np.ndarray:
    """Solve a simplex linear objective with TV cost by exact mass transport."""

    current = np.asarray(current_policy, dtype=float)
    rewards = np.asarray(route_rewards, dtype=float)
    if current.shape != rewards.shape or current.ndim != 1:
        raise ValueError("trust-region policy and rewards have wrong shapes")
    policy = current.copy()
    budget = float(maximum_total_variation)
    donors = np.argsort(rewards, kind="stable")
    recipients = np.argsort(-rewards, kind="stable")
    donor_pointer = 0
    recipient_pointer = 0
    tolerance = 1e-15

    while budget > tolerance:
        while (
            donor_pointer < donors.size
            and policy[donors[donor_pointer]] <= tolerance
        ):
            donor_pointer += 1
        while (
            recipient_pointer < recipients.size
            and 1.0 - policy[recipients[recipient_pointer]]
            <= tolerance
        ):
            recipient_pointer += 1
        if (
            donor_pointer >= donors.size
            or recipient_pointer >= recipients.size
        ):
            break
        donor = int(donors[donor_pointer])
        recipient = int(recipients[recipient_pointer])
        if donor == recipient:
            break
        marginal_gain = (
            rewards[recipient]
            - rewards[donor]
            - float(switching_penalty)
        )
        if marginal_gain <= tolerance:
            break
        moved = min(
            float(policy[donor]),
            float(1.0 - policy[recipient]),
            budget,
        )
        if moved <= tolerance:
            break
        policy[donor] -= moved
        policy[recipient] += moved
        budget -= moved

    return _probability_vector(
        policy,
        expected_size=current.size,
        label="posterior trust-region policy",
    )


def commissioned_detector_compatible_actions(
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    channel: str,
    direction: str,
    effect_floor: float,
) -> tuple[int, ...]:
    """Select a commissioned threat set from one detector alternative.

    The returned set is based only on the feedback endpoint bank.  It includes
    ``no_attack`` as a benign compatibility column and every action whose
    commissioned channel effect moves in the detector-leading direction on at
    least one route.
    """

    if direction not in {"increase", "decrease"}:
        raise ValueError("detector direction must be increase or decrease")
    floor = float(effect_floor)
    if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("effect_floor must lie in [0, 1]")
    matrix = _feedback_channel_matrix(evidence_bank, channel)
    no_attack = evidence_bank.action_ids.index("no_attack")
    baseline = matrix[:, no_attack][:, None]
    signed_effect = (
        matrix - baseline
        if direction == "increase"
        else baseline - matrix
    )
    compatible = set(map(int, np.flatnonzero(
        np.max(signed_effect, axis=0) >= floor
    )))
    compatible.add(no_attack)
    return tuple(sorted(compatible))


def discover_alarm_response_record(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    config: HistogramCoLearningConfig,
    plan: ControllerPrimeExploitPlan,
    *,
    schedule: ControllerPrimeExploitSchedule,
    alice: PlayerLearner,
    observation_design: RouteObservationDesign,
    detectors: MatchedClassicalQuantumEndpointDetectors,
    active_detector: str,
    seed: int,
    attack_enabled: bool,
    execution_split: str = "audit",
    execution_draw_start: int = 0,
    execution_draw_stop: int | None = None,
    selection_digest_blake2b_128: str | None = None,
) -> dict[str, Any]:
    """Replay only until one detector alarms and freeze its public record."""

    if active_detector not in ACTIVE_RESPONSE_DETECTORS:
        raise ValueError(f"unsupported active detector {active_detector!r}")
    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        payoff_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    validate_endpoint_evidence_bank_against_case(
        evidence_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    _validate_response_inputs(
        payoff_bank,
        evidence_bank,
        plan,
        alice,
        observation_design,
    )

    execution_block = _endpoint_execution_block(
        evidence_bank,
        execution_split,
    )
    draw_start, draw_stop = _draw_bounds(
        execution_block.draws_per_cell,
        execution_draw_start,
        execution_draw_stop,
        label=f"{execution_split} alarm execution",
    )
    no_attack_index = payoff_bank.action_ids.index("no_attack")
    stream = CommonCellOutcomeStream(
        route_count=len(payoff_bank.route_ids),
        action_count=len(payoff_bank.action_ids),
        draws_per_cell=draw_stop - draw_start,
        seed=int(seed),
    )
    prime_rng = np.random.default_rng([int(seed), 613])
    prime_indices = np.asarray(plan.prime_action_indices, dtype=np.int64)
    prime_probabilities = np.asarray(plan.prime_probabilities, dtype=float)
    target_set = set(map(int, plan.target_route_indices))
    passive_target_observations: list[int] = []
    trigger: dict[str, Any] | None = None
    started = time.perf_counter()

    for turn_index in range(schedule.total_turns):
        step = turn_index + 1
        if turn_index < schedule.prime_turns:
            planned_action_index = int(
                prime_rng.choice(prime_indices, p=prime_probabilities)
            )
        else:
            if trigger is None:
                trigger = passive_exploit_trigger(
                    passive_target_observations,
                    threshold=plan.profitable_target_mass_threshold,
                    z=schedule.trigger_confidence_z,
                )
            planned_action_index = (
                plan.exploit_action_index
                if bool(trigger["triggered"])
                else no_attack_index
            )
        action_index = (
            planned_action_index if attack_enabled else no_attack_index
        )
        route_index = alice.sample()
        draw_index = (
            draw_start + stream.next_draw(route_index, action_index)
        )
        observation = execution_block.draw(
            route_index,
            action_index,
            draw_index,
            requested_pair_count=evidence_bank.requested_pairs_per_draw,
        )
        alice.update(float(observation["alice_reward"]))
        detector_rows = detectors.update(
            route_index,
            observation["endpoint_evidence"],
        )
        detector, row = _active_detector_and_row(
            detectors,
            detector_rows,
            active_detector,
        )

        signature = observation_design.signature_for_route(route_index)
        observed_routes = observation_design.routes_for_signature(signature)
        if len(observed_routes) != 1:
            raise AssertionError(
                "declared exact reservation signature became ambiguous"
            )
        if turn_index < schedule.prime_turns:
            passive_target_observations.append(
                int(int(observed_routes[0]) in target_set)
            )
        if row["newly_alarmed"]:
            return {
                "schema": "frozen_endpoint_alarm_record.v1",
                "active_detector": active_detector,
                "alarm_turn": step,
                "alarm_phase": (
                    "prime"
                    if step <= schedule.prime_turns
                    else "exploit"
                ),
                "alarm_log_e_value": float(row["log_e_value"]),
                "leading_alternative": dict(row["leading_alternative"]),
                "detector_diagnostics_at_alarm": detector.diagnostics(),
                "state_at_alarm": {
                    "alice_sample_calls": int(alice.sample_calls),
                    "alice_update_calls": int(alice.update_calls),
                    "alice_policy": alice.policy().tolist(),
                },
                "elapsed_seconds": time.perf_counter() - started,
                "execution_split": str(execution_split),
                "execution_draw_start": draw_start,
                "execution_draw_stop": draw_stop,
                "information_boundary": (
                    "public active-detector evidence on the held-out audit "
                    "trace; no Eve action or target identity"
                ),
            }

    detector, _ = _active_detector_and_row(
        detectors,
        None,
        active_detector,
    )
    return {
        "schema": "frozen_endpoint_alarm_record.v1",
        "active_detector": active_detector,
        "alarm_turn": None,
        "alarm_phase": None,
        "alarm_log_e_value": None,
        "leading_alternative": None,
        "detector_diagnostics_at_horizon": detector.diagnostics(),
        "state_at_horizon": {
            "alice_sample_calls": int(alice.sample_calls),
            "alice_update_calls": int(alice.update_calls),
            "alice_policy": alice.policy().tolist(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "execution_split": str(execution_split),
        "execution_draw_start": draw_start,
        "execution_draw_stop": draw_stop,
        "information_boundary": (
            "public active-detector evidence on the held-out audit trace; "
            "no Eve action or target identity"
        ),
    }


def run_alarm_conditioned_response_continuation(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    config: HistogramCoLearningConfig,
    plan: ControllerPrimeExploitPlan,
    *,
    schedule: ControllerPrimeExploitSchedule,
    alice: PlayerLearner,
    observation_design: RouteObservationDesign,
    detectors: MatchedClassicalQuantumEndpointDetectors,
    response_config: AlarmResponseConfig,
    seed: int,
    attack_enabled: bool,
    condition: str,
    frozen_alarm_record: Mapping[str, Any] | None = None,
    execution_split: str = "audit",
    execution_draw_start: int = 0,
    execution_draw_stop: int | None = None,
    selection_digest_blake2b_128: str | None = None,
) -> dict[str, Any]:
    """Continue one exact Alice state and apply a response after public alarm."""

    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        payoff_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    validate_endpoint_evidence_bank_against_case(
        evidence_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    _validate_response_inputs(
        payoff_bank,
        evidence_bank,
        plan,
        alice,
        observation_design,
    )

    execution_block = _endpoint_execution_block(
        evidence_bank,
        execution_split,
    )
    draw_start, draw_stop = _draw_bounds(
        execution_block.draws_per_cell,
        execution_draw_start,
        execution_draw_stop,
        label=f"{execution_split} response execution",
    )
    if frozen_alarm_record is not None:
        _validate_frozen_alarm_execution(
            frozen_alarm_record,
            execution_split=execution_split,
            draw_start=draw_start,
            draw_stop=draw_stop,
        )

    no_attack_index = payoff_bank.action_ids.index("no_attack")
    stream = CommonCellOutcomeStream(
        route_count=len(payoff_bank.route_ids),
        action_count=len(payoff_bank.action_ids),
        draws_per_cell=draw_stop - draw_start,
        seed=int(seed),
    )
    prime_rng = np.random.default_rng([int(seed), 613])
    response_rng = np.random.default_rng([int(seed), 881])
    prime_indices = np.asarray(plan.prime_action_indices, dtype=np.int64)
    prime_probabilities = np.asarray(plan.prime_probabilities, dtype=float)
    target_set = set(map(int, plan.target_route_indices))
    passive_target_observations: list[int] = []
    trigger: dict[str, Any] | None = None
    fixed_response_policy: np.ndarray | None = None
    response_selection: dict[str, Any] | None = None
    sequential_posterior: SequentialInterventionPosterior | None = None
    response_policy_refreshes = 0
    response_alarm_turn: int | None = None
    response_effective_turn: int | None = None
    response_reward_sum = 0.0
    response_reward_count = 0
    response_learner_controlled_turns = 0
    response_containment_controlled_turns = 0
    response_behavior_mixture_turns = 0
    response_containment_probability_sum = 0.0
    reward_sum = 0.0
    phase_reward_sums: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    route_counts = np.zeros(len(payoff_bank.route_ids), dtype=np.int64)
    response_route_counts = np.zeros(
        len(payoff_bank.route_ids), dtype=np.int64
    )
    trajectory: list[dict[str, Any]] = []
    starting_sample_calls = int(alice.sample_calls)
    starting_update_calls = int(alice.update_calls)
    started = time.perf_counter()

    for turn_index in range(schedule.total_turns):
        step = turn_index + 1
        phase = "prime" if turn_index < schedule.prime_turns else "exploit"
        if phase == "prime":
            planned_action_index = int(
                prime_rng.choice(prime_indices, p=prime_probabilities)
            )
        else:
            if trigger is None:
                trigger = passive_exploit_trigger(
                    passive_target_observations,
                    threshold=plan.profitable_target_mass_threshold,
                    z=schedule.trigger_confidence_z,
                )
            planned_action_index = (
                plan.exploit_action_index
                if bool(trigger["triggered"])
                else no_attack_index
            )
        action_index = (
            planned_action_index if attack_enabled else no_attack_index
        )

        response_active = (
            response_effective_turn is not None
            and step >= response_effective_turn
        )
        containment_probability = 0.0
        containment_controlled_turn = False
        learner_controlled_turn = True
        behavior_mixture_turn = False
        if fixed_response_policy is not None and response_active:
            containment_probability = _containment_probability(
                response_selection,
                step=step,
                response_alarm_turn=response_alarm_turn,
            )
            if _uses_behavior_conditioned_updates(response_selection):
                behavior_policy = _current_response_behavior_policy(
                    alice.policy(),
                    fixed_response_policy,
                    containment_probability=containment_probability,
                    response_active=True,
                )
                route_index = alice.sample_from_behavior(behavior_policy)
                behavior_mixture_turn = True
            else:
                containment_controlled_turn = (
                    containment_probability >= 1.0
                    or response_rng.random() < containment_probability
                )
                if containment_controlled_turn:
                    route_index = int(
                        response_rng.choice(
                            len(payoff_bank.route_ids),
                            p=fixed_response_policy,
                        )
                    )
                    learner_controlled_turn = False
                else:
                    route_index = alice.sample()
        else:
            route_index = alice.sample()
        draw_index = (
            draw_start + stream.next_draw(route_index, action_index)
        )
        observation = execution_block.draw(
            route_index,
            action_index,
            draw_index,
            requested_pair_count=evidence_bank.requested_pairs_per_draw,
        )
        reward = float(observation["alice_reward"])
        if learner_controlled_turn:
            alice.update(reward)

        if frozen_alarm_record is None:
            detector_rows = detectors.update(
                route_index,
                observation["endpoint_evidence"],
            )
            active_detector, active_row = _active_detector_and_row(
                detectors,
                detector_rows,
                response_config.active_detector,
            )
        else:
            active_detector = None
            active_row = _frozen_alarm_row(
                frozen_alarm_record,
                step=step,
                active_detector=response_config.active_detector,
            )
        if active_row["newly_alarmed"]:
            response_alarm_turn = step
            response_effective_turn = (
                step + 1 if step < schedule.total_turns else None
            )
            live_policy = alice.policy().copy()
            if active_detector is None:
                detector_diagnostics = frozen_alarm_record.get(
                    "detector_diagnostics_at_alarm"
                )
                if not isinstance(detector_diagnostics, Mapping):
                    raise ValueError(
                        "frozen alarm record lacks detector diagnostics"
                    )
            else:
                detector_diagnostics = active_detector.diagnostics()
            fixed_response_policy, response_selection = _select_response(
                response_config,
                payoff_bank,
                evidence_bank,
                live_policy,
                active_row["leading_alternative"],
                detector_diagnostics,
                alice.metadata,
            )
            if (
                response_config.strategy
                == RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION
            ):
                initial_posterior = CommissionedInterventionPosterior(
                    **response_selection[
                        "commissioned_intervention_posterior"
                    ]
                )
                sequential_posterior = SequentialInterventionPosterior(
                    evidence_bank,
                    initial_posterior,
                    window_turns=response_config.posterior_window_turns,
                    pseudocount=response_config.posterior_pseudocount,
                    feedback_draw_start=(
                        response_config.posterior_model_draw_start
                    ),
                    feedback_draw_stop=(
                        response_config.posterior_model_draw_stop
                    ),
                )
                response_selection["posterior_history"].append({
                    "turn": step,
                    "reason": "alarm_initialization",
                    "posterior_attack_mass": response_selection[
                        "posterior_attack_mass"
                    ],
                    "posterior_entropy_nats": (
                        initial_posterior.posterior_entropy_nats
                    ),
                    "target_policy_total_variation": (
                        response_selection[
                            "posterior_trust_region_policy"
                        ]["total_variation_from_live_policy"]
                    ),
                })

        if response_active and sequential_posterior is not None:
            updated_posterior = sequential_posterior.update(
                route_index,
                observation["endpoint_evidence"],
            )
            post_alarm_turn = step - int(response_alarm_turn)
            if (
                post_alarm_turn
                % int(response_config.posterior_refresh_turns)
                == 0
            ):
                refreshed = solve_posterior_trust_region_route_policy(
                    payoff_bank,
                    alice.policy(),
                    updated_posterior,
                    switching_penalty=response_config.switching_penalty,
                    maximum_total_variation=float(
                        response_selection["staged_containment"][
                            "maximum_total_variation"
                        ]
                    ),
                    feedback_draw_start=(
                        response_config.posterior_model_draw_start
                    ),
                    feedback_draw_stop=(
                        response_config.posterior_model_draw_stop
                    ),
                )
                fixed_response_policy = np.asarray(
                    refreshed.probabilities,
                    dtype=float,
                )
                no_attack = updated_posterior.action_ids.index(
                    "no_attack"
                )
                response_selection[
                    "commissioned_intervention_posterior"
                ] = updated_posterior.to_dict()
                response_selection[
                    "posterior_trust_region_policy"
                ] = refreshed.to_dict()
                response_selection["fixed_policy"] = (
                    fixed_response_policy.tolist()
                )
                response_selection["posterior_attack_mass"] = float(
                    1.0 - updated_posterior.probabilities[no_attack]
                )
                response_policy_refreshes += 1
                if (
                    response_policy_refreshes == 1
                    or step == schedule.total_turns
                    or step % response_config.record_every == 0
                    or step == schedule.prime_turns
                    or step == schedule.prime_turns + 1
                ):
                    response_selection["posterior_history"].append({
                        "turn": step,
                        "reason": "sequential_refresh",
                        "posterior_attack_mass": response_selection[
                            "posterior_attack_mass"
                        ],
                        "posterior_entropy_nats": (
                            updated_posterior.posterior_entropy_nats
                        ),
                        "target_policy_total_variation": (
                            refreshed.total_variation_from_live_policy
                        ),
                    })

        signature = observation_design.signature_for_route(route_index)
        observed_routes = observation_design.routes_for_signature(signature)
        if len(observed_routes) != 1:
            raise AssertionError(
                "declared exact reservation signature became ambiguous"
            )
        observed_target = int(int(observed_routes[0]) in target_set)
        if phase == "prime":
            passive_target_observations.append(observed_target)

        reward_sum += reward
        phase_reward_sums[phase] += reward
        phase_counts[phase] += 1
        action_counts[action_index] += 1
        route_counts[route_index] += 1
        if response_active:
            response_reward_sum += reward
            response_reward_count += 1
            response_route_counts[route_index] += 1
            response_learner_controlled_turns += int(
                learner_controlled_turn
            )
            response_containment_controlled_turns += int(
                containment_controlled_turn
            )
            response_behavior_mixture_turns += int(
                behavior_mixture_turn
            )
            response_containment_probability_sum += float(
                containment_probability
            )

        if (
            step == 1
            or step == schedule.total_turns
            or step % response_config.record_every == 0
            or bool(active_row["newly_alarmed"])
        ):
            current_behavior_policy = (
                _current_response_behavior_policy(
                    alice.policy(),
                    fixed_response_policy,
                    containment_probability=containment_probability,
                    response_active=response_active,
                )
            )
            trajectory.append({
                "turn": step,
                "phase": phase,
                "active_detector_log_e_value": float(
                    active_row["log_e_value"]
                ) if active_row["log_e_value"] is not None else None,
                "active_detector_alarmed": bool(
                    active_row["alarmed"]
                ),
                "response_active": response_active,
                "containment_probability": containment_probability,
                "containment_controlled_turn": containment_controlled_turn,
                "learner_controlled_turn": learner_controlled_turn,
                "behavior_mixture_turn": behavior_mixture_turn,
                "mean_reward": reward_sum / step,
                "target_policy_mass": float(
                    np.sum(current_behavior_policy[list(target_set)])
                ),
            })

    if trigger is None:
        trigger = passive_exploit_trigger(
            passive_target_observations,
            threshold=plan.profitable_target_mass_threshold,
            z=schedule.trigger_confidence_z,
        )
    final_containment_probability = (
        _containment_probability(
            response_selection,
            step=schedule.total_turns,
            response_alarm_turn=response_alarm_turn,
        )
        if response_reward_count > 0
        else 0.0
    )
    final_policy = _current_response_behavior_policy(
        alice.policy(),
        fixed_response_policy,
        containment_probability=final_containment_probability,
        response_active=response_reward_count > 0,
    )
    if frozen_alarm_record is None:
        active_detector, _ = _active_detector_and_row(
            detectors,
            None,
            response_config.active_detector,
        )
        active_detector_diagnostics = active_detector.diagnostics()
        all_detector_diagnostics = detectors.diagnostics()
    else:
        active_detector_diagnostics = dict(frozen_alarm_record)
        all_detector_diagnostics = {
            "schema": "frozen_endpoint_alarm_record.v1",
            "active_detector": response_config.active_detector,
            "record": dict(frozen_alarm_record),
            "replayed_without_post_alarm_detector_updates": True,
        }
    audit_response = _audit_response_diagnostics(
        payoff_bank,
        fixed_response_policy,
        response_selection,
    )
    return {
        "schema": ENDPOINT_RESPONSE_RUN_SCHEMA,
        "condition": str(condition),
        "attack_enabled": bool(attack_enabled),
        "seed": int(seed),
        "case_id": payoff_bank.case_id,
        "turns": schedule.total_turns,
        "schedule": schedule.to_dict(),
        "plan": plan.to_dict(),
        "response_config": response_config.to_dict(),
        "alice_learner": dict(alice.metadata),
        "active_detector": active_detector_diagnostics,
        "all_detectors": all_detector_diagnostics,
        "response_alarm_turn": response_alarm_turn,
        "response_effective_turn": response_effective_turn,
        "response_selection": response_selection,
        "response_audit_diagnostics_for_scoring_only": audit_response,
        "mean_alice_reward": reward_sum / schedule.total_turns,
        "mean_response_window_reward": (
            response_reward_sum / response_reward_count
            if response_reward_count
            else None
        ),
        "response_window_turns": response_reward_count,
        "response_control": {
            "learner_controlled_turns": (
                response_learner_controlled_turns
            ),
            "containment_controlled_turns": (
                response_containment_controlled_turns
            ),
            "behavior_mixture_turns": (
                response_behavior_mixture_turns
            ),
            "mean_containment_probability": (
                response_containment_probability_sum
                / response_reward_count
                if response_reward_count
                else 0.0
            ),
            "final_containment_probability": (
                final_containment_probability
            ),
            "response_policy_refreshes": response_policy_refreshes,
            "sequential_posterior": (
                sequential_posterior.diagnostics()
                if sequential_posterior is not None
                else None
            ),
        },
        "phase_metrics": {
            phase: {
                "turns": int(phase_counts[phase]),
                "mean_alice_reward": (
                    phase_reward_sums[phase] / phase_counts[phase]
                ),
            }
            for phase in ("prime", "exploit")
        },
        "passive_exploit_trigger": trigger,
        "action_multiset": {
            payoff_bank.action_ids[index]: int(count)
            for index, count in sorted(action_counts.items())
        },
        "route_counts": route_counts.tolist(),
        "response_route_counts": response_route_counts.tolist(),
        "final_behavior_policy": final_policy.tolist(),
        "final_behavior_policy_statistics": (
            policy_distribution_statistics(final_policy)
        ),
        "trajectory": trajectory,
        "state_continuity": {
            "starting_alice_sample_calls": starting_sample_calls,
            "starting_alice_update_calls": starting_update_calls,
            "ending_alice_sample_calls": int(alice.sample_calls),
            "ending_alice_update_calls": int(alice.update_calls),
            "exact_post_follow_state_continued_until_response": True,
            "learner_reinitialized": False,
            "fixed_response_stops_learner_updates": (
                fixed_response_policy is not None
                and not _is_staged_response(response_selection)
            ),
            "staged_response_preserves_native_updates_on_native_turns": (
                _is_staged_response(response_selection)
            ),
            "staged_response_updates_every_behavior_turn": (
                response_behavior_mixture_turns
                == response_reward_count
                and response_reward_count > 0
            ),
            "response_behavior_mixture_turns": (
                response_behavior_mixture_turns
            ),
            "response_learner_controlled_turns": (
                response_learner_controlled_turns
            ),
            "response_containment_controlled_turns": (
                response_containment_controlled_turns
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "execution_split": str(execution_split),
        "execution_draw_start": draw_start,
        "execution_draw_stop": draw_stop,
        "information_boundary": {
            "online_detector": (
                "selected route and endpoint counts declared by the active "
                "classical-service or quantum endpoint detector"
            ),
            "response_selection": (
                "exact live Alice policy, detector-leading route/channel/"
                "direction, single-channel intervention posterior, and "
                "disjoint feedback-bank commissioned model"
            ),
            "excluded_online": [
                "realized Eve action identity",
                "realized Eve target identity",
                "target-active simulator flag",
                "audit payoff matrix",
                "counterfactual route rewards",
            ],
            "learner_specific_adaptation": {
                "tsallis_inf": (
                    "native continuation retains independent route-arm "
                    "statistics; the declared behavior-IW response samples "
                    "the controller mixture and updates every selected arm "
                    "with its actual behavior probability"
                ),
                "linear_exp2": (
                    "native continuation retains route-resource features and "
                    "G-optimal structured exploration"
                ),
            },
        },
        "claim_boundary": (
            "The fixed robust response is optimal only for its declared "
            "feedback-bank maximin objective. The posterior trust-region "
            "response is optimal only for its single-leading-channel "
            "commissioned posterior objective and declared TV constraint. "
            "Learner-aware arming schedules are experimental controls, not "
            "empirical optima. The Tsallis behavior-IW extension preserves an "
            "unbiased loss-vector estimate under the declared mixture but is "
            "outside the native Tsallis-INF regret theorem. Held-out audit "
            "values score decisions but are not inputs."
        ),
    }


def _is_staged_response(
    response_selection: Mapping[str, Any] | None,
) -> bool:
    return bool(
        response_selection is not None
        and response_selection.get("strategy") in {
            RESPONSE_POSTERIOR_TRUST_REGION,
            RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
        }
    )


def _uses_behavior_conditioned_updates(
    response_selection: Mapping[str, Any] | None,
) -> bool:
    if not _is_staged_response(response_selection):
        return False
    assert response_selection is not None
    staged = response_selection.get("staged_containment")
    return bool(
        isinstance(staged, Mapping)
        and staged.get("learner_update_mode")
        == RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW
    )


def _containment_probability(
    response_selection: Mapping[str, Any] | None,
    *,
    step: int,
    response_alarm_turn: int | None,
) -> float:
    if response_selection is None or response_alarm_turn is None:
        return 0.0
    post_alarm_turn = int(step) - int(response_alarm_turn)
    if post_alarm_turn <= 0:
        return 0.0
    if not _is_staged_response(response_selection):
        return 1.0
    staged = response_selection.get("staged_containment")
    if not isinstance(staged, Mapping):
        raise ValueError("staged response lacks its arming profile")
    arming_turns = int(staged["arming_turns"])
    ceiling = float(staged["containment_ceiling"])
    if arming_turns < 1:
        raise ValueError("staged response arming_turns must be positive")
    if not math.isfinite(ceiling) or not 0.0 <= ceiling <= 1.0:
        raise ValueError("staged response containment ceiling is invalid")
    attack_mass = (
        float(response_selection.get("posterior_attack_mass", 1.0))
        if response_selection.get("strategy")
        == RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION
        else 1.0
    )
    if not math.isfinite(attack_mass) or not 0.0 <= attack_mass <= 1.0:
        raise ValueError("sequential posterior attack mass is invalid")
    return float(
        ceiling
        * attack_mass
        * min(1.0, post_alarm_turn / arming_turns)
    )


def _current_response_behavior_policy(
    live_policy: Sequence[float] | np.ndarray,
    containment_policy: Sequence[float] | np.ndarray | None,
    *,
    containment_probability: float,
    response_active: bool,
) -> np.ndarray:
    live = np.asarray(live_policy, dtype=float)
    if containment_policy is None or not response_active:
        return live.copy()
    containment = np.asarray(containment_policy, dtype=float)
    probability = float(containment_probability)
    if (
        containment.shape != live.shape
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("invalid staged response behavior mixture")
    mixed = (1.0 - probability) * live + probability * containment
    mixed /= np.sum(mixed)
    return mixed


def _select_response(
    config: AlarmResponseConfig,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    live_policy: np.ndarray,
    leading: Mapping[str, Any],
    detector_diagnostics: Mapping[str, Any],
    alice_metadata: Mapping[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if config.strategy == RESPONSE_NATIVE_CONTINUATION:
        return None, {
            "strategy": config.strategy,
            "live_policy_at_alarm": live_policy.tolist(),
            "semantics": (
                "continue the exact learner; Tsallis-INF adapts per route arm "
                "and LinearExp2 shares feedback through topology features"
            ),
        }
    if config.strategy == RESPONSE_FREEZE_LIVE_POLICY:
        return live_policy.copy(), {
            "strategy": config.strategy,
            "live_policy_at_alarm": live_policy.tolist(),
            "fixed_policy": live_policy.tolist(),
            "total_variation_from_live_policy": 0.0,
        }
    if config.strategy == RESPONSE_QUARANTINE_LEADING_ROUTE:
        route = int(leading["route_index"])
        fixed = live_policy.copy()
        fixed[route] = 0.0
        if float(np.sum(fixed)) <= 0.0:
            raise RuntimeError("leading-route quarantine removed all policy mass")
        fixed /= np.sum(fixed)
        return fixed, {
            "strategy": config.strategy,
            "live_policy_at_alarm": live_policy.tolist(),
            "fixed_policy": fixed.tolist(),
            "quarantined_route_index": route,
            "quarantined_route_id": payoff_bank.route_ids[route],
            "total_variation_from_live_policy": (
                0.5 * float(np.sum(np.abs(fixed - live_policy)))
            ),
        }

    if config.strategy in {
        RESPONSE_POSTERIOR_TRUST_REGION,
        RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
    }:
        profile = _learner_aware_response_profile(
            config,
            alice_metadata,
        )
        posterior = commissioned_intervention_posterior(
            evidence_bank,
            detector_diagnostics,
            leading,
            effect_floor=config.compatibility_effect_floor,
            pseudocount=config.posterior_pseudocount,
            feedback_draw_start=config.posterior_model_draw_start,
            feedback_draw_stop=config.posterior_model_draw_stop,
        )
        response = solve_posterior_trust_region_route_policy(
            payoff_bank,
            live_policy,
            posterior,
            switching_penalty=config.switching_penalty,
            maximum_total_variation=float(
                profile["maximum_total_variation"]
            ),
            feedback_draw_start=config.posterior_model_draw_start,
            feedback_draw_stop=config.posterior_model_draw_stop,
        )
        fixed = np.asarray(response.probabilities, dtype=float)
        selection = {
            "strategy": config.strategy,
            "live_policy_at_alarm": live_policy.tolist(),
            "fixed_policy": fixed.tolist(),
            "commissioned_intervention_posterior": posterior.to_dict(),
            "posterior_trust_region_policy": response.to_dict(),
            "staged_containment": profile,
        }
        if config.strategy == RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION:
            no_attack = posterior.action_ids.index("no_attack")
            selection["posterior_attack_mass"] = float(
                1.0 - posterior.probabilities[no_attack]
            )
            selection["sequential_posterior_config"] = {
                "window_turns": config.posterior_window_turns,
                "refresh_turns": config.posterior_refresh_turns,
                "feedback_model_draw_start": (
                    response.feedback_draw_start
                ),
                "feedback_model_draw_stop": response.feedback_draw_stop,
                "attack_mass_gates_containment": True,
            }
            selection["posterior_history"] = []
        return fixed, selection

    candidates = commissioned_detector_compatible_actions(
        evidence_bank,
        channel=str(leading["channel"]),
        direction=str(leading["direction"]),
        effect_floor=config.compatibility_effect_floor,
    )
    robust = solve_commissioned_robust_route_policy(
        payoff_bank,
        live_policy,
        candidates,
        switching_penalty=config.switching_penalty,
        leading_channel=str(leading["channel"]),
        leading_direction=str(leading["direction"]),
    )
    fixed = np.asarray(robust.probabilities, dtype=float)
    return fixed, {
        "strategy": config.strategy,
        "live_policy_at_alarm": live_policy.tolist(),
        "fixed_policy": fixed.tolist(),
        "commissioned_robust_policy": robust.to_dict(),
    }


def _learner_aware_response_profile(
    config: AlarmResponseConfig,
    alice_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    topology_sharing = bool(
        alice_metadata.get("cross_route_information_sharing", False)
    )
    if topology_sharing:
        radius = float(config.topology_sharing_trust_region_tv)
        arming_turns = int(config.topology_sharing_arming_turns)
        containment_ceiling = float(
            config.topology_sharing_containment_ceiling
        )
        profile = "topology_sharing"
        learner_update_mode = RESPONSE_UPDATE_NATIVE_ONLY
    else:
        radius = float(config.independent_arm_trust_region_tv)
        arming_turns = int(config.independent_arm_arming_turns)
        containment_ceiling = float(
            config.independent_arm_containment_ceiling
        )
        profile = "independent_route_arms"
        learner_update_mode = config.independent_arm_update_mode
    return {
        "profile": profile,
        "learner_mode": str(alice_metadata.get("mode") or "unknown"),
        "cross_route_information_sharing": topology_sharing,
        "maximum_total_variation": radius,
        "arming_turns": arming_turns,
        "containment_ceiling": containment_ceiling,
        "learner_update_mode": learner_update_mode,
        "arming_schedule": (
            "containment probability containment_ceiling * min(1, "
            "post_alarm_turn/arming_turns)"
        ),
        "learner_update_semantics": (
            "sample the full native/containment behavior mixture and divide "
            "the selected Tsallis loss correction by its actual behavior "
            "probability on every response turn"
            if learner_update_mode == RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW
            else (
                "sample and update the learner only on native-policy turns; "
                "fixed-policy containment turns do not update the learner"
            )
        ),
        "selection_status": (
            "declared learner-aware experimental controller; not an "
            "empirical optimum"
        ),
    }


def _active_detector_and_row(
    detectors: MatchedClassicalQuantumEndpointDetectors,
    rows: dict[str, Any] | None,
    active_detector: str,
) -> tuple[RouteConditionedBinomialEProcess, dict[str, Any] | None]:
    if active_detector == DETECTOR_CLASSICAL_SERVICE:
        return (
            detectors.classical,
            None if rows is None else rows["classical_service_baseline"],
        )
    if active_detector == DETECTOR_QUANTUM_ENDPOINT:
        return (
            detectors.quantum,
            None if rows is None else rows["quantum_endpoint_integrity"],
        )
    raise ValueError(f"unsupported active detector {active_detector!r}")


def _frozen_alarm_row(
    record: Mapping[str, Any],
    *,
    step: int,
    active_detector: str,
) -> dict[str, Any]:
    if str(record.get("active_detector")) != active_detector:
        raise ValueError("frozen alarm record uses a different detector")
    alarm_turn_raw = record.get("alarm_turn")
    alarm_turn = None if alarm_turn_raw is None else int(alarm_turn_raw)
    if alarm_turn is not None and alarm_turn < 1:
        raise ValueError("frozen alarm turn must be positive")
    leading = record.get("leading_alternative")
    if alarm_turn is not None and not isinstance(leading, Mapping):
        raise ValueError(
            "alarmed frozen record requires a leading alternative"
        )
    alarmed = alarm_turn is not None and step >= alarm_turn
    return {
        "turn": int(step),
        "log_e_value": (
            float(record["alarm_log_e_value"])
            if step == alarm_turn
            and record.get("alarm_log_e_value") is not None
            else None
        ),
        "alarmed": alarmed,
        "newly_alarmed": bool(alarm_turn is not None and step == alarm_turn),
        "alarm_turn": alarm_turn,
        "leading_alternative": (
            dict(leading) if isinstance(leading, Mapping) else None
        ),
    }


def _feedback_channel_matrix(
    bank: HistogramEndpointEvidenceBank,
    channel: str,
) -> np.ndarray:
    block = bank.feedback
    if channel == PHYSICAL_DELIVERY_CHANNEL:
        return np.mean(
            block.delivered_pair_count / bank.requested_pairs_per_draw,
            axis=2,
        )
    if channel == PUBLIC_SERVICE_ACCEPTANCE_CHANNEL:
        return np.mean(
            block.public_outcome_code == PUBLIC_OUTCOME_CODES["accepted"],
            axis=2,
        )
    if channel == KEY_ERROR_CHANNEL:
        successes = np.sum(block.key_error_count, axis=2, dtype=float)
        trials = np.sum(block.key_pair_count, axis=2, dtype=float)
        return np.divide(
            successes,
            trials,
            out=np.zeros_like(successes),
            where=trials > 0.0,
        )
    if channel.startswith(CHSH_SAME_CHANNEL_PREFIX):
        label = channel[len(CHSH_SAME_CHANNEL_PREFIX):]
        try:
            index = CHSH_SETTING_LABELS.index(label)
        except ValueError as exc:
            raise ValueError(f"unknown CHSH channel {channel!r}") from exc
        successes = np.sum(
            block.chsh_setting_same_counts[:, :, :, index],
            axis=2,
            dtype=float,
        )
        trials = np.sum(
            block.chsh_setting_total_counts[:, :, :, index],
            axis=2,
            dtype=float,
        )
        return np.divide(
            successes,
            trials,
            out=np.zeros_like(successes),
            where=trials > 0.0,
        )
    raise ValueError(f"unsupported endpoint channel {channel!r}")


def _feedback_channel_counts(
    bank: HistogramEndpointEvidenceBank,
    channel: str,
    *,
    draw_start: int = 0,
    draw_stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate feedback-bank binomial counts by route and intervention."""

    block = bank.feedback
    start, stop = _draw_bounds(
        bank.draws_per_cell,
        draw_start,
        draw_stop,
        label="endpoint feedback counts",
    )
    route_count, action_count, _ = block.alice_reward.shape
    draw_count = stop - start
    shape = (route_count, action_count)
    if channel == PHYSICAL_DELIVERY_CHANNEL:
        successes = np.sum(
            block.delivered_pair_count[:, :, start:stop],
            axis=2,
            dtype=np.int64,
        )
        trials = np.full(
            shape,
            draw_count * bank.requested_pairs_per_draw,
            dtype=np.int64,
        )
        return successes, trials
    if channel == PUBLIC_SERVICE_ACCEPTANCE_CHANNEL:
        successes = np.sum(
            block.public_outcome_code[:, :, start:stop]
            == PUBLIC_OUTCOME_CODES["accepted"],
            axis=2,
            dtype=np.int64,
        )
        trials = np.full(shape, draw_count, dtype=np.int64)
        return successes, trials
    if channel == KEY_ERROR_CHANNEL:
        return (
            np.sum(
                block.key_error_count[:, :, start:stop],
                axis=2,
                dtype=np.int64,
            ),
            np.sum(
                block.key_pair_count[:, :, start:stop],
                axis=2,
                dtype=np.int64,
            ),
        )
    if channel.startswith(CHSH_SAME_CHANNEL_PREFIX):
        label = channel[len(CHSH_SAME_CHANNEL_PREFIX):]
        try:
            index = CHSH_SETTING_LABELS.index(label)
        except ValueError as exc:
            raise ValueError(f"unknown CHSH channel {channel!r}") from exc
        return (
            np.sum(
                block.chsh_setting_same_counts[
                    :, :, start:stop, index
                ],
                axis=2,
                dtype=np.int64,
            ),
            np.sum(
                block.chsh_setting_total_counts[
                    :, :, start:stop, index
                ],
                axis=2,
                dtype=np.int64,
            ),
        )
    raise ValueError(f"unsupported endpoint channel {channel!r}")


def _draw_bounds(
    draws_per_cell: int,
    draw_start: int,
    draw_stop: int | None,
    *,
    label: str,
) -> tuple[int, int]:
    start = int(draw_start)
    stop = (
        int(draws_per_cell)
        if draw_stop is None
        else int(draw_stop)
    )
    if not 0 <= start < stop <= int(draws_per_cell):
        raise ValueError(
            f"{label} draw range [{start}, {stop}) is outside "
            f"[0, {int(draws_per_cell)})"
        )
    return start, stop


def _endpoint_execution_block(
    bank: HistogramEndpointEvidenceBank,
    split: str,
) -> Any:
    name = str(split)
    if name not in {"feedback", "audit"}:
        raise ValueError(
            "endpoint response execution_split must be feedback or audit"
        )
    return getattr(bank, name)


def _validate_frozen_alarm_execution(
    record: Mapping[str, Any],
    *,
    execution_split: str,
    draw_start: int,
    draw_stop: int,
) -> None:
    recorded_split = record.get("execution_split")
    if (
        recorded_split is not None
        and str(recorded_split) != str(execution_split)
    ):
        raise ValueError(
            "frozen alarm record uses a different execution split"
        )
    for key, expected in (
        ("execution_draw_start", draw_start),
        ("execution_draw_stop", draw_stop),
    ):
        value = record.get(key)
        if value is not None and int(value) != int(expected):
            raise ValueError(
                "frozen alarm record uses a different execution draw range"
            )


def _audit_response_diagnostics(
    bank: HistogramPayoffBank,
    fixed_policy: np.ndarray | None,
    selection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if fixed_policy is None or selection is None:
        return None
    robust = selection.get("commissioned_robust_policy")
    posterior_response = selection.get("posterior_trust_region_policy")
    if isinstance(posterior_response, Mapping):
        posterior = posterior_response["posterior"]
        candidates = tuple(map(int, posterior["action_indices"]))
        probabilities = _probability_vector(
            posterior["probabilities"],
            expected_size=len(candidates),
            label="recorded intervention posterior",
        )
        values = (
            fixed_policy
            @ bank.audit.alice_payoff_matrix()[:, candidates]
        )
        return {
            "audit_use": "scoring_only",
            "candidate_action_indices": list(candidates),
            "candidate_action_ids": [
                bank.action_ids[index] for index in candidates
            ],
            "audit_reward_by_candidate_action": values.tolist(),
            "audit_posterior_expected_reward": float(
                values @ probabilities
            ),
            "audit_worst_case_reward": float(np.min(values)),
        }
    if robust is None:
        return {
            "audit_use": "scoring_only",
            "audit_mean_reward_by_action": (
                fixed_policy @ bank.audit.alice_payoff_matrix()
            ).tolist(),
        }
    candidates = tuple(map(int, robust["candidate_action_indices"]))
    values = (
        fixed_policy
        @ bank.audit.alice_payoff_matrix()[:, candidates]
    )
    return {
        "audit_use": "scoring_only",
        "candidate_action_indices": list(candidates),
        "candidate_action_ids": [
            bank.action_ids[index] for index in candidates
        ],
        "audit_reward_by_candidate_action": values.tolist(),
        "audit_worst_case_reward": float(np.min(values)),
    }


def _validate_response_inputs(
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    plan: ControllerPrimeExploitPlan,
    alice: PlayerLearner,
    observation_design: RouteObservationDesign,
) -> None:
    if (
        payoff_bank.route_ids != evidence_bank.route_ids
        or payoff_bank.action_ids != evidence_bank.action_ids
    ):
        raise ValueError("payoff and endpoint evidence bank axes differ")
    if alice.player != "alice" or alice.action_ids != payoff_bank.route_ids:
        raise ValueError("continuation Alice does not match the bank")
    if alice.sample_calls != alice.update_calls:
        raise RuntimeError(
            "continuation Alice has an outstanding selected action"
        )
    if observation_design.route_ids != payoff_bank.route_ids:
        raise ValueError("observation design and bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError(
            "passive controller execution requires exact route separation"
        )
    action_count = len(payoff_bank.action_ids)
    indices = (
        int(plan.exploit_action_index),
        *map(int, plan.prime_action_indices),
    )
    if any(index < 0 or index >= action_count for index in indices):
        raise ValueError("controller plan action index is outside the bank")
    if payoff_bank.action_ids[plan.exploit_action_index] != plan.exploit_action_id:
        raise ValueError("controller exploit action identity changed")


def _probability_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int,
    label: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if (
        vector.shape != (expected_size,)
        or not np.all(np.isfinite(vector))
        or np.any(vector < -1e-12)
        or not math.isclose(
            float(np.sum(vector)),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"{label} must lie on the declared simplex")
    clipped = np.maximum(vector, 0.0)
    clipped /= np.sum(clipped)
    return clipped
