"""Matched endpoint observation sets for Stackelberg change detection.

The delivery arm observes physical delivery counts.  The service arm adds the
public accept/reject result.  The full-endpoint arm additionally observes
finite key-error and per-setting CHSH same/different counts.  A separate E91
measurement-count arm receives only the numbers and outcomes of key-basis and
CHSH-setting measurements recorded in the endpoint transcript.  No arm
observes Eve's action, target activity, expected simulator probabilities, or
other private diagnostics.

Each detector is a route-conditioned mixture e-process.  A component bets on
one binomial evidence channel, route, change direction, change magnitude, and
change time.  Evidence channels are mixed rather than multiplied, so the
detector does not assume independence between delivery, key, and CHSH counts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.special import betaln

from histograms.models import ProtocolConfig
from sequence_game.corpus.compiled_histogram_backend import (
    CompiledHistogramStepResult,
)


PHYSICAL_DELIVERY_CHANNEL = "physical_delivery"
PUBLIC_SERVICE_ACCEPTANCE_CHANNEL = "public_service_acceptance"
KEY_ERROR_CHANNEL = "key_error"
KEY_PAIR_COUNT_CHANNEL = "key_pair_count_per_request"
CHSH_SETTING_LABELS = tuple(
    f"{alice}|{bob}"
    for alice, bob in ProtocolConfig(requested_pairs=1).chsh_pairs
)
CHSH_SAME_CHANNEL_PREFIX = "chsh_same::"
CHSH_TRIAL_COUNT_CHANNEL_PREFIX = "chsh_trial_count_per_request::"
CLASSICAL_DELIVERY_CHANNELS = (PHYSICAL_DELIVERY_CHANNEL,)
CLASSICAL_SERVICE_CHANNELS = (
    PHYSICAL_DELIVERY_CHANNEL,
    PUBLIC_SERVICE_ACCEPTANCE_CHANNEL,
)
QUANTUM_ENDPOINT_CHANNELS = (
    PHYSICAL_DELIVERY_CHANNEL,
    PUBLIC_SERVICE_ACCEPTANCE_CHANNEL,
    KEY_ERROR_CHANNEL,
    *tuple(
        f"{CHSH_SAME_CHANNEL_PREFIX}{label}"
        for label in CHSH_SETTING_LABELS
    ),
)
E91_MEASUREMENT_COUNT_CHANNELS = (
    KEY_PAIR_COUNT_CHANNEL,
    *tuple(
        f"{CHSH_TRIAL_COUNT_CHANNEL_PREFIX}{label}"
        for label in CHSH_SETTING_LABELS
    ),
    KEY_ERROR_CHANNEL,
    *tuple(
        f"{CHSH_SAME_CHANNEL_PREFIX}{label}"
        for label in CHSH_SETTING_LABELS
    ),
)


@dataclass(frozen=True)
class QuantumEndpointEvidence:
    """Finite endpoint-visible counts from one compiled E91/BBM92 request."""

    requested_pair_count: int
    delivered_pair_count: int
    key_pair_count: int
    key_error_count: int
    public_service_accepted: bool
    chsh_setting_total_counts: tuple[int, ...]
    chsh_setting_same_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        integer_fields = {
            "requested_pair_count": self.requested_pair_count,
            "delivered_pair_count": self.delivered_pair_count,
            "key_pair_count": self.key_pair_count,
            "key_error_count": self.key_error_count,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.delivered_pair_count > self.requested_pair_count:
            raise ValueError(
                "delivered_pair_count cannot exceed requested_pair_count"
            )
        if self.key_pair_count > self.delivered_pair_count:
            raise ValueError(
                "key_pair_count cannot exceed delivered_pair_count"
            )
        if self.key_error_count > self.key_pair_count:
            raise ValueError("key_error_count cannot exceed key_pair_count")
        if not isinstance(self.public_service_accepted, bool):
            raise ValueError("public_service_accepted must be boolean")
        if (
            len(self.chsh_setting_total_counts) != len(CHSH_SETTING_LABELS)
            or len(self.chsh_setting_same_counts) != len(CHSH_SETTING_LABELS)
        ):
            raise ValueError(
                "CHSH count tuples must follow the configured four-setting "
                "protocol order"
            )
        for total, same in zip(
            self.chsh_setting_total_counts,
            self.chsh_setting_same_counts,
        ):
            if (
                isinstance(total, bool)
                or isinstance(same, bool)
                or int(total) != total
                or int(same) != same
                or total < 0
                or same < 0
                or same > total
            ):
                raise ValueError(
                    "each CHSH same-outcome count must be an integer in "
                    "[0, setting total]"
                )
        chsh_pair_count = sum(self.chsh_setting_total_counts)
        if self.key_pair_count + chsh_pair_count > self.delivered_pair_count:
            raise ValueError(
                "key and CHSH trial counts cannot exceed delivered pairs"
            )

    @classmethod
    def from_step_result(
        cls,
        result: CompiledHistogramStepResult,
    ) -> "QuantumEndpointEvidence":
        """Extract only Alice/Bob endpoint evidence from a compiled step."""

        return cls(
            requested_pair_count=int(result.requested_pair_count),
            delivered_pair_count=int(result.delivered_pair_count),
            key_pair_count=int(result.key_pair_count),
            key_error_count=int(result.key_error_count),
            public_service_accepted=(
                str(result.public_outcome) == "accepted"
            ),
            chsh_setting_total_counts=tuple(
                int(result.chsh_setting_total_counts.get(label, 0))
                for label in CHSH_SETTING_LABELS
            ),
            chsh_setting_same_counts=tuple(
                int(result.chsh_setting_same_counts.get(label, 0))
                for label in CHSH_SETTING_LABELS
            ),
        )

    @property
    def qber(self) -> float | None:
        if self.key_pair_count == 0:
            return None
        return self.key_error_count / self.key_pair_count

    @property
    def chsh_correlations(self) -> dict[str, float | None]:
        correlations: dict[str, float | None] = {}
        for label, total, same in zip(
            CHSH_SETTING_LABELS,
            self.chsh_setting_total_counts,
            self.chsh_setting_same_counts,
        ):
            correlations[label] = (
                (2.0 * same - total) / total
                if total > 0
                else None
            )
        return correlations

    @property
    def chsh_s(self) -> float | None:
        correlations = tuple(self.chsh_correlations.values())
        if any(value is None for value in correlations):
            return None
        return -(
            float(correlations[0])
            + float(correlations[1])
            + float(correlations[2])
            - float(correlations[3])
        )

    def binomial_counts(
        self,
        channel_labels: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return successes and trials in the declared detector channel order."""

        successes: list[int] = []
        trials: list[int] = []
        chsh_index = {
            f"{CHSH_SAME_CHANNEL_PREFIX}{label}": index
            for index, label in enumerate(CHSH_SETTING_LABELS)
        }
        chsh_trial_index = {
            f"{CHSH_TRIAL_COUNT_CHANNEL_PREFIX}{label}": index
            for index, label in enumerate(CHSH_SETTING_LABELS)
        }
        for label in channel_labels:
            if label == PHYSICAL_DELIVERY_CHANNEL:
                successes.append(self.delivered_pair_count)
                trials.append(self.requested_pair_count)
            elif label == KEY_PAIR_COUNT_CHANNEL:
                successes.append(self.key_pair_count)
                trials.append(self.requested_pair_count)
            elif label == KEY_ERROR_CHANNEL:
                successes.append(self.key_error_count)
                trials.append(self.key_pair_count)
            elif label == PUBLIC_SERVICE_ACCEPTANCE_CHANNEL:
                successes.append(int(self.public_service_accepted))
                trials.append(1)
            elif label in chsh_trial_index:
                index = chsh_trial_index[label]
                successes.append(self.chsh_setting_total_counts[index])
                trials.append(self.requested_pair_count)
            elif label in chsh_index:
                index = chsh_index[label]
                successes.append(self.chsh_setting_same_counts[index])
                trials.append(self.chsh_setting_total_counts[index])
            else:
                raise ValueError(f"unsupported endpoint evidence channel: {label}")
        return (
            np.asarray(successes, dtype=np.int64),
            np.asarray(trials, dtype=np.int64),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pair_count": int(self.requested_pair_count),
            "delivered_pair_count": int(self.delivered_pair_count),
            "key_pair_count": int(self.key_pair_count),
            "key_error_count": int(self.key_error_count),
            "public_service_accepted": self.public_service_accepted,
            "qber": self.qber,
            "chsh_setting_total_counts": dict(
                zip(CHSH_SETTING_LABELS, self.chsh_setting_total_counts)
            ),
            "chsh_setting_same_counts": dict(
                zip(CHSH_SETTING_LABELS, self.chsh_setting_same_counts)
            ),
            "chsh_correlations": self.chsh_correlations,
            "chsh_s": self.chsh_s,
        }


@dataclass(frozen=True)
class BinomialEvidenceDetectorConfig:
    """Commissioning-relative configuration for count-valued evidence."""

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


class RouteConditionedBinomialEProcess:
    """Mixture e-process over route-conditioned binomial evidence channels."""

    def __init__(
        self,
        commissioning_successes: np.ndarray,
        commissioning_trials: np.ndarray,
        channel_labels: Sequence[str],
        config: BinomialEvidenceDetectorConfig,
    ) -> None:
        successes = _integer_count_array(
            commissioning_successes,
            "commissioning_successes",
        )
        trials = _integer_count_array(
            commissioning_trials,
            "commissioning_trials",
        )
        labels = tuple(str(label) for label in channel_labels)
        if (
            successes.ndim != 2
            or successes.shape != trials.shape
            or successes.shape[0] < 1
            or successes.shape[1] < 1
            or successes.shape[1] != len(labels)
            or np.any(successes < 0)
            or np.any(trials < 0)
            or np.any(successes > trials)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "commissioning successes/trials must be compatible "
                "nonnegative route x channel arrays"
            )
        if np.any(np.sum(trials, axis=1) == 0):
            raise ValueError(
                "commissioning evidence must contain a trial on every route"
            )

        pseudocount = float(config.commissioning_pseudocount)
        self.config = config
        self.channel_labels = labels
        self.commissioning_successes = successes.copy()
        self.commissioning_trials = trials.copy()
        self.null_probabilities = (
            successes.astype(float) + pseudocount
        ) / (trials.astype(float) + 2.0 * pseudocount)
        self._online_successes = np.zeros_like(successes)
        self._online_trials = np.zeros_like(trials)

        routes: list[int] = []
        channels: list[int] = []
        probabilities: list[float] = []
        alternative_labels: list[tuple[int, int, str, float]] = []
        for route_index in range(successes.shape[0]):
            for channel_index in range(len(labels)):
                if trials[route_index, channel_index] == 0:
                    continue
                baseline = self.null_probabilities[
                    route_index, channel_index
                ]
                for direction, endpoint in (
                    ("decrease", 0.0),
                    ("increase", 1.0),
                ):
                    for fraction in config.change_fractions:
                        routes.append(route_index)
                        channels.append(channel_index)
                        probabilities.append(
                            (1.0 - fraction) * baseline
                            + fraction * endpoint
                        )
                        alternative_labels.append(
                            (
                                route_index,
                                channel_index,
                                direction,
                                float(fraction),
                            )
                        )
        if not alternative_labels:
            raise ValueError(
                "commissioning evidence did not activate any detector channel"
            )
        self._alternative_routes = np.asarray(routes, dtype=np.int64)
        self._alternative_channels = np.asarray(channels, dtype=np.int64)
        self._alternative_probabilities = np.asarray(
            probabilities, dtype=float
        )
        self._alternative_labels = tuple(alternative_labels)
        self._log_change_time_sums = np.full(
            len(alternative_labels), -np.inf, dtype=float
        )
        self.turn = 0
        self.alarm_turn: int | None = None
        self.log_e_value = 0.0
        self.maximum_log_e_value = 0.0
        self.leading_alternative: tuple[int, int, str, float] | None = None

    @property
    def alarmed(self) -> bool:
        return self.alarm_turn is not None

    @property
    def threshold_log_e_value(self) -> float:
        return -math.log(self.config.model_false_alarm_level)

    def update(
        self,
        route_index: int,
        successes: Sequence[int] | np.ndarray,
        trials: Sequence[int] | np.ndarray,
    ) -> dict[str, Any]:
        route = int(route_index)
        observed_successes = _integer_count_array(
            successes,
            "observed successes",
        )
        observed_trials = _integer_count_array(
            trials,
            "observed trials",
        )
        if self.turn >= self.config.horizon:
            raise RuntimeError(
                "detector received more observations than its horizon"
            )
        if not 0 <= route < self.commissioning_successes.shape[0]:
            raise IndexError("route_index is outside the commissioning model")
        if (
            observed_successes.shape != (len(self.channel_labels),)
            or observed_trials.shape != observed_successes.shape
            or np.any(observed_successes < 0)
            or np.any(observed_trials < 0)
            or np.any(observed_successes > observed_trials)
        ):
            raise ValueError(
                "observed successes/trials must be compatible channel vectors"
            )

        self.turn += 1
        log_likelihood_ratio = np.zeros(
            self._alternative_routes.shape[0], dtype=float
        )
        selected_route = self._alternative_routes == route
        for channel_index in np.flatnonzero(observed_trials > 0):
            selected = selected_route & (
                self._alternative_channels == channel_index
            )
            if not np.any(selected):
                continue
            success_count = int(observed_successes[channel_index])
            trial_count = int(observed_trials[channel_index])
            failure_count = trial_count - success_count
            alpha = (
                float(self.commissioning_successes[route, channel_index])
                + float(self._online_successes[route, channel_index])
                + self.config.commissioning_pseudocount
            )
            beta = (
                float(
                    self.commissioning_trials[route, channel_index]
                    - self.commissioning_successes[route, channel_index]
                )
                + float(
                    self._online_trials[route, channel_index]
                    - self._online_successes[route, channel_index]
                )
                + self.config.commissioning_pseudocount
            )
            log_null_predictive = float(
                betaln(alpha + success_count, beta + failure_count)
                - betaln(alpha, beta)
            )
            alternative_probability = self._alternative_probabilities[selected]
            log_alternative = (
                success_count * np.log(alternative_probability)
                + failure_count * np.log1p(-alternative_probability)
            )
            log_likelihood_ratio[selected] = (
                log_alternative - log_null_predictive
            )

        self._log_change_time_sums = (
            log_likelihood_ratio
            + np.logaddexp(0.0, self._log_change_time_sums)
        )
        self._online_successes[route] += observed_successes
        self._online_trials[route] += observed_trials

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
            "turn": int(self.turn),
            "log_e_value": float(self.log_e_value),
            "e_value": _finite_exponential(self.log_e_value),
            "threshold_log_e_value": self.threshold_log_e_value,
            "alarmed": self.alarmed,
            "newly_alarmed": newly_alarmed,
            "alarm_turn": self.alarm_turn,
            "leading_alternative": self._leading_alternative_dict(),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema": "route_conditioned_binomial_e_process.v1",
            "config": self.config.to_dict(),
            "channel_labels": list(self.channel_labels),
            "turn": int(self.turn),
            "alarm_turn": self.alarm_turn,
            "alarmed": self.alarmed,
            "log_e_value": float(self.log_e_value),
            "e_value": _finite_exponential(self.log_e_value),
            "maximum_log_e_value": float(self.maximum_log_e_value),
            "null_probabilities": self.null_probabilities.tolist(),
            "online_successes": self._online_successes.tolist(),
            "online_trials": self._online_trials.tolist(),
            "commissioning_successes": (
                self.commissioning_successes.tolist()
            ),
            "commissioning_trials": self.commissioning_trials.tolist(),
            "alternative_count": int(self._alternative_routes.size),
            "leading_alternative": self._leading_alternative_dict(),
            "model_boundary": {
                "null": (
                    "route/channel beta-binomial posterior predictive "
                    "conditioned on each observed trial count"
                ),
                "alternatives": (
                    "fixed binomial shifts mixed over route, evidence "
                    "channel, direction, magnitude, and change time"
                ),
                "channel_dependence": (
                    "channels are mixed, not multiplied; no cross-channel "
                    "independence assumption is used"
                ),
                "false_alarm_scope": (
                    "model-based under the declared commissioning-relative "
                    "posterior-predictive null"
                ),
            },
        }

    def _leading_alternative_dict(self) -> dict[str, Any] | None:
        if self.leading_alternative is None:
            return None
        route, channel, direction, fraction = self.leading_alternative
        return {
            "route_index": int(route),
            "channel": self.channel_labels[int(channel)],
            "direction": direction,
            "change_fraction": float(fraction),
        }


class MatchedClassicalQuantumEndpointDetectors:
    """Matched observation ablation using one endpoint evidence stream."""

    def __init__(
        self,
        delivery_only: RouteConditionedBinomialEProcess,
        classical: RouteConditionedBinomialEProcess,
        quantum: RouteConditionedBinomialEProcess,
        measurement_counts: RouteConditionedBinomialEProcess,
    ) -> None:
        if delivery_only.channel_labels != CLASSICAL_DELIVERY_CHANNELS:
            raise ValueError(
                "delivery-only detector has the wrong channels"
            )
        if classical.channel_labels != CLASSICAL_SERVICE_CHANNELS:
            raise ValueError(
                "classical detector must use delivery and public service"
            )
        if quantum.channel_labels != QUANTUM_ENDPOINT_CHANNELS:
            raise ValueError(
                "quantum detector must use the declared endpoint channels"
            )
        if (
            measurement_counts.channel_labels
            != E91_MEASUREMENT_COUNT_CHANNELS
        ):
            raise ValueError(
                "measurement-count detector must use only the declared "
                "E91 count channels"
            )
        self.delivery_only = delivery_only
        self.classical = classical
        self.quantum = quantum
        self.measurement_counts = measurement_counts

    @classmethod
    def from_commissioning(
        cls,
        evidence_by_route: Sequence[Iterable[QuantumEndpointEvidence]],
        config: BinomialEvidenceDetectorConfig,
    ) -> "MatchedClassicalQuantumEndpointDetectors":
        delivery_successes, delivery_trials = aggregate_route_evidence(
            evidence_by_route,
            CLASSICAL_DELIVERY_CHANNELS,
        )
        classical_successes, classical_trials = aggregate_route_evidence(
            evidence_by_route,
            CLASSICAL_SERVICE_CHANNELS,
        )
        quantum_successes, quantum_trials = aggregate_route_evidence(
            evidence_by_route,
            QUANTUM_ENDPOINT_CHANNELS,
        )
        measurement_successes, measurement_trials = aggregate_route_evidence(
            evidence_by_route,
            E91_MEASUREMENT_COUNT_CHANNELS,
        )
        return cls(
            RouteConditionedBinomialEProcess(
                delivery_successes,
                delivery_trials,
                CLASSICAL_DELIVERY_CHANNELS,
                config,
            ),
            RouteConditionedBinomialEProcess(
                classical_successes,
                classical_trials,
                CLASSICAL_SERVICE_CHANNELS,
                config,
            ),
            RouteConditionedBinomialEProcess(
                quantum_successes,
                quantum_trials,
                QUANTUM_ENDPOINT_CHANNELS,
                config,
            ),
            RouteConditionedBinomialEProcess(
                measurement_successes,
                measurement_trials,
                E91_MEASUREMENT_COUNT_CHANNELS,
                config,
            ),
        )

    def update(
        self,
        route_index: int,
        evidence: QuantumEndpointEvidence,
    ) -> dict[str, Any]:
        delivery_successes, delivery_trials = evidence.binomial_counts(
            CLASSICAL_DELIVERY_CHANNELS
        )
        classical_successes, classical_trials = evidence.binomial_counts(
            CLASSICAL_SERVICE_CHANNELS
        )
        quantum_successes, quantum_trials = evidence.binomial_counts(
            QUANTUM_ENDPOINT_CHANNELS
        )
        measurement_successes, measurement_trials = (
            evidence.binomial_counts(E91_MEASUREMENT_COUNT_CHANNELS)
        )
        return {
            "classical_delivery_only": self.delivery_only.update(
                route_index,
                delivery_successes,
                delivery_trials,
            ),
            "classical_service_baseline": self.classical.update(
                route_index,
                classical_successes,
                classical_trials,
            ),
            "quantum_endpoint_integrity": self.quantum.update(
                route_index,
                quantum_successes,
                quantum_trials,
            ),
            "e91_measurement_counts_only": self.measurement_counts.update(
                route_index,
                measurement_successes,
                measurement_trials,
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema": "matched_classical_quantum_endpoint_detectors.v1",
            "classical_delivery_only": self.delivery_only.diagnostics(),
            "classical_service_baseline": self.classical.diagnostics(),
            "quantum_endpoint_integrity": self.quantum.diagnostics(),
            "e91_measurement_counts_only": (
                self.measurement_counts.diagnostics()
            ),
            "information_boundary": {
                "shared_context": [
                    "selected route identity",
                    "requested pair count",
                ],
                "delivery_only_observations": [
                    "physically delivered pair count",
                ],
                "service_decision_observations": [
                    "physically delivered pair count",
                    "public accepted/rejected service outcome",
                ],
                "full_endpoint_observations": [
                    "physically delivered pair count",
                    "public accepted/rejected service outcome",
                    "key error count conditioned on sifted key-pair count",
                    (
                        "same/different counts conditioned on each configured "
                        "CHSH setting-pair count"
                    ),
                ],
                "measurement_count_only_observations": [
                    (
                        "sifted key-pair count conditioned on the requested "
                        "pair count"
                    ),
                    (
                        "each CHSH setting-pair trial count conditioned on "
                        "the requested pair count"
                    ),
                    "key errors conditioned on sifted key-pair count",
                    (
                        "same outcomes conditioned on each CHSH setting-pair "
                        "trial count"
                    ),
                ],
                "excluded": [
                    "Eve action identity",
                    "Eve target identity",
                    "target-active simulator flag",
                    "expected delivery, QBER, or CHSH values",
                    "private no-delivery-stage diagnostics",
                    "private Eve observation transcript",
                    (
                        "memory age, BSM-controller, and herald fields until "
                        "their endpoint-visible observation contracts exist"
                    ),
                ],
            },
            "claim_scope": (
                "matched observation ablation on one compiled "
                "quantum-network trace; the measurement-count-only arm "
                "receives no delivery telemetry, public accept/reject field, "
                "Eve action, or simulator expectation"
            ),
        }


def aggregate_route_evidence(
    evidence_by_route: Sequence[Iterable[QuantumEndpointEvidence]],
    channel_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate commissioning counts without collapsing evidence channels."""

    labels = tuple(channel_labels)
    if not evidence_by_route:
        raise ValueError("commissioning evidence must include a route")
    successes = np.zeros((len(evidence_by_route), len(labels)), dtype=np.int64)
    trials = np.zeros_like(successes)
    for route_index, route_evidence in enumerate(evidence_by_route):
        observed = False
        for evidence in route_evidence:
            observed = True
            row_successes, row_trials = evidence.binomial_counts(labels)
            successes[route_index] += row_successes
            trials[route_index] += row_trials
        if not observed:
            raise ValueError(
                "commissioning evidence must include an observation per route"
            )
    return successes, trials


def _finite_exponential(value: float) -> float:
    if value >= math.log(np.finfo(float).max):
        return float("inf")
    return math.exp(value)


def _integer_count_array(
    values: Sequence[int] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must contain integers, not booleans")
    if np.issubdtype(array.dtype, np.unsignedinteger) and np.any(
        array > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} exceeds the supported count range")
    return np.asarray(array, dtype=np.int64)
