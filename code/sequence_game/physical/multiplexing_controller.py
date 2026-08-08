"""Opt-in controller model for synchronized elementary-link time bins.

The optical histogram instrument remains a *single-mode* instrument.  This
module composes repeated uses of that instrument into one controller round
without changing the conditional memory state produced by a successful mode.

Only the narrow case whose reduction is exact is executable here:

* every mode has an independent memory/source resource;
* every mode uses the same full optical instrument;
* the two link ends launch synchronized modes at a midpoint BSM;
* the first conclusive mode is retained and later conclusive modes discarded;
* the mode spacing and the inter-round gap satisfy the active device recovery
  time; and
* every round has one constant duration.

Repeated excitation of one ion, mode-dependent instruments, periodic cooling,
and stateful controller/resource reuse are represented as explicit blockers.
They must not be approximated by silently replacing a per-excitation emission
probability with an effective branching ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONTROLLER_SCHEMA_VERSION = 1
SUPPORTED_CONTROLLER_KIND = "independent_time_bin_bank"
SUPPORTED_SCOPES = frozenset({"toy", "literature", "empirical_fit"})


class MultiplexingControllerError(ValueError):
    """Invalid controller schema or operating point."""


class MultiplexingScientificBlocker(NotImplementedError):
    """A requested controller mapping lacks an implemented physical model."""


def _finite_nonnegative(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise MultiplexingControllerError(f"{name} must be finite and >= 0")
    return out


def _probability(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise MultiplexingControllerError(f"{name} must be finite and in [0, 1]")
    return out


@dataclass(frozen=True)
class MultiplexingRoundResult:
    """Exact reduction of an executable controller to one geometric round."""

    single_mode_success_probability: float
    round_success_probability: float
    round_failure_probability: float
    round_period_s: float
    herald_latency_s: float
    mode_offsets_s: tuple[float, ...]
    conditional_first_success_weights: tuple[float, ...]
    success_count_distribution: tuple[float, ...]
    multiple_success_probability: float
    expected_successful_modes: float
    expected_discarded_successes: float
    expected_first_success_herald_time_s: float | None
    effective_time_per_launched_mode_s: float
    mean_time_to_round_success_s: float
    success_rate_per_s: float
    single_mode_reference_rate_per_s: float
    rate_enhancement_vs_first_mode: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_mode_success_probability": self.single_mode_success_probability,
            "round_success_probability": self.round_success_probability,
            "round_failure_probability": self.round_failure_probability,
            "round_period_s": self.round_period_s,
            "herald_latency_s": self.herald_latency_s,
            "mode_count": len(self.mode_offsets_s),
            "mode_offsets_s": list(self.mode_offsets_s),
            "conditional_first_success_weights": list(
                self.conditional_first_success_weights
            ),
            "success_count_distribution": list(self.success_count_distribution),
            "multiple_success_probability": self.multiple_success_probability,
            "expected_successful_modes": self.expected_successful_modes,
            "expected_discarded_successes": self.expected_discarded_successes,
            "expected_first_success_herald_time_s": (
                self.expected_first_success_herald_time_s
            ),
            "effective_time_per_launched_mode_s": self.effective_time_per_launched_mode_s,
            "mean_time_to_round_success_s": self.mean_time_to_round_success_s,
            "mean_time_to_round_success_semantics": (
                "failed rounds consume T_round; the successful round ends for the filled slot at "
                "the selected mode's herald, so E[T_fill]=((1-p_round)/p_round)T_round+E[T_herald|success]"
            ),
            "success_rate_per_s": self.success_rate_per_s,
            "success_rate_semantics": (
                "continuous saturated controller throughput p_round/T_round; distinct from the inverse "
                "single-slot fill time when a success is accepted before round close"
            ),
            "single_mode_reference_rate_per_s": self.single_mode_reference_rate_per_s,
            "rate_enhancement_vs_first_mode": self.rate_enhancement_vs_first_mode,
            "probability_model": "iid repeated full optical instrument",
            "selection_policy": "first conclusive mode retained; later conclusive modes discarded",
        }


@dataclass(frozen=True)
class MultiplexingController:
    """Serializable constant-cadence independent time-bin controller.

    ``mode_offsets_s`` are launch times measured from the start of a round.
    A round closes after the last mode's herald returns plus
    ``fixed_round_overhead_s``.  This is the midpoint-BSM mapping of the
    time-bin timing relation: for a node-to-node edge of length ``D``, each
    endpoint-to-midpoint photon traversal plus midpoint-to-endpoint herald
    return is ``D/c`` in total.
    """

    mode_offsets_s: tuple[float, ...]
    mode_resource_ids: tuple[str, ...]
    scope: str
    reference: str
    fixed_round_overhead_s: float = 0.0
    periodic_cooling_duration_s: float = 0.0
    periodic_cooling_every_rounds: int | None = None
    controller_kind: str = SUPPORTED_CONTROLLER_KIND
    topology_mapping: str = "synchronized_two_node_midpoint_bsm"
    mode_probability_model: str = "iid_full_instrument"
    conditional_instrument_model: str = "mode_invariant"
    herald_selection_policy: str = "first_success"
    multiple_success_policy: str = "keep_first_discard_later"
    schema_version: int = CONTROLLER_SCHEMA_VERSION
    activation_blockers: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if int(self.schema_version) != CONTROLLER_SCHEMA_VERSION:
            raise MultiplexingControllerError(
                f"unsupported controller schema_version {self.schema_version!r}; "
                f"expected {CONTROLLER_SCHEMA_VERSION}"
            )
        if str(self.controller_kind) != SUPPORTED_CONTROLLER_KIND:
            raise MultiplexingScientificBlocker(
                f"controller kind {self.controller_kind!r} is not implemented; "
                f"supported executable kind is {SUPPORTED_CONTROLLER_KIND!r}"
            )
        if str(self.scope) not in SUPPORTED_SCOPES:
            raise MultiplexingControllerError(
                "controller scope must be one of toy, literature, empirical_fit"
            )
        if str(self.scope) == "literature" and not str(self.reference).strip():
            raise MultiplexingControllerError(
                "literature-scoped multiplexing requires a nonempty reference"
            )
        offsets = tuple(
            _finite_nonnegative(f"mode_offsets_s[{index}]", value)
            for index, value in enumerate(self.mode_offsets_s)
        )
        if not offsets:
            raise MultiplexingControllerError("at least one mode offset is required")
        if any(right <= left for left, right in zip(offsets, offsets[1:])):
            raise MultiplexingControllerError(
                "mode_offsets_s must be strictly increasing for a time-bin controller"
            )
        resources = tuple(str(value).strip() for value in self.mode_resource_ids)
        if len(resources) != len(offsets):
            raise MultiplexingControllerError(
                "mode_resource_ids must provide one resource id per mode"
            )
        if any(not value for value in resources):
            raise MultiplexingControllerError("mode resource ids must be nonempty")
        if len(set(resources)) != len(resources):
            raise MultiplexingScientificBlocker(
                "independent_time_bin_bank requires a distinct memory/source resource per mode; "
                "repeated excitation of one resource needs a population-state controller model"
            )
        _finite_nonnegative("fixed_round_overhead_s", self.fixed_round_overhead_s)
        cooling = _finite_nonnegative(
            "periodic_cooling_duration_s", self.periodic_cooling_duration_s
        )
        if cooling > 0.0:
            if self.periodic_cooling_every_rounds is None or int(
                self.periodic_cooling_every_rounds
            ) < 1:
                raise MultiplexingControllerError(
                    "positive periodic cooling requires periodic_cooling_every_rounds >= 1"
                )
        elif self.periodic_cooling_every_rounds is not None:
            raise MultiplexingControllerError(
                "periodic_cooling_every_rounds requires positive periodic_cooling_duration_s"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MultiplexingController":
        periodic = raw.get("periodic_cooling") or {}
        return cls(
            schema_version=int(raw.get("schema_version", CONTROLLER_SCHEMA_VERSION)),
            controller_kind=str(raw.get("controller_kind", raw.get("kind", ""))),
            scope=str(raw.get("scope", "")),
            reference=str(raw.get("reference", raw.get("reference_tag", ""))),
            mode_offsets_s=tuple(float(value) for value in raw.get("mode_offsets_s", ())),
            mode_resource_ids=tuple(str(value) for value in raw.get("mode_resource_ids", ())),
            fixed_round_overhead_s=float(raw.get("fixed_round_overhead_s", 0.0)),
            periodic_cooling_duration_s=float(
                periodic.get("duration_s", raw.get("periodic_cooling_duration_s", 0.0))
            ),
            periodic_cooling_every_rounds=(
                int(periodic.get("every_rounds", raw.get("periodic_cooling_every_rounds")))
                if periodic.get("every_rounds", raw.get("periodic_cooling_every_rounds"))
                is not None
                else None
            ),
            topology_mapping=str(
                raw.get("topology_mapping", "synchronized_two_node_midpoint_bsm")
            ),
            mode_probability_model=str(
                raw.get("mode_probability_model", "iid_full_instrument")
            ),
            conditional_instrument_model=str(
                raw.get("conditional_instrument_model", "mode_invariant")
            ),
            herald_selection_policy=str(
                raw.get("herald_selection_policy", "first_success")
            ),
            multiple_success_policy=str(
                raw.get("multiple_success_policy", "keep_first_discard_later")
            ),
            activation_blockers=tuple(str(value) for value in raw.get("activation_blockers", ())),
            notes=str(raw.get("notes", "")),
        )

    @property
    def mode_count(self) -> int:
        return len(self.mode_offsets_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "controller_kind": self.controller_kind,
            "scope": self.scope,
            "reference": self.reference,
            "topology_mapping": self.topology_mapping,
            "mode_probability_model": self.mode_probability_model,
            "conditional_instrument_model": self.conditional_instrument_model,
            "herald_selection_policy": self.herald_selection_policy,
            "multiple_success_policy": self.multiple_success_policy,
            "mode_count": self.mode_count,
            "mode_offsets_s": list(self.mode_offsets_s),
            "mode_resource_ids": list(self.mode_resource_ids),
            "fixed_round_overhead_s": self.fixed_round_overhead_s,
            "periodic_cooling": {
                "duration_s": self.periodic_cooling_duration_s,
                "every_rounds": self.periodic_cooling_every_rounds,
            },
            "activation_blockers": list(self.activation_blockers),
            "notes": self.notes,
        }

    def execution_blockers(self) -> tuple[str, ...]:
        blockers = list(self.activation_blockers)
        expected_fields = {
            "topology_mapping": (
                self.topology_mapping,
                "synchronized_two_node_midpoint_bsm",
            ),
            "mode_probability_model": (
                self.mode_probability_model,
                "iid_full_instrument",
            ),
            "conditional_instrument_model": (
                self.conditional_instrument_model,
                "mode_invariant",
            ),
            "herald_selection_policy": (
                self.herald_selection_policy,
                "first_success",
            ),
            "multiple_success_policy": (
                self.multiple_success_policy,
                "keep_first_discard_later",
            ),
        }
        for field, (actual, expected) in expected_fields.items():
            if actual != expected:
                blockers.append(
                    f"{field}={actual!r} is unsupported; executable value is {expected!r}"
                )
        if self.periodic_cooling_duration_s > 0.0:
            blockers.append(
                "periodic cooling requires a phase-aware nonuniform waiting-time distribution; "
                "the histogram waiting kernel currently consumes one constant round period"
            )
        return tuple(blockers)

    def round_timing(
        self,
        *,
        herald_latency_s: float,
        minimum_shared_device_recovery_s: float = 0.0,
    ) -> dict[str, Any]:
        """Validate cadence and return the exact constant round timing."""

        blockers = self.execution_blockers()
        if blockers:
            raise MultiplexingScientificBlocker(
                "multiplexing controller cannot be activated: " + "; ".join(blockers)
            )
        herald = _finite_nonnegative("herald_latency_s", herald_latency_s)
        recovery = _finite_nonnegative(
            "minimum_shared_device_recovery_s", minimum_shared_device_recovery_s
        )
        overhead = _finite_nonnegative(
            "fixed_round_overhead_s", self.fixed_round_overhead_s
        )
        offsets = tuple(float(value) for value in self.mode_offsets_s)
        separations = [right - left for left, right in zip(offsets, offsets[1:])]
        round_period = offsets[-1] + herald + overhead
        if round_period <= 0.0:
            raise MultiplexingControllerError("controller round period must be > 0")
        cross_round_separation = round_period - offsets[-1] + offsets[0]
        all_separations = separations + [cross_round_separation]
        if recovery > 0.0 and any(value + 1e-18 < recovery for value in all_separations):
            shortest = min(all_separations)
            raise MultiplexingScientificBlocker(
                "multiplexing cadence violates the shared detector/source recovery constraint: "
                f"shortest separation {shortest:.17g} s < required {recovery:.17g} s"
            )
        return {
            "round_period_s": round_period,
            "herald_latency_s": herald,
            "mode_count": self.mode_count,
            "mode_offsets_s": list(offsets),
            "within_round_mode_separations_s": separations,
            "cross_round_mode_separation_s": cross_round_separation,
            "minimum_shared_device_recovery_s": recovery,
            "fixed_round_overhead_s": overhead,
            "constant_round_period": True,
        }

    def evaluate_round(
        self,
        *,
        single_mode_success_probability: float,
        herald_latency_s: float,
        minimum_shared_device_recovery_s: float = 0.0,
    ) -> MultiplexingRoundResult:
        """Compose the per-mode instrument into a constant-duration round."""

        p = _probability(
            "single_mode_success_probability", single_mode_success_probability
        )
        timing = self.round_timing(
            herald_latency_s=herald_latency_s,
            minimum_shared_device_recovery_s=minimum_shared_device_recovery_s,
        )
        herald = float(timing["herald_latency_s"])
        overhead = float(timing["fixed_round_overhead_s"])
        offsets = tuple(float(value) for value in self.mode_offsets_s)
        round_period = float(timing["round_period_s"])

        n = self.mode_count
        if p >= 1.0:
            failure = 0.0
            p_round = 1.0
        elif p <= 0.0:
            failure = 1.0
            p_round = 0.0
        else:
            log_failure = n * math.log1p(-p)
            failure = math.exp(log_failure)
            p_round = -math.expm1(log_failure)
        count_distribution = tuple(
            math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
            for k in range(n + 1)
        )
        count_total = math.fsum(count_distribution)
        if count_total > 0.0:
            count_distribution = tuple(value / count_total for value in count_distribution)

        first_unconditional: list[float] = []
        preceding_failure = 1.0
        for _index in range(n):
            first_unconditional.append(preceding_failure * p)
            preceding_failure *= 1.0 - p
        first_weights = tuple(
            value / p_round for value in first_unconditional
        ) if p_round > 0.0 else tuple(0.0 for _value in first_unconditional)
        expected_first_herald = (
            math.fsum(
                weight * (offset + herald)
                for weight, offset in zip(first_weights, offsets)
            )
            if p_round > 0.0
            else None
        )
        multiple = math.fsum(count_distribution[2:]) if n >= 2 else 0.0
        expected_successes = n * p
        expected_discarded = max(0.0, expected_successes - p_round)
        success_rate = p_round / round_period
        # A slot becomes live when the selected mode's herald arrives, not at
        # the end of the successful round. Failed rounds consume the full
        # period; the successful round consumes only its conditional herald
        # time. This distinction matters when a mode train is not negligible.
        mean_success_time = (
            ((1.0 - p_round) / p_round) * round_period
            + float(expected_first_herald)
            if p_round > 0.0 and expected_first_herald is not None
            else math.inf
        )
        first_mode_period = offsets[0] + herald + overhead
        single_mode_rate = p / first_mode_period if first_mode_period > 0.0 else 0.0
        enhancement = (
            success_rate / single_mode_rate if single_mode_rate > 0.0 else None
        )
        return MultiplexingRoundResult(
            single_mode_success_probability=p,
            round_success_probability=p_round,
            round_failure_probability=failure,
            round_period_s=round_period,
            herald_latency_s=herald,
            mode_offsets_s=offsets,
            conditional_first_success_weights=first_weights,
            success_count_distribution=count_distribution,
            multiple_success_probability=multiple,
            expected_successful_modes=expected_successes,
            expected_discarded_successes=expected_discarded,
            expected_first_success_herald_time_s=expected_first_herald,
            effective_time_per_launched_mode_s=round_period / n,
            mean_time_to_round_success_s=mean_success_time,
            success_rate_per_s=success_rate,
            single_mode_reference_rate_per_s=single_mode_rate,
            rate_enhancement_vs_first_mode=enhancement,
        )


def controller_from_config(
    raw: Mapping[str, Any] | MultiplexingController | None,
) -> MultiplexingController | None:
    if raw is None:
        return None
    if isinstance(raw, MultiplexingController):
        return raw
    return MultiplexingController.from_dict(raw)


def poisson_binomial_distribution(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Reference Poisson-binomial fold used by tests and future explicit modes."""

    distribution = [1.0]
    for index, raw_probability in enumerate(probabilities):
        probability = _probability(f"probabilities[{index}]", raw_probability)
        updated = [0.0] * (len(distribution) + 1)
        for successes, weight in enumerate(distribution):
            updated[successes] += weight * (1.0 - probability)
            updated[successes + 1] += weight * probability
        distribution = updated
    return tuple(distribution)
