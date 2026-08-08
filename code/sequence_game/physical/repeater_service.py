"""Literature-scoped asynchronous repeater service and Bell-quality models.

The histogram kernel evaluates the conditional quantum transcript quickly, but
it cannot infer a repeater controller, memory inventory, or service rate from a
single optical-attempt probability.  This module supplies that missing layer
without importing SeQUeNCe or the histogram implementation.

Only equations explicitly selected by a validated service profile are
implemented.  In particular, the Krutyanskiy et al. profile uses

* the paper's two-photon midpoint-link rate model;
* its nested-chain approximation (Supplementary Eq. S12);
* its elementary-link phase-error model (Supplementary Eqs. S13--S14); and
* its local-swap single-qubit depolarizing map (Supplementary Eq. S15).

The rate and Bell-quality calculations are deliberately separate.  A route can
have a scientifically meaningful delivery rate while its unpurified final
state is too noisy for a Bell violation or E91 key generation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .entanglement_purification import (
    BilateralRecurrenceModel,
    PurificationError,
    bell_diagonal_density,
    bilateral_phase_error_recurrence,
)
from .empirical_bell import psi_minus_distribution_from_model
from .local_bell_measurement import LocalBellMeasurementError, LocalBellMeasurementModel


REPEATER_SERVICE_SCHEMA_VERSION = 1
REPEATER_SERVICE_SCOPES = ("literature", "empirical_fit", "toy", "placeholder")
REPEATER_CLAIM_TIERS = (
    "demonstrated_system",
    "demonstrated_component",
    "near_term_composite",
    "near_future_projection",
    "diagnostic_reference",
)
RATE_MODEL_KINDS = (
    "krutyanskiy_midpoint_two_photon",
    "cui_note8_midpoint_two_photon_projection",
    "empirical_elementary_link",
    "empirical_ion_photon_node_component",
)
STATE_MODEL_KINDS = (
    "krutyanskiy_equations_S13_S15",
    "detector_conditioned_fock_bsm",
    "pauli_twirled_bell_fidelity",
    "bell_diagonal_correlations",
    "scalar_bell_fidelity_metadata_only",
)
ELEMENTARY_LINK_INSTRUMENT_KINDS = (
    "fock_midpoint_two_photon",
    "empirical_remote_memory_pair",
)
PAULI_LABELS = ("I", "X", "Z", "Y")
_PURIFIED_LINK_ESTIMATE_CACHE: dict[
    tuple[str, str, float], "PurifiedElementaryLinkEstimate"
] = {}


class RepeaterServiceError(ValueError):
    """Invalid or inapplicable repeater-service configuration."""


def _probability(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RepeaterServiceError(f"{name} must be a finite probability in [0, 1]")
    return result


def _positive(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RepeaterServiceError(f"{name} must be finite and > 0")
    return result


def _nonnegative(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RepeaterServiceError(f"{name} must be finite and >= 0")
    return result


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RepeaterServiceError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class RepeaterServiceProfile:
    """Validated physical/controller profile carried into the histogram path."""

    profile_id: str
    scope: str
    claim_tier: str
    architecture: str
    source: dict[str, Any]
    validity: dict[str, Any]
    hardware: dict[str, Any]
    rate_model: dict[str, Any]
    state_model: dict[str, Any]
    memory_model: dict[str, Any]
    controller_model: dict[str, Any]
    histogram_adapter: dict[str, Any]
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepeaterServiceProfile":
        data = dict(payload)
        version = int(data.get("schema_version", 0))
        if version != REPEATER_SERVICE_SCHEMA_VERSION:
            raise RepeaterServiceError(
                f"unsupported repeater-service schema_version {version}; "
                f"expected {REPEATER_SERVICE_SCHEMA_VERSION}"
            )
        profile_id = str(data.get("profile_id", "")).strip()
        if not profile_id:
            raise RepeaterServiceError("profile_id must be non-empty")
        scope = str(data.get("scope", "")).strip()
        if scope not in REPEATER_SERVICE_SCOPES:
            raise RepeaterServiceError(
                f"scope must be one of {REPEATER_SERVICE_SCOPES}, got {scope!r}"
            )
        claim_tier = str(data.get("claim_tier", "")).strip()
        if claim_tier not in REPEATER_CLAIM_TIERS:
            raise RepeaterServiceError(
                f"claim_tier must be one of {REPEATER_CLAIM_TIERS}, got {claim_tier!r}"
            )
        architecture = str(data.get("architecture", "")).strip()
        if not architecture:
            raise RepeaterServiceError("architecture must be non-empty")

        source = _require_mapping("source", data.get("source"))
        if not str(source.get("citation", "")).strip():
            raise RepeaterServiceError("source.citation must be non-empty")
        validity = _require_mapping("validity", data.get("validity"))
        lower = _positive("validity.edge_length_min_km", validity.get("edge_length_min_km"))
        upper = _positive("validity.edge_length_max_km", validity.get("edge_length_max_km"))
        if upper < lower:
            raise RepeaterServiceError("validity edge-length maximum must be >= minimum")

        hardware = _require_mapping("hardware", data.get("hardware"))
        if int(hardware.get("memory_qubits_per_node", 0)) < 1:
            raise RepeaterServiceError("hardware.memory_qubits_per_node must be >= 1")
        if int(hardware.get("memory_slots_per_adjacent_edge_side", 0)) < 1:
            raise RepeaterServiceError(
                "hardware.memory_slots_per_adjacent_edge_side must be >= 1"
            )
        output_slots = int(
            hardware.get(
                "purified_output_slots_per_adjacent_edge_side",
                hardware["memory_slots_per_adjacent_edge_side"],
            )
        )
        if output_slots < 1:
            raise RepeaterServiceError(
                "hardware.purified_output_slots_per_adjacent_edge_side must be >= 1"
            )
        if output_slots > int(hardware["memory_slots_per_adjacent_edge_side"]):
            raise RepeaterServiceError(
                "purified output slots cannot exceed physical memory slots per edge side"
            )

        rate_model = _require_mapping("rate_model", data.get("rate_model"))
        rate_kind = str(rate_model.get("kind", ""))
        if rate_kind not in RATE_MODEL_KINDS:
            raise RepeaterServiceError(
                f"rate_model.kind must be one of {RATE_MODEL_KINDS}, got {rate_kind!r}"
            )
        if rate_kind == "krutyanskiy_midpoint_two_photon":
            _probability(
                "rate_model.per_node_zero_loss_detection_probability",
                rate_model.get("per_node_zero_loss_detection_probability"),
            )
            _nonnegative(
                "rate_model.attenuation_exponent_per_km",
                rate_model.get("attenuation_exponent_per_km"),
            )
            _positive("rate_model.attempt_overhead_s", rate_model.get("attempt_overhead_s"))
            _positive("rate_model.fiber_speed_m_per_s", rate_model.get("fiber_speed_m_per_s"))
            if str(rate_model.get("nested_time_model", "")) != "supplementary_equation_S12":
                raise RepeaterServiceError(
                    "krutyanskiy rate_model requires nested_time_model='supplementary_equation_S12'"
                )
        elif rate_kind == "cui_note8_midpoint_two_photon_projection":
            _positive(
                "rate_model.reference_edge_length_km",
                rate_model.get("reference_edge_length_km"),
            )
            _positive(
                "rate_model.reference_success_rate_hz",
                rate_model.get("reference_success_rate_hz"),
            )
            _nonnegative(
                "rate_model.fiber_attenuation_db_per_km",
                rate_model.get("fiber_attenuation_db_per_km"),
            )
            _positive(
                "rate_model.fiber_speed_m_per_s",
                rate_model.get("fiber_speed_m_per_s"),
            )
            if str(rate_model.get("completion_time_model", "")) != (
                "empirical_exponential_renewal"
            ):
                raise RepeaterServiceError(
                    "Cui Note 8 projection requires "
                    "completion_time_model='empirical_exponential_renewal'"
                )
            if int(rate_model.get("source_data_total_modes", 0)) < 1:
                raise RepeaterServiceError(
                    "Cui Note 8 projection requires source_data_total_modes >= 1"
                )
        else:
            _positive("rate_model.reference_edge_length_km", rate_model.get("reference_edge_length_km"))
            empirical_rate = _positive(
                "rate_model.measured_success_rate_hz",
                rate_model.get("measured_success_rate_hz"),
            )
            measured_mean = rate_model.get("measured_mean_distribution_time_s")
            if measured_mean is not None and not math.isclose(
                _positive(
                    "rate_model.measured_mean_distribution_time_s",
                    measured_mean,
                ),
                1.0 / empirical_rate,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise RepeaterServiceError(
                    "empirical measured mean distribution time must equal the inverse "
                    "of measured_success_rate_hz"
                )
            completion_kind = rate_model.get("completion_time_model")
            if completion_kind is not None and str(completion_kind) not in {
                "empirical_exponential_renewal",
            }:
                raise RepeaterServiceError(
                    "unsupported empirical rate_model.completion_time_model "
                    f"{completion_kind!r}"
                )

        state_model = _require_mapping("state_model", data.get("state_model"))
        state_kind = str(state_model.get("kind", ""))
        if state_kind == "krutyanskiy_equations_S13_S15":
            _probability("state_model.ion_photon_bell_fidelity", state_model.get("ion_photon_bell_fidelity"))
            _probability("state_model.photonic_bsm_fidelity", state_model.get("photonic_bsm_fidelity"))
            _probability("state_model.local_swap_bell_fidelity", state_model.get("local_swap_bell_fidelity"))
        elif state_kind == "detector_conditioned_fock_bsm":
            _probability(
                "state_model.service_estimate_raw_bell_fidelity",
                state_model.get("service_estimate_raw_bell_fidelity"),
            )
            if str(state_model.get("conditional_state_semantics", "")) != (
                "histogram_detector_conditioned_density_matrix"
            ):
                raise RepeaterServiceError(
                    "detector-conditioned Fock state requires "
                    "conditional_state_semantics='histogram_detector_conditioned_density_matrix'"
                )
        elif state_kind == "scalar_bell_fidelity_metadata_only":
            _probability("state_model.reported_bell_fidelity", state_model.get("reported_bell_fidelity"))
        elif state_kind == "pauli_twirled_bell_fidelity":
            _probability(
                "state_model.reported_bell_fidelity",
                state_model.get("reported_bell_fidelity"),
            )
            if str(state_model.get("target_bell_state", "")) != "psi_minus":
                raise RepeaterServiceError(
                    "pauli-twirled elementary pairs currently require "
                    "state_model.target_bell_state='psi_minus'"
                )
            if str(state_model.get("twirling_map", "")) != "bilateral_clifford_isotropic_twirl":
                raise RepeaterServiceError(
                    "pauli-twirled elementary pairs require "
                    "state_model.twirling_map='bilateral_clifford_isotropic_twirl'"
                )
        elif state_kind == "bell_diagonal_correlations":
            try:
                psi_minus_distribution_from_model(state_model)
            except ValueError as exc:
                raise RepeaterServiceError(str(exc)) from exc
            local_swap = state_model.get("local_swap_model")
            if local_swap is not None:
                try:
                    LocalBellMeasurementModel.from_mapping(
                        _require_mapping("state_model.local_swap_model", local_swap)
                    )
                except LocalBellMeasurementError as exc:
                    raise RepeaterServiceError(str(exc)) from exc
        else:
            raise RepeaterServiceError(
                f"state_model.kind must be one of {STATE_MODEL_KINDS}"
            )

        restoration = state_model.get("quality_restoration")
        if restoration is not None:
            try:
                recurrence = BilateralRecurrenceModel.from_mapping(
                    _require_mapping("state_model.quality_restoration", restoration)
                )
            except PurificationError as exc:
                raise RepeaterServiceError(str(exc)) from exc
            memory_slots = int(hardware["memory_slots_per_adjacent_edge_side"])
            lanes = int(hardware.get("parallel_purification_lanes_per_edge", 0))
            if lanes < 1:
                raise RepeaterServiceError(
                    "quality-restored hardware requires "
                    "parallel_purification_lanes_per_edge >= 1"
                )
            if output_slots != lanes:
                raise RepeaterServiceError(
                    "each independently maintained purified output slot must have one "
                    "declared parallel purification lane"
                )
            required_edge_memories = recurrence.raw_pairs_per_round * lanes
            if memory_slots < required_edge_memories:
                raise RepeaterServiceError(
                    "bilateral recurrence requires two physical raw-pair memory slots "
                    "per parallel purification lane and adjacent edge side"
                )
            simultaneous = int(hardware.get("simultaneous_elementary_links_per_node", 0))
            required_memories = memory_slots * max(1, simultaneous)
            if int(hardware["memory_qubits_per_node"]) < required_memories:
                raise RepeaterServiceError(
                    "hardware.memory_qubits_per_node is smaller than the declared "
                    "per-edge-side inventory times simultaneous adjacent links"
                )
            communication = int(hardware.get("communication_qubits_per_node", 0))
            communication_per_mode = int(
                hardware.get("communication_qubits_per_raw_generation_mode", 1)
            )
            if communication_per_mode < 1:
                raise RepeaterServiceError(
                    "communication_qubits_per_raw_generation_mode must be >= 1"
                )
            required_communication = (
                recurrence.parallel_raw_pair_generation_modes_per_edge
                * lanes
                * max(1, simultaneous)
                * communication_per_mode
            )
            if communication < required_communication:
                raise RepeaterServiceError(
                    "hardware.communication_qubits_per_node is smaller than the declared "
                    "raw-generation modules per purification lane and simultaneous edge count"
                )

        memory_model = _require_mapping("memory_model", data.get("memory_model"))
        storage = _require_mapping("memory_model.storage_decoherence_model", memory_model.get("storage_decoherence_model"))
        storage_kind = str(storage.get("kind", ""))
        if storage_kind not in {
            "none",
            "gaussian_dephasing",
            "exponential_dephasing",
            "liu_2026_anisotropic_pair_fit",
        }:
            raise RepeaterServiceError(
                "unsupported memory storage kind"
            )
        if storage_kind in {"gaussian_dephasing", "exponential_dephasing"}:
            _positive("memory storage coherence_time_s", storage.get("coherence_time_s"))
        elif storage_kind == "liu_2026_anisotropic_pair_fit":
            _positive(
                "memory storage xx_pair_coherence_time_s",
                storage.get("xx_pair_coherence_time_s"),
            )
            _probability(
                "memory storage zz_initial_visibility",
                storage.get("zz_initial_visibility"),
            )
            _nonnegative(
                "memory storage zz_linear_decay_rate_per_s",
                storage.get("zz_linear_decay_rate_per_s"),
            )
            _positive("memory storage fit_domain_max_s", storage.get("fit_domain_max_s"))
            if str(storage.get("local_factorization", "")) != "symmetric_square_root":
                raise RepeaterServiceError(
                    "Liu anisotropic storage requires local_factorization='symmetric_square_root'"
                )
        decay = memory_model.get("detected_decay_lifetime_s")
        if decay is not None:
            _positive("memory_model.detected_decay_lifetime_s", decay)

        controller_model = _require_mapping("controller_model", data.get("controller_model"))
        if str(controller_model.get("kind", "")) != "balanced_nested_asynchronous":
            raise RepeaterServiceError(
                "controller_model.kind must be 'balanced_nested_asynchronous'"
            )
        _nonnegative(
            "controller_model.local_swap_duration_s",
            controller_model.get("local_swap_duration_s", 0.0),
        )
        _probability(
            "controller_model.local_swap_success_probability",
            controller_model.get("local_swap_success_probability", 1.0),
        )
        completion_model = controller_model.get("empirical_multilink_completion_model")
        if completion_model is not None and str(completion_model) != (
            "independent_exponential_balanced_tree"
        ):
            raise RepeaterServiceError(
                "unsupported empirical_multilink_completion_model"
            )
        if controller_model.get("accepted_output_alive_time_s") is not None:
            _positive(
                "controller_model.accepted_output_alive_time_s",
                controller_model.get("accepted_output_alive_time_s"),
            )

        histogram_adapter = _require_mapping(
            "histogram_adapter", data.get("histogram_adapter", {})
        )
        instrument_kind = str(
            histogram_adapter.get(
                "elementary_link_instrument_kind", "fock_midpoint_two_photon"
            )
        )
        if instrument_kind not in ELEMENTARY_LINK_INSTRUMENT_KINDS:
            raise RepeaterServiceError(
                "histogram_adapter.elementary_link_instrument_kind must be one of "
                f"{ELEMENTARY_LINK_INSTRUMENT_KINDS}, got {instrument_kind!r}"
            )
        if instrument_kind == "empirical_remote_memory_pair":
            if state_kind not in {"pauli_twirled_bell_fidelity", "bell_diagonal_correlations"}:
                raise RepeaterServiceError(
                    "an empirical remote-memory pair instrument requires a validated "
                    "executable Bell-diagonal state model"
                )
            if histogram_adapter.get("apply_elementary_state_channel", False):
                raise RepeaterServiceError(
                    "an empirical remote-memory pair is a replacement elementary instrument, "
                    "not a post-Fock state channel"
                )
        local_swap = state_model.get("local_swap_model")
        if local_swap is not None or state_kind == "krutyanskiy_equations_S13_S15":
            if rate_kind == "empirical_elementary_link" and completion_model is None:
                raise RepeaterServiceError(
                    "multi-hop empirical links require an explicit completion controller"
                )
        if state_model.get("local_swap_model") is not None:
            local_bsm = LocalBellMeasurementModel.from_mapping(
                _require_mapping(
                    "state_model.local_swap_model", state_model["local_swap_model"]
                )
            )
            if not math.isclose(
                float(controller_model.get("local_swap_duration_s", 0.0)),
                local_bsm.duration_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise RepeaterServiceError(
                    "controller local_swap_duration_s must equal the local Bell-measurement duration"
                )
            if float(controller_model.get("local_swap_success_probability", 1.0)) != 1.0:
                raise RepeaterServiceError(
                    "the selected complete trapped-ion Bell measurement is deterministic"
                )
            if not histogram_adapter.get("apply_local_bell_measurement_override", False):
                raise RepeaterServiceError(
                    "local swap model requires apply_local_bell_measurement_override=true"
                )
        return cls(
            profile_id=profile_id,
            scope=scope,
            claim_tier=claim_tier,
            architecture=architecture,
            source=source,
            validity=validity,
            hardware=hardware,
            rate_model=rate_model,
            state_model=state_model,
            memory_model=memory_model,
            controller_model=controller_model,
            histogram_adapter=histogram_adapter,
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPEATER_SERVICE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "scope": self.scope,
            "claim_tier": self.claim_tier,
            "architecture": self.architecture,
            "source": dict(self.source),
            "validity": dict(self.validity),
            "hardware": dict(self.hardware),
            "rate_model": dict(self.rate_model),
            "state_model": dict(self.state_model),
            "memory_model": dict(self.memory_model),
            "controller_model": dict(self.controller_model),
            "histogram_adapter": dict(self.histogram_adapter),
            "notes": self.notes,
        }

    def validate_edge_length(self, edge_length_km: float) -> float:
        length = _positive("edge_length_km", edge_length_km)
        lower = float(self.validity["edge_length_min_km"])
        upper = float(self.validity["edge_length_max_km"])
        if not lower <= length <= upper:
            raise RepeaterServiceError(
                f"profile {self.profile_id!r} is scoped to {lower:g}--{upper:g} km "
                f"elementary links, got {length:g} km"
            )
        return length


@dataclass(frozen=True)
class ElementaryLinkEstimate:
    edge_length_km: float
    attempt_period_s: float | None
    success_probability_per_attempt: float | None
    mean_distribution_time_s: float
    success_rate_hz: float
    half_link_transmission: float | None
    rate_model_kind: str
    source_equation: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_length_km": self.edge_length_km,
            "attempt_period_s": self.attempt_period_s,
            "success_probability_per_attempt": self.success_probability_per_attempt,
            "mean_distribution_time_s": self.mean_distribution_time_s,
            "success_rate_hz": self.success_rate_hz,
            "half_link_transmission": self.half_link_transmission,
            "rate_model_kind": self.rate_model_kind,
            "source_equation": self.source_equation,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class PurifiedElementaryLinkEstimate:
    """Mean renewal time and conditional state for one purified edge pair."""

    edge_length_km: float
    raw_link: ElementaryLinkEstimate
    raw_pairs_consumed_per_cycle: int
    parallel_generation_modes: int
    parallel_purification_lanes: int
    mean_raw_pair_batch_time_s: float
    local_operation_duration_s: float
    classical_comparison_duration_s: float
    mean_cycle_duration_s: float
    purification_success_probability: float
    per_lane_mean_distribution_time_s: float
    mean_distribution_time_s: float
    success_rate_hz: float
    sacrificial_wait_coherence: float
    output_pauli_error_distribution: tuple[float, float, float, float]
    source_equations: tuple[str, ...]
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_length_km": self.edge_length_km,
            "raw_link": self.raw_link.to_dict(),
            "raw_pairs_consumed_per_cycle": self.raw_pairs_consumed_per_cycle,
            "parallel_generation_modes": self.parallel_generation_modes,
            "parallel_purification_lanes": self.parallel_purification_lanes,
            "mean_raw_pair_batch_time_s": self.mean_raw_pair_batch_time_s,
            "local_operation_duration_s": self.local_operation_duration_s,
            "classical_comparison_duration_s": self.classical_comparison_duration_s,
            "mean_cycle_duration_s": self.mean_cycle_duration_s,
            "purification_success_probability": self.purification_success_probability,
            "per_lane_mean_distribution_time_s": self.per_lane_mean_distribution_time_s,
            "mean_distribution_time_s": self.mean_distribution_time_s,
            "success_rate_hz": self.success_rate_hz,
            "sacrificial_wait_coherence": self.sacrificial_wait_coherence,
            "output_pauli_error_distribution": {
                label: value
                for label, value in zip(PAULI_LABELS, self.output_pauli_error_distribution)
            },
            "source_equations": list(self.source_equations),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class RouteServiceEstimate:
    hop_count: int
    total_distance_km: float
    mean_distribution_time_s: float
    success_rate_hz: float
    published_nested_time_s: float | None
    controller_inclusive_time_s: float
    calculation_mode: str
    exact_for_declared_model: bool
    memory_coherence_time_s: float | None
    coherence_to_service_time_ratio: float | None
    memory_regime: str
    node_memory_qubits: int
    memory_slots_per_adjacent_edge_side: int
    assumptions: tuple[str, ...]
    quality_restoration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_count": self.hop_count,
            "total_distance_km": self.total_distance_km,
            "mean_distribution_time_s": self.mean_distribution_time_s,
            "success_rate_hz": self.success_rate_hz,
            "published_nested_time_s": self.published_nested_time_s,
            "controller_inclusive_time_s": self.controller_inclusive_time_s,
            "calculation_mode": self.calculation_mode,
            "exact_for_declared_model": self.exact_for_declared_model,
            "memory_coherence_time_s": self.memory_coherence_time_s,
            "coherence_to_service_time_ratio": self.coherence_to_service_time_ratio,
            "memory_regime": self.memory_regime,
            "node_memory_qubits": self.node_memory_qubits,
            "memory_slots_per_adjacent_edge_side": self.memory_slots_per_adjacent_edge_side,
            "assumptions": list(self.assumptions),
            "quality_restoration": self.quality_restoration,
        }


@dataclass(frozen=True)
class BellQualityEstimate:
    hop_count: int
    raw_elementary_bell_fidelity: float
    elementary_bell_fidelity: float
    local_swap_bell_fidelity: float
    elementary_pauli_error_distributions: tuple[
        tuple[float, float, float, float], ...
    ]
    pauli_error_distribution: tuple[float, float, float, float]
    end_to_end_bell_fidelity: float
    z_basis_qber: float
    maximal_chsh_s: float
    entangled_by_bell_diagonal_ppt_test: bool
    violates_chsh: bool
    storage_decoherence_included: bool
    source_equations: tuple[str, ...]
    quality_restoration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_count": self.hop_count,
            "raw_elementary_bell_fidelity": self.raw_elementary_bell_fidelity,
            "elementary_bell_fidelity": self.elementary_bell_fidelity,
            "local_swap_bell_fidelity": self.local_swap_bell_fidelity,
            "elementary_pauli_error_distributions": [
                {label: value for label, value in zip(PAULI_LABELS, distribution)}
                for distribution in self.elementary_pauli_error_distributions
            ],
            "pauli_error_distribution": {
                label: value for label, value in zip(PAULI_LABELS, self.pauli_error_distribution)
            },
            "end_to_end_bell_fidelity": self.end_to_end_bell_fidelity,
            "z_basis_qber": self.z_basis_qber,
            "maximal_chsh_s": self.maximal_chsh_s,
            "entangled_by_bell_diagonal_ppt_test": self.entangled_by_bell_diagonal_ppt_test,
            "violates_chsh": self.violates_chsh,
            "storage_decoherence_included": self.storage_decoherence_included,
            "source_equations": list(self.source_equations),
            "quality_restoration": self.quality_restoration,
        }


def load_repeater_service_profile(path: str | Path) -> RepeaterServiceProfile:
    with Path(path).open(encoding="utf-8") as handle:
        return RepeaterServiceProfile.from_dict(json.load(handle))


def repeater_service_profile_from_context(value: Any) -> RepeaterServiceProfile | None:
    if value is None:
        return None
    if isinstance(value, RepeaterServiceProfile):
        return value
    if isinstance(value, Mapping):
        return RepeaterServiceProfile.from_dict(value)
    raise RepeaterServiceError("repeater service profile must be a mapping or RepeaterServiceProfile")


def elementary_link_estimate(
    profile: RepeaterServiceProfile,
    edge_length_km: float,
) -> ElementaryLinkEstimate:
    """Mean elementary-link service time under the profile's cited rate model."""

    length = profile.validate_edge_length(edge_length_km)
    model = profile.rate_model
    kind = str(model["kind"])
    if kind == "empirical_ion_photon_node_component":
        raise RepeaterServiceError(
            f"profile {profile.profile_id!r} reports a one-node ion-photon component rate; "
            "it cannot be substituted for a two-node memory-memory elementary-link rate"
        )
    if kind == "cui_note8_midpoint_two_photon_projection":
        reference_length = float(model["reference_edge_length_km"])
        reference_rate = float(model["reference_success_rate_hz"])
        attenuation = float(model["fiber_attenuation_db_per_km"])
        # Cui Supplementary Note 8 defines two synchronized node-to-midpoint
        # arms.  Their combined fiber length is the neighboring-node edge
        # length, so the two-photon coincidence rate acquires one ordinary
        # power-transmission factor over the *total* edge-length difference.
        relative_transmission = 10.0 ** (
            -attenuation * (length - reference_length) / 10.0
        )
        rate = reference_rate * relative_transmission
        return ElementaryLinkEstimate(
            edge_length_km=length,
            attempt_period_s=None,
            success_probability_per_attempt=None,
            mean_distribution_time_s=1.0 / rate,
            success_rate_hz=rate,
            half_link_transmission=10.0 ** (-attenuation * length / 20.0),
            rate_model_kind=kind,
            source_equation=(
                "R(D)=R_ref*10^[-alpha_db_per_km*(D-D_ref)/10]"
            ),
            assumptions=(
                "the source-data two-node rate is an aggregate renewal rate after the Cui mode train",
                "the two node-to-midpoint arms have combined fiber length equal to the elementary edge length",
                "distance extrapolation changes loss and herald latency but not the source-data controller hardware",
            ),
        )
    if kind == "empirical_elementary_link":
        reference = float(model["reference_edge_length_km"])
        tolerance = float(model.get("reference_length_tolerance_km", 1e-9))
        if abs(length - reference) > tolerance:
            raise RepeaterServiceError(
                f"empirical profile {profile.profile_id!r} is measured only at "
                f"{reference:g} km; interpolation/extrapolation is not supplied"
            )
        rate = float(model["measured_success_rate_hz"])
        return ElementaryLinkEstimate(
            edge_length_km=length,
            attempt_period_s=None,
            success_probability_per_attempt=None,
            mean_distribution_time_s=1.0 / rate,
            success_rate_hz=rate,
            half_link_transmission=None,
            rate_model_kind=kind,
            source_equation="source-reported measured elementary-link rate",
            assumptions=("no interpolation or extrapolation away from the measured length",),
        )

    p0 = float(model["per_node_zero_loss_detection_probability"])
    gamma = float(model["attenuation_exponent_per_km"])
    overhead = float(model["attempt_overhead_s"])
    speed = float(model["fiber_speed_m_per_s"])
    period = length * 1000.0 / speed + overhead
    half_transmission = 10.0 ** (-gamma * length / 2.0)
    # This is the probability convention used by the source's 0.07 s
    # two-node/50 km statement.  The nested S12 approximation has its own
    # leading factor two, retained in route_service_estimate below.
    success = (p0 * half_transmission) ** 2
    mean = period / success
    return ElementaryLinkEstimate(
        edge_length_km=length,
        attempt_period_s=period,
        success_probability_per_attempt=success,
        mean_distribution_time_s=mean,
        success_rate_hz=1.0 / mean,
        half_link_transmission=half_transmission,
        rate_model_kind=kind,
        source_equation="T_link=(L/c'+T0)/[(P_link^0)^2 eta_t^2]",
        assumptions=(
            "per-node P_link^0 already aggregates photon generation and detection excluding fiber loss",
            "eta_t=10^(-gamma*L/2) is the half-link transmission",
            "the source's direct two-node prediction is retained without inserting an extra BSM factor",
        ),
    )


def quality_restoration_model(
    profile: RepeaterServiceProfile,
) -> BilateralRecurrenceModel | None:
    configured = profile.state_model.get("quality_restoration")
    if configured is None:
        return None
    try:
        return BilateralRecurrenceModel.from_mapping(
            _require_mapping("state_model.quality_restoration", configured)
        )
    except PurificationError as exc:
        raise RepeaterServiceError(str(exc)) from exc


def _parallel_two_mode_batch_rounds(success_probability: float) -> float:
    """E[max(G1,G2)] for iid geometric raw-link completion rounds."""

    probability = _probability("raw-link success probability", success_probability)
    if probability <= 0.0:
        raise RepeaterServiceError("raw-link success probability must be > 0")
    return 2.0 / probability - 1.0 / (probability * (2.0 - probability))


def _sacrificial_wait_coherence(
    profile: RepeaterServiceProfile,
    raw_link: ElementaryLinkEstimate,
) -> float:
    """Average C(|G1-G2| Delta) for the older of two parallel raw pairs.

    For two iid geometric completion rounds with q=1-p,
    ``P(K=0)=p/(2-p)`` and ``P(K=k)=2*p*q^k/(2-p)`` for k>=1.  Averaging the
    selected memory's declared coherence law over that exact gap distribution
    accounts for the sacrificial pair waiting while the younger retained pair
    is still being generated.
    """

    storage = profile.memory_model["storage_decoherence_model"]
    kind = str(storage.get("kind", "none"))
    if kind == "none":
        return 1.0
    coherence_time = float(storage["coherence_time_s"])
    if raw_link.attempt_period_s is None or raw_link.success_probability_per_attempt is None:
        if str(profile.rate_model.get("completion_time_model", "")) != (
            "empirical_exponential_renewal"
        ):
            raise RepeaterServiceError(
                "purification wait decoherence requires either an attempt decomposition "
                "or an empirical exponential renewal declaration"
            )
        rate = 1.0 / float(raw_link.mean_distribution_time_s)
        if kind == "exponential_dephasing":
            return _probability(
                "average sacrificial wait coherence",
                rate / (rate + 1.0 / coherence_time),
            )
        if kind != "gaussian_dephasing":
            raise RepeaterServiceError(
                f"unsupported purification wait storage model {kind!r}"
            )
        # For iid exponential completion times, |X-Y| is exponential with
        # the same rate.  This is the exact Gaussian-coherence average
        # integral int_0^inf lambda exp(-lambda*t-(t/tau)^2) dt.
        scaled = rate * coherence_time
        value = (
            0.5
            * math.sqrt(math.pi)
            * scaled
            * math.exp(0.25 * scaled * scaled)
            * math.erfc(0.5 * scaled)
        )
        return _probability("average sacrificial wait coherence", value)
    period = float(raw_link.attempt_period_s)
    probability = float(raw_link.success_probability_per_attempt)
    q = 1.0 - probability
    value = probability / (2.0 - probability)
    k = 1
    residual = 2.0 * q / (2.0 - probability)
    while residual > 1e-14:
        elapsed = k * period
        if kind == "gaussian_dephasing":
            coherence = math.exp(-((elapsed / coherence_time) ** 2))
        elif kind == "exponential_dephasing":
            coherence = math.exp(-(elapsed / coherence_time))
        else:
            raise RepeaterServiceError(
                f"unsupported purification wait storage model {kind!r}"
            )
        mass = 2.0 * probability * (q ** k) / (2.0 - probability)
        value += mass * coherence
        residual -= mass
        k += 1
        if k > 10_000_000:
            raise RepeaterServiceError(
                "purification wait-coherence series did not converge"
            )
    return _probability("average sacrificial wait coherence", value)


def purified_elementary_link_estimate(
    profile: RepeaterServiceProfile,
    edge_length_km: float,
) -> PurifiedElementaryLinkEstimate | None:
    """Resource, timing, acceptance, and state result for one purified edge."""

    model = quality_restoration_model(profile)
    if model is None:
        return None
    cache_key = (
        profile.profile_id,
        json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":")),
        float(edge_length_km),
    )
    cached = _PURIFIED_LINK_ESTIMATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw_link = elementary_link_estimate(profile, edge_length_km)
    if model.parallel_raw_pair_generation_modes_per_edge != 2:
        raise RepeaterServiceError(
            "only two-mode parallel raw-pair batches are implemented"
        )
    if raw_link.attempt_period_s is None or raw_link.success_probability_per_attempt is None:
        if str(profile.rate_model.get("completion_time_model", "")) != (
            "empirical_exponential_renewal"
        ):
            raise RepeaterServiceError(
                "bilateral recurrence requires either a raw attempt decomposition "
                "or an empirical exponential renewal declaration"
            )
        batch_time = 1.5 * raw_link.mean_distribution_time_s
        batch_equation = "E[max(X1,X2)]=3/(2*lambda) for iid exponential renewals"
    else:
        batch_time = (
            _parallel_two_mode_batch_rounds(raw_link.success_probability_per_attempt)
            * raw_link.attempt_period_s
        )
        batch_equation = (
            "E[max(G1,G2)]=2/p-1/[p(2-p)] for two iid geometric generation modes"
        )
    comparison = (
        float(edge_length_km) * 1000.0 / model.classical_comparison_speed_m_per_s
    )
    cycle = batch_time + model.local_operation_duration_s + comparison
    raw_fidelity = elementary_bell_fidelity(profile)
    raw_distribution = (raw_fidelity, 0.0, 1.0 - raw_fidelity, 0.0)
    raw_density = bell_diagonal_density(raw_distribution, target_label="psi_minus")
    sacrificial_coherence = _sacrificial_wait_coherence(profile, raw_link)
    purification = bilateral_phase_error_recurrence(
        raw_density,
        raw_density,
        model,
        target_label="psi_minus",
        sacrificial_wait_coherence=sacrificial_coherence,
    )
    lanes = int(profile.hardware.get("parallel_purification_lanes_per_edge", 1))
    per_lane_mean = cycle / purification.success_probability
    # Independent continuously operated lanes have exactly additive steady-
    # state renewal rates.  The route controller consumes that aggregate edge
    # service rate; the rolling reservation model separately assigns the
    # per-lane empty time to each physical output slot.
    mean = per_lane_mean / lanes
    wait_equation = (
        "|X1-X2|~Exponential(lambda) for iid exponential raw-pair renewals"
        if raw_link.attempt_period_s is None
        else "P(|G1-G2|=0)=p/(2-p); P(k>=1)=2p(1-p)^k/(2-p)"
    )
    result = PurifiedElementaryLinkEstimate(
        edge_length_km=float(edge_length_km),
        raw_link=raw_link,
        raw_pairs_consumed_per_cycle=model.raw_pairs_per_round,
        parallel_generation_modes=model.parallel_raw_pair_generation_modes_per_edge,
        parallel_purification_lanes=lanes,
        mean_raw_pair_batch_time_s=batch_time,
        local_operation_duration_s=model.local_operation_duration_s,
        classical_comparison_duration_s=comparison,
        mean_cycle_duration_s=cycle,
        purification_success_probability=purification.success_probability,
        per_lane_mean_distribution_time_s=per_lane_mean,
        mean_distribution_time_s=mean,
        success_rate_hz=1.0 / mean,
        sacrificial_wait_coherence=sacrificial_coherence,
        output_pauli_error_distribution=purification.bell_probabilities,
        source_equations=(
            "Bennett/Deutsch-family bilateral recurrence phase-parity circuit",
            batch_equation,
            wait_equation,
        ),
        assumptions=(
            "two raw-pair optical generation modes are independently resourced and run in parallel",
            "parallel purification lanes have independent raw optical and memory resources; their steady-state renewal rates add",
            "the younger completed raw pair is selected as the retained control pair",
            "Alice and Bob local gates and measurements execute in parallel",
            "one one-way edge-length classical message is required to confirm parity acceptance",
            "failed purification consumes both raw pairs and restarts the elementary-edge cycle",
        ),
    )
    if len(_PURIFIED_LINK_ESTIMATE_CACHE) >= 512:
        _PURIFIED_LINK_ESTIMATE_CACHE.pop(next(iter(_PURIFIED_LINK_ESTIMATE_CACHE)))
    _PURIFIED_LINK_ESTIMATE_CACHE[cache_key] = result
    return result


def _mean_max_two_exponentials(left_mean_s: float, right_mean_s: float) -> float:
    """E[max(X,Y)] for independent exponentials with the supplied means."""

    left = _positive("left segment mean", left_mean_s)
    right = _positive("right segment mean", right_mean_s)
    return left + right - (left * right) / (left + right)


def _balanced_controller_time_s(
    elementary: Sequence[ElementaryLinkEstimate | PurifiedElementaryLinkEstimate],
    *,
    swap_duration_s: float,
    swap_success_probability: float,
) -> float:
    if len(elementary) == 1:
        return float(elementary[0].mean_distribution_time_s)

    def recurse(start: int, stop: int) -> float:
        if stop - start == 1:
            return float(elementary[start].mean_distribution_time_s)
        middle = start + (stop - start) // 2
        left = recurse(start, middle)
        right = recurse(middle, stop)
        return (
            _mean_max_two_exponentials(left, right) + float(swap_duration_s)
        ) / float(swap_success_probability)

    # Supplementary Eq. S12 is exactly twice the homogeneous exponential
    # balanced-tree recurrence.  Retaining that normalization makes the
    # general adjacent balanced tree conservative and reproduces the paper's
    # 16-link value instead of silently using the more optimistic recurrence.
    return 2.0 * recurse(0, len(elementary))


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def _homogeneous_elementary(estimates: Sequence[ElementaryLinkEstimate]) -> bool:
    first = estimates[0]
    return all(
        math.isclose(row.mean_distribution_time_s, first.mean_distribution_time_s, rel_tol=1e-12, abs_tol=1e-15)
        for row in estimates[1:]
    )


def route_service_estimate(
    profile: RepeaterServiceProfile,
    edge_lengths_km: Sequence[float],
) -> RouteServiceEstimate:
    """Scalable adjacent balanced-tree service estimate for an Alice route.

    Homogeneous power-of-two Krutyanskiy routes are evaluated with the cited
    Supplementary Eq. S12.  Other route lengths use the same factor-two
    normalization with a balanced exponential-completion recurrence.  That
    generalization is a labelled controller approximation, not a claim made by
    the paper.
    """

    if not edge_lengths_km:
        raise RepeaterServiceError("at least one elementary edge is required")
    raw_estimates = [elementary_link_estimate(profile, value) for value in edge_lengths_km]
    restoration = quality_restoration_model(profile)
    purified_estimates = (
        [
            purified_elementary_link_estimate(profile, value)
            for value in edge_lengths_km
        ]
        if restoration is not None else None
    )
    estimates: list[ElementaryLinkEstimate | PurifiedElementaryLinkEstimate]
    if purified_estimates is not None:
        if any(value is None for value in purified_estimates):
            raise RepeaterServiceError("quality-restoration estimate construction failed")
        estimates = [
            value for value in purified_estimates if value is not None
        ]
    else:
        estimates = list(raw_estimates)
    if (
        profile.rate_model["kind"] == "empirical_elementary_link"
        and len(estimates) != 1
        and str(profile.controller_model.get("empirical_multilink_completion_model", ""))
        != "independent_exponential_balanced_tree"
    ):
        raise RepeaterServiceError(
            "an empirical elementary-link rate without a controller equation cannot be composed into a route"
        )
    swap_duration = float(profile.controller_model.get("local_swap_duration_s", 0.0))
    swap_success = float(profile.controller_model.get("local_swap_success_probability", 1.0))
    inclusive = _balanced_controller_time_s(
        estimates,
        swap_duration_s=swap_duration,
        swap_success_probability=swap_success,
    )
    published: float | None = None
    exact = False
    if (
        profile.rate_model["kind"] == "krutyanskiy_midpoint_two_photon"
        and restoration is None
        and _is_power_of_two(len(estimates))
        and _homogeneous_elementary(raw_estimates)
    ):
        nesting_levels = int(math.log2(len(estimates)))
        base = raw_estimates[0].mean_distribution_time_s
        published = base if len(estimates) == 1 else 2.0 * base * (1.5 ** nesting_levels)
        if swap_duration == 0.0 and swap_success == 1.0:
            inclusive = published
        mode = "supplementary_equation_S12"
        exact = True
    elif len(estimates) == 1:
        inclusive = estimates[0].mean_distribution_time_s
        if restoration is None:
            published = inclusive
            mode = "elementary_link_source_rate"
        else:
            published = None
            mode = "resource_accounted_purified_elementary_link_renewal"
        exact = True
    else:
        mode = "balanced_exponential_completion_generalization_of_S12"

    storage = profile.memory_model["storage_decoherence_model"]
    coherence = (
        float(storage["coherence_time_s"])
        if storage.get("kind") != "none" and storage.get("coherence_time_s") is not None
        else None
    )
    ratio = coherence / inclusive if coherence is not None else None
    if ratio is None:
        memory_regime = "not_quantified"
    elif ratio >= 10.0:
        memory_regime = "tau_at_least_10x_mean_service_time"
    elif ratio >= 1.0:
        memory_regime = "tau_comparable_to_mean_service_time"
    else:
        memory_regime = "mean_service_time_exceeds_tau"
    assumptions = [
        "adjacent elementary links are generated asynchronously and in parallel",
        "a successful segment is stored while its partner segment is prepared",
        "swapping consumes both input segments and failed swaps restart that segment",
        "the general heterogeneous/non-power-of-two path uses exponential segment-completion times",
        "no encoding, queue contention, or Eve disruption is included",
    ]
    restoration_payload = None
    if restoration is None:
        assumptions.append("no purification is included")
    else:
        assumptions.extend((
            "one elementary-link phase-parity recurrence round precedes every swap",
            "purification acceptance and raw-pair generation time are included in edge renewal time",
        ))
        assert purified_estimates is not None
        restoration_payload = {
            "model": restoration.to_dict(),
            "placement": restoration.placement,
            "per_edge": [
                value.to_dict() for value in purified_estimates if value is not None
            ],
            "rate_semantics": (
                "the balanced asynchronous route controller consumes purified-pair "
                "renewal times, not raw optical-attempt rates"
            ),
        }
    return RouteServiceEstimate(
        hop_count=len(estimates),
        total_distance_km=math.fsum(float(value) for value in edge_lengths_km),
        mean_distribution_time_s=inclusive,
        success_rate_hz=1.0 / inclusive,
        published_nested_time_s=published,
        controller_inclusive_time_s=inclusive,
        calculation_mode=mode,
        exact_for_declared_model=exact,
        memory_coherence_time_s=coherence,
        coherence_to_service_time_ratio=ratio,
        memory_regime=memory_regime,
        node_memory_qubits=int(profile.hardware["memory_qubits_per_node"]),
        memory_slots_per_adjacent_edge_side=int(
            profile.hardware["memory_slots_per_adjacent_edge_side"]
        ),
        assumptions=tuple(assumptions),
        quality_restoration=restoration_payload,
    )


def elementary_bell_fidelity(profile: RepeaterServiceProfile) -> float:
    model = profile.state_model
    if model["kind"] == "detector_conditioned_fock_bsm":
        return _probability(
            "state_model.service_estimate_raw_bell_fidelity",
            model["service_estimate_raw_bell_fidelity"],
        )
    if model["kind"] == "pauli_twirled_bell_fidelity":
        return _probability(
            "state_model.reported_bell_fidelity",
            model["reported_bell_fidelity"],
        )
    if model["kind"] == "bell_diagonal_correlations":
        return psi_minus_distribution_from_model(model)["I"]
    if model["kind"] != "krutyanskiy_equations_S13_S15":
        raise RepeaterServiceError(
            "a scalar reported Bell fidelity does not define an executable process channel; "
            "the profile is rate/provenance-only for histogram state generation"
        )
    f0 = float(model["ion_photon_bell_fidelity"])
    f_photons = float(model["photonic_bsm_fidelity"])
    visibility = 2.0 * f_photons - 1.0
    return _probability(
        "derived elementary Bell fidelity",
        0.5 * (1.0 + visibility * ((1.0 - 2.0 * f0) ** 2)),
    )


def _pauli_convolution(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    out = [0.0, 0.0, 0.0, 0.0]
    for left_code, left_probability in enumerate(left):
        for right_code, right_probability in enumerate(right):
            out[left_code ^ right_code] += float(left_probability) * float(right_probability)
    total = math.fsum(out)
    return tuple(value / total for value in out)  # type: ignore[return-value]


def route_bell_quality_estimate(
    profile: RepeaterServiceProfile,
    hop_count: int,
    edge_lengths_km: Sequence[float] | None = None,
) -> BellQualityEstimate:
    """Bell-diagonal chain quality from the paper's S13--S15 maps.

    Storage decoherence is intentionally excluded here because the source's
    800 km fidelity calculation also assumes ``tau >> T_tot``.  The histogram
    adapter applies the profile's storage model to explicit memory ages.
    """

    hops = int(hop_count)
    if hops < 1:
        raise RepeaterServiceError("hop_count must be >= 1")
    raw_edge_fidelity = elementary_bell_fidelity(profile)
    state_kind = str(profile.state_model["kind"])
    local_bsm = local_bell_measurement_model(profile)
    if state_kind in {
        "pauli_twirled_bell_fidelity",
        "bell_diagonal_correlations",
        "detector_conditioned_fock_bsm",
    }:
        if hops > 1 and local_bsm is None:
            raise RepeaterServiceError(
                f"profile {profile.profile_id!r} requires an explicit local-swap "
                "process for multi-hop execution"
            )
        swap_fidelity = (
            local_bsm.frame_error_distribution["I"] if local_bsm is not None else 1.0
        )
    else:
        swap_fidelity = float(profile.state_model["local_swap_bell_fidelity"])
    restoration = quality_restoration_model(profile)
    restoration_payload = None
    if restoration is None:
        if state_kind == "pauli_twirled_bell_fidelity":
            other = (1.0 - raw_edge_fidelity) / 3.0
            edge_errors = tuple(
                (raw_edge_fidelity, other, other, other)
                for _index in range(hops)
            )
        elif state_kind == "bell_diagonal_correlations":
            calibrated = psi_minus_distribution_from_model(profile.state_model)
            one_edge = tuple(calibrated[label] for label in PAULI_LABELS)
            edge_errors = tuple(one_edge for _index in range(hops))
        else:
            edge_errors = tuple(
                (raw_edge_fidelity, 0.0, 1.0 - raw_edge_fidelity, 0.0)
                for _index in range(hops)
            )
        edge_fidelity = raw_edge_fidelity
    else:
        if edge_lengths_km is None:
            raise RepeaterServiceError(
                "a quality-restored route requires explicit edge_lengths_km because raw-pair "
                "wait decoherence and classical comparison time are length dependent"
            )
        lengths = [float(value) for value in edge_lengths_km]
        if len(lengths) != hops:
            raise RepeaterServiceError(
                "edge_lengths_km must contain one value per route hop"
            )
        estimates = [
            purified_elementary_link_estimate(profile, value) for value in lengths
        ]
        if any(value is None for value in estimates):
            raise RepeaterServiceError("quality-restoration estimate construction failed")
        restored = [value for value in estimates if value is not None]
        edge_errors = tuple(value.output_pauli_error_distribution for value in restored)
        edge_fidelity = math.fsum(value[0] for value in edge_errors) / hops
        restoration_payload = {
            "model": restoration.to_dict(),
            "edge_lengths_km": lengths,
            "per_edge": [value.to_dict() for value in restored],
            "storage_scope": (
                "additional older-sacrificial-pair wait decoherence is averaged into each "
                "recurrence input; post-acceptance live-pair storage remains excluded here "
                "and is applied by the histogram age model"
            ),
        }
    if local_bsm is not None:
        swap_error = tuple(
            local_bsm.frame_error_distribution[label] for label in PAULI_LABELS
        )
    else:
        swap_error = (
            swap_fidelity,
            (1.0 - swap_fidelity) / 3.0,
            (1.0 - swap_fidelity) / 3.0,
            (1.0 - swap_fidelity) / 3.0,
        )
    distribution: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    for edge_error in edge_errors:
        distribution = _pauli_convolution(distribution, edge_error)
    for _index in range(hops - 1):
        distribution = _pauli_convolution(distribution, swap_error)
    p_i, p_x, p_z, p_y = distribution
    correlations = (
        -p_i - p_x + p_z + p_y,
        -p_i + p_x + p_z - p_y,
        -p_i + p_x - p_z + p_y,
    )
    squares = sorted(value * value for value in correlations)
    chsh = 2.0 * math.sqrt(squares[-1] + squares[-2])
    fidelity = p_i
    if state_kind == "pauli_twirled_bell_fidelity":
        source_equations = (
            "Liu et al. measured Bell fidelity followed by the declared "
            "fidelity-preserving bilateral Clifford isotropic twirl",
        )
    elif state_kind == "bell_diagonal_correlations":
        source_equations = (
            str(profile.state_model.get("correlation_source", "source-reported Bell correlations")),
            str(profile.state_model.get("completion_semantics", "declared Bell-diagonal completion")),
        )
    elif state_kind == "detector_conditioned_fock_bsm":
        source_equations = (
            "detector-conditioned Fock/PNR midpoint-BSM density instrument",
            str(
                profile.state_model.get(
                    "service_estimate_calibration",
                    "declared detector-conditioned service-estimate calibration",
                )
            ),
        )
    else:
        source_equations = (
            "Krutyanskiy et al. Supplementary Eqs. S13--S14 "
            "(elementary phase-error mixture)",
            "Krutyanskiy et al. Supplementary Eq. S15 "
            "(local single-qubit depolarizing map)",
        )
    if restoration is not None:
        source_equations += (
            "Bennett/Deutsch-family bilateral phase-parity recurrence with "
            "explicit noisy local operations",
        )
    return BellQualityEstimate(
        hop_count=hops,
        raw_elementary_bell_fidelity=raw_edge_fidelity,
        elementary_bell_fidelity=edge_fidelity,
        local_swap_bell_fidelity=swap_fidelity,
        elementary_pauli_error_distributions=edge_errors,
        pauli_error_distribution=distribution,
        end_to_end_bell_fidelity=fidelity,
        z_basis_qber=p_x + p_y,
        maximal_chsh_s=chsh,
        entangled_by_bell_diagonal_ppt_test=max(distribution) > 0.5 + 1e-15,
        violates_chsh=chsh > 2.0 + 1e-12,
        storage_decoherence_included=False,
        source_equations=source_equations,
        quality_restoration=restoration_payload,
    )


def profile_elementary_state_channel(profile: RepeaterServiceProfile) -> dict[str, Any]:
    if profile.state_model["kind"] == "detector_conditioned_fock_bsm":
        raise RepeaterServiceError(
            "the detector-conditioned Fock elementary state is the active instrument; "
            "it must not receive a second scalar post-channel"
        )
    fidelity = elementary_bell_fidelity(profile)
    return {
        "kind": "single_qubit_phase_flip",
        "qubit": 1,
        "phase_flip_probability": 1.0 - fidelity,
        "bell_fidelity": fidelity,
        "source": "Krutyanskiy et al. Supplementary Eqs. S13--S14",
        "semantics": (
            "rho=F|Psi-><Psi-|+(1-F)|Psi+><Psi+|; applied after the "
            "detector-conditioned elementary-link instrument"
        ),
    }


def profile_elementary_pair_instrument(
    profile: RepeaterServiceProfile,
) -> dict[str, Any] | None:
    """Conditional accepted-pair instrument for a demonstrated remote-memory link.

    This is deliberately distinct from ``profile_elementary_state_channel``:
    it replaces the two-photon Fock elementary instrument instead of applying
    another channel after it.  The isotropic Bell distribution is the explicit
    output of the profile's declared fidelity-preserving bilateral twirl.
    """

    kind = str(
        profile.histogram_adapter.get(
            "elementary_link_instrument_kind", "fock_midpoint_two_photon"
        )
    )
    if kind != "empirical_remote_memory_pair":
        return None
    state_kind = str(profile.state_model["kind"])
    if state_kind == "pauli_twirled_bell_fidelity":
        fidelity = elementary_bell_fidelity(profile)
        other = (1.0 - fidelity) / 3.0
        distribution = {"I": fidelity, "X": other, "Z": other, "Y": other}
        completion = "declared fidelity-preserving bilateral isotropic twirl"
    elif state_kind == "bell_diagonal_correlations":
        distribution = psi_minus_distribution_from_model(profile.state_model)
        fidelity = distribution["I"]
        completion = str(profile.state_model.get("completion_semantics", "measured Bell correlations"))
    else:
        raise RepeaterServiceError("empirical remote-memory pair has no executable Bell state")
    return {
        "kind": kind,
        "target_bell_state": "psi_minus",
        "pauli_error_distribution": distribution,
        "reported_bell_fidelity": fidelity,
        "state_model_kind": state_kind,
        "state_completion": completion,
        "twirling_map": profile.state_model.get("twirling_map"),
        "source": profile.source.get("citation"),
        "semantics": (
            "conditional remote memory-memory pair after herald-frame correction "
            "and the declared fidelity-preserving bilateral isotropic twirl; the "
            "measured renewal rate and rolling availability carry delivery mass"
        ),
    }


def profile_quality_restoration(
    profile: RepeaterServiceProfile,
    edge_length_km: float,
) -> dict[str, Any] | None:
    model = quality_restoration_model(profile)
    if model is None:
        return None
    estimate = purified_elementary_link_estimate(profile, edge_length_km)
    if estimate is None:
        raise RepeaterServiceError("quality-restoration estimate construction failed")
    return {
        "kind": "bilateral_phase_error_recurrence",
        "target_bell_state": "psi_minus",
        "model": model.to_dict(),
        "edge_estimate": estimate.to_dict(),
        "sacrificial_wait_coherence": estimate.sacrificial_wait_coherence,
        "conditional_success_probability_for_profile_raw_state": (
            estimate.purification_success_probability
        ),
        "kernel_semantics": (
            "the histogram applies the full two-copy noisy LOCC density-matrix "
            "instrument to detector-conditioned raw elementary pairs; this profile-only "
            "success probability is used by the asynchronous renewal-rate model"
        ),
    }


def profile_swap_state_channel(profile: RepeaterServiceProfile) -> dict[str, Any]:
    local_bsm = local_bell_measurement_model(profile)
    if local_bsm is not None:
        return {
            "kind": "bell_label_confusion_only",
            "qubit": 1,
            "bell_fidelity": local_bsm.frame_error_distribution["I"],
            "depolarizing_probability": 0.0,
            "source": list(local_bsm.source),
            "semantics": (
                "local Bell-transform and readout errors enter through the explicit "
                "Bell-label confusion override, not a second output-state channel"
            ),
        }
    if profile.state_model["kind"] != "krutyanskiy_equations_S13_S15":
        raise RepeaterServiceError("profile has no executable local-swap channel")
    fidelity = float(profile.state_model["local_swap_bell_fidelity"])
    return {
        "kind": "single_qubit_bell_depolarizing",
        "qubit": 1,
        "bell_fidelity": fidelity,
        "depolarizing_probability": 1.0 - fidelity,
        "source": "Krutyanskiy et al. Supplementary Eq. S15",
        "semantics": "F*rho+(1-F)/3*(XrhoX+YrhoY+ZrhoZ) on one output qubit",
    }


def local_bell_measurement_model(
    profile: RepeaterServiceProfile,
) -> LocalBellMeasurementModel | None:
    configured = profile.state_model.get("local_swap_model")
    if configured is None:
        return None
    try:
        return LocalBellMeasurementModel.from_mapping(
            _require_mapping("state_model.local_swap_model", configured)
        )
    except LocalBellMeasurementError as exc:
        raise RepeaterServiceError(str(exc)) from exc


def profile_summary(profile: RepeaterServiceProfile) -> dict[str, Any]:
    state_executable = profile.state_model["kind"] in {
        "krutyanskiy_equations_S13_S15",
        "detector_conditioned_fock_bsm",
        "pauli_twirled_bell_fidelity",
        "bell_diagonal_correlations",
    }
    restoration = quality_restoration_model(profile)
    physical_slots = int(profile.hardware["memory_slots_per_adjacent_edge_side"])
    output_slots = int(
        profile.hardware.get(
            "purified_output_slots_per_adjacent_edge_side", physical_slots
        )
    )
    return {
        "profile_id": profile.profile_id,
        "scope": profile.scope,
        "claim_tier": profile.claim_tier,
        "architecture": profile.architecture,
        "citation": profile.source.get("citation"),
        "rate_model_kind": profile.rate_model["kind"],
        "elementary_link_instrument_kind": profile.histogram_adapter.get(
            "elementary_link_instrument_kind", "fock_midpoint_two_photon"
        ),
        "histogram_state_channel_executable": state_executable,
        "edge_length_range_km": [
            float(profile.validity["edge_length_min_km"]),
            float(profile.validity["edge_length_max_km"]),
        ],
        "memory_qubits_per_node": int(profile.hardware["memory_qubits_per_node"]),
        "communication_qubits_per_node": profile.hardware.get(
            "communication_qubits_per_node"
        ),
        "memory_slots_per_adjacent_edge_side": physical_slots,
        "rolling_output_pair_slots_per_adjacent_edge_side": output_slots,
        "parallel_purification_lanes_per_edge": profile.hardware.get(
            "parallel_purification_lanes_per_edge"
        ),
        "quality_restoration_active": restoration is not None,
        "quality_restoration_kind": (
            "bilateral_phase_error_recurrence" if restoration is not None else None
        ),
    }
