"""Core constants and immutable histogram model objects."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = "e91_game_turn_histogram.v6"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "physical"
DEFAULT_LENGTH_M = 1000.0
DEFAULT_DEPHASING_THETA = math.pi / 8.0
DEFAULT_FOCK_CUTOFF = 3
TOL = 1e-9

PHYSICAL_PATHS = ("sequence_bk_reference", "fock_pnr_snspd")
SOURCE_MODELS = ("memory_emission", "spdc")
DETECTOR_KINDS = ("repository_default", "threshold_snspd", "pnr_snspd", "ideal_pnr")

SOURCE_FIELDS = ("frequency", "wavelength", "bandwidth", "mean_photon_num", "phase_error")
FIBER_FIELDS = ("attenuation", "polarization_fidelity", "light_speed", "frequency")
DETECTOR_FIELDS = ("efficiency", "dark_count", "count_rate", "time_resolution")
MEMORY_FIELDS = (
    "fidelity",
    "frequency",
    "efficiency",
    "coherence_time",
    "wavelength",
    "memory_process_model",
    "memory_storage_decoherence_model",
    "memory_readout_model",
    "detected_decay_lifetime_s",
    "endpoint_analysis_rotation_model",
)
SWAP_FIELDS = ("phase_error", "success_probability")

DELIVERY_STATUS = {"delivered": 0, "no_delivery": 1}
FAILURE_STAGE = {
    "none": 0,
    "elementary_generation_or_channel_failure": 1,
    "swap_failure": 2,
    "endpoint_detection_failure": 3,
    "stale_pair_cutoff": 4,
    "memory_decay_detected": 5,
}
SIFTING_CLASS = {"key": 0, "chsh": 1, "discarded": 2, "no_delivery": 3}
FRAME_LABELS = ("I", "X", "Z", "Y")
FRAME_ID = {label: idx for idx, label in enumerate(FRAME_LABELS)}
FRAME_BITS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
BITS_FRAME = {bits: label for label, bits in FRAME_BITS.items()}
BELL_LABELS = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
DETECTOR_CLICK_RECORD_NOT_EMITTED = "not_emitted"
DETECTOR_CLICK_RECORD_NO_DELIVERY = "no_delivered_memory_pair"
# Public Pauli frame that expresses the SeQUeNCe Barrett-Kok phi_plus target in
# this generator's E91 psi_minus endpoint convention: (I x Y)|phi+> ~ |psi->.
BK_REFERENCE_ENDPOINT_CONVENTION_FRAME = "Y"

I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULI_BY_LABEL = {"I": I2, "X": X, "Y": Y, "Z": Z}


LEGACY_MOVE_ID_REPLACEMENTS = (
    ("no_attack_miss", "no_move_baseline"),
    ("edge_intercept_resend", "edge_measure_resend"),
    ("swap_denial", "swap_block"),
)
LEGACY_MOVE_KIND_ALIASES = {
    "edge_intercept_resend": "edge_measure_resend",
    "swap_denial": "swap_block",
}


def canonical_move_id(value: str) -> str:
    out = str(value)
    for legacy, canonical in LEGACY_MOVE_ID_REPLACEMENTS:
        out = out.replace(legacy, canonical)
    return out


def canonical_move_kind(value: str) -> str:
    return LEGACY_MOVE_KIND_ALIASES.get(str(value), str(value))


@dataclass(frozen=True, init=False)
class EveMove:
    move_id: str
    move_kind: str
    target_kind: str = "none"
    target_id: str = ""
    memory_side: str = "both"
    target_active: bool = False
    histogram_source: str = "active"
    loss_multiplier: float = 1.0
    drop_probability: float = 0.0
    basis_choice: str = "Z"
    edge_measure_basis_label: str = "Z"
    edge_measure_probability: float = 1.0
    measurement_probability: float = 1.0
    memory_fidelity_multiplier: float = 1.0
    memory_depolarizing_probability: float = 0.0
    memory_dephasing_probability: float = 0.0
    dephasing_theta: float = DEFAULT_DEPHASING_THETA
    target_memory_side: str = "both"
    information_probability: float = 1.0

    def __init__(
        self,
        move_id: str | None = None,
        move_kind: str | None = None,
        *,
        attack_id: str | None = None,
        attack_kind: str | None = None,
        target_kind: str = "none",
        target_id: str = "",
        memory_side: str = "both",
        target_active: bool = False,
        histogram_source: str = "active",
        loss_multiplier: float = 1.0,
        drop_probability: float = 0.0,
        basis_choice: str = "Z",
        edge_measure_basis_label: str | None = None,
        edge_measure_probability: float | None = None,
        intercept_basis_label: str | None = None,
        intercept_probability: float | None = None,
        measurement_probability: float = 1.0,
        memory_fidelity_multiplier: float = 1.0,
        memory_depolarizing_probability: float = 0.0,
        memory_dephasing_probability: float = 0.0,
        dephasing_theta: float = DEFAULT_DEPHASING_THETA,
        target_memory_side: str = "both",
        information_probability: float = 1.0,
    ) -> None:
        chosen_id = move_id if move_id is not None else attack_id
        chosen_kind = move_kind if move_kind is not None else attack_kind
        if chosen_id is None:
            chosen_id = "no_move_baseline"
        if chosen_kind is None:
            chosen_kind = "none"
        measure_basis = (
            edge_measure_basis_label
            if edge_measure_basis_label is not None else
            intercept_basis_label
            if intercept_basis_label is not None else
            "Z"
        )
        measure_probability = (
            edge_measure_probability
            if edge_measure_probability is not None else
            intercept_probability
            if intercept_probability is not None else
            1.0
        )
        if histogram_source == "no_attack_miss":
            histogram_source = "no_move_baseline"
        object.__setattr__(self, "move_id", canonical_move_id(chosen_id))
        object.__setattr__(self, "move_kind", canonical_move_kind(chosen_kind))
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "memory_side", memory_side)
        object.__setattr__(self, "target_active", bool(target_active))
        object.__setattr__(self, "histogram_source", histogram_source)
        object.__setattr__(self, "loss_multiplier", float(loss_multiplier))
        object.__setattr__(self, "drop_probability", float(drop_probability))
        object.__setattr__(self, "basis_choice", basis_choice)
        object.__setattr__(self, "edge_measure_basis_label", measure_basis)
        object.__setattr__(self, "edge_measure_probability", float(measure_probability))
        object.__setattr__(self, "measurement_probability", float(measurement_probability))
        object.__setattr__(self, "memory_fidelity_multiplier", float(memory_fidelity_multiplier))
        object.__setattr__(self, "memory_depolarizing_probability", float(memory_depolarizing_probability))
        object.__setattr__(self, "memory_dephasing_probability", float(memory_dephasing_probability))
        object.__setattr__(self, "dephasing_theta", float(dephasing_theta))
        object.__setattr__(self, "target_memory_side", target_memory_side)
        object.__setattr__(self, "information_probability", float(information_probability))

    @property
    def attack_id(self) -> str:
        """Compatibility alias for older helper code."""
        return self.move_id

    @property
    def attack_kind(self) -> str:
        """Compatibility alias for older helper code."""
        return self.move_kind

    @property
    def intercept_basis_label(self) -> str:
        """Compatibility alias for the edge measure-resend basis label."""
        return self.edge_measure_basis_label

    @property
    def intercept_probability(self) -> float:
        """Compatibility alias for the edge measure-resend probability."""
        return self.edge_measure_probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_id": self.move_id,
            "move_kind": self.move_kind,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "memory_side": self.memory_side,
            "target_active": self.target_active,
            "histogram_source": self.histogram_source,
            "loss_multiplier": self.loss_multiplier,
            "drop_probability": self.drop_probability,
            "basis_choice": self.basis_choice,
            "edge_measure_basis_label": self.edge_measure_basis_label,
            "edge_measure_probability": self.edge_measure_probability,
            "measurement_probability": self.measurement_probability,
            "memory_fidelity_multiplier": self.memory_fidelity_multiplier,
            "memory_depolarizing_probability": self.memory_depolarizing_probability,
            "memory_dephasing_probability": self.memory_dephasing_probability,
            "dephasing_theta": self.dephasing_theta,
            "target_memory_side": self.target_memory_side,
            "information_probability": self.information_probability,
        }


HistogramAttack = EveMove


@dataclass
class Branch:
    probability: float
    rho: np.ndarray
    frame: str = "I"
    eve_observation: str = "none"
    public_bsm_record: str = "none"
    detector_click_record: str = "not_recorded"

    def copy(self) -> "Branch":
        return Branch(
            probability=float(self.probability),
            rho=np.array(self.rho, dtype=complex, copy=True),
            frame=self.frame,
            eve_observation=self.eve_observation,
            public_bsm_record=self.public_bsm_record,
            detector_click_record=self.detector_click_record,
        )


@dataclass(frozen=True)
class ModelContext:
    physical_path: str = "sequence_bk_reference"
    source_model: str = "memory_emission"
    detector_kind: str = "repository_default"
    fock_cutoff: int = DEFAULT_FOCK_CUTOFF
    detector_profile_path: str | None = None
    detector_profile: dict[str, Any] | None = None
    detector_response: dict[str, Any] | None = None
    fiber_response: dict[str, Any] | None = None
    conversion_response: dict[str, Any] | None = None
    multiplexing_controller: dict[str, Any] | None = None
    repeater_service_profile_path: str | None = None
    repeater_service_profile: dict[str, Any] | None = None
    allow_wavelength_mismatch: bool = False
    allow_generic_memory_fidelity_depolarizing: bool = False
    waiting_tail_tolerance: float = 1e-6
    waiting_max_attempts: int = 4096
    waiting_max_history_pairs: int = 250_000
    waiting_max_wait_signatures: int = 2048
    waiting_budget_mode: str = "fixed_attempts"
    waiting_budget_wall_clock_ps: float | None = None
    waiting_precision_mode: str = "fixed_caps"
    waiting_tv_tolerance: float = 1e-6
    waiting_metric_tolerance: float = 1e-3
    waiting_max_refinement_rounds: int = 4
    waiting_max_runtime_s: float | None = None
    waiting_max_memory_mb: float | None = 512.0
    waiting_protocol: str = "single_buffer_swap_asap"
    rolling_edge_availability: float | tuple[float, ...] | None = None
    rolling_age_distribution_kind: str = "discrete"
    rolling_age_distribution_points_s: tuple[float, ...] = (0.0,)
    rolling_age_distribution_weights: tuple[float, ...] = (1.0,)
    rolling_edge_age_distribution_points_s: tuple[tuple[float, ...], ...] | None = None
    rolling_edge_age_distribution_weights: tuple[tuple[float, ...], ...] | None = None
    rolling_age_evaluation: str = "cartesian"
    rolling_age_support_product_limit: int = 1_000_000
    rolling_accepted_alive_time_s: float | None = None
    rolling_memory_slots_per_edge_side: int = 1
    rolling_selection_policy: str = "configured_distribution"
    rolling_age_distribution_source: str = "configured"
    memory_cutoff_ps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "physical_path": self.physical_path,
            "source_model": self.source_model,
            "detector_kind": self.detector_kind,
            "fock_cutoff": self.fock_cutoff,
            "detector_profile_path": self.detector_profile_path,
            "detector_profile": self.detector_profile,
            "detector_response_loaded": self.detector_response is not None,
            "detector_response": self.detector_response,
            "detector_response_hash": stable_hash(self.detector_response),
            "fiber_response_loaded": self.fiber_response is not None,
            "fiber_response": self.fiber_response,
            "fiber_response_hash": stable_hash(self.fiber_response),
            "conversion_response_loaded": self.conversion_response is not None,
            "conversion_response": self.conversion_response,
            "conversion_response_hash": stable_hash(self.conversion_response),
            "allow_wavelength_mismatch": self.allow_wavelength_mismatch,
            "allow_generic_memory_fidelity_depolarizing": self.allow_generic_memory_fidelity_depolarizing,
            "waiting_tail_tolerance": self.waiting_tail_tolerance,
            "waiting_max_attempts": self.waiting_max_attempts,
            "waiting_max_history_pairs": self.waiting_max_history_pairs,
            "waiting_max_wait_signatures": self.waiting_max_wait_signatures,
            "waiting_budget_mode": self.waiting_budget_mode,
            "waiting_budget_wall_clock_ps": self.waiting_budget_wall_clock_ps,
            "waiting_precision_mode": self.waiting_precision_mode,
            "waiting_tv_tolerance": self.waiting_tv_tolerance,
            "waiting_metric_tolerance": self.waiting_metric_tolerance,
            "waiting_max_refinement_rounds": self.waiting_max_refinement_rounds,
            "waiting_max_runtime_s": self.waiting_max_runtime_s,
            "waiting_max_memory_mb": self.waiting_max_memory_mb,
            "waiting_protocol": self.waiting_protocol,
            "rolling_edge_availability": self.rolling_edge_availability,
            "rolling_age_distribution_kind": self.rolling_age_distribution_kind,
            "rolling_age_distribution_points_s": list(self.rolling_age_distribution_points_s),
            "rolling_age_distribution_weights": list(self.rolling_age_distribution_weights),
            "rolling_edge_age_distribution_points_s": (
                [list(row) for row in self.rolling_edge_age_distribution_points_s]
                if self.rolling_edge_age_distribution_points_s is not None else None
            ),
            "rolling_edge_age_distribution_weights": (
                [list(row) for row in self.rolling_edge_age_distribution_weights]
                if self.rolling_edge_age_distribution_weights is not None else None
            ),
            "rolling_age_evaluation": self.rolling_age_evaluation,
            "rolling_age_support_product_limit": self.rolling_age_support_product_limit,
            "rolling_accepted_alive_time_s": self.rolling_accepted_alive_time_s,
            "rolling_memory_slots_per_edge_side": self.rolling_memory_slots_per_edge_side,
            "rolling_selection_policy": self.rolling_selection_policy,
            "rolling_age_distribution_source": self.rolling_age_distribution_source,
            "memory_cutoff_ps": self.memory_cutoff_ps,
        }
        # Preserve the historical disabled-path theta/cache identity.  The
        # controller becomes part of the identity only when explicitly active.
        if self.multiplexing_controller is not None:
            payload["multiplexing_controller"] = self.multiplexing_controller
            payload["multiplexing_controller_hash"] = stable_hash(
                self.multiplexing_controller
            )
        if self.repeater_service_profile is not None:
            payload["repeater_service_profile_path"] = self.repeater_service_profile_path
            payload["repeater_service_profile"] = self.repeater_service_profile
            payload["repeater_service_profile_hash"] = stable_hash(
                self.repeater_service_profile
            )
        return payload


@dataclass(frozen=True)
class ProtocolSetting:
    label: str
    party: str
    axis: tuple[float, float, float]
    use: str
    probability_weight: float


@dataclass(frozen=True)
class ProtocolConfig:
    requested_pairs: int
    qber_abort_threshold: float = 0.11
    chsh_abort_threshold: float = 2.0
    key_outcome_relation: str = "anti_correlated"

    @property
    def settings(self) -> tuple[ProtocolSetting, ...]:
        inv = 1.0 / math.sqrt(2.0)
        return (
            ProtocolSetting("A_key_Z", "alice", (0.0, 0.0, 1.0), "key", 2.0),
            ProtocolSetting("A0_Z", "alice", (0.0, 0.0, 1.0), "chsh", 1.0),
            ProtocolSetting("A1_X", "alice", (1.0, 0.0, 0.0), "chsh", 1.0),
            ProtocolSetting("B_key_Z", "bob", (0.0, 0.0, 1.0), "key", 2.0),
            ProtocolSetting("B0_Z_plus_X", "bob", (inv, 0.0, inv), "chsh", 1.0),
            ProtocolSetting("B1_Z_minus_X", "bob", (-inv, 0.0, inv), "chsh", 1.0),
        )

    @property
    def chsh_pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("A0_Z", "B0_Z_plus_X"),
            ("A0_Z", "B1_Z_minus_X"),
            ("A1_X", "B0_Z_plus_X"),
            ("A1_X", "B1_Z_minus_X"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pairs": self.requested_pairs,
            "qber_abort_threshold": self.qber_abort_threshold,
            "chsh_abort_threshold": self.chsh_abort_threshold,
            "key_outcome_relation": self.key_outcome_relation,
            "measurement_settings": [setting.__dict__ for setting in self.settings],
            "chsh_setting_pairs": [list(pair) for pair in self.chsh_pairs],
        }


@dataclass(frozen=True)
class TranscriptDistribution:
    """Evaluated finite chain-scope E91 transcript distribution."""

    metadata: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def probabilities(self) -> np.ndarray:
        return np.asarray([row["probability"] for row in self.rows], dtype=float)

    def to_case(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "rows": self.rows}


@dataclass(frozen=True)
class FrozenHistogramKernel:
    """Frozen-theta E91 transcript histogram kernel.

    Once constructed, the front matter, physical model context, hop count,
    protocol settings, and Eve move are the model family.  ``evaluate`` only
    accepts the route fiber lengths.
    """

    front: dict[str, Any]
    context: ModelContext
    hop_count: int
    attack: EveMove
    key_length: int = 256
    fiber_length_symbols: tuple[str, ...] | None = None
    emit_full_bsm_audit: bool = False
    max_full_bsm_repeaters: int = 2
    emit_detector_click_audit: bool = False
    emit_approximation_audit: bool = False
    max_approximation_audit_repeaters: int = 2

    def __post_init__(self) -> None:
        from .topology import default_fiber_length_symbols
        from sequence_game.physical.repeater_service import (
            repeater_service_profile_from_context,
        )

        if self.hop_count < 1:
            raise ValueError("hop_count must be >= 1")
        if self.key_length < 1:
            raise ValueError("key_length must be >= 1")
        service_profile = repeater_service_profile_from_context(
            self.context.repeater_service_profile
        )
        if service_profile is not None:
            max_hops = service_profile.histogram_adapter.get("max_executable_hops")
            if max_hops is not None and self.hop_count > int(max_hops):
                raise ValueError(
                    f"repeater service profile {service_profile.profile_id!r} is executable "
                    f"through at most {int(max_hops)} hop(s), got {self.hop_count}"
                )
        symbols = (
            tuple(default_fiber_length_symbols(self.hop_count))
            if self.fiber_length_symbols is None else
            tuple(self.fiber_length_symbols)
        )
        if len(symbols) != self.hop_count:
            raise ValueError("fiber_length_symbols must match hop_count")
        object.__setattr__(self, "front", copy.deepcopy(self.front))
        object.__setattr__(self, "context", copy.deepcopy(self.context))
        object.__setattr__(self, "attack", copy.deepcopy(self.attack))
        object.__setattr__(self, "fiber_length_symbols", symbols)

    @property
    def protocol(self) -> ProtocolConfig:
        return ProtocolConfig(requested_pairs=self.key_length)

    @property
    def theta_payload(self) -> dict[str, Any]:
        return frozen_theta_identity_payload(
            front=self.front,
            context=self.context,
            hop_count=self.hop_count,
            attack=self.attack,
            protocol=self.protocol,
            fiber_length_symbols=self.fiber_length_symbols or (),
            emit_full_bsm_audit=self.emit_full_bsm_audit,
            max_full_bsm_repeaters=self.max_full_bsm_repeaters,
            emit_detector_click_audit=self.emit_detector_click_audit,
            emit_approximation_audit=self.emit_approximation_audit,
            max_approximation_audit_repeaters=self.max_approximation_audit_repeaters,
        )

    @property
    def theta_hash(self) -> str:
        return stable_hash(self.theta_payload)

    def evaluate(self, fiber_lengths_m: Sequence[float]) -> TranscriptDistribution:
        from .generation import generate_histogram_case
        from .topology import make_route

        lengths = [float(length) for length in fiber_lengths_m]
        if len(lengths) != self.hop_count:
            raise ValueError(
                f"expected {self.hop_count} fiber lengths, got {len(lengths)}"
            )
        route = make_route(self.hop_count, lengths)
        route["fiber_length_symbols"] = list(self.fiber_length_symbols or ())
        case = generate_histogram_case(
            route=route,
            front=self.front,
            attack=self.attack,
            key_length=self.key_length,
            context=self.context,
            emit_full_bsm_audit=self.emit_full_bsm_audit,
            max_full_bsm_repeaters=self.max_full_bsm_repeaters,
            emit_detector_click_audit=self.emit_detector_click_audit,
            emit_approximation_audit=self.emit_approximation_audit,
            max_approximation_audit_repeaters=self.max_approximation_audit_repeaters,
            seed=None,
        )
        case["metadata"]["frozen_kernel"] = frozen_theta_identity_summary(self.theta_payload)
        case["metadata"]["frozen_theta_hash"] = self.theta_hash
        return TranscriptDistribution(metadata=case["metadata"], rows=case["rows"])


def _json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-15:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def stable_hash(payload: Any) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def front_matter_identity_payload(front: dict[str, Any]) -> dict[str, Any]:
    """Canonical hardware/protocol front-matter content used in theta identity."""

    keys = (
        "generator_version",
        "front_matter_role",
        "hardware_config_dir",
        "hardware_config_source_paths",
        "selected_models",
        "models",
        "device_constants",
        "device_parameter_inventory",
        "mapped_device_parameters",
        "unsupported_device_parameters",
        "adapter_conversion_notes",
        "fallback_defaults_used",
        "event_free_model_scope",
    )
    return {key: copy.deepcopy(front[key]) for key in keys if key in front}


def frozen_theta_identity_payload(
    *,
    front: dict[str, Any],
    context: ModelContext,
    hop_count: int,
    attack: EveMove,
    protocol: ProtocolConfig,
    fiber_length_symbols: Sequence[str],
    emit_full_bsm_audit: bool,
    max_full_bsm_repeaters: int,
    emit_detector_click_audit: bool,
    emit_approximation_audit: bool,
    max_approximation_audit_repeaters: int,
) -> dict[str, Any]:
    """Complete frozen-theta identity for H_{theta,h,move}(L)->P(tau)."""

    from .devices import detector_profile_metadata, pnr_response_model_metadata
    from .fock import detector_max_count, resolved_detector_kind

    resolved_kind = resolved_detector_kind(front, context)
    profile = detector_profile_metadata(front, context)
    max_count = detector_max_count(context, profile)
    pnr_meta = pnr_response_model_metadata(profile, resolved_kind, max_count)
    return {
        "generator_version": GENERATOR_VERSION,
        "kernel_kind": "frozen_e91_transcript_histogram_kernel",
        "front_matter": front_matter_identity_payload(front),
        "model_context": context.to_dict(),
        "resolved_detector_kind": resolved_kind,
        "detector_response_model": pnr_meta,
        "hop_count": int(hop_count),
        "fiber_length_symbols": list(fiber_length_symbols),
        "eve_move": attack.to_dict(),
        "protocol": protocol.to_dict(),
        "physics_flags": {
            "emit_full_bsm_audit": bool(emit_full_bsm_audit),
            "max_full_bsm_repeaters": int(max_full_bsm_repeaters),
            "emit_detector_click_audit": bool(emit_detector_click_audit),
            "emit_approximation_audit": bool(emit_approximation_audit),
            "max_approximation_audit_repeaters": int(max_approximation_audit_repeaters),
        },
    }


def frozen_theta_identity_summary(theta_payload: dict[str, Any]) -> dict[str, Any]:
    context = theta_payload["model_context"]
    front = theta_payload["front_matter"]
    return {
        "theta_hash": stable_hash(theta_payload),
        "front_matter_identity_hash": stable_hash(front),
        "model_context_hash": stable_hash(context),
        "detector_profile_hash": stable_hash(context.get("detector_profile")),
        "detector_response_hash": context.get("detector_response_hash"),
        "fiber_response_hash": context.get("fiber_response_hash"),
        "conversion_response_hash": context.get("conversion_response_hash"),
        "eve_move_hash": stable_hash(theta_payload["eve_move"]),
        "protocol_hash": stable_hash(theta_payload["protocol"]),
        "hop_count": theta_payload["hop_count"],
        "fiber_length_symbols": theta_payload["fiber_length_symbols"],
    }
