"""Direct E91 transcript simulation from density-matrix probabilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal

import numpy as np

MODEL_VERSION = "direct_e91_transcript.v1"
BACKEND = "direct_e91_transcript"
BACKEND_SCOPE = "direct_e91_protocol_transcript_simulator"

AttackKind = Literal[
    "none",
    "edge_added_loss",
    "edge_intercept_resend",
    "swap_denial",
    "memory_degradation",
    "repeater_memory_measure_Z",
    "repeater_memory_measure_X",
    "repeater_memory_dephasing_probe",
]
TargetKind = Literal["none", "edge", "node", "memory"]
SettingUse = Literal["key", "chsh"]
Party = Literal["alice", "bob"]

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_PAULI = (_X, _Y, _Z)
_I4 = np.eye(4, dtype=complex)


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _stable_hash(payload: Any) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _normalize_axis(axis: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray(tuple(axis), dtype=float)
    if arr.shape != (3,):
        raise ValueError("axis vector must contain exactly three values")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("axis vector must be nonzero")
    return tuple(float(x) for x in arr / norm)


@dataclass(frozen=True)
class DirectE91MeasurementSetting:
    label: str
    party: Party
    axis: tuple[float, float, float]
    use: SettingUse
    probability_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.party not in {"alice", "bob"}:
            raise ValueError("party must be 'alice' or 'bob'")
        if self.use not in {"key", "chsh"}:
            raise ValueError("use must be 'key' or 'chsh'")
        if self.probability_weight < 0.0:
            raise ValueError("probability_weight must be nonnegative")
        object.__setattr__(self, "axis", _normalize_axis(self.axis))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "party": self.party,
            "axis": list(self.axis),
            "use": self.use,
            "probability_weight": float(self.probability_weight),
        }


@dataclass
class DirectE91HardwareProvenance:
    hardware_profile_id: str = "direct_e91_defaults"
    hardware_profile_source_paths: list[str] = field(default_factory=list)
    mapped_parameter_names: list[str] = field(default_factory=list)
    fallback_default_names: list[str] = field(default_factory=list)
    unmapped_parameter_names: list[str] = field(default_factory=list)
    literature_source_ids: list[str] = field(default_factory=list)
    conversion_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_profile_id": self.hardware_profile_id,
            "hardware_profile_source_paths": list(self.hardware_profile_source_paths),
            "mapped_parameter_names": sorted(set(self.mapped_parameter_names)),
            "fallback_default_names": sorted(set(self.fallback_default_names)),
            "unmapped_parameter_names": sorted(set(self.unmapped_parameter_names)),
            "literature_source_ids": sorted(set(self.literature_source_ids)),
            "conversion_notes": dict(sorted(self.conversion_notes.items())),
        }


@dataclass
class DirectE91HardwareConfig:
    fiber_loss_db_per_km: float = 0.2
    source_pair_success_probability: float = 1.0
    detector_efficiency: float = 1.0
    dark_count_probability: float = 0.0
    misalignment_probability: float = 0.0
    elementary_pair_fidelity: float = 1.0
    swap_success_probability: float = 1.0
    swap_depolarizing_probability: float = 0.0
    memory_depolarizing_probability: float = 0.0
    memory_dephasing_probability: float = 0.0
    provenance: DirectE91HardwareProvenance = field(
        default_factory=DirectE91HardwareProvenance
    )

    def __post_init__(self) -> None:
        for name in (
            "source_pair_success_probability",
            "detector_efficiency",
            "dark_count_probability",
            "misalignment_probability",
            "elementary_pair_fidelity",
            "swap_success_probability",
            "swap_depolarizing_probability",
            "memory_depolarizing_probability",
            "memory_dephasing_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.fiber_loss_db_per_km < 0.0:
            raise ValueError("fiber_loss_db_per_km must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiber_loss_db_per_km": float(self.fiber_loss_db_per_km),
            "source_pair_success_probability": float(
                self.source_pair_success_probability
            ),
            "detector_efficiency": float(self.detector_efficiency),
            "dark_count_probability": float(self.dark_count_probability),
            "misalignment_probability": float(self.misalignment_probability),
            "elementary_pair_fidelity": float(self.elementary_pair_fidelity),
            "swap_success_probability": float(self.swap_success_probability),
            "swap_depolarizing_probability": float(
                self.swap_depolarizing_probability
            ),
            "memory_depolarizing_probability": float(
                self.memory_depolarizing_probability
            ),
            "memory_dephasing_probability": float(self.memory_dephasing_probability),
            "provenance": self.provenance.to_dict(),
        }


def default_e91_measurement_settings() -> tuple[DirectE91MeasurementSetting, ...]:
    inv = 1.0 / math.sqrt(2.0)
    return (
        DirectE91MeasurementSetting("A_key_Z", "alice", (0.0, 0.0, 1.0), "key", 2.0),
        DirectE91MeasurementSetting("A0_Z", "alice", (0.0, 0.0, 1.0), "chsh", 1.0),
        DirectE91MeasurementSetting("A1_X", "alice", (1.0, 0.0, 0.0), "chsh", 1.0),
        DirectE91MeasurementSetting("B_key_Z", "bob", (0.0, 0.0, 1.0), "key", 2.0),
        DirectE91MeasurementSetting("B0_Z_plus_X", "bob", (inv, 0.0, inv), "chsh", 1.0),
        DirectE91MeasurementSetting("B1_Z_minus_X", "bob", (-inv, 0.0, inv), "chsh", 1.0),
    )


@dataclass
class DirectE91ProtocolConfig:
    requested_pairs: int = 256
    qber_abort_threshold: float = 0.11
    chsh_abort_threshold: float = 2.0
    min_key_pairs_for_qber_abort: int = 16
    min_chsh_pairs_for_chsh_abort: int = 16
    finite_key_epsilon: float = 1e-9
    error_correction_efficiency: float = 1.16
    finite_key_formula_id: str = "diagnostic_one_minus_two_hq"
    chsh_sign_convention: str = "singlet_negative_standard"
    target_bell_label: str = "psi_minus"
    key_outcome_relation: str = "anti_correlated"
    measurement_settings: tuple[DirectE91MeasurementSetting, ...] = field(
        default_factory=default_e91_measurement_settings
    )
    chsh_setting_pairs: tuple[tuple[str, str], ...] = (
        ("A0_Z", "B0_Z_plus_X"),
        ("A0_Z", "B1_Z_minus_X"),
        ("A1_X", "B0_Z_plus_X"),
        ("A1_X", "B1_Z_minus_X"),
    )

    def __post_init__(self) -> None:
        if self.requested_pairs < 0:
            raise ValueError("requested_pairs must be nonnegative")
        if not 0.0 <= self.qber_abort_threshold <= 1.0:
            raise ValueError("qber_abort_threshold must be in [0, 1]")
        if self.chsh_abort_threshold < 0.0:
            raise ValueError("chsh_abort_threshold must be nonnegative")
        if self.min_key_pairs_for_qber_abort < 0:
            raise ValueError("min_key_pairs_for_qber_abort must be nonnegative")
        if self.min_chsh_pairs_for_chsh_abort < 0:
            raise ValueError("min_chsh_pairs_for_chsh_abort must be nonnegative")
        if not 0.0 < self.finite_key_epsilon < 1.0:
            raise ValueError("finite_key_epsilon must be in (0, 1)")
        if self.error_correction_efficiency < 0.0:
            raise ValueError("error_correction_efficiency must be nonnegative")
        if self.key_outcome_relation not in {"correlated", "anti_correlated"}:
            raise ValueError("key_outcome_relation must be correlated or anti_correlated")
        object.__setattr__(self, "measurement_settings", tuple(self.measurement_settings))
        self._validate_settings()

    def _validate_settings(self) -> None:
        labels = [setting.label for setting in self.measurement_settings]
        if len(set(labels)) != len(labels):
            raise ValueError("measurement setting labels must be unique")
        alice = [s for s in self.measurement_settings if s.party == "alice"]
        bob = [s for s in self.measurement_settings if s.party == "bob"]
        if not alice or not bob:
            raise ValueError("at least one Alice and Bob setting is required")
        if not any(s.use == "key" for s in alice) or not any(s.use == "key" for s in bob):
            raise ValueError("at least one key setting per party is required")
        label_set = set(labels)
        for a_label, b_label in self.chsh_setting_pairs:
            if a_label not in label_set or b_label not in label_set:
                raise ValueError("CHSH setting pair references unknown labels")
        if len(self.chsh_setting_pairs) != 4:
            raise ValueError("exactly four CHSH setting pairs are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_pairs": int(self.requested_pairs),
            "qber_abort_threshold": float(self.qber_abort_threshold),
            "chsh_abort_threshold": float(self.chsh_abort_threshold),
            "min_key_pairs_for_qber_abort": int(self.min_key_pairs_for_qber_abort),
            "min_chsh_pairs_for_chsh_abort": int(self.min_chsh_pairs_for_chsh_abort),
            "finite_key_epsilon": float(self.finite_key_epsilon),
            "error_correction_efficiency": float(self.error_correction_efficiency),
            "finite_key_formula_id": self.finite_key_formula_id,
            "chsh_sign_convention": self.chsh_sign_convention,
            "target_bell_label": self.target_bell_label,
            "key_outcome_relation": self.key_outcome_relation,
            "measurement_settings": [s.to_dict() for s in self.measurement_settings],
            "chsh_setting_pairs": [list(pair) for pair in self.chsh_setting_pairs],
        }


@dataclass(frozen=True)
class DirectE91AttackSpec:
    attack_id: str = "none"
    attack_kind: AttackKind = "none"
    target_kind: TargetKind = "none"
    target_id: str = ""
    loss_multiplier: float = 1.0
    intercept_basis_label: str = "Z"
    intercept_probability: float = 1.0
    memory_depolarizing_probability: float = 0.0
    memory_dephasing_probability: float = 0.0
    dephasing_theta: float = 0.0
    measurement_probability: float = 1.0
    information_probability: float = 1.0
    target_memory_side: str = "both"

    def __post_init__(self) -> None:
        valid_attacks = set(AttackKind.__args__)  # type: ignore[attr-defined]
        valid_targets = set(TargetKind.__args__)  # type: ignore[attr-defined]
        if self.attack_kind not in valid_attacks:
            raise ValueError(f"unsupported attack_kind {self.attack_kind!r}")
        if self.target_kind not in valid_targets:
            raise ValueError(f"unsupported target_kind {self.target_kind!r}")
        for name in (
            "loss_multiplier",
            "intercept_probability",
            "memory_depolarizing_probability",
            "memory_dephasing_probability",
            "measurement_probability",
            "information_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.dephasing_theta < 0.0 or self.dephasing_theta > math.pi / 2.0:
            raise ValueError("dephasing_theta must be in [0, pi/2]")
        if self.intercept_basis_label not in {"Z", "X"}:
            raise ValueError("intercept_basis_label must be Z or X")
        if self.target_memory_side not in {"left", "right", "both"}:
            raise ValueError("target_memory_side must be left, right, or both")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "attack_kind": self.attack_kind,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "loss_multiplier": float(self.loss_multiplier),
            "intercept_basis_label": self.intercept_basis_label,
            "intercept_probability": float(self.intercept_probability),
            "memory_depolarizing_probability": float(
                self.memory_depolarizing_probability
            ),
            "memory_dephasing_probability": float(self.memory_dephasing_probability),
            "dephasing_theta": float(self.dephasing_theta),
            "measurement_probability": float(self.measurement_probability),
            "information_probability": float(self.information_probability),
            "target_memory_side": self.target_memory_side,
        }


@dataclass(frozen=True)
class DirectE91Route:
    route_id: str
    node_path: tuple[str, ...]
    edge_ids: tuple[str, ...]
    edge_lengths_m: dict[str, float]
    total_length_m: float
    edge_fidelities: dict[str, float] = field(default_factory=dict)
    edge_transmittances: dict[str, float] = field(default_factory=dict)
    edge_loss_db: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.node_path) < 2:
            raise ValueError("node_path must include Alice and Bob")
        if len(self.edge_ids) != len(self.node_path) - 1:
            raise ValueError("edge_ids must match route hops")
        object.__setattr__(self, "node_path", tuple(str(n) for n in self.node_path))
        object.__setattr__(self, "edge_ids", tuple(str(e) for e in self.edge_ids))
        if len(set(self.node_path)) != len(self.node_path):
            raise ValueError("node_path must be simple; repeated route nodes are not supported")
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("edge_ids must be unique within a route")
        object.__setattr__(
            self,
            "edge_lengths_m",
            {str(k): float(v) for k, v in self.edge_lengths_m.items()},
        )
        object.__setattr__(
            self,
            "edge_fidelities",
            {str(k): float(v) for k, v in self.edge_fidelities.items()},
        )
        object.__setattr__(
            self,
            "edge_transmittances",
            {str(k): _clip_probability(v) for k, v in self.edge_transmittances.items()},
        )
        object.__setattr__(
            self,
            "edge_loss_db",
            {str(k): float(v) for k, v in self.edge_loss_db.items()},
        )

    @property
    def internal_nodes(self) -> tuple[str, ...]:
        return self.node_path[1:-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "node_path": list(self.node_path),
            "edge_ids": list(self.edge_ids),
            "edge_lengths_m": dict(self.edge_lengths_m),
            "total_length_m": float(self.total_length_m),
            "edge_fidelities": dict(self.edge_fidelities),
            "edge_transmittances": dict(self.edge_transmittances),
            "edge_loss_db": dict(self.edge_loss_db),
        }


@dataclass
class QuantumBranch:
    probability: float
    rho: np.ndarray
    eve_observation_labels: tuple[str, ...] = ()
    public_bsm_labels: tuple[str, ...] = ()
    eve_outcome: int | None = None


@dataclass
class SwapResult:
    rho: np.ndarray
    outcome_probabilities: dict[str, float]
    pauli_frame_corrections_applied: bool
    target_bell_label: str


@dataclass
class RouteStateResult:
    rho: np.ndarray
    branch_model: dict[str, Any] | None
    channel_record: dict[str, Any]
    elementary_link_count: int
    bsm_swap_count: int
    bsm_outcome_probabilities: list[dict[str, Any]]
    pauli_frame_corrections_applied: bool
    target_memory_side_effective: str | None
    edge_attack_subsystem: str | None
    edge_attack_edge_id: str | None
    effective_end_to_end_fidelity_model: float
    actual_final_bell_fidelity: float
    link_fidelity_product_diagnostic: float
    route_fidelity_application: str
    pauli_frame_correction_model: str


@dataclass
class DirectE91PreparedTransition:
    alice_node_id: str
    bob_node_id: str
    selected_route: DirectE91Route
    eve_attack: DirectE91AttackSpec
    hardware: DirectE91HardwareConfig
    protocol: DirectE91ProtocolConfig
    target_active: bool
    delivery_probability: float
    rho_public: np.ndarray
    eve_branch_model: dict[str, Any] | None
    eve_guess_probability: float
    eve_guess_model: str
    chsh_correlations_expected: dict[str, float]
    chsh_S_expected: float | None
    channel_record: dict[str, Any]
    elementary_link_count: int
    bsm_swap_count: int
    bsm_outcome_probabilities: list[dict[str, Any]]
    pauli_frame_corrections_applied: bool
    pauli_frame_correction_model: str
    target_memory_side_effective: str | None
    edge_attack_subsystem: str | None
    edge_attack_edge_id: str | None
    elementary_pair_fidelity: float
    effective_end_to_end_fidelity_model: float
    actual_final_bell_fidelity: float
    link_fidelity_product_diagnostic: float
    route_fidelity_application: str
    provenance_base: dict[str, Any]
    training_light_provenance_base: dict[str, Any]
    transcript_static_summary: dict[str, Any]
    cache_key: Any = None


@dataclass(frozen=True)
class DirectE91StepInput:
    topology: Any
    alice_node_id: str
    bob_node_id: str
    selected_route: DirectE91Route
    eve_attack: DirectE91AttackSpec
    hardware: DirectE91HardwareConfig
    protocol: DirectE91ProtocolConfig
    seed: int
    step_index: int


@dataclass
class DirectE91StepResult:
    backend: str
    backend_scope: str
    no_sequence_runtime: bool
    model_version: str
    alice_node_id: str
    bob_node_id: str
    route_id: str
    selected_route: dict[str, Any]
    attack_id: str
    attack_kind: str
    target_kind: str
    target_id: str
    target_active: bool
    requested_pair_count: int
    delivered_pair_count: int
    delivery_probability: float
    setting_pair_counts: dict[str, int]
    key_pair_count: int
    chsh_pair_count: int
    discarded_setting_pair_count: int
    key_basis_count: int
    test_basis_count: int
    sifted_key_count: int
    key_error_count: int
    qber_key: float | None
    qber_abort: bool
    chsh_correlations_expected: dict[str, float]
    chsh_S_expected: float | None
    chsh_correlations_observed: dict[str, float | None]
    chsh_S_observed: float | None
    chsh_correlations: dict[str, float | None]
    chsh_S: float | None
    chsh_abort: bool
    public_outcome: str
    information_exposed: bool
    eve_observation_count: int
    eve_observation_distribution: dict[str, int]
    eve_guess_probability: float
    secure_delivery: bool
    finite_key_estimate_bits: float
    finite_key_formula_id: str
    finite_key_claim_scope: str
    elementary_pair_fidelity: float
    effective_end_to_end_fidelity_model: float
    actual_final_bell_fidelity: float
    link_fidelity_product_diagnostic: float
    route_fidelity_application: str
    source_pair_success_application: str
    route_state_model: str
    bsm_swap_count: int
    elementary_link_count: int
    bsm_outcome_probabilities: list[dict[str, Any]]
    pauli_frame_corrections_applied: bool
    pauli_frame_correction_model: str
    target_memory_side_effective: str | None
    edge_attack_subsystem: str | None
    edge_attack_edge_id: str | None
    eve_guess_model: str
    eve_win: bool
    reward_components: dict[str, bool]
    provenance: dict[str, Any]
    transcript_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_step_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_scope": self.backend_scope,
            "no_sequence_runtime": self.no_sequence_runtime,
            "public_outcome": self.public_outcome,
            "route_id": self.route_id,
            "alice_route_id": self.route_id,
            "eve_action_id": self.attack_id,
            "eve_action": {
                "id": self.attack_id,
                "attack_kind": self.attack_kind,
                "target_kind": self.target_kind,
                "target": self.target_id,
            },
            "delivered": self.delivered_pair_count > 0,
            "delivered_count": self.delivered_pair_count,
            "qber_key": self.qber_key,
            "qber": self.qber_key,
            "chsh_S": self.chsh_S_observed,
            "chsh_S_expected": self.chsh_S_expected,
            "chsh_S_observed": self.chsh_S_observed,
            "information_exposed": self.information_exposed,
            "secure_delivery": self.secure_delivery,
            "eve_win": self.eve_win,
            "route_state_model": self.route_state_model,
            "provenance": dict(self.provenance),
        }


def bell_state_density(label: str) -> np.ndarray:
    states = {
        "phi_plus": np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0),
        "phi_minus": np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) / math.sqrt(2.0),
        "psi_plus": np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / math.sqrt(2.0),
        "psi_minus": np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / math.sqrt(2.0),
    }
    if label not in states:
        raise ValueError(f"unknown Bell label {label!r}")
    ket = states[label].reshape(4, 1)
    return ket @ ket.conj().T


def werner_state(fidelity: float, target_bell_label: str = "psi_minus") -> np.ndarray:
    fidelity = _clip_probability(fidelity)
    labels = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
    if target_bell_label not in labels:
        raise ValueError(f"unknown Bell label {target_bell_label!r}")
    rho = fidelity * bell_state_density(target_bell_label)
    for label in labels:
        if label != target_bell_label:
            rho += ((1.0 - fidelity) / 3.0) * bell_state_density(label)
    return _clean_density(rho)


def apply_one_qubit_channel(
    rho: np.ndarray,
    qubit_index: int,
    kraus_ops: Iterable[np.ndarray],
) -> np.ndarray:
    if qubit_index not in {0, 1}:
        raise ValueError("qubit_index must be 0 or 1")
    result = np.zeros((4, 4), dtype=complex)
    for op in kraus_ops:
        op = np.asarray(op, dtype=complex)
        if op.shape != (2, 2):
            raise ValueError("one-qubit Kraus operator must be 2x2")
        lifted = np.kron(op, _I2) if qubit_index == 0 else np.kron(_I2, op)
        result += lifted @ rho @ lifted.conj().T
    return _clean_density(result)


def apply_two_qubit_channel(
    rho: np.ndarray,
    kraus_ops: Iterable[np.ndarray],
) -> np.ndarray:
    result = np.zeros((4, 4), dtype=complex)
    for op in kraus_ops:
        op = np.asarray(op, dtype=complex)
        if op.shape != (4, 4):
            raise ValueError("two-qubit Kraus operator must be 4x4")
        result += op @ rho @ op.conj().T
    return _clean_density(result)


def projectors_for_axis(axis_vector: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = _normalize_axis(axis_vector)
    observable = x * _X + y * _Y + z * _Z
    p0 = (_I2 + observable) / 2.0
    p1 = (_I2 - observable) / 2.0
    return _clean_projector(p0), _clean_projector(p1)


def joint_outcome_probabilities(
    rho: np.ndarray,
    alice_axis: Iterable[float],
    bob_axis: Iterable[float],
) -> np.ndarray:
    alice_projectors = projectors_for_axis(alice_axis)
    bob_projectors = projectors_for_axis(bob_axis)
    probs = np.zeros((2, 2), dtype=float)
    for a_idx, pa in enumerate(alice_projectors):
        for b_idx, pb in enumerate(bob_projectors):
            effect = np.kron(pa, pb)
            probs[a_idx, b_idx] = float(np.real_if_close(np.trace(effect @ rho)))
    probs = np.clip(probs, 0.0, 1.0)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full((2, 2), 0.25, dtype=float)
    return probs / total


def binary_entropy(q: float) -> float:
    q = _clip_probability(q)
    if q in {0.0, 1.0}:
        return 0.0
    return -q * math.log2(q) - (1.0 - q) * math.log2(1.0 - q)


def run_direct_e91_transcript_batch(
    inputs: Iterable[DirectE91StepInput],
    prepared_cache: dict[Any, DirectE91PreparedTransition] | None = None,
    *,
    training_light: bool = False,
) -> list[DirectE91StepResult]:
    return [
        run_direct_e91_transcript_step(
            step_input,
            prepared_cache=prepared_cache,
            training_light=training_light,
        )
        for step_input in inputs
    ]


def prepare_direct_e91_transition(
    step_input: DirectE91StepInput,
    *,
    training_light: bool = False,
) -> DirectE91PreparedTransition:
    route = step_input.selected_route
    attack = step_input.eve_attack
    hardware = step_input.hardware
    protocol = step_input.protocol
    _validate_step_route(step_input)
    target_active = _attack_target_active(attack, route)
    delivery_probability = _route_delivery_probability(
        route,
        hardware,
        attack,
        target_active,
    )
    route_state = route_density_matrix_from_elementary_swaps(
        route,
        hardware,
        protocol,
        attack,
        target_active,
    )
    rho_public = _apply_measurement_noise(route_state.rho, hardware)
    chsh_correlations_expected, chsh_s_expected = _chsh_from_density(
        rho_public,
        protocol,
    )
    eve_guess_probability = _eve_guess_probability(
        rho_public,
        route_state.branch_model,
        attack,
        target_active,
        protocol,
        hardware,
    )
    eve_guess_model = _eve_guess_model(attack, target_active, route_state.branch_model)
    training_light_provenance_base = {
        "backend": BACKEND,
        "backend_scope": BACKEND_SCOPE,
        "no_sequence_runtime": True,
        "model_version": MODEL_VERSION,
        "training_light": True,
    }
    provenance_base = (
        dict(training_light_provenance_base)
        if training_light
        else _result_provenance_base(
            step_input,
            route,
            attack,
            hardware,
            protocol,
        )
    )
    transcript_static_summary = {
        "channel_record": route_state.channel_record,
        "route_state_model": "explicit_elementary_links_with_bsm_swapping",
        "elementary_link_count": route_state.elementary_link_count,
        "bsm_swap_count": route_state.bsm_swap_count,
        "bsm_outcome_probabilities": route_state.bsm_outcome_probabilities,
        "pauli_frame_corrections_applied": (
            route_state.pauli_frame_corrections_applied
        ),
        "pauli_frame_correction_model": route_state.pauli_frame_correction_model,
        "actual_final_bell_fidelity": route_state.actual_final_bell_fidelity,
        "link_fidelity_product_diagnostic": (
            route_state.link_fidelity_product_diagnostic
        ),
        "hardware_provenance": hardware.provenance.to_dict(),
        "chsh_sign_convention": protocol.chsh_sign_convention,
        "chsh_expected_is_density_matrix_diagnostic": True,
        "chsh_observed_is_sampled_transcript_statistic": True,
        "endpoint_sampling_state": "rho_public_density_matrix",
        "eve_branch_model_scope": (
            "exact_eve_posterior_and_private_observation_accounting"
            if route_state.branch_model is not None
            else "none"
        ),
        "eve_guess_key_bit_definition": "alice_raw_key_bit_first_configured_key_pair",
        "eve_guess_conditioning": (
            "eve_retained_observation_record_and_public_bsm_record"
        ),
    }
    return DirectE91PreparedTransition(
        alice_node_id=step_input.alice_node_id,
        bob_node_id=step_input.bob_node_id,
        selected_route=route,
        eve_attack=attack,
        hardware=hardware,
        protocol=protocol,
        target_active=target_active,
        delivery_probability=delivery_probability,
        rho_public=rho_public,
        eve_branch_model=route_state.branch_model,
        eve_guess_probability=float(eve_guess_probability),
        eve_guess_model=eve_guess_model,
        chsh_correlations_expected=chsh_correlations_expected,
        chsh_S_expected=chsh_s_expected,
        channel_record=route_state.channel_record,
        elementary_link_count=route_state.elementary_link_count,
        bsm_swap_count=route_state.bsm_swap_count,
        bsm_outcome_probabilities=route_state.bsm_outcome_probabilities,
        pauli_frame_corrections_applied=route_state.pauli_frame_corrections_applied,
        pauli_frame_correction_model=route_state.pauli_frame_correction_model,
        target_memory_side_effective=route_state.target_memory_side_effective,
        edge_attack_subsystem=route_state.edge_attack_subsystem,
        edge_attack_edge_id=route_state.edge_attack_edge_id,
        elementary_pair_fidelity=float(hardware.elementary_pair_fidelity),
        effective_end_to_end_fidelity_model=float(
            route_state.effective_end_to_end_fidelity_model
        ),
        actual_final_bell_fidelity=float(route_state.actual_final_bell_fidelity),
        link_fidelity_product_diagnostic=float(
            route_state.link_fidelity_product_diagnostic
        ),
        route_fidelity_application=route_state.route_fidelity_application,
        provenance_base=provenance_base,
        training_light_provenance_base=training_light_provenance_base,
        transcript_static_summary=transcript_static_summary,
        cache_key=_prepared_transition_cache_key(
            step_input,
            target_active,
            training_light=training_light,
        ),
    )


def sample_prepared_direct_e91_transition(
    prepared: DirectE91PreparedTransition,
    seed: int,
    step_index: int,
    requested_pairs: int | None = None,
    *,
    training_light: bool = False,
) -> DirectE91StepResult:
    rng = np.random.default_rng([int(seed), int(step_index)])
    route = prepared.selected_route
    attack = prepared.eve_attack
    hardware = prepared.hardware
    protocol = prepared.protocol
    requested_pair_count = (
        int(protocol.requested_pairs)
        if requested_pairs is None
        else int(requested_pairs)
    )
    if requested_pair_count < 0:
        raise ValueError("requested_pairs must be nonnegative")
    delivered_pair_count = int(
        rng.binomial(requested_pair_count, prepared.delivery_probability)
    )

    setting_pair_counts: dict[str, int] = {}
    outcome_counts: dict[str, dict[str, int]] = {}
    eve_observations: dict[str, int] = {}
    key_pair_count = 0
    chsh_pair_count = 0
    discarded_setting_pair_count = 0
    key_error_count = 0

    if delivered_pair_count > 0:
        pair_labels, pair_probs, pair_settings = _setting_pair_distribution(protocol)
        sampled_setting_counts = rng.multinomial(delivered_pair_count, pair_probs)
        chsh_pair_labels = _chsh_pair_labels(protocol)
        for pair_label, pair_count, (alice_setting, bob_setting) in zip(
            pair_labels,
            sampled_setting_counts,
            pair_settings,
        ):
            count = int(pair_count)
            if count <= 0:
                continue
            setting_pair_counts[pair_label] = count
            if alice_setting.use == "key" and bob_setting.use == "key":
                key_pair_count += count
            elif pair_label in chsh_pair_labels:
                chsh_pair_count += count
            else:
                discarded_setting_pair_count += count
            sampled = _sample_joint_counts(
                rng,
                count,
                prepared.rho_public,
                alice_setting,
                bob_setting,
                hardware,
            )
            pair_outcomes: dict[str, int] = {}
            for row in sampled:
                outcome_label = f"{row['alice_outcome']}{row['bob_outcome']}"
                pair_outcomes[outcome_label] = (
                    pair_outcomes.get(outcome_label, 0) + int(row["count"])
                )
                if alice_setting.use == "key" and bob_setting.use == "key":
                    if _is_key_error(
                        int(row["alice_outcome"]),
                        int(row["bob_outcome"]),
                        protocol,
                    ):
                        key_error_count += int(row["count"])
            outcome_counts[pair_label] = pair_outcomes

        if prepared.eve_branch_model is not None:
            eve_observations.update(
                _sample_eve_observations_from_branch_model(
                    rng,
                    delivered_pair_count,
                    prepared.eve_branch_model,
                )
            )

    if (
        delivered_pair_count > 0
        and prepared.target_active
        and attack.attack_kind == "repeater_memory_dephasing_probe"
        and attack.dephasing_theta > 0.0
        and attack.information_probability > 0.0
    ):
        observation_probability = _clip_probability(
            math.sin(float(attack.dephasing_theta)) * attack.information_probability
        )
        if observation_probability > 0.0:
            observed = int(rng.binomial(delivered_pair_count, observation_probability))
            if observed > 0:
                eve_observations["dephasing_probe"] = observed

    chsh_correlations_observed, chsh_s_observed = _observed_chsh_from_counts(
        outcome_counts,
        protocol,
    )

    sifted_key_count = key_pair_count
    qber_key = (
        key_error_count / sifted_key_count
        if sifted_key_count > 0
        else None
    )
    qber_abort = (
        qber_key is not None
        and sifted_key_count >= protocol.min_key_pairs_for_qber_abort
        and qber_key > protocol.qber_abort_threshold
    )
    chsh_abort = (
        not qber_abort
        and chsh_s_observed is not None
        and chsh_pair_count >= protocol.min_chsh_pairs_for_chsh_abort
        and chsh_s_observed < protocol.chsh_abort_threshold
    )
    public_outcome = _public_outcome(delivered_pair_count, qber_abort, chsh_abort)
    finite_key = _finite_key_estimate(protocol, sifted_key_count, qber_key)
    information_exposed = (
        public_outcome == "accepted"
        and sifted_key_count > 0
        and prepared.eve_guess_probability > 0.5 + 1e-12
        and _attack_can_expose_information(attack, prepared.target_active)
    )
    secure_delivery = public_outcome == "accepted" and not information_exposed
    eve_win = public_outcome != "accepted" or information_exposed
    reward_components = {
        "delivery_failure": public_outcome == "delivery_failure",
        "qber_abort": qber_abort,
        "chsh_abort": chsh_abort,
        "information_exposure": information_exposed,
        "secure_delivery": secure_delivery,
    }
    provenance = _prepared_provenance(
        prepared,
        seed,
        step_index,
        training_light=training_light,
    )
    transcript_summary = _prepared_transcript_summary(
        prepared,
        setting_pair_counts,
        outcome_counts,
        training_light=training_light,
    )
    result = DirectE91StepResult(
        backend=BACKEND,
        backend_scope=BACKEND_SCOPE,
        no_sequence_runtime=True,
        model_version=MODEL_VERSION,
        alice_node_id=prepared.alice_node_id,
        bob_node_id=prepared.bob_node_id,
        route_id=route.route_id,
        selected_route=route.to_dict(),
        attack_id=attack.attack_id,
        attack_kind=attack.attack_kind,
        target_kind=attack.target_kind,
        target_id=attack.target_id,
        target_active=prepared.target_active,
        requested_pair_count=requested_pair_count,
        delivered_pair_count=delivered_pair_count,
        delivery_probability=prepared.delivery_probability,
        setting_pair_counts=setting_pair_counts,
        key_pair_count=key_pair_count,
        chsh_pair_count=chsh_pair_count,
        discarded_setting_pair_count=discarded_setting_pair_count,
        key_basis_count=key_pair_count,
        test_basis_count=chsh_pair_count,
        sifted_key_count=sifted_key_count,
        key_error_count=key_error_count,
        qber_key=qber_key,
        qber_abort=bool(qber_abort),
        chsh_correlations_expected=prepared.chsh_correlations_expected,
        chsh_S_expected=prepared.chsh_S_expected,
        chsh_correlations_observed=chsh_correlations_observed,
        chsh_S_observed=chsh_s_observed,
        chsh_correlations=chsh_correlations_observed,
        chsh_S=chsh_s_observed,
        chsh_abort=bool(chsh_abort),
        public_outcome=public_outcome,
        information_exposed=information_exposed,
        eve_observation_count=sum(eve_observations.values()),
        eve_observation_distribution=dict(sorted(eve_observations.items())),
        eve_guess_probability=float(prepared.eve_guess_probability),
        secure_delivery=secure_delivery,
        finite_key_estimate_bits=finite_key,
        finite_key_formula_id=protocol.finite_key_formula_id,
        finite_key_claim_scope="diagnostic_estimate",
        elementary_pair_fidelity=prepared.elementary_pair_fidelity,
        effective_end_to_end_fidelity_model=prepared.effective_end_to_end_fidelity_model,
        actual_final_bell_fidelity=prepared.actual_final_bell_fidelity,
        link_fidelity_product_diagnostic=prepared.link_fidelity_product_diagnostic,
        route_fidelity_application=prepared.route_fidelity_application,
        source_pair_success_application="per_elementary_link",
        route_state_model="explicit_elementary_links_with_bsm_swapping",
        bsm_swap_count=prepared.bsm_swap_count,
        elementary_link_count=prepared.elementary_link_count,
        bsm_outcome_probabilities=prepared.bsm_outcome_probabilities,
        pauli_frame_corrections_applied=prepared.pauli_frame_corrections_applied,
        pauli_frame_correction_model=prepared.pauli_frame_correction_model,
        target_memory_side_effective=prepared.target_memory_side_effective,
        edge_attack_subsystem=prepared.edge_attack_subsystem,
        edge_attack_edge_id=prepared.edge_attack_edge_id,
        eve_guess_model=prepared.eve_guess_model,
        eve_win=eve_win,
        reward_components=reward_components,
        provenance=provenance,
        transcript_summary=transcript_summary,
    )
    return result


def run_direct_e91_transcript_step(
    step_input: DirectE91StepInput,
    prepared_cache: dict[Any, DirectE91PreparedTransition] | None = None,
    *,
    training_light: bool = False,
) -> DirectE91StepResult:
    if prepared_cache is None:
        prepared = prepare_direct_e91_transition(
            step_input,
            training_light=training_light,
        )
    else:
        target_active = _attack_target_active(step_input.eve_attack, step_input.selected_route)
        cache_key = _prepared_transition_cache_key(
            step_input,
            target_active,
            training_light=training_light,
        )
        prepared = prepared_cache.get(cache_key)
        if prepared is None:
            prepared = prepare_direct_e91_transition(
                step_input,
                training_light=training_light,
            )
            prepared_cache[cache_key] = prepared
    return sample_prepared_direct_e91_transition(
        prepared,
        step_input.seed,
        step_input.step_index,
        requested_pairs=step_input.protocol.requested_pairs,
        training_light=training_light,
    )


def _clean_projector(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(np.real_if_close(matrix), dtype=complex)


def _clean_density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    rho = (rho + rho.conj().T) / 2.0
    trace = complex(np.trace(rho))
    if abs(trace) <= 0.0:
        raise ValueError("density matrix trace must be nonzero")
    rho = rho / trace
    return np.asarray(np.real_if_close(rho), dtype=complex)


def _depolarizing_kraus(probability: float) -> tuple[np.ndarray, ...]:
    p = _clip_probability(probability)
    return (
        math.sqrt(max(0.0, 1.0 - p)) * _I2,
        math.sqrt(p / 3.0) * _X,
        math.sqrt(p / 3.0) * _Y,
        math.sqrt(p / 3.0) * _Z,
    )


def _dephasing_kraus(probability: float, basis: str = "Z") -> tuple[np.ndarray, ...]:
    p = _clip_probability(probability)
    op = _Z if basis == "Z" else _X
    return (
        math.sqrt(max(0.0, 1.0 - p)) * _I2,
        math.sqrt(p) * op,
    )


def _bit_flip_kraus(probability: float) -> tuple[np.ndarray, ...]:
    p = _clip_probability(probability)
    return (
        math.sqrt(max(0.0, 1.0 - p)) * _I2,
        math.sqrt(p) * _X,
    )


def _two_qubit_depolarizing_kraus(probability: float) -> tuple[np.ndarray, ...]:
    p = _clip_probability(probability)
    ops = [math.sqrt(max(0.0, 1.0 - p)) * _I4]
    paulis = (_I2, _X, _Y, _Z)
    non_identity = [
        np.kron(a, b)
        for a in paulis
        for b in paulis
        if not (np.array_equal(a, _I2) and np.array_equal(b, _I2))
    ]
    ops.extend(math.sqrt(p / len(non_identity)) * op for op in non_identity)
    return tuple(ops)


def bell_projectors() -> dict[str, np.ndarray]:
    return {
        label: bell_state_density(label)
        for label in ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
    }


def bell_projector(label: str) -> np.ndarray:
    return bell_projectors()[label]


def elementary_link_density(
    edge_id: str,
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
) -> np.ndarray:
    fidelity = _clip_probability(
        route.edge_fidelities.get(edge_id, hardware.elementary_pair_fidelity)
    )
    return werner_state(fidelity, protocol.target_bell_label)


def apply_pauli_correction_for_bsm_outcome(
    rho_ab: np.ndarray,
    outcome_label: str,
    target_bell_label: str,
) -> np.ndarray:
    correction_label = _pauli_correction_label_for_bsm_outcome(
        outcome_label,
        target_bell_label,
    )
    unitary = np.kron(_I2, _pauli_by_label(correction_label))
    return _clean_density(unitary @ rho_ab @ unitary.conj().T)


def swap_two_links(
    left_rho: np.ndarray,
    right_rho: np.ndarray,
    *,
    target_bell_label: str,
    bsm_success: bool = True,
) -> SwapResult:
    return _swap_four_qubit_state(
        np.kron(left_rho, right_rho),
        target_bell_label=target_bell_label,
        bsm_success=bsm_success,
    )


def partial_trace_repeater_pair(rho_arlc: np.ndarray) -> np.ndarray:
    return _partial_trace_qubits(rho_arlc, keep=(0, 3), n_qubits=4)


def route_density_matrix_from_elementary_swaps(
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
    attack: DirectE91AttackSpec | None = None,
    target_active: bool | None = None,
) -> RouteStateResult:
    if not route.edge_ids:
        raise ValueError("route must include at least one edge")
    attack = attack or DirectE91AttackSpec()
    if target_active is None:
        target_active = _attack_target_active(attack, route)

    route_state_model = "explicit_elementary_links_with_bsm_swapping"
    link_fidelity_product = _route_elementary_fidelity_product(route, hardware)
    route_fidelity_application = "actual_final_bell_fidelity_from_explicit_bsm_state"
    channel_record: dict[str, Any] = {
        "route_state_model": route_state_model,
        "route_fidelity_application": route_fidelity_application,
        "link_fidelity_product_diagnostic": link_fidelity_product,
        "link_fidelity_product_application": (
            "product_of_configured_elementary_link_fidelities"
        ),
        "elementary_link_fidelities": {
            edge_id: _edge_elementary_fidelity(edge_id, route, hardware)
            for edge_id in route.edge_ids
        },
        "edge_transmittance_source": {
            edge_id: _edge_transmittance_source(edge_id, route)
            for edge_id in route.edge_ids
        },
        "attack_channel": "none",
        "target_memory_side_effective": None,
        "edge_attack_subsystem": None,
        "edge_attack_edge_id": None,
        "memory_noise_applied_before_bsm": bool(
            hardware.memory_depolarizing_probability > 0.0
            or hardware.memory_dephasing_probability > 0.0
        ),
        "swap_depolarizing_applied_after_bsm": bool(
            hardware.swap_depolarizing_probability > 0.0
        ),
    }
    if target_active and attack.attack_kind == "edge_added_loss":
        channel_record["attack_channel"] = "edge_added_loss_delivery_only"
    if target_active and attack.attack_kind == "swap_denial":
        channel_record["attack_channel"] = "swap_denial_delivery_only"

    branches = _elementary_link_branches(
        route.edge_ids[0],
        route,
        hardware,
        protocol,
        attack,
        target_active,
        channel_record,
    )
    bsm_records: list[dict[str, Any]] = []
    pauli_corrections_applied = bool(route.internal_nodes)
    pauli_frame_correction_model = (
        "nonselective_bsm_outcome_correction"
        if route.internal_nodes
        else "not_applicable_no_bsm"
    )

    for swap_index, repeater_node in enumerate(route.internal_nodes):
        right_edge_id = route.edge_ids[swap_index + 1]
        right_branches = _elementary_link_branches(
            right_edge_id,
            route,
            hardware,
            protocol,
            attack,
            target_active,
            channel_record,
        )
        combined = _combine_link_branch_ensembles(branches, right_branches)
        combined = _apply_repeater_memory_operations(
            combined,
            repeater_node,
            hardware,
            attack,
            bool(target_active),
            channel_record,
        )

        next_branches: list[QuantumBranch] = []
        aggregate_outcomes = {label: 0.0 for label in bell_projectors()}
        aggregate_weight = 0.0
        for branch in combined:
            if branch.probability <= 0.0:
                continue
            aggregate_weight += branch.probability
            for outcome_label in bell_projectors():
                branch_state, outcome_probability = _bsm_branch_state_no_correction(
                    branch.rho,
                    outcome_label,
                )
                aggregate_outcomes[outcome_label] += branch.probability * outcome_probability
                if outcome_probability <= 0.0:
                    continue
                corrected_rho = apply_pauli_correction_for_bsm_outcome(
                    branch_state,
                    outcome_label,
                    protocol.target_bell_label,
                )
                if hardware.swap_depolarizing_probability > 0.0:
                    corrected_rho = apply_two_qubit_channel(
                        corrected_rho,
                        _two_qubit_depolarizing_kraus(
                            hardware.swap_depolarizing_probability
                        ),
                    )
                next_branches.append(
                    QuantumBranch(
                        probability=branch.probability * outcome_probability,
                        rho=corrected_rho,
                        eve_observation_labels=branch.eve_observation_labels,
                        public_bsm_labels=branch.public_bsm_labels
                        + (f"{repeater_node}:{outcome_label}",),
                        eve_outcome=branch.eve_outcome,
                    )
                )
        if aggregate_weight > 0.0:
            aggregate_outcomes = {
                label: probability / aggregate_weight
                for label, probability in aggregate_outcomes.items()
            }
        bsm_records.append({
            "swap_index": swap_index,
            "repeater_node_id": repeater_node,
            "outcome_probabilities": {
                label: float(probability)
                for label, probability in sorted(aggregate_outcomes.items())
            },
            "pauli_frame_corrections": _pauli_correction_map(
                protocol.target_bell_label
            ),
            "nonselective_corrected_state": True,
        })
        branches = _normalize_quantum_branches(next_branches)

    public_state = _public_state_from_branches(branches)
    branch_model = _branch_model_from_branches(branches)
    actual_bell_fidelity = _bell_fidelity(public_state, protocol.target_bell_label)
    channel_record["actual_final_bell_fidelity"] = actual_bell_fidelity
    return RouteStateResult(
        rho=public_state,
        branch_model=branch_model,
        channel_record=channel_record,
        elementary_link_count=len(route.edge_ids),
        bsm_swap_count=len(route.internal_nodes),
        bsm_outcome_probabilities=bsm_records,
        pauli_frame_corrections_applied=pauli_corrections_applied,
        target_memory_side_effective=channel_record["target_memory_side_effective"],
        edge_attack_subsystem=channel_record["edge_attack_subsystem"],
        edge_attack_edge_id=channel_record["edge_attack_edge_id"],
        effective_end_to_end_fidelity_model=actual_bell_fidelity,
        actual_final_bell_fidelity=actual_bell_fidelity,
        link_fidelity_product_diagnostic=link_fidelity_product,
        route_fidelity_application=route_fidelity_application,
        pauli_frame_correction_model=pauli_frame_correction_model,
    )


def _swap_four_qubit_state(
    rho_arlc: np.ndarray,
    *,
    target_bell_label: str,
    bsm_success: bool = True,
) -> SwapResult:
    if not bsm_success:
        return SwapResult(
            rho=_clean_density(partial_trace_repeater_pair(rho_arlc)),
            outcome_probabilities={label: 0.0 for label in bell_projectors()},
            pauli_frame_corrections_applied=False,
            target_bell_label=target_bell_label,
        )

    corrected = np.zeros((4, 4), dtype=complex)
    outcome_probabilities: dict[str, float] = {}
    total_probability = 0.0
    for outcome_label in bell_projectors():
        branch_state, branch_probability = _bsm_branch_state_no_correction(
            rho_arlc,
            outcome_label,
        )
        outcome_probabilities[outcome_label] = branch_probability
        if branch_probability <= 0.0:
            continue
        corrected += branch_probability * apply_pauli_correction_for_bsm_outcome(
            branch_state,
            outcome_label,
            target_bell_label,
        )
        total_probability += branch_probability
    if total_probability <= 0.0:
        corrected_state = _clean_density(partial_trace_repeater_pair(rho_arlc))
    else:
        corrected_state = _clean_density(corrected / total_probability)
        outcome_probabilities = {
            label: probability / total_probability
            for label, probability in outcome_probabilities.items()
        }
    return SwapResult(
        rho=corrected_state,
        outcome_probabilities={
            label: float(probability)
            for label, probability in sorted(outcome_probabilities.items())
        },
        pauli_frame_corrections_applied=True,
        target_bell_label=target_bell_label,
    )


def _bsm_branch_state_no_correction(
    rho_arlc: np.ndarray,
    outcome_label: str,
) -> tuple[np.ndarray, float]:
    projector = _bsm_projector_on_repeater_pair(outcome_label)
    projected = projector @ rho_arlc @ projector.conj().T
    unnormalized = partial_trace_repeater_pair(projected)
    probability = float(np.real_if_close(np.trace(unnormalized)))
    probability = max(0.0, probability)
    if probability <= 0.0:
        return _clean_density(partial_trace_repeater_pair(rho_arlc)), 0.0
    return _clean_density(unnormalized / probability), probability


def _bsm_projector_on_repeater_pair(outcome_label: str) -> np.ndarray:
    return np.kron(np.kron(_I2, bell_projector(outcome_label)), _I2)


@lru_cache(maxsize=None)
def _pauli_correction_label_for_bsm_outcome(
    outcome_label: str,
    target_bell_label: str,
) -> str:
    ideal_four_qubit = np.kron(
        bell_state_density(target_bell_label),
        bell_state_density(target_bell_label),
    )
    raw_state, probability = _bsm_branch_state_no_correction(
        ideal_four_qubit,
        outcome_label,
    )
    if probability <= 0.0:
        return "I"
    target = bell_state_density(target_bell_label)
    best_label = "I"
    best_overlap = -1.0
    for label in ("I", "X", "Y", "Z"):
        pauli = _pauli_by_label(label)
        unitary = np.kron(_I2, pauli)
        corrected = unitary @ raw_state @ unitary.conj().T
        overlap = float(np.real_if_close(np.trace(target @ corrected)))
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    return best_label


def _pauli_correction_map(target_bell_label: str) -> dict[str, str]:
    return {
        outcome_label: _pauli_correction_label_for_bsm_outcome(
            outcome_label,
            target_bell_label,
        )
        for outcome_label in sorted(bell_projectors())
    }


def _pauli_by_label(label: str) -> np.ndarray:
    if label == "I":
        return _I2
    if label == "X":
        return _X
    if label == "Y":
        return _Y
    if label == "Z":
        return _Z
    raise ValueError(f"unknown Pauli label {label!r}")


def _partial_trace_qubits(
    rho: np.ndarray,
    *,
    keep: tuple[int, ...],
    n_qubits: int,
) -> np.ndarray:
    keep = tuple(sorted(keep))
    row_labels = list("abcdefghijklmnopqrstuvwxyz"[:n_qubits])
    col_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_qubits])
    for qubit in range(n_qubits):
        if qubit not in keep:
            col_labels[qubit] = row_labels[qubit]
    output_labels = [row_labels[qubit] for qubit in keep] + [
        col_labels[qubit] for qubit in keep
    ]
    tensor = rho.reshape((2,) * (2 * n_qubits))
    if not output_labels:
        return np.asarray([[np.einsum("".join(row_labels + col_labels), tensor)]])
    expression = (
        "".join(row_labels + col_labels)
        + "->"
        + "".join(output_labels)
    )
    result = np.einsum(expression, tensor)
    dim = 2 ** len(keep)
    return result.reshape(dim, dim)


def _operator_on_n_qubits(
    op: np.ndarray,
    *,
    n_qubits: int,
    qubit_index: int,
) -> np.ndarray:
    if qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError("qubit_index out of range")
    lifted = np.array([[1.0]], dtype=complex)
    for index in range(n_qubits):
        lifted = np.kron(lifted, op if index == qubit_index else _I2)
    return lifted


def _apply_one_qubit_channel_n(
    rho: np.ndarray,
    *,
    n_qubits: int,
    qubit_index: int,
    kraus_ops: Iterable[np.ndarray],
) -> np.ndarray:
    dim = 2 ** n_qubits
    result = np.zeros((dim, dim), dtype=complex)
    for op in kraus_ops:
        lifted = _operator_on_n_qubits(
            np.asarray(op, dtype=complex),
            n_qubits=n_qubits,
            qubit_index=qubit_index,
        )
        result += lifted @ rho @ lifted.conj().T
    return _clean_density(result)


def _edge_elementary_fidelity(
    edge_id: str,
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
) -> float:
    return _clip_probability(
        route.edge_fidelities.get(edge_id, hardware.elementary_pair_fidelity)
    )


def _route_elementary_fidelity_product(
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
) -> float:
    fidelity = 1.0
    for edge_id in route.edge_ids:
        fidelity *= _edge_elementary_fidelity(edge_id, route, hardware)
    return _clip_probability(fidelity)


def _bell_fidelity(rho: np.ndarray, target_bell_label: str) -> float:
    projector = bell_projector(target_bell_label)
    return _clip_probability(float(np.real_if_close(np.trace(projector @ rho))))


def _elementary_link_branches(
    edge_id: str,
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
    attack: DirectE91AttackSpec,
    target_active: bool,
    channel_record: dict[str, Any],
) -> list[QuantumBranch]:
    branches = [
        QuantumBranch(
            probability=1.0,
            rho=elementary_link_density(edge_id, route, hardware, protocol),
        )
    ]
    if (
        target_active
        and attack.attack_kind == "edge_intercept_resend"
        and attack.target_kind == "edge"
        and attack.target_id == edge_id
    ):
        branches = _projective_branch_ensemble(
            branches,
            n_qubits=2,
            qubit_index=1,
            basis=attack.intercept_basis_label,
            probability=attack.intercept_probability,
            information_probability=attack.information_probability,
            label_prefix=f"intercept_{edge_id}_receiver",
        )
        channel_record["attack_channel"] = "edge_intercept_resend"
        channel_record["basis"] = attack.intercept_basis_label
        channel_record["probability"] = float(attack.intercept_probability)
        channel_record["information_probability"] = float(attack.information_probability)
        channel_record["edge_attack_subsystem"] = "receiver_side_qubit"
        channel_record["edge_attack_edge_id"] = edge_id
    return _normalize_quantum_branches(branches)


def _combine_link_branch_ensembles(
    left_branches: list[QuantumBranch],
    right_branches: list[QuantumBranch],
) -> list[QuantumBranch]:
    combined: list[QuantumBranch] = []
    for left in left_branches:
        for right in right_branches:
            probability = left.probability * right.probability
            if probability <= 0.0:
                continue
            combined.append(
                QuantumBranch(
                    probability=probability,
                    rho=np.kron(left.rho, right.rho),
                    eve_observation_labels=(
                        left.eve_observation_labels
                        + right.eve_observation_labels
                    ),
                    public_bsm_labels=(
                        left.public_bsm_labels
                        + right.public_bsm_labels
                    ),
                    eve_outcome=_combine_eve_outcome(
                        left.eve_outcome,
                        right.eve_outcome,
                    ),
                )
            )
    return _normalize_quantum_branches(combined)


def _combine_eve_outcome(
    left: int | None,
    right: int | None,
) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return None


def _apply_repeater_memory_operations(
    branches: list[QuantumBranch],
    repeater_node: str,
    hardware: DirectE91HardwareConfig,
    attack: DirectE91AttackSpec,
    target_active: bool,
    channel_record: dict[str, Any],
) -> list[QuantumBranch]:
    operated = branches
    for qubit_index in (1, 2):
        if hardware.memory_depolarizing_probability > 0.0:
            operated = _apply_branch_channel(
                operated,
                n_qubits=4,
                qubit_index=qubit_index,
                kraus_ops=_depolarizing_kraus(
                    hardware.memory_depolarizing_probability
                ),
            )
        if hardware.memory_dephasing_probability > 0.0:
            operated = _apply_branch_channel(
                operated,
                n_qubits=4,
                qubit_index=qubit_index,
                kraus_ops=_dephasing_kraus(
                    hardware.memory_dephasing_probability,
                    "Z",
                ),
            )

    if not (
        target_active
        and attack.target_kind in {"node", "memory"}
        and attack.target_id == repeater_node
    ):
        return operated
    if attack.attack_kind not in {
        "memory_degradation",
        "repeater_memory_measure_Z",
        "repeater_memory_measure_X",
        "repeater_memory_dephasing_probe",
    }:
        return operated

    side = getattr(attack, "target_memory_side", "both")
    qubit_indices = _memory_side_qubits(side)
    channel_record["target_memory_side_effective"] = side

    if attack.attack_kind == "memory_degradation":
        depol = attack.memory_depolarizing_probability
        dephase = attack.memory_dephasing_probability
        for qubit_index in qubit_indices:
            if depol > 0.0:
                operated = _apply_branch_channel(
                    operated,
                    n_qubits=4,
                    qubit_index=qubit_index,
                    kraus_ops=_depolarizing_kraus(depol),
                )
            if dephase > 0.0:
                operated = _apply_branch_channel(
                    operated,
                    n_qubits=4,
                    qubit_index=qubit_index,
                    kraus_ops=_dephasing_kraus(dephase, "Z"),
                )
        channel_record["attack_channel"] = "memory_degradation"
        channel_record["memory_depolarizing_probability"] = float(depol)
        channel_record["memory_dephasing_probability"] = float(dephase)
        return operated

    if attack.attack_kind in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
        basis = "Z" if attack.attack_kind.endswith("_Z") else "X"
        if len(qubit_indices) > 1:
            operated = _projective_branch_ensemble_joint(
                operated,
                n_qubits=4,
                qubit_indices=qubit_indices,
                basis=basis,
                probability=attack.measurement_probability,
                information_probability=attack.information_probability,
                label_prefixes=tuple(
                    f"memory_measure_{basis}_{repeater_node}_{'left' if qubit == 1 else 'right'}"
                    for qubit in qubit_indices
                ),
            )
            channel_record["measurement_probability_semantics"] = (
                "single_joint_attempt_for_selected_memory_sides"
            )
        else:
            qubit_index = qubit_indices[0]
            side_label = "left" if qubit_index == 1 else "right"
            operated = _projective_branch_ensemble(
                operated,
                n_qubits=4,
                qubit_index=qubit_index,
                basis=basis,
                probability=attack.measurement_probability,
                information_probability=attack.information_probability,
                label_prefix=f"memory_measure_{basis}_{repeater_node}_{side_label}",
            )
            channel_record["measurement_probability_semantics"] = (
                "single_attempt_for_selected_memory_side"
            )
        channel_record["attack_channel"] = attack.attack_kind
        channel_record["basis"] = basis
        channel_record["probability"] = float(attack.measurement_probability)
        channel_record["information_probability"] = float(attack.information_probability)
        return operated

    if attack.attack_kind == "repeater_memory_dephasing_probe":
        theta = float(attack.dephasing_theta)
        p = (1.0 - math.cos(theta)) / 2.0
        for qubit_index in qubit_indices:
            operated = _apply_branch_channel(
                operated,
                n_qubits=4,
                qubit_index=qubit_index,
                kraus_ops=_dephasing_kraus(p, "Z"),
            )
        channel_record["attack_channel"] = "repeater_memory_dephasing_probe"
        channel_record["theta"] = theta
        channel_record["dephasing_probability"] = p
        return operated

    return operated


def _memory_side_qubits(side: str) -> tuple[int, ...]:
    if side == "left":
        return (1,)
    if side == "right":
        return (2,)
    return (1, 2)


def _apply_branch_channel(
    branches: list[QuantumBranch],
    *,
    n_qubits: int,
    qubit_index: int,
    kraus_ops: Iterable[np.ndarray],
) -> list[QuantumBranch]:
    return [
        QuantumBranch(
            probability=branch.probability,
            rho=_apply_one_qubit_channel_n(
                branch.rho,
                n_qubits=n_qubits,
                qubit_index=qubit_index,
                kraus_ops=kraus_ops,
            ),
            eve_observation_labels=branch.eve_observation_labels,
            public_bsm_labels=branch.public_bsm_labels,
            eve_outcome=branch.eve_outcome,
        )
        for branch in branches
    ]


def _projective_branch_ensemble(
    branches: list[QuantumBranch],
    *,
    n_qubits: int,
    qubit_index: int,
    basis: str,
    probability: float,
    information_probability: float,
    label_prefix: str,
    normalize: bool = True,
) -> list[QuantumBranch]:
    probability = _clip_probability(probability)
    information_probability = _clip_probability(information_probability)
    projectors = projectors_for_axis(_axis_for_basis(basis))
    expanded: list[QuantumBranch] = []
    if probability < 1.0:
        expanded.extend(
            QuantumBranch(
                probability=branch.probability * (1.0 - probability),
                rho=branch.rho,
                eve_observation_labels=branch.eve_observation_labels,
                public_bsm_labels=branch.public_bsm_labels,
                eve_outcome=branch.eve_outcome,
            )
            for branch in branches
        )
    for branch in branches:
        for outcome, projector in enumerate(projectors):
            op = _operator_on_n_qubits(
                projector,
                n_qubits=n_qubits,
                qubit_index=qubit_index,
            )
            unnormalized = op @ branch.rho @ op.conj().T
            branch_probability = float(np.real_if_close(np.trace(unnormalized)))
            branch_probability = max(0.0, branch_probability)
            if branch_probability <= 0.0:
                continue
            measured_probability = branch.probability * probability * branch_probability
            measured_state = _clean_density(unnormalized / branch_probability)
            if information_probability > 0.0:
                expanded.append(
                    QuantumBranch(
                        probability=measured_probability * information_probability,
                        rho=measured_state,
                        eve_observation_labels=(
                            branch.eve_observation_labels
                            + (f"{label_prefix}_{outcome}",)
                        ),
                        public_bsm_labels=branch.public_bsm_labels,
                        eve_outcome=(
                            outcome if branch.eve_outcome is None else None
                        ),
                    )
                )
            if information_probability < 1.0:
                expanded.append(
                    QuantumBranch(
                        probability=measured_probability * (1.0 - information_probability),
                        rho=measured_state,
                        eve_observation_labels=branch.eve_observation_labels,
                        public_bsm_labels=branch.public_bsm_labels,
                        eve_outcome=branch.eve_outcome,
                    )
                )
    if normalize:
        return _normalize_quantum_branches(expanded)
    return [
        QuantumBranch(
            probability=float(branch.probability),
            rho=_clean_density(branch.rho),
            eve_observation_labels=branch.eve_observation_labels,
            public_bsm_labels=branch.public_bsm_labels,
            eve_outcome=branch.eve_outcome,
        )
        for branch in expanded
        if branch.probability > 0.0
    ]


def _projective_branch_ensemble_joint(
    branches: list[QuantumBranch],
    *,
    n_qubits: int,
    qubit_indices: tuple[int, ...],
    basis: str,
    probability: float,
    information_probability: float,
    label_prefixes: tuple[str, ...],
) -> list[QuantumBranch]:
    probability = _clip_probability(probability)
    expanded: list[QuantumBranch] = []
    if probability < 1.0:
        expanded.extend(
            QuantumBranch(
                probability=branch.probability * (1.0 - probability),
                rho=branch.rho,
                eve_observation_labels=branch.eve_observation_labels,
                public_bsm_labels=branch.public_bsm_labels,
                eve_outcome=branch.eve_outcome,
            )
            for branch in branches
        )
    measured = [
        QuantumBranch(
            probability=branch.probability * probability,
            rho=branch.rho,
            eve_observation_labels=branch.eve_observation_labels,
            public_bsm_labels=branch.public_bsm_labels,
            eve_outcome=branch.eve_outcome,
        )
        for branch in branches
        if probability > 0.0
    ]
    for qubit_index, label_prefix in zip(qubit_indices, label_prefixes):
        measured = _projective_branch_ensemble(
            measured,
            n_qubits=n_qubits,
            qubit_index=qubit_index,
            basis=basis,
            probability=1.0,
            information_probability=information_probability,
            label_prefix=label_prefix,
            normalize=False,
        )
    expanded.extend(measured)
    return _normalize_quantum_branches(expanded)


def _normalize_quantum_branches(
    branches: list[QuantumBranch],
) -> list[QuantumBranch]:
    total = sum(float(branch.probability) for branch in branches)
    if total <= 0.0:
        return []
    return [
        QuantumBranch(
            probability=float(branch.probability) / total,
            rho=_clean_density(branch.rho),
            eve_observation_labels=branch.eve_observation_labels,
            public_bsm_labels=branch.public_bsm_labels,
            eve_outcome=branch.eve_outcome,
        )
        for branch in branches
        if branch.probability > 0.0
    ]


def _public_state_from_branches(branches: list[QuantumBranch]) -> np.ndarray:
    public_state = np.zeros((4, 4), dtype=complex)
    for branch in branches:
        public_state += branch.probability * branch.rho
    return _clean_density(public_state)


def _branch_model_from_branches(
    branches: list[QuantumBranch],
) -> dict[str, Any] | None:
    if not any(branch.eve_observation_labels for branch in branches):
        return None
    normalized = _normalize_quantum_branches(branches)
    return {
        "branches": [
            {
                "label": _branch_condition_label(
                    branch.eve_observation_labels,
                    branch.public_bsm_labels,
                ),
                "eve_observation_label": (
                    "|".join(branch.eve_observation_labels)
                    if branch.eve_observation_labels
                    else "none"
                ),
                "eve_observation_labels": tuple(branch.eve_observation_labels),
                "public_bsm_label": (
                    "|".join(branch.public_bsm_labels)
                    if branch.public_bsm_labels
                    else "none"
                ),
                "public_bsm_labels": tuple(branch.public_bsm_labels),
                "probability": float(branch.probability),
                "state": branch.rho,
                "outcome": branch.eve_outcome,
            }
            for branch in normalized
        ],
        "public_state": _public_state_from_branches(normalized),
    }


def _branch_condition_label(
    eve_observation_labels: tuple[str, ...],
    public_bsm_labels: tuple[str, ...],
) -> str:
    eve = "|".join(eve_observation_labels) if eve_observation_labels else "none"
    bsm = "|".join(public_bsm_labels) if public_bsm_labels else "none"
    return f"eve={eve};public_bsm={bsm}"


def _route_delivery_probability(
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
    attack: DirectE91AttackSpec,
    target_active: bool,
) -> float:
    probability = 1.0
    for edge_id in route.edge_ids:
        probability *= float(hardware.source_pair_success_probability)
        eta = _edge_delivery_transmittance(edge_id, route, hardware)
        if (
            target_active
            and attack.attack_kind == "edge_added_loss"
            and attack.target_kind == "edge"
            and attack.target_id == edge_id
        ):
            eta *= attack.loss_multiplier
        probability *= eta
    endpoint_detection = 1.0 - (
        (1.0 - hardware.detector_efficiency) * (1.0 - hardware.dark_count_probability)
    )
    probability *= endpoint_detection * endpoint_detection
    swap_count = len(route.internal_nodes)
    swap_success = float(hardware.swap_success_probability)
    if target_active and attack.attack_kind == "swap_denial":
        swap_success = 0.0
    probability *= swap_success ** swap_count
    return _clip_probability(probability)


def _edge_delivery_transmittance(
    edge_id: str,
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
) -> float:
    if edge_id in route.edge_transmittances:
        return _clip_probability(route.edge_transmittances[edge_id])
    if edge_id in route.edge_loss_db:
        return _clip_probability(10.0 ** (-float(route.edge_loss_db[edge_id]) / 10.0))
    length_km = float(route.edge_lengths_m.get(edge_id, 0.0)) / 1000.0
    return _clip_probability(
        10.0 ** (-(hardware.fiber_loss_db_per_km * length_km) / 10.0)
    )


def _edge_transmittance_source(edge_id: str, route: DirectE91Route) -> str:
    if edge_id in route.edge_loss_db:
        return "explicit_edge_loss_db"
    if edge_id in route.edge_transmittances:
        return "explicit_edge_transmittance"
    return "fiber_loss_db_per_km_times_edge_length"


def _effective_route_fidelity(
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
) -> float:
    return _route_elementary_fidelity_product(route, hardware)


def _route_density_matrix(
    route: DirectE91Route,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
    effective_fidelity: float | None = None,
) -> np.ndarray:
    _ = effective_fidelity
    return route_density_matrix_from_elementary_swaps(
        route,
        hardware,
        protocol,
    ).rho


def _apply_measurement_noise(
    rho: np.ndarray,
    hardware: DirectE91HardwareConfig,
) -> np.ndarray:
    if hardware.misalignment_probability <= 0.0:
        return rho
    rho = apply_one_qubit_channel(
        rho,
        0,
        _bit_flip_kraus(hardware.misalignment_probability),
    )
    rho = apply_one_qubit_channel(
        rho,
        1,
        _bit_flip_kraus(hardware.misalignment_probability),
    )
    return rho


def _axis_for_basis(basis: str) -> tuple[float, float, float]:
    if basis == "Z":
        return (0.0, 0.0, 1.0)
    if basis == "X":
        return (1.0, 0.0, 0.0)
    raise ValueError("basis must be Z or X")


def _validate_step_route(step_input: DirectE91StepInput) -> None:
    route = step_input.selected_route
    if route.node_path[0] != step_input.alice_node_id:
        raise ValueError("selected_route must start at alice_node_id")
    if route.node_path[-1] != step_input.bob_node_id:
        raise ValueError("selected_route must end at bob_node_id")
    if len(set(route.node_path)) != len(route.node_path):
        raise ValueError("selected_route node_path must be simple")
    if len(set(route.edge_ids)) != len(route.edge_ids):
        raise ValueError("selected_route edge_ids must be unique")
    if len(route.edge_ids) != len(route.node_path) - 1:
        raise ValueError("selected_route edge_ids must match route hops")

    topology = step_input.topology
    for edge_id, u, v in zip(route.edge_ids, route.node_path, route.node_path[1:]):
        topology_edge_id = _topology_edge_id_between(topology, u, v)
        if topology_edge_id is not None and topology_edge_id != edge_id:
            raise ValueError(
                f"route edge_id {edge_id!r} does not match topology edge "
                f"{topology_edge_id!r} for hop {u!r}->{v!r}"
            )


def _topology_edge_id_between(
    topology: Any,
    u: str,
    v: str,
) -> str | None:
    if topology is None:
        return None
    if hasattr(topology, "edge_between"):
        edge = topology.edge_between(u, v)
        if edge is None:
            raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")
        return str(edge.edge_id)
    if hasattr(topology, "to_dict"):
        return _topology_edge_id_between(topology.to_dict(), u, v)
    if not isinstance(topology, dict):
        return None
    edges = topology.get("edges")
    if edges is None:
        return None
    matches = []
    for edge in edges:
        edge_u = str(edge.get("u", ""))
        edge_v = str(edge.get("v", ""))
        if {edge_u, edge_v} == {u, v}:
            matches.append(edge)
    if not matches:
        raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous topology edge for route hop {u!r}->{v!r}")
    return str(matches[0].get("edge_id", ""))


def _attack_target_active(attack: DirectE91AttackSpec, route: DirectE91Route) -> bool:
    if attack.attack_kind == "none" or attack.target_kind == "none":
        return False
    if attack.target_kind == "edge":
        return attack.target_id in set(route.edge_ids)
    if attack.target_kind in {"node", "memory"}:
        return attack.target_id in set(route.internal_nodes)
    return False


def _setting_pair_distribution(
    protocol: DirectE91ProtocolConfig,
) -> tuple[list[str], np.ndarray, list[tuple[DirectE91MeasurementSetting, DirectE91MeasurementSetting]]]:
    alice_settings = [s for s in protocol.measurement_settings if s.party == "alice"]
    bob_settings = [s for s in protocol.measurement_settings if s.party == "bob"]
    alice_weights = _weights(alice_settings)
    bob_weights = _weights(bob_settings)
    labels: list[str] = []
    probabilities: list[float] = []
    pairs: list[tuple[DirectE91MeasurementSetting, DirectE91MeasurementSetting]] = []
    for a_setting, a_prob in zip(alice_settings, alice_weights):
        for b_setting, b_prob in zip(bob_settings, bob_weights):
            labels.append(f"{a_setting.label}|{b_setting.label}")
            probabilities.append(float(a_prob * b_prob))
            pairs.append((a_setting, b_setting))
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / probs.sum()
    return labels, probs, pairs


def _weights(settings: list[DirectE91MeasurementSetting]) -> np.ndarray:
    weights = np.asarray([s.probability_weight for s in settings], dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(settings), dtype=float)
    return weights / weights.sum()


def _sample_joint_counts(
    rng: np.random.Generator,
    count: int,
    rho: np.ndarray,
    alice_setting: DirectE91MeasurementSetting,
    bob_setting: DirectE91MeasurementSetting,
    hardware: DirectE91HardwareConfig,
) -> list[dict[str, Any]]:
    probs = joint_outcome_probabilities(rho, alice_setting.axis, bob_setting.axis)
    probs = _apply_dark_count_outcome_model(probs, hardware)
    flat = probs.reshape(4)
    sampled = rng.multinomial(count, flat)
    return [
        {
            "alice_outcome": idx // 2,
            "bob_outcome": idx % 2,
            "eve_observation_label": "none",
            "count": int(value),
        }
        for idx, value in enumerate(sampled)
        if int(value) > 0
    ]


def _sample_eve_observations_from_branch_model(
    rng: np.random.Generator,
    delivered_pair_count: int,
    branch_model: dict[str, Any],
) -> dict[str, int]:
    weights_by_label: dict[str, float] = {}
    none_weight = 0.0
    for branch in branch_model.get("branches", []):
        probability = float(branch.get("probability", 0.0))
        if probability <= 0.0:
            continue
        eve_label = str(branch.get("eve_observation_label", "none"))
        if eve_label == "none":
            none_weight += probability
        else:
            weights_by_label[eve_label] = weights_by_label.get(eve_label, 0.0) + probability
    labels = sorted(weights_by_label)
    probabilities = [weights_by_label[label] for label in labels]
    none_probability = max(0.0, none_weight)
    if none_probability > 0.0:
        labels.append("none")
        probabilities.append(none_probability)
    total = sum(probabilities)
    if delivered_pair_count <= 0 or total <= 0.0:
        return {}
    probs = np.asarray(probabilities, dtype=float) / total
    sampled = rng.multinomial(int(delivered_pair_count), probs)
    return {
        label: int(count)
        for label, count in zip(labels, sampled)
        if label != "none" and int(count) > 0
    }


def _apply_dark_count_outcome_model(
    quantum_probs: np.ndarray,
    hardware: DirectE91HardwareConfig,
) -> np.ndarray:
    alice_dark = _dark_only_probability_conditioned_on_detection(hardware)
    bob_dark = alice_dark
    if alice_dark <= 0.0 and bob_dark <= 0.0:
        return quantum_probs

    quantum_probs = np.asarray(quantum_probs, dtype=float).reshape(2, 2)
    alice_marginal = quantum_probs.sum(axis=1)
    bob_marginal = quantum_probs.sum(axis=0)
    alice_random = np.full(2, 0.5, dtype=float)
    bob_random = np.full(2, 0.5, dtype=float)
    mixed = (
        (1.0 - alice_dark) * (1.0 - bob_dark) * quantum_probs
        + alice_dark * (1.0 - bob_dark) * np.outer(alice_random, bob_marginal)
        + (1.0 - alice_dark) * bob_dark * np.outer(alice_marginal, bob_random)
        + alice_dark * bob_dark * np.outer(alice_random, bob_random)
    )
    mixed = np.clip(mixed, 0.0, 1.0)
    total = float(mixed.sum())
    if total <= 0.0:
        return np.full((2, 2), 0.25, dtype=float)
    return mixed / total


def _dark_only_probability_conditioned_on_detection(
    hardware: DirectE91HardwareConfig,
) -> float:
    signal = _clip_probability(hardware.detector_efficiency)
    dark = _clip_probability(hardware.dark_count_probability)
    endpoint_detection = 1.0 - (1.0 - signal) * (1.0 - dark)
    if endpoint_detection <= 0.0:
        return 0.0
    return _clip_probability(((1.0 - signal) * dark) / endpoint_detection)


def _is_key_error(a_outcome: int, b_outcome: int, protocol: DirectE91ProtocolConfig) -> bool:
    if protocol.key_outcome_relation == "correlated":
        return a_outcome != b_outcome
    return a_outcome == b_outcome


def _chsh_from_density(
    rho: np.ndarray,
    protocol: DirectE91ProtocolConfig,
) -> tuple[dict[str, float], float | None]:
    settings = {setting.label: setting for setting in protocol.measurement_settings}
    correlations: dict[str, float] = {}
    for a_label, b_label in protocol.chsh_setting_pairs:
        probs = joint_outcome_probabilities(rho, settings[a_label].axis, settings[b_label].axis)
        same = float(probs[0, 0] + probs[1, 1])
        different = float(probs[0, 1] + probs[1, 0])
        correlations[f"{a_label}|{b_label}"] = same - different
    if len(protocol.chsh_setting_pairs) != 4:
        return correlations, None
    labels = [f"{a}|{b}" for a, b in protocol.chsh_setting_pairs]
    raw = (
        correlations[labels[0]]
        + correlations[labels[1]]
        + correlations[labels[2]]
        - correlations[labels[3]]
    )
    if protocol.chsh_sign_convention == "singlet_negative_standard":
        return correlations, float(-raw)
    if protocol.chsh_sign_convention == "absolute_standard":
        return correlations, float(abs(raw))
    return correlations, float(raw)


def _observed_chsh_from_counts(
    outcome_counts: dict[str, dict[str, int]],
    protocol: DirectE91ProtocolConfig,
) -> tuple[dict[str, float | None], float | None]:
    correlations: dict[str, float | None] = {}
    for a_label, b_label in protocol.chsh_setting_pairs:
        pair_label = f"{a_label}|{b_label}"
        counts = outcome_counts.get(pair_label, {})
        total = sum(int(value) for value in counts.values())
        if total <= 0:
            correlations[pair_label] = None
            continue
        same = int(counts.get("00", 0)) + int(counts.get("11", 0))
        different = int(counts.get("01", 0)) + int(counts.get("10", 0))
        correlations[pair_label] = (same - different) / total
    if any(value is None for value in correlations.values()):
        return correlations, None
    labels = [f"{a}|{b}" for a, b in protocol.chsh_setting_pairs]
    raw = (
        float(correlations[labels[0]])
        + float(correlations[labels[1]])
        + float(correlations[labels[2]])
        - float(correlations[labels[3]])
    )
    if protocol.chsh_sign_convention == "singlet_negative_standard":
        return correlations, float(-raw)
    if protocol.chsh_sign_convention == "absolute_standard":
        return correlations, float(abs(raw))
    return correlations, float(raw)


def _chsh_pair_labels(protocol: DirectE91ProtocolConfig) -> set[str]:
    return {f"{a_label}|{b_label}" for a_label, b_label in protocol.chsh_setting_pairs}


def _eve_guess_probability(
    rho: np.ndarray,
    branch_model: dict[str, Any] | None,
    attack: DirectE91AttackSpec,
    target_active: bool,
    protocol: DirectE91ProtocolConfig,
    hardware: DirectE91HardwareConfig,
) -> float:
    _ = rho
    if not target_active:
        return 0.5
    if attack.attack_kind == "repeater_memory_dephasing_probe":
        retained = _clip_probability(attack.information_probability)
        return float(0.5 + retained * math.sin(float(attack.dephasing_theta)) / 2.0)
    if attack.attack_kind not in {
        "edge_intercept_resend",
        "repeater_memory_measure_Z",
        "repeater_memory_measure_X",
    }:
        return 0.5
    if branch_model is None:
        return 0.5
    key_pair = _first_key_pair(protocol)
    if key_pair is None:
        return 0.5
    alice_setting, bob_setting = key_pair
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], list[float]] = {}
    for branch in branch_model["branches"]:
        branch_probability = float(branch["probability"])
        if branch_probability <= 0.0:
            continue
        condition = _posterior_condition_record(branch)
        if condition not in grouped:
            grouped[condition] = [0.0, 0.0]
        branch_state = _apply_measurement_noise(branch["state"], hardware)
        probs = joint_outcome_probabilities(
            branch_state,
            alice_setting.axis,
            bob_setting.axis,
        )
        grouped[condition][0] += branch_probability * float(probs[0, :].sum())
        grouped[condition][1] += branch_probability * float(probs[1, :].sum())
    total = sum(sum(values) for values in grouped.values())
    if total <= 0.0:
        return 0.5
    guess = sum(max(values) for values in grouped.values()) / total
    return _clip_probability(guess)


def _posterior_condition_record(branch: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    eve_labels = branch.get("eve_observation_labels")
    if eve_labels is None:
        eve_label = str(branch.get("eve_observation_label", "none"))
        eve_labels = () if eve_label == "none" else tuple(eve_label.split("|"))
    public_labels = branch.get("public_bsm_labels")
    if public_labels is None:
        public_label = str(branch.get("public_bsm_label", "none"))
        public_labels = () if public_label == "none" else tuple(public_label.split("|"))
    return tuple(str(label) for label in eve_labels), tuple(str(label) for label in public_labels)


def _eve_guess_model(
    attack: DirectE91AttackSpec,
    target_active: bool,
    branch_model: dict[str, Any] | None,
) -> str:
    if not target_active or attack.attack_kind in {"none", "edge_added_loss", "swap_denial"}:
        return "none"
    if attack.attack_kind == "repeater_memory_dephasing_probe":
        return "probe_angle_diagnostic"
    if (
        attack.attack_kind
        in {"edge_intercept_resend", "repeater_memory_measure_Z", "repeater_memory_measure_X"}
        and branch_model is not None
    ):
        return "branch_conditioned_exact_key_posterior"
    return "none"


def _first_key_pair(
    protocol: DirectE91ProtocolConfig,
) -> tuple[DirectE91MeasurementSetting, DirectE91MeasurementSetting] | None:
    alice = next(
        (s for s in protocol.measurement_settings if s.party == "alice" and s.use == "key"),
        None,
    )
    bob = next(
        (s for s in protocol.measurement_settings if s.party == "bob" and s.use == "key"),
        None,
    )
    if alice is None or bob is None:
        return None
    return alice, bob


def _attack_can_expose_information(
    attack: DirectE91AttackSpec,
    target_active: bool,
) -> bool:
    if not target_active:
        return False
    return (
        attack.attack_kind
        in {
            "edge_intercept_resend",
            "repeater_memory_measure_Z",
            "repeater_memory_measure_X",
            "repeater_memory_dephasing_probe",
            "memory_degradation",
        }
        and attack.information_probability > 0.0
    )


def _finite_key_estimate(
    protocol: DirectE91ProtocolConfig,
    sifted_key_count: int,
    qber_key: float | None,
) -> float:
    if sifted_key_count <= 0 or qber_key is None:
        return 0.0
    hq = binary_entropy(qber_key)
    leak_ec = protocol.error_correction_efficiency * sifted_key_count * hq
    asymptotic_bits = sifted_key_count * max(0.0, 1.0 - 2.0 * hq)
    finite_penalty = math.sqrt(
        max(1, sifted_key_count) * math.log2(1.0 / protocol.finite_key_epsilon)
    )
    return float(max(0.0, asymptotic_bits - leak_ec - finite_penalty))


def _public_outcome(
    delivered_pair_count: int,
    qber_abort: bool,
    chsh_abort: bool,
) -> str:
    if delivered_pair_count <= 0:
        return "delivery_failure"
    if qber_abort:
        return "qber_abort"
    if chsh_abort:
        return "chsh_abort"
    return "accepted"


def _prepared_transition_cache_key(
    step_input: DirectE91StepInput,
    target_active: bool,
    *,
    training_light: bool,
) -> tuple[Any, ...]:
    return (
        id(step_input.topology),
        step_input.alice_node_id,
        step_input.bob_node_id,
        id(step_input.selected_route),
        id(step_input.hardware),
        id(step_input.protocol),
        id(step_input.eve_attack),
        bool(target_active),
        bool(training_light),
    )


def _prepared_provenance(
    prepared: DirectE91PreparedTransition,
    seed: int,
    step_index: int,
    *,
    training_light: bool,
) -> dict[str, Any]:
    base = (
        prepared.training_light_provenance_base
        if training_light
        else prepared.provenance_base
    )
    provenance = dict(base)
    provenance["seed"] = int(seed)
    provenance["step_index"] = int(step_index)
    return provenance


def _prepared_transcript_summary(
    prepared: DirectE91PreparedTransition,
    setting_pair_counts: dict[str, int],
    outcome_counts: dict[str, dict[str, int]],
    *,
    training_light: bool,
) -> dict[str, Any]:
    if training_light:
        return {
            "training_light": True,
            "endpoint_sampling_state": "rho_public_density_matrix",
            "eve_branch_model_scope": (
                prepared.transcript_static_summary["eve_branch_model_scope"]
            ),
        }
    summary = dict(prepared.transcript_static_summary)
    summary["public_setting_labels"] = sorted(setting_pair_counts)
    summary["outcome_counts"] = outcome_counts
    return summary


def _result_provenance_base(
    step_input: DirectE91StepInput,
    route: DirectE91Route,
    attack: DirectE91AttackSpec,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
) -> dict[str, Any]:
    topology_payload = (
        step_input.topology.to_dict()
        if hasattr(step_input.topology, "to_dict")
        else step_input.topology
    )
    hardware_payload = hardware.to_dict()
    protocol_payload = protocol.to_dict()
    config_hash = _stable_hash({
        "hardware": hardware_payload,
        "protocol": protocol_payload,
    })
    hp = hardware.provenance.to_dict()
    return {
        "backend": BACKEND,
        "backend_scope": BACKEND_SCOPE,
        "no_sequence_runtime": True,
        "model_version": MODEL_VERSION,
        "topology_hash": _stable_hash(topology_payload),
        "route_hash": _stable_hash(route.to_dict()),
        "action_hash": _stable_hash(attack.to_dict()),
        "config_hash": config_hash,
        "hardware_profile_id": hp["hardware_profile_id"],
        "hardware_profile_source_paths": hp["hardware_profile_source_paths"],
        "mapped_parameter_names": hp["mapped_parameter_names"],
        "fallback_default_names": hp["fallback_default_names"],
        "unmapped_parameter_names": hp["unmapped_parameter_names"],
        "literature_source_ids": hp["literature_source_ids"],
    }


def _result_provenance(
    step_input: DirectE91StepInput,
    route: DirectE91Route,
    attack: DirectE91AttackSpec,
    hardware: DirectE91HardwareConfig,
    protocol: DirectE91ProtocolConfig,
) -> dict[str, Any]:
    provenance = _result_provenance_base(
        step_input,
        route,
        attack,
        hardware,
        protocol,
    )
    provenance["seed"] = int(step_input.seed)
    provenance["step_index"] = int(step_input.step_index)
    return provenance
