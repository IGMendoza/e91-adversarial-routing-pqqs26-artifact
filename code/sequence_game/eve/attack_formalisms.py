"""Pure formalisms for source-scoped Eve attack surfaces.

These functions are mathematical helpers and capability gates. They do not
wire blocked attacks into the selected-path runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..claims import (
    BLOCKED,
    HARDWARE_DIAGNOSTIC,
    assert_no_private_public_fields,
)
from .attack_capabilities import (
    BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
    BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
    BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC,
    BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
)


class FormalismError(ValueError):
    """Invalid density matrix, probability, or attack formalism input."""


class CapabilityBlockedError(RuntimeError):
    """A source-motivated attack formalism lacks its runtime hook."""

    def __init__(self, capability_code: str, message: str):
        super().__init__(f"{capability_code}: {message}")
        self.capability_code = capability_code


@dataclass(frozen=True)
class MeasurementBranch:
    outcome: int
    probability: float
    state: np.ndarray


KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)
KET_PLUS = (KET_0 + KET_1) / math.sqrt(2)
KET_MINUS = (KET_0 - KET_1) / math.sqrt(2)


def density(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=complex).reshape(-1, 1)
    return vec @ vec.conj().T


def basis_ket(basis: str, outcome: int) -> np.ndarray:
    basis = basis.upper()
    if outcome not in (0, 1):
        raise FormalismError("outcome must be 0 or 1")
    if basis == "Z":
        return KET_0 if outcome == 0 else KET_1
    if basis == "X":
        return KET_PLUS if outcome == 0 else KET_MINUS
    raise FormalismError("basis must be Z or X")


def projector(basis: str, outcome: int) -> np.ndarray:
    return density(basis_ket(basis, outcome))


def single_qubit_projective_measurement(
        rho: np.ndarray,
        *,
        target_qubit: int = 0,
        basis: str = "Z",
        outcome: int) -> MeasurementBranch:
    rho = _as_density(rho)
    op = lift_single_qubit_operator(
        projector(basis, outcome), target_qubit=target_qubit,
        num_qubits=_num_qubits(rho))
    unnormalized = op @ rho @ op
    probability = float(np.real_if_close(np.trace(unnormalized)))
    if probability > 0:
        state = unnormalized / probability
    else:
        state = unnormalized
    return MeasurementBranch(outcome=outcome, probability=probability, state=state)


def projective_measurement_branches(
        rho: np.ndarray,
        *,
        target_qubit: int = 0,
        basis: str = "Z") -> tuple[MeasurementBranch, MeasurementBranch]:
    return (
        single_qubit_projective_measurement(
            rho, target_qubit=target_qubit, basis=basis, outcome=0),
        single_qubit_projective_measurement(
            rho, target_qubit=target_qubit, basis=basis, outcome=1),
    )


def averaged_projective_measurement(
        rho: np.ndarray,
        *,
        target_qubit: int = 0,
        basis: str = "Z") -> np.ndarray:
    branches = projective_measurement_branches(
        rho, target_qubit=target_qubit, basis=basis)
    return sum(branch.probability * branch.state for branch in branches)


def ancilla_output_for_basis_state(bit: int, theta: float) -> np.ndarray:
    _validate_theta(theta)
    if bit == 0:
        return KET_0.copy()
    if bit == 1:
        return math.cos(theta) * KET_0 + math.sin(theta) * KET_1
    raise FormalismError("bit must be 0 or 1")


def ancilla_coupling_reduced_state(rho_q: np.ndarray, theta: float) -> np.ndarray:
    _validate_theta(theta)
    rho_q = _as_density(rho_q, expected_dim=2)
    reduced = np.array(rho_q, dtype=complex, copy=True)
    reduced[0, 1] *= math.cos(theta)
    reduced[1, 0] *= math.cos(theta)
    _require_density_matrix(reduced)
    return reduced


def repeater_memory_dephasing_probe_channel(
        rho: np.ndarray,
        theta: float,
        *,
        target_qubit: int = 0) -> np.ndarray:
    """Reduced channel from the specified controlled-Ry Eve probe model.

    This is the modeled attack operation used by this implementation. It is not
    attributed as an exact formula from Harkness/Satoh/Naik.
    """

    _validate_theta(theta)
    rho = _as_density(rho)
    num_qubits = _num_qubits(rho)
    p0 = lift_single_qubit_operator(projector("Z", 0), target_qubit=target_qubit,
                                    num_qubits=num_qubits)
    p1 = lift_single_qubit_operator(projector("Z", 1), target_qubit=target_qubit,
                                    num_qubits=num_qubits)
    c = math.cos(theta)
    result = p0 @ rho @ p0 + p1 @ rho @ p1 + c * (p0 @ rho @ p1 + p1 @ rho @ p0)
    _require_density_matrix(result)
    return result


def dephasing_probe_qx(theta: float) -> float:
    _validate_theta(theta)
    return (1.0 - math.cos(theta)) / 2.0


def dephasing_probe_guess_probability(theta: float) -> float:
    _validate_theta(theta)
    return (1.0 + math.sin(theta)) / 2.0


def strong_z_recording_unitary() -> np.ndarray:
    return ancilla_coupling_unitary(math.pi / 2)


def ancilla_coupling_unitary(theta: float) -> np.ndarray:
    _validate_theta(theta)
    c = math.cos(theta)
    s = math.sin(theta)
    # Controlled Ry(2 theta) with basis |q,E>.
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, -s],
        [0, 0, s, c],
    ], dtype=complex)


def endpoint_intercept_resend_channel(rho_ab: np.ndarray,
                                      *,
                                      basis: str,
                                      fraction: float,
                                      target_qubit: int = 1) -> np.ndarray:
    if not 0.0 <= fraction <= 1.0:
        raise FormalismError("fraction must be in [0, 1]")
    rho_ab = _as_density(rho_ab)
    attacked = averaged_projective_measurement(
        rho_ab, target_qubit=target_qubit, basis=basis)
    result = (1.0 - fraction) * rho_ab + fraction * attacked
    _require_density_matrix(result)
    return result


def endpoint_qnd_resend_runtime_blocker() -> dict[str, Any]:
    return _blocked(
        BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
        "endpoint attack transcript with Eve private record is absent",
    )


def detector_time_shift_povm(
        ideal_effects: Sequence[np.ndarray],
        eta_profile: Optional[Mapping[str, Sequence[float]]] = None,
        *,
        timing_action: str = "nominal",
        profile_label: str = "caller-specified sensitivity") -> dict[str, Any]:
    if eta_profile is None:
        raise CapabilityBlockedError(
            BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
            "eta_j(delta_t) detector-efficiency mismatch profile is required",
        )
    if timing_action not in eta_profile:
        raise FormalismError(f"timing_action {timing_action!r} missing from eta_profile")
    effects = [np.asarray(effect, dtype=complex) for effect in ideal_effects]
    if not effects:
        raise FormalismError("ideal_effects must be non-empty")
    dim = effects[0].shape[0]
    if any(effect.shape != (dim, dim) for effect in effects):
        raise FormalismError("all ideal effects must have the same square dimension")
    etas = [float(v) for v in eta_profile[timing_action]]
    if len(etas) != len(effects):
        raise FormalismError("eta profile length must match ideal effects")
    if any(eta < 0.0 or eta > 1.0 for eta in etas):
        raise FormalismError("eta values must be in [0, 1]")
    shifted = [eta * effect for eta, effect in zip(etas, effects)]
    no_click = np.eye(dim, dtype=complex) - sum(shifted)
    if not is_psd(no_click):
        raise FormalismError("no-click operator is not PSD")
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "profile_label": profile_label,
        "timing_action": timing_action,
        "effects": shifted,
        "no_click": no_click,
    }


def korzh_timing_reference_unlocks_time_shift(profile: Mapping[str, Any]) -> bool:
    _ = profile
    return False


DETECTOR_BLINDING_REQUIRED_FIELDS = frozenset({
    "detector_family",
    "operating_mode_transition",
    "blinding_threshold_or_illumination_parameter",
    "trigger_pulse_response",
    "basis_dependent_faked_state_click_rule",
    "countermeasure_assumptions",
})


def validate_detector_blinding_model(model: Mapping[str, Any]) -> dict[str, Any]:
    missing = DETECTOR_BLINDING_REQUIRED_FIELDS - set(model)
    if missing:
        raise CapabilityBlockedError(
            BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
            f"classical detector-control model missing fields {sorted(missing)}",
        )
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "model_fields": sorted(model),
        "capability": "detector blinding formal requirements present",
    }


def information_gain_metric_blocker() -> dict[str, Any]:
    return _blocked(
        BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC,
        "raw key K, Eve private record E, and public transcript P are required",
    )


def raw_key_guessing_probability_formula() -> str:
    return "P_guess(K | E,P) = sum_{e,p} P(e,p) max_k P(k | e,p)"


def conditional_mutual_information_formula() -> str:
    return (
        "I(K;E | P) = sum_p P(p) sum_{k,e} P(k,e | p) "
        "log2(P(k,e | p)/(P(k | p) P(e | p)))"
    )


def lift_single_qubit_operator(operator: np.ndarray,
                               *,
                               target_qubit: int,
                               num_qubits: int) -> np.ndarray:
    if target_qubit < 0 or target_qubit >= num_qubits:
        raise FormalismError("target_qubit out of range")
    op = np.asarray(operator, dtype=complex)
    if op.shape != (2, 2):
        raise FormalismError("single-qubit operator must be 2x2")
    factors = [
        op if i == target_qubit else np.eye(2, dtype=complex)
        for i in range(num_qubits)
    ]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def is_hermitian(matrix: np.ndarray, *, atol: float = 1e-9) -> bool:
    matrix = np.asarray(matrix, dtype=complex)
    return bool(np.allclose(matrix, matrix.conj().T, atol=atol))


def is_psd(matrix: np.ndarray, *, atol: float = 1e-9) -> bool:
    matrix = np.asarray(matrix, dtype=complex)
    if not is_hermitian(matrix, atol=atol):
        return False
    return bool(np.min(np.linalg.eigvalsh(matrix)) >= -atol)


def _blocked(capability_code: str, reason: str) -> dict[str, Any]:
    payload = {
        "scope_label": BLOCKED,
        "blocked": True,
        "capability_code": capability_code,
        "blocked_reason": reason,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def _validate_theta(theta: float) -> None:
    if not 0.0 <= theta <= math.pi / 2:
        raise FormalismError("theta must be in [0, pi/2]")


def _num_qubits(rho: np.ndarray) -> int:
    dim = rho.shape[0]
    num = int(round(math.log2(dim)))
    if 2 ** num != dim:
        raise FormalismError("density matrix dimension must be a power of two")
    return num


def _as_density(rho: np.ndarray, *, expected_dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(rho, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise FormalismError("density matrix must be square")
    if expected_dim is not None and arr.shape != (expected_dim, expected_dim):
        raise FormalismError(f"density matrix must be {expected_dim}x{expected_dim}")
    _require_density_matrix(arr)
    return arr


def _require_density_matrix(rho: np.ndarray) -> None:
    if not is_hermitian(rho):
        raise FormalismError("density matrix must be Hermitian")
    if not np.isclose(np.trace(rho), 1.0, atol=1e-9):
        raise FormalismError("density matrix must have trace 1")
    if not is_psd(rho):
        raise FormalismError("density matrix must be PSD")


__all__ = [
    "CapabilityBlockedError",
    "DETECTOR_BLINDING_REQUIRED_FIELDS",
    "FormalismError",
    "KET_0",
    "KET_1",
    "KET_MINUS",
    "KET_PLUS",
    "MeasurementBranch",
    "ancilla_coupling_reduced_state",
    "ancilla_coupling_unitary",
    "ancilla_output_for_basis_state",
    "averaged_projective_measurement",
    "basis_ket",
    "conditional_mutual_information_formula",
    "dephasing_probe_guess_probability",
    "dephasing_probe_qx",
    "density",
    "detector_time_shift_povm",
    "endpoint_intercept_resend_channel",
    "endpoint_qnd_resend_runtime_blocker",
    "information_gain_metric_blocker",
    "is_hermitian",
    "is_psd",
    "korzh_timing_reference_unlocks_time_shift",
    "lift_single_qubit_operator",
    "projective_measurement_branches",
    "projector",
    "raw_key_guessing_probability_formula",
    "repeater_memory_dephasing_probe_channel",
    "single_qubit_projective_measurement",
    "strong_z_recording_unitary",
    "validate_detector_blinding_model",
]
