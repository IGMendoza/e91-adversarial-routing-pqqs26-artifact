"""Resource-accounted bilateral recurrence purification primitives.

The implementation is deliberately a small density-matrix circuit rather than
an empirical fidelity boost.  It implements one phase-error recurrence round:

1. express the requested Bell target in the ``phi_plus`` frame;
2. exchange phase and bit errors with bilateral Hadamards;
3. apply a local CNOT at Alice and at Bob;
4. measure the sacrificial pair in Z and retain the control pair when the two
   *reported* outcomes have even parity; and
5. undo the bilateral Hadamards and Bell-frame change.

Two-qubit and one-qubit gate errors are Pauli-depolarizing channels derived
from explicitly average-gate-fidelity parameters.  Readout errors enter the
post-selection POVM rather than being turned into state noise after the fact.
This is an explicit bilateral recurrence circuit in the Bennett/Deutsch
protocol family, specialized to phase-biased Bell-diagonal elementary links.
It does not silently substitute the distinct DEJMPS local-rotation circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
PAULIS = (I2, X, Y, Z)
PAULI_LABELS = ("I", "X", "Z", "Y")


class PurificationError(ValueError):
    """Invalid recurrence input or physical-operation model."""


def _probability(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise PurificationError(f"{name} must be a finite probability in [0, 1]")
    return result


def _nonnegative(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PurificationError(f"{name} must be finite and >= 0")
    return result


def _density_matrix(name: str, value: np.ndarray, dimension: int) -> np.ndarray:
    rho = np.asarray(value, dtype=complex)
    if rho.shape != (dimension, dimension):
        raise PurificationError(
            f"{name} must have shape {(dimension, dimension)}, got {rho.shape}"
        )
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.real_if_close(np.trace(rho)))
    if not math.isfinite(trace) or trace <= 0.0:
        raise PurificationError(f"{name} must have positive finite trace")
    rho = rho / trace
    minimum = float(np.linalg.eigvalsh(rho).min())
    if minimum < -1e-10:
        raise PurificationError(
            f"{name} must be positive semidefinite (minimum eigenvalue {minimum:.3g})"
        )
    return np.asarray(np.real_if_close(rho), dtype=complex)


def average_gate_fidelity_to_pauli_error(
    average_gate_fidelity: float,
    *,
    dimension: int,
) -> float:
    """Convert average gate fidelity to total non-identity Pauli probability.

    For a d-dimensional Pauli channel with identity weight ``1-p``,
    ``F_avg = (d*(1-p)+1)/(d+1)``.  This conversion is valid only because the
    selected operation model explicitly declares Pauli depolarization.
    """

    fidelity = _probability("average_gate_fidelity", average_gate_fidelity)
    if dimension < 2:
        raise PurificationError("dimension must be >= 2")
    error = ((dimension + 1.0) / dimension) * (1.0 - fidelity)
    if error > 1.0 + 1e-12:
        raise PurificationError(
            "average gate fidelity is below the range representable by the "
            f"selected d={dimension} Pauli channel"
        )
    return min(1.0, max(0.0, error))


@dataclass(frozen=True)
class BilateralRecurrenceModel:
    """Validated local-operation model for one recurrence round."""

    protocol: str
    rounds: int
    placement: str
    raw_pairs_per_round: int
    parallel_raw_pair_generation_modes_per_edge: int
    retained_pair_selection: str
    local_two_qubit_average_gate_fidelity: float
    local_two_qubit_gate_duration_s: float
    local_single_qubit_average_gate_fidelity: float
    local_single_qubit_layer_duration_s: float
    measurement_readout_fidelity: float
    measurement_duration_s: float
    classical_comparison_speed_m_per_s: float
    component_sources: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BilateralRecurrenceModel":
        data = dict(value)
        kind = str(data.get("kind", ""))
        if kind != "bilateral_phase_error_recurrence":
            raise PurificationError(
                "quality restoration kind must be 'bilateral_phase_error_recurrence'"
            )
        protocol = str(data.get("protocol", ""))
        if protocol != "bilateral_phase_parity_basis_exchange":
            raise PurificationError(
                "quality restoration protocol must be "
                "'bilateral_phase_parity_basis_exchange'"
            )
        rounds = int(data.get("rounds", 0))
        if rounds != 1:
            raise PurificationError(
                "this bounded implementation supports exactly one recurrence round"
            )
        placement = str(data.get("placement", ""))
        if placement != "elementary_links_before_swapping":
            raise PurificationError(
                "quality restoration placement must be 'elementary_links_before_swapping'"
            )
        raw_pairs = int(data.get("raw_pairs_per_round", 0))
        if raw_pairs != 2:
            raise PurificationError("one recurrence round consumes exactly two raw pairs")
        modes = int(data.get("parallel_raw_pair_generation_modes_per_edge", 0))
        if modes != 2:
            raise PurificationError(
                "the implemented batch-time model requires exactly two independently "
                "resourced raw-pair generation modes per edge"
            )
        retained = str(data.get("retained_pair_selection", ""))
        if retained != "youngest_raw_pair_as_control":
            raise PurificationError(
                "retained_pair_selection must be 'youngest_raw_pair_as_control'"
            )

        two_qubit = data.get("local_two_qubit_gate_model")
        one_qubit = data.get("local_single_qubit_gate_model")
        measurement = data.get("measurement_model")
        comparison = data.get("classical_comparison_model")
        for name, model in (
            ("local_two_qubit_gate_model", two_qubit),
            ("local_single_qubit_gate_model", one_qubit),
            ("measurement_model", measurement),
            ("classical_comparison_model", comparison),
        ):
            if not isinstance(model, Mapping):
                raise PurificationError(f"{name} must be an object")

        assert isinstance(two_qubit, Mapping)
        assert isinstance(one_qubit, Mapping)
        assert isinstance(measurement, Mapping)
        assert isinstance(comparison, Mapping)
        if str(two_qubit.get("kind", "")) != "pauli_depolarizing_from_average_gate_fidelity":
            raise PurificationError(
                "local two-qubit gate model must explicitly select average-fidelity "
                "Pauli depolarization"
            )
        if str(one_qubit.get("kind", "")) != "pauli_depolarizing_from_average_gate_fidelity":
            raise PurificationError(
                "local one-qubit gate model must explicitly select average-fidelity "
                "Pauli depolarization"
            )
        if str(measurement.get("kind", "")) != "independent_symmetric_bit_confusion":
            raise PurificationError(
                "measurement model must be 'independent_symmetric_bit_confusion'"
            )
        if str(comparison.get("kind", "")) != "one_way_parity_comparison":
            raise PurificationError(
                "classical comparison model must be 'one_way_parity_comparison'"
            )

        two_fidelity = _probability(
            "local_two_qubit_gate_model.average_gate_fidelity",
            two_qubit.get("average_gate_fidelity"),
        )
        one_fidelity = _probability(
            "local_single_qubit_gate_model.average_gate_fidelity",
            one_qubit.get("average_gate_fidelity"),
        )
        readout = _probability(
            "measurement_model.readout_fidelity", measurement.get("readout_fidelity")
        )
        speed = float(comparison.get("speed_m_per_s", 0.0))
        if not math.isfinite(speed) or speed <= 0.0:
            raise PurificationError(
                "classical_comparison_model.speed_m_per_s must be finite and > 0"
            )
        sources_raw = data.get("component_sources", ())
        if not isinstance(sources_raw, (list, tuple)) or not sources_raw:
            raise PurificationError("quality restoration requires component_sources")
        sources = tuple(str(item).strip() for item in sources_raw)
        if any(not item for item in sources):
            raise PurificationError("quality restoration component sources must be non-empty")
        return cls(
            protocol=protocol,
            rounds=rounds,
            placement=placement,
            raw_pairs_per_round=raw_pairs,
            parallel_raw_pair_generation_modes_per_edge=modes,
            retained_pair_selection=retained,
            local_two_qubit_average_gate_fidelity=two_fidelity,
            local_two_qubit_gate_duration_s=_nonnegative(
                "local_two_qubit_gate_model.duration_s", two_qubit.get("duration_s", 0.0)
            ),
            local_single_qubit_average_gate_fidelity=one_fidelity,
            local_single_qubit_layer_duration_s=_nonnegative(
                "local_single_qubit_gate_model.layer_duration_s",
                one_qubit.get("layer_duration_s", 0.0),
            ),
            measurement_readout_fidelity=readout,
            measurement_duration_s=_nonnegative(
                "measurement_model.duration_s", measurement.get("duration_s", 0.0)
            ),
            classical_comparison_speed_m_per_s=speed,
            component_sources=sources,
        )

    @property
    def local_operation_duration_s(self) -> float:
        # One pre-rotation layer and one post-rotation layer; Alice and Bob act
        # in parallel within each layer and during their local CNOT/readout.
        return (
            2.0 * self.local_single_qubit_layer_duration_s
            + self.local_two_qubit_gate_duration_s
            + self.measurement_duration_s
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "bilateral_phase_error_recurrence",
            "protocol": self.protocol,
            "rounds": self.rounds,
            "placement": self.placement,
            "raw_pairs_per_round": self.raw_pairs_per_round,
            "parallel_raw_pair_generation_modes_per_edge": (
                self.parallel_raw_pair_generation_modes_per_edge
            ),
            "retained_pair_selection": self.retained_pair_selection,
            "local_two_qubit_average_gate_fidelity": (
                self.local_two_qubit_average_gate_fidelity
            ),
            "local_two_qubit_pauli_error_probability": average_gate_fidelity_to_pauli_error(
                self.local_two_qubit_average_gate_fidelity, dimension=4
            ),
            "local_two_qubit_gate_duration_s": self.local_two_qubit_gate_duration_s,
            "local_single_qubit_average_gate_fidelity": (
                self.local_single_qubit_average_gate_fidelity
            ),
            "local_single_qubit_pauli_error_probability": average_gate_fidelity_to_pauli_error(
                self.local_single_qubit_average_gate_fidelity, dimension=2
            ),
            "local_single_qubit_layer_duration_s": (
                self.local_single_qubit_layer_duration_s
            ),
            "measurement_readout_fidelity": self.measurement_readout_fidelity,
            "measurement_duration_s": self.measurement_duration_s,
            "classical_comparison_speed_m_per_s": self.classical_comparison_speed_m_per_s,
            "local_operation_duration_s": self.local_operation_duration_s,
            "component_sources": list(self.component_sources),
        }


@dataclass(frozen=True)
class PurificationResult:
    output_density_matrix: np.ndarray
    success_probability: float
    bell_probabilities: tuple[float, float, float, float]
    input_pair_count: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_probability": self.success_probability,
            "bell_probabilities": {
                label: value for label, value in zip(PAULI_LABELS, self.bell_probabilities)
            },
            "input_pair_count": self.input_pair_count,
        }


def _lift_one(operator: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    if not 0 <= qubit < n_qubits:
        raise PurificationError("qubit index out of range")
    out = np.array([[1.0]], dtype=complex)
    for index in range(n_qubits):
        out = np.kron(out, operator if index == qubit else I2)
    return out


def _cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    if control == target:
        raise PurificationError("CNOT control and target must differ")
    dimension = 1 << n_qubits
    out = np.zeros((dimension, dimension), dtype=complex)
    for source in range(dimension):
        bits = [(source >> (n_qubits - 1 - index)) & 1 for index in range(n_qubits)]
        target_bits = list(bits)
        target_bits[target] ^= bits[control]
        destination = sum(
            bit << (n_qubits - 1 - index) for index, bit in enumerate(target_bits)
        )
        out[destination, source] = 1.0
    return out


def _apply_unitary(rho: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return operator @ rho @ operator.conj().T


def _apply_one_qubit_pauli_channel(
    rho: np.ndarray,
    qubit: int,
    total_error_probability: float,
) -> np.ndarray:
    p = _probability("one-qubit Pauli error probability", total_error_probability)
    out = (1.0 - p) * rho
    for pauli in PAULIS[1:]:
        lifted = _lift_one(pauli, qubit, 4)
        out = out + (p / 3.0) * _apply_unitary(rho, lifted)
    return out


def _apply_two_qubit_pauli_channel(
    rho: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    total_error_probability: float,
) -> np.ndarray:
    p = _probability("two-qubit Pauli error probability", total_error_probability)
    out = (1.0 - p) * rho
    for left_index, left in enumerate(PAULIS):
        for right_index, right in enumerate(PAULIS):
            if left_index == 0 and right_index == 0:
                continue
            lifted = _lift_one(left, qubit_a, 4) @ _lift_one(right, qubit_b, 4)
            out = out + (p / 15.0) * _apply_unitary(rho, lifted)
    return out


def _apply_two_qubit_local_dephasing(
    rho: np.ndarray,
    qubit: int,
    coherence_factor: float,
) -> np.ndarray:
    coherence = _probability("coherence_factor", coherence_factor)
    phase_flip = 0.5 * (1.0 - coherence)
    lifted = _lift_one(Z, qubit, 4)
    return (1.0 - phase_flip) * rho + phase_flip * _apply_unitary(rho, lifted)


def _bell_ket(label: str) -> np.ndarray:
    inv = 1.0 / math.sqrt(2.0)
    states = {
        "phi_plus": np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) * inv,
        "phi_minus": np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) * inv,
        "psi_plus": np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) * inv,
        "psi_minus": np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) * inv,
    }
    try:
        return states[label]
    except KeyError as exc:
        raise PurificationError(f"unsupported Bell target {label!r}") from exc


def bell_diagonal_density(
    probabilities: tuple[float, float, float, float],
    *,
    target_label: str = "psi_minus",
) -> np.ndarray:
    values = tuple(_probability(f"probabilities[{index}]", value) for index, value in enumerate(probabilities))
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise PurificationError(f"Bell probabilities must sum to one, got {total:.17g}")
    target = _bell_ket(target_label)
    out = np.zeros((4, 4), dtype=complex)
    for probability, pauli in zip(values, (I2, X, Z, Y)):
        state = np.kron(I2, pauli) @ target
        out = out + probability * np.outer(state, state.conj())
    return np.asarray(np.real_if_close(out), dtype=complex)


def bell_error_probabilities(
    rho: np.ndarray,
    *,
    target_label: str = "psi_minus",
) -> tuple[float, float, float, float]:
    state = _density_matrix("rho", rho, 4)
    target = _bell_ket(target_label)
    values = []
    for pauli in (I2, X, Z, Y):
        vector = np.kron(I2, pauli) @ target
        values.append(max(0.0, float(np.real(vector.conj() @ state @ vector))))
    total = math.fsum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def bilateral_phase_error_recurrence(
    control_pair: np.ndarray,
    sacrificial_pair: np.ndarray,
    model: BilateralRecurrenceModel,
    *,
    target_label: str = "psi_minus",
    sacrificial_wait_coherence: float = 1.0,
) -> PurificationResult:
    """Apply one noisy bilateral phase-parity recurrence round exactly.

    Qubit order is ``A_control, B_control, A_sacrificial, B_sacrificial``.
    The optional coherence factor applies the additional wait of the older
    sacrificial raw pair before local purification begins.  The retained pair
    is explicitly selected as the younger input by the service controller.
    """

    control = _density_matrix("control_pair", control_pair, 4)
    sacrificial = _density_matrix("sacrificial_pair", sacrificial_pair, 4)
    coherence = _probability("sacrificial_wait_coherence", sacrificial_wait_coherence)
    rho = np.kron(control, sacrificial)

    # Virtual local Pauli frames map psi_minus to phi_plus.  These are tracked
    # in software and therefore do not add a physical gate error or duration.
    if target_label == "psi_minus":
        for qubit in (1, 3):
            rho = _apply_unitary(rho, _lift_one(Y, qubit, 4))
    elif target_label != "phi_plus":
        raise PurificationError(
            "the implemented recurrence supports target_label psi_minus or phi_plus"
        )

    if coherence < 1.0:
        rho = _apply_two_qubit_local_dephasing(rho, 2, coherence)
        rho = _apply_two_qubit_local_dephasing(rho, 3, coherence)

    one_qubit_error = average_gate_fidelity_to_pauli_error(
        model.local_single_qubit_average_gate_fidelity, dimension=2
    )
    two_qubit_error = average_gate_fidelity_to_pauli_error(
        model.local_two_qubit_average_gate_fidelity, dimension=4
    )

    # Bilateral basis exchange turns phase parity into the Z-measurement bit
    # parity selected by the recurrence protocol.
    for qubit in range(4):
        rho = _apply_unitary(rho, _lift_one(H, qubit, 4))
        rho = _apply_one_qubit_pauli_channel(rho, qubit, one_qubit_error)

    bilateral_cnot = _cnot(0, 2, 4) @ _cnot(1, 3, 4)
    rho = _apply_unitary(rho, bilateral_cnot)
    rho = _apply_two_qubit_pauli_channel(rho, 0, 2, two_qubit_error)
    rho = _apply_two_qubit_pauli_channel(rho, 1, 3, two_qubit_error)

    readout_error = 1.0 - model.measurement_readout_fidelity
    retained = np.zeros((4, 4), dtype=complex)
    for alice_actual in (0, 1):
        for bob_actual in (0, 1):
            actual_even = alice_actual == bob_actual
            if actual_even:
                accept_probability = (1.0 - readout_error) ** 2 + readout_error ** 2
            else:
                accept_probability = 2.0 * readout_error * (1.0 - readout_error)
            indices = [
                (alice_control << 3)
                | (bob_control << 2)
                | (alice_actual << 1)
                | bob_actual
                for alice_control in (0, 1)
                for bob_control in (0, 1)
            ]
            retained = retained + accept_probability * rho[np.ix_(indices, indices)]

    success = max(0.0, float(np.real_if_close(np.trace(retained))))
    if success <= 0.0:
        raise PurificationError("purification acceptance probability is zero")
    retained = retained / success

    for qubit in (0, 1):
        retained_operator = np.kron(H if qubit == 0 else I2, H if qubit == 1 else I2)
        retained = _apply_unitary(retained, retained_operator)
        p = one_qubit_error
        noisy = (1.0 - p) * retained
        for pauli in PAULIS[1:]:
            local = np.kron(pauli if qubit == 0 else I2, pauli if qubit == 1 else I2)
            noisy = noisy + (p / 3.0) * _apply_unitary(retained, local)
        retained = noisy

    if target_label == "psi_minus":
        retained = _apply_unitary(retained, np.kron(I2, Y))
    retained = _density_matrix("purified output", retained, 4)
    return PurificationResult(
        output_density_matrix=retained,
        success_probability=success,
        bell_probabilities=bell_error_probabilities(retained, target_label=target_label),
    )
