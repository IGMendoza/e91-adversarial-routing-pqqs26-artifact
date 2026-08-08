"""Density-matrix primitives, channels, measurements, and swap algebra."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import numpy as np

from .devices import *
from .models import *


def join_labels(left: str, right: str) -> str:
    parts = [label for label in (left, right) if label and label != "none"]
    return "|".join(parts) if parts else "none"

def density(ket: np.ndarray) -> np.ndarray:
    ket = np.asarray(ket, dtype=complex).reshape(-1, 1)
    return ket @ ket.conj().T


def bell_ket(label: str) -> np.ndarray:
    inv = 1.0 / math.sqrt(2.0)
    states = {
        "phi_plus": np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) * inv,
        "phi_minus": np.array([1.0, 0.0, 0.0, -1.0], dtype=complex) * inv,
        "psi_plus": np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) * inv,
        "psi_minus": np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) * inv,
    }
    return states[label]


def bell_density(label: str) -> np.ndarray:
    return density(bell_ket(label))


def clean_density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    rho = (rho + rho.conj().T) / 2.0
    trace = complex(np.trace(rho))
    if abs(trace) <= 0.0:
        raise ValueError("density matrix trace must be nonzero")
    return np.asarray(np.real_if_close(rho / trace), dtype=complex)


def sequence_bell_state_with_fidelity(target_label: str, fidelity: float) -> np.ndarray:
    """Density-matrix form of ``sequence.components.bsm._set_state_with_fidelity``.

    SeQUeNCe uses this exact Bell-diagonal assignment in density-matrix
    formalism: the desired Bell state has weight ``fidelity`` and the other
    three Bell states each have weight ``(1-fidelity)/3``.
    """
    target = bell_density(target_label)
    labels = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
    rho = fidelity * target
    for label in labels:
        if label != target_label:
            rho += ((1.0 - fidelity) / 3.0) * bell_density(label)
    return clean_density(rho)


def elementary_memory_pair_density(
    front: dict[str, Any],
    *,
    fidelity_multiplier: float = 1.0,
) -> np.ndarray:
    fidelity = clip_probability(
        front["device_constants"]["memory"]["fidelity"] * fidelity_multiplier
    )
    return sequence_bell_state_with_fidelity("phi_plus", fidelity)


def apply_one_qubit_channel(rho: np.ndarray, qubit: int, ops: Iterable[np.ndarray]) -> np.ndarray:
    n_qubits = int(round(math.log2(rho.shape[0])))
    out = np.zeros_like(rho, dtype=complex)
    for op in ops:
        lifted = lift_operator(op, qubit, n_qubits)
        out += lifted @ rho @ lifted.conj().T
    return clean_density(out)


def depolarizing_kraus(probability: float) -> tuple[np.ndarray, ...]:
    p = clip_probability(probability)
    return (
        math.sqrt(1.0 - p) * I2,
        math.sqrt(p / 3.0) * X,
        math.sqrt(p / 3.0) * Y,
        math.sqrt(p / 3.0) * Z,
    )


def dephasing_kraus(probability: float, basis: str = "Z") -> tuple[np.ndarray, ...]:
    p = clip_probability(probability)
    op = Z if basis == "Z" else X
    return (math.sqrt(1.0 - p) * I2, math.sqrt(p) * op)


def amplitude_damping_kraus(probability: float) -> tuple[np.ndarray, ...]:
    gamma = clip_probability(probability)
    return (
        np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex),
        np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    )


def two_qubit_depolarizing_kraus(probability: float) -> tuple[np.ndarray, ...]:
    p = clip_probability(probability)
    identity_4 = np.eye(4, dtype=complex)
    ops: list[np.ndarray] = [math.sqrt(1.0 - p) * identity_4]
    paulis = (I2, X, Y, Z)
    non_identity = [
        np.kron(left, right)
        for left in paulis
        for right in paulis
        if not (np.array_equal(left, I2) and np.array_equal(right, I2))
    ]
    ops.extend(math.sqrt(p / len(non_identity)) * op for op in non_identity)
    return tuple(ops)


def apply_two_qubit_channel(rho: np.ndarray, ops: Iterable[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for op in ops:
        out += op @ rho @ op.conj().T
    return clean_density(out)


def apply_branch_channel(
    branches: list[Branch],
    *,
    qubit: int,
    ops: Iterable[np.ndarray],
) -> list[Branch]:
    return normalize_branches([
        Branch(
            probability=branch.probability,
            rho=apply_one_qubit_channel(branch.rho, qubit, ops),
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches
    ])


def lift_operator(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    lifted = np.array([[1.0]], dtype=complex)
    for idx in range(n_qubits):
        lifted = np.kron(lifted, op if idx == qubit else I2)
    return lifted


def projectors_for_axis(axis: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("measurement axis must be nonzero")
    observable = (x / norm) * X + (y / norm) * Y + (z / norm) * Z
    return (I2 + observable) / 2.0, (I2 - observable) / 2.0


def basis_axis(basis: str) -> tuple[float, float, float]:
    if basis == "Z":
        return (0.0, 0.0, 1.0)
    if basis == "X":
        return (1.0, 0.0, 0.0)
    raise ValueError("basis must be Z or X")


def projective_expand(
    branches: list[Branch],
    *,
    qubit: int,
    basis: str,
    label_prefix: str,
    probability: float = 1.0,
    information_probability: float = 1.0,
    retain_observation: bool = True,
    normalize: bool = True,
) -> list[Branch]:
    probability = clip_probability(probability)
    information_probability = clip_probability(information_probability if retain_observation else 0.0)
    projectors = projectors_for_axis(basis_axis(basis))
    expanded: list[Branch] = []
    if probability < 1.0:
        expanded.extend(
            Branch(
                probability=branch.probability * (1.0 - probability),
                rho=branch.rho,
                frame=branch.frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=branch.public_bsm_record,
                detector_click_record=branch.detector_click_record,
            )
            for branch in branches
        )
    for branch in branches:
        n_qubits = int(round(math.log2(branch.rho.shape[0])))
        for outcome, projector in enumerate(projectors):
            op = lift_operator(projector, qubit, n_qubits)
            unnormalized = op @ branch.rho @ op.conj().T
            outcome_probability = max(0.0, float(np.real_if_close(np.trace(unnormalized))))
            if outcome_probability <= 0.0 or probability <= 0.0:
                continue
            measured_probability = branch.probability * probability * outcome_probability
            measured_state = clean_density(unnormalized / outcome_probability)
            if information_probability > 0.0:
                label = f"{label_prefix}:basis={basis}:outcome={outcome}"
                if branch.eve_observation != "none":
                    label = branch.eve_observation + "|" + label
                expanded.append(Branch(
                    probability=measured_probability * information_probability,
                    rho=measured_state,
                    frame=branch.frame,
                    eve_observation=label,
                    public_bsm_record=branch.public_bsm_record,
                    detector_click_record=branch.detector_click_record,
                ))
            if information_probability < 1.0:
                expanded.append(Branch(
                    probability=measured_probability * (1.0 - information_probability),
                    rho=measured_state,
                    frame=branch.frame,
                    eve_observation=branch.eve_observation,
                    public_bsm_record=branch.public_bsm_record,
                    detector_click_record=branch.detector_click_record,
                ))
    return normalize_branches(expanded) if normalize else expanded


def projective_expand_joint(
    branches: list[Branch],
    *,
    qubits: tuple[int, ...],
    basis: str,
    label_prefixes: tuple[str, ...],
    probability: float,
    information_probability: float,
) -> list[Branch]:
    probability = clip_probability(probability)
    expanded: list[Branch] = []
    if probability < 1.0:
        expanded.extend(
            Branch(
                probability=branch.probability * (1.0 - probability),
                rho=branch.rho,
                frame=branch.frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=branch.public_bsm_record,
                detector_click_record=branch.detector_click_record,
            )
            for branch in branches
        )
    measured = [
        Branch(
            probability=branch.probability * probability,
            rho=branch.rho,
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches
        if probability > 0.0
    ]
    for qubit, label_prefix in zip(qubits, label_prefixes):
        measured = projective_expand(
            measured,
            qubit=qubit,
            basis=basis,
            label_prefix=label_prefix,
            probability=1.0,
            information_probability=information_probability,
            retain_observation=True,
            normalize=False,
        )
    expanded.extend(measured)
    return normalize_branches(expanded)


def normalize_branches(branches: list[Branch]) -> list[Branch]:
    total = sum(max(0.0, branch.probability) for branch in branches)
    if total <= 0.0:
        return []
    return [
        Branch(
            probability=max(0.0, branch.probability) / total,
            rho=clean_density(branch.rho),
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches if branch.probability > 0.0
    ]


def detector_click_record_for_policy(record: str, *, keep_detector_click_records: bool) -> str:
    return record if keep_detector_click_records else DETECTOR_CLICK_RECORD_NOT_EMITTED


def join_detector_click_records(
    left: str,
    right: str,
    *,
    keep_detector_click_records: bool,
) -> str:
    if not keep_detector_click_records:
        return DETECTOR_CLICK_RECORD_NOT_EMITTED
    return join_labels(left, right)


def aggregate_branches(
    branches: list[Branch],
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool = True,
) -> list[Branch]:
    grouped: dict[tuple[str, str, str, str], tuple[float, np.ndarray]] = {}
    for branch in branches:
        detector_record = detector_click_record_for_policy(
            branch.detector_click_record,
            keep_detector_click_records=keep_detector_click_records,
        )
        key = (
            branch.frame,
            branch.eve_observation,
            branch.public_bsm_record if keep_full_bsm else "frame_compressed",
            detector_record,
        )
        probability, rho_sum = grouped.get(
            key, (0.0, np.zeros_like(branch.rho, dtype=complex))
        )
        grouped[key] = (
            probability + branch.probability,
            rho_sum + branch.probability * branch.rho,
        )
    out = []
    for (frame, eve, bsm, detector_record), (probability, rho_sum) in grouped.items():
        if probability <= 0.0:
            continue
        out.append(Branch(probability, clean_density(rho_sum / probability), frame, eve, bsm, detector_record))
    return normalize_branches(out)


def partial_trace(rho: np.ndarray, keep: tuple[int, ...], n_qubits: int) -> np.ndarray:
    keep = tuple(sorted(keep))
    row = list("abcdefghijklmnopqrstuvwxyz"[:n_qubits])
    col = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_qubits])
    for idx in range(n_qubits):
        if idx not in keep:
            col[idx] = row[idx]
    out_labels = [row[idx] for idx in keep] + [col[idx] for idx in keep]
    tensor = rho.reshape((2,) * (2 * n_qubits))
    expr = "".join(row + col) + "->" + "".join(out_labels)
    result = np.einsum(expr, tensor)
    dim = 2 ** len(keep)
    return result.reshape(dim, dim)


_BSM_PROJECTOR_CACHE: dict[str, np.ndarray] = {}


def bsm_projector(outcome: str) -> np.ndarray:
    cached = _BSM_PROJECTOR_CACHE.get(outcome)
    if cached is None:
        cached = np.kron(np.kron(I2, bell_density(outcome)), I2)
        _BSM_PROJECTOR_CACHE[outcome] = cached
    return cached


def bsm_project(branch_rho: np.ndarray, outcome: str) -> tuple[np.ndarray, float]:
    projector = bsm_projector(outcome)
    projected = projector @ branch_rho @ projector.conj().T
    reduced = partial_trace(projected, (0, 3), 4)
    probability = max(0.0, float(np.real_if_close(np.trace(reduced))))
    if probability <= 0.0:
        fallback = partial_trace(branch_rho, (0, 3), 4)
        return clean_density(fallback), 0.0
    return clean_density(reduced / probability), probability


def correction_for_bsm(outcome: str) -> str:
    target = bell_density("psi_minus")
    raw, probability = bsm_project(np.kron(target, target), outcome)
    if probability <= 0.0:
        return "I"
    best = "I"
    best_overlap = -1.0
    for label in FRAME_LABELS:
        unitary = np.kron(I2, PAULI_BY_LABEL[label])
        corrected = unitary @ raw @ unitary.conj().T
        overlap = float(np.real_if_close(np.trace(target @ corrected)))
        if overlap > best_overlap:
            best = label
            best_overlap = overlap
    return best


CORRECTION_BY_BSM = {
    label: correction_for_bsm(label)
    for label in ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
}


def compose_frame(left: str, right: str) -> str:
    lx, lz = FRAME_BITS[left]
    rx, rz = FRAME_BITS[right]
    return BITS_FRAME[((lx ^ rx), (lz ^ rz))]


def apply_frame_correction(rho: np.ndarray, frame: str) -> np.ndarray:
    unitary = np.kron(I2, PAULI_BY_LABEL[frame])
    return clean_density(unitary @ rho @ unitary.conj().T)


def apply_swap_degradation(
    rho: np.ndarray,
    front: dict[str, Any],
    context: ModelContext | None = None,
) -> np.ndarray:
    service_channel = (
        repeater_swap_state_channel(context)
        if context is not None else None
    )
    if service_channel is not None:
        return apply_one_qubit_channel(
            rho,
            int(service_channel["qubit"]),
            depolarizing_kraus(float(service_channel["depolarizing_probability"])),
        )
    degradation = float(front["device_constants"]["swap_bsm"].get("swapping_degradation", 1.0))
    depolarizing_probability = clip_probability(1.0 - degradation)
    if depolarizing_probability > 0.0:
        return apply_two_qubit_channel(rho, two_qubit_depolarizing_kraus(depolarizing_probability))
    return clean_density(rho)


def apply_swap_phase_error(rho: np.ndarray, front: dict[str, Any]) -> np.ndarray:
    phase_error = float(front["device_constants"]["swap_bsm"].get("phase_error", 0.0))
    if phase_error > 0.0:
        return apply_one_qubit_channel(rho, 1, dephasing_kraus(phase_error, "Z"))
    return clean_density(rho)


def swap_branch_bell_outcomes(
    branch: Branch,
    repeater: str,
    front: dict[str, Any],
    context: ModelContext | None = None,
) -> list[Branch]:
    """Repeater swap Bell measurement with optional outcome-label confusion.

    The middle memory pair is projected onto the true Bell outcome; the
    repeater then announces a (possibly wrong) label drawn from the typed
    bell_outcome_confusion_model. The announced label drives the public
    record, the composed Pauli frame, and the applied feed-forward
    correction, so label confusion misframes the true projected state.
    """
    confusion = swap_bell_outcome_confusion_model(front)
    out: list[Branch] = []
    for outcome in BELL_LABELS:
        reduced, probability = bsm_project(branch.rho, outcome)
        if probability <= 0.0:
            continue
        for reported, report_probability in bell_outcome_reported_distribution(confusion, outcome):
            correction = CORRECTION_BY_BSM[reported]
            frame = compose_frame(branch.frame, correction)
            corrected = apply_frame_correction(reduced, correction)
            corrected = apply_swap_degradation(corrected, front, context)
            corrected = apply_swap_phase_error(corrected, front)
            full_record = join_labels(branch.public_bsm_record, f"{repeater}:{reported}")
            out.append(Branch(
                probability=branch.probability * probability * report_probability,
                rho=corrected,
                frame=frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=full_record,
                detector_click_record=branch.detector_click_record,
            ))
    return out
