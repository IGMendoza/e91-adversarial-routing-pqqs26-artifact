"""Exact Bell-state diagnostics for event-delivered endpoint memories.

The event backend normally obtains CHSH and QBER from destructive, finite-shot
endpoint measurements.  For backend validation that is needlessly expensive:
hundreds of delivered pairs are required before every Ekert setting cell is
adequately sampled.  This module instead evaluates the same fixed Ekert
observables on the two-memory density matrix immediately before either endpoint
measures it.

The resulting value is a state-expectation diagnostic.  It answers what the
event-delivered state predicts for an infinite ensemble measured at the declared
settings.  It is not an executed Bell-test transcript, a loophole-free Bell
test, or a finite-key security proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import numpy as np

from .chsh_core import (
    ALICE_ANGLES,
    BOB_ANGLES,
    CHSH_PAIRS,
    CLASSICAL_BOUND,
    KEY_PAIRS,
    bloch_angle,
)


_I = np.eye(2, dtype=complex)
_X = np.asarray([[0, 1], [1, 0]], dtype=complex)
_Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.asarray([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_X, _Y, _Z)


def _reduced_density_matrix(state: Any, target_keys: list[int]) -> np.ndarray:
    """Return a reduced density matrix in the requested target-key order."""

    keys = list(state.keys)
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("target state keys must be distinct")
    if any(key not in keys for key in target_keys):
        raise ValueError(
            f"target keys {target_keys} not all present in state keys {keys}"
        )
    if int(getattr(state, "truncation", 1)) != 1:
        raise ValueError("endpoint Bell witness requires dense qubit states")

    payload = np.asarray(state.state, dtype=complex)
    if payload.ndim == 1:
        rho = np.outer(payload, payload.conj())
    elif payload.ndim == 2 and payload.shape[0] == payload.shape[1]:
        rho = payload
    else:
        raise ValueError("state payload must be a ket vector or square density matrix")

    qubits = len(keys)
    expected_dimension = 2 ** qubits
    if rho.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f"state shape {rho.shape} is incompatible with {qubits} qubit keys"
        )

    target = [keys.index(key) for key in target_keys]
    environment = [index for index in range(qubits) if index not in target]
    permutation = target + environment + [index + qubits for index in target + environment]
    permuted = rho.reshape((2,) * (2 * qubits)).transpose(permutation)
    environment_dimension = 2 ** len(environment)
    target_dimension = 2 ** len(target)
    permuted = permuted.reshape(
        target_dimension, environment_dimension,
        target_dimension, environment_dimension,
    )
    reduced = np.einsum("aebe->ab", permuted)
    # Remove harmless numerical asymmetry without projecting an invalid state
    # onto the positive cone (which would hide upstream physics errors).
    return 0.5 * (reduced + reduced.conj().T)


def reduced_two_qubit_density_matrix(
        state: Any,
        alice_key: int,
        bob_key: int,
) -> np.ndarray:
    """Return ``rho_AB`` in Alice-then-Bob order from a dense qubit state.

    ``state`` is a SeQUeNCe ``KetState`` or ``DensityState``-compatible object
    with ``state``, ``keys``, and (optionally) ``truncation`` attributes. Extra
    qubits are traced out rather than silently assumed absent.
    """

    return _reduced_density_matrix(state, [alice_key, bob_key])


def _measurement_observable(polarization_angle: float) -> np.ndarray:
    phi = bloch_angle(polarization_angle)
    return np.cos(phi) * _Z + np.sin(phi) * _X


def _expectation(rho: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    value = np.trace(rho @ np.kron(left, right))
    if abs(float(np.imag(value))) > 1e-9:
        raise ValueError(f"observable expectation has non-negligible imaginary part {value}")
    return float(np.real(value))


@dataclass(frozen=True)
class BellStateWitness:
    """Fixed-setting and optimized Bell diagnostics of one two-qubit state."""

    fixed_setting_chsh_s: float
    fixed_setting_violates_bell: bool
    key_qber: float
    correlations: dict[str, float]
    horodecki_max_chsh_s: float
    phi_plus_fidelity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_setting_chsh_s": self.fixed_setting_chsh_s,
            "fixed_setting_violates_bell": self.fixed_setting_violates_bell,
            "key_qber": self.key_qber,
            "correlations": dict(self.correlations),
            "horodecki_max_chsh_s": self.horodecki_max_chsh_s,
            "phi_plus_fidelity": self.phi_plus_fidelity,
        }


def bell_state_witness(rho: np.ndarray) -> BellStateWitness:
    """Evaluate the repository's Ekert settings and Horodecki CHSH maximum."""

    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"two-qubit density matrix must have shape (4, 4), got {rho.shape}")
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=1e-8):
        raise ValueError(f"two-qubit density matrix must have trace 1, got {trace}")

    all_pairs = set(CHSH_PAIRS) | set(KEY_PAIRS)
    pair_correlations = {
        pair: _expectation(
            rho,
            _measurement_observable(ALICE_ANGLES[pair[0]]),
            _measurement_observable(BOB_ANGLES[pair[1]]),
        )
        for pair in all_pairs
    }
    chsh_s = (
        pair_correlations[(0, 0)]
        - pair_correlations[(0, 2)]
        + pair_correlations[(2, 0)]
        + pair_correlations[(2, 2)]
    )
    key_qber = float(np.mean([
        (1.0 - pair_correlations[pair]) / 2.0 for pair in KEY_PAIRS
    ]))
    # Clip only floating-point spill outside the probability interval.
    key_qber = min(1.0, max(0.0, key_qber))

    correlation_tensor = np.asarray([
        [_expectation(rho, left, right) for right in _PAULIS]
        for left in _PAULIS
    ])
    eigenvalues = np.linalg.eigvalsh(correlation_tensor.T @ correlation_tensor)
    horodecki_max = 2.0 * sqrt(max(0.0, float(eigenvalues[-1] + eigenvalues[-2])))
    phi_plus = np.asarray([1, 0, 0, 1], dtype=complex) / sqrt(2.0)
    fidelity = float(np.real(phi_plus.conj() @ rho @ phi_plus))

    return BellStateWitness(
        fixed_setting_chsh_s=float(chsh_s),
        fixed_setting_violates_bell=abs(chsh_s) > CLASSICAL_BOUND,
        key_qber=key_qber,
        correlations={
            f"E({ALICE_ANGLES[ai]:.3f},{BOB_ANGLES[bi]:.3f})": pair_correlations[(ai, bi)]
            for ai, bi in CHSH_PAIRS
        },
        horodecki_max_chsh_s=float(horodecki_max),
        phi_plus_fidelity=fidelity,
    )


class EndpointBellStateWitnessCollector:
    """Capture each delivered Alice--Bob state once, before measurement."""

    def __init__(
            self,
            *,
            quantum_manager: Any,
            alice_node: str,
            bob_node: str,
            qstate_key_by_memory_name: Mapping[str, int],
    ) -> None:
        self.quantum_manager = quantum_manager
        self.alice_node = str(alice_node)
        self.bob_node = str(bob_node)
        self.qstate_key_by_memory_name = dict(qstate_key_by_memory_name)
        self._captured_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
        self._density_matrices: list[np.ndarray] = []
        self._per_pair_witnesses: list[BellStateWitness] = []

    def observe(self, app: Any, info: Any) -> None:
        node = str(app.node.name)
        remote_node = str(info.remote_node)
        if {node, remote_node} != {self.alice_node, self.bob_node}:
            return
        identity = tuple(sorted((
            (node, str(info.memory.name)),
            (remote_node, str(info.remote_memo)),
        )))
        if identity in self._captured_pairs:
            return

        local_key = int(info.memory.qstate_key)
        try:
            remote_key = int(self.qstate_key_by_memory_name[str(info.remote_memo)])
        except KeyError as exc:
            raise ValueError(
                f"no endpoint qstate key registered for remote memory {info.remote_memo!r}"
            ) from exc
        if node == self.alice_node:
            alice_key, bob_key = local_key, remote_key
        else:
            alice_key, bob_key = remote_key, local_key
        state = self.quantum_manager.get(local_key)
        if alice_key in state.keys and bob_key in state.keys:
            rho = reduced_two_qubit_density_matrix(state, alice_key, bob_key)
        else:
            # A selective pre-swap measurement legitimately separates the two
            # endpoint ket objects. Their conditioned joint state is the tensor
            # product of the two one-qubit reductions and must remain visible as
            # Bell-nonviolating rather than making the diagnostic unavailable.
            alice_state = self.quantum_manager.get(alice_key)
            bob_state = self.quantum_manager.get(bob_key)
            rho = np.kron(
                _reduced_density_matrix(alice_state, [alice_key]),
                _reduced_density_matrix(bob_state, [bob_key]),
            )
        self._captured_pairs.add(identity)
        self._density_matrices.append(rho)
        self._per_pair_witnesses.append(bell_state_witness(rho))

    def summary(self, *, qber_threshold: float) -> dict[str, Any]:
        if not self._density_matrices:
            return {
                "pair_count": 0,
                "fixed_setting_chsh_s": None,
                "fixed_setting_violates_bell": None,
                "key_qber": None,
                "bell_qualified": False,
                "horodecki_max_chsh_s": None,
                "mean_per_pair_horodecki_max_chsh_s": None,
                "phi_plus_fidelity": None,
                "semantics": "pre_measurement_state_expectation_not_finite_transcript",
            }
        ensemble = bell_state_witness(np.mean(self._density_matrices, axis=0))
        out = ensemble.to_dict()
        out.update({
            "pair_count": len(self._density_matrices),
            "bell_qualified": bool(
                ensemble.fixed_setting_violates_bell
                and ensemble.key_qber <= float(qber_threshold)
            ),
            "qber_threshold": float(qber_threshold),
            "mean_per_pair_horodecki_max_chsh_s": float(np.mean([
                witness.horodecki_max_chsh_s
                for witness in self._per_pair_witnesses
            ])),
            "semantics": "pre_measurement_state_expectation_not_finite_transcript",
            "claim_boundary": (
                "exact expectation for the event-delivered state at the declared Ekert "
                "settings; not an executed finite-shot Bell test or security proof"
            ),
        })
        return out


class EndpointPairMeasurementCoordinator:
    """Wait until both endpoint swap messages arrive before either measurement.

    SeQUeNCe's swapping protocol sends the Bell-measurement feed-forward bits to
    only one end of a swap.  The other endpoint may otherwise measure before the
    remote Pauli correction is applied.  Those operations commute for the raw
    transcript, but a state observer would see an uncorrected random Bell frame.
    Pair coordination exposes the operational, corrected endpoint state and
    then performs the same two local measurements.
    """

    def __init__(self) -> None:
        self._pending: dict[
            tuple[tuple[str, str], tuple[str, str]], tuple[Any, Any]
        ] = {}

    @staticmethod
    def _identity(app: Any, info: Any) -> tuple[tuple[str, str], tuple[str, str]]:
        return tuple(sorted((
            (str(app.node.name), str(info.memory.name)),
            (str(info.remote_node), str(info.remote_memo)),
        )))

    def coordinate(self, app: Any, info: Any) -> bool:
        """Return True to defer the first side; release it on the second side."""

        identity = self._identity(app, info)
        pending = self._pending.pop(identity, None)
        if pending is None:
            self._pending[identity] = (app, info)
            return True
        pending_app, pending_info = pending
        pending_app.measure_qualified_memory(pending_info)
        # False tells the currently executing app.get_memory to measure now.
        return False
