"""Endpoint application for repeater-delivered E91/BBM92 memory measurement.

First-party port of the former submodule patch ``sequence/app/e91_endpoint.py``
(see ``sequence-submodule-2-commits.patch``). The submodule now tracks upstream
SeQUeNCe v1.0.0 unmodified, so this app lives on our side of the boundary. It
extends :class:`sequence.app.request_app.RequestApp` for the application phase
after the repeater stack has delivered an end-to-end entangled memory. It
measures local communication memories in Z or X, records the private
measurement transcript, and releases the memory back to the resource manager.

Security scope: this is raw-key/QBER measurement only. It does not implement
finite-key security, privacy amplification, error correction, or a CHSH monitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sequence.app.request_app import RequestApp
from sequence.components.circuit import Circuit
from sequence.resource_management.memory_manager import MemoryInfo

if TYPE_CHECKING:
    from sequence.network_management.reservation import Reservation
    from sequence.topology.node import QuantumRouter


_Z_MEASURE = Circuit(1)
_Z_MEASURE.measure(0)

_X_MEASURE = Circuit(1)
_X_MEASURE.h(0)
_X_MEASURE.measure(0)


def _angle_measure_circuit(bloch_angle_rad: float) -> Circuit:
    """Circuit that measures a memory qubit along Bloch angle phi in the X-Z plane.

    Realises Ry(-phi) then a Z measurement. SeQUeNCe's Circuit has no parametric
    Ry, so Ry(-phi) is synthesised (up to global phase, irrelevant to outcomes)
    as ``S . H . phase(-phi) . H . Sdg`` -- validated to match Ry(-phi) exactly
    for the Ekert Bloch angles {0, pi/4, pi/2, 3pi/4}.
    """

    circuit = Circuit(1)
    if bloch_angle_rad != 0.0:
        circuit.sdg(0)
        circuit.h(0)
        circuit.phase(0, -bloch_angle_rad)
        circuit.h(0)
        circuit.s(0)
    circuit.measure(0)
    return circuit


@dataclass(frozen=True)
class E91MemoryMeasurement:
    """One endpoint measurement of a repeater-delivered entangled memory."""

    node: str
    local_memo: str
    remote_node: str
    remote_memo: str
    basis: str
    outcome: int
    fidelity: float
    time_ps: int

    def pair_key(self) -> tuple[str, str, str, str]:
        """Directed key identifying this local half and its remote half."""

        return (self.node, self.local_memo, self.remote_node, self.remote_memo)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "local_memo": self.local_memo,
            "remote_node": self.remote_node,
            "remote_memo": self.remote_memo,
            "basis": self.basis,
            "outcome": self.outcome,
            "fidelity": self.fidelity,
            "time_ps": self.time_ps,
        }


class E91EndpointApp(RequestApp):
    """Measure delivered endpoint memories in Z/X for raw key + QBER.

    ``RequestApp.start`` still controls the reservation and delivery threshold.
    This class treats that threshold only as a simulator gate: once a qualified
    memory is delivered, the actual delivered fidelity is recorded separately for
    downstream diagnostics.
    """

    def __init__(self, node: "QuantumRouter", *, seed: int,
                 basis_labels: tuple[str, str] = ("Z", "X"),
                 release_after_measure: bool = True):
        super().__init__(node)
        if len(basis_labels) != 2 or set(basis_labels) != {"Z", "X"}:
            raise ValueError("E91EndpointApp currently supports exactly Z/X bases")
        self.name = f"{self.node.name}.E91EndpointApp"
        self.basis_labels = tuple(basis_labels)
        self.release_after_measure = release_after_measure
        self.rng = np.random.default_rng(seed)
        self.measurements: list[E91MemoryMeasurement] = []
        # Optional coordinator invoked once this local memory passes the
        # reservation/delivery checks. Returning True defers measurement; the
        # coordinator may later call measure_qualified_memory after the remote
        # endpoint has also processed its swap feed-forward message.
        self.on_qualified_memory = None
        # Optional observer invoked after delivery qualification but before the
        # destructive endpoint measurement.  Validation code uses this to
        # inspect the actual delivered two-memory state without changing it.
        self.on_pre_measurement = None
        # Optional observer called with this app after each recorded
        # measurement; the trial runner uses it to stop the timeline once the
        # pair target is reached on both endpoints.
        self.on_measurement = None

    def get_memory(self, info: MemoryInfo) -> None:
        """Measure a qualified end-to-end entangled memory, then release it."""

        if info.state != MemoryInfo.ENTANGLED:
            return
        if info.index not in self.memo_to_reservation:
            return

        reservation: Reservation = self.memo_to_reservation[info.index]
        if info.remote_node not in (reservation.initiator, reservation.responder):
            return
        if info.fidelity < reservation.fidelity:
            return

        if (self.on_qualified_memory is not None
                and self.on_qualified_memory(self, info)):
            return
        self.measure_qualified_memory(info)

    def measure_qualified_memory(self, info: MemoryInfo) -> None:
        """Measure an already-qualified memory (used by pair coordination)."""

        self._notify_pre_measurement(info)
        basis = self.basis_labels[int(self.rng.integers(0, 2))]
        circuit = _X_MEASURE if basis == "X" else _Z_MEASURE
        key = info.memory.qstate_key
        # Clamp away exact 0.0 to keep the same RNG stream and outcomes as the
        # pre-v1.0.0 patch, which had to avoid a truthiness assert upstream.
        meas_samp = max(float(self.rng.random()), 1e-15)
        result = self.node.timeline.quantum_manager.run_circuit(
            circuit, [key], meas_samp)
        outcome = int(result[key])

        self.measurements.append(E91MemoryMeasurement(
            node=self.node.name,
            local_memo=info.memory.name,
            remote_node=info.remote_node,
            remote_memo=info.remote_memo,
            basis=basis,
            outcome=outcome,
            fidelity=float(info.fidelity),
            time_ps=int(self.node.timeline.now()),
        ))
        self.memory_counter += 1
        if self.release_after_measure:
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
        self._notify_measurement()

    def _notify_measurement(self) -> None:
        if self.on_measurement is not None:
            self.on_measurement(self)

    def _notify_pre_measurement(self, info: MemoryInfo) -> None:
        if self.on_pre_measurement is not None:
            self.on_pre_measurement(self, info)


@dataclass(frozen=True)
class E91CHSHMeasurement:
    """One endpoint measurement of a delivered memory at an Ekert angle setting."""

    node: str
    local_memo: str
    remote_node: str
    remote_memo: str
    setting_index: int      # index into this side's angle set
    bloch_angle: float      # measurement Bloch angle (radians)
    outcome: int
    fidelity: float
    time_ps: int

    def pair_key(self) -> tuple[str, str, str, str]:
        return (self.node, self.local_memo, self.remote_node, self.remote_memo)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "local_memo": self.local_memo,
            "remote_node": self.remote_node,
            "remote_memo": self.remote_memo,
            "setting_index": self.setting_index,
            "bloch_angle": self.bloch_angle,
            "outcome": self.outcome,
            "fidelity": self.fidelity,
            "time_ps": self.time_ps,
        }


class E91CHSHEndpointApp(RequestApp):
    """Measure delivered endpoint memories at random Ekert-angle settings.

    Each side (Alice/Bob) carries its own polarization angle set; on delivery of
    a qualified entangled memory it picks a uniform random setting index and
    measures the memory qubit along the corresponding Bloch angle
    (``bloch_angle`` = 2x polarization angle) via :func:`_angle_measure_circuit`.
    The trial runner pairs Alice/Bob measurements and feeds the setting indices
    and outcomes to :mod:`sequence_game.protocol.chsh_core` for the CHSH-S
    statistic and the device-independent E91 accept/abort decision.
    """

    def __init__(self, node: "QuantumRouter", *, seed: int,
                 polarization_angles: tuple[float, ...],
                 release_after_measure: bool = True):
        super().__init__(node)
        if len(polarization_angles) < 2:
            raise ValueError("need at least two measurement angles")
        self.name = f"{self.node.name}.E91CHSHEndpointApp"
        self.polarization_angles = tuple(float(a) for a in polarization_angles)
        # Precompute the measurement circuit per setting index (Bloch = 2x pol).
        self._circuits = [
            _angle_measure_circuit(2.0 * angle) for angle in self.polarization_angles
        ]
        self.release_after_measure = release_after_measure
        self.rng = np.random.default_rng(seed)
        self.measurements: list[E91CHSHMeasurement] = []
        self.on_qualified_memory = None
        self.on_pre_measurement = None
        # Same observer contract as E91EndpointApp.on_measurement.
        self.on_measurement = None

    def get_memory(self, info: MemoryInfo) -> None:
        if info.state != MemoryInfo.ENTANGLED:
            return
        if info.index not in self.memo_to_reservation:
            return
        reservation: Reservation = self.memo_to_reservation[info.index]
        if info.remote_node not in (reservation.initiator, reservation.responder):
            return
        if info.fidelity < reservation.fidelity:
            return

        if (self.on_qualified_memory is not None
                and self.on_qualified_memory(self, info)):
            return
        self.measure_qualified_memory(info)

    def measure_qualified_memory(self, info: MemoryInfo) -> None:
        """Measure an already-qualified memory (used by pair coordination)."""

        self._notify_pre_measurement(info)
        setting_index = int(self.rng.integers(0, len(self.polarization_angles)))
        circuit = self._circuits[setting_index]
        key = info.memory.qstate_key
        meas_samp = max(float(self.rng.random()), 1e-15)
        result = self.node.timeline.quantum_manager.run_circuit(
            circuit, [key], meas_samp)
        outcome = int(result[key])

        self.measurements.append(E91CHSHMeasurement(
            node=self.node.name,
            local_memo=info.memory.name,
            remote_node=info.remote_node,
            remote_memo=info.remote_memo,
            setting_index=setting_index,
            bloch_angle=2.0 * self.polarization_angles[setting_index],
            outcome=outcome,
            fidelity=float(info.fidelity),
            time_ps=int(self.node.timeline.now()),
        ))
        self.memory_counter += 1
        if self.release_after_measure:
            self.node.resource_manager.update(None, info.memory, MemoryInfo.RAW)
        self._notify_measurement()

    def _notify_measurement(self) -> None:
        if self.on_measurement is not None:
            self.on_measurement(self)

    def _notify_pre_measurement(self, info: MemoryInfo) -> None:
        if self.on_pre_measurement is not None:
            self.on_pre_measurement(self, info)
