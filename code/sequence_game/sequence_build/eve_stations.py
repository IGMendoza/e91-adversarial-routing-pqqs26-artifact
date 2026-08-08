"""Eve attack stations: SeQUeNCe components placed on an attacked fiber edge.

These are hardware-level components (hence under ``sequence_build``, where the
``sequence`` import lives). Each is inserted as the forwarding component of an
Eve node sitting at the far end of the attacked fiber, so Eve is never inside
Alice's or Bob's node. They share the ``get(photon)`` interface with
``PhotonForwarder``.

- ``AddedLossStation``: availability attack. Drops each photon with a configured
  probability (1.0 = full denial of the edge); otherwise forwards unchanged.
- ``InterceptResendStation``: intercept-resend. Measures each in-flight photon in
  a chosen polarization basis (collapsing the shared entangled state), records the
  classical (basis, outcome, time), and resends a fresh product-state photon in
  the measured eigenstate. This is a genuine measurement on SeQUeNCe state, so the
  resulting QBER and Eve's record are simulator truth, not invented values.

Randomness uses the owning node's seeded generator, so a fixed build seed makes
every attack deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sequence.components.photon import Photon
from sequence.kernel.entity import Entity
from sequence.utils.encoding import polarization

if TYPE_CHECKING:
    from sequence.kernel.timeline import Timeline
    from sequence.topology.node import Node

BASIS_LABELS = ("Z", "X")  # index 0 = rectilinear, index 1 = diagonal (matches encoding)


class AddedLossStation(Entity):
    """Drop photons with ``drop_probability`` (DoS); forward the rest unchanged."""

    def __init__(self, name: str, timeline: "Timeline", node: "Node", dest_name: str,
                 drop_probability: float = 1.0):
        super().__init__(name, timeline)
        if not 0.0 <= drop_probability <= 1.0:
            raise ValueError(f"drop_probability must be in [0, 1], got {drop_probability}")
        self.node = node
        self.dest_name = dest_name
        self.drop_probability = drop_probability
        self.received = 0
        self.dropped = 0

    def init(self) -> None:
        pass

    def get(self, photon: "Photon", **kwargs) -> None:
        self.received += 1
        if self.node.get_generator().random() < self.drop_probability:
            self.dropped += 1
            return  # photon absorbed -> never reaches Bob
        self.node.send_qubit(self.dest_name, photon)


class InterceptResendStation(Entity):
    """Measure each photon in a chosen basis, record (basis, outcome, time), and
    resend a fresh eigenstate photon toward the next hop."""

    def __init__(self, name: str, timeline: "Timeline", node: "Node", dest_name: str,
                 basis_choice: str = "random"):
        super().__init__(name, timeline)
        if basis_choice not in ("random", "Z", "X"):
            raise ValueError(f"basis_choice must be 'random'|'Z'|'X', got {basis_choice!r}")
        self.node = node
        self.dest_name = dest_name
        self.basis_choice = basis_choice
        self.received = 0
        #: Eve's private classical record, one dict per intercepted photon.
        self.observations: list[dict[str, Any]] = []

    def init(self) -> None:
        pass

    def get(self, photon: "Photon", **kwargs) -> None:
        self.received += 1
        rng = self.node.get_generator()
        if self.basis_choice == "random":
            basis_idx = int(rng.integers(0, 2))
        else:
            basis_idx = BASIS_LABELS.index(self.basis_choice)

        outcome = Photon.measure(polarization["bases"][basis_idx], photon, rng)
        self.observations.append({
            "time": self.timeline.now(),
            "basis": BASIS_LABELS[basis_idx],
            "basis_idx": basis_idx,
            "outcome": int(outcome),
        })

        eigenstate = tuple(complex(c) for c in polarization["bases"][basis_idx][outcome])
        resent = Photon(f"{self.name}.resend{self.received}", self.timeline,
                        location=self.node, encoding_type=polarization,
                        quantum_state=eigenstate)
        self.node.send_qubit(self.dest_name, resent)
