"""Passive photon relay for multi-hop bare-fiber routes.

A route in this game is a chain of fiber edges with no quantum repeaters: the
Bob-arm photon of each entangled pair is forwarded hop-to-hop, accumulating the
physical loss/noise of each ``QuantumChannel`` it crosses. ``PhotonForwarder`` is
the thin first-party component that does this forwarding. It performs no
measurement and adds no loss of its own (all physical loss lives in the fibers);
it only provides connectivity, so the shared entangled ``FreeQuantumState`` of
the pair is preserved as the photon travels toward Bob.

Eve's attack stations (see ``sequence_game.eve``) are *alternatives* to a passive
forwarder at an inserted Eve node; they share the same ``get(photon)`` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sequence.kernel.entity import Entity

from .quantum_ops import apply_polarization_depolarization

if TYPE_CHECKING:
    from sequence.components.photon import Photon
    from sequence.kernel.timeline import Timeline
    from sequence.topology.node import Node


class PhotonForwarder(Entity):
    """Forward each received photon over its owner node's quantum channel to the
    next hop, applying that outgoing fiber's polarization depolarization.

    ``polarization_fidelity`` is the *outgoing* hop's literature fidelity. With
    probability ``1 - fidelity`` an entanglement-correct random-unitary
    depolarization is applied to the photon's arm (``quantum_ops``), replacing
    SeQUeNCe's entanglement-unsafe channel noise. fidelity 1.0 = lossless/
    noiseless forwarding."""

    def __init__(self, name: str, timeline: "Timeline", node: "Node", dest_name: str,
                 polarization_fidelity: float = 1.0):
        super().__init__(name, timeline)
        self.node = node
        self.dest_name = dest_name
        self.polarization_fidelity = polarization_fidelity
        self.forward_count = 0
        self.noised_count = 0

    def init(self) -> None:
        pass

    def get(self, photon: "Photon", **kwargs) -> None:
        self.forward_count += 1
        if apply_polarization_depolarization(
                photon, self.polarization_fidelity, self.node.get_generator()):
            self.noised_count += 1
        self.node.send_qubit(self.dest_name, photon)
