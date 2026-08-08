"""Legacy bare-fiber builder for one E91/BBM92 route.

This module is retained for regression coverage and historical BBM92/bare-fiber
experiments. New repeater-network E91 work should use
``sequence_build.repeater_e91_builder``.

A *route* is an ordered node path ``alice -> r1 -> ... -> bob`` whose consecutive
pairs are fiber edges. This builder lays out the entanglement-based QKD hardware
along that path:

- **Alice** (source): an ``SPDCSource`` (polarization) emitting a Bell pair per
  period; receiver[0] is Alice's local ``QSDetectorPolarization`` (she measures
  one arm locally), receiver[1] is a ``PhotonForwarder`` that injects the other
  arm onto the first fiber toward Bob.
- **Intermediate nodes**: passive ``PhotonForwarder``s (bare fiber junctions), or
  an Eve attack station supplied by ``station_factory`` for an inserted Eve node.
- **Bob** (sink): a ``QSDetectorPolarization`` that measures the arriving arm.
- One directed fiber ``QuantumChannel`` per hop (loss/latency/polarization noise
  per the resolved fiber model).

Scientific scope: the *physics* (entangled emission, fiber loss/noise, detection)
is SeQUeNCe's; this module only wires verified components together. There are no
quantum memories, repeaters, or entanglement swapping (bare-fiber design), so the
shared ``FreeQuantumState`` of each pair survives end-to-end and Alice/Bob
measurements are correlated by SeQUeNCe's quantum-state module (validated:
matched-basis correlation, mismatched-basis independence).

This module imports ``sequence`` (allowed, like the other builders).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sequence.components.detector import QSDetectorPolarization
from sequence.components.light_source import SPDCSource
from sequence.components.optical_channel import QuantumChannel
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from ..physical.registry import PhysicalModel
from ..topology.ir import TopologyIR
from .device_adapters import create_detector, create_fiber, create_source
from .relay import PhotonForwarder

#: Signature: (node_name, node, dest_name, timeline) -> forwarding component with
#: a ``get(photon)`` method, or None to use a passive PhotonForwarder.
StationFactory = Callable[[str, Node, str, Timeline], Optional[Any]]


class E91BuildError(ValueError):
    """Inconsistent route/hop specification for an E91 line build."""


@dataclass(frozen=True)
class E91Hop:
    """One directed fiber hop of the Bob-arm photon."""

    src: str
    dst: str
    length_m: float
    fiber_model: PhysicalModel


@dataclass
class BuiltE91Line:
    timeline: Timeline
    node_path: tuple[str, ...]
    alice: str
    bob: str
    source: SPDCSource
    alice_detector: QSDetectorPolarization
    bob_detector: QSDetectorPolarization
    nodes: dict[str, Node]
    qchannels: dict[tuple[str, str], QuantumChannel]
    forwarders: dict[str, Any]
    hops: tuple[E91Hop, ...]
    eve_nodes: tuple[str, ...] = ()

    @property
    def bob_arm_delay_ps(self) -> int:
        """Total propagation delay of the Bob-arm photon over all hops (ps).

        Valid only after ``timeline.init()`` (which the builder calls)."""
        return sum(qc.delay for qc in self.qchannels.values())

    def summary(self) -> dict[str, Any]:
        return {
            "node_path": list(self.node_path),
            "num_hops": len(self.hops),
            "num_qchannels": len(self.qchannels),
            "bob_arm_delay_ps": self.bob_arm_delay_ps,
            "eve_nodes": list(self.eve_nodes),
        }


def hops_for_path(ir: TopologyIR, node_path: list[str],
                  fiber_model: PhysicalModel) -> list[E91Hop]:
    """Build a hop list from a topology node path, using one fiber model for every
    edge and reading each edge's length from the topology."""
    if len(node_path) < 2:
        raise E91BuildError("node_path must have at least 2 nodes (alice, bob)")
    hops: list[E91Hop] = []
    for u, v in zip(node_path, node_path[1:]):
        edge = ir.edge_between(u, v)
        if edge is None:
            raise E91BuildError(f"no edge between {u!r} and {v!r} in topology")
        hops.append(E91Hop(u, v, edge.length_m, fiber_model))
    return hops


def build_e91_line(hops: list[E91Hop], *, alice: str, bob: str,
                   source_model: PhysicalModel, detector_model: PhysicalModel,
                   stop_time_ps: int, seed: int,
                   station_factory: Optional[StationFactory] = None,
                   eve_nodes: tuple[str, ...] = ()) -> BuiltE91Line:
    """Build the SeQUeNCe line network for one route.

    ``hops`` is the ordered physical hop list from ``alice`` to ``bob`` (callers
    insert Eve nodes by expanding a hop into two; see ``sequence_game.eve``).
    ``station_factory`` may override the forwarding component at intermediate nodes
    (used to place an Eve station); it is never consulted for ``alice``/``bob``.
    """
    if not hops:
        raise E91BuildError("need at least one hop")
    node_path = (hops[0].src,) + tuple(h.dst for h in hops)
    for i, h in enumerate(hops):
        if h.src != node_path[i] or h.dst != node_path[i + 1]:
            raise E91BuildError(f"hop {i} {(h.src, h.dst)} breaks the path chain")
    if node_path[0] != alice or node_path[-1] != bob:
        raise E91BuildError(
            f"path endpoints {node_path[0], node_path[-1]} != ({alice}, {bob})")
    if len(set(node_path)) != len(node_path):
        raise E91BuildError(f"route is not a simple path: {node_path}")

    timeline = Timeline(stop_time_ps)
    nodes: dict[str, Node] = {}
    for idx, nid in enumerate(node_path):
        node = Node(nid, timeline)
        node.set_seed(seed + idx)
        nodes[nid] = node

    dest_of = {h.src: h.dst for h in hops}
    # Each node's outgoing-hop polarization fidelity (the passive forwarder applies
    # that fiber's depolarization on the way out; see PhotonForwarder/quantum_ops).
    outhop_fidelity = {
        h.src: float(h.fiber_model.parameters["polarization_fidelity"]) for h in hops}

    # Alice: SPDC source + local detector + passive Bob-arm forwarder.
    alice_node = nodes[alice]
    source = create_source(source_model, f"{alice}.spdc", timeline)
    alice_node.add_component(source)
    alice_detector = create_detector(detector_model, f"{alice}.qsd", timeline)
    alice_node.add_component(alice_detector)
    bob_arm = PhotonForwarder(f"{alice}.bobarm", timeline, alice_node, dest_of[alice],
                              polarization_fidelity=outhop_fidelity[alice])
    alice_node.add_component(bob_arm)
    source.add_receiver(alice_detector)  # receiver[0] -> Alice's local arm
    source.add_receiver(bob_arm)         # receiver[1] -> Bob arm onto the fiber

    forwarders: dict[str, Any] = {alice: bob_arm}

    # Intermediate forwarding nodes (passive, or an Eve station via station_factory).
    for nid in node_path[1:-1]:
        node = nodes[nid]
        dest = dest_of[nid]
        comp = station_factory(nid, node, dest, timeline) if station_factory else None
        if comp is None:
            comp = PhotonForwarder(f"{nid}.fwd", timeline, node, dest,
                                   polarization_fidelity=outhop_fidelity[nid])
        node.add_component(comp)
        node.set_first_component(comp.name)
        forwarders[nid] = comp

    # Bob: detector sink.
    bob_node = nodes[bob]
    bob_detector = create_detector(detector_model, f"{bob}.qsd", timeline)
    bob_node.add_component(bob_detector)
    bob_node.set_first_component(bob_detector.name)

    # One directed fiber per hop. polarization_fidelity is forced to 1.0 because
    # SeQUeNCe's polarization depolarization (FreeQuantumState.random_noise) is not
    # entanglement-aware and corrupts the shared Bell state (see create_fiber).
    qchannels: dict[tuple[str, str], QuantumChannel] = {}
    for h in hops:
        qc = create_fiber(h.fiber_model, f"qc.{h.src}->{h.dst}", timeline, h.length_m,
                          polarization_fidelity=1.0)
        qc.set_ends(nodes[h.src], h.dst)
        qchannels[(h.src, h.dst)] = qc

    timeline.init()
    return BuiltE91Line(
        timeline=timeline, node_path=node_path, alice=alice, bob=bob,
        source=source, alice_detector=alice_detector, bob_detector=bob_detector,
        nodes=nodes, qchannels=qchannels, forwarders=forwarders,
        hops=tuple(hops), eve_nodes=tuple(eve_nodes),
    )
