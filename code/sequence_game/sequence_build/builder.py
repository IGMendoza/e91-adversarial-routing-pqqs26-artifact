"""Minimal SeQUeNCe network builder: topology IR + physical registry -> timeline,
base nodes, and classical/quantum channels.

This is wiring only. It does not create memories, sources, detectors, or BSMs
(those adapters fail closed in ``device_adapters``), does not attach protocols,
and does not fake entanglement generation or QKD success. Channel parameters
are passed through to SeQUeNCe's ``QuantumChannel`` exactly as named in the
registry model; no values are invented here, so building requires a registry
whose channel models are fully resolved (toy or user-supplied).

This module is the only place (besides device_adapters) allowed to import
``sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sequence.components.optical_channel import ClassicalChannel, QuantumChannel
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from ..physical.registry import PhysicalModel, PhysicalRegistry
from ..topology.ir import TopologyIR

#: QuantumChannel constructor args we know how to pass through (verified
#: against sequence/components/optical_channel.py); distance comes from the
#: edge, name/timeline from the builder.
_QC_REQUIRED_PARAMS = ("attenuation", "polarization_fidelity")
_QC_OPTIONAL_PARAMS = ("light_speed", "frequency")


class BuildError(ValueError):
    """Topology/registry cannot be built into a SeQUeNCe network."""


@dataclass(frozen=True)
class BuildOptions:
    stop_time_ps: int
    with_quantum_channels: bool = True


@dataclass
class BuiltNetwork:
    timeline: Timeline
    nodes: dict[str, Node]
    cchannels: list[ClassicalChannel] = field(default_factory=list)
    qchannels: list[QuantumChannel] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_cchannels": len(self.cchannels),
            "num_qchannels": len(self.qchannels),
            "stop_time_ps": self.timeline.stop_time,
        }


def _channel_model_for_edge(ir: TopologyIR, registry: PhysicalRegistry,
                            edge_id: str, profile_id: str | None) -> PhysicalModel:
    if profile_id is None:
        raise BuildError(
            f"edge {edge_id!r} has no channel_profile_id; every edge must name an "
            "edge hardware profile to be built")
    if profile_id not in registry.edge_profiles:
        raise BuildError(f"edge {edge_id!r} references unknown edge profile {profile_id!r}")
    model = registry.get_model(registry.edge_profiles[profile_id].channel_model)
    model.require_resolved()
    missing = [p for p in _QC_REQUIRED_PARAMS if p not in model.parameters]
    if missing:
        raise BuildError(
            f"channel model {model.model_name!r} lacks required parameters {missing}")
    return model


def build_network(ir: TopologyIR, registry: PhysicalRegistry,
                  options: BuildOptions, seed: int) -> BuiltNetwork:
    """Build timeline + base nodes + channels. Fails closed on any device
    hardware profile (no device adapters are implemented yet)."""
    ir.validate()
    registry.validate_structure()

    for node_id in sorted(ir.nodes):
        profile_id = ir.nodes[node_id].hardware_profile_id
        if profile_id is not None:
            raise NotImplementedError(
                f"node {node_id!r} has hardware profile {profile_id!r}, but device "
                "adapters (source/memory/swap-BSM/detector) are not implemented; "
                "see sequence_game/sequence_build/device_adapters.py")

    timeline = Timeline(options.stop_time_ps)
    nodes: dict[str, Node] = {}
    for index, node_id in enumerate(sorted(ir.nodes)):
        node = Node(node_id, timeline)
        node.set_seed(seed + index)
        nodes[node_id] = node

    built = BuiltNetwork(timeline=timeline, nodes=nodes)

    for edge in ir.edges:
        for sender_id, receiver_id in ((edge.u, edge.v), (edge.v, edge.u)):
            cc = ClassicalChannel(f"cc.{edge.edge_id}.{sender_id}->{receiver_id}",
                                  timeline, distance=edge.length_m)
            cc.set_ends(nodes[sender_id], receiver_id)
            built.cchannels.append(cc)

        if options.with_quantum_channels:
            model = _channel_model_for_edge(ir, registry, edge.edge_id,
                                            edge.channel_profile_id)
            kwargs = {p: model.parameters[p] for p in _QC_REQUIRED_PARAMS}
            kwargs.update({p: model.parameters[p] for p in _QC_OPTIONAL_PARAMS
                           if p in model.parameters})
            for sender_id, receiver_id in ((edge.u, edge.v), (edge.v, edge.u)):
                qc = QuantumChannel(f"qc.{edge.edge_id}.{sender_id}->{receiver_id}",
                                    timeline, distance=edge.length_m, **kwargs)
                qc.set_ends(nodes[sender_id], receiver_id)
                built.qchannels.append(qc)

    timeline.init()
    return built
