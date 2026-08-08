"""Hardware-profile assignment for generated topology IRs.

This is a pure-data pass. It assigns existing node/edge profile identifiers to
Alice/Bob endpoints, repeater/interior nodes, and graph edges. It does not
create physical parameters, adapters, or simulator objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..physical.registry import PhysicalRegistry
from .ir import NodeRecord, TopologyError, TopologyIR


@dataclass(frozen=True)
class HardwareAssignmentConfig:
    endpoint_profile_id: str
    repeater_profile_id: str
    edge_profile_id: str
    require_roles: bool = True

    def to_dict(self) -> dict:
        return {
            "endpoint_profile_id": self.endpoint_profile_id,
            "repeater_profile_id": self.repeater_profile_id,
            "edge_profile_id": self.edge_profile_id,
            "require_roles": self.require_roles,
        }


def _validate_profile_refs(config: HardwareAssignmentConfig,
                           registry: Optional[PhysicalRegistry]) -> None:
    if registry is None:
        return
    missing_nodes = [
        profile_id for profile_id in
        (config.endpoint_profile_id, config.repeater_profile_id)
        if profile_id not in registry.node_profiles
    ]
    if missing_nodes:
        raise TopologyError(f"unknown node hardware profiles {missing_nodes}")
    if config.edge_profile_id not in registry.edge_profiles:
        raise TopologyError(f"unknown edge hardware profile {config.edge_profile_id!r}")
    registry.validate_structure()


def assign_hardware_profiles(
        ir: TopologyIR,
        config: HardwareAssignmentConfig,
        *,
        registry: Optional[PhysicalRegistry] = None,
) -> TopologyIR:
    """Return a topology with hardware profile IDs assigned.

    Nodes with role ``alice`` or ``bob`` receive ``endpoint_profile_id``.
    All other nodes receive ``repeater_profile_id``. If ``require_roles`` is
    true, the topology must already have exactly one Alice and one Bob role.
    """

    ir.validate()
    _validate_profile_refs(config, registry)

    alice_nodes = ir.nodes_with_role("alice")
    bob_nodes = ir.nodes_with_role("bob")
    if config.require_roles and (len(alice_nodes) != 1 or len(bob_nodes) != 1):
        raise TopologyError(
            "hardware assignment requires exactly one alice and one bob role")

    new_nodes: dict[str, NodeRecord] = {}
    for node_id, node in ir.nodes.items():
        if "alice" in node.roles or "bob" in node.roles:
            profile_id = config.endpoint_profile_id
        else:
            profile_id = config.repeater_profile_id
        new_nodes[node_id] = replace(node, hardware_profile_id=profile_id)

    new_edges = [
        replace(edge, channel_profile_id=config.edge_profile_id)
        for edge in ir.edges
    ]
    result = TopologyIR(nodes=new_nodes, edges=new_edges, metadata=ir.metadata)
    result.validate()
    return result
