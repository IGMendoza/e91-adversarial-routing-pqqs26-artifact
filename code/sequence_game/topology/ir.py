"""Topology intermediate representation, independent of SeQUeNCe objects.

Pure-data structures describing a network graph with role and hardware-profile
annotations. No simulator imports, no physics: edge ``length_m`` is carried as
data and is only given meaning by whichever (scope-labelled) channel model
later consumes it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

VALID_ROLES = frozenset({
    "alice",
    "bob",
    "swap_candidate",
    "memory_node",
    "detector_node",
    "eve_eligible",
})


class TopologyError(ValueError):
    """Invalid topology IR contents."""


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    roles: frozenset[str] = frozenset()
    hardware_profile_id: Optional[str] = None
    coordinates: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise TopologyError("node_id must be non-empty")
        object.__setattr__(self, "roles", frozenset(self.roles))
        bad = self.roles - VALID_ROLES
        if bad:
            raise TopologyError(
                f"node {self.node_id!r}: unknown roles {sorted(bad)}; valid: {sorted(VALID_ROLES)}")
        if self.coordinates is not None:
            object.__setattr__(self, "coordinates", tuple(float(c) for c in self.coordinates))

    def with_roles(self, roles: frozenset[str]) -> "NodeRecord":
        return NodeRecord(self.node_id, roles, self.hardware_profile_id, self.coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "roles": sorted(self.roles),
            "hardware_profile_id": self.hardware_profile_id,
            "coordinates": list(self.coordinates) if self.coordinates else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeRecord":
        coords = data.get("coordinates")
        return cls(
            node_id=data["node_id"],
            roles=frozenset(data.get("roles", [])),
            hardware_profile_id=data.get("hardware_profile_id"),
            coordinates=tuple(coords) if coords else None,
        )


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    u: str
    v: str
    length_m: float
    channel_profile_id: Optional[str] = None
    eve_eligible: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise TopologyError("edge_id must be non-empty")
        if self.u == self.v:
            raise TopologyError(f"edge {self.edge_id!r}: self-loop on {self.u!r}")
        if self.length_m < 0:
            raise TopologyError(f"edge {self.edge_id!r}: negative length {self.length_m}")

    def endpoints(self) -> frozenset[str]:
        return frozenset((self.u, self.v))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "u": self.u,
            "v": self.v,
            "length_m": self.length_m,
            "channel_profile_id": self.channel_profile_id,
            "eve_eligible": self.eve_eligible,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeRecord":
        return cls(**data)


@dataclass(frozen=True)
class TopologyMetadata:
    """Provenance of a generated topology (generator name, its parameters, seed)."""

    generator: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {"generator": self.generator, "params": dict(self.params), "seed": self.seed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyMetadata":
        return cls(generator=data["generator"], params=dict(data.get("params", {})),
                   seed=data.get("seed"))


@dataclass
class TopologyIR:
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    edges: list[EdgeRecord] = field(default_factory=list)
    metadata: TopologyMetadata = field(default_factory=lambda: TopologyMetadata("manual"))

    def validate(self) -> None:
        for node_id, record in self.nodes.items():
            if node_id != record.node_id:
                raise TopologyError(
                    f"node key {node_id!r} does not match record id {record.node_id!r}")
        seen_ids: set[str] = set()
        seen_pairs: set[frozenset[str]] = set()
        for edge in self.edges:
            if edge.edge_id in seen_ids:
                raise TopologyError(f"duplicate edge_id {edge.edge_id!r}")
            seen_ids.add(edge.edge_id)
            if edge.endpoints() in seen_pairs:
                raise TopologyError(f"duplicate edge between {edge.u!r} and {edge.v!r}")
            seen_pairs.add(edge.endpoints())
            for endpoint in (edge.u, edge.v):
                if endpoint not in self.nodes:
                    raise TopologyError(
                        f"edge {edge.edge_id!r} references unknown node {endpoint!r}")

    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            adj[edge.u].add(edge.v)
            adj[edge.v].add(edge.u)
        return adj

    def edge_between(self, u: str, v: str) -> Optional[EdgeRecord]:
        pair = frozenset((u, v))
        for edge in self.edges:
            if edge.endpoints() == pair:
                return edge
        return None

    def nodes_with_role(self, role: str) -> list[str]:
        if role not in VALID_ROLES:
            raise TopologyError(f"unknown role {role!r}")
        return sorted(nid for nid, rec in self.nodes.items() if role in rec.roles)

    def is_connected(self) -> bool:
        if not self.nodes:
            return False
        adj = self.adjacency()
        start = next(iter(self.nodes))
        seen = {start}
        stack = [start]
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return len(seen) == len(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: rec.to_dict() for nid, rec in sorted(self.nodes.items())},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyIR":
        ir = cls(
            nodes={nid: NodeRecord.from_dict(nd) for nid, nd in data.get("nodes", {}).items()},
            edges=[EdgeRecord.from_dict(ed) for ed in data.get("edges", [])],
            metadata=TopologyMetadata.from_dict(data.get("metadata", {"generator": "manual"})),
        )
        ir.validate()
        return ir

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TopologyIR":
        return cls.from_dict(json.loads(text))
