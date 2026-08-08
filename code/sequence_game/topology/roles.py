"""Deterministic role assignment for Alice, Bob, swap candidates, and Eve
eligibility, as a pure function on the topology IR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Union

import numpy as np

from .ir import NodeRecord, TopologyError, TopologyIR


@dataclass(frozen=True)
class RoleAssignmentConfig:
    alice: Optional[str] = None
    bob: Optional[str] = None
    # "none", "all_non_endpoint", or an explicit list of node ids
    eve_eligible_nodes: Union[str, tuple[str, ...]] = "none"
    # "none", "all", or an explicit list of edge ids
    eve_eligible_edges: Union[str, tuple[str, ...]] = "none"
    allow_endpoint_eve: bool = False
    require_route: bool = True

    def to_dict(self) -> dict:
        return {
            "alice": self.alice,
            "bob": self.bob,
            "eve_eligible_nodes": list(self.eve_eligible_nodes)
            if isinstance(self.eve_eligible_nodes, tuple) else self.eve_eligible_nodes,
            "eve_eligible_edges": list(self.eve_eligible_edges)
            if isinstance(self.eve_eligible_edges, tuple) else self.eve_eligible_edges,
            "allow_endpoint_eve": self.allow_endpoint_eve,
            "require_route": self.require_route,
        }


def _has_route(ir: TopologyIR, a: str, b: str) -> bool:
    adj = ir.adjacency()
    seen, stack = {a}, [a]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt == b:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def assign_roles(ir: TopologyIR, config: RoleAssignmentConfig,
                 seed: Optional[int] = None) -> TopologyIR:
    """Return a new TopologyIR with roles assigned; the input is not mutated.

    Random Alice/Bob selection requires an explicit seed.
    """
    ir.validate()
    node_ids = sorted(ir.nodes)

    alice, bob = config.alice, config.bob
    if alice is None or bob is None:
        if seed is None:
            raise TopologyError("random Alice/Bob selection requires an explicit seed")
        rng = np.random.default_rng(seed)
        remaining = [nid for nid in node_ids if nid not in (alice, bob)]
        if alice is None:
            alice = remaining[int(rng.integers(len(remaining)))]
            remaining = [nid for nid in remaining if nid != alice]
        if bob is None:
            if not remaining:
                raise TopologyError("not enough nodes to pick distinct Alice and Bob")
            bob = remaining[int(rng.integers(len(remaining)))]

    for label, nid in (("alice", alice), ("bob", bob)):
        if nid not in ir.nodes:
            raise TopologyError(f"{label} node {nid!r} does not exist")
    if alice == bob:
        raise TopologyError(f"Alice and Bob must be distinct, both are {alice!r}")
    if config.require_route and not _has_route(ir, alice, bob):
        raise TopologyError(f"no route between Alice {alice!r} and Bob {bob!r}")

    endpoints = {alice, bob}

    if config.eve_eligible_nodes == "none":
        eve_nodes: set[str] = set()
    elif config.eve_eligible_nodes == "all_non_endpoint":
        eve_nodes = set(node_ids) - endpoints
    else:
        eve_nodes = set(config.eve_eligible_nodes)
        unknown = eve_nodes - set(node_ids)
        if unknown:
            raise TopologyError(f"unknown eve_eligible_nodes {sorted(unknown)}")
        forbidden = eve_nodes & endpoints
        if forbidden and not config.allow_endpoint_eve:
            raise TopologyError(
                f"eve_eligible_nodes includes endpoints {sorted(forbidden)}; "
                "set allow_endpoint_eve=True to permit this explicitly")

    edge_ids = {e.edge_id for e in ir.edges}
    if config.eve_eligible_edges == "none":
        eve_edges: set[str] = set()
    elif config.eve_eligible_edges == "all":
        eve_edges = set(edge_ids)
    else:
        eve_edges = set(config.eve_eligible_edges)
        unknown = eve_edges - edge_ids
        if unknown:
            raise TopologyError(f"unknown eve_eligible_edges {sorted(unknown)}")

    new_nodes: dict[str, NodeRecord] = {}
    for nid, record in ir.nodes.items():
        roles = set(record.roles)
        if nid == alice:
            roles.add("alice")
        elif nid == bob:
            roles.add("bob")
        else:
            roles.add("swap_candidate")
        if nid in eve_nodes:
            roles.add("eve_eligible")
        new_nodes[nid] = record.with_roles(frozenset(roles))

    new_edges = [replace(edge, eve_eligible=edge.edge_id in eve_edges) for edge in ir.edges]

    result = TopologyIR(nodes=new_nodes, edges=new_edges, metadata=ir.metadata)
    result.validate()
    return result
