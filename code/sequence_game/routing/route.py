"""Route representation and path helpers over the topology IR."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..topology.ir import TopologyError, TopologyIR


class NoRouteError(TopologyError):
    """No route exists between the requested endpoints."""


@dataclass(frozen=True)
class Route:
    route_id: str
    path: tuple[str, ...]
    edge_ids: tuple[str, ...]
    total_length_m: float

    @property
    def hop_count(self) -> int:
        return len(self.edge_ids)

    def to_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "path": list(self.path),
            "edge_ids": list(self.edge_ids),
            "total_length_m": self.total_length_m,
            "hop_count": self.hop_count,
        }


def route_id_for_path(path: tuple[str, ...]) -> str:
    digest = hashlib.sha256("->".join(path).encode("utf-8")).hexdigest()
    return f"route-{digest[:16]}"


def make_route(ir: TopologyIR, path: tuple[str, ...]) -> Route:
    """Build a Route from an ordered node path, validating every hop."""
    if len(path) < 2:
        raise TopologyError(f"path must have at least 2 nodes, got {path}")
    if len(set(path)) != len(path):
        raise TopologyError(f"path revisits a node: {path}")
    edge_ids = []
    total = 0.0
    for u, v in zip(path, path[1:]):
        if u not in ir.nodes or v not in ir.nodes:
            raise TopologyError(f"path references unknown node in hop ({u!r}, {v!r})")
        edge = ir.edge_between(u, v)
        if edge is None:
            raise TopologyError(f"no edge between consecutive path nodes {u!r} and {v!r}")
        edge_ids.append(edge.edge_id)
        total += edge.length_m
    return Route(route_id_for_path(tuple(path)), tuple(path), tuple(edge_ids), total)


def enumerate_simple_paths(
    ir: TopologyIR,
    source: str,
    target: str,
) -> list[tuple[str, ...]]:
    """Enumerate every simple path in deterministic lexicographic DFS order."""
    for nid in (source, target):
        if nid not in ir.nodes:
            raise TopologyError(f"unknown node {nid!r}")
    adj = {nid: sorted(nbrs) for nid, nbrs in ir.adjacency().items()}
    results: list[tuple[str, ...]] = []

    def dfs(node: str, path: list[str]) -> None:
        if node == target:
            results.append(tuple(path))
            return
        for nbr in adj[node]:
            if nbr not in path:
                path.append(nbr)
                dfs(nbr, path)
                path.pop()

    dfs(source, [source])
    return results
