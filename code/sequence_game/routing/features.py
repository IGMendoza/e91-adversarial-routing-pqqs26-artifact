"""Deterministic route feature extraction for control-game route sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..claims import CONTROL_GAME, IMPLEMENTED_CONTROL_ONLY
from ..topology.ir import TopologyIR
from .route import Route, enumerate_simple_paths, make_route

DETERMINISTIC_TIE_BREAK_NOTE = "deterministic_lexicographic_no_physical_preference"


@dataclass(frozen=True)
class RouteFeatureRow:
    route_id: str
    path: tuple[str, ...]
    hops: int
    length: float
    internal_nodes: tuple[str, ...]
    edges: tuple[str, ...]
    overlap_with_other_routes: dict[str, int]
    route_bottleneck_score: float
    eligible_eve_node_targets: tuple[str, ...]
    eligible_eve_edge_targets: tuple[str, ...]
    tie_break_basis: str = DETERMINISTIC_TIE_BREAK_NOTE
    scope_label: str = CONTROL_GAME
    claim_status: str = IMPLEMENTED_CONTROL_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "path": list(self.path),
            "hops": self.hops,
            "length": self.length,
            "internal_nodes": list(self.internal_nodes),
            "edges": list(self.edges),
            "overlap_with_other_routes": dict(sorted(self.overlap_with_other_routes.items())),
            "route_bottleneck_score": self.route_bottleneck_score,
            "eligible_eve_node_targets": list(self.eligible_eve_node_targets),
            "eligible_eve_edge_targets": list(self.eligible_eve_edge_targets),
            "tie_break_basis": self.tie_break_basis,
            "scope_label": self.scope_label,
            "claim_status": self.claim_status,
        }


def k_shortest_simple_routes(ir: TopologyIR,
                             source: str,
                             target: str,
                             *,
                             k: int = 12) -> list[Route]:
    """Return a stable capped list of simple routes ranked for control tests."""

    if k < 1:
        raise ValueError("k must be >= 1")
    paths = enumerate_simple_paths(ir, source, target)
    routes = [make_route(ir, path) for path in paths]
    routes.sort(key=route_sort_key)
    return routes[:k]


def route_sort_key(route: Route) -> tuple[float, int, tuple[str, ...]]:
    """Control-game ordering only; no physical preference is implied."""

    return (route.total_length_m, route.hop_count, route.path)


def route_feature_table(ir: TopologyIR,
                        routes: Iterable[Route],
                        *,
                        alice: str,
                        bob: str) -> list[RouteFeatureRow]:
    """Build deterministic per-route features for Alice/Eve control actions."""

    route_list = sorted(list(routes), key=route_sort_key)
    edge_counts: dict[str, int] = {}
    for route in route_list:
        for edge_id in route.edge_ids:
            edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1

    rows = []
    for route in route_list:
        route_edges = set(route.edge_ids)
        overlaps = {
            other.route_id: len(route_edges & set(other.edge_ids))
            for other in route_list
            if other.route_id != route.route_id
        }
        if route.edge_ids and route_list:
            bottleneck_score = max(edge_counts[edge_id] for edge_id in route.edge_ids) / len(route_list)
        else:
            bottleneck_score = 0.0
        internal_nodes = tuple(route.path[1:-1])
        rows.append(RouteFeatureRow(
            route_id=route.route_id,
            path=route.path,
            hops=route.hop_count,
            length=route.total_length_m,
            internal_nodes=internal_nodes,
            edges=route.edge_ids,
            overlap_with_other_routes=overlaps,
            route_bottleneck_score=float(bottleneck_score),
            eligible_eve_node_targets=tuple(
                node for node in internal_nodes
                if node not in {alice, bob} and "alice" not in ir.nodes[node].roles and "bob" not in ir.nodes[node].roles
            ),
            eligible_eve_edge_targets=route.edge_ids,
        ))
    return rows
