"""Graph metrics and route-structure analysis for corpus generation.

These helpers are intentionally graph-structural. They do not invoke the
SeQUeNCe runtime, attacks, learners, hardware assignment, or XAI layers.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Hashable, Optional

from sequence_game.routing.route import Route, make_route
from sequence_game.topology import TopologyIR

SHORTEST_PATH_COUNT_CAP = 1_000_000
FLOW_INFINITY_CAPACITY = 1_000_000


@dataclass(frozen=True)
class RouteStructureRecord:
    route_count: int
    shortest_hops: Optional[int]
    longest_hops: Optional[int]
    shortest_path_count: int
    node_cut_size: Optional[int]
    edge_cut_size: Optional[int]
    candidate_node_hitting_size: Optional[int]
    candidate_edge_hitting_size: Optional[int]
    route_overlap_summary: dict[str, Any]
    candidate_routes: list[dict[str, Any]]
    node_cut: dict[str, Any]
    edge_cut: dict[str, Any]
    candidate_node_hitting_set: dict[str, Any]
    candidate_edge_hitting_set: dict[str, Any]
    route_enumeration_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_count": self.route_count,
            "shortest_hops": self.shortest_hops,
            "longest_hops": self.longest_hops,
            "shortest_path_count": self.shortest_path_count,
            "node_cut_size": self.node_cut_size,
            "edge_cut_size": self.edge_cut_size,
            "candidate_node_hitting_size": self.candidate_node_hitting_size,
            "candidate_edge_hitting_size": self.candidate_edge_hitting_size,
            "route_overlap_summary": self.route_overlap_summary,
            "candidate_routes": self.candidate_routes,
            "node_cut": self.node_cut,
            "edge_cut": self.edge_cut,
            "candidate_node_hitting_set": self.candidate_node_hitting_set,
            "candidate_edge_hitting_set": self.candidate_edge_hitting_set,
            "route_enumeration_truncated": self.route_enumeration_truncated,
        }


def analyze_route_structure(
    graph: TopologyIR,
    alice: str,
    bob: str,
    *,
    max_route_candidates: int,
    route_probe_limit: int,
    exact_hitting_set_max_size: int,
    shortest_data: Optional[dict[str, tuple[int, int]]] = None,
) -> RouteStructureRecord:
    if shortest_data is None:
        shortest_data = single_source_shortest_path_data(graph, alice)
    if bob not in shortest_data:
        return _disconnected_route_structure()

    shortest_hops, shortest_path_count = shortest_data[bob]
    node_cut = minimum_st_internal_node_cut(graph, alice, bob)
    edge_cut = minimum_st_edge_cut(graph, alice, bob)
    routes, truncated = candidate_routes_bounded(
        graph,
        alice,
        bob,
        max_candidates=max_route_candidates,
        probe_limit=route_probe_limit,
    )
    shortest_route = min(routes, key=lambda route: (route.hop_count, route.path)) if routes else None
    longest_route = max(routes, key=lambda route: (route.hop_count, route.path)) if routes else None
    internal_node_sets = [set(route.path[1:-1]) for route in routes]
    edge_sets = [set(route.edge_ids) for route in routes]
    node_hitting = minimum_hitting_set(
        internal_node_sets,
        label="internal_nodes",
        max_exact_size=exact_hitting_set_max_size,
    )
    edge_hitting = minimum_hitting_set(
        edge_sets,
        label="edges",
        max_exact_size=exact_hitting_set_max_size,
    )
    return RouteStructureRecord(
        route_count=len(routes),
        shortest_hops=shortest_route.hop_count if shortest_route else shortest_hops,
        longest_hops=longest_route.hop_count if longest_route else shortest_hops,
        shortest_path_count=shortest_path_count,
        node_cut_size=_size_or_none(node_cut),
        edge_cut_size=_size_or_none(edge_cut),
        candidate_node_hitting_size=_size_or_none(node_hitting),
        candidate_edge_hitting_size=_size_or_none(edge_hitting),
        route_overlap_summary=route_overlap_summary(routes),
        candidate_routes=[route_row(route) for route in routes],
        node_cut=node_cut,
        edge_cut=edge_cut,
        candidate_node_hitting_set=node_hitting,
        candidate_edge_hitting_set=edge_hitting,
        route_enumeration_truncated=truncated,
    )


def _disconnected_route_structure() -> RouteStructureRecord:
    empty_cut = {
        "set_kind": "none",
        "size": None,
        "sets": [],
        "exact": True,
        "reason": "alice and bob are disconnected",
    }
    return RouteStructureRecord(
        route_count=0,
        shortest_hops=None,
        longest_hops=None,
        shortest_path_count=0,
        node_cut_size=None,
        edge_cut_size=None,
        candidate_node_hitting_size=None,
        candidate_edge_hitting_size=None,
        route_overlap_summary={"route_count": 0},
        candidate_routes=[],
        node_cut=empty_cut,
        edge_cut=empty_cut,
        candidate_node_hitting_set=empty_cut,
        candidate_edge_hitting_set=empty_cut,
        route_enumeration_truncated=False,
    )


def candidate_routes_bounded(
    graph: TopologyIR,
    alice: str,
    bob: str,
    *,
    max_candidates: int,
    probe_limit: int,
) -> tuple[list[Route], bool]:
    adjacency = {node: sorted(neighbors) for node, neighbors in graph.adjacency().items()}
    edge_lengths = {frozenset((edge.u, edge.v)): edge.length_m for edge in graph.edges}
    routes: list[Route] = []
    frontier: list[tuple[float, int, tuple[str, ...]]] = [(0.0, 0, (alice,))]
    probes = 0
    truncated = False

    while frontier and len(routes) < max_candidates:
        if probes >= probe_limit:
            truncated = True
            break
        total_length, hop_count, path = heapq.heappop(frontier)
        probes += 1
        node = path[-1]
        if node == bob:
            routes.append(make_route(graph, path))
            continue
        for neighbor in adjacency[node]:
            if neighbor in path:
                continue
            next_path = (*path, neighbor)
            heapq.heappush(
                frontier,
                (
                    total_length + edge_lengths[frozenset((node, neighbor))],
                    hop_count + 1,
                    next_path,
                ),
            )
    return routes, truncated or bool(frontier)


def all_pairs_shortest_path_data(
    graph: TopologyIR,
) -> dict[str, dict[str, tuple[int, int]]]:
    adjacency = {node: sorted(neighbors) for node, neighbors in graph.adjacency().items()}
    return {
        source: single_source_shortest_path_data(graph, source, adjacency=adjacency)
        for source in sorted(graph.nodes)
    }


def single_source_shortest_path_data(
    graph: TopologyIR,
    source: str,
    *,
    adjacency: Optional[dict[str, list[str]]] = None,
) -> dict[str, tuple[int, int]]:
    if adjacency is None:
        adjacency = {node: sorted(neighbors) for node, neighbors in graph.adjacency().items()}
    distance = {source: 0}
    path_count = {source: 1}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distance[node] + 1
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = next_distance
                path_count[neighbor] = path_count[node]
                queue.append(neighbor)
            elif distance[neighbor] == next_distance:
                path_count[neighbor] = min(
                    SHORTEST_PATH_COUNT_CAP,
                    path_count[neighbor] + path_count[node],
                )
    return {node: (hops, path_count[node]) for node, hops in distance.items()}


def minimum_st_edge_cut(graph: TopologyIR, alice: str, bob: str) -> dict[str, Any]:
    capacity: dict[Hashable, dict[Hashable, int]] = {}
    for edge in graph.edges:
        _add_capacity(capacity, edge.u, edge.v, 1)
        _add_capacity(capacity, edge.v, edge.u, 1)
    _, reachable = _max_flow_min_cut(capacity, alice, bob)
    cut_edges = sorted(
        edge.edge_id
        for edge in graph.edges
        if (edge.u in reachable) != (edge.v in reachable)
    )
    return {
        "set_kind": "edges",
        "size": len(cut_edges),
        "sets": [cut_edges],
        "exact": True,
        "method": "unit_capacity_st_edge_cut",
    }


def minimum_st_internal_node_cut(graph: TopologyIR, alice: str, bob: str) -> dict[str, Any]:
    capacity: dict[Hashable, dict[Hashable, int]] = {}
    for node in graph.nodes:
        node_capacity = FLOW_INFINITY_CAPACITY if node in {alice, bob} else 1
        _add_capacity(capacity, _node_in(node), _node_out(node), node_capacity)
    for edge in graph.edges:
        _add_capacity(capacity, _node_out(edge.u), _node_in(edge.v), FLOW_INFINITY_CAPACITY)
        _add_capacity(capacity, _node_out(edge.v), _node_in(edge.u), FLOW_INFINITY_CAPACITY)

    flow, reachable = _max_flow_min_cut(capacity, _node_out(alice), _node_in(bob))
    if flow >= FLOW_INFINITY_CAPACITY:
        return {
            "set_kind": "internal_nodes",
            "size": None,
            "sets": [],
            "exact": True,
            "method": "node_split_unit_capacity_st_vertex_cut",
            "reason": "no finite internal-node cut exists for this pair",
        }
    cut_nodes = sorted(
        node
        for node in graph.nodes
        if node not in {alice, bob}
        and _node_in(node) in reachable
        and _node_out(node) not in reachable
    )
    return {
        "set_kind": "internal_nodes",
        "size": len(cut_nodes),
        "sets": [cut_nodes],
        "exact": True,
        "method": "node_split_unit_capacity_st_vertex_cut",
    }


def route_overlap_summary(routes: list[Route]) -> dict[str, Any]:
    if len(routes) < 2:
        return {
            "route_count": len(routes),
            "mean_internal_node_jaccard": 0.0,
            "mean_edge_jaccard": 0.0,
            "max_shared_internal_nodes": 0,
            "max_shared_edges": 0,
        }
    node_jaccards = []
    edge_jaccards = []
    max_shared_nodes = 0
    max_shared_edges = 0
    for left, right in itertools.combinations(routes, 2):
        left_nodes = set(left.path[1:-1])
        right_nodes = set(right.path[1:-1])
        left_edges = set(left.edge_ids)
        right_edges = set(right.edge_ids)
        max_shared_nodes = max(max_shared_nodes, len(left_nodes & right_nodes))
        max_shared_edges = max(max_shared_edges, len(left_edges & right_edges))
        node_jaccards.append(_jaccard(left_nodes, right_nodes))
        edge_jaccards.append(_jaccard(left_edges, right_edges))
    return {
        "route_count": len(routes),
        "mean_internal_node_jaccard": sum(node_jaccards) / len(node_jaccards),
        "mean_edge_jaccard": sum(edge_jaccards) / len(edge_jaccards),
        "max_shared_internal_nodes": max_shared_nodes,
        "max_shared_edges": max_shared_edges,
    }


def minimum_hitting_set(
    sets: list[set[str]],
    *,
    label: str,
    max_exact_size: int,
) -> dict[str, Any]:
    universe = sorted(set().union(*sets)) if sets else []
    if any(not route_set for route_set in sets):
        return {
            "set_kind": label,
            "size": None,
            "sets": [],
            "exact": True,
            "reason": "at least one candidate route has no elements of this kind",
        }
    for size in range(1, min(max_exact_size, len(universe)) + 1):
        found = []
        for combo in itertools.combinations(universe, size):
            candidate = set(combo)
            if all(candidate & route_set for route_set in sets):
                found.append(list(combo))
                if len(found) >= 8:
                    break
        if found:
            return {
                "set_kind": label,
                "size": size,
                "sets": found,
                "exact": True,
            }
    return {
        "set_kind": label,
        "size": None,
        "sets": [],
        "exact": False,
        "lower_bound": min(max_exact_size + 1, len(universe) + 1),
        "reason": f"no hitting set found up to size {max_exact_size}",
    }


def route_row(route: Route) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "path": list(route.path),
        "edge_ids": list(route.edge_ids),
        "hop_count": route.hop_count,
        "total_length_m": route.total_length_m,
        "internal_nodes": list(route.path[1:-1]),
    }


def connected_components(graph: TopologyIR) -> list[set[str]]:
    unseen = set(graph.nodes)
    adjacency = graph.adjacency()
    components = []
    while unseen:
        start = min(unseen)
        component = {start}
        queue = deque([start])
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda comp: (-len(comp), sorted(comp)[0]))
    return components


def _max_flow_min_cut(
    capacity: dict[Hashable, dict[Hashable, int]],
    source: Hashable,
    sink: Hashable,
) -> tuple[int, set[Hashable]]:
    residual = {node: dict(neighbors) for node, neighbors in capacity.items()}
    for node, neighbors in list(residual.items()):
        for neighbor in list(neighbors):
            residual.setdefault(neighbor, {})
            residual[neighbor].setdefault(node, 0)

    flow = 0
    while True:
        parents: dict[Hashable, Optional[Hashable]] = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbor, available in residual[node].items():
                if available <= 0 or neighbor in parents:
                    continue
                parents[neighbor] = node
                queue.append(neighbor)
                if neighbor == sink:
                    break
        if sink not in parents:
            break

        bottleneck = FLOW_INFINITY_CAPACITY
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            bottleneck = min(bottleneck, residual[parent][node])
            node = parent

        node = sink
        while parents[node] is not None:
            parent = parents[node]
            residual[parent][node] -= bottleneck
            residual[node][parent] += bottleneck
            node = parent
        flow += bottleneck

    reachable = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor, available in residual[node].items():
            if available > 0 and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return flow, reachable


def _add_capacity(
    capacity: dict[Hashable, dict[Hashable, int]],
    u: Hashable,
    v: Hashable,
    amount: int,
) -> None:
    capacity.setdefault(u, {})
    capacity.setdefault(v, {})
    capacity[u][v] = capacity[u].get(v, 0) + amount


def _node_in(node: str) -> tuple[str, str]:
    return node, "in"


def _node_out(node: str) -> tuple[str, str]:
    return node, "out"


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _size_or_none(payload: dict[str, Any]) -> Optional[int]:
    value = payload.get("size")
    return int(value) if isinstance(value, int) else None
