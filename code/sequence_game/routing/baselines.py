"""Baseline routing policies: shortest-hop, shortest-length, seeded-random
simple path, and fixed route. Purely graph-level, no physical claims."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Optional

import numpy as np

from ..topology.ir import TopologyError, TopologyIR
from .policy import RoutingPolicy
from .route import NoRouteError, Route, enumerate_simple_paths, make_route


class ShortestHopPolicy(RoutingPolicy):
    """BFS shortest path by hop count, deterministic tie-break by node id."""

    name = "shortest_hop"

    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        if source not in topology.nodes or target not in topology.nodes:
            raise TopologyError(f"unknown endpoint {source!r} or {target!r}")
        adj = {nid: sorted(nbrs) for nid, nbrs in topology.adjacency().items()}
        parent: dict[str, str] = {}
        seen = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    parent[nbr] = node
                    queue.append(nbr)
        if target not in seen:
            raise NoRouteError(f"no route from {source!r} to {target!r}")
        path = [target]
        while path[-1] != source:
            path.append(parent[path[-1]])
        return make_route(topology, tuple(reversed(path)))


class ShortestLengthPolicy(RoutingPolicy):
    """Dijkstra over edge length_m, deterministic tie-break by node id."""

    name = "shortest_length"

    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        if source not in topology.nodes or target not in topology.nodes:
            raise TopologyError(f"unknown endpoint {source!r} or {target!r}")
        lengths: dict[frozenset[str], float] = {
            edge.endpoints(): edge.length_m for edge in topology.edges}
        adj = {nid: sorted(nbrs) for nid, nbrs in topology.adjacency().items()}
        dist: dict[str, float] = {source: 0.0}
        parent: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, source)]
        done: set[str] = set()
        while heap:
            d, node = heapq.heappop(heap)
            if node in done:
                continue
            done.add(node)
            if node == target:
                break
            for nbr in adj[node]:
                nd = d + lengths[frozenset((node, nbr))]
                if nbr not in dist or nd < dist[nbr]:
                    dist[nbr] = nd
                    parent[nbr] = node
                    heapq.heappush(heap, (nd, nbr))
        if target not in done:
            raise NoRouteError(f"no route from {source!r} to {target!r}")
        path = [target]
        while path[-1] != source:
            path.append(parent[path[-1]])
        return make_route(topology, tuple(reversed(path)))


class SeededRandomSimplePathPolicy(RoutingPolicy):
    """Uniform choice among simple paths; requires an explicit RNG."""

    name = "seeded_random_simple_path"

    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        if rng is None:
            raise ValueError(f"{self.name} requires an explicit rng")
        candidates = enumerate_simple_paths(topology, source, target)
        if not candidates:
            raise NoRouteError(f"no route from {source!r} to {target!r}")
        path = candidates[int(rng.integers(len(candidates)))]
        return make_route(topology, path)

    def metadata(self) -> dict[str, Any]:
        return {"policy": self.name}


class FixedRoutePolicy(RoutingPolicy):
    """Always returns the configured path (validated against the topology)."""

    name = "fixed_route"

    def __init__(self, path: tuple[str, ...]):
        self.path = tuple(path)

    def select_route(self, topology: TopologyIR, source: str, target: str, *,
                     rng: Optional[np.random.Generator] = None,
                     context: Optional[dict[str, Any]] = None) -> Route:
        if not self.path or self.path[0] != source or self.path[-1] != target:
            raise NoRouteError(
                f"fixed path {self.path} does not connect {source!r} to {target!r}")
        return make_route(topology, self.path)

    def metadata(self) -> dict[str, Any]:
        return {"policy": self.name, "path": list(self.path)}
