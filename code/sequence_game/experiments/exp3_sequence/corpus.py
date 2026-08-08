"""Capped graph corpus for Exp3/oracle SeQUeNCe runs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from sequence_game.corpus.runner import corpus_candidate_routes
from sequence_game.topology import EdgeRecord, NodeRecord, TopologyIR, TopologyMetadata

from .config import CORPUS_BASE_EDGE_LENGTH_M, CORPUS_EDGE_LENGTH_STEP_M

CORPUS_SQLITE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GraphCase:
    graph_id: str
    family: str
    alice: str
    bob: str
    ir_dict: dict[str, Any]
    assignment: dict[str, Any]
    routes: list[dict[str, Any]]
    route_selection: dict[str, Any]
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "alice": self.alice,
            "bob": self.bob,
            "assignment": self.assignment,
            "routes": self.routes,
            "route_selection": self.route_selection,
            "features": self.features,
            "topology": self.ir_dict,
        }


def graph_case_from_dict(payload: dict[str, Any]) -> GraphCase:
    topology = payload.get("topology", payload.get("ir_dict"))
    if not isinstance(topology, dict):
        raise ValueError("graph case payload is missing topology")
    return GraphCase(
        graph_id=str(payload["graph_id"]),
        family=str(payload["family"]),
        alice=str(payload["alice"]),
        bob=str(payload["bob"]),
        ir_dict=topology,
        assignment=dict(payload.get("assignment", {})),
        routes=list(payload.get("routes", [])),
        route_selection=dict(payload.get("route_selection", {})),
        features={
            str(key): float(value)
            for key, value in dict(payload.get("features", {})).items()
        },
    )


def write_graph_cases_sqlite(
        path: Path,
        cases: list[GraphCase],
        *,
        metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_payload = {
        "schema_version": CORPUS_SQLITE_SCHEMA_VERSION,
        **(metadata or {}),
    }
    con = sqlite3.connect(path)
    try:
        with con:
            con.executescript(
                """
                DROP TABLE IF EXISTS metadata;
                DROP TABLE IF EXISTS graph_cases;

                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE graph_cases (
                    position INTEGER PRIMARY KEY,
                    graph_id TEXT NOT NULL UNIQUE,
                    family TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            con.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                [
                    (str(key), json.dumps(value, sort_keys=True))
                    for key, value in sorted(metadata_payload.items())
                ],
            )
            con.executemany(
                """
                INSERT INTO graph_cases(position, graph_id, family, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        index,
                        case.graph_id,
                        case.family,
                        json.dumps(case.to_dict(), sort_keys=True),
                    )
                    for index, case in enumerate(cases)
                ],
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_cases_family ON graph_cases(family)"
            )
    finally:
        con.close()


def load_graph_cases_from_sqlite(path: Path, *, limit: int | None = None) -> list[GraphCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exp3 corpus SQLite file not found: {path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    con = sqlite3.connect(path)
    try:
        query = "SELECT payload_json FROM graph_cases ORDER BY position"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()
    return [graph_case_from_dict(json.loads(row[0])) for row in rows]


def read_corpus_sqlite_metadata(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exp3 corpus SQLite file not found: {path}")
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT key, value_json FROM metadata ORDER BY key"
        ).fetchall()
    finally:
        con.close()
    return {str(key): json.loads(value_json) for key, value_json in rows}


def corpus_summary(cases: list[GraphCase]) -> dict[str, Any]:
    route_hops = [
        int(route["hop_count"])
        for case in cases
        for route in case.routes
    ]
    route_lengths_m = [
        float(route["total_length_m"])
        for case in cases
        for route in case.routes
    ]
    families: dict[str, int] = {}
    for case in cases:
        families[case.family] = families.get(case.family, 0) + 1
    return {
        "graph_count": len(cases),
        "route_count": len(route_hops),
        "max_route_hops": max(route_hops) if route_hops else 0,
        "max_routes_per_graph": max((len(case.routes) for case in cases), default=0),
        "max_node_disjoint_paths": max(
            (case.features["node_disjoint_paths"] for case in cases),
            default=0.0,
        ),
        "mean_route_length_m": (
            sum(route_lengths_m) / len(route_lengths_m)
            if route_lengths_m else 0.0
        ),
        "families": families,
    }


def build_graph_cases(*, max_graphs: int, max_routes: int, max_route_hops: int,
                      base_seed: int) -> list[GraphCase]:
    specs = _default_specs(base_seed)
    cases: list[GraphCase] = []
    for graph_id, family, ir in specs:
        if len(cases) >= max_graphs:
            break
        alice = "a"
        bob = "b"
        assignment = _assignment_shell(graph_id, family)
        routes, route_selection = corpus_candidate_routes(
            ir.to_dict(),
            alice,
            bob,
            assignment,
        )
        routes = [
            route
            for route in routes
            if int(route["hop_count"]) <= int(max_route_hops)
        ]
        route_selection = {
            **route_selection,
            "route_selection_policy": "historical_hop_bounded_exact",
            "route_hop_cutoff": int(max_route_hops),
        }
        if not routes:
            continue
        if len(routes) > max_routes:
            routes = routes[:max_routes]
            route_selection = {
                **route_selection,
                "route_enumeration_truncated": True,
                "route_truncated_to_max_routes": max_routes,
            }
        features = graph_features(ir, routes, alice, bob)
        assignment = {
            **assignment,
            "route_count": len(routes),
            "shortest_hops": int(features["shortest_hops"]),
            "longest_hops": int(features["longest_hops"]),
            "node_cut_size": int(features["node_disjoint_paths"]),
            "edge_cut_size": int(features["edge_disjoint_paths"]),
        }
        cases.append(GraphCase(
            graph_id=graph_id,
            family=family,
            alice=alice,
            bob=bob,
            ir_dict=ir.to_dict(),
            assignment=assignment,
            routes=routes,
            route_selection=route_selection,
            features=features,
        ))
    return cases


def graph_features(ir: TopologyIR, routes: list[dict[str, Any]],
                   alice: str, bob: str) -> dict[str, float]:
    graph = _nx_graph(ir)
    node_disjoint = nx.node_connectivity(graph, alice, bob)
    edge_disjoint = nx.edge_connectivity(graph, alice, bob)
    hops = [int(route["hop_count"]) for route in routes]
    lengths = [float(route["total_length_m"]) for route in routes]
    bottlenecks = bottleneck_nodes(routes)
    overlaps = []
    route_sets = [set(route.get("internal_nodes") or []) for route in routes]
    for i, left in enumerate(route_sets):
        for right in route_sets[i + 1:]:
            overlaps.append(len(left & right))
    return {
        "num_nodes": float(len(ir.nodes)),
        "num_edges": float(len(ir.edges)),
        "num_routes": float(len(routes)),
        "node_disjoint_paths": float(node_disjoint),
        "edge_disjoint_paths": float(edge_disjoint),
        "shortest_hops": float(min(hops)),
        "longest_hops": float(max(hops)),
        "route_hop_ratio": float(max(hops) / max(1, min(hops))),
        "bottleneck_node_count": float(len(bottlenecks)),
        "mean_edge_length_m": float(sum(edge.length_m for edge in ir.edges) / len(ir.edges)),
        "mean_route_length_m": float(sum(lengths) / len(lengths)),
        "max_route_overlap": float(max(overlaps) if overlaps else 0),
    }


def bottleneck_nodes(routes: list[dict[str, Any]]) -> set[str]:
    if not routes:
        return set()
    common = set(routes[0].get("internal_nodes") or [])
    for route in routes[1:]:
        common &= set(route.get("internal_nodes") or [])
    return common


def _default_specs(base_seed: int) -> list[tuple[str, str, TopologyIR]]:
    specs: list[tuple[str, str, TopologyIR]] = []
    ordinal = 0
    # Repeated physical variants keep the same strategic structure while changing
    # fiber lengths and seeds. The first 50 specs are intentionally ordered to
    # cover disjoint, bottleneck, mixed-overlap, and layered cases.
    for variant, scale in enumerate((0.85, 1.0, 1.15)):
        for width in (2, 3, 4, 8, 16, 32):
            specs.append((
                f"disjoint_parallel_{width}_v{variant}",
                "disjoint_parallel",
                _disjoint_parallel(width, seed=base_seed + ordinal, length_scale=scale),
            ))
            ordinal += 1
    for width in (2, 4, 8, 16):
        for variant, scale in enumerate((0.9, 1.05, 1.2)):
            specs.append((
                f"single_bottleneck_{width}_v{variant}",
                "single_bottleneck",
                _single_bottleneck(width, seed=base_seed + ordinal, length_scale=scale),
            ))
            ordinal += 1
    for cuts, width in ((2, 4), (2, 8), (3, 4), (3, 8)):
        for variant, scale in enumerate((0.9, 1.15)):
            specs.append((
                f"multi_bottleneck_{cuts}x{width}_v{variant}",
                "multi_bottleneck",
                _multi_bottleneck(cuts, width, seed=base_seed + ordinal, length_scale=scale),
            ))
            ordinal += 1
    for repeats in (1, 2, 3):
        for variant, scale in enumerate((0.9, 1.15)):
            specs.append((
                f"wheatstone_chain_{repeats}_v{variant}",
                "wheatstone_chain",
                _wheatstone_chain(repeats, seed=base_seed + ordinal, length_scale=scale),
            ))
            ordinal += 1
    for layers, width in ((2, 3), (2, 4), (3, 3), (3, 4)):
        for variant, scale in enumerate((0.9, 1.1)):
            specs.append((
                f"layered_parallel_{layers}x{width}_v{variant}",
                "layered_parallel",
                _layered_parallel(layers, width, seed=base_seed + ordinal, length_scale=scale),
            ))
            ordinal += 1
    for width, scale in ((4, 0.7), (4, 1.3), (8, 0.75), (8, 1.35), (16, 0.8), (16, 1.25)):
        specs.append((
            f"length_variant_disjoint_{width}_{scale:.2f}".replace(".", "p"),
            "length_variant",
            _disjoint_parallel(width, seed=base_seed + ordinal, length_scale=scale),
        ))
        ordinal += 1
    for hops in (6, 7):
        for width in (2, 3):
            specs.append((
                f"deep_parallel_h{hops}_w{width}",
                "deep_parallel",
                _parallel_chains(hops, width, seed=base_seed + ordinal),
            ))
            ordinal += 1
    for hops in (6, 7):
        for width in (4, 8):
            specs.append((
                f"deep_bottleneck_h{hops}_w{width}",
                "deep_bottleneck",
                _deep_bottleneck(hops, width, seed=base_seed + ordinal),
            ))
            ordinal += 1
    return _interleave_by_family(specs)


def _interleave_by_family(
    specs: list[tuple[str, str, TopologyIR]],
) -> list[tuple[str, str, TopologyIR]]:
    buckets: dict[str, list[tuple[str, str, TopologyIR]]] = {}
    for spec in specs:
        buckets.setdefault(spec[1], []).append(spec)
    ordered = []
    families = list(buckets)
    while any(buckets.values()):
        for family in families:
            bucket = buckets[family]
            if bucket:
                ordered.append(bucket.pop(0))
    return ordered


def _assignment_shell(graph_id: str, family: str) -> dict[str, Any]:
    return {
        "assignment_id": "generator_preferred_pair",
        "graph_id": graph_id,
        "assignment_policy": "generator_preferred_pair",
        "family": family,
    }


def _disjoint_parallel(width: int, *, seed: int, length_scale: float = 1.0) -> TopologyIR:
    nodes = ["a", "b"] + [f"r{i}" for i in range(width)]
    edges = []
    for i in range(width):
        length = _length(seed, i, length_scale)
        edges.extend([("a", f"r{i}", length), (f"r{i}", "b", length)])
    return _manual_ir("disjoint_parallel", nodes, edges, seed, {"width": width})


def _parallel_chains(hops: int, width: int, *, seed: int,
                     length_scale: float = 1.0) -> TopologyIR:
    if hops < 2:
        raise ValueError("parallel chain corpus cases need hops >= 2")
    nodes = ["a", "b"]
    edges = []
    ordinal = 0
    for branch in range(width):
        path = ["a"]
        internal = [f"p{branch}_{depth}" for depth in range(1, hops)]
        nodes.extend(internal)
        path.extend(internal)
        path.append("b")
        for u, v in zip(path, path[1:]):
            edges.append((u, v, _length(seed, ordinal, length_scale)))
            ordinal += 1
    return _manual_ir("deep_parallel", nodes, edges, seed, {
        "width": width,
        "route_hops": hops,
    })


def _deep_bottleneck(hops: int, width: int, *, seed: int,
                     length_scale: float = 1.0) -> TopologyIR:
    if hops < 2:
        raise ValueError("deep bottleneck corpus cases need hops >= 2")
    nodes = ["a", "s", "b"]
    edges = [("a", "s", _length(seed, 0, length_scale))]
    ordinal = 1
    for branch in range(width):
        path = ["s"]
        internal = [f"q{branch}_{depth}" for depth in range(1, hops - 1)]
        nodes.extend(internal)
        path.extend(internal)
        path.append("b")
        for u, v in zip(path, path[1:]):
            edges.append((u, v, _length(seed, ordinal, length_scale)))
            ordinal += 1
    return _manual_ir("deep_bottleneck", nodes, edges, seed, {
        "width": width,
        "route_hops": hops,
        "shared_bottleneck_node": "s",
    })


def _single_bottleneck(width: int, *, seed: int, length_scale: float = 1.0) -> TopologyIR:
    nodes = ["a", "s", "b"] + [f"r{i}" for i in range(width)]
    edges = [("a", "s", _length(seed, 0, length_scale))]
    for i in range(width):
        edges.extend([
            ("s", f"r{i}", _length(seed, i + 1, length_scale)),
            (f"r{i}", "b", _length(seed, i + 10, length_scale)),
        ])
    return _manual_ir("single_bottleneck", nodes, edges, seed, {"width": width})


def _multi_bottleneck(cuts: int, width: int, *, seed: int,
                      length_scale: float = 1.0) -> TopologyIR:
    nodes = ["a", "b"] + [f"s{i}" for i in range(cuts)]
    edges = [("a", "s0", _length(seed, 0, length_scale))]
    previous = "s0"
    for cut in range(1, cuts):
        layer = [f"r{cut}_{i}" for i in range(width)]
        nodes.extend(layer)
        for i, node in enumerate(layer):
            edges.append((previous, node, _length(seed, cut * 100 + i, length_scale)))
            edges.append((node, f"s{cut}", _length(seed, cut * 100 + i + 50, length_scale)))
        previous = f"s{cut}"
    if cuts == 1:
        previous = "s0"
    tail = [f"t{i}" for i in range(width)]
    nodes.extend(tail)
    for i, node in enumerate(tail):
        edges.append((previous, node, _length(seed, 700 + i, length_scale)))
        edges.append((node, "b", _length(seed, 750 + i, length_scale)))
    return _manual_ir("multi_bottleneck", nodes, edges, seed, {"cuts": cuts, "width": width})


def _wheatstone_chain(repeats: int, *, seed: int, length_scale: float = 1.0) -> TopologyIR:
    nodes = ["a"]
    edges = []
    left = "a"
    for r in range(repeats):
        upper = f"u{r}"
        lower = f"v{r}"
        right = "b" if r == repeats - 1 else f"c{r}"
        nodes.extend([upper, lower, right])
        edges.extend([
            (left, upper, _length(seed, r * 10, length_scale)),
            (upper, right, _length(seed, r * 10 + 1, length_scale)),
            (left, lower, _length(seed, r * 10 + 2, length_scale)),
            (lower, right, _length(seed, r * 10 + 3, length_scale)),
            (upper, lower, _length(seed, r * 10 + 4, length_scale)),
        ])
        left = right
    return _manual_ir("wheatstone_chain", nodes, edges, seed, {"repeats": repeats})


def _layered_parallel(layers: int, width: int, *, seed: int,
                      length_scale: float = 1.0) -> TopologyIR:
    levels = [["a"]]
    for layer in range(layers):
        levels.append([f"l{layer}_{i}" for i in range(width)])
    levels.append(["b"])
    nodes = [node for level in levels for node in level]
    edges = []
    ordinal = 0
    for left, right in zip(levels, levels[1:]):
        for u in left:
            for v in right:
                edges.append((u, v, _length(seed, ordinal, length_scale)))
                ordinal += 1
    return _manual_ir("layered_parallel", nodes, edges, seed, {"layers": layers, "width": width})


def _manual_ir(family: str, nodes: list[str], edges: list[tuple[str, str, float]],
               seed: int, params: dict[str, Any]) -> TopologyIR:
    unique_nodes = list(dict.fromkeys(nodes))
    node_records = {}
    for node in unique_nodes:
        roles = []
        if node == "a":
            roles.append("alice")
        elif node == "b":
            roles.append("bob")
        else:
            roles.append("swap_candidate")
        node_records[node] = NodeRecord(
            node,
            roles=frozenset(roles),
            coordinates=_coord(node),
        )
    edge_records = [
        EdgeRecord(f"e{i}", u, v, float(length), eve_eligible=True)
        for i, (u, v, length) in enumerate(edges)
    ]
    ir = TopologyIR(
        nodes=node_records,
        edges=edge_records,
        metadata=TopologyMetadata(
            "exp3_sequence_corpus",
            {
                "graph_family": family,
                "preferred_alice": "a",
                "preferred_bob": "b",
                "edge_length_semantics": "meters; km-scale SeQUeNCe repeater corpus",
                "base_edge_length_m": CORPUS_BASE_EDGE_LENGTH_M,
                "edge_length_step_m": CORPUS_EDGE_LENGTH_STEP_M,
                **params,
            },
            seed,
        ),
    )
    ir.validate()
    return ir


def _length(seed: int, ordinal: int, scale: float = 1.0) -> float:
    return float(scale * (
        CORPUS_BASE_EDGE_LENGTH_M
        + CORPUS_EDGE_LENGTH_STEP_M * ((seed + ordinal * 17) % 5)
    ))


def _coord(node: str) -> tuple[float, float]:
    if node == "a":
        return (0.0, 0.0)
    if node == "b":
        return (10.0, 0.0)
    total = sum(ord(ch) for ch in node)
    return (float(1 + total % 8), float((total // 7) % 7 - 3))


def _nx_graph(ir: TopologyIR) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(ir.nodes)
    graph.add_edges_from((edge.u, edge.v) for edge in ir.edges)
    return graph
