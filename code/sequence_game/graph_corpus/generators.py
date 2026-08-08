"""Synthetic graph generators for graph-corpus construction."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import numpy as np

from sequence_game.claims import CONTROL_GAME, IMPLEMENTED_CONTROL_ONLY
from sequence_game.topology import EdgeRecord, NodeRecord, TopologyIR, TopologyMetadata

SOURCE_SYNTHETIC = "synthetic"
PHYSICAL_SYNTHETIC_CONTROL = "synthetic_control"
PHYSICAL_STRUCTURAL_STRESS_ONLY = "structural_stress_only"
EDGE_LENGTH_SYNTHETIC = "synthetic_length"

DEFAULT_FIXED_EDGE_LENGTH_M = 10.0

DEFAULT_GRAPH_SPECS: list[dict[str, Any]] = [
    {"family": "braess", "count": 1},
    {"family": "bottleneck", "count": 1},
    {"family": "ladder", "count": 1},
    {"family": "diamond", "count": 1},
    {
        "family": "layered_parallel",
        "count": 25,
        "layers": 3,
        "width": 3,
        "same_layer_chord_probability": 0.0,
    },
    {
        "family": "layered_parallel",
        "count": 25,
        "layers": 3,
        "width": 3,
        "same_layer_chord_probability": 0.25,
    },
    {
        "family": "layered_parallel",
        "count": 25,
        "layers": 4,
        "width": 3,
        "same_layer_chord_probability": 0.25,
    },
    {
        "family": "layered_parallel",
        "count": 25,
        "layers": 4,
        "width": 4,
        "same_layer_chord_probability": 0.15,
    },
    {
        "family": "bounded_er",
        "count": 50,
        "num_nodes": 16,
        "edge_probability": 0.18,
        "max_degree": 4,
    },
    {
        "family": "bounded_er",
        "count": 50,
        "num_nodes": 24,
        "edge_probability": 0.12,
        "max_degree": 4,
    },
    {
        "family": "bounded_er",
        "count": 50,
        "num_nodes": 32,
        "edge_probability": 0.09,
        "max_degree": 5,
    },
]


def generated_graph_count(graph_specs: Iterable[dict[str, Any]]) -> int:
    return sum(int(spec.get("count", 1)) for spec in graph_specs)


def generate_synthetic_graphs(
    graph_specs: Iterable[dict[str, Any]],
    *,
    base_seed: int,
    fixed_edge_length_m: float = DEFAULT_FIXED_EDGE_LENGTH_M,
) -> Iterator[TopologyIR]:
    ordinal = 0
    for spec in graph_specs:
        family = str(spec["family"])
        for _ in range(int(spec.get("count", 1))):
            seed = base_seed + ordinal
            ordinal += 1
            yield generate_synthetic_graph(
                family,
                seed=seed,
                fixed_edge_length_m=fixed_edge_length_m,
                **{key: value for key, value in spec.items() if key not in {"family", "count"}},
            )


def generate_synthetic_graph(
    family: str,
    *,
    seed: int,
    fixed_edge_length_m: float = DEFAULT_FIXED_EDGE_LENGTH_M,
    **params: Any,
) -> TopologyIR:
    if family == "braess":
        return _braess_graph(seed=seed, fixed_edge_length_m=fixed_edge_length_m)
    if family == "bottleneck":
        return _bottleneck_graph(seed=seed, fixed_edge_length_m=fixed_edge_length_m)
    if family == "ladder":
        return _ladder_graph(seed=seed, fixed_edge_length_m=fixed_edge_length_m)
    if family == "diamond":
        return _diamond_graph(seed=seed, fixed_edge_length_m=fixed_edge_length_m)
    if family == "layered_parallel":
        return _layered_parallel_graph(
            seed=seed,
            fixed_edge_length_m=fixed_edge_length_m,
            layers=int(params.get("layers", 3)),
            width=int(params.get("width", 3)),
            same_layer_chord_probability=float(params.get("same_layer_chord_probability", 0.0)),
        )
    if family == "bounded_er":
        return _bounded_er_graph(
            seed=seed,
            fixed_edge_length_m=fixed_edge_length_m,
            num_nodes=int(params.get("num_nodes", 12)),
            edge_probability=float(params.get("edge_probability", 0.2)),
            max_degree=int(params.get("max_degree", 4)),
        )
    raise ValueError(f"unknown graph family {family!r}")


def _braess_graph(*, seed: int, fixed_edge_length_m: float) -> TopologyIR:
    return _manual_graph(
        family="braess",
        nodes=("n0", "n1", "n2", "n3"),
        edges=(
            ("n0", "n1"),
            ("n1", "n3"),
            ("n0", "n2"),
            ("n2", "n3"),
            ("n1", "n2"),
        ),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        params={"preferred_alice": "n0", "preferred_bob": "n3"},
    )


def _bottleneck_graph(*, seed: int, fixed_edge_length_m: float) -> TopologyIR:
    return _manual_graph(
        family="bottleneck",
        nodes=("n0", "n1", "n2", "n3", "n4"),
        edges=(
            ("n0", "n1"),
            ("n0", "n2"),
            ("n1", "n3"),
            ("n2", "n3"),
            ("n3", "n4"),
        ),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        params={"preferred_alice": "n0", "preferred_bob": "n4"},
    )


def _ladder_graph(*, seed: int, fixed_edge_length_m: float) -> TopologyIR:
    return _manual_graph(
        family="ladder",
        nodes=("n0", "n1", "n2", "n3", "n4", "n5"),
        edges=(
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n5"),
            ("n0", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n1", "n3"),
            ("n2", "n4"),
        ),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        params={"preferred_alice": "n0", "preferred_bob": "n5"},
    )


def _diamond_graph(*, seed: int, fixed_edge_length_m: float) -> TopologyIR:
    return _manual_graph(
        family="diamond",
        nodes=("n0", "n1", "n2", "n3"),
        edges=(
            ("n0", "n1"),
            ("n1", "n3"),
            ("n0", "n2"),
            ("n2", "n3"),
        ),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        params={"preferred_alice": "n0", "preferred_bob": "n3"},
    )


def _layered_parallel_graph(
    *,
    seed: int,
    fixed_edge_length_m: float,
    layers: int,
    width: int,
    same_layer_chord_probability: float,
) -> TopologyIR:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if width < 2:
        raise ValueError("width must be >= 2")
    if not 0.0 <= same_layer_chord_probability <= 1.0:
        raise ValueError("same_layer_chord_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    level_nodes = [["n0"]]
    next_id = 1
    for _ in range(layers):
        level = [f"n{next_id + offset}" for offset in range(width)]
        next_id += width
        level_nodes.append(level)
    sink = f"n{next_id}"
    level_nodes.append([sink])

    edges: list[tuple[str, str]] = []
    for left, right in zip(level_nodes, level_nodes[1:]):
        for u in left:
            for v in right:
                edges.append((u, v))
    for level in level_nodes[1:-1]:
        for u, v in itertools.combinations(level, 2):
            if rng.random() < same_layer_chord_probability:
                edges.append((u, v))
    nodes = tuple(node for level in level_nodes for node in level)
    return _manual_graph(
        family="layered_parallel",
        nodes=nodes,
        edges=tuple(edges),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        params={
            "layers": layers,
            "width": width,
            "same_layer_chord_probability": same_layer_chord_probability,
            "preferred_alice": "n0",
            "preferred_bob": sink,
        },
    )


def _bounded_er_graph(
    *,
    seed: int,
    fixed_edge_length_m: float,
    num_nodes: int,
    edge_probability: float,
    max_degree: int,
) -> TopologyIR:
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    if max_degree < 2:
        raise ValueError("max_degree must be >= 2 for connected bounded ER graphs")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    nodes = [f"n{i}" for i in range(num_nodes)]
    edges: set[frozenset[str]] = set()
    degree = {node: 0 for node in nodes}

    connected = [nodes[0]]
    for node in nodes[1:]:
        available = [existing for existing in connected if degree[existing] < max_degree]
        if not available:
            raise ValueError("max_degree too small for connected backbone construction")
        parent = str(available[int(rng.integers(len(available)))])
        edges.add(frozenset((node, parent)))
        degree[node] += 1
        degree[parent] += 1
        connected.append(node)

    pairs = [
        (u, v)
        for u, v in itertools.combinations(nodes, 2)
        if frozenset((u, v)) not in edges
    ]
    rng.shuffle(pairs)
    for u, v in pairs:
        if degree[u] >= max_degree or degree[v] >= max_degree:
            continue
        if rng.random() < edge_probability:
            edges.add(frozenset((u, v)))
            degree[u] += 1
            degree[v] += 1

    return _manual_graph(
        family="bounded_er",
        nodes=tuple(nodes),
        edges=tuple(tuple(sorted(edge)) for edge in sorted(edges, key=lambda pair: tuple(sorted(pair)))),
        seed=seed,
        fixed_edge_length_m=fixed_edge_length_m,
        physical_role=PHYSICAL_STRUCTURAL_STRESS_ONLY,
        params={
            "num_nodes": num_nodes,
            "edge_probability": edge_probability,
            "max_degree": max_degree,
        },
    )


def _manual_graph(
    *,
    family: str,
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str]],
    seed: int,
    fixed_edge_length_m: float,
    physical_role: str = PHYSICAL_SYNTHETIC_CONTROL,
    params: Optional[dict[str, Any]] = None,
) -> TopologyIR:
    node_ids = tuple(nodes)
    edge_records = [
        EdgeRecord(f"e{index}", u, v, fixed_edge_length_m, eve_eligible=True)
        for index, (u, v) in enumerate(_unique_edges(edges))
    ]
    graph = TopologyIR(
        nodes={node: NodeRecord(node) for node in node_ids},
        edges=edge_records,
        metadata=TopologyMetadata(
            "graph_corpus_generator",
            {
                "graph_family": family,
                "source": SOURCE_SYNTHETIC,
                "physical_role": physical_role,
                "edge_length_semantics": EDGE_LENGTH_SYNTHETIC,
                "scope_label": CONTROL_GAME,
                "claim_status": IMPLEMENTED_CONTROL_ONLY,
                **(params or {}),
            },
            seed,
        ),
    )
    graph.validate()
    return graph


def _unique_edges(edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    unique = []
    for u, v in edges:
        key = frozenset((u, v))
        if key in seen:
            continue
        seen.add(key)
        unique.append((u, v))
    return unique
