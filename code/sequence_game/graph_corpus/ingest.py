"""External graph ingestion for graph-corpus construction."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from sequence_game.claims import CONTROL_GAME, IMPLEMENTED_CONTROL_ONLY
from sequence_game.topology import EdgeRecord, NodeRecord, TopologyIR, TopologyMetadata

SOURCE_TOPOLOGY_IR = "ingested_topology_ir"
SOURCE_TOPOLOGY_ZOO = "topology_zoo"
SOURCE_SNDLIB = "sndlib"
SOURCE_SNAP_AS = "snap_as"
SOURCE_CAIDA = "caida"

PHYSICAL_TELECOM_TOPOLOGY = "telecom_topology"
PHYSICAL_STRUCTURAL_STRESS_ONLY = "structural_stress_only"
PHYSICAL_TOPOLOGY_CANDIDATE = "physical_topology_candidate"

EDGE_LENGTH_PHYSICAL = "physical_length"
EDGE_LENGTH_SYNTHETIC = "synthetic_length"
EDGE_LENGTH_HOP_ONLY = "hop_only"

LENGTH_KEYS = ("length_m", "length", "Length", "distance", "Distance", "weight")
LAT_KEYS = ("lat", "latitude", "Latitude", "y")
LON_KEYS = ("lon", "lng", "longitude", "Longitude", "x")


def ingest_topology_ir_directory(path: Path) -> list[TopologyIR]:
    graphs = []
    for file_path in sorted(path.glob("*.json")):
        graph = TopologyIR.from_json(file_path.read_text(encoding="utf-8"))
        graph.metadata = _metadata(
            graph.metadata.params.get("graph_family", file_path.stem),
            source=SOURCE_TOPOLOGY_IR,
            physical_role=str(graph.metadata.params.get("physical_role", PHYSICAL_TOPOLOGY_CANDIDATE)),
            edge_length_semantics=str(
                graph.metadata.params.get("edge_length_semantics", EDGE_LENGTH_SYNTHETIC)
            ),
            seed=graph.metadata.seed,
            extra={
                **graph.metadata.params,
                "ingested_from": str(file_path),
            },
        )
        graph.validate()
        graphs.append(graph)
    return graphs


def ingest_graphml_gml_directory(
    path: Path,
    *,
    source: str = SOURCE_TOPOLOGY_ZOO,
    physical_role: str = PHYSICAL_TELECOM_TOPOLOGY,
    synthetic_edge_length_m: float = 10.0,
    largest_connected_component: bool = True,
    compute_geo_lengths: bool = True,
) -> list[TopologyIR]:
    nx = _networkx()
    graphs = []
    for file_path in sorted(_graphml_gml_files(path)):
        if file_path.suffix.lower() == ".graphml":
            nx_graph = nx.read_graphml(file_path)
        else:
            nx_graph = nx.read_gml(file_path)
        if largest_connected_component:
            nx_graph = _largest_nx_component(nx_graph, nx)
        graph = _nx_to_topology_ir(
            nx_graph,
            graph_family=file_path.stem,
            source=source,
            physical_role=physical_role,
            synthetic_edge_length_m=synthetic_edge_length_m,
            compute_geo_lengths=compute_geo_lengths,
            ingested_from=file_path,
        )
        graphs.append(graph)
    return graphs


def ingest_snap_edge_list_directory(
    path: Path,
    *,
    source: str = SOURCE_SNAP_AS,
    physical_role: str = PHYSICAL_STRUCTURAL_STRESS_ONLY,
    synthetic_edge_length_m: float = 10.0,
    edge_length_semantics: str = EDGE_LENGTH_SYNTHETIC,
    largest_connected_component: bool = True,
) -> list[TopologyIR]:
    graphs = []
    for file_path in sorted(_edge_list_files(path)):
        graph = _read_edge_list(
            file_path,
            source=source,
            physical_role=physical_role,
            synthetic_edge_length_m=synthetic_edge_length_m,
            edge_length_semantics=edge_length_semantics,
            largest_connected_component=largest_connected_component,
        )
        graphs.append(graph)
    return graphs


def _nx_to_topology_ir(
    nx_graph: Any,
    *,
    graph_family: str,
    source: str,
    physical_role: str,
    synthetic_edge_length_m: float,
    compute_geo_lengths: bool,
    ingested_from: Path,
) -> TopologyIR:
    node_ids = [str(node) for node in nx_graph.nodes]
    nodes = {node_id: NodeRecord(node_id) for node_id in node_ids}
    edge_records = []
    length_semantics = EDGE_LENGTH_PHYSICAL
    seen_pairs = set()

    for index, edge_data in enumerate(_nx_edges(nx_graph)):
        u, v, edge_id, attrs = edge_data
        u = str(u)
        v = str(v)
        if u == v:
            continue
        pair = frozenset((u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        length = _edge_length_from_attrs(attrs)
        if length is None and compute_geo_lengths:
            length = _geo_edge_length(nx_graph.nodes[u], nx_graph.nodes[v])
        if length is None:
            length = synthetic_edge_length_m
            length_semantics = EDGE_LENGTH_SYNTHETIC
        edge_records.append(
            EdgeRecord(str(edge_id or f"e{index}"), u, v, float(length), eve_eligible=True)
        )

    graph = TopologyIR(
        nodes=nodes,
        edges=edge_records,
        metadata=_metadata(
            graph_family,
            source=source,
            physical_role=physical_role,
            edge_length_semantics=length_semantics,
            seed=None,
            extra={"ingested_from": str(ingested_from)},
        ),
    )
    graph.validate()
    return graph


def _read_edge_list(
    file_path: Path,
    *,
    source: str,
    physical_role: str,
    synthetic_edge_length_m: float,
    edge_length_semantics: str,
    largest_connected_component: bool,
) -> TopologyIR:
    edges = []
    nodes = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        if u == v:
            continue
        nodes.update((u, v))
        edges.append((u, v))
    if largest_connected_component:
        nodes, edges = _largest_component_from_edges(nodes, edges)

    edge_records = [
        EdgeRecord(f"e{index}", u, v, synthetic_edge_length_m, eve_eligible=True)
        for index, (u, v) in enumerate(_unique_edges(edges))
    ]
    graph = TopologyIR(
        nodes={node: NodeRecord(node) for node in sorted(nodes)},
        edges=edge_records,
        metadata=_metadata(
            file_path.stem,
            source=source,
            physical_role=physical_role,
            edge_length_semantics=edge_length_semantics,
            seed=None,
            extra={"ingested_from": str(file_path)},
        ),
    )
    graph.validate()
    return graph


def _networkx() -> Any:
    try:
        import networkx as nx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - covered when dependency absent
        raise RuntimeError(
            "GraphML/GML ingestion requires networkx. It is not currently importable; "
            "install or enable networkx before using graphml_gml ingestion."
        ) from exc
    return nx


def _graphml_gml_files(path: Path) -> Iterator[Path]:
    for suffix in ("*.graphml", "*.gml"):
        yield from path.glob(suffix)


def _edge_list_files(path: Path) -> Iterator[Path]:
    for suffix in ("*.txt", "*.edges", "*.edgelist", "*.csv"):
        yield from path.glob(suffix)


def _nx_edges(nx_graph: Any) -> Iterable[tuple[Any, Any, Optional[str], dict[str, Any]]]:
    if nx_graph.is_multigraph():
        for u, v, key, attrs in nx_graph.edges(keys=True, data=True):
            yield u, v, str(attrs.get("id", key)), dict(attrs)
    else:
        for index, (u, v, attrs) in enumerate(nx_graph.edges(data=True)):
            yield u, v, str(attrs.get("id", f"e{index}")), dict(attrs)


def _largest_nx_component(nx_graph: Any, nx: Any) -> Any:
    undirected = nx_graph.to_undirected()
    components = list(nx.connected_components(undirected))
    if not components:
        return nx_graph
    largest = max(components, key=len)
    return nx_graph.subgraph(largest).copy()


def _edge_length_from_attrs(attrs: dict[str, Any]) -> Optional[float]:
    for key in LENGTH_KEYS:
        if key not in attrs:
            continue
        try:
            value = float(attrs[key])
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _geo_edge_length(left_attrs: dict[str, Any], right_attrs: dict[str, Any]) -> Optional[float]:
    left_lat = _first_float(left_attrs, LAT_KEYS)
    left_lon = _first_float(left_attrs, LON_KEYS)
    right_lat = _first_float(right_attrs, LAT_KEYS)
    right_lon = _first_float(right_attrs, LON_KEYS)
    if None in (left_lat, left_lon, right_lat, right_lon):
        return None
    return _haversine_m(left_lat, left_lon, right_lat, right_lon)


def _first_float(attrs: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key not in attrs:
            continue
        try:
            return float(attrs[key])
        except (TypeError, ValueError):
            continue
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _largest_component_from_edges(
    nodes: set[str],
    edges: list[tuple[str, str]],
) -> tuple[set[str], list[tuple[str, str]]]:
    adjacency = {node: set() for node in nodes}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    unseen = set(nodes)
    components = []
    while unseen:
        start = min(unseen)
        component = {start}
        unseen.remove(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    if not components:
        return nodes, edges
    largest = max(components, key=len)
    filtered_edges = [(u, v) for u, v in edges if u in largest and v in largest]
    return largest, filtered_edges


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


def _metadata(
    graph_family: str,
    *,
    source: str,
    physical_role: str,
    edge_length_semantics: str,
    seed: Optional[int],
    extra: Optional[dict[str, Any]] = None,
) -> TopologyMetadata:
    return TopologyMetadata(
        "graph_corpus_ingest",
        {
            "graph_family": graph_family,
            "source": source,
            "physical_role": physical_role,
            "edge_length_semantics": edge_length_semantics,
            "scope_label": CONTROL_GAME,
            "claim_status": IMPLEMENTED_CONTROL_ONLY,
            **(extra or {}),
        },
        seed,
    )
