"""Pure graph-structural ("shape") features for a (topology, Alice, Bob) triple.

Everything in this module is exact graph theory computed on the undirected
topology graph induced by a ``TopologyIR``: min s-t cuts, articulation
points, bridges, edge-betweenness centrality, algebraic connectivity
(Fiedler value), k-core numbers, and basic size/diameter statistics. None of
it touches SeQUeNCe, the E91/CHSH runtime, Eve's attack model, or Exp3
learning dynamics.

Scientific-scope note: this module makes no claim about physical validity,
security, or learned Alice/Eve strategy quality. It exists purely so that a
downstream decision tree can be fit on graph *shape* (independent of route
identity or node naming) against *separately measured* Exp3 outcomes (e.g.
``matrix_retention`` / ``exploitability`` from
``experiments/exp3_sequence/online.py``). A feature being predictive here
says only "this shape statistic correlates with the observed outcome on this
corpus"; it is not a security or physical claim about the shape itself.

Min-cut implementation choice: this module standardizes on networkx
(``nx.node_connectivity`` / ``nx.edge_connectivity`` / ``nx.minimum_node_cut``
/ ``nx.minimum_edge_cut``) rather than the hand-rolled max-flow min-cut in
``sequence_game/graph_corpus/metrics.py`` (``minimum_st_edge_cut`` /
``minimum_st_internal_node_cut``). Reason: the Exp3-side consumer this module
feeds, ``sequence_game/experiments/exp3_sequence/corpus.py:graph_features``,
already computes ``node_disjoint_paths`` / ``edge_disjoint_paths`` via
``nx.node_connectivity`` / ``nx.edge_connectivity`` for its synthetic corpus.
Standardizing here keeps the whole Exp3 ``GraphCase.features`` vocabulary on
one min-cut implementation instead of introducing a second, independently
tested one just to get the concrete cut *edge/node sets* (needed here for
edge-betweenness-of-cut-edges, which ``graph_corpus/metrics.py`` does not
expose). ``graph_corpus/metrics.py``'s hand-rolled max-flow remains the right
tool for the *real* Topology-Bench-corpus route/hitting-set analysis
(``analyze_route_structure``), which is a different consumer and is left
untouched.

By Menger's theorem, for non-adjacent Alice/Bob the min s-t edge cut size
equals the max number of edge-disjoint Alice-Bob paths, and the min s-t node
cut size equals the max number of internally node-disjoint Alice-Bob paths.
This module therefore reports one integer for each (``edge_cut_size``,
``node_cut_size``) rather than four redundant fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from sequence_game.topology import TopologyIR

# Enumerating every shortest Alice-Bob path is only used to decide whether an
# articulation point / bridge lies "on a shortest path"; cap it so a highly
# symmetric graph with many tied-shortest paths cannot blow up this function.
MAX_SHORTEST_PATHS_PROBED = 200


@dataclass(frozen=True)
class ShapeFeatures:
    """Pure graph-structural features for one (topology, alice, bob) triple.

    All fields are exact graph theory; see the module docstring for the
    explicit non-claim about physical validity or learned strategy quality.
    """

    node_count: int
    edge_count: int
    is_connected: bool
    diameter: Optional[int]  # None if the whole graph is disconnected

    alice_bob_connected: bool
    shortest_hops: Optional[int]  # Alice-Bob distance; None if disconnected

    # Min s-t cuts / Menger disjoint-path counts (see module docstring).
    edge_cut_size: int
    node_cut_size: int

    articulation_point_count: int  # graph-wide
    articulation_points_on_shortest_path: int
    bridge_count: int  # graph-wide
    bridges_on_shortest_path: int

    max_edge_betweenness: float  # graph-wide max edge-betweenness centrality
    max_cut_edge_betweenness: Optional[float]  # None if the cut is empty
    mean_cut_edge_betweenness: Optional[float]  # None if the cut is empty

    algebraic_connectivity: float  # Fiedler value; 0.0 if graph disconnected

    alice_core_number: int
    bob_core_number: int
    max_core_number: int  # graph-wide

    # Node identity for narrative claims (e.g. "node X is the bottleneck").
    # `node_cut_size == 1` only means *a* size-1 cut exists, not that only
    # one node could be named -- two distinct articulation points in series
    # on the only alice-bob path each individually form a size-1 cut, so
    # `has_unique_bottleneck` additionally requires exactly one node to be
    # individually critical (brute-force checked in compute_shape_features,
    # not merely inferred from one nx.minimum_node_cut call). It is
    # therefore NOT fully redundant with `node_cut_size == 1`: on the real
    # 234-graph small_synthetic corpus, 11 graphs have node_cut_size == 1 but
    # only 8 have has_unique_bottleneck == True (3 have two competing
    # critical nodes). Deliberately excluded from FEATURE_NAMES /
    # feature_vector() regardless: `bottleneck_node_id` is a string (not a
    # numeric DT input), and even the corrected `has_unique_bottleneck`
    # would only ever refine what a narrative layer can safely *say* about
    # a node_cut_size==1 graph, not give the DT any information it lacks --
    # the DT already sees node_cut_size directly.
    has_unique_bottleneck: bool
    bottleneck_node_id: Optional[str]  # the single cut node, iff node_cut_size == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "is_connected": self.is_connected,
            "diameter": self.diameter,
            "alice_bob_connected": self.alice_bob_connected,
            "shortest_hops": self.shortest_hops,
            "edge_cut_size": self.edge_cut_size,
            "node_cut_size": self.node_cut_size,
            "articulation_point_count": self.articulation_point_count,
            "articulation_points_on_shortest_path": self.articulation_points_on_shortest_path,
            "bridge_count": self.bridge_count,
            "bridges_on_shortest_path": self.bridges_on_shortest_path,
            "max_edge_betweenness": self.max_edge_betweenness,
            "max_cut_edge_betweenness": self.max_cut_edge_betweenness,
            "mean_cut_edge_betweenness": self.mean_cut_edge_betweenness,
            "algebraic_connectivity": self.algebraic_connectivity,
            "alice_core_number": self.alice_core_number,
            "bob_core_number": self.bob_core_number,
            "max_core_number": self.max_core_number,
            "has_unique_bottleneck": self.has_unique_bottleneck,
            "bottleneck_node_id": self.bottleneck_node_id,
        }

    def feature_vector(self) -> list[float]:
        """Numeric vector in ``FEATURE_NAMES`` order for DT fitting.

        ``None`` fields (undefined because the graph, or the Alice-Bob pair,
        is disconnected) become ``-1.0``, a value outside every other
        field's natural range so a DT can still split on "was this
        undefined" without silently being treated as zero.
        """
        return [_sentinel(getattr(self, name)) if name != "is_connected"
                and name != "alice_bob_connected"
                else float(getattr(self, name))
                for name in FEATURE_NAMES]


FEATURE_NAMES: tuple[str, ...] = (
    "node_count",
    "edge_count",
    "is_connected",
    "diameter",
    "alice_bob_connected",
    "shortest_hops",
    "edge_cut_size",
    "node_cut_size",
    "articulation_point_count",
    "articulation_points_on_shortest_path",
    "bridge_count",
    "bridges_on_shortest_path",
    "max_edge_betweenness",
    "max_cut_edge_betweenness",
    "mean_cut_edge_betweenness",
    "algebraic_connectivity",
    "alice_core_number",
    "bob_core_number",
    "max_core_number",
)


def _sentinel(value: Any) -> float:
    if value is None:
        return -1.0
    return float(value)


def compute_shape_features(ir: TopologyIR, alice: str, bob: str) -> ShapeFeatures:
    """Compute :class:`ShapeFeatures` for one (topology, alice, bob) triple.

    ``alice``/``bob`` must be node ids present in ``ir.nodes``; this function
    does not require they be role-tagged "alice"/"bob" in the IR.
    """
    if alice not in ir.nodes:
        raise ValueError(f"alice node {alice!r} not present in topology")
    if bob not in ir.nodes:
        raise ValueError(f"bob node {bob!r} not present in topology")

    graph = _nx_graph(ir)
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    is_connected = nx.is_connected(graph) if node_count > 0 else False
    diameter = _safe_diameter(graph, is_connected)

    alice_bob_connected = nx.has_path(graph, alice, bob)
    shortest_hops: Optional[int] = None
    shortest_paths: list[list[str]] = []
    if alice_bob_connected:
        shortest_hops = nx.shortest_path_length(graph, alice, bob)
        shortest_paths = _bounded_shortest_paths(graph, alice, bob)

    edge_cut_size = 0
    node_cut_size = 0
    cut_edges: set[frozenset[str]] = set()
    bottleneck_node_id: Optional[str] = None
    if alice_bob_connected and alice != bob:
        edge_cut_size = nx.edge_connectivity(graph, alice, bob)
        node_cut_size = nx.node_connectivity(graph, alice, bob)
        cut_edges = {frozenset(edge) for edge in nx.minimum_edge_cut(graph, alice, bob)}
        if node_cut_size == 1:
            # nx.minimum_node_cut returns ONE minimum cut, not every node
            # whose individual removal disconnects alice/bob. Two distinct
            # articulation points in series on the only alice-bob path each
            # form their own size-1 cut; naming just one of them "the"
            # bottleneck would be arbitrary. Brute-force every non-alice/bob
            # node (cheap: this branch only runs when node_cut_size == 1,
            # graphs here are small) and only name a bottleneck when exactly
            # one node is individually critical.
            critical_nodes = [
                node for node in graph.nodes
                if node not in (alice, bob) and _removal_disconnects(graph, node, alice, bob)
            ]
            if len(critical_nodes) == 1:
                bottleneck_node_id = critical_nodes[0]

    articulation_points = set(nx.articulation_points(graph)) if node_count > 0 else set()
    bridges = {frozenset(edge) for edge in nx.bridges(graph)} if is_connected else set()

    shortest_path_nodes: set[str] = set()
    shortest_path_edges: set[frozenset[str]] = set()
    for path in shortest_paths:
        shortest_path_nodes.update(path[1:-1])  # internal nodes only
        shortest_path_edges.update(
            frozenset((u, v)) for u, v in zip(path, path[1:])
        )

    articulation_points_on_shortest_path = len(
        articulation_points & shortest_path_nodes
    )
    bridges_on_shortest_path = len(bridges & shortest_path_edges)

    betweenness = nx.edge_betweenness_centrality(graph) if edge_count > 0 else {}
    betweenness_by_pair = {
        frozenset(edge): value for edge, value in betweenness.items()
    }
    max_edge_betweenness = max(betweenness.values()) if betweenness else 0.0
    cut_edge_betweenness = [
        betweenness_by_pair.get(edge, 0.0) for edge in cut_edges
    ]
    max_cut_edge_betweenness = max(cut_edge_betweenness) if cut_edge_betweenness else None
    mean_cut_edge_betweenness = (
        sum(cut_edge_betweenness) / len(cut_edge_betweenness)
        if cut_edge_betweenness else None
    )

    algebraic_connectivity = (
        nx.algebraic_connectivity(graph) if node_count > 1 else 0.0
    )

    core_numbers = nx.core_number(graph) if node_count > 0 else {}
    alice_core_number = int(core_numbers.get(alice, 0))
    bob_core_number = int(core_numbers.get(bob, 0))
    max_core_number = max(core_numbers.values()) if core_numbers else 0

    return ShapeFeatures(
        node_count=node_count,
        edge_count=edge_count,
        is_connected=is_connected,
        diameter=diameter,
        alice_bob_connected=alice_bob_connected,
        shortest_hops=shortest_hops,
        edge_cut_size=int(edge_cut_size),
        node_cut_size=int(node_cut_size),
        articulation_point_count=len(articulation_points),
        articulation_points_on_shortest_path=articulation_points_on_shortest_path,
        bridge_count=len(bridges),
        bridges_on_shortest_path=bridges_on_shortest_path,
        max_edge_betweenness=float(max_edge_betweenness),
        max_cut_edge_betweenness=(
            float(max_cut_edge_betweenness) if max_cut_edge_betweenness is not None else None
        ),
        mean_cut_edge_betweenness=(
            float(mean_cut_edge_betweenness) if mean_cut_edge_betweenness is not None else None
        ),
        algebraic_connectivity=float(algebraic_connectivity),
        alice_core_number=alice_core_number,
        bob_core_number=bob_core_number,
        max_core_number=int(max_core_number),
        has_unique_bottleneck=bottleneck_node_id is not None,
        bottleneck_node_id=bottleneck_node_id,
    )


def _nx_graph(ir: TopologyIR) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(ir.nodes)
    graph.add_edges_from((edge.u, edge.v) for edge in ir.edges)
    return graph


def _removal_disconnects(graph: nx.Graph, node: str, alice: str, bob: str) -> bool:
    reduced = graph.copy()
    reduced.remove_node(node)
    return not nx.has_path(reduced, alice, bob)


def _safe_diameter(graph: nx.Graph, is_connected: bool) -> Optional[int]:
    if not is_connected or graph.number_of_nodes() == 0:
        return None
    return nx.diameter(graph)


def _bounded_shortest_paths(graph: nx.Graph, alice: str, bob: str) -> list[list[str]]:
    paths: list[list[str]] = []
    for path in nx.all_shortest_paths(graph, alice, bob):
        paths.append(path)
        if len(paths) >= MAX_SHORTEST_PATHS_PROBED:
            break
    return paths
