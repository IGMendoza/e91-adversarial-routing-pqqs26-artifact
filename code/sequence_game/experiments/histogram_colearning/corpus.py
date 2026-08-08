"""Load one normalized Topology Bench assignment as a histogram game case."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sequence_game.corpus.e91_runtime_game import _normalize_routes, _runtime_actions
from sequence_game.corpus.sqlite_store import SQLiteGraphCorpusSource
from sequence_game.routing.route import enumerate_simple_paths, make_route
from sequence_game.topology import TopologyIR

ROUTE_UNIVERSE_ALL_SIMPLE_PATHS = "all_simple_paths"
ROUTE_UNIVERSE_STORED_ASSIGNMENT = "stored_assignment"
ROUTE_UNIVERSE_MODES = frozenset({
    ROUTE_UNIVERSE_ALL_SIMPLE_PATHS,
    ROUTE_UNIVERSE_STORED_ASSIGNMENT,
})


@dataclass(frozen=True)
class HistogramGameCase:
    graph_id: str
    assignment_id: str
    family: str
    alice: str
    bob: str
    ir_dict: dict[str, Any]
    routes: tuple[dict[str, Any], ...]
    assignment: dict[str, Any]
    corpus_id: str
    corpus_db_path: str
    route_universe: dict[str, Any] = field(default_factory=dict)

    def actions(self, game_mode: str) -> tuple[dict[str, Any], ...]:
        ir = TopologyIR.from_dict(self.ir_dict)
        return tuple(_runtime_actions(ir, list(self.routes), game_mode))

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "family": self.family,
            "alice": self.alice,
            "bob": self.bob,
            "route_count": len(self.routes),
            "route_ids": [str(route["route_id"]) for route in self.routes],
            "route_hops": [int(route["hop_count"]) for route in self.routes],
            "route_lengths_m": [
                float(route["total_length_m"]) for route in self.routes
            ],
            "corpus_id": self.corpus_id,
            "corpus_db_path": self.corpus_db_path,
            "route_universe": dict(self.route_universe),
        }


def load_topology_bench_case(
    db_path: str | Path,
    *,
    corpus_id: str | None = None,
    graph_id: str | None = None,
    assignment_id: str | None = None,
    assignment_policy: str | None = None,
    assignment_index: int = 0,
    route_universe_mode: str = ROUTE_UNIVERSE_ALL_SIMPLE_PATHS,
) -> HistogramGameCase:
    """Load a corpus assignment with an explicit cycle-free route universe.

    The production default enumerates every simple Alice--Bob path.  No
    service rate, length ranking, predicted physics quality, or hop count is
    used to admit or reject a route.  ``stored_assignment`` remains available
    solely to reproduce historical artifacts.

    """

    if route_universe_mode not in ROUTE_UNIVERSE_MODES:
        raise ValueError(
            f"unsupported route_universe_mode {route_universe_mode!r}"
        )
    source = SQLiteGraphCorpusSource.open(db_path, corpus_id)
    manifest = source.load_manifest()
    candidates = [
        row for row in manifest.get("assignments", [])
        if isinstance(row, dict)
        and (graph_id is None or str(row.get("graph_id")) == str(graph_id))
        and (
            assignment_id is None
            or str(row.get("assignment_id")) == str(assignment_id)
        )
        and (
            assignment_policy is None
            or str(row.get("assignment_policy")) == str(assignment_policy)
        )
    ]
    if not candidates:
        raise ValueError(
            "no Topology Bench assignment matched "
            f"graph_id={graph_id!r}, assignment_id={assignment_id!r}, "
            f"assignment_policy={assignment_policy!r}"
        )
    if assignment_index < 0 or assignment_index >= len(candidates):
        raise IndexError(
            f"assignment_index {assignment_index} outside 0..{len(candidates) - 1}"
        )
    selected = candidates[assignment_index]
    assignment = source.load_json(selected["assignment_path"])
    selected_graph_id = str(selected["graph_id"])
    graph = next(
        (
            row for row in manifest.get("graphs", [])
            if str(row.get("graph_id")) == selected_graph_id
        ),
        None,
    )
    if graph is None:
        raise ValueError(f"manifest has no graph record for {selected_graph_id!r}")
    ir_dict = source.load_json(graph["topology_path"])
    ir = TopologyIR.from_dict(ir_dict)
    stored_route_rows = list(
        (assignment.get("route_structure") or {}).get("candidate_routes") or []
    )
    if route_universe_mode == ROUTE_UNIVERSE_ALL_SIMPLE_PATHS:
        route_rows = enumerate_cycle_free_route_rows(
            ir,
            str(assignment["alice"]),
            str(assignment["bob"]),
        )
        enumeration_source = "active_normalized_topology_ir"
    else:
        route_rows = stored_route_rows
        enumeration_source = "historical_corpus_assignment"
    routes = _normalize_routes(
        ir,
        str(assignment["alice"]),
        str(assignment["bob"]),
        route_rows,
    )
    _validate_cycle_free_routes(
        routes,
        alice=str(assignment["alice"]),
        bob=str(assignment["bob"]),
    )
    # Arm order is deliberately independent of predicted route quality.
    routes.sort(key=lambda row: tuple(str(node) for node in row["path"]))
    if not routes:
        raise ValueError(f"assignment {selected['assignment_id']!r} has no routes")
    return HistogramGameCase(
        graph_id=selected_graph_id,
        assignment_id=str(selected["assignment_id"]),
        family=str(graph.get("family") or graph.get("source") or "unknown"),
        alice=str(assignment["alice"]),
        bob=str(assignment["bob"]),
        ir_dict=ir_dict,
        routes=tuple(routes),
        assignment=dict(assignment),
        corpus_id=source.corpus_id,
        corpus_db_path=str(source.db_path),
        route_universe={
            "mode": route_universe_mode,
            "enumeration_source": enumeration_source,
            "path_constraint": "simple_cycle_free",
            "quality_based_route_pruning": False,
            "enumeration_complete": bool(
                route_universe_mode == ROUTE_UNIVERSE_ALL_SIMPLE_PATHS
            ),
            "stored_assignment_route_count": len(stored_route_rows),
            "enumerated_route_count": len(routes),
            "returned_route_count": len(routes),
            "interpretation": (
                "all simple cycle-free Alice--Bob paths"
                if route_universe_mode == ROUTE_UNIVERSE_ALL_SIMPLE_PATHS
                else "explicit incomplete route subset"
            ),
        },
    )


def build_topology_bench_case_for_endpoints(
    *,
    ir_dict: dict[str, Any],
    graph_id: str,
    family: str,
    alice: str,
    bob: str,
    corpus_id: str,
    corpus_db_path: str,
    assignment_id: str,
    assignment: dict[str, Any] | None = None,
) -> HistogramGameCase:
    """Build a complete cycle-free case for an explicitly declared pair.

    Corpus-scale experiments sometimes need a graph-level endpoint rule rather
    than one of Topology Bench's pre-generated assignment policies.  This
    constructor preserves the same route normalization and validation contract
    as :func:`load_topology_bench_case`; it enumerates every simple Alice--Bob
    path and applies no hop, route-count, or predicted-quality filter.
    """

    ir = TopologyIR.from_dict(ir_dict)
    if alice == bob:
        raise ValueError("Alice and Bob must be distinct endpoints")
    for endpoint in (alice, bob):
        if endpoint not in ir.nodes:
            raise ValueError(f"unknown endpoint {endpoint!r}")
    route_rows = enumerate_cycle_free_route_rows(
        ir,
        alice,
        bob,
    )
    routes = _normalize_routes(ir, alice, bob, route_rows)
    _validate_cycle_free_routes(routes, alice=alice, bob=bob)
    routes.sort(key=lambda row: tuple(str(node) for node in row["path"]))
    if not routes:
        raise ValueError(f"no route connects {alice!r} to {bob!r}")
    assignment_payload = dict(assignment or {})
    assignment_payload.update({
        "assignment_id": assignment_id,
        "graph_id": graph_id,
        "alice": alice,
        "bob": bob,
    })
    return HistogramGameCase(
        graph_id=str(graph_id),
        assignment_id=str(assignment_id),
        family=str(family),
        alice=str(alice),
        bob=str(bob),
        ir_dict=dict(ir_dict),
        routes=tuple(routes),
        assignment=assignment_payload,
        corpus_id=str(corpus_id),
        corpus_db_path=str(corpus_db_path),
        route_universe={
            "mode": ROUTE_UNIVERSE_ALL_SIMPLE_PATHS,
            "enumeration_source": "active_normalized_topology_ir",
            "path_constraint": "simple_cycle_free",
            "quality_based_route_pruning": False,
            "enumeration_complete": True,
            "enumerated_route_count": len(routes),
            "returned_route_count": len(routes),
            "interpretation": "all simple cycle-free Alice--Bob paths",
        },
    )


def enumerate_cycle_free_route_rows(
    ir: TopologyIR,
    alice: str,
    bob: str,
) -> list[dict[str, Any]]:
    """Enumerate every simple path without physics-quality pruning."""
    rows: list[dict[str, Any]] = []
    for path in enumerate_simple_paths(ir, alice, bob):
        route = make_route(ir, path)
        rows.append({
            **route.to_dict(),
            "internal_nodes": list(route.path[1:-1]),
        })
    rows.sort(key=lambda row: tuple(str(node) for node in row["path"]))
    return rows


def _validate_cycle_free_routes(
    routes: list[dict[str, Any]],
    *,
    alice: str,
    bob: str,
) -> None:
    for route in routes:
        path = tuple(str(node) for node in route.get("path") or ())
        route_id = str(route.get("route_id") or "<unknown>")
        if not path or path[0] != alice or path[-1] != bob:
            raise ValueError(
                f"route {route_id} does not connect declared Alice to Bob"
            )
        if len(path) != len(set(path)):
            raise ValueError(f"route {route_id} contains a cycle: {path}")
