"""Per-graph corpus runner and structural feature extraction.

Converts TopologyIR dicts from the corpus into demo_runner topology format,
runs the configured game suite, then computes separate structural emissions for
cross-graph pooled DTs. Attack-surface combos use the graph-route SeQUeNCe E91
runtime backend in ``corpus.e91_runtime_game``. Legacy binary-collision combos
remain dispatchable only when explicitly supplied.

* Eve attack-choice rows from learned Eve Q tables.
* Alice route-choice rows from learned Alice route selections.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from sequence_game.corpus.e91_runtime_game import (  # noqa: E402
    e91_runtime_action_count,
    run_e91_graph_runtime_game,
)


# ── Topology conversion ───────────────────────────────────────────────────────

def ir_to_demo_topology(ir_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert TopologyIR.to_dict() format to the dict expected by demo_runner."""
    raw_nodes = ir_dict.get("nodes", {})
    demo_nodes = []
    for node_id, rec in (raw_nodes.items() if isinstance(raw_nodes, dict)
                          else {n["node_id"]: n for n in raw_nodes}.items()):
        coords = rec.get("coordinates") or [0.0, 0.0]
        demo_nodes.append({"id": node_id, "x": float(coords[0]), "y": float(coords[1])})

    raw_edges = ir_dict.get("edges", [])
    demo_edges = []
    for edge in raw_edges:
        demo_edges.append({
            "u": edge["u"],
            "v": edge["v"],
            "length": float(edge.get("length_m", edge.get("length", 1.0))),
        })

    return {"nodes": demo_nodes, "edges": demo_edges, "warnings": []}


# ── Action parsing ────────────────────────────────────────────────────────────

def parse_action_parts(action_id: str) -> tuple[str, str]:
    """Return (kind, target) from an action id string.

    Examples:
      "node_collision:n3"  → ("node", "n3")
      "edge_collision:n0--n1" → ("edge", "n0--n1")
      "no_attack"          → ("none", "")
    """
    if action_id in {"no_attack", "none"}:
        return ("none", "")
    if ":" in action_id:
        kind_str, target = action_id.split(":", 1)
        node_actions = {
            "node_collision",
            "node_dos",
            "swap_denial",
            "memory_degradation",
            "repeater_memory_measure_Z",
            "repeater_memory_measure_X",
        }
        edge_actions = {
            "edge_collision",
            "edge_dos",
            "edge_information_probe",
            "added_loss",
            "intercept_resend",
        }
        if kind_str in node_actions or "node" in kind_str:
            kind = "node"
        elif kind_str in edge_actions or "edge" in kind_str:
            kind = "edge"
        else:
            kind = kind_str
        return (kind, target)
    return ("none", "")


def parse_attack_type(action_id: str) -> str:
    if action_id in {"no_attack", "none"}:
        return "no_attack"
    if ":" in action_id:
        return action_id.split(":", 1)[0]
    return "unknown"


# ── Structural feature extraction ─────────────────────────────────────────────

STRUCTURAL_FEATURE_NAMES = [
    "is_node",
    "is_edge",
    "in_node_hs",       # target in any minimum node hitting set
    "in_edge_hs",       # target in any minimum edge hitting set
    "route_coverage",   # fraction of game routes containing this target
    "graph_N",          # node hitting set size
    "graph_E",          # edge hitting set size
    "graph_C_log2",     # log2(actual runner route-action count)
    "graph_S",          # shortest hops
    "graph_L",          # longest hops
    "graph_L_over_S",   # longest/shortest ratio
    "is_added_loss_edge",
    "is_swap_denial_node",
    "is_memory_degradation_node",
    "is_intercept_resend_edge",
    "is_repeater_memory_measure_Z_node",
    "is_repeater_memory_measure_X_node",
    "is_information_bearing",
    "is_delivery_disruption",
]

ROUTE_CHOICE_FEATURE_NAMES = [
    "route_hops",
    "route_length",
    "route_length_over_shortest",
    "route_hops_over_shortest",
    "internal_node_count",
    "edge_count",
    "route_contains_node_hs_member",
    "route_contains_edge_hs_member",
    "route_node_hs_fraction",
    "route_edge_hs_fraction",
    "graph_N",
    "graph_E",
    "graph_C_log2",
    "graph_S",
    "graph_L",
    "graph_L_over_S",
]

def _node_hs_members(assignment: dict[str, Any]) -> frozenset[str]:
    payload = assignment.get("smallest_set_of_nodes_on_each_candidate_route", {})
    if not payload:
        payload = (assignment.get("route_structure") or {}).get("candidate_node_hitting_set", {})
    members: set[str] = set()
    for hs in payload.get("sets", []):
        members.update(hs)
    return frozenset(members)


def _edge_hs_members(assignment: dict[str, Any]) -> frozenset[str]:
    payload = assignment.get("smallest_set_of_edges_on_each_candidate_route", {})
    if not payload:
        payload = (assignment.get("route_structure") or {}).get("candidate_edge_hitting_set", {})
    members: set[str] = set()
    for hs in payload.get("sets", []):
        members.update(hs)
    members.update(_edge_hs_route_variants(assignment, members))
    return frozenset(members)


def _edge_hs_route_variants(assignment: dict[str, Any], raw_members: set[str]) -> set[str]:
    """Map corpus edge ids such as e20 onto route edge labels such as n1-n2."""
    variants: set[str] = set()
    route_structure = assignment.get("route_structure") or {}
    for route in route_structure.get("candidate_routes") or []:
        path = route.get("path") or []
        edge_ids = route.get("edge_ids") or []
        for edge_id, u, v in zip(edge_ids, path, path[1:]):
            if edge_id not in raw_members:
                continue
            a, b = sorted((str(u), str(v)))
            variants.add(str(edge_id))
            variants.add(f"{a}-{b}")
            variants.add(f"{a}--{b}")
    return variants


def _route_coverage(target: str, kind: str, routes: list[dict[str, Any]]) -> float:
    if not routes or kind == "none":
        return 0.0
    hits = 0
    for route in routes:
        if kind == "node" and target in route.get("path", [])[1:-1]:
            hits += 1
        elif kind == "edge" and target in route.get("edge_ids", []):
            hits += 1
    return hits / len(routes)


def action_structural_features(
    action_id: str,
    routes: list[dict[str, Any]],
    assignment: dict[str, Any],
) -> list[float]:
    """Return the STRUCTURAL_FEATURE_NAMES vector for one action."""
    kind, target = parse_action_parts(action_id)
    attack_type = parse_attack_type(action_id)
    is_node = 1.0 if kind == "node" else 0.0
    is_edge = 1.0 if kind == "edge" else 0.0

    node_hs = _node_hs_members(assignment)
    edge_hs = _edge_hs_members(assignment)
    in_node_hs = 1.0 if (kind == "node" and target in node_hs) else 0.0
    in_edge_hs = 1.0 if (kind == "edge" and target in edge_hs) else 0.0
    coverage = _route_coverage(target, kind, routes)

    route_structure = assignment.get("route_structure") or {}
    n_payload = assignment.get("smallest_set_of_nodes_on_each_candidate_route", {})
    if not n_payload:
        n_payload = route_structure.get("candidate_node_hitting_set", {})
    e_payload = assignment.get("smallest_set_of_edges_on_each_candidate_route", {})
    if not e_payload:
        e_payload = route_structure.get("candidate_edge_hitting_set", {})
    graph_N = float(n_payload.get("size") or 0)
    graph_E = float(e_payload.get("size") or 0)

    c_routes = (
        len(routes)
        or assignment.get("route_count")
        or len(route_structure.get("candidate_routes") or [])
        or assignment.get("candidate_routes", {}).get("count", 1)
    )
    graph_C_log2 = math.log2(max(1, c_routes))
    graph_S = float(
        min((_route_hops(r) for r in routes), default=0)
        or assignment.get("shortest_hops")
        or route_structure.get("shortest_hops")
        or assignment.get("shortest_route", {}).get("hop_count", 1)
    )
    graph_L = float(
        max((_route_hops(r) for r in routes), default=0)
        or assignment.get("longest_hops")
        or route_structure.get("longest_hops")
        or assignment.get("longest_route", {}).get("hop_count", 1)
    )
    graph_L_over_S = graph_L / max(1.0, graph_S)

    is_added_loss_edge = 1.0 if attack_type in {"edge_dos", "added_loss"} else 0.0
    is_swap_denial_node = 1.0 if attack_type == "swap_denial" else 0.0
    is_memory_degradation_node = 1.0 if attack_type == "memory_degradation" else 0.0
    is_intercept_resend_edge = 1.0 if attack_type in {
        "edge_information_probe",
        "intercept_resend",
    } else 0.0
    is_repeater_memory_measure_z_node = (
        1.0 if attack_type == "repeater_memory_measure_Z" else 0.0
    )
    is_repeater_memory_measure_x_node = (
        1.0 if attack_type == "repeater_memory_measure_X" else 0.0
    )
    is_information_bearing = 1.0 if attack_type in {
        "edge_information_probe",
        "intercept_resend",
        "repeater_memory_measure_Z",
        "repeater_memory_measure_X",
    } else 0.0
    is_delivery_disruption = 1.0 if attack_type in {
        "edge_dos",
        "added_loss",
        "swap_denial",
        "memory_degradation",
    } else 0.0

    return [
        is_node, is_edge, in_node_hs, in_edge_hs, coverage,
        graph_N, graph_E, graph_C_log2, graph_S, graph_L, graph_L_over_S,
        is_added_loss_edge,
        is_swap_denial_node,
        is_memory_degradation_node,
        is_intercept_resend_edge,
        is_repeater_memory_measure_z_node,
        is_repeater_memory_measure_x_node,
        is_information_bearing,
        is_delivery_disruption,
    ]


def route_choice_structural_features(
    route: dict[str, Any],
    routes: list[dict[str, Any]],
    assignment: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
) -> list[float]:
    """Return ROUTE_CHOICE_FEATURE_NAMES for one Alice candidate route."""
    ctx = context or _route_choice_feature_context(routes, assignment)
    route_hops = float(_route_hops(route))
    route_length = float(_route_length(route))
    shortest_length = float(ctx["shortest_length"])
    shortest_hops = int(ctx["shortest_hops"])

    internal_nodes = set(_route_internal_nodes(route))
    edge_ids = set(_route_edge_ids(route))
    node_hs = ctx["node_hs"]
    edge_hs = ctx["edge_hs"]
    node_hs_hits = internal_nodes.intersection(node_hs)
    edge_hs_hits = edge_ids.intersection(edge_hs)

    graph_N, graph_E, graph_C_log2, graph_S, graph_L, graph_L_over_S = ctx[
        "graph_metrics"
    ]

    return [
        route_hops,
        route_length,
        route_length / max(1e-9, shortest_length),
        route_hops / max(1.0, float(shortest_hops)),
        float(len(internal_nodes)),
        float(len(edge_ids)),
        1.0 if node_hs_hits else 0.0,
        1.0 if edge_hs_hits else 0.0,
        len(node_hs_hits) / max(1, len(internal_nodes)),
        len(edge_hs_hits) / max(1, len(edge_ids)),
        graph_N,
        graph_E,
        graph_C_log2,
        graph_S,
        graph_L,
        graph_L_over_S,
    ]


def _route_choice_feature_context(
    routes: list[dict[str, Any]],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "shortest_length": min((_route_length(r) for r in routes), default=1.0),
        "shortest_hops": min((_route_hops(r) for r in routes), default=1),
        "node_hs": _node_hs_members(assignment),
        "edge_hs": _edge_hs_members(assignment),
        "graph_metrics": _assignment_graph_metrics(assignment, routes),
    }


# Per-route diagnostic rows embed full edge/node payloads; on route-dense
# graphs (~30k routes) an unbounded dump reaches hundreds of MB per case.
_MAX_ROUTE_DIAGNOSTIC_ROWS = 512


def sequence_route_diagnostics(
    ir_dict: dict[str, Any],
    routes: list[dict[str, Any]],
    assignment: dict[str, Any] | None = None,
    *,
    max_route_rows: int = _MAX_ROUTE_DIAGNOSTIC_ROWS,
) -> dict[str, Any]:
    """Static post-transpilation route diagnostics.

    This is an inspection artifact, not an Alice observation and not a learning
    input.  QBER/fidelity are runtime measurements, so they remain null here
    unless a separate calibration pass is run. Routes beyond max_route_rows
    (in the runner's hops/length ordering) are counted but not emitted.
    """
    from sequence_game.physical import load_models_from_dir
    from sequence_game.topology import TopologyIR

    ir = TopologyIR.from_dict(ir_dict)
    registry = load_models_from_dir(_ROOT / "configs" / "physical", require_resolved=True)

    def by_kind(kind: str) -> Any:
        return next(model for model in registry.models.values() if model.device_kind == kind)

    fiber_model = by_kind("fiber_channel")
    memory_model = by_kind("memory")
    fiber_params = dict(fiber_model.parameters)
    memory_params = dict(memory_model.parameters)
    attenuation = _float_or_none(fiber_params.get("attenuation"))
    polarization_fidelity = _float_or_none(fiber_params.get("polarization_fidelity"))

    edge_by_label: dict[str, Any] = {}
    for edge in ir.edges:
        label = _demo_edge_id(edge.u, edge.v)
        double_label = f"{min(str(edge.u), str(edge.v))}--{max(str(edge.u), str(edge.v))}"
        total_loss_db = (
            attenuation * float(edge.length_m)
            if attenuation is not None
            else None
        )
        edge_payload = {
            "edge_id": edge.edge_id,
            "edge_label": label,
            "u": edge.u,
            "v": edge.v,
            "length_m": float(edge.length_m),
            "channel_profile_id": edge.channel_profile_id,
            "eve_eligible": bool(edge.eve_eligible),
            "fiber_model": _model_digest(fiber_model),
            "fiber_parameters": {
                "attenuation": fiber_params.get("attenuation"),
                "polarization_fidelity": fiber_params.get("polarization_fidelity"),
                "light_speed": fiber_params.get("light_speed"),
                "frequency": fiber_params.get("frequency"),
            },
            "static_total_loss_db": total_loss_db,
            "static_transmission_probability_proxy": (
                10 ** (-total_loss_db / 10.0)
                if total_loss_db is not None
                else None
            ),
            "static_polarization_fidelity_parameter": polarization_fidelity,
        }
        for key in {str(edge.edge_id), label, double_label}:
            edge_by_label[key] = edge_payload

    route_rows = []
    emitted_routes = routes[:max(0, int(max_route_rows))]
    for route in emitted_routes:
        edge_ids = [str(edge_id) for edge_id in route.get("edge_ids") or []]
        internal_nodes = _route_internal_nodes(route)
        edges = [edge_by_label[edge_id] for edge_id in edge_ids if edge_id in edge_by_label]
        total_length = float(
            route.get("total_length_m")
            or route.get("total_length")
            or sum(float(edge["length_m"]) for edge in edges)
        )
        total_loss_db = (
            attenuation * total_length if attenuation is not None else None
        )
        route_rows.append({
            "route_id": str(route.get("route_id")),
            "path": [str(node) for node in route.get("path") or []],
            "edge_ids": edge_ids,
            "internal_nodes": internal_nodes,
            "hop_count": int(route.get("hop_count") or len(edge_ids)),
            "total_length_m": total_length,
            "edge_count": len(edge_ids),
            "internal_node_count": len(internal_nodes),
            "edges": edges,
            "nodes": [
                _node_diagnostic(ir.nodes[node])
                for node in route.get("path") or []
                if node in ir.nodes
            ],
            "attack_exposure": {
                "edge_dos_targets": edge_ids,
                "edge_information_probe_targets": edge_ids,
                "swap_denial_targets": internal_nodes,
                "memory_degradation_targets": internal_nodes,
                "repeater_memory_measure_Z_targets": internal_nodes,
                "repeater_memory_measure_X_targets": internal_nodes,
            },
            "static_channel_summary": {
                "total_loss_db": total_loss_db,
                "transmission_probability_proxy": (
                    10 ** (-total_loss_db / 10.0)
                    if total_loss_db is not None
                    else None
                ),
                "fiber_polarization_fidelity_parameter": polarization_fidelity,
                "memory_model": _model_digest(memory_model),
                "memory_fidelity_parameter": memory_params.get("fidelity"),
                "memory_efficiency_parameter": memory_params.get("efficiency"),
                "memory_coherence_time_parameter": memory_params.get("coherence_time"),
            },
            "sampled_quality_metrics": {
                "qber": None,
                "fidelity": None,
                "delivery_success_rate": None,
                "calibration_required": True,
                "reason": (
                    "QBER/fidelity are sampled SeQUeNCe trial outputs, not "
                    "static route-transpilation fields."
                ),
            },
        })

    return {
        "scope_label": "REPEATER_RUNTIME",
        "diagnostic_kind": "post_transpilation_static_route_diagnostics",
        "metric_semantics": {
            "static_channel_summary": (
                "Inspectable edge/node/model parameters and simple loss proxies."
            ),
            "sampled_quality_metrics": (
                "Null until an explicit calibration pass runs route trials."
            ),
            "policy_visibility": (
                "Not fed into Alice's current DQN/Q update; sidecar artifact only."
            ),
        },
        "assignment_route_count": (
            (assignment or {}).get("candidate_route_count")
            or (assignment or {}).get("route_count")
            or (assignment or {}).get("candidate_routes", {}).get("count")
        ),
        "runtime_route_count": len(routes),
        "route_rows_emitted": len(route_rows),
        "route_rows_truncated": len(routes) > len(route_rows),
        "route_row_limit": int(max_route_rows),
        "routes": route_rows,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _model_digest(model: Any) -> dict[str, Any]:
    return {
        "model_name": model.model_name,
        "device_kind": model.device_kind,
        "scope": model.scope,
        "reference_tag": model.reference_tag,
    }


def _node_diagnostic(node: Any) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "roles": sorted(node.roles),
        "hardware_profile_id": node.hardware_profile_id,
        "coordinates": list(node.coordinates) if node.coordinates else None,
    }


def _assignment_graph_metrics(
    assignment: dict[str, Any],
    routes: list[dict[str, Any]],
) -> tuple[float, float, float, float, float, float]:
    route_structure = assignment.get("route_structure") or {}
    n_payload = assignment.get("smallest_set_of_nodes_on_each_candidate_route", {})
    if not n_payload:
        n_payload = route_structure.get("candidate_node_hitting_set", {})
    e_payload = assignment.get("smallest_set_of_edges_on_each_candidate_route", {})
    if not e_payload:
        e_payload = route_structure.get("candidate_edge_hitting_set", {})

    graph_N = float(
        assignment.get("candidate_node_hitting_size")
        or route_structure.get("candidate_node_hitting_size")
        or n_payload.get("size")
        or 0
    )
    graph_E = float(
        assignment.get("candidate_edge_hitting_size")
        or route_structure.get("candidate_edge_hitting_size")
        or e_payload.get("size")
        or 0
    )
    route_count = (
        len(routes)
        or assignment.get("route_count")
        or len(route_structure.get("candidate_routes") or [])
        or assignment.get("candidate_routes", {}).get("count")
        or 1
    )
    graph_C_log2 = math.log2(max(1, int(route_count)))
    graph_S = float(
        min((_route_hops(r) for r in routes), default=0)
        or assignment.get("shortest_hops")
        or route_structure.get("shortest_hops")
        or assignment.get("shortest_route", {}).get("hop_count")
        or min((_route_hops(r) for r in routes), default=1)
    )
    graph_L = float(
        max((_route_hops(r) for r in routes), default=0)
        or assignment.get("longest_hops")
        or route_structure.get("longest_hops")
        or assignment.get("longest_route", {}).get("hop_count")
        or max((_route_hops(r) for r in routes), default=1)
    )
    return graph_N, graph_E, graph_C_log2, graph_S, graph_L, graph_L / max(1.0, graph_S)


def _route_hops(route: dict[str, Any]) -> int:
    return int(route.get("hop_count") or len(route.get("edge_ids") or []) or 0)


def _route_length(route: dict[str, Any]) -> float:
    return float(
        route.get("total_length")
        or route.get("total_length_m")
        or route.get("length")
        or max(1, _route_hops(route))
    )


def _route_internal_nodes(route: dict[str, Any]) -> list[str]:
    if "internal_nodes" in route:
        return [str(node) for node in route.get("internal_nodes") or []]
    path = route.get("path") or []
    return [str(node) for node in path[1:-1]]


def _route_edge_ids(route: dict[str, Any]) -> list[str]:
    return [str(edge_id) for edge_id in route.get("edge_ids") or []]


def corpus_candidate_routes(
    ir_dict: dict[str, Any],
    alice: str,
    bob: str,
    assignment: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate every simple cycle-free Alice--Bob route."""
    from sequence_game.routing.route import route_id_for_path
    from sequence_game.topology import TopologyIR

    graph = TopologyIR.from_dict(ir_dict)
    route_selection_policy = "all_simple_paths"

    adjacency = {node: sorted(neighbors) for node, neighbors in graph.adjacency().items()}
    if alice not in adjacency or bob not in adjacency:
        return [], {
            "route_selection_policy": route_selection_policy,
            "route_enumeration_complete": True,
            "route_enumeration_truncated": False,
        }

    route_rows: list[dict[str, Any]] = []
    stack: list[tuple[str, ...]] = [(alice,)]
    while stack:
        path = stack.pop()
        node = path[-1]
        hop_count = len(path) - 1
        if node == bob:
            edge_ids = [_demo_edge_id(u, v) for u, v in zip(path, path[1:])]
            total_length = sum(
                float(graph.edge_between(u, v).length_m)  # type: ignore[union-attr]
                for u, v in zip(path, path[1:])
            )
            route_rows.append({
                "route_id": route_id_for_path(path),
                "path": list(path),
                "edge_ids": edge_ids,
                "total_length": total_length,
                "total_length_m": total_length,
                "hop_count": hop_count,
                "internal_nodes": list(path[1:-1]),
            })
            continue
        for neighbor in reversed(adjacency[node]):
            if neighbor in path:
                continue
            stack.append((*path, neighbor))

    route_rows.sort(key=lambda route: (
        int(route["hop_count"]),
        float(route["total_length"]),
        tuple(str(node) for node in route["path"]),
    ))
    return route_rows, {
        "route_selection_policy": route_selection_policy,
        "route_enumeration_complete": True,
        "route_enumeration_truncated": False,
    }


def _demo_edge_id(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


# ── Q-table → per-action avg rank ────────────────────────────────────────────

def _avg_q_per_action(q_table: dict[str, Any]) -> list[float]:
    entries = q_table.get("entries", [])
    num_actions = q_table.get("num_actions", 0)
    if not entries or not num_actions:
        return []
    sums = [0.0] * num_actions
    counts = [0] * num_actions
    for entry in entries:
        for i, v in enumerate(entry.get("q", [])):
            if i < num_actions:
                sums[i] += v
                counts[i] += 1
    return [s / max(1, c) for s, c in zip(sums, counts)]


def _rank_normalize(values: list[float]) -> list[float]:
    """Map each value to its fractional rank in [0, 1]; ties get mean rank."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        mean_rank = ((i + j - 1) / 2.0) / max(1, n - 1)
        for k in range(i, j):
            orig_idx, _ = indexed[k]
            ranks[orig_idx] = mean_rank
        i = j
    return ranks


# ── Per-graph structural rows ─────────────────────────────────────────────────

def per_graph_structural_rows(
    action_names: list[str],
    q_table: dict[str, Any],
    routes: list[dict[str, Any]],
    assignment: dict[str, Any],
    graph_id: str,
    family: str,
    *,
    combo: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per non-trivial action with structural features + Q-rank.

    Only node/edge attack actions are included (no_attack excluded).
    """
    avg_q = _avg_q_per_action(q_table)
    if not avg_q:
        return []
    ranks = _rank_normalize(avg_q)

    rows = []
    for i, action_id in enumerate(action_names):
        kind, _target = parse_action_parts(action_id)
        if kind == "none":
            continue
        feats = action_structural_features(action_id, routes, assignment)
        rows.append({
            "graph_id": graph_id,
            "family": family,
            "emission_type": "eve_attack_choice",
            "action_id": action_id,
            "features": feats,
            "avg_q": avg_q[i] if i < len(avg_q) else 0.0,
            "q_rank": ranks[i] if i < len(ranks) else 0.0,
            **_combo_row_metadata(combo),
        })
    return rows


def per_graph_route_choice_rows(
    routes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    assignment: dict[str, Any],
    graph_id: str,
    family: str,
    *,
    combo: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per candidate route with learned Alice route-choice signal.

    The target is empirical route choice rate from this backend corpus run. This
    is intentionally separate from Eve's attack-action Q-rank emission.
    """
    if not routes or not steps:
        return []

    counts: defaultdict[str, int] = defaultdict(int)
    reward_sums: defaultdict[str, float] = defaultdict(float)
    total_choices = 0
    for step in steps:
        route_id = step.get("alice_route_id")
        if not route_id:
            continue
        route_id = str(route_id)
        counts[route_id] += 1
        reward_sums[route_id] += float(step.get("alice_reward") or 0.0)
        total_choices += 1

    if total_choices <= 0:
        return []

    rows: list[dict[str, Any]] = []
    feature_context = _route_choice_feature_context(routes, assignment)
    for route in routes:
        route_id = str(route.get("route_id", ""))
        chosen_count = counts.get(route_id, 0)
        rows.append({
            "graph_id": graph_id,
            "family": family,
            "emission_type": "alice_route_choice",
            "route_id": route_id,
            "features": route_choice_structural_features(
                route,
                routes,
                assignment,
                context=feature_context,
            ),
            "choice_rate": chosen_count / total_choices,
            "mean_alice_reward": reward_sums.get(route_id, 0.0) / max(1, chosen_count),
            "chosen_count": chosen_count,
            "total_choices": total_choices,
            **_combo_row_metadata(combo),
        })
    return rows


def _combo_row_metadata(combo: dict[str, Any] | None) -> dict[str, Any]:
    if not combo:
        return {}
    return {
        "combo_key": combo.get("key"),
        "game_mode": combo.get("game_mode"),
        "eve_algo": combo.get("eve"),
        "alice_algo": combo.get("alice"),
        "combo_group": combo.get("group"),
    }


def _dt_payload(dt_result: Any) -> dict[str, Any]:
    importances = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in zip(
            dt_result.feature_names,
            dt_result.dt.feature_importances_,
        )
        if float(importance) > 0.0
    ]
    importances.sort(key=lambda row: row["importance"], reverse=True)
    return {
        "target_key": dt_result.target_key,
        "num_rows": dt_result.num_rows,
        "num_topologies": dt_result.num_topologies,
        "r2_score": dt_result.r2_score,
        "feature_names": dt_result.feature_names,
        "feature_importances": importances,
        "rules_text": dt_result.rules_text,
        "dt_depth": int(dt_result.dt.get_depth()),
        "dt_leaf_count": int(dt_result.dt.get_n_leaves()),
    }


def _safe_path_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _route_corridor_action_count(
    demo_topology: dict[str, Any],
    alice: str,
    bob: str,
    routes: list[dict[str, Any]],
    control_model: str,
) -> int:
    from sequence_game.gui_demo.demo_runner import (
        _attack_actions,
        _control_model_config,
        _route_corridor_attack_scope,
    )

    model = _control_model_config(control_model)
    attack_node_ids, attack_edge_ids = _route_corridor_attack_scope(routes, alice, bob)
    actions = _attack_actions(
        demo_topology,
        alice,
        bob,
        set(model["attack_kinds"]),
        attack_node_ids=attack_node_ids,
        attack_edge_ids=attack_edge_ids,
    )
    return len(actions)


# ── Eve-win helper (matches paper_results.py _is_eve_win) ────────────────────
# binary_collision: "hit" / "miss"
# attack_surface runtime rows use "accepted" / "delivery_failure" / "qber_abort"
# and may also emit "information_exposure" for accepted mixed probes.
# Direct/compiled transcript backends additionally emit "chsh_abort" and carry
# the authoritative per-step ``eve_win`` flag (which also covers accepted
# deliveries whose key material leaked); prefer the flag when present so the
# reported win rates match the reward the agents trained on.
_EVE_WIN_OUTCOMES: frozenset[str] = frozenset({
    "hit",
    "delivery_failure",
    "qber_abort",
    "chsh_abort",
    "information_exposure",
})


def _is_step_eve_win(step: dict) -> bool:
    flag = step.get("eve_win")
    if isinstance(flag, bool):
        return flag
    return step.get("public_outcome") in _EVE_WIN_OUTCOMES

_PUBLIC_ATTACK_FAMILY_LABELS: dict[str, str] = {
    "edge_dos": "added_loss_edge",
    "swap_denial": "swap_denial_node",
    "memory_degradation": "memory_degradation_node",
    "edge_information_probe": "intercept_resend_edge",
    "repeater_memory_measure_Z": "repeater_memory_measure_Z_node",
    "repeater_memory_measure_X": "repeater_memory_measure_X_node",
}


def _public_attack_family(action_id: str) -> str:
    if action_id == "no_attack":
        return "no_attack"
    action_kind = action_id.split(":", 1)[0]
    return _PUBLIC_ATTACK_FAMILY_LABELS.get(action_kind, action_kind)


def _attack_family_counts(action_names: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter(_public_attack_family(str(name)) for name in action_names)
    return dict(sorted(counts.items()))


def _eve_win_rates(steps: list[dict], final_window: list[dict]) -> tuple[float, float]:
    def rate(lst: list[dict]) -> float:
        if not lst:
            return 0.0
        return sum(1 for s in lst if _is_step_eve_win(s)) / len(lst)
    return rate(steps), rate(final_window)


def eve_win_curve(steps: list[dict], max_points: int = 25) -> dict[str, list]:
    """Windowed eve-win rates over up to max_points equal step bins.

    Compact per-case digest (a few hundred bytes) so corpus figures can show
    within-run learning curves without retaining full step records.
    """
    if not steps:
        return {"step_end": [], "eve_win_rate": []}
    n = len(steps)
    points = max(1, min(int(max_points), n))
    step_ends: list[int] = []
    rates: list[float] = []
    start = 0
    for i in range(points):
        end = round((i + 1) * n / points)
        if end <= start:
            continue
        window = steps[start:end]
        step_ends.append(end)
        rates.append(
            round(sum(1 for s in window if _is_step_eve_win(s)) / len(window), 4)
        )
        start = end
    return {"step_end": step_ends, "eve_win_rate": rates}


_RUNTIME_METRIC_DIGEST_KEYS = (
    "scope_label",
    "backend",
    "sequence_runtime_executed",
    "direct_e91_runtime_executed",
    "compiled_histogram_runtime_executed",
    "full_graph_runtime",
    "selected_path_runtime",
    "training_execution",
    "control_model",
    "reward_model",
    "runtime_engines",
    "runtime_case_count",
    "enabled_attack_kinds",
    "attack_surface_scope",
    "attack_surface_node_count",
    "attack_surface_edge_count",
    "information_gain_reward_enabled",
    "information_gain_metric_source",
    "public_outcome_counts",
    "information_exposures",
    "delivery_failures",
    "aborts",
    "eve_attack_selection_counts",
    "qber",
    "qber_summary",
    "qber_by_eve_win",
    "fidelity",
    "fidelity_summary",
    "fidelity_by_eve_win",
    "information_gain_mean_bits",
    "information_gain_max_bits",
    "information_gain_by_eve_win",
    "quality_by_public_outcome",
    "quality_by_attack_type",
    "delivered_pair_count",
    "route_entropy",
    "route_switches",
    "compiled_histogram_backend",
)


def _runtime_metric_digest(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in _RUNTIME_METRIC_DIGEST_KEYS
        if key in metrics
    }


# ── Single-graph runner ───────────────────────────────────────────────────────

# Default combo: primary main-object condition used for structural rows + DT training.
_PRIMARY_COMBO: dict[str, str] = {
    "game_mode": "attack_surface_all",
    "eve":       "deep_q",
    "alice":     "cautious_greedy",
    "key":       "attack_surface_all__deep_q__cautious_greedy",
}

LEARNED_EVE_ALGOS = frozenset({"q_learning", "deep_q"})
LEARNED_ALICE_ALGOS = frozenset({"q_learning", "deep_q"})


def run_one_corpus_graph(
    graph_record: dict[str, Any],
    corpus_root: Any,
    *,
    num_steps: int = 300,
    seed: int = 42,
    eve_algo: str = "deep_q",
    combos: list[dict] | None = None,
    dt_refit_config: Any = None,
    dt_refit_output_dir: Path | None = None,
    dt_refit_output_per_case: bool = False,
    dt_depth: int = 4,
    progress_every: int = 0,
    progress_label: str | None = None,
) -> dict[str, Any]:
    """Run game combos on one accepted corpus graph.

    combos: list of {game_mode, eve, alice, key} dicts.
            Defaults to the primary attack_surface_all × deep_q × cautious_greedy run.

    Structural rows (for pooled DTs) are emitted separately for every learned
    Eve attack-choice combo and every learned Alice route-choice combo.
    Per-combo win rates are stored in combo_results for LLM synthesis.
    """
    from sequence_game.gui_demo.demo_runner import run_binary_collision

    graph_id = graph_record["graph_id"]
    ir_dict = _load_corpus_json(corpus_root, graph_record["topology_path"])
    assignment = _load_corpus_json(corpus_root, graph_record["assignment_path"])

    demo_topo = ir_to_demo_topology(ir_dict)
    alice = graph_record["alice"]
    bob = graph_record["bob"]
    routes, route_selection = corpus_candidate_routes(
        ir_dict,
        alice,
        bob,
        assignment,
    )
    route_diagnostics = sequence_route_diagnostics(ir_dict, routes, assignment)

    family = str(graph_record.get("family") or "unknown")
    if family == "unknown":
        for fam in ("braess", "bottleneck", "ladder", "diamond", "layered_parallel", "bounded_er"):
            if fam in graph_id:
                family = fam
                break

    # Determine which combos to run.  When a caller deliberately supplies a
    # subset that excludes the legacy primary, report the first selected combo
    # as the top-level condition instead of silently running an extra DOS pass.
    run_combos = combos if combos is not None else [_PRIMARY_COMBO]
    if not run_combos:
        raise ValueError("combos must contain at least one condition")

    combo_keys = {
        combo.get("key", f"{combo['game_mode']}__{combo['eve']}__{combo['alice']}")
        for combo in run_combos
    }
    primary_combo = _PRIMARY_COMBO if _PRIMARY_COMBO["key"] in combo_keys else run_combos[0]
    primary_key = primary_combo.get(
        "key",
        f"{primary_combo['game_mode']}__{primary_combo['eve']}__{primary_combo['alice']}",
    )
    combo_results: dict[str, dict] = {}
    structural_rows: list[dict[str, Any]] = []
    route_choice_rows: list[dict[str, Any]] = []
    structural_row_counts: dict[str, int] = {}
    route_choice_row_counts: dict[str, int] = {}
    dt_refit_summaries: dict[str, dict[str, Any]] = {}
    dt_refit_artifact_dirs: dict[str, str] = {}
    primary_wr = primary_fwr = 0.0
    progress_interval = max(0, int(progress_every or 0))
    progress_prefix = progress_label or f"{graph_id}/{graph_record.get('assignment_id', '')}"
    num_actions = 0
    attack_surface_scope = ""
    attack_surface_node_count = 0
    attack_surface_edge_count = 0
    eve_action_family_counts: dict[str, int] = {}

    def _make_dt_refit_callback(combo: dict[str, Any], combo_dir: Path):
        from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt

        def _callback(event: Any, context: dict[str, Any]) -> None:
            fit_start = time.perf_counter()
            event_dir = combo_dir / f"refit_{int(event.refit_step):05d}"
            event_dir.mkdir(parents=True, exist_ok=True)
            output_paths: list[str] = []
            sample_count = 0
            max_depth_seen = 0
            leaf_count = 0

            if combo["eve"] in LEARNED_EVE_ALGOS and context.get("eve_q") is not None:
                eve_q = context["eve_q"].to_dict()
                action_names = [str(action["id"]) for action in context["actions"]]
                rows = per_graph_structural_rows(
                    action_names,
                    eve_q,
                    routes,
                    assignment,
                    graph_id,
                    family,
                    combo=combo,
                )
                if rows:
                    eve_dt = fit_cross_graph_dt(
                        rows,
                        max_depth=dt_depth,
                        target_key="q_rank",
                        feature_names=STRUCTURAL_FEATURE_NAMES,
                    )
                    payload = _dt_payload(eve_dt)
                    json_path = event_dir / "eve_attack_choice_dt.json"
                    rules_path = event_dir / "eve_attack_choice_dt_rules.txt"
                    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                    rules_path.write_text(payload["rules_text"], encoding="utf-8")
                    output_paths.extend([str(json_path), str(rules_path)])
                    sample_count += int(eve_dt.num_rows)
                    max_depth_seen = max(max_depth_seen, int(eve_dt.dt.get_depth()))
                    leaf_count += int(eve_dt.dt.get_n_leaves())

            if combo["alice"] in LEARNED_ALICE_ALGOS:
                rows = per_graph_route_choice_rows(
                    routes,
                    context["steps"],
                    assignment,
                    graph_id,
                    family,
                    combo=combo,
                )
                if rows:
                    alice_dt = fit_cross_graph_dt(
                        rows,
                        max_depth=dt_depth,
                        target_key="choice_rate",
                        feature_names=ROUTE_CHOICE_FEATURE_NAMES,
                    )
                    payload = _dt_payload(alice_dt)
                    json_path = event_dir / "alice_route_choice_dt.json"
                    rules_path = event_dir / "alice_route_choice_dt_rules.txt"
                    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                    rules_path.write_text(payload["rules_text"], encoding="utf-8")
                    output_paths.extend([str(json_path), str(rules_path)])
                    sample_count += int(alice_dt.num_rows)
                    max_depth_seen = max(max_depth_seen, int(alice_dt.dt.get_depth()))
                    leaf_count += int(alice_dt.dt.get_n_leaves())

            event.record_fit(
                wall_clock_seconds=time.perf_counter() - fit_start,
                training_sample_count=sample_count,
                dt_depth=max_depth_seen,
                dt_leaf_count=leaf_count,
                output_paths=output_paths,
            )

        return _callback

    for combo in run_combos:
        key = combo.get("key", f"{combo['game_mode']}__{combo['eve']}__{combo['alice']}")
        # oracle only valid vs cautious_greedy
        if combo["eve"] == "route_aware_oracle" and combo["alice"] != "cautious_greedy":
            combo_results[key] = {"skipped": "oracle only valid vs cautious_greedy"}
            continue
        try:
            observer = None
            refit_callback = None
            combo_dt_dir = None
            if dt_refit_config is not None and (
                combo["eve"] in LEARNED_EVE_ALGOS or combo["alice"] in LEARNED_ALICE_ALGOS
            ):
                from sequence_game.xai.dt_refit_observer import DynamicDTRefitObserver

                if combo["game_mode"] == "binary_collision":
                    eve_action_count = _route_corridor_action_count(
                        demo_topo,
                        alice,
                        bob,
                        routes,
                        combo["game_mode"],
                    )
                else:
                    eve_action_count = e91_runtime_action_count(
                        ir_dict,
                        routes,
                        combo["game_mode"],
                    )
                observer = DynamicDTRefitObserver(
                    dt_refit_config,
                    total_steps=num_steps,
                    alice_action_count=(
                        len(routes) if combo["alice"] in LEARNED_ALICE_ALGOS else 0
                    ),
                    eve_action_count=(
                        eve_action_count if combo["eve"] in LEARNED_EVE_ALGOS else 0
                    ),
                )
                if dt_refit_output_dir is not None:
                    if dt_refit_output_per_case:
                        combo_dt_dir = Path(dt_refit_output_dir) / "dt"
                    else:
                        combo_dt_dir = (
                            Path(dt_refit_output_dir)
                            / _safe_path_part(graph_id)
                            / _safe_path_part(graph_record.get("assignment_id") or "assignment")
                            / _safe_path_part(key)
                        )
                    combo_dt_dir.mkdir(parents=True, exist_ok=True)
                    refit_callback = _make_dt_refit_callback(combo, combo_dt_dir)
            if combo["game_mode"] == "binary_collision":
                result = run_binary_collision(
                    demo_topo, alice, bob,
                    eve_algo=combo["eve"],
                    alice_algo=combo["alice"],
                    control_model=combo["game_mode"],
                    num_steps=num_steps,
                    seed=seed,
                    route_candidates=routes,
                    dt_refit_observer=observer,
                    dt_refit_callback=refit_callback,
                )
            else:
                result = run_e91_graph_runtime_game(
                    ir_dict,
                    alice,
                    bob,
                    routes=routes,
                    eve_algo=combo["eve"],
                    alice_algo=combo["alice"],
                    game_mode=combo["game_mode"],
                    num_steps=num_steps,
                    seed=seed,
                    final_window_size=min(200, num_steps),
                    dt_refit_observer=observer,
                    dt_refit_callback=refit_callback,
                    progress_every=progress_interval,
                    progress_label=f"{progress_prefix} {key}",
                )
            if observer is not None:
                if combo_dt_dir is not None:
                    observer.write_outputs(combo_dt_dir)
                    dt_refit_artifact_dirs[key] = str(combo_dt_dir)
                dt_summary = observer.summary()
                dt_refit_summaries[key] = dt_summary
            steps    = result["steps"]
            final_w  = result.get("final_window", steps[-20:])
            wr, fwr  = _eve_win_rates(steps, final_w)
            metrics = result["summary"].get("metrics", {})
            combo_results[key] = {
                "eve_win_rate":   round(wr, 4),
                "final_win_rate": round(fwr, 4),
                "eve_win_curve":  eve_win_curve(steps),
                "runtime_metrics": _runtime_metric_digest(metrics),
            }
            if observer is not None:
                combo_results[key]["dt_refit_mode"] = dt_summary["refit_mode"]
                combo_results[key]["dt_refit_count"] = dt_summary["refit_count"]
                combo_results[key]["dt_refit_artifact_dir"] = (
                    dt_refit_artifact_dirs.get(key)
                )
            if key == primary_key:
                primary_wr = combo_results[key]["eve_win_rate"]
                primary_fwr = combo_results[key]["final_win_rate"]
                action_names = result["summary"]["q_tables"]["eve"]["action_names"]
                num_actions = len(action_names)
                eve_action_family_counts = _attack_family_counts(action_names)
                attack_surface_scope = str(metrics.get("attack_surface_scope", ""))
                attack_surface_node_count = int(metrics.get("attack_surface_node_count") or 0)
                attack_surface_edge_count = int(metrics.get("attack_surface_edge_count") or 0)
            if combo["eve"] in LEARNED_EVE_ALGOS:
                q_table = result["summary"]["q_tables"]["eve"]["raw"]
                action_names = result["summary"]["q_tables"]["eve"]["action_names"]
                combo_rows = per_graph_structural_rows(
                    action_names, q_table, routes, assignment, graph_id, family, combo=combo
                )
                structural_rows.extend(combo_rows)
                structural_row_counts[key] = len(combo_rows)
            if combo["alice"] in LEARNED_ALICE_ALGOS:
                combo_route_rows = per_graph_route_choice_rows(
                    routes, steps, assignment, graph_id, family, combo=combo
                )
                route_choice_rows.extend(combo_route_rows)
                route_choice_row_counts[key] = len(combo_route_rows)
        except Exception as exc:
            combo_results[key] = {"error": str(exc)}

    # Fall back: if primary wasn't in combos, run it separately for backward-compatible
    # top-level metrics and structural rows.
    if primary_key not in combo_results:
        try:
            result = run_e91_graph_runtime_game(
                ir_dict,
                alice,
                bob,
                routes=routes,
                eve_algo=_PRIMARY_COMBO["eve"],
                alice_algo=_PRIMARY_COMBO["alice"],
                game_mode=_PRIMARY_COMBO["game_mode"],
                num_steps=num_steps,
                seed=seed,
                final_window_size=min(200, num_steps),
                progress_every=progress_interval,
                progress_label=f"{progress_prefix} {primary_key}",
            )
            steps = result["steps"]
            final_w = result.get("final_window", steps[-20:])
            primary_wr, primary_fwr = _eve_win_rates(steps, final_w)
            primary_wr = round(primary_wr, 4)
            primary_fwr = round(primary_fwr, 4)
            action_names = result["summary"]["q_tables"]["eve"]["action_names"]
            num_actions = len(action_names)
            eve_action_family_counts = _attack_family_counts(action_names)
            metrics = result["summary"].get("metrics", {})
            attack_surface_scope = str(metrics.get("attack_surface_scope", ""))
            attack_surface_node_count = int(metrics.get("attack_surface_node_count") or 0)
            attack_surface_edge_count = int(metrics.get("attack_surface_edge_count") or 0)
            combo_results[primary_key] = {
                "eve_win_rate": primary_wr,
                "final_win_rate": primary_fwr,
                "eve_win_curve": eve_win_curve(steps),
                "runtime_metrics": _runtime_metric_digest(metrics),
            }
            combo_rows = per_graph_structural_rows(
                result["summary"]["q_tables"]["eve"]["action_names"],
                result["summary"]["q_tables"]["eve"]["raw"],
                routes,
                assignment,
                graph_id,
                family,
                combo=_PRIMARY_COMBO,
            )
            structural_rows.extend(combo_rows)
            structural_row_counts[primary_key] = len(combo_rows)
        except Exception:
            combo_results[primary_key] = {"error": "primary combo failed"}

    return {
        "graph_id": graph_id,
        "assignment_id": graph_record.get("assignment_id"),
        "assignment_policy": graph_record.get("assignment_policy"),
        "strategy_validity": graph_record.get("strategy_validity"),
        "family": family,
        "alice": alice,
        "bob": bob,
        "graph_N": _metric_size(graph_record, "N", "candidate_node_hitting_size"),
        "graph_E": _metric_size(graph_record, "E", "candidate_edge_hitting_size"),
        "graph_C": len(routes),
        "assignment_graph_C": (
            graph_record.get("candidate_route_count") or graph_record.get("route_count")
        ),
        "graph_S": min((_route_hops(route) for route in routes), default=0)
        or graph_record.get("shortest_hops"),
        "graph_L": max((_route_hops(route) for route in routes), default=0)
        or graph_record.get("longest_hops"),
        "num_steps": num_steps,
        "run_seed": seed,
        **route_selection,
        "eve_win_rate":   primary_wr,
        "final_win_rate": primary_fwr,
        "num_actions": num_actions,
        "eve_action_family_counts": eve_action_family_counts,
        "eve_attack_families": sorted(
            family for family in eve_action_family_counts if family != "no_attack"
        ),
        "attack_surface_scope": attack_surface_scope,
        "attack_surface_node_count": attack_surface_node_count,
        "attack_surface_edge_count": attack_surface_edge_count,
        "num_routes": len(routes),
        "sequence_route_diagnostics": route_diagnostics,
        "primary_combo_key": primary_key,
        "learned_combo_keys": sorted(structural_row_counts),
        "learned_alice_combo_keys": sorted(route_choice_row_counts),
        "structural_row_counts": structural_row_counts,
        "route_choice_row_counts": route_choice_row_counts,
        "dt_refit_summaries": dt_refit_summaries,
        "dt_refit_artifact_dirs": dt_refit_artifact_dirs,
        "structural_rows": structural_rows,
        "route_choice_rows": route_choice_rows,
        "combo_results": combo_results,
    }


def _resolve_corpus_path(corpus_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for candidate in (corpus_root / path, path):
        if candidate.exists():
            return candidate
    return corpus_root / path


def _load_corpus_json(corpus_root: Any, raw_path: str) -> Any:
    loader = getattr(corpus_root, "load_json", None)
    if callable(loader):
        return loader(raw_path)
    path = _resolve_corpus_path(Path(corpus_root), raw_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_size(record: dict[str, Any], old_key: str, new_key: str) -> Any:
    value = record.get(new_key)
    if value is not None:
        return value
    old_value = record.get(old_key)
    if isinstance(old_value, dict):
        return old_value.get("size")
    return old_value
