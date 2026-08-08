"""Decision-tree rows for Exp3/oracle outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt

from .backend import ActionSpec
from .corpus import GraphCase, bottleneck_nodes
from .online import OnlineRunSummary
from .oracle import OracleSummary
from .payoff import PayoffEstimate


GRAPH_FEATURE_NAMES = [
    "num_nodes",
    "num_edges",
    "num_routes",
    "node_disjoint_paths",
    "edge_disjoint_paths",
    "shortest_hops",
    "longest_hops",
    "route_hop_ratio",
    "bottleneck_node_count",
    "mean_edge_length_m",
    "mean_route_length_m",
    "max_route_overlap",
]

ACTION_FEATURE_NAMES = [
    "is_no_attack",
    "is_edge_intercept_resend",
    "is_memory_degradation",
    "target_route_coverage",
    "target_is_bottleneck",
    "num_routes",
    "node_disjoint_paths",
]

ROUTE_FEATURE_NAMES = [
    "hop_count",
    "length_m",
    "length_over_shortest",
    "internal_node_count",
    "contains_bottleneck",
    "node_disjoint_paths",
    "mean_overlap_with_other_routes",
]


def build_dt_payload(cases: list[GraphCase],
                     payoffs: dict[str, PayoffEstimate],
                     oracles: dict[str, OracleSummary],
                     online: dict[str, dict[str, OnlineRunSummary]],
                     actions_by_graph: dict[str, list[ActionSpec]],
                     *, max_depth: int) -> dict[str, Any]:
    graph_rows = graph_value_rows(cases, oracles, online)
    action_rows = action_strategy_rows(cases, payoffs, actions_by_graph, oracles, online)
    route_rows = route_strategy_rows(cases, payoffs, oracles, online)
    payload: dict[str, Any] = {
        "graph_rows": graph_rows,
        "action_rows": action_rows,
        "route_rows": route_rows,
        "trees": {},
    }
    if graph_rows:
        payload["trees"]["graph_oracle_retention"] = (
            _fit(graph_rows, "oracle_retention", GRAPH_FEATURE_NAMES, max_depth)
            if any(row["oracle_retention"] is not None for row in graph_rows)
            else None
        )
    if action_rows:
        payload["trees"]["action_oracle_eve_strategy_prob"] = _fit(
            action_rows, "oracle_eve_strategy_prob", ACTION_FEATURE_NAMES, max_depth)
        payload["trees"]["action_expected_denial_under_oracle_alice"] = _fit(
            action_rows, "action_expected_denial_under_oracle_alice", ACTION_FEATURE_NAMES, max_depth)
    if route_rows:
        payload["trees"]["route_oracle_alice_strategy_prob"] = _fit(
            route_rows, "oracle_alice_strategy_prob", ROUTE_FEATURE_NAMES, max_depth)
    return payload


def graph_value_rows(cases: list[GraphCase], oracles: dict[str, OracleSummary],
                     online: dict[str, dict[str, OnlineRunSummary]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        oracle = oracles[case.graph_id]
        exp3 = online.get(case.graph_id, {}).get("exp3_vs_exp3")
        cautious = online.get(case.graph_id, {}).get(
            "exp3_eve_vs_cautious_greedy_alice")
        rows.append({
            "graph_id": case.graph_id,
            "family": case.family,
            "features": [case.features[name] for name in GRAPH_FEATURE_NAMES],
            "oracle_retention": oracle.retention,
            "oracle_value": oracle.value,
            "baseline_rate": oracle.baseline_rate,
            "exp3_final_retention": exp3.final_retention if exp3 else None,
            "exp3_exploitability_vs_oracle": exp3.exploitability_vs_payoff if exp3 else None,
            "cautious_greedy_final_retention": (
                cautious.final_retention if cautious else None
            ),
            "cautious_greedy_hit_rate": cautious.hit_rate if cautious else None,
            "cautious_greedy_exploitability_vs_payoff": (
                cautious.exploitability_vs_payoff if cautious else None
            ),
            "node_disjoint_paths": case.features["node_disjoint_paths"],
            "bottleneck_count": case.features["bottleneck_node_count"],
        })
    return rows


def action_strategy_rows(cases: list[GraphCase],
                         payoffs: dict[str, PayoffEstimate],
                         actions_by_graph: dict[str, list[ActionSpec]],
                         oracles: dict[str, OracleSummary],
                         online: dict[str, dict[str, OnlineRunSummary]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        actions = actions_by_graph[case.graph_id]
        payoff = payoffs[case.graph_id]
        oracle = oracles[case.graph_id]
        exp3 = online.get(case.graph_id, {}).get("exp3_vs_exp3")
        cautious = online.get(case.graph_id, {}).get(
            "exp3_eve_vs_cautious_greedy_alice")
        bottlenecks = bottleneck_nodes(case.routes)
        for index, action in enumerate(actions):
            action_key_rate = float(oracle.alice_strategy @ payoff.payoff[:, index])
            hit_diagnostics = _action_hit_diagnostics(payoff, action.action_id)
            rows.append({
                "graph_id": case.graph_id,
                "family": case.family,
                "action_id": action.action_id,
                "features": _action_features(case, action, bottlenecks),
                "oracle_eve_strategy_prob": float(oracle.eve_strategy[index]),
                "exp3_eve_empirical_prob": (
                    float(exp3.eve_strategy[index]) if exp3 and index < len(exp3.eve_strategy) else None
                ),
                "cautious_control_eve_empirical_prob": (
                    float(cautious.eve_strategy[index])
                    if cautious and index < len(cautious.eve_strategy) else None
                ),
                "action_expected_denial_under_oracle_alice": float(1.0 - action_key_rate),
                "active_hit_sample_count": hit_diagnostics["sample_count"],
                "active_hit_accepted_rate": hit_diagnostics["accepted_rate"],
                "active_hit_mean_chsh_s": hit_diagnostics["mean_chsh_s"],
                "active_hit_mean_qber": hit_diagnostics["mean_qber"],
            })
    return rows


def route_strategy_rows(cases: list[GraphCase], payoffs: dict[str, PayoffEstimate],
                        oracles: dict[str, OracleSummary],
                        online: dict[str, dict[str, OnlineRunSummary]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        payoff = payoffs[case.graph_id]
        oracle = oracles[case.graph_id]
        exp3 = online.get(case.graph_id, {}).get("exp3_vs_exp3")
        cautious = online.get(case.graph_id, {}).get(
            "exp3_eve_vs_cautious_greedy_alice")
        bottlenecks = bottleneck_nodes(case.routes)
        no_attack_index = payoff.action_ids.index("no_attack")
        for index, route in enumerate(case.routes):
            rows.append({
                "graph_id": case.graph_id,
                "family": case.family,
                "route_id": route["route_id"],
                "features": _route_features(case, route, bottlenecks),
                "oracle_alice_strategy_prob": float(oracle.alice_strategy[index]),
                "exp3_alice_empirical_prob": (
                    float(exp3.alice_strategy[index]) if exp3 and index < len(exp3.alice_strategy) else None
                ),
                "cautious_greedy_alice_empirical_prob": (
                    float(cautious.alice_strategy[index])
                    if cautious and index < len(cautious.alice_strategy) else None
                ),
                "no_attack_key_rate": float(payoff.payoff[index, no_attack_index]),
                "worst_case_key_rate": float(np.min(payoff.payoff[index, :])),
            })
    return rows


def _fit(rows: list[dict[str, Any]], target: str, feature_names: list[str],
         max_depth: int) -> dict[str, Any]:
    valid = [row for row in rows if row.get(target) is not None]
    result = fit_cross_graph_dt(
        valid,
        max_depth=max_depth,
        target_key=target,
        feature_names=feature_names,
    )
    return {
        "target": target,
        "feature_names": feature_names,
        "num_rows": result.num_rows,
        "num_topologies": result.num_topologies,
        "r2_score": result.r2_score,
        "rules_text": result.rules_text,
    }


def _action_features(case: GraphCase, action: ActionSpec, bottlenecks: set[str]) -> list[float]:
    coverage = 0.0
    if action.target:
        if action.kind == "edge":
            coverage = sum(
                1 for route in case.routes
                if action.target in _route_edge_labels(route)
            ) / max(1, len(case.routes))
        else:
            coverage = sum(
                1 for route in case.routes
                if action.target in set(route.get("internal_nodes") or [])
            ) / max(1, len(case.routes))
    return [
        1.0 if action.attack_type == "no_attack" else 0.0,
        1.0 if action.attack_type == "edge_intercept_resend" else 0.0,
        1.0 if action.attack_type == "memory_degradation" else 0.0,
        float(coverage),
        1.0 if action.target in bottlenecks else 0.0,
        case.features["num_routes"],
        case.features["node_disjoint_paths"],
    ]


def _route_edge_labels(route: dict[str, Any]) -> set[str]:
    path = [str(node) for node in route.get("path", [])]
    labels = {
        f"{min(u, v)}-{max(u, v)}"
        for u, v in zip(path, path[1:])
    }
    labels.update(str(edge_id) for edge_id in route.get("edge_ids", ()))
    return labels


def _route_features(case: GraphCase, route: dict[str, Any], bottlenecks: set[str]) -> list[float]:
    shortest = min(float(row["total_length_m"]) for row in case.routes)
    internal = set(route.get("internal_nodes") or [])
    overlaps = []
    for other in case.routes:
        if other is route:
            continue
        overlaps.append(len(internal & set(other.get("internal_nodes") or [])))
    return [
        float(route["hop_count"]),
        float(route["total_length_m"]),
        float(route["total_length_m"]) / max(1e-9, shortest),
        float(len(internal)),
        1.0 if internal & bottlenecks else 0.0,
        case.features["node_disjoint_paths"],
        float(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
    ]


def _action_hit_diagnostics(
        payoff: PayoffEstimate,
        action_id: str,
) -> dict[str, Any]:
    cells = [
        cell for cell in payoff.cells
        if (
            cell.action_id == action_id
            and cell.sequence_timing.get("payoff_model") == "attack_route_profile_cache"
        )
    ]
    sample_count = sum(cell.trial_count for cell in cells)
    accepted_count = sum(cell.accepted_count for cell in cells)
    chsh_values = [
        float(cell.mean_chsh_s)
        for cell in cells
        if cell.mean_chsh_s is not None
    ]
    qber_values = [
        float(cell.mean_qber)
        for cell in cells
        if cell.mean_qber is not None
    ]
    return {
        "sample_count": sample_count,
        "accepted_rate": (
            accepted_count / sample_count if sample_count else None
        ),
        "mean_chsh_s": (
            float(np.mean(chsh_values)) if chsh_values else None
        ),
        "mean_qber": (
            float(np.mean(qber_values)) if qber_values else None
        ),
    }
