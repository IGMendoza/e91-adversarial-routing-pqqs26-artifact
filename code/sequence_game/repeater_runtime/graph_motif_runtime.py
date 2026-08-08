"""Legacy two-route bridge over the graph-to-SeQUeNCe compiler.

The graph-to-runtime compiler in ``graph_to_sequence`` is the source of truth.
This module keeps the older ``route_r0`` / ``route_r1`` live Q-Eve path working
as a legacy selected-path scaffold.  It does not collapse graph paths into one
repeater as the scientific runtime model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..claims import (
    BLOCKED,
    BLOCKED_PENDING_GRAPH_INTERFACE,
    CONTROL_GAME,
    IMPLEMENTED_CONTROL_ONLY,
    GRAPH_OBJECTIVE_CANDIDATE,
    blocked_result,
    validate_blocked_response,
)
from ..topology.ir import TopologyIR
from .graph_to_sequence import (
    STRATEGY_VALID_MULTI_ROUTE,
    GraphToSequenceRouteMapping,
    compile_graph_to_sequence_runtime,
)
from .two_route_control import ROUTE_PATHS
from .two_route_live_q_eve import (
    LiveQEveTrainingConfig,
    train_live_q_eve_two_route,
)

GRAPH_MOTIF_ABSTRACTION = "GRAPH_TO_SELECTED_REPEATER_RUNTIME_MOTIF"
GRAPH_MOTIF_BLOCK_REASON = (
    "graph cannot supply two non-degenerate Alice-Bob routes for the legacy "
    "route_r0/route_r1 selected-path scaffold"
)


@dataclass(frozen=True)
class GraphMotifSelectionConfig:
    alice: str
    bob: str
    seed: int = 1

    def __post_init__(self) -> None:
        if not self.alice or not self.bob:
            raise ValueError("alice and bob must be non-empty")
        if self.alice == self.bob:
            raise ValueError("alice and bob must be distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alice": self.alice,
            "bob": self.bob,
            "seed": self.seed,
        }


def select_graph_two_route_motif(
        topology: TopologyIR,
        config: GraphMotifSelectionConfig) -> dict[str, Any]:
    """Select two compiled graph routes for the legacy route_r0/route_r1 runner."""

    compile_result = compile_graph_to_sequence_runtime(
        topology,
        alice=config.alice,
        bob=config.bob,
    )
    compile_dict = compile_result.to_dict()
    candidates = [_candidate_route_row(route) for route in compile_result.candidate_routes]
    legacy_supported = (
        compile_result.runtime_supported
        and compile_result.strategy_label == STRATEGY_VALID_MULTI_ROUTE
        and len(compile_result.candidate_routes) >= 2
    )
    selected_pair: Optional[tuple[GraphToSequenceRouteMapping, GraphToSequenceRouteMapping]]
    relation = "unavailable"
    selected_pair = None
    if legacy_supported:
        selected_pair, relation = _select_route_pair(compile_result.candidate_routes)
        mapping = _mapping_table(config, selected_pair)
    else:
        mapping = []

    report = {
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "abstraction": GRAPH_MOTIF_ABSTRACTION,
        "compiler_source_of_truth": "sequence_game.repeater_runtime.graph_to_sequence",
        "runtime_supported": compile_result.runtime_supported,
        "runtime_blocked_code": compile_result.runtime_blocked_code,
        "runtime_blocked_reason": compile_result.runtime_blocked_reason,
        "strategy_benchmark_supported": compile_result.strategy_benchmark_supported,
        "strategy_label": compile_result.strategy_label,
        "strategy_blocked_reason": compile_result.strategy_blocked_reason,
        "legacy_selected_path_supported": legacy_supported,
        "selection_config": config.to_dict(),
        "candidate_routes": candidates,
        "selected_relation": relation,
        "selected_routes": (
            [_selected_route_row(route) for route in selected_pair]
            if selected_pair is not None else []
        ),
        "mapping_table": mapping,
        "runtime_route_ids": ["route_r0", "route_r1"],
        "legacy_runtime_topology_label": "selected_two_route_reference_alice_r0_r1_bob",
        "compile_result": compile_dict,
        "abstraction_note": (
            "The compiled graph route preserves every repeater and fiber edge. "
            "route_r0/route_r1 are retained only as a legacy selected-path "
            "Q-Eve scaffold."
        ),
        "graph_objective": _graph_objective_block(),
    }
    if not legacy_supported:
        report["legacy_scaffold_blocked_reason"] = _legacy_scaffold_blocked_reason(
            compile_result.strategy_label,
            compile_result.strategy_blocked_reason,
            compile_result.runtime_blocked_reason,
        )
    return report


def run_live_q_eve_on_graph_motif(
        topology: TopologyIR,
        motif_config: GraphMotifSelectionConfig,
        *,
        live_config: Optional[LiveQEveTrainingConfig] = None) -> dict[str, Any]:
    """Run live selected-path Q-Eve only after a valid graph motif compiles."""

    motif_report = select_graph_two_route_motif(topology, motif_config)
    if not motif_report.get("legacy_selected_path_supported", False):
        response = blocked_result(
            GRAPH_MOTIF_BLOCK_REASON,
            claim_status=BLOCKED_PENDING_GRAPH_INTERFACE,
            extra={
                "runtime_supported": motif_report.get("runtime_supported", False),
                "strategy_label": motif_report.get("strategy_label"),
                "legacy_selected_path_supported": False,
                "motif_report": motif_report,
                "graph_objective": _graph_objective_block(),
            },
        )
        validate_blocked_response(response)
        return response

    training = train_live_q_eve_two_route(
        live_config or LiveQEveTrainingConfig(target_steps=3, seed=motif_config.seed)
    )
    return {
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "abstraction": GRAPH_MOTIF_ABSTRACTION,
        "motif_report": motif_report,
        "mapping_table": motif_report["mapping_table"],
        "runtime_training": training,
        "runtime_scope_label": training["summary"]["scope_label"],
        "graph_objective": _graph_objective_block(),
    }


def write_graph_motif_live_q_eve_artifacts(
        topology: TopologyIR,
        motif_config: GraphMotifSelectionConfig,
        *,
        report_path: Path,
        mapping_path: Path,
        runtime_summary_path: Path,
        runtime_steps_path: Path,
        runtime_q_table_path: Path,
        live_config: Optional[LiveQEveTrainingConfig] = None) -> dict[str, Any]:
    """Write motif report, mapping CSV, and a small selected-path live run."""

    live_config = live_config or LiveQEveTrainingConfig(
        target_steps=3,
        seed=motif_config.seed,
    )
    result = run_live_q_eve_on_graph_motif(
        topology,
        motif_config,
        live_config=live_config,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    report = result.get("motif_report", result)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_mapping_csv(result.get("mapping_table", []), mapping_path)

    if result.get("scope_label") != BLOCKED:
        training = result["runtime_training"]
        runtime_summary_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_steps_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_q_table_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_summary_path.write_text(
            json.dumps(training["summary"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with runtime_steps_path.open("w", encoding="utf-8") as f:
            for step in training["steps"]:
                f.write(json.dumps(step, sort_keys=True) + "\n")
        runtime_q_table_path.write_text(
            json.dumps(training["q_table"].to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _select_route_pair(
        routes: list[GraphToSequenceRouteMapping]
        ) -> tuple[tuple[GraphToSequenceRouteMapping, GraphToSequenceRouteMapping], str]:
    scored = []
    for i, first in enumerate(routes):
        for second in routes[i + 1:]:
            relation = _route_pair_relation(first, second)
            relation_rank = {
                "internally_node_disjoint": 0,
                "edge_disjoint": 1,
                "degenerate_bottleneck": 2,
            }[relation]
            scored.append((
                relation_rank,
                first.total_length_m + second.total_length_m,
                first.hop_count + second.hop_count,
                _compiled_route_sort_key(first),
                _compiled_route_sort_key(second),
                first,
                second,
                relation,
            ))
    scored.sort(key=lambda row: row[:5])
    best = scored[0]
    return (best[5], best[6]), best[7]


def _route_pair_relation(first: GraphToSequenceRouteMapping,
                         second: GraphToSequenceRouteMapping) -> str:
    first_internal = set(first.internal_node_set)
    second_internal = set(second.internal_node_set)
    if first_internal.isdisjoint(second_internal):
        return "internally_node_disjoint"
    if set(first.edge_set).isdisjoint(second.edge_set):
        return "edge_disjoint"
    return "degenerate_bottleneck"


def _compiled_route_sort_key(route: GraphToSequenceRouteMapping) -> tuple[float, int, tuple[str, ...]]:
    return (route.total_length_m, route.hop_count, route.graph_path)


def _mapping_table(config: GraphMotifSelectionConfig,
                   selected_pair: tuple[GraphToSequenceRouteMapping,
                                        GraphToSequenceRouteMapping]) -> list[dict[str, Any]]:
    rows = []
    for index, route in enumerate(selected_pair):
        legacy_route_id = f"route_r{index}"
        rows.append({
            "scope_label": CONTROL_GAME,
            "claim_status": IMPLEMENTED_CONTROL_ONLY,
            "original_route_id": route.route_id,
            "runtime_route_id": route.runtime_route_id,
            "original_path": list(route.graph_path),
            "original_edge_ids": list(route.edge_ids),
            "original_edge_lengths_m": list(route.edge_lengths_m),
            "original_total_length_m": route.total_length_m,
            "original_hop_count": route.hop_count,
            "compiled_runtime_path": list(route.runtime_path),
            "compiled_intermediate_repeaters": list(route.intermediate_repeaters),
            "legacy_selected_path_scaffold": True,
            "legacy_route_id": legacy_route_id,
            "legacy_scaffold_path": ROUTE_PATHS[legacy_route_id],
            "legacy_runtime_intermediate": f"r{index}",
            "original_alice": config.alice,
            "original_bob": config.bob,
            "mapping_visibility": "analysis_artifact_not_eve_observation",
        })
    return rows


def _candidate_route_row(route: GraphToSequenceRouteMapping) -> dict[str, Any]:
    row = route.to_dict()
    row.update({
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "legacy_two_route_candidate": True,
    })
    return row


def _selected_route_row(route: GraphToSequenceRouteMapping) -> dict[str, Any]:
    row = _candidate_route_row(route)
    row.update({
        "internal_nodes": list(route.intermediate_repeaters),
    })
    return row


def _legacy_scaffold_blocked_reason(
        strategy_label: str,
        strategy_reason: Optional[str],
        runtime_reason: Optional[str]) -> str:
    if runtime_reason:
        return runtime_reason
    if strategy_reason:
        return strategy_reason
    return (
        f"{strategy_label} does not provide the non-degenerate two-route "
        "choice expected by the legacy selected-path Q-Eve scaffold"
    )


def _graph_objective_block() -> dict[str, Any]:
    return {
        "scope_label": BLOCKED,
        "claim_status": BLOCKED_PENDING_GRAPH_INTERFACE,
        "blocked": True,
        "candidate_scope_label": GRAPH_OBJECTIVE_CANDIDATE,
        "graph_objective_supported": False,
        "blocked_reason": (
            "final graph hardness / operational-entanglement objective is not "
            "defined or mapped; compiled selected-route runtime does not prove "
            "the graph objective"
        ),
    }


def _write_mapping_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scope_label",
        "claim_status",
        "original_route_id",
        "original_path",
        "original_edge_ids",
        "original_edge_lengths_m",
        "original_total_length_m",
        "original_hop_count",
        "runtime_route_id",
        "compiled_runtime_path",
        "compiled_intermediate_repeaters",
        "legacy_selected_path_scaffold",
        "legacy_route_id",
        "legacy_scaffold_path",
        "legacy_runtime_intermediate",
        "original_alice",
        "original_bob",
        "mapping_visibility",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


__all__ = [
    "GRAPH_MOTIF_ABSTRACTION",
    "GRAPH_MOTIF_BLOCK_REASON",
    "GraphMotifSelectionConfig",
    "run_live_q_eve_on_graph_motif",
    "select_graph_two_route_motif",
    "write_graph_motif_live_q_eve_artifacts",
]
