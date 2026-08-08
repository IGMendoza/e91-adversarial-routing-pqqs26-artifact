"""Topology Bench census for histogram learner experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import mean, median
import time
from typing import Any, Callable, Sequence

import numpy as np

from sequence_game.corpus.sqlite_store import SQLiteGraphCorpusSource
from sequence_game.graph_corpus.metrics import route_overlap_summary
from sequence_game.routing.route import Route, make_route
from sequence_game.topology import TopologyIR

from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .path_polytope import build_histogram_case_path_polytope
from .physics import (
    CACHE_ENTRY_RAM_ESTIMATE_BYTES,
    CACHE_ENTRY_RAM_ESTIMATE_SOURCE,
    histogram_cache_plan,
)
from .structure import eve_reward_embedding, route_resource_embedding


MIB = 1024**2
PHYSICS_COMPILE_SECONDS_PER_CANONICAL_ENTRY = 419.8741160375066 / 2098.0
PHYSICS_SAMPLE_SECONDS_PER_DRAW = 39.618773465568665 / 481_248.0
TIMING_ESTIMATE_SOURCE = (
    "2026-07-15 jointly structured long-case seed 42: 2,098 cold cache "
    "misses and 481,248 sampled turns/audit draws"
)


@dataclass(frozen=True)
class TopologyCensusConfig:
    """Explicit structural and resource boundaries for one corpus census."""

    game_mode: str = "attack_surface_all"
    fock_cutoff: int = 3
    evaluated_cache_size: int = 256
    enumeration_cache_limit_mib: float = 4096.0
    enumeration_dfs_expansion_limit: int = 2_000_000
    selected_case_cache_limit_mib: float = 768.0
    audit_samples_per_cell: int = 64
    selected_case_audit_draw_limit: int = 2_000_000

    def __post_init__(self) -> None:
        if self.game_mode not in {
            "attack_surface_dos",
            "attack_surface_mixed",
            "attack_surface_all",
        }:
            raise ValueError(f"unsupported game_mode {self.game_mode!r}")
        if self.fock_cutoff < 2 or self.evaluated_cache_size < 1:
            raise ValueError("fock_cutoff and evaluated_cache_size are invalid")
        if self.enumeration_cache_limit_mib <= 0.0:
            raise ValueError("enumeration_cache_limit_mib must be positive")
        if self.enumeration_dfs_expansion_limit < 1:
            raise ValueError("enumeration_dfs_expansion_limit must be positive")
        if not 0.0 < self.selected_case_cache_limit_mib <= (
            self.enumeration_cache_limit_mib
        ):
            raise ValueError(
                "selected_case_cache_limit_mib must be positive and no larger "
                "than the enumeration cache limit"
            )
        if self.audit_samples_per_cell < 2:
            raise ValueError("audit_samples_per_cell must be at least two")
        if self.selected_case_audit_draw_limit < 1:
            raise ValueError("selected_case_audit_draw_limit must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def enumeration_entry_limit(self) -> int:
        return int(
            math.floor(
                self.enumeration_cache_limit_mib
                * MIB
                / CACHE_ENTRY_RAM_ESTIMATE_BYTES
            )
        )


def census_topology_bench(
    db_path: str | Path,
    *,
    config: TopologyCensusConfig | None = None,
    corpus_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Census every assignment without constructing physical histograms."""

    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    process_start = time.process_time()
    active = config or TopologyCensusConfig()
    source = SQLiteGraphCorpusSource.open(db_path, corpus_id)
    manifest = source.load_manifest()
    graph_rows = {
        str(row["graph_id"]): row
        for row in manifest.get("graphs", [])
        if isinstance(row, dict)
    }
    ir_cache: dict[str, tuple[dict[str, Any], TopologyIR]] = {}
    assignments = sorted(
        (
            row
            for row in manifest.get("assignments", [])
            if isinstance(row, dict)
        ),
        key=lambda row: (str(row["graph_id"]), str(row["assignment_id"])),
    )
    rows: list[dict[str, Any]] = []
    for index, assignment in enumerate(assignments, start=1):
        graph_id = str(assignment["graph_id"])
        graph = graph_rows.get(graph_id)
        if graph is None:
            raise ValueError(f"manifest has no graph record for {graph_id!r}")
        if graph_id not in ir_cache:
            ir_dict = source.load_json(graph["topology_path"])
            ir_cache[graph_id] = (ir_dict, TopologyIR.from_dict(ir_dict))
        ir_dict, ir = ir_cache[graph_id]
        row = census_assignment(
            ir=ir,
            ir_dict=ir_dict,
            graph=graph,
            assignment=assignment,
            corpus_id=source.corpus_id,
            corpus_db_path=str(source.db_path),
            config=active,
        )
        rows.append(row)
        if progress_callback is not None and (
            index == 1 or index == len(assignments) or index % 25 == 0
        ):
            progress_callback(
                f"assignment {index}/{len(assignments)}: "
                f"complete={sum(item['enumeration_complete'] for item in rows)} "
                f"eligible={sum(item['selection_eligible'] for item in rows)}"
            )
    completed_at = datetime.now(timezone.utc)
    return {
        "schema": "histogram_topology_bench_census.v1",
        "corpus": {
            "corpus_id": source.corpus_id,
            "db_path": str(source.db_path),
            "manifest_hash_blake2b_128": source.stable_file_hash(
                "corpus_manifest.json"
            ),
            "scope_label": manifest.get("scope_label"),
            "graph_count": len(graph_rows),
            "assignment_count": len(assignments),
        },
        "config": active.to_dict(),
        "cache_ram_estimate": {
            "bytes_per_canonical_entry": CACHE_ENTRY_RAM_ESTIMATE_BYTES,
            "source": CACHE_ENTRY_RAM_ESTIMATE_SOURCE,
        },
        "timing_estimate": {
            "compile_seconds_per_canonical_entry": (
                PHYSICS_COMPILE_SECONDS_PER_CANONICAL_ENTRY
            ),
            "sample_seconds_per_draw": PHYSICS_SAMPLE_SECONDS_PER_DRAW,
            "source": TIMING_ESTIMATE_SOURCE,
            "boundary": (
                "physics-only planning values; learner optimization and file "
                "serialization are not included"
            ),
        },
        "census_execution": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "wall_seconds": time.perf_counter() - wall_start,
            "process_seconds": time.process_time() - process_start,
            "physical_histograms_constructed": False,
        },
        "selection_information_boundary": {
            "physical_histograms_constructed": False,
            "learner_runs_read": False,
            "service_quality_used": False,
            "payoff_fit_used": False,
            "payoff_residual_status": (
                "deferred until after the case manifest is frozen"
            ),
        },
        "host_ram_snapshot": available_ram_snapshot(),
        "summary": summarize_census(rows),
        "cases": rows,
    }


def census_assignment(
    *,
    ir: TopologyIR,
    ir_dict: dict[str, Any],
    graph: dict[str, Any],
    assignment: dict[str, Any],
    corpus_id: str,
    corpus_db_path: str,
    config: TopologyCensusConfig,
) -> dict[str, Any]:
    """Compute structural geometry and resource demand for one assignment."""

    alice = str(assignment["alice"])
    bob = str(assignment["bob"])
    enumeration = enumerate_census_routes(
        ir,
        alice,
        bob,
        config=config,
    )
    base = {
        "case_id": f"{assignment['graph_id']}::{assignment['assignment_id']}",
        "graph_id": str(assignment["graph_id"]),
        "assignment_id": str(assignment["assignment_id"]),
        "family": str(graph.get("family") or graph.get("source") or "unknown"),
        "assignment_policy": str(assignment.get("assignment_policy") or "unknown"),
        "strategy_validity": str(assignment.get("strategy_validity") or "unknown"),
        "alice": alice,
        "bob": bob,
        "graph_node_count": len(ir.nodes),
        "graph_edge_count": len(ir.edges),
        "node_cut_size": _optional_int(assignment.get("node_cut_size")),
        "edge_cut_size": _optional_int(assignment.get("edge_cut_size")),
        "stored_route_count": int(assignment.get("route_count") or 0),
        "stored_route_enumeration_truncated": bool(
            assignment.get("route_enumeration_truncated")
        ),
        **enumeration["diagnostics"],
        "payoff_residual": None,
        "payoff_residual_status": "not_measured_before_case_selection",
    }
    if not enumeration["complete"]:
        return {
            **base,
            **_unavailable_geometry(),
            "selection_eligible": False,
            "selection_exclusion_reasons": [
                f"route_enumeration:{enumeration['diagnostics']['enumeration_status']}"
            ],
        }

    routes = enumeration["routes"]
    if not routes:
        return {
            **base,
            **_unavailable_geometry(),
            "selection_eligible": False,
            "selection_exclusion_reasons": ["no_connecting_route"],
        }
    route_rows = tuple({
        **route.to_dict(),
        # Runtime Eve targets use endpoint-derived undirected labels rather
        # than source-corpus edge IDs. Keep the census case contract identical
        # to load_topology_bench_case's normalized routes.
        "edge_ids": [
            _edge_label(left, right)
            for left, right in zip(route.path, route.path[1:])
        ],
        "internal_nodes": list(route.path[1:-1]),
        "edge_lengths_m": [
            float(ir.edge_between(left, right).length_m)
            for left, right in zip(route.path, route.path[1:])
        ],
        "total_length": float(route.total_length_m),
    } for route in routes)
    case = HistogramGameCase(
        graph_id=str(assignment["graph_id"]),
        assignment_id=str(assignment["assignment_id"]),
        family=base["family"],
        alice=alice,
        bob=bob,
        ir_dict=ir_dict,
        routes=route_rows,
        assignment=dict(assignment),
        corpus_id=corpus_id,
        corpus_db_path=corpus_db_path,
        route_universe={
            "mode": "all_simple_paths",
            "path_constraint": "simple_cycle_free",
            "quality_based_route_pruning": False,
            "enumeration_complete": True,
        },
    )
    actions = case.actions(config.game_mode)
    route_embedding = route_resource_embedding(case)
    eve_embedding = eve_reward_embedding(case, actions)
    learning_config = HistogramCoLearningConfig(
        game_mode=config.game_mode,
        fock_cutoff=config.fock_cutoff,
        evaluated_cache_size=config.evaluated_cache_size,
    )
    cache = histogram_cache_plan(case, learning_config)
    if (
        cache["canonical_unique_move_count_total"]
        != enumeration["diagnostics"]["canonical_cache_entries_lower_bound"]
    ):
        raise RuntimeError(
            "census canonical-entry accounting disagrees with the runtime planner"
        )
    overlap = route_overlap_summary(list(routes))
    hop_counts = [route.hop_count for route in routes]
    lengths = [float(route.total_length_m) for route in routes]
    resource_union = {
        *(edge for route in routes for edge in route.edge_ids),
        *(node for route in routes for node in route.path[1:-1]),
    }
    incidence_mass = sum(2 * route.hop_count - 1 for route in routes)
    route_resource_density = (
        incidence_mass / (len(routes) * len(resource_union))
        if resource_union else 0.0
    )
    dag = _dag_compatibility(case)
    audit_draws = (
        cache["global_route_action_pair_count"]
        * config.audit_samples_per_cell
    )
    estimated_physics_seconds = (
        cache["canonical_unique_move_count_total"]
        * PHYSICS_COMPILE_SECONDS_PER_CANONICAL_ENTRY
        + (100_000 + audit_draws) * PHYSICS_SAMPLE_SECONDS_PER_DRAW
    )
    exclusion_reasons = []
    if len(routes) < 2:
        exclusion_reasons.append("fewer_than_two_routes")
    if cache["estimated_full_action_surface_cache_mib"] > (
        config.selected_case_cache_limit_mib
    ):
        exclusion_reasons.append("selected_cache_limit_exceeded")
    if audit_draws > config.selected_case_audit_draw_limit:
        exclusion_reasons.append("selected_audit_draw_limit_exceeded")
    if not cache["sufficient_for_declared_action_surface_without_eviction"]:
        exclusion_reasons.append("configured_lru_would_evict")
    return {
        **base,
        "route_count": len(routes),
        "route_count_lower_bound": len(routes),
        "route_hops_minimum": min(hop_counts),
        "route_hops_mean": float(mean(hop_counts)),
        "route_hops_maximum": max(hop_counts),
        "route_hop_histogram": {
            str(hops): count
            for hops, count in sorted(Counter(hop_counts).items())
        },
        "route_length_m_minimum": min(lengths),
        "route_length_m_median": float(median(lengths)),
        "route_length_m_maximum": max(lengths),
        "route_union_resource_count": len(resource_union),
        "route_resource_incidence_density": float(route_resource_density),
        "mean_internal_node_jaccard": float(
            overlap["mean_internal_node_jaccard"]
        ),
        "mean_edge_jaccard": float(overlap["mean_edge_jaccard"]),
        "max_shared_internal_nodes": int(overlap["max_shared_internal_nodes"]),
        "max_shared_edges": int(overlap["max_shared_edges"]),
        "alice_feature_rank": int(route_embedding.rank),
        "alice_compression_ratio": float(len(routes) / route_embedding.rank),
        "alice_dimension_reduction_active": bool(
            route_embedding.rank < len(routes)
        ),
        "eve_action_count": len(actions),
        "eve_feature_rank": int(eve_embedding.rank),
        "eve_compression_ratio": float(len(actions) / eve_embedding.rank),
        "nominal_route_action_cells": int(
            cache["global_route_action_pair_count"]
        ),
        "canonical_cache_entries": int(
            cache["canonical_unique_move_count_total"]
        ),
        "maximum_cache_entries_per_route": int(
            cache["maximum_canonical_moves_on_one_route"]
        ),
        "configured_cache_entries_per_route": int(
            cache["configured_entries_per_route"]
        ),
        "cache_sufficient_without_eviction": bool(
            cache["sufficient_for_declared_action_surface_without_eviction"]
        ),
        "estimated_cache_bytes": int(
            cache["estimated_full_action_surface_cache_bytes"]
        ),
        "estimated_cache_mib": float(
            cache["estimated_full_action_surface_cache_mib"]
        ),
        "audit_samples_per_cell": config.audit_samples_per_cell,
        "audit_draws_per_seed": int(audit_draws),
        "estimated_first_configuration_physics_seconds": float(
            estimated_physics_seconds
        ),
        **dag,
        "selection_eligible": not exclusion_reasons,
        "selection_exclusion_reasons": exclusion_reasons,
    }


def enumerate_census_routes(
    ir: TopologyIR,
    source: str,
    target: str,
    *,
    config: TopologyCensusConfig,
) -> dict[str, Any]:
    """Enumerate routes or stop when an explicit resource bound is proven."""

    for node in (source, target):
        if node not in ir.nodes:
            raise ValueError(f"unknown route endpoint {node!r}")
    adjacency = {
        node: sorted(neighbors)
        for node, neighbors in ir.adjacency().items()
    }
    routes: list[Route] = []
    canonical_entries = 0
    expansions = 0
    stop_status: str | None = None

    def dfs(node: str, path: list[str]) -> None:
        nonlocal canonical_entries, expansions, stop_status
        if stop_status is not None:
            return
        expansions += 1
        if expansions > config.enumeration_dfs_expansion_limit:
            stop_status = "dfs_expansion_limit_reached"
            return
        if node == target:
            route = make_route(ir, tuple(path))
            routes.append(route)
            canonical_entries += canonical_moves_for_route(
                route.hop_count,
                config.game_mode,
            )
            if canonical_entries > config.enumeration_entry_limit:
                stop_status = "cache_lower_bound_exceeds_enumeration_limit"
            return
        for neighbor in adjacency[node]:
            if neighbor not in path:
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()
                if stop_status is not None:
                    return

    dfs(source, [source])
    complete = stop_status is None
    status = "complete" if complete else str(stop_status)
    return {
        "complete": complete,
        "routes": tuple(routes) if complete else (),
        "diagnostics": {
            "enumeration_complete": complete,
            "enumeration_status": status,
            "route_path_constraint": "simple_cycle_free",
            "quality_based_route_pruning": False,
            "route_count": len(routes) if complete else None,
            "route_count_lower_bound": len(routes),
            "canonical_cache_entries_lower_bound": canonical_entries,
            "estimated_cache_mib_lower_bound": (
                canonical_entries * CACHE_ENTRY_RAM_ESTIMATE_BYTES / MIB
            ),
            "dfs_expansions": expansions,
            "enumeration_cache_entry_limit": config.enumeration_entry_limit,
            "enumeration_cache_limit_mib": config.enumeration_cache_limit_mib,
            "enumeration_dfs_expansion_limit": (
                config.enumeration_dfs_expansion_limit
            ),
            "incomplete_universe_used_for_learning": False,
        },
    }


def canonical_moves_for_route(hop_count: int, game_mode: str) -> int:
    """Exact route-local canonical move count for the declared action model."""

    if hop_count < 1:
        raise ValueError("hop_count must be positive")
    if game_mode == "attack_surface_dos":
        return 3 * hop_count - 1
    if game_mode in {"attack_surface_mixed", "attack_surface_all"}:
        return 6 * hop_count - 3
    raise ValueError(f"unsupported game_mode {game_mode!r}")


def summarize_census(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["enumeration_complete"]]
    measured = [row for row in complete if row.get("route_count") is not None]
    eligible = [row for row in complete if row["selection_eligible"]]
    return {
        "assignment_count": len(rows),
        "enumeration_complete_count": len(complete),
        "enumeration_incomplete_count": len(rows) - len(complete),
        "enumeration_status_counts": dict(sorted(Counter(
            str(row["enumeration_status"]) for row in rows
        ).items())),
        "selection_eligible_count": len(eligible),
        "selection_exclusion_reason_counts": dict(sorted(Counter(
            reason
            for row in rows
            for reason in row["selection_exclusion_reasons"]
        ).items())),
        "strategy_validity_counts": dict(sorted(Counter(
            str(row["strategy_validity"]) for row in rows
        ).items())),
        "complete_route_count_quantiles": _quantiles(
            [float(row["route_count"]) for row in measured]
        ),
        "eligible_route_count_quantiles": _quantiles(
            [float(row["route_count"]) for row in eligible]
        ),
        "eligible_alice_compression_quantiles": _quantiles(
            [float(row["alice_compression_ratio"]) for row in eligible]
        ),
        "eligible_cache_mib_quantiles": _quantiles(
            [float(row["estimated_cache_mib"]) for row in eligible]
        ),
        "eligible_audit_draw_quantiles": _quantiles(
            [float(row["audit_draws_per_seed"]) for row in eligible]
        ),
        "eligible_dag_compatible_count": sum(
            bool(row["exact_bounded_dag_compatible"]) for row in eligible
        ),
    }


def available_ram_snapshot() -> dict[str, Any]:
    """Read Linux memory availability without influencing case selection."""

    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "total_gib": (
            values["MemTotal"] / 1024**3 if "MemTotal" in values else None
        ),
        "available_gib": (
            values["MemAvailable"] / 1024**3
            if "MemAvailable" in values else None
        ),
        "used_for_case_selection": False,
    }


def _dag_compatibility(case: HistogramGameCase) -> dict[str, Any]:
    try:
        adapter = build_histogram_case_path_polytope(case)
    except ValueError as exc:
        return {
            "exact_bounded_dag_compatible": False,
            "dag_dimension": None,
            "dag_path_count": None,
            "dag_incompatibility_reason": str(exc),
        }
    return {
        "exact_bounded_dag_compatible": True,
        "dag_dimension": int(adapter.polytope.dimension),
        "dag_path_count": int(adapter.polytope.path_count),
        "dag_incompatibility_reason": None,
    }


def _unavailable_geometry() -> dict[str, Any]:
    keys = (
        "route_count",
        "route_hops_minimum",
        "route_hops_mean",
        "route_hops_maximum",
        "route_hop_histogram",
        "route_length_m_minimum",
        "route_length_m_median",
        "route_length_m_maximum",
        "route_union_resource_count",
        "route_resource_incidence_density",
        "mean_internal_node_jaccard",
        "mean_edge_jaccard",
        "max_shared_internal_nodes",
        "max_shared_edges",
        "alice_feature_rank",
        "alice_compression_ratio",
        "alice_dimension_reduction_active",
        "eve_action_count",
        "eve_feature_rank",
        "eve_compression_ratio",
        "nominal_route_action_cells",
        "canonical_cache_entries",
        "maximum_cache_entries_per_route",
        "configured_cache_entries_per_route",
        "cache_sufficient_without_eviction",
        "estimated_cache_bytes",
        "estimated_cache_mib",
        "audit_samples_per_cell",
        "audit_draws_per_seed",
        "estimated_first_configuration_physics_seconds",
        "exact_bounded_dag_compatible",
        "dag_dimension",
        "dag_path_count",
        "dag_incompatibility_reason",
    )
    return {key: None for key in keys}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _edge_label(left: str, right: str) -> str:
    first, second = sorted((str(left), str(right)))
    return f"{first}-{second}"


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "q25", "median", "q75", "maximum")}
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }
