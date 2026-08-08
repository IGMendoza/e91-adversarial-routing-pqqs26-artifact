"""Manifest-locked corpus-scale histogram learner comparisons."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sequence_game.corpus.sqlite_store import SQLiteGraphCorpusSource
from sequence_game.topology import TopologyIR

from .calibration_bank import build_histogram_payoff_bank
from .calibration_game import (
    CalibrationGameOracle,
    build_calibration_game_oracle,
    score_calibration_behavior,
)
from .config import (
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_TSALLIS_INF,
    DEFAULT_CORPUS_DB_PATH,
    EVE_LEARNER_TSALLIS_INF,
    HistogramCoLearningConfig,
    REPO_ROOT,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
)
from .corpus import (
    HistogramGameCase,
    build_topology_bench_case_for_endpoints,
    load_topology_bench_case,
)
from .learning import run_histogram_colearning
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .topology_census import TopologyCensusConfig, enumerate_census_routes


CORPUS_SCALE_MANIFEST_SCHEMA = "histogram_corpus_scale_manifest.v1"
CORPUS_SCALE_RESULT_SCHEMA = "histogram_corpus_scale_result.v1"
CORPUS_SCALE_GRAPH_CHECKPOINT_SCHEMA = "histogram_corpus_scale_graph_checkpoint.v2"
CORPUS_SCALE_EXECUTION_PLAN_SCHEMA = "histogram_corpus_scale_execution_plan.v2"
DEFAULT_CENSUS_PATH = REPO_ROOT / "docs" / "topology_bench_step6_census" / "census.json"
DEFAULT_CORPUS_SCALE_OUTPUT_DIR = REPO_ROOT / "runs" / "histogram_corpus_500k"

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CorpusScalePlan:
    """Compute, endpoint-selection, and scoring contract for one corpus run."""

    turns: int = 500_000
    final_window: int = 50_000
    record_every: int = 10_000
    case_cache_limit_mib: float = 768.0
    maximum_case_cache_limit_mib: float = 6_144.0
    enumeration_dfs_expansion_limit: int = 2_000_000
    audit_samples_per_cell_per_split: int = 16
    root_seed: int = 20_260_716
    payoff_bank_root_seed: int = 20_260_717
    workers_per_graph: int = 2
    evaluated_cache_size: int = 256
    requested_pairs_per_turn: int = 1_200
    age_points_per_edge: int = 8

    def __post_init__(self) -> None:
        if self.turns < 1 or self.final_window < 1 or self.record_every < 1:
            raise ValueError("turn, final-window, and record intervals must be positive")
        if self.case_cache_limit_mib <= 0.0:
            raise ValueError("case_cache_limit_mib must be positive")
        if self.maximum_case_cache_limit_mib < self.case_cache_limit_mib:
            raise ValueError(
                "maximum_case_cache_limit_mib must be at least the preferred limit"
            )
        if self.audit_samples_per_cell_per_split < 2:
            raise ValueError("audit samples per cell must be at least two")
        if self.workers_per_graph != 2:
            raise ValueError("this comparison requires exactly two workers per graph")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusLearnerPair:
    label: str
    alice_family: str
    eve_family: str
    config_overrides: dict[str, Any]
    calibration_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "alice_family": self.alice_family,
            "eve_family": self.eve_family,
            "config_overrides": dict(self.config_overrides),
            "calibration_source": self.calibration_source,
        }


CORPUS_SCALE_PAIRINGS = (
    CorpusLearnerPair(
        label="tsallis_rv_eta1_vs_tsallis_rv_eta1",
        alice_family="tsallis_inf",
        eve_family="tsallis_inf",
        config_overrides={
            "alice_learner_mode": ALICE_LEARNER_TSALLIS_INF,
            "alice_tsallis_inf_estimator": "rv",
            "alice_tsallis_inf_eta_scale": 1.0,
            "eve_learner_mode": EVE_LEARNER_TSALLIS_INF,
            "eve_tsallis_inf_estimator": "rv",
            "eve_tsallis_inf_eta_scale": 1.0,
        },
        calibration_source=(
            "2026-07-15 five-topology Tsallis refinement selected RV eta-scale "
            "1 for both roles"
        ),
    ),
    CorpusLearnerPair(
        label="linear_goptimal_eta2_vs_tsallis_rv_eta1",
        alice_family="linear_exp2",
        eve_family="tsallis_inf",
        config_overrides={
            "alice_learner_mode": ALICE_LEARNER_LINEAR_ROUTE_EXP2,
            "alice_exp3_eta_c": 2.0,
            "structured_exploration_mode": STRUCTURED_EXPLORATION_G_OPTIMAL,
            "eve_learner_mode": EVE_LEARNER_TSALLIS_INF,
            "eve_tsallis_inf_estimator": "rv",
            "eve_tsallis_inf_eta_scale": 1.0,
        },
        calibration_source=(
            "2026-07-15 five-topology Step 7 selected LinearExp2 G-optimal "
            "eta-scale 2 for Alice; the same cross-play found Tsallis-INF RV "
            "eta-scale 1 to be the strongest tested Eve family"
        ),
    ),
)


_POLICY_ORDER = {
    "best_route_diverse": 0,
    "random_valid_pair": 1,
    "long_distance_pair": 2,
    "strongest_edge_bottleneck": 3,
    "strongest_node_bottleneck": 4,
}
_GRAPH_WORKER_CONTEXT: dict[str, Any] | None = None


def build_corpus_scale_manifest(
    *,
    db_path: str | Path = DEFAULT_CORPUS_DB_PATH,
    census_path: str | Path = DEFAULT_CENSUS_PATH,
    plan: CorpusScalePlan | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Select one complete, outcome-blind Alice--Bob case per real topology."""

    active = plan or CorpusScalePlan()
    source = SQLiteGraphCorpusSource.open(db_path)
    corpus_manifest = source.load_manifest()
    census_target = Path(census_path)
    census = json.loads(census_target.read_text(encoding="utf-8"))
    census_rows: dict[str, list[dict[str, Any]]] = {}
    for row in census.get("cases", []):
        census_rows.setdefault(str(row["graph_id"]), []).append(row)
    assignment_rows = {
        str(row["assignment_id"]): row
        for row in corpus_manifest.get("assignments", [])
        if isinstance(row, dict)
    }
    graph_rows = sorted(
        (
            row for row in corpus_manifest.get("graphs", [])
            if isinstance(row, dict)
        ),
        key=lambda row: str(row["graph_id"]),
    )
    cases: list[dict[str, Any]] = []
    for index, graph in enumerate(graph_rows, start=1):
        graph_id = str(graph["graph_id"])
        selected = _select_graph_case(
            source=source,
            graph=graph,
            census_rows=census_rows.get(graph_id, []),
            assignment_rows=assignment_rows,
            plan=active,
        )
        cases.append(selected)
        _notify(
            progress_callback,
            f"manifest {index}/{len(graph_rows)} {selected['family']}: "
            f"{selected['route_count']} routes, {selected['eve_action_count']} "
            f"Eve actions, {selected['estimated_cache_mib']:.1f} MiB",
        )
    payload = {
        "schema": CORPUS_SCALE_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "corpus_id": source.corpus_id,
            "db_path": str(source.db_path),
            "normalized_physical_lengths": True,
            "source_scope_label": corpus_manifest.get("scope_label"),
            "manifest_hash_blake2b_128": source.stable_file_hash(
                "corpus_manifest.json"
            ),
            "graph_count": len(graph_rows),
        },
        "census": {
            "path": str(census_target),
            "hash_blake2b_128": _file_digest(census_target),
        },
        "plan": active.to_dict(),
        "pairings": [row.to_dict() for row in CORPUS_SCALE_PAIRINGS],
        "endpoint_selection": {
            "outcome_information_used": False,
            "physics_reward_information_used": False,
            "route_quality_prior_used": False,
            "route_pruning_used": False,
            "route_universe": "every simple cycle-free Alice--Bob path",
            "existing_assignment_order": list(_POLICY_ORDER),
            "resource_boundary": (
                "prefer the first declared Topology Bench assignment whose "
                "complete action surface fits the preferred case cache envelope; "
                "if none fits, search all endpoint pairs in deterministic local-"
                "degree order, first under the preferred envelope and then under "
                "the declared maximum envelope"
            ),
            "single_route_handling": (
                "retain a complete single-route case only when no in-scope "
                "multi-route edge case can be enumerated inside the envelope"
            ),
        },
        "evaluation_partition_loaded": True,
        "evaluation_partition_status": (
            "the previously held-out graphs are included by explicit corpus-wide request"
        ),
        "cases": cases,
    }
    payload["selection_digest_blake2b_128"] = _selection_digest(payload)
    return payload


def write_corpus_scale_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_corpus_scale_case(
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> HistogramGameCase:
    db_path = str(manifest["corpus"]["db_path"])
    if entry["selection_source"] == "topology_bench_assignment":
        return load_topology_bench_case(
            db_path,
            corpus_id=str(manifest["corpus"]["corpus_id"]),
            graph_id=str(entry["graph_id"]),
            assignment_id=str(entry["assignment_id"]),
        )
    source = SQLiteGraphCorpusSource.open(
        db_path,
        str(manifest["corpus"]["corpus_id"]),
    )
    ir_dict = source.load_json(entry["topology_path"])
    return build_topology_bench_case_for_endpoints(
        ir_dict=ir_dict,
        graph_id=str(entry["graph_id"]),
        family=str(entry["family"]),
        alice=str(entry["alice"]),
        bob=str(entry["bob"]),
        corpus_id=source.corpus_id,
        corpus_db_path=str(source.db_path),
        assignment_id=str(entry["assignment_id"]),
        assignment={
            "assignment_policy": "deterministic_endpoint_pair_search",
            "selection_source": "corpus_scale_endpoint_rule",
        },
    )


def run_corpus_scale_experiment(
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_CORPUS_SCALE_OUTPUT_DIR,
    progress_callback: ProgressCallback | None = None,
    requested_graph_workers: int = 6,
    available_memory_mib: float | None = None,
    memory_reserve_mib: float = 8_192.0,
    per_graph_overhead_mib: float = 1_024.0,
) -> dict[str, Any]:
    """Run RAM-bounded graph processes with one on-demand pool per graph.

    Each graph-owner process runs the two learner pairings concurrently as
    threads against the same mutable :class:`HistogramPhysicsPool`.  The pool
    starts with only the per-route no-attack calibration entries required by
    the public reward normalization.  Learner-selected route/action cells then
    populate it on demand.  The complete independent payoff audit runs only
    after both learners finish and fills any remaining cells.

    Parallelism across graph-owner processes supplies concurrent cold
    compilation without duplicating one graph's cache.  The initial resident
    upper bound is the sum of the largest selected full-surface estimates plus
    an explicit per-process overhead allowance.
    """

    _validate_manifest(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "graph_checkpoints"
    progress_dir = output / "progress"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    plan = CorpusScalePlan(**dict(manifest["plan"]))
    pairings = tuple(
        CorpusLearnerPair(
            label=str(row["label"]),
            alice_family=str(row["alice_family"]),
            eve_family=str(row["eve_family"]),
            config_overrides=dict(row["config_overrides"]),
            calibration_source=str(row["calibration_source"]),
        )
        for row in manifest["pairings"]
    )
    expected = len(manifest["cases"]) * len(pairings)
    checkpoints = _load_graph_checkpoints(checkpoint_dir, manifest=manifest)
    existing = _load_jsonl(output / "results.jsonl")
    rows_by_key = {str(row["run_key"]): row for row in existing}
    physics_rows = _load_jsonl(output / "physics_compilation.jsonl")
    physics_by_graph = {str(row["graph_id"]): row for row in physics_rows}
    for checkpoint in checkpoints.values():
        for row in checkpoint["results"]:
            rows_by_key[str(row["run_key"])] = row
        physics = checkpoint["physics"]
        physics_by_graph[str(physics["graph_id"])] = physics
    _validate_existing_rows(
        rows_by_key,
        manifest=manifest,
        pairings=pairings,
        plan=plan,
    )
    ordered_cases = sorted(
        manifest["cases"],
        key=lambda row: (-float(row["estimated_cache_mib"]), str(row["graph_id"])),
    )
    pending = [
        entry
        for entry in ordered_cases
        if str(entry["graph_id"]) not in checkpoints
    ]
    available = (
        _available_memory_mib()
        if available_memory_mib is None
        else float(available_memory_mib)
    )
    execution = build_corpus_execution_plan(
        ordered_cases,
        pending_graph_ids={str(row["graph_id"]) for row in pending},
        requested_graph_workers=requested_graph_workers,
        available_memory_mib=available,
        memory_reserve_mib=memory_reserve_mib,
        per_graph_overhead_mib=per_graph_overhead_mib,
    )
    _write_json_atomic(output / "execution_plan.json", execution)
    _notify(
        progress_callback,
        "execution plan: "
        f"{execution['selected_graph_workers']} graph processes x "
        f"{plan.workers_per_graph} shared-pool learner threads; "
        f"initial resident upper bound {execution['initial_resident_upper_bound_mib']:.1f} MiB; "
        f"{len(pending)} graphs pending",
    )
    if pending:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("on-demand shared-pool corpus execution requires fork")
        global _GRAPH_WORKER_CONTEXT
        if _GRAPH_WORKER_CONTEXT is not None:
            raise RuntimeError("corpus graph-worker context is already active")
        _GRAPH_WORKER_CONTEXT = {
            "manifest": dict(manifest),
            "plan": plan,
            "pairings": pairings,
            "output_dir": str(output),
            "graph_workers": int(execution["selected_graph_workers"]),
        }
        context = mp.get_context("fork")
        try:
            with context.Pool(
                processes=int(execution["selected_graph_workers"]),
                maxtasksperchild=1,
            ) as pool:
                iterator = pool.imap_unordered(
                    _run_corpus_graph_worker,
                    pending,
                    chunksize=1,
                )
                remaining = len(pending)
                while remaining:
                    try:
                        checkpoint = iterator.next(timeout=30.0)
                    except mp.TimeoutError:
                        _notify(
                            progress_callback,
                            _progress_snapshot(progress_dir, remaining=remaining),
                        )
                        continue
                    graph_id = str(checkpoint["graph_id"])
                    checkpoints[graph_id] = checkpoint
                    for row in checkpoint["results"]:
                        rows_by_key[str(row["run_key"])] = row
                    physics = checkpoint["physics"]
                    physics_by_graph[graph_id] = physics
                    remaining -= 1
                    _materialize_corpus_rows(
                        output,
                        rows_by_key=rows_by_key,
                        physics_by_graph=physics_by_graph,
                    )
                    _notify(
                        progress_callback,
                        f"graph complete {len(checkpoints)}/{len(ordered_cases)}: "
                        f"{checkpoint['family']}; results {len(rows_by_key)}/{expected}",
                    )
        finally:
            _GRAPH_WORKER_CONTEXT = None
    _materialize_corpus_rows(
        output,
        rows_by_key=rows_by_key,
        physics_by_graph=physics_by_graph,
    )
    rows = list(rows_by_key.values())
    summary = summarize_corpus_scale_results(manifest, rows, physics_by_graph)
    _write_json_atomic(output / "summary.json", summary)
    return summary


def summarize_corpus_scale_results(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    physics_by_graph: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pair_labels = [str(row["label"]) for row in manifest["pairings"]]
    aggregate = []
    for label in pair_labels:
        selected = [row for row in rows if row["pairing_label"] == label]
        metrics = {}
        for key in (
            "exploitability",
            "alice_external_regret_per_turn",
            "eve_external_regret_per_turn",
            "mean_alice_reward",
            "mean_bell_qualified_rate_hz",
            "false_signal_rate",
            "false_signal_policy_total_variation",
            "turns_per_second",
        ):
            values = np.asarray([float(row[key]) for row in selected], dtype=float)
            metrics[key] = _distribution(values)
        aggregate.append({
            "pairing_label": label,
            "graph_count": len(selected),
            "metrics": metrics,
        })
    return {
        "schema": "histogram_corpus_scale_summary.v1",
        "selection_digest_blake2b_128": manifest[
            "selection_digest_blake2b_128"
        ],
        "expected_run_count": len(manifest["cases"]) * len(pair_labels),
        "run_count": len(rows),
        "complete": len(rows) == len(manifest["cases"]) * len(pair_labels),
        "graph_count": len({str(row["graph_id"]) for row in rows}),
        "pairings": aggregate,
        "physics_compiled_graph_count": len(physics_by_graph or {}),
        "evaluation_partition_loaded": True,
        "rows": list(rows),
    }


def _select_graph_case(
    *,
    source: SQLiteGraphCorpusSource,
    graph: Mapping[str, Any],
    census_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Mapping[str, Mapping[str, Any]],
    plan: CorpusScalePlan,
) -> dict[str, Any]:
    graph_id = str(graph["graph_id"])
    family = str(graph.get("family") or graph.get("source") or "unknown")
    complete_multi = [
        row for row in census_rows
        if bool(row.get("enumeration_complete"))
        and int(row.get("route_count") or 0) >= 2
        and float(row.get("estimated_cache_mib") or float("inf"))
        <= plan.case_cache_limit_mib
    ]
    complete_multi.sort(key=lambda row: (
        _POLICY_ORDER.get(str(row.get("assignment_policy")), 99),
        str(row["assignment_id"]),
    ))
    for row in complete_multi:
        assignment = assignment_rows[str(row["assignment_id"])]
        try:
            case = load_topology_bench_case(
                source.db_path,
                corpus_id=source.corpus_id,
                graph_id=graph_id,
                assignment_id=str(row["assignment_id"]),
            )
        except ValueError:
            continue
        cache = histogram_cache_plan(
            case,
            _base_config(plan, plan.root_seed, case=case),
        )
        if len(case.routes) < 2 or float(
            cache["estimated_full_action_surface_cache_mib"]
        ) > plan.case_cache_limit_mib:
            continue
        selection_reason = "first_complete_multi_route_assignment_in_declared_policy_order"
        return _manifest_case_row(
            case,
            graph=graph,
            topology_path=str(graph["topology_path"]),
            selection_source="topology_bench_assignment",
            selection_reason=selection_reason,
            assignment_policy=str(assignment.get("assignment_policy") or "unknown"),
            plan=plan,
        )

    ir_dict = source.load_json(graph["topology_path"])
    ir = TopologyIR.from_dict(ir_dict)
    endpoint = _select_complete_endpoint_pair(ir, plan=plan)
    if endpoint is not None:
        alice, bob, route_count, selection_tier = endpoint
        assignment_id = f"corpus_scale_endpoint_pair__{alice}__{bob}"
        case = build_topology_bench_case_for_endpoints(
            ir_dict=ir_dict,
            graph_id=graph_id,
            family=family,
            alice=alice,
            bob=bob,
            corpus_id=source.corpus_id,
            corpus_db_path=str(source.db_path),
            assignment_id=assignment_id,
            assignment={
                "assignment_policy": "deterministic_endpoint_pair_search",
                "enumerated_route_count": route_count,
                "selection_tier": selection_tier,
            },
        )
        return _manifest_case_row(
            case,
            graph=graph,
            topology_path=str(graph["topology_path"]),
            selection_source="corpus_scale_endpoint_rule",
            selection_reason=(
                "no declared multi-route assignment fit the preferred cache "
                "envelope; selected the first complete pair in deterministic "
                f"all-pairs order at resource tier {selection_tier}"
            ),
            assignment_policy="deterministic_endpoint_pair_search",
            plan=plan,
        )

    complete_single = [
        row for row in census_rows
        if bool(row.get("enumeration_complete"))
        and int(row.get("route_count") or 0) == 1
        and float(row.get("estimated_cache_mib") or float("inf"))
        <= plan.case_cache_limit_mib
    ]
    complete_single.sort(key=lambda row: (
        _POLICY_ORDER.get(str(row.get("assignment_policy")), 99),
        str(row["assignment_id"]),
    ))
    for row in complete_single:
        assignment = assignment_rows[str(row["assignment_id"])]
        try:
            case = load_topology_bench_case(
                source.db_path,
                corpus_id=source.corpus_id,
                graph_id=graph_id,
                assignment_id=str(row["assignment_id"]),
            )
        except ValueError:
            continue
        if len(case.routes) != 1:
            continue
        return _manifest_case_row(
            case,
            graph=graph,
            topology_path=str(graph["topology_path"]),
            selection_source="topology_bench_assignment",
            selection_reason=(
                "no complete multi-route edge case fit the declared envelope; "
                "retained the complete single-route corpus assignment"
            ),
            assignment_policy=str(assignment.get("assignment_policy") or "unknown"),
            plan=plan,
        )
    raise RuntimeError(
        f"graph {graph_id} has no complete in-scope Alice--Bob case inside "
        f"the declared {plan.maximum_case_cache_limit_mib:.1f} MiB maximum envelope"
    )


def _select_complete_endpoint_pair(
    ir: TopologyIR,
    *,
    plan: CorpusScalePlan,
) -> tuple[str, str, int, str] | None:
    adjacency = ir.adjacency()
    bridges = _bridges(ir)
    candidates: list[tuple[int, int, int, int, str, str]] = []
    edge_pairs = {
        frozenset((str(edge.u), str(edge.v))) for edge in ir.edges
    }
    nodes = sorted(ir.nodes)
    for left_index, alice in enumerate(nodes):
        for bob in nodes[left_index + 1:]:
            pair = frozenset((alice, bob))
            # Cycle edges are tried first because they guarantee a direct path
            # and at least one graph-theoretic alternate path. Remaining pairs
            # then extend the search across sparse cuts.
            if pair in edge_pairs and pair not in bridges:
                pair_class = 0
            elif pair not in edge_pairs:
                pair_class = 1
            else:
                pair_class = 2
            candidates.append((
                pair_class,
                max(len(adjacency[alice]), len(adjacency[bob])),
                len(adjacency[alice]) + len(adjacency[bob]),
                len(adjacency[alice]) * len(adjacency[bob]),
                alice,
                bob,
            ))
    tiers = (
        ("preferred", plan.case_cache_limit_mib),
        ("maximum", plan.maximum_case_cache_limit_mib),
    )
    for tier, cache_limit_mib in tiers:
        config = TopologyCensusConfig(
            enumeration_cache_limit_mib=cache_limit_mib,
            enumeration_dfs_expansion_limit=plan.enumeration_dfs_expansion_limit,
            selected_case_cache_limit_mib=cache_limit_mib,
            audit_samples_per_cell=plan.audit_samples_per_cell_per_split,
        )
        for (
            _pair_class,
            _max_degree,
            _degree_sum,
            _degree_product,
            alice,
            bob,
        ) in sorted(candidates):
            enumeration = enumerate_census_routes(ir, alice, bob, config=config)
            routes = enumeration["routes"]
            if enumeration["complete"] and len(routes) >= 2:
                return alice, bob, len(routes), tier
    return None


def _bridges(ir: TopologyIR) -> set[frozenset[str]]:
    adjacency = ir.adjacency()
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    result: set[frozenset[str]] = set()
    clock = 0

    def visit(node: str, parent: str | None) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = low[node] = clock
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    result.add(frozenset((node, neighbor)))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(ir.nodes):
        if node not in discovery:
            visit(node, None)
    return result


def _manifest_case_row(
    case: HistogramGameCase,
    *,
    graph: Mapping[str, Any],
    topology_path: str,
    selection_source: str,
    selection_reason: str,
    assignment_policy: str,
    plan: CorpusScalePlan,
) -> dict[str, Any]:
    config = _base_config(plan, plan.root_seed, case=case)
    cache = histogram_cache_plan(case, config)
    if not cache["sufficient_for_declared_action_surface_without_eviction"]:
        raise RuntimeError(f"cache capacity is insufficient for {case.family}")
    if cache["estimated_full_action_surface_cache_mib"] > plan.maximum_case_cache_limit_mib:
        raise RuntimeError(f"selected case exceeds cache envelope for {case.family}")
    actions = case.actions(config.game_mode)
    return {
        "case_id": f"{case.graph_id}::{case.assignment_id}",
        "graph_id": case.graph_id,
        "family": case.family,
        "topology_path": topology_path,
        "assignment_id": case.assignment_id,
        "assignment_policy": assignment_policy,
        "selection_source": selection_source,
        "selection_reason": selection_reason,
        "alice": case.alice,
        "bob": case.bob,
        "graph_node_count": len(case.ir_dict.get("nodes", {})),
        "graph_edge_count": len(case.ir_dict.get("edges", [])),
        "route_count": len(case.routes),
        "route_ids": [str(route["route_id"]) for route in case.routes],
        "route_ids_digest_blake2b_128": _object_digest(
            [str(route["route_id"]) for route in case.routes]
        ),
        "eve_action_count": len(actions),
        "eve_action_ids": [str(action["id"]) for action in actions],
        "canonical_cache_entries": cache["canonical_unique_move_count_total"],
        "estimated_cache_mib": cache["estimated_full_action_surface_cache_mib"],
        "nominal_route_action_cells": cache["global_route_action_pair_count"],
        "route_universe": dict(case.route_universe),
        "source_graph": dict(graph),
    }


def _run_corpus_graph_worker(entry: Mapping[str, Any]) -> dict[str, Any]:
    context = _GRAPH_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("corpus graph-worker context is not initialized")
    manifest = context["manifest"]
    plan: CorpusScalePlan = context["plan"]
    pairings: tuple[CorpusLearnerPair, ...] = context["pairings"]
    output = Path(context["output_dir"])
    graph_id = str(entry["graph_id"])
    progress = _GraphProgress(
        output / "progress" / f"{graph_id}.json",
        graph_id=graph_id,
        family=str(entry["family"]),
    )
    started = time.perf_counter()
    seed = _graph_seed(plan.root_seed, graph_id)
    case = load_corpus_scale_case(entry, manifest)
    base_config = _base_config(plan, seed, case=case)
    actual_plan = histogram_cache_plan(case, base_config)
    _assert_case_contract(entry, case, actual_plan, plan)

    progress.update(
        phase="baseline_calibration",
        message=(
            f"building {len(case.routes)} required no-attack route calibrations; "
            f"full surface has {actual_plan['canonical_unique_move_count_total']} "
            "canonical entries and remains on demand"
        ),
    )
    initialization_started = time.perf_counter()
    physics = HistogramPhysicsPool(case, base_config)
    initialization_seconds = time.perf_counter() - initialization_started
    cache_after_initialization = physics.cache_state()

    tasks = [
        (_run_key(graph_id, pair.label, seed), pair, seed)
        for pair in pairings
    ]
    progress.update(
        phase="online_learning",
        message="two learner pairings started against one mutable on-demand cache",
    )
    learning_started = time.perf_counter()
    live_rows: list[tuple[tuple[str, CorpusLearnerPair, int], dict[str, Any]]] = []
    with ThreadPoolExecutor(
        max_workers=plan.workers_per_graph,
        thread_name_prefix=f"{case.family}-learner",
    ) as executor:
        futures = [
            executor.submit(
                _run_pairing_live,
                task,
                case=case,
                physics=physics,
                base_config=base_config,
                progress_callback=(
                    lambda message, label=task[1].label: progress.learner(
                        label,
                        message,
                    )
                ),
            )
            for task in tasks
        ]
        for task, future in zip(tasks, futures):
            live_rows.append((task, future.result()))
    learning_seconds = time.perf_counter() - learning_started
    cache_after_learning = physics.cache_state()

    progress.update(
        phase="independent_audit",
        message=(
            "online learning complete; filling remaining cache entries through "
            "the disjoint equal-coverage payoff audit"
        ),
    )
    audit_started = time.perf_counter()
    bank = build_histogram_payoff_bank(
        case,
        base_config,
        samples_per_cell=plan.audit_samples_per_cell_per_split,
        root_seed=_graph_seed(plan.payoff_bank_root_seed, graph_id),
        physics=physics,
        progress_callback=progress.audit,
        selection_digest_blake2b_128=str(
            manifest["selection_digest_blake2b_128"]
        ),
        evaluation_partition_loaded=True,
        sampling_purpose="corpus_scale_strategic_scoring",
        sampling_design=(
            "two disjoint equal-coverage histogram transcript banks constructed "
            "after both online learners; the audit split defines the strategic "
            "scoring matrix, the feedback split remains available for "
            "reproducibility checks, and online learning used fresh on-demand "
            "compiled-histogram transcript draws"
        ),
    )
    audit_seconds = time.perf_counter() - audit_started
    oracle = build_calibration_game_oracle(bank)
    cache_after_audit = physics.cache_state()
    diagnostics = physics.diagnostics()
    _assert_compiled_surface(diagnostics)
    _assert_complete_audit_surface(
        diagnostics,
        expected_canonical_entries=int(
            actual_plan["canonical_unique_move_count_total"]
        ),
    )

    results = [
        _score_pairing_live(
            task,
            live=live,
            case=case,
            bank=bank,
            oracle=oracle,
            selection_digest=str(manifest["selection_digest_blake2b_128"]),
            graph_worker_count=int(context["graph_workers"]),
        )
        for task, live in live_rows
    ]
    physics_row = {
        "schema": "histogram_corpus_scale_physics_compilation.v2",
        "graph_id": graph_id,
        "family": case.family,
        "assignment_id": case.assignment_id,
        "elapsed_seconds": time.perf_counter() - started,
        "baseline_calibration_seconds": initialization_seconds,
        "concurrent_online_learning_seconds": learning_seconds,
        "post_learning_audit_seconds": audit_seconds,
        "cache_plan": actual_plan,
        "cache_population_order": (
            "required_no_attack_route_calibration_then_two_concurrent_online_"
            "learners_then_complete_independent_audit"
        ),
        "cache_mutation_model": (
            "one_mutable_on_demand_pool_owned_by_one_graph_process_and_shared_"
            "by_two_learner_threads"
        ),
        "eager_full_surface_compilation": False,
        "cache_after_initialization": cache_after_initialization,
        "cache_after_online_learning": cache_after_learning,
        "cache_after_independent_audit": cache_after_audit,
        "actual_cache_memory": diagnostics["cache_memory"],
        "physics_stats": diagnostics["stats"],
        "payoff_bank_content_digest_blake2b_128": bank.content_digest(),
        "audit_samples_per_cell_per_split": (
            plan.audit_samples_per_cell_per_split
        ),
        "selection_digest_blake2b_128": manifest[
            "selection_digest_blake2b_128"
        ],
    }
    checkpoint = {
        "schema": CORPUS_SCALE_GRAPH_CHECKPOINT_SCHEMA,
        "selection_digest_blake2b_128": manifest[
            "selection_digest_blake2b_128"
        ],
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "physics": physics_row,
    }
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress.update(
        phase="complete",
        message=(
            f"two 500k results and independent audit complete in "
            f"{time.perf_counter() - started:.1f} s"
        ),
    )
    del oracle, bank, physics, case
    gc.collect()
    return checkpoint


def _run_pairing_live(
    task: tuple[str, CorpusLearnerPair, int],
    *,
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    base_config: HistogramCoLearningConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _run_key_value, pairing, seed = task
    config = replace(base_config, seed=seed, **pairing.config_overrides)
    return run_histogram_colearning(
        case,
        config,
        physics=physics,
        progress_callback=progress_callback,
    ).to_dict()


def _score_pairing_live(
    task: tuple[str, CorpusLearnerPair, int],
    *,
    live: Mapping[str, Any],
    case: HistogramGameCase,
    bank: Any,
    oracle: CalibrationGameOracle,
    selection_digest: str,
    graph_worker_count: int,
) -> dict[str, Any]:
    run_key, pairing, seed = task
    joint_counts = np.asarray(
        live["online_learning_telemetry"]["joint_action_counts"],
        dtype=np.int64,
    )
    strategic = score_calibration_behavior(
        oracle,
        joint_counts=joint_counts,
        alice_strategy=np.asarray(live["alice_strategy"], dtype=float),
        eve_strategy=np.asarray(live["eve_strategy"], dtype=float),
        include_equilibrium_distance=True,
    )
    false_signal = live["alice_false_signal_telemetry"]
    return {
        "schema": CORPUS_SCALE_RESULT_SCHEMA,
        "run_key": run_key,
        "selection_digest_blake2b_128": selection_digest,
        "graph_id": case.graph_id,
        "family": case.family,
        "assignment_id": case.assignment_id,
        "route_count": len(case.routes),
        "eve_action_count": len(live["actions"]),
        "pairing_label": pairing.label,
        "alice_family": pairing.alice_family,
        "eve_family": pairing.eve_family,
        "pairing": pairing.to_dict(),
        "seed": seed,
        "turns": live["turns"],
        "elapsed_seconds": live["elapsed_seconds"],
        "turns_per_second": live["turns_per_second"],
        "execution_worker_count": 2,
        "graph_worker_count": int(graph_worker_count),
        "cache_sharing": (
            "two_learner_threads_share_one_mutable_on_demand_pool_owned_by_the_"
            "graph_process"
        ),
        "cache_population_order": (
            "required_no_attack_route_calibration_then_online_learning_then_"
            "independent_audit"
        ),
        "mean_alice_reward": live["mean_alice_utility"],
        "final_alice_reward": live["final_alice_utility"],
        "mean_bell_qualified_rate_hz": live["mean_bell_qualified_rate_hz"],
        "final_bell_qualified_rate_hz": live["final_bell_qualified_rate_hz"],
        "mean_eve_bandit_reward": live["mean_eve_bandit_reward"],
        "false_signal_rate": false_signal["false_signal_rate_all_turns"],
        "false_signal_policy_total_variation": false_signal[
            "policy_attribution"
        ]["policy_total_variation"],
        "target_active_rate_diagnostic": live[
            "target_overlap_rate_diagnostic"
        ],
        "information_observation_rate": live["information_observation_rate"],
        "alice_strategy": live["alice_strategy"],
        "eve_strategy": live["eve_strategy"],
        "alice_current_policy": live["alice_current_strategy"],
        "eve_current_policy": live["eve_current_strategy"],
        "trajectory": live["learning_curve"],
        "outcome_counts": live["outcome_counts"],
        "alice_runtime": live["online_learning_telemetry"]["alice"],
        "eve_runtime": live["online_learning_telemetry"]["eve"],
        "bank_content_digest_blake2b_128": bank.content_digest(),
        "learner_outcome_source": "fresh_compiled_histogram_transcript_per_turn",
        "scoring_source": "independent_equal_coverage_histogram_audit_bank",
        **strategic,
    }


class _GraphProgress:
    """Atomic, worker-authored progress for one graph process."""

    def __init__(self, path: Path, *, graph_id: str, family: str) -> None:
        self.path = path
        self.graph_id = str(graph_id)
        self.family = str(family)
        self._lock = threading.Lock()
        self._phase = "starting"
        self._message = "graph worker starting"
        self._learner: dict[str, str] = {}
        self._audit: str | None = None
        self._write_locked()

    def update(self, *, phase: str, message: str) -> None:
        with self._lock:
            self._phase = str(phase)
            self._message = str(message)
            self._write_locked()

    def learner(self, label: str, message: str) -> None:
        with self._lock:
            self._phase = "online_learning"
            self._learner[str(label)] = str(message)
            self._message = "two learner pairings share the graph's on-demand cache"
            self._write_locked()

    def audit(self, message: str) -> None:
        with self._lock:
            self._phase = "independent_audit"
            self._audit = str(message)
            self._message = "post-learning equal-coverage audit"
            self._write_locked()

    def _write_locked(self) -> None:
        _write_json_atomic(self.path, {
            "schema": "histogram_corpus_scale_progress.v2",
            "graph_id": self.graph_id,
            "family": self.family,
            "phase": self._phase,
            "message": self._message,
            "learner_progress": dict(sorted(self._learner.items())),
            "audit_progress": self._audit,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })


def build_corpus_execution_plan(
    ordered_cases: Sequence[Mapping[str, Any]],
    *,
    pending_graph_ids: set[str],
    requested_graph_workers: int,
    available_memory_mib: float,
    memory_reserve_mib: float,
    per_graph_overhead_mib: float,
) -> dict[str, Any]:
    """Choose graph concurrency from the declared full-cache upper bounds."""

    if requested_graph_workers < 1:
        raise ValueError("requested_graph_workers must be positive")
    for name, value in (
        ("available_memory_mib", available_memory_mib),
        ("memory_reserve_mib", memory_reserve_mib),
        ("per_graph_overhead_mib", per_graph_overhead_mib),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    pending = [
        row for row in ordered_cases
        if str(row["graph_id"]) in pending_graph_ids
    ]
    cpu_count = int(os.cpu_count() or 1)
    cpu_limit = max(1, cpu_count // 2)
    selected = min(requested_graph_workers, cpu_limit, len(pending))
    usable = float(available_memory_mib) - float(memory_reserve_mib)
    if pending and usable <= 0.0:
        raise MemoryError("memory reserve leaves no capacity for a graph worker")

    def initial_bound(worker_count: int) -> float:
        return float(sum(
            float(row["estimated_cache_mib"]) + per_graph_overhead_mib
            for row in pending[:worker_count]
        ))

    while selected > 1 and initial_bound(selected) > usable:
        selected -= 1
    if selected and initial_bound(selected) > usable:
        largest = float(pending[0]["estimated_cache_mib"])
        raise MemoryError(
            "largest pending graph cannot fit the declared available-memory "
            f"budget: cache={largest:.1f} MiB, overhead={per_graph_overhead_mib:.1f} "
            f"MiB, usable={usable:.1f} MiB"
        )
    top = pending[:selected]
    return {
        "schema": CORPUS_SCALE_EXECUTION_PLAN_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_graph_workers": int(requested_graph_workers),
        "selected_graph_workers": int(selected),
        "learner_threads_per_graph": 2,
        "logical_cpu_count": cpu_count,
        "cpu_worker_limit": cpu_limit,
        "available_memory_mib": float(available_memory_mib),
        "memory_reserve_mib": float(memory_reserve_mib),
        "usable_memory_mib": usable,
        "per_graph_overhead_mib": float(per_graph_overhead_mib),
        "initial_resident_upper_bound_mib": initial_bound(selected),
        "resident_bound_semantics": (
            "sum of the selected largest full-surface cache estimates plus one "
            "declared process/audit overhead allowance per graph worker; pending "
            "tasks are globally nonincreasing in estimated cache size"
        ),
        "pending_graph_count": len(pending),
        "initial_graphs": [
            {
                "graph_id": str(row["graph_id"]),
                "family": str(row["family"]),
                "estimated_cache_mib": float(row["estimated_cache_mib"]),
            }
            for row in top
        ],
        "cache_execution": (
            "one graph process owns one mutable on-demand physics pool; two "
            "learner threads share it; the complete audit follows learning"
        ),
    }


def _available_memory_mib() -> float:
    """Read the kernel's MemAvailable estimate without launching a command."""

    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise RuntimeError(
            "automatic graph concurrency requires /proc/meminfo or an explicit "
            "available_memory_mib value"
        )
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kib = float(line.split()[1])
            return kib / 1024.0
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _load_graph_checkpoints(
    directory: Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    digest = str(manifest["selection_digest_blake2b_128"])
    expected_labels = {
        str(row["label"]) for row in manifest["pairings"]
    }
    graph_ids = {str(row["graph_id"]) for row in manifest["cases"]}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != CORPUS_SCALE_GRAPH_CHECKPOINT_SCHEMA:
            raise ValueError(f"unsupported graph checkpoint schema at {path}")
        if row.get("selection_digest_blake2b_128") != digest:
            raise ValueError(f"stale graph checkpoint at {path}")
        graph_id = str(row.get("graph_id"))
        if graph_id not in graph_ids:
            raise ValueError(f"unexpected graph checkpoint at {path}")
        labels = {str(item["pairing_label"]) for item in row.get("results", [])}
        if labels != expected_labels:
            raise ValueError(f"incomplete graph checkpoint at {path}")
        result[graph_id] = row
    return result


def _materialize_corpus_rows(
    output: Path,
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    physics_by_graph: Mapping[str, Mapping[str, Any]],
) -> None:
    _write_jsonl_atomic(
        output / "results.jsonl",
        [rows_by_key[key] for key in sorted(rows_by_key)],
    )
    _write_jsonl_atomic(
        output / "physics_compilation.jsonl",
        [physics_by_graph[key] for key in sorted(physics_by_graph)],
    )


def _progress_snapshot(directory: Path, *, remaining: int) -> str:
    active: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("phase") == "complete":
            continue
        family = str(row.get("family") or row.get("graph_id"))
        phase = str(row.get("phase") or "unknown")
        detail = row.get("audit_progress")
        if not detail:
            learners = row.get("learner_progress") or {}
            if learners:
                detail = " | ".join(str(value) for value in learners.values())
        active.append(f"{family}:{phase}" + (f" [{detail}]" if detail else ""))
    preview = "; ".join(active[:6]) or "workers starting"
    if len(active) > 6:
        preview += f"; +{len(active) - 6} more"
    return f"{remaining} graphs pending; {preview}"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _base_config(
    plan: CorpusScalePlan,
    seed: int,
    *,
    case: HistogramGameCase,
) -> HistogramCoLearningConfig:
    return HistogramCoLearningConfig(
        turns=plan.turns,
        final_window=plan.final_window,
        record_every=plan.record_every,
        posthoc_samples_per_cell=0,
        seed=int(seed),
        requested_pairs_per_turn=plan.requested_pairs_per_turn,
        age_points_per_edge=plan.age_points_per_edge,
        evaluated_cache_size=plan.evaluated_cache_size,
    )


def _assert_case_contract(
    entry: Mapping[str, Any],
    case: HistogramGameCase,
    cache: Mapping[str, Any],
    plan: CorpusScalePlan,
) -> None:
    if len(case.routes) != int(entry["route_count"]):
        raise ValueError(f"route count changed for {case.graph_id}")
    route_digest = _object_digest([str(row["route_id"]) for row in case.routes])
    if route_digest != entry["route_ids_digest_blake2b_128"]:
        raise ValueError(f"route universe changed for {case.graph_id}")
    if int(cache["canonical_unique_move_count_total"]) != int(
        entry["canonical_cache_entries"]
    ):
        raise ValueError(f"cache action surface changed for {case.graph_id}")
    if float(cache["estimated_full_action_surface_cache_mib"]) > (
        plan.maximum_case_cache_limit_mib
    ):
        raise MemoryError(f"case cache envelope exceeded for {case.graph_id}")
    if not cache["sufficient_for_declared_action_surface_without_eviction"]:
        raise MemoryError(f"configured LRU cannot retain {case.graph_id}")


def _assert_compiled_surface(diagnostics: Mapping[str, Any]) -> None:
    cache = diagnostics["cache_plan"]
    if not cache["sufficient_for_declared_action_surface_without_eviction"]:
        raise RuntimeError("compiled action surface cannot be retained")
    evictions = float(diagnostics.get("stats", {}).get("evaluated_cache_evictions", 0.0))
    if evictions != 0.0:
        raise RuntimeError("physics cache evicted a compiled histogram")


def _assert_complete_audit_surface(
    diagnostics: Mapping[str, Any],
    *,
    expected_canonical_entries: int,
) -> None:
    misses = int(
        round(float(diagnostics.get("stats", {}).get("evaluated_cache_misses", 0.0)))
    )
    if misses != int(expected_canonical_entries):
        raise RuntimeError(
            "post-learning equal-coverage audit did not populate the complete "
            f"canonical surface: misses={misses}, expected={expected_canonical_entries}"
        )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != CORPUS_SCALE_MANIFEST_SCHEMA:
        raise ValueError("unsupported corpus-scale manifest schema")
    expected = _selection_digest(manifest)
    if manifest.get("selection_digest_blake2b_128") != expected:
        raise ValueError("corpus-scale manifest selection digest mismatch")
    if len(manifest.get("pairings", [])) != 2:
        raise ValueError("corpus-scale experiment requires two pairings")
    if manifest.get("endpoint_selection", {}).get("route_universe") != (
        "every simple cycle-free Alice--Bob path"
    ):
        raise ValueError("corpus-scale manifest route universe changed")


def _validate_existing_rows(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    pairings: Sequence[CorpusLearnerPair],
    plan: CorpusScalePlan,
) -> None:
    graph_ids = {str(row["graph_id"]) for row in manifest["cases"]}
    labels = {row.label for row in pairings}
    digest = str(manifest["selection_digest_blake2b_128"])
    for key, row in rows.items():
        if row.get("selection_digest_blake2b_128") != digest:
            raise ValueError(f"stale resumable row {key}")
        if row.get("graph_id") not in graph_ids or row.get("pairing_label") not in labels:
            raise ValueError(f"unexpected resumable row {key}")
        if int(row.get("turns", -1)) != plan.turns:
            raise ValueError(f"turn contract changed for resumable row {key}")


def _selection_digest(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: manifest[key]
        for key in (
            "schema",
            "corpus",
            "census",
            "plan",
            "pairings",
            "endpoint_selection",
            "evaluation_partition_loaded",
            "evaluation_partition_status",
            "cases",
        )
    }
    return _object_digest(payload)


def _object_digest(value: Any) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_seed(root_seed: int, graph_id: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    digest.update(b"\0")
    digest.update(graph_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)


def _run_key(graph_id: str, pairing_label: str, seed: int) -> str:
    return f"{graph_id}|{pairing_label}|seed-{seed}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {key: None for key in ("mean", "median", "q25", "q75", "minimum", "maximum")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
