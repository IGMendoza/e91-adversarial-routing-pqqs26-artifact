"""Corpus writer and orchestration for graph-corpus generation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from sequence_game.claims import CONTROL_GAME
from sequence_game.topology import TopologyIR, TopologyMetadata

from .assignment import (
    AssignmentBuildResult,
    AssignmentSearchConfig,
    GraphAssignmentRecord,
    assignment_class_counts,
    build_assignments_for_graph,
)
from .generators import (
    DEFAULT_FIXED_EDGE_LENGTH_M,
    DEFAULT_GRAPH_SPECS,
    EDGE_LENGTH_SYNTHETIC,
    PHYSICAL_STRUCTURAL_STRESS_ONLY,
    PHYSICAL_SYNTHETIC_CONTROL,
    SOURCE_SYNTHETIC,
    generate_synthetic_graphs,
    generated_graph_count,
)
from .ingest import (
    EDGE_LENGTH_SYNTHETIC as INGEST_EDGE_LENGTH_SYNTHETIC,
    PHYSICAL_STRUCTURAL_STRESS_ONLY as INGEST_PHYSICAL_STRUCTURAL_STRESS_ONLY,
    SOURCE_SNAP_AS,
    SOURCE_TOPOLOGY_ZOO,
    ingest_graphml_gml_directory,
    ingest_snap_edge_list_directory,
    ingest_topology_ir_directory,
)
from .topology_bench import ingest_topology_bench_directory

DEFAULT_OUTPUT_ROOT = Path("runs") / "graph_corpus_paper_v2"

CORPUS_SUMMARY_COLUMNS = [
    "graph_id",
    "family",
    "source",
    "scope_label",
    "physical_role",
    "node_count",
    "edge_count",
    "edge_length_semantics",
    "topology_path",
    "assignment_count",
]

ASSIGNMENT_SUMMARY_COLUMNS = [
    "assignment_id",
    "graph_id",
    "alice",
    "bob",
    "assignment_policy",
    "strategy_validity",
    "route_count",
    "shortest_hops",
    "longest_hops",
    "shortest_path_count",
    "node_cut_size",
    "edge_cut_size",
    "candidate_node_hitting_size",
    "candidate_edge_hitting_size",
    "route_overlap_summary",
    "assignment_path",
]


@dataclass(frozen=True)
class GraphCorpusRecord:
    graph_id: str
    family: str
    source: str
    scope_label: str
    physical_role: str
    node_count: int
    edge_count: int
    edge_length_semantics: str
    topology_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "source": self.source,
            "scope_label": self.scope_label,
            "physical_role": self.physical_role,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "edge_length_semantics": self.edge_length_semantics,
            "topology_path": self.topology_path,
        }


@dataclass(frozen=True)
class GraphCorpusBuildConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    base_seed: int = 20260627
    graph_specs: tuple[dict[str, Any], ...] = field(
        default_factory=lambda: tuple(dict(spec) for spec in DEFAULT_GRAPH_SPECS)
    )
    fixed_edge_length_m: float = DEFAULT_FIXED_EDGE_LENGTH_M
    assignment_config: AssignmentSearchConfig = field(default_factory=AssignmentSearchConfig)
    topology_ir_dirs: tuple[Path, ...] = ()
    graphml_gml_dirs: tuple[Path, ...] = ()
    snap_edge_list_dirs: tuple[Path, ...] = ()
    topology_bench_dirs: tuple[Path, ...] = ()
    include_degenerate_controls: bool = True
    synthetic_graph_limit: Optional[int] = None

    def resolved(self, root: Path) -> "GraphCorpusBuildConfig":
        output_root = self.output_root
        if not output_root.is_absolute():
            output_root = root / output_root
        return replace(
            self,
            output_root=output_root,
            assignment_config=replace(
                self.assignment_config,
                include_degenerate_controls=self.include_degenerate_controls,
            ),
        )


@dataclass(frozen=True)
class GraphCorpusBuildResult:
    output_root: Path
    graph_count: int
    assignment_count: int
    skipped_count: int
    strategy_validity_counts: dict[str, int]
    source_counts: dict[str, int]
    generated_candidate_count: int


def write_graph_corpus(
    config: GraphCorpusBuildConfig,
    *,
    repo_root: Optional[Path] = None,
) -> GraphCorpusBuildResult:
    root = repo_root or Path.cwd()
    config = config.resolved(root)
    output_root = config.output_root
    accepted_dir = output_root / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    skipped_path = output_root / "pruned_or_skipped.jsonl"
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text("", encoding="utf-8")

    graph_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    manifest_graphs = []
    manifest_assignments = []
    skipped_records = []
    graph_count = 0
    assignment_count = 0
    class_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    graphs = list(_collect_graphs(config))
    total = len(graphs)
    print(_config_summary(config, total), flush=True)
    for index, graph in enumerate(graphs):
        graph_id = _graph_id(graph, index)
        family = str(graph.metadata.params.get("graph_family", graph.metadata.generator))
        print(
            f"[{index + 1}/{total}] graph={graph_id} family={family} "
            f"nodes={len(graph.nodes)} edges={len(graph.edges)}",
            flush=True,
        )
        assignment_result = build_assignments_for_graph(
            graph,
            graph_id,
            config.assignment_config,
        )
        if not assignment_result.assignments:
            skipped = _skipped_record(graph_id, graph, assignment_result)
            skipped_records.append(skipped)
            _append_jsonl(skipped, skipped_path)
            print(f"  SKIP {skipped['reason']}", flush=True)
            continue

        graph_dir = accepted_dir / graph_id
        assignments_dir = graph_dir / "assignments"
        assignments_dir.mkdir(parents=True, exist_ok=True)
        topology_path = graph_dir / "topology.json"
        graph.metadata = TopologyMetadata(
            graph.metadata.generator,
            {
                **graph.metadata.params,
                "graph_id": graph_id,
                "scope_label": CONTROL_GAME,
            },
            graph.metadata.seed,
        )
        _write_json(graph.to_dict(), topology_path)

        corpus_record = _corpus_record(graph_id, graph, topology_path, root)
        assignments = []
        for assignment in assignment_result.assignments:
            assignment_dir = assignments_dir / assignment.assignment_id
            assignment_path = assignment_dir / "assignment.json"
            assignment_with_path = assignment.with_assignment_path(_display_path(assignment_path, root))
            _write_json(assignment_with_path.to_dict(), assignment_path)
            assignments.append(assignment_with_path)
            assignment_rows.append(assignment_with_path.to_summary_row())
            manifest_assignments.append(assignment_with_path.to_summary_row())
            class_counter[assignment_with_path.strategy_validity] += 1
            assignment_count += 1

        graph_summary = _graph_summary(corpus_record, assignments, assignment_result)
        _write_json(graph_summary, graph_dir / "graph_summary.json")
        graph_row = {
            **corpus_record.to_dict(),
            "assignment_count": len(assignments),
        }
        graph_rows.append(graph_row)
        manifest_graphs.append(graph_summary)
        source_counter[corpus_record.source] += 1
        graph_count += 1
        print(
            f"  ACCEPT assignments={len(assignments)} "
            f"classes={assignment_class_counts(assignments)}",
            flush=True,
        )

    _write_csv(output_root / "corpus_summary.csv", CORPUS_SUMMARY_COLUMNS, graph_rows)
    _write_csv(
        output_root / "assignment_summary.csv",
        ASSIGNMENT_SUMMARY_COLUMNS,
        assignment_rows,
    )
    manifest = {
        "scope_label": CONTROL_GAME,
        "script": "scripts/generate_graph_corpus.py",
        "config": _config_payload(config),
        "counts": {
            "graphs_accepted": graph_count,
            "assignments": assignment_count,
            "pruned_or_skipped": len(skipped_records),
            "strategy_validity": dict(class_counter),
            "sources": dict(source_counter),
        },
        "graphs": manifest_graphs,
        "assignments": manifest_assignments,
        "pruned_or_skipped": skipped_records,
    }
    _write_json(manifest, output_root / "corpus_manifest.json")
    print(
        f"done graphs={graph_count} assignments={assignment_count} "
        f"skipped={len(skipped_records)} manifest={_display_path(output_root / 'corpus_manifest.json', root)}",
        flush=True,
    )
    return GraphCorpusBuildResult(
        output_root=output_root,
        graph_count=graph_count,
        assignment_count=assignment_count,
        skipped_count=len(skipped_records),
        strategy_validity_counts=dict(class_counter),
        source_counts=dict(source_counter),
        generated_candidate_count=_configured_synthetic_graph_count(config),
    )


def _collect_graphs(config: GraphCorpusBuildConfig) -> Iterable[TopologyIR]:
    synthetic_iter = generate_synthetic_graphs(
        config.graph_specs,
        base_seed=config.base_seed,
        fixed_edge_length_m=config.fixed_edge_length_m,
    )
    if config.synthetic_graph_limit is not None:
        synthetic_iter = list(synthetic_iter)[: config.synthetic_graph_limit]
    yield from synthetic_iter
    for path in config.topology_ir_dirs:
        yield from ingest_topology_ir_directory(path)
    for path in config.graphml_gml_dirs:
        yield from ingest_graphml_gml_directory(path, source=SOURCE_TOPOLOGY_ZOO)
    for path in config.snap_edge_list_dirs:
        yield from ingest_snap_edge_list_directory(
            path,
            source=SOURCE_SNAP_AS,
            physical_role=INGEST_PHYSICAL_STRUCTURAL_STRESS_ONLY,
            edge_length_semantics=INGEST_EDGE_LENGTH_SYNTHETIC,
            synthetic_edge_length_m=config.fixed_edge_length_m,
        )
    for path in config.topology_bench_dirs:
        yield from ingest_topology_bench_directory(
            path,
            synthetic_edge_length_m=config.fixed_edge_length_m,
        )


def _corpus_record(
    graph_id: str,
    graph: TopologyIR,
    topology_path: Path,
    root: Path,
) -> GraphCorpusRecord:
    params = graph.metadata.params
    return GraphCorpusRecord(
        graph_id=graph_id,
        family=str(params.get("graph_family", graph.metadata.generator)),
        source=str(params.get("source", SOURCE_SYNTHETIC)),
        scope_label=str(params.get("scope_label", CONTROL_GAME)),
        physical_role=str(params.get("physical_role", _default_physical_role(params))),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        edge_length_semantics=str(params.get("edge_length_semantics", EDGE_LENGTH_SYNTHETIC)),
        topology_path=_display_path(topology_path, root),
    )


def _graph_summary(
    corpus_record: GraphCorpusRecord,
    assignments: list[GraphAssignmentRecord],
    assignment_result: AssignmentBuildResult,
) -> dict[str, Any]:
    return {
        **corpus_record.to_dict(),
        "assignment_count": len(assignments),
        "assignment_search_mode": assignment_result.assignment_search_mode,
        "pair_evaluations": assignment_result.pair_evaluations,
        "candidate_node_count": assignment_result.candidate_node_count,
        "assignment_classes": assignment_class_counts(assignments),
        "assignment_ids": [assignment.assignment_id for assignment in assignments],
    }


def _skipped_record(
    graph_id: str,
    graph: TopologyIR,
    assignment_result: AssignmentBuildResult,
) -> dict[str, Any]:
    return {
        "graph_id": graph_id,
        "family": graph.metadata.params.get("graph_family", graph.metadata.generator),
        "source": graph.metadata.params.get("source", SOURCE_SYNTHETIC),
        "reason": assignment_result.skipped_reason or "no assignments emitted",
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "assignment_search_mode": assignment_result.assignment_search_mode,
        "pair_evaluations": assignment_result.pair_evaluations,
        "candidate_node_count": assignment_result.candidate_node_count,
    }


def _config_summary(config: GraphCorpusBuildConfig, total_graphs: int) -> str:
    return (
        "Graph corpus v2 generation\n"
        f"  output_root: {config.output_root}\n"
        f"  planned_graphs: {total_graphs} "
        f"(synthetic={_configured_synthetic_graph_count(config)}, "
        f"topology_ir_dirs={len(config.topology_ir_dirs)}, "
        f"graphml_gml_dirs={len(config.graphml_gml_dirs)}, "
        f"snap_edge_list_dirs={len(config.snap_edge_list_dirs)}, "
        f"topology_bench_dirs={len(config.topology_bench_dirs)})\n"
        f"  include_degenerate_controls: {config.include_degenerate_controls}\n"
        f"  large_graph_node_threshold: {config.assignment_config.large_graph_node_threshold}\n"
        f"  max_pair_evaluations_large: {config.assignment_config.max_pair_evaluations_large}"
    )


def _config_payload(config: GraphCorpusBuildConfig) -> dict[str, Any]:
    assignment_config = config.assignment_config
    return {
        "output_root": str(config.output_root),
        "base_seed": config.base_seed,
        "graph_specs": list(config.graph_specs),
        "generated_candidate_count": _configured_synthetic_graph_count(config),
        "fixed_edge_length_m": config.fixed_edge_length_m,
        "include_degenerate_controls": config.include_degenerate_controls,
        "topology_ir_dirs": [str(path) for path in config.topology_ir_dirs],
        "graphml_gml_dirs": [str(path) for path in config.graphml_gml_dirs],
        "snap_edge_list_dirs": [str(path) for path in config.snap_edge_list_dirs],
        "topology_bench_dirs": [str(path) for path in config.topology_bench_dirs],
        "assignment": {
            "max_assignments_per_graph": assignment_config.max_assignments_per_graph,
            "assignment_policies": list(assignment_config.assignment_policies),
            "exclude_direct_links": assignment_config.exclude_direct_links,
            "min_ab_hops": assignment_config.min_ab_hops,
            "max_route_candidates": assignment_config.max_route_candidates,
            "route_probe_limit": assignment_config.route_probe_limit,
            "exact_hitting_set_max_size": assignment_config.exact_hitting_set_max_size,
            "large_graph_node_threshold": assignment_config.large_graph_node_threshold,
            "large_graph_candidate_node_limit": assignment_config.large_graph_candidate_node_limit,
            "max_pair_evaluations_large": assignment_config.max_pair_evaluations_large,
            "large_graph_selection_mode": assignment_config.large_graph_selection_mode,
            "large_graph_finalist_count": assignment_config.large_graph_finalist_count,
            "max_graph_nodes_for_analysis": assignment_config.max_graph_nodes_for_analysis,
        },
    }


def _configured_synthetic_graph_count(config: GraphCorpusBuildConfig) -> int:
    generated = generated_graph_count(config.graph_specs)
    if config.synthetic_graph_limit is None:
        return generated
    return min(generated, max(0, config.synthetic_graph_limit))


def _graph_id(graph: TopologyIR, index: int) -> str:
    params = graph.metadata.params
    family = str(params.get("graph_family", graph.metadata.generator))
    seed = graph.metadata.seed if graph.metadata.seed is not None else "na"
    source = str(params.get("source", SOURCE_SYNTHETIC))
    return f"{index:04d}_{_slug(source)}_{_slug(family)}_seed_{seed}"


def _default_physical_role(params: dict[str, Any]) -> str:
    source = params.get("source")
    if source == SOURCE_SYNTHETIC:
        return PHYSICAL_SYNTHETIC_CONTROL
    if source == SOURCE_SNAP_AS:
        return PHYSICAL_STRUCTURAL_STRESS_ONLY
    return PHYSICAL_STRUCTURAL_STRESS_ONLY


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "none"
