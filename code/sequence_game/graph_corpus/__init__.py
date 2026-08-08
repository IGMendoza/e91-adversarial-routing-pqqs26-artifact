"""Graph-corpus generation for control/XAI/adversarial-learning experiments."""

from .assignment import (
    DEGENERATE_EDGE_BOTTLENECK,
    DEGENERATE_NODE_BOTTLENECK,
    DEGENERATE_SINGLE_ROUTE,
    INVALID_DISCONNECTED,
    LARGE_GRAPH_SELECTION_POLICY,
    LARGE_GRAPH_SELECTION_TOP_SAMPLED,
    POLICY_TOP_SAMPLED_PAIR,
    VALID_MULTI_ROUTE,
    AssignmentBuildResult,
    AssignmentSearchConfig,
    GraphAssignmentRecord,
    build_assignments_for_graph,
    classify_strategy_validity,
)
from .generators import DEFAULT_GRAPH_SPECS, generate_synthetic_graph, generate_synthetic_graphs
from .ingest import (
    ingest_graphml_gml_directory,
    ingest_snap_edge_list_directory,
    ingest_topology_ir_directory,
)
from .topology_bench import (
    SOURCE_TOPOLOGY_BENCH,
    ingest_topology_bench_directory,
    ingest_topology_bench_workbook,
)
from .metrics import RouteStructureRecord, analyze_route_structure
from .writer import (
    DEFAULT_OUTPUT_ROOT,
    ASSIGNMENT_SUMMARY_COLUMNS,
    CORPUS_SUMMARY_COLUMNS,
    GraphCorpusBuildConfig,
    GraphCorpusBuildResult,
    GraphCorpusRecord,
    write_graph_corpus,
)

__all__ = [
    "ASSIGNMENT_SUMMARY_COLUMNS",
    "CORPUS_SUMMARY_COLUMNS",
    "DEFAULT_GRAPH_SPECS",
    "DEFAULT_OUTPUT_ROOT",
    "DEGENERATE_EDGE_BOTTLENECK",
    "DEGENERATE_NODE_BOTTLENECK",
    "DEGENERATE_SINGLE_ROUTE",
    "INVALID_DISCONNECTED",
    "LARGE_GRAPH_SELECTION_POLICY",
    "LARGE_GRAPH_SELECTION_TOP_SAMPLED",
    "POLICY_TOP_SAMPLED_PAIR",
    "SOURCE_TOPOLOGY_BENCH",
    "VALID_MULTI_ROUTE",
    "AssignmentBuildResult",
    "AssignmentSearchConfig",
    "GraphAssignmentRecord",
    "GraphCorpusBuildConfig",
    "GraphCorpusBuildResult",
    "GraphCorpusRecord",
    "RouteStructureRecord",
    "analyze_route_structure",
    "build_assignments_for_graph",
    "classify_strategy_validity",
    "generate_synthetic_graph",
    "generate_synthetic_graphs",
    "ingest_graphml_gml_directory",
    "ingest_snap_edge_list_directory",
    "ingest_topology_bench_directory",
    "ingest_topology_bench_workbook",
    "ingest_topology_ir_directory",
    "write_graph_corpus",
]
