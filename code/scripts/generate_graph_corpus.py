"""Generate the graph-corpus v2 dataset.

Run with:

    .venv/bin/python scripts/generate_graph_corpus.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.graph_corpus import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    LARGE_GRAPH_SELECTION_POLICY,
    LARGE_GRAPH_SELECTION_TOP_SAMPLED,
    AssignmentSearchConfig,
    GraphCorpusBuildConfig,
    write_graph_corpus,
)


def main(argv: list[str] | None = None) -> int:
    default_assignment_config = AssignmentSearchConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--topology-ir-dir", type=Path, action="append", default=[])
    parser.add_argument("--graphml-gml-dir", type=Path, action="append", default=[])
    parser.add_argument("--snap-edge-list-dir", type=Path, action="append", default=[])
    parser.add_argument("--base-seed", type=int, default=20260627)
    parser.add_argument(
        "--max-assignments-per-graph",
        type=int,
        default=default_assignment_config.max_assignments_per_graph,
    )
    parser.add_argument(
        "--large-graph-node-threshold",
        type=int,
        default=default_assignment_config.large_graph_node_threshold,
    )
    parser.add_argument(
        "--max-pair-evaluations-large",
        type=int,
        default=default_assignment_config.max_pair_evaluations_large,
    )
    parser.add_argument(
        "--large-graph-selection-mode",
        choices=(LARGE_GRAPH_SELECTION_POLICY, LARGE_GRAPH_SELECTION_TOP_SAMPLED),
        default=default_assignment_config.large_graph_selection_mode,
    )
    parser.add_argument(
        "--large-graph-finalist-count",
        type=int,
        default=default_assignment_config.large_graph_finalist_count,
    )
    parser.add_argument("--synthetic-graph-limit", type=int, default=None)
    parser.add_argument(
        "--exclude-direct-links",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-degenerate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    assignment_config = replace(
        default_assignment_config,
        max_assignments_per_graph=args.max_assignments_per_graph,
        exclude_direct_links=args.exclude_direct_links,
        large_graph_node_threshold=args.large_graph_node_threshold,
        max_pair_evaluations_large=args.max_pair_evaluations_large,
        large_graph_selection_mode=args.large_graph_selection_mode,
        large_graph_finalist_count=args.large_graph_finalist_count,
        random_seed=args.base_seed,
    )
    config = GraphCorpusBuildConfig(
        output_root=args.output_root,
        base_seed=args.base_seed,
        assignment_config=assignment_config,
        topology_ir_dirs=tuple(args.topology_ir_dir),
        graphml_gml_dirs=tuple(args.graphml_gml_dir),
        snap_edge_list_dirs=tuple(args.snap_edge_list_dir),
        include_degenerate_controls=args.include_degenerate_controls,
        synthetic_graph_limit=args.synthetic_graph_limit,
    )
    write_graph_corpus(config, repo_root=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
