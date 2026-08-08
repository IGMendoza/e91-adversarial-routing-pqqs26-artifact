#!/usr/bin/env python3
"""Build a queryable SQLite graph corpus with Alice/Bob assignments.

Primary pipeline:

1. Collect the Topology Bench real optical-core topologies. SNAP AS-733 and
   synthetic controls are opt-in structural extensions.
2. Compute Alice/Bob (source/sink) assignments per graph with the existing
   assignment policies and retain every cycle-free route.
3. Write the on-disk corpus layout, then ingest it losslessly into SQLite and
   index graphs/assignments into the structured ``graph_corpus_graphs`` /
   ``graph_corpus_assignments`` tables for direct SQL querying.

Run with:

    .venv/bin/python scripts/build_graph_corpus_sqlite.py

Query examples once built:

    SELECT alice, bob, shortest_hops FROM graph_corpus_assignments
    WHERE shortest_hops BETWEEN 5 AND 15;
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.corpus.sqlite_store import (  # noqa: E402
    connect_corpus_db,
    ingest_graph_corpus_to_sqlite,
)
from sequence_game.graph_corpus import (  # noqa: E402
    LARGE_GRAPH_SELECTION_POLICY,
    LARGE_GRAPH_SELECTION_TOP_SAMPLED,
    AssignmentSearchConfig,
    GraphCorpusBuildConfig,
    write_graph_corpus,
)

DEFAULT_OUTPUT_ROOT = Path("runs") / "graph_corpus_sqlite_v1"
DEFAULT_TOPOLOGY_BENCH_DIR = Path("data") / "external" / "topology_bench" / "extracted"
DEFAULT_SNAP_DIR = Path("data") / "external" / "snap_as_733" / "extracted"


def main(argv: list[str] | None = None) -> int:
    default_assignment_config = AssignmentSearchConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite output path. Defaults to <output-root>/graph_corpus.sqlite.",
    )
    parser.add_argument(
        "--corpus-id",
        type=str,
        default=None,
        help="Corpus identifier in the DB. Defaults to the output-root name.",
    )
    parser.add_argument(
        "--topology-bench-dir",
        type=Path,
        default=DEFAULT_TOPOLOGY_BENCH_DIR,
        help="Directory of Topology Bench real-topology .xlsx workbooks.",
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=DEFAULT_SNAP_DIR,
        help="Directory of SNAP AS-733 edge-list snapshots.",
    )
    parser.add_argument(
        "--snap-limit",
        type=int,
        default=0,
        help="Number of SNAP snapshots to include, evenly spaced over time. "
        "0 disables SNAP ingestion.",
    )
    parser.add_argument(
        "--synthetic-graph-limit",
        type=int,
        default=0,
        help="Number of synthetic control graphs to include (0 disables). "
        "The first four are the named braess/bottleneck/ladder/diamond controls.",
    )
    parser.add_argument("--base-seed", type=int, default=20260705)
    parser.add_argument(
        "--min-ab-hops",
        type=int,
        default=default_assignment_config.min_ab_hops,
    )
    parser.add_argument(
        "--max-assignments-per-graph",
        type=int,
        default=default_assignment_config.max_assignments_per_graph,
    )
    parser.add_argument(
        "--large-graph-selection-mode",
        choices=(LARGE_GRAPH_SELECTION_POLICY, LARGE_GRAPH_SELECTION_TOP_SAMPLED),
        default=default_assignment_config.large_graph_selection_mode,
    )
    parser.add_argument(
        "--max-pair-evaluations-large",
        type=int,
        default=default_assignment_config.max_pair_evaluations_large,
    )
    parser.add_argument(
        "--include-degenerate-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    db_path = args.db_path or (output_root / "graph_corpus.sqlite")
    corpus_id = args.corpus_id or output_root.name

    snap_dirs: tuple[Path, ...] = ()
    if args.snap_limit > 0:
        staged = _stage_snap_snapshots(args.snap_dir, output_root, args.snap_limit)
        if staged is not None:
            snap_dirs = (staged,)

    topology_bench_dirs: tuple[Path, ...] = ()
    bench_dir = args.topology_bench_dir
    if not bench_dir.is_absolute():
        bench_dir = ROOT / bench_dir
    if bench_dir.is_dir() and any(bench_dir.glob("*.xlsx")):
        topology_bench_dirs = (bench_dir,)
    else:
        print(f"topology bench dir not usable, skipping: {bench_dir}", flush=True)

    assignment_config = replace(
        AssignmentSearchConfig(),
        max_assignments_per_graph=args.max_assignments_per_graph,
        min_ab_hops=args.min_ab_hops,
        max_pair_evaluations_large=args.max_pair_evaluations_large,
        large_graph_selection_mode=args.large_graph_selection_mode,
        random_seed=args.base_seed,
    )
    config = GraphCorpusBuildConfig(
        output_root=output_root,
        base_seed=args.base_seed,
        assignment_config=assignment_config,
        snap_edge_list_dirs=snap_dirs,
        topology_bench_dirs=topology_bench_dirs,
        include_degenerate_controls=args.include_degenerate_controls,
        synthetic_graph_limit=args.synthetic_graph_limit,
    )
    build_result = write_graph_corpus(config, repo_root=ROOT)

    ingest_summary = ingest_graph_corpus_to_sqlite(
        output_root,
        db_path,
        corpus_id=corpus_id,
        progress=lambda message: print(message, flush=True),
    )

    payload = {
        "build": {
            "output_root": str(build_result.output_root),
            "graph_count": build_result.graph_count,
            "assignment_count": build_result.assignment_count,
            "skipped_count": build_result.skipped_count,
            "strategy_validity_counts": build_result.strategy_validity_counts,
            "source_counts": build_result.source_counts,
        },
        "sqlite": ingest_summary.to_dict(),
        "assignment_hop_histogram": _hop_histogram(db_path, corpus_id),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _stage_snap_snapshots(
    snap_dir: Path,
    output_root: Path,
    snap_limit: int,
) -> Path | None:
    """Copy an evenly spaced selection of SNAP snapshots into the corpus root."""
    if not snap_dir.is_absolute():
        snap_dir = ROOT / snap_dir
    if not snap_dir.is_dir():
        print(f"snap dir not found, skipping SNAP ingestion: {snap_dir}", flush=True)
        return None
    snapshots = sorted(snap_dir.glob("*.txt"))
    if not snapshots:
        print(f"no SNAP snapshots under {snap_dir}, skipping", flush=True)
        return None
    if len(snapshots) > snap_limit:
        step = (len(snapshots) - 1) / max(snap_limit - 1, 1)
        indices = sorted({round(index * step) for index in range(snap_limit)})
        snapshots = [snapshots[index] for index in indices]
    staging = output_root / "ingest_staging" / "snap_as_733"
    staging.mkdir(parents=True, exist_ok=True)
    for existing in staging.glob("*.txt"):
        existing.unlink()
    for snapshot in snapshots:
        shutil.copy2(snapshot, staging / snapshot.name)
    print(
        f"staged {len(snapshots)} SNAP snapshots into {staging}",
        flush=True,
    )
    return staging


def _hop_histogram(db_path: Path, corpus_id: str) -> dict[str, int]:
    con = connect_corpus_db(db_path)
    try:
        rows = con.execute(
            """
            SELECT shortest_hops, COUNT(*) AS n
            FROM graph_corpus_assignments
            WHERE corpus_id = ?
            GROUP BY shortest_hops
            ORDER BY shortest_hops
            """,
            (corpus_id,),
        ).fetchall()
    finally:
        con.close()
    return {str(row["shortest_hops"]): int(row["n"]) for row in rows}


if __name__ == "__main__":
    raise SystemExit(main())
