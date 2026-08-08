#!/usr/bin/env python3
"""Census Topology Bench and freeze histogram learner case partitions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.case_selection import (  # noqa: E402
    freeze_case_partitions,
)
from sequence_game.experiments.histogram_colearning.census_artifacts import (  # noqa: E402
    write_census_artifacts,
)
from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    DEFAULT_CORPUS_DB_PATH,
)
from sequence_game.experiments.histogram_colearning.paper import (  # noqa: E402
    DEFAULT_PAPER_CASES,
)
from sequence_game.experiments.histogram_colearning.topology_census import (  # noqa: E402
    TopologyCensusConfig,
    census_topology_bench,
)


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "topology_bench_step6_census"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enumeration-cache-limit-mib", type=float, default=4096.0)
    parser.add_argument("--dfs-expansion-limit", type=int, default=2_000_000)
    parser.add_argument("--selected-cache-limit-mib", type=float, default=768.0)
    parser.add_argument("--audit-samples-per-cell", type=int, default=64)
    parser.add_argument("--selected-audit-draw-limit", type=int, default=2_000_000)
    parser.add_argument("--calibration-count", type=int, default=5)
    parser.add_argument("--evaluation-count", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TopologyCensusConfig(
        enumeration_cache_limit_mib=args.enumeration_cache_limit_mib,
        enumeration_dfs_expansion_limit=args.dfs_expansion_limit,
        selected_case_cache_limit_mib=args.selected_cache_limit_mib,
        audit_samples_per_cell=args.audit_samples_per_cell,
        selected_case_audit_draw_limit=args.selected_audit_draw_limit,
    )
    census = census_topology_bench(
        args.corpus_db,
        config=config,
        progress_callback=print,
    )
    development_case_ids = [
        f"{spec.graph_id}::{spec.assignment_id}" for spec in DEFAULT_PAPER_CASES
    ]
    case_manifest = freeze_case_partitions(
        census,
        development_case_ids=development_case_ids,
        calibration_count=args.calibration_count,
        evaluation_count=args.evaluation_count,
    )
    artifacts = write_census_artifacts(census, case_manifest, args.out_dir)
    print(json.dumps({
        "summary": census["summary"],
        "selection_digest_blake2b_128": case_manifest[
            "selection_digest_blake2b_128"
        ],
        "partitions": {
            role: [row["case_id"] for row in rows]
            for role, rows in case_manifest["partitions"].items()
        },
        "artifact_manifest_path": artifacts["artifact_manifest_path"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
