#!/usr/bin/env python3
"""Build and execute the normalized Real Topologies 500k learner study."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.corpus_scale import (  # noqa: E402
    DEFAULT_CENSUS_PATH,
    DEFAULT_CORPUS_SCALE_OUTPUT_DIR,
    CorpusScalePlan,
    build_corpus_scale_manifest,
    run_corpus_scale_experiment,
    write_corpus_scale_manifest,
)
from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    DEFAULT_CORPUS_DB_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CORPUS_SCALE_OUTPUT_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--census-path", type=Path, default=DEFAULT_CENSUS_PATH)
    parser.add_argument("--turns", type=int, default=500_000)
    parser.add_argument("--record-every", type=int, default=10_000)
    parser.add_argument("--final-window", type=int, default=50_000)
    parser.add_argument("--case-cache-limit-mib", type=float, default=768.0)
    parser.add_argument("--maximum-case-cache-limit-mib", type=float, default=6_144.0)
    parser.add_argument("--audit-samples-per-cell", type=int, default=16)
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=6,
        help=(
            "maximum concurrent graph-owner processes; the runtime may reduce "
            "this count to satisfy the declared RAM reserve"
        ),
    )
    parser.add_argument(
        "--memory-reserve-mib",
        type=float,
        default=8_192.0,
        help="MemAvailable reserve retained outside corpus graph workers",
    )
    parser.add_argument(
        "--per-graph-overhead-mib",
        type=float,
        default=1_024.0,
        help=(
            "RAM allowance per graph process in addition to its full-cache "
            "estimate (learners, audit arrays, interpreter, allocator)"
        ),
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="load output-dir/run_manifest.json instead of rebuilding selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = CorpusScalePlan(
        turns=args.turns,
        record_every=args.record_every,
        final_window=args.final_window,
        case_cache_limit_mib=args.case_cache_limit_mib,
        maximum_case_cache_limit_mib=args.maximum_case_cache_limit_mib,
        audit_samples_per_cell_per_split=args.audit_samples_per_cell,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.json"
    if args.reuse_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_corpus_scale_manifest(
            db_path=args.db_path,
            census_path=args.census_path,
            plan=plan,
            progress_callback=_progress,
        )
        write_corpus_scale_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selection_digest_blake2b_128": manifest[
                    "selection_digest_blake2b_128"
                ],
                "graph_count": len(manifest["cases"]),
                "expected_run_count": len(manifest["cases"])
                * len(manifest["pairings"]),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.manifest_only:
        return
    summary = run_corpus_scale_experiment(
        manifest,
        output_dir=args.output_dir,
        progress_callback=_progress,
        requested_graph_workers=args.graph_workers,
        memory_reserve_mib=args.memory_reserve_mib,
        per_graph_overhead_mib=args.per_graph_overhead_mib,
    )
    print(
        f"complete={summary['complete']} runs={summary['run_count']}/"
        f"{summary['expected_run_count']} output={args.output_dir}",
        flush=True,
    )


def _progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    main()
