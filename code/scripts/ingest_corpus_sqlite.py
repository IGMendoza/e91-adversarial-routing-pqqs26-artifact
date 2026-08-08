#!/usr/bin/env python3
"""Ingest corpus run JSON outputs into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sequence_game.corpus.sqlite_store import (
    ingest_graph_corpus_to_sqlite,
    ingest_run_to_sqlite,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest corpus run JSON artifacts into corpus_results.sqlite."
    )
    parser.add_argument(
        "run_path",
        type=Path,
        help="Run root, outer run directory, or data directory to ingest.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite output path. Defaults to <run>/data/corpus_results.sqlite.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N checkpoint records.",
    )
    parser.add_argument(
        "--refresh-checkpoint",
        action="store_true",
        help="Rewrite checkpoint payload rows even if they already exist.",
    )
    parser.add_argument(
        "--with-source-files",
        action="store_true",
        help="Store exact run output source files, excluding the giant checkpoint JSON.",
    )
    parser.add_argument(
        "--graph-corpus",
        type=Path,
        action="append",
        default=[],
        help="Additional graph corpus directory to store losslessly in the same DB.",
    )
    args = parser.parse_args(argv)

    summary = ingest_run_to_sqlite(
        args.run_path,
        args.db_path,
        progress=lambda message: print(message, flush=True),
        progress_every=args.progress_every,
        refresh_checkpoint=args.refresh_checkpoint,
        ingest_source_files=args.with_source_files,
    )
    payload = {"run": summary.to_dict(), "graph_corpora": []}
    db_path = args.db_path or summary.db_path
    for corpus_path in args.graph_corpus:
        graph_summary = ingest_graph_corpus_to_sqlite(
            corpus_path,
            db_path,
            progress=lambda message: print(message, flush=True),
            progress_every=max(args.progress_every, 100),
        )
        payload["graph_corpora"].append(graph_summary.to_dict())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
