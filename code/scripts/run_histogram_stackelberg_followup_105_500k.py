#!/usr/bin/env python3
"""Evaluate the frozen Stackelberg route margin on all 105 corpus graphs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_followup_corpus import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    build_stackelberg_followup_manifest,
    run_stackelberg_followup_study,
)


DEFAULT_CORPUS_RUN_DIR = REPO_ROOT / "runs" / "histogram_corpus_500k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-run-dir", type=Path, default=DEFAULT_CORPUS_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph-workers", type=int, default=8)
    parser.add_argument("--route-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    corpus_manifest = json.loads(
        (args.corpus_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.is_file():
        study = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        study = build_stackelberg_followup_manifest(corpus_manifest)
    print(json.dumps(study["preflight"], indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    summary = run_stackelberg_followup_study(
        corpus_manifest,
        study,
        output_dir=args.output_dir,
        graph_workers=args.graph_workers,
        route_workers=args.route_workers,
        progress_callback=_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    main()
