#!/usr/bin/env python3
"""Run the five-topology 500k physical Prime-and-Probe study."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe import (  # noqa: E402
    PrimeProbeSchedule,
)
from sequence_game.experiments.histogram_colearning.prime_probe_corpus import (  # noqa: E402
    DEFAULT_CORPUS_RUN_DIR,
    DEFAULT_PRIME_PROBE_RUN_DIR,
    build_prime_probe_manifest,
    run_prime_probe_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-run-dir", type=Path, default=DEFAULT_CORPUS_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRIME_PROBE_RUN_DIR)
    parser.add_argument("--graph-workers", type=int, default=5)
    parser.add_argument("--samples-per-cell", type=int, default=64)
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args()

    corpus_manifest = json.loads(
        (args.corpus_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest_path = args.output_dir / "run_manifest.json"
    if args.reuse_manifest or manifest_path.exists():
        study_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        study_manifest = build_prime_probe_manifest(
            corpus_manifest,
            schedule=PrimeProbeSchedule(),
            samples_per_cell_per_split=args.samples_per_cell,
        )
    summary = run_prime_probe_study(
        corpus_manifest,
        study_manifest,
        output_dir=args.output_dir,
        graph_workers=args.graph_workers,
        progress_callback=_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    main()
