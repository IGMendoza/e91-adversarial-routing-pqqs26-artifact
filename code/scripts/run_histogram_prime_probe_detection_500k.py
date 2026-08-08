#!/usr/bin/env python3
"""Run the matched-transcript Prime-and-Probe Alice-detection study."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe_detection_corpus import (  # noqa: E402
    DEFAULT_CORPUS_RUN_DIR,
    DEFAULT_MATCHED_SAMPLES_PER_CELL_PER_SPLIT,
    DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
    DEFAULT_PRIME_PROBE_DETECTION_SMOKE_RUN_DIR,
    DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
    build_prime_probe_detection_manifest,
    load_source_prime_probe_bundle,
    prime_probe_detection_preflight,
    run_prime_probe_detection_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-run-dir",
        type=Path,
        default=DEFAULT_CORPUS_RUN_DIR,
    )
    parser.add_argument(
        "--source-prime-probe-run-dir",
        type=Path,
        default=DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--graph-workers", type=int, default=5)
    parser.add_argument("--samples-per-cell", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run one short, non-paper verification on the smallest frozen "
            "source case."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Print the frozen resource plan without starting physics.",
    )
    args = parser.parse_args()

    output = args.output_dir or (
        DEFAULT_PRIME_PROBE_DETECTION_SMOKE_RUN_DIR
        if args.smoke
        else DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR
    )
    source_manifest, source_checkpoints = load_source_prime_probe_bundle(
        args.source_prime_probe_run_dir
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        study_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if bool(study_manifest.get("smoke")) != bool(args.smoke):
            raise ValueError(
                "existing output manifest smoke mode differs from --smoke"
            )
    else:
        samples = (
            args.samples_per_cell
            if args.samples_per_cell is not None
            else DEFAULT_MATCHED_SAMPLES_PER_CELL_PER_SPLIT
        )
        study_manifest = build_prime_probe_detection_manifest(
            source_manifest,
            source_checkpoints,
            samples_per_cell_per_split=samples,
            smoke=args.smoke,
        )
    preflight = prime_probe_detection_preflight(study_manifest)
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    corpus_manifest = json.loads(
        (args.corpus_run_dir / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    summary = run_prime_probe_detection_study(
        corpus_manifest,
        study_manifest,
        output_dir=output,
        graph_workers=(1 if args.smoke else args.graph_workers),
        progress_callback=_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _progress(message: str) -> None:
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


if __name__ == "__main__":
    main()
