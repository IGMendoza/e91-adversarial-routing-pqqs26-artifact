#!/usr/bin/env python3
"""Run the opt-in Cernet Stackelberg detection and mitigation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_detection_study import (
    DEFAULT_CORPUS_RUN_DIR,
    DEFAULT_STACKELBERG_DETECTION_RUN_DIR,
    build_cernet_detection_manifest,
    run_cernet_detection_study,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-extension",
        action="store_true",
        help=(
            "explicitly enable the separate detection/mitigation experiment; "
            "the existing Stackelberg runners are not modified"
        ),
    )
    parser.add_argument(
        "--corpus-run-dir",
        type=Path,
        default=DEFAULT_CORPUS_RUN_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STACKELBERG_DETECTION_RUN_DIR,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--turns", type=int, default=100_000)
    parser.add_argument("--samples-per-cell", type=int, default=96)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--bank-only",
        action="store_true",
        help="build and validate the disjoint payoff bank, then stop",
    )
    parser.add_argument("--force-rebuild-bank", action="store_true")
    args = parser.parse_args()
    if not args.enable_extension:
        parser.error(
            "--enable-extension is required; this keeps the study opt-in"
        )

    corpus_manifest = json.loads(
        (args.corpus_run_dir / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_cernet_detection_manifest(
        corpus_manifest,
        turns=args.turns,
        samples_per_cell_per_split=args.samples_per_cell,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        print(json.dumps(manifest["preflight"], indent=2, sort_keys=True))
        return
    summary = run_cernet_detection_study(
        corpus_manifest,
        manifest,
        output_dir=args.output_dir,
        workers=args.workers,
        force_rebuild_bank=args.force_rebuild_bank,
        bank_only=args.bank_only,
        progress_callback=lambda message: print(message, flush=True),
    )
    print(json.dumps(
        summary.get("aggregates", summary),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
