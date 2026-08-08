#!/usr/bin/env python3
"""Calibrate and evaluate strict Stackelberg route steering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_steering_corpus import (
    DEFAULT_CORPUS_RUN_DIR,
    DEFAULT_STACKELBERG_STEERING_RUN_DIR,
    build_stackelberg_steering_manifest,
    run_stackelberg_steering_study,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-run-dir",
        type=Path,
        default=DEFAULT_CORPUS_RUN_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STACKELBERG_STEERING_RUN_DIR,
    )
    parser.add_argument(
        "--reuse-calibration-from",
        type=Path,
        default=None,
        help=(
            "reuse verified calibration rows from a completed steering run; "
            "the deterministic payoff bank is rebuilt and digest-checked"
        ),
    )
    parser.add_argument("--graph-workers", type=int, default=5)
    parser.add_argument(
        "--evaluation-margin-override",
        type=float,
        default=None,
        help=(
            "evaluate one declared margin on every graph/defender while "
            "retaining the calibrated positive-margin selection"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    corpus_manifest = json.loads(
        (args.corpus_run_dir / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    study = build_stackelberg_steering_manifest(
        corpus_manifest,
        evaluation_margin_override=args.evaluation_margin_override,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        print(json.dumps(study["preflight"], indent=2, sort_keys=True))
        return
    result = run_stackelberg_steering_study(
        corpus_manifest,
        study,
        output_dir=args.output_dir,
        graph_workers=args.graph_workers,
        calibration_source_dir=args.reuse_calibration_from,
        progress_callback=lambda message: print(message, flush=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
