#!/usr/bin/env python3
"""Render publication artifacts for strict Stackelberg route steering."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_steering_artifacts import (
    write_stackelberg_steering_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "histogram_stackelberg_steering_500k",
    )
    parser.add_argument(
        "--boundary-run-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "histogram_stackelberg_steering_zero_margin_500k"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "artifacts"
            / "histogram_stackelberg_steering_500k"
        ),
    )
    parser.add_argument(
        "--corpus-run-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "histogram_corpus_500k",
    )
    args = parser.parse_args()
    result = write_stackelberg_steering_artifacts(
        args.strict_run_dir,
        args.boundary_run_dir,
        args.corpus_run_dir,
        args.output_dir,
    )
    print(
        f"rendered {len(result['files'])} Stackelberg steering artifacts "
        f"in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
