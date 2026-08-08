#!/usr/bin/env python3
"""Validate and summarize the completed 500k Alice-detection rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe_detection_analysis import (  # noqa: E402
    PrimeProbeDetectionAnalysisError,
    analyze_prime_probe_detection_bundle,
)
from sequence_game.experiments.histogram_colearning.prime_probe_detection_corpus import (  # noqa: E402
    DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
    )
    args = parser.parse_args()
    try:
        analysis = analyze_prime_probe_detection_bundle(args.run_dir)
    except PrimeProbeDetectionAnalysisError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            analysis,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
