#!/usr/bin/env python3
"""Render publication figures for the Cernet detection study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_detection_artifacts import (
    render_stackelberg_detection_artifacts,
)
from sequence_game.experiments.histogram_colearning.stackelberg_detection_study import (
    DEFAULT_STACKELBERG_DETECTION_RUN_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_STACKELBERG_DETECTION_RUN_DIR,
    )
    args = parser.parse_args()
    for path in render_stackelberg_detection_artifacts(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()
