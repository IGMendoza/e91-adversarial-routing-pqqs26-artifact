#!/usr/bin/env python3
"""Build the corpus-wide stationary Stackelberg paper figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.stackelberg_followup_artifacts import (  # noqa: E402
    build_stackelberg_followup_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs/histogram_stackelberg_followup_105_500k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper/pqqs2026/figures/stackelberg_target_route_mass.pdf",
    )
    args = parser.parse_args()
    print(json.dumps(build_stackelberg_followup_artifacts(args.run_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
