#!/usr/bin/env python3
"""Render the Iris outlier figures from the frozen 500k corpus run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.iris_outlier_artifacts import (
    write_iris_outlier_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/histogram_corpus_500k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "docs/artifacts/histogram_corpus_500k/iris_outlier"
        ),
    )
    args = parser.parse_args()
    result = write_iris_outlier_artifacts(args.run_dir, args.output_dir)
    print(
        f"rendered {len(result['files'])} Iris artifacts in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
