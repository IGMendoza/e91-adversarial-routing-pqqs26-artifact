#!/usr/bin/env python3
"""Render the normalized Real Topologies 500k learner artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.corpus_scale import (  # noqa: E402
    DEFAULT_CORPUS_SCALE_OUTPUT_DIR,
)
from sequence_game.experiments.histogram_colearning.corpus_scale_artifacts import (  # noqa: E402
    write_corpus_scale_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_CORPUS_SCALE_OUTPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "artifacts" / "histogram_corpus_500k",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = write_corpus_scale_artifacts(
        args.run_dir,
        args.output_dir,
        require_complete=not args.allow_incomplete,
    )
    print(
        f"rendered {len(result['files'])} artifacts for "
        f"{result['graph_count']} matched graphs in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
