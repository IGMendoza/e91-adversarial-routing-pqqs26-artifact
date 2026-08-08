#!/usr/bin/env python3
"""Render paper artifacts from the completed Tsallis-INF refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.tsallis_refinement import (  # noqa: E402
    DEFAULT_TSALLIS_REFINEMENT_OUTPUT_DIR,
)
from sequence_game.experiments.histogram_colearning.tsallis_refinement_artifacts import (  # noqa: E402
    render_tsallis_refinement_artifacts,
)


DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "docs" / "artifacts" / "tsallis_refinement_5topology"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_TSALLIS_REFINEMENT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    args = parser.parse_args()
    result = render_tsallis_refinement_artifacts(
        args.source_dir,
        args.artifact_dir,
    )
    print(
        f"wrote {len(result['files'])} Tsallis refinement artifacts to "
        f"{args.artifact_dir}"
    )


if __name__ == "__main__":
    main()
