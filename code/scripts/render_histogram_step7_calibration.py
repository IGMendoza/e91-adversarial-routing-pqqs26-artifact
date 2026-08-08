#!/usr/bin/env python3
"""Render tables, figures, and a report from a completed Step 7 run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.calibration_artifacts import (  # noqa: E402
    write_step7_calibration_artifacts,
)
from sequence_game.experiments.histogram_colearning.calibration_pipeline import (  # noqa: E402
    DEFAULT_STEP7_OUTPUT_DIR,
)
from sequence_game.experiments.histogram_colearning.calibration_cases import (  # noqa: E402
    load_frozen_calibration_cases,
)
from sequence_game.experiments.histogram_colearning.paper_topology_plots import (  # noqa: E402
    TopologyFigureCase,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STEP7_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--without-topologies",
        action="store_true",
        help="skip the frozen-network context figures",
    )
    args = parser.parse_args()
    topology_cases = None
    if not args.without_topologies:
        topology_cases = [
            TopologyFigureCase(
                slug=spec.family,
                structural_regime=(
                    spec.assignment_id.split("__", 1)[0].replace("_", " ")
                ),
                case=case,
            )
            for spec, case in load_frozen_calibration_cases()
        ]
    result = write_step7_calibration_artifacts(
        args.output_dir,
        artifact_dir=args.artifact_dir,
        topology_cases=topology_cases,
    )
    print(
        f"wrote {len(result['files'])} Step 7 artifacts to "
        f"{Path(result['manifest_path']).parent}"
    )


if __name__ == "__main__":
    main()
