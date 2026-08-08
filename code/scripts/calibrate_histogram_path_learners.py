#!/usr/bin/env python3
"""Calibrate Maiti/Ito path learners without using histogram payoff oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    DEFAULT_CORPUS_DB_PATH,
)
from sequence_game.experiments.histogram_colearning.corpus import (  # noqa: E402
    load_topology_bench_case,
)
from sequence_game.experiments.histogram_colearning.modern_path_calibration import (  # noqa: E402
    calibrate_modern_path_learners,
)
from sequence_game.experiments.histogram_colearning.paper import (  # noqa: E402
    DEFAULT_PAPER_CASES,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    long_case = next(
        case for case in DEFAULT_PAPER_CASES if case.slug == "long_composable"
    )
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--graph-id", default=long_case.graph_id)
    parser.add_argument("--assignment-id", default=long_case.assignment_id)
    parser.add_argument("--horizon", type=int, default=3_000)
    parser.add_argument(
        "--learner-horizon",
        type=int,
        default=None,
        help="Maiti schedule horizon; defaults to --horizon",
    )
    parser.add_argument("--calibration-seeds", default="101,102,103")
    parser.add_argument("--evaluation-seeds", default="201,202,203")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/modern_path_learner_calibration/calibration.json"),
    )
    return parser.parse_args(argv)


def _seeds(value: str) -> tuple[int, ...]:
    return tuple(int(row.strip()) for row in value.split(",") if row.strip())


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    case = load_topology_bench_case(
        args.corpus_db,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
    )
    result = calibrate_modern_path_learners(
        case,
        horizon=args.horizon,
        learner_horizon=args.learner_horizon,
        calibration_seeds=_seeds(args.calibration_seeds),
        evaluation_seeds=_seeds(args.evaluation_seeds),
        progress_callback=lambda message: print(message, flush=True),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(f"wrote {args.out}")
    for family, row in result["selected_by_family"].items():
        held_out = row["held_out_evaluation"]
        print(
            f"{family}: {row['candidate']['label']}; held-out mean regret/turn="
            f"{held_out['selection_score']:.6f}; worst scenario="
            f"{held_out['worst_scenario_mean_regret_per_turn']:.6f}"
        )
    return result


if __name__ == "__main__":
    main()
