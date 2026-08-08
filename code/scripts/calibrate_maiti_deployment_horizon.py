#!/usr/bin/env python3
"""Calibrate Maiti on a short prefix using the declared deployment schedule."""

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
    ModernPathCandidate,
    calibrate_modern_path_learners,
)
from sequence_game.experiments.histogram_colearning.paper import (  # noqa: E402
    DEFAULT_PAPER_CASES,
)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(row.strip()) for row in value.split(",") if row.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(row.strip()) for row in value.split(",") if row.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    long_case = next(
        case for case in DEFAULT_PAPER_CASES if case.slug == "long_composable"
    )
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DB_PATH)
    parser.add_argument("--graph-id", default=long_case.graph_id)
    parser.add_argument("--assignment-id", default=long_case.assignment_id)
    parser.add_argument("--prefix-turns", type=int, default=1_000)
    parser.add_argument("--deployment-turns", type=int, default=100_000)
    parser.add_argument("--eta-scales", default="1,4,16")
    parser.add_argument("--ix-scales", default="1")
    parser.add_argument("--calibration-seeds", default="111,112,113")
    parser.add_argument("--evaluation-seeds", default="211,212,213")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/maiti_deployment_calibration/calibration.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    case = load_topology_bench_case(
        args.corpus_db,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
    )
    candidates = tuple(
        ModernPathCandidate(
            family="maiti",
            label=f"maiti_eta{eta:g}_ix{ix:g}",
            parameters={
                "eta_scale": eta,
                "implicit_exploration_scale": ix,
                "failure_probability": 0.05,
                "reward_loss_baseline": 1.0,
            },
            literature_default=eta == 1.0 and ix == 1.0,
        )
        for eta in _floats(args.eta_scales)
        for ix in _floats(args.ix_scales)
    )
    result = calibrate_modern_path_learners(
        case,
        horizon=args.prefix_turns,
        learner_horizon=args.deployment_turns,
        calibration_seeds=_ints(args.calibration_seeds),
        evaluation_seeds=_ints(args.evaluation_seeds),
        candidates=candidates,
        progress_callback=lambda message: print(message, flush=True),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    selected = result["selected_by_family"]["maiti"]
    print(f"wrote {args.out}")
    print(
        f"selected {selected['candidate']['label']}; raw best "
        f"{selected['raw_empirical_best']['candidate']['label']}; held-out "
        f"regret/turn "
        f"{selected['held_out_evaluation']['selection_score']:.6f}"
    )
    return result


if __name__ == "__main__":
    main()
