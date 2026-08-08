#!/usr/bin/env python3
"""Run manifest-locked two-player histogram learner calibration."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.calibration_pipeline import (  # noqa: E402
    DEFAULT_STEP7_OUTPUT_DIR,
    Step7CalibrationPlan,
    run_step7_calibration,
    step7_preflight,
)
from sequence_game.experiments.histogram_colearning.config import (  # noqa: E402
    HistogramCoLearningConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STEP7_OUTPUT_DIR)
    parser.add_argument(
        "--phase",
        choices=("banks", "screening", "confirmation", "all"),
        default="all",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help=(
            "run an explicit calibration-case subset; repeat for multiple "
            "cases (subset executions do not make a five-case claim)"
        ),
    )
    parser.add_argument("--samples-per-cell", type=int, default=64)
    parser.add_argument("--screening-turns", type=int, default=10_000)
    parser.add_argument("--screening-seeds", default="101,102,103")
    parser.add_argument("--confirmation-turns", type=int, default=100_000)
    parser.add_argument("--confirmation-seeds", default="201,202,203")
    parser.add_argument("--confirmation-record-every", type=int, default=10_000)
    parser.add_argument("--confirmation-workers", type=int, default=4)
    parser.add_argument("--confirmation-timing-turns", type=int, default=10_000)
    parser.add_argument("--confirmation-timing-seed", type=int, default=301)
    parser.add_argument("--requested-pairs-per-turn", type=int, default=1_200)
    parser.add_argument("--age-points-per-edge", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = Step7CalibrationPlan(
        samples_per_cell_per_split=args.samples_per_cell,
        screening_turns=args.screening_turns,
        screening_seeds=_parse_seeds(args.screening_seeds),
        confirmation_turns=args.confirmation_turns,
        confirmation_seeds=_parse_seeds(args.confirmation_seeds),
        confirmation_record_every=args.confirmation_record_every,
        confirmation_workers=args.confirmation_workers,
        confirmation_timing_turns=args.confirmation_timing_turns,
        confirmation_timing_seed=args.confirmation_timing_seed,
    )
    config = HistogramCoLearningConfig(
        turns=plan.screening_turns,
        requested_pairs_per_turn=args.requested_pairs_per_turn,
        posthoc_samples_per_cell=0,
        age_points_per_edge=args.age_points_per_edge,
        seed=plan.screening_seeds[0],
    )
    if args.preflight_only:
        import json

        print(json.dumps(step7_preflight(plan), indent=2, sort_keys=True))
        return
    result = run_step7_calibration(
        output_dir=args.output_dir,
        base_config=config,
        plan=plan,
        case_ids=args.case_ids,
        phase=args.phase,
        progress_callback=_progress,
    )
    print(
        f"completed phase={result['phase'] if 'phase' in result else args.phase} "
        f"output={args.output_dir}"
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds:
        raise ValueError("seed list must be nonempty")
    return seeds


def _progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


if __name__ == "__main__":
    main()
