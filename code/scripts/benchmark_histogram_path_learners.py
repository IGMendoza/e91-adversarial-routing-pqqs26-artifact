#!/usr/bin/env python3
"""Run a matched histogram benchmark of calibrated Maiti/Ito and baselines."""

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
    HistogramCoLearningConfig,
)
from sequence_game.experiments.histogram_colearning.corpus import (  # noqa: E402
    load_topology_bench_case,
)
from sequence_game.experiments.histogram_colearning.modern_path_benchmark import (  # noqa: E402
    run_modern_path_benchmark,
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
    parser.add_argument("--turns", type=int, default=1_000)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument("--posthoc-samples-per-cell", type=int, default=0)
    parser.add_argument("--evaluated-cache-size", type=int, default=256)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("docs/modern_path_learner_calibration.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/modern_path_benchmark/benchmark.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    case = load_topology_bench_case(
        args.corpus_db,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
    )
    config = HistogramCoLearningConfig(
        turns=args.turns,
        final_window=min(args.final_window, args.turns),
        record_every=args.record_every,
        posthoc_samples_per_cell=args.posthoc_samples_per_cell,
        evaluated_cache_size=args.evaluated_cache_size,
    )
    result = run_modern_path_benchmark(
        case,
        config,
        calibration,
        seeds=tuple(int(value) for value in args.seeds.split(",")),
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
    for row in result["comparison_rows"]:
        print(
            f"{row['label']}: mean Alice={row['mean_alice_utility']:.4f}; "
            f"final Alice={row['final_alice_utility']:.4f}; "
            f"throughput={row['turns_per_second']:.1f} turns/s"
        )
    return result


if __name__ == "__main__":
    main()
