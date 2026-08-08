#!/usr/bin/env python3
"""Generate full two-sided cross-play artifacts from four learner bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.crossplay_artifacts import (  # noqa: E402,E501
    write_two_sided_crossplay_artifacts,
)
from sequence_game.experiments.histogram_colearning.crossplay_validation import (  # noqa: E402,E501
    two_sided_crossplay_validation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--alice-structured-bundle", type=Path, required=True)
    parser.add_argument("--eve-structured-bundle", type=Path, required=True)
    parser.add_argument("--both-structured-bundle", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    bundles = {
        "baseline": args.baseline_bundle,
        "alice_structured": args.alice_structured_bundle,
        "eve_structured": args.eve_structured_bundle,
        "both_structured": args.both_structured_bundle,
    }
    cases = args.case or _common_cases(*bundles.values())
    if not cases:
        raise ValueError("no common run cases found across the four bundles")
    results = {
        case: two_sided_crossplay_validation(
            baseline_payloads=_load_runs(args.baseline_bundle, case),
            alice_structured_payloads=_load_runs(
                args.alice_structured_bundle,
                case,
            ),
            eve_structured_payloads=_load_runs(args.eve_structured_bundle, case),
            both_structured_payloads=_load_runs(
                args.both_structured_bundle,
                case,
            ),
        )
        for case in cases
    }
    manifest = write_two_sided_crossplay_artifacts(
        args.out_dir,
        case_results=results,
        source_bundles={key: str(value) for key, value in bundles.items()},
    )
    print(
        f"wrote two-sided cross-play for {len(cases)} case(s) to {args.out_dir}",
        flush=True,
    )
    return {"results": results, "manifest": manifest}


def _load_runs(bundle: Path, case: str) -> list[dict[str, Any]]:
    paths = sorted((bundle / "runs" / case).glob("seed_*.json"))
    if not paths:
        raise ValueError(f"no runs found for {case!r} under {bundle}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _common_cases(*bundles: Path) -> list[str]:
    case_sets = []
    for bundle in bundles:
        root = bundle / "runs"
        case_sets.append({
            path.name
            for path in root.iterdir()
            if path.is_dir() and any(path.glob("seed_*.json"))
        })
    return sorted(set.intersection(*case_sets))


if __name__ == "__main__":
    main()
