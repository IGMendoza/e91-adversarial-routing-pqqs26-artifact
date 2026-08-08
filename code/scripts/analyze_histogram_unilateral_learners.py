#!/usr/bin/env python3
"""Compare matched baseline, Alice-structured, and Eve-structured runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from sequence_game.experiments.histogram_colearning.learner_validation import (  # noqa: E402
    unilateral_learner_validation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--alice-structured-bundle", type=Path, required=True)
    parser.add_argument("--eve-structured-bundle", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="case slug to analyze; repeat for several",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    bundles = (
        args.baseline_bundle,
        args.alice_structured_bundle,
        args.eve_structured_bundle,
    )
    cases = args.case or _common_cases(*bundles)
    if not cases:
        raise ValueError("no common run cases found across the three bundles")
    results = {
        case: unilateral_learner_validation(
            baseline_payloads=_load_runs(args.baseline_bundle, case),
            alice_structured_payloads=_load_runs(
                args.alice_structured_bundle,
                case,
            ),
            eve_structured_payloads=_load_runs(args.eve_structured_bundle, case),
        )
        for case in cases
    }
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    combined = {
        "schema": "histogram_unilateral_learner_validation_bundle.v1",
        "cases": results,
        "case_count": len(results),
        "case_slugs": list(results),
        "source_bundles": {
            "baseline": str(args.baseline_bundle),
            "alice_structured": str(args.alice_structured_bundle),
            "eve_structured": str(args.eve_structured_bundle),
        },
        "online_information_boundary": (
            "all strategies were learned from selected-action scalar rewards; "
            "the shared audit is used only after learning"
        ),
    }
    (output / "unilateral_learner_validation.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for case, result in results.items():
        case_dir = output / "cases" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(case_dir / "unilateral_learner_validation.csv", result)
        _write_figure(case_dir / "unilateral_learner_validation", result)
    print(
        f"wrote unilateral validation for {len(cases)} case(s) to {output}",
        flush=True,
    )
    return combined


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


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    metrics = list(result["pairs"][0]["unilateral_minus_baseline"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", *metrics],
            lineterminator="\n",
        )
        writer.writeheader()
        for pair in result["pairs"]:
            writer.writerow({
                "seed": pair["seed"],
                **pair["unilateral_minus_baseline"],
            })


def _write_figure(stem: Path, result: dict[str, Any]) -> None:
    pairs = result["pairs"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    fig.patch.set_facecolor("white")
    for index, pair in enumerate(pairs):
        alice = pair["changed_player_common_opponent_regret"]["alice"]
        eve = pair["changed_player_common_opponent_regret"]["eve"]
        profiles = pair["offline_recombined_profiles"]
        axes[0].plot(
            [0, 1],
            [alice["baseline_eve"]["baseline"],
             alice["baseline_eve"]["structured"]],
            marker="o",
            label=f"seed {pair['seed']}",
        )
        axes[1].plot(
            [0, 1],
            [eve["baseline_alice"]["baseline"],
             eve["baseline_alice"]["structured"]],
            marker="o",
        )
        axes[2].plot(
            [0, 1],
            [profiles["baseline"]["reference_exploitability"],
             profiles["alice_changed_only"]["reference_exploitability"]],
            marker="o",
            color="#0072B2",
            alpha=0.75,
            label="structured Alice" if index == 0 else None,
        )
        axes[2].plot(
            [0, 1],
            [profiles["baseline"]["reference_exploitability"],
             profiles["eve_changed_only"]["reference_exploitability"]],
            marker="s",
            color="#D55E00",
            alpha=0.75,
            label="structured Eve" if index == 0 else None,
        )
    axes[0].set_title("Alice regret against baseline Eve")
    axes[0].set_xticks([0, 1], ["baseline", "structured Alice"])
    axes[1].set_title("Eve regret against baseline Alice")
    axes[1].set_xticks([0, 1], ["baseline", "structured Eve"])
    axes[2].set_title("Nash gap under one-player substitution")
    axes[2].set_xticks([0, 1], ["baseline", "one learner changed"])
    axes[0].set_ylabel("best-response gap")
    axes[1].set_ylabel("best-response gap")
    axes[2].set_ylabel("Nash gap")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
