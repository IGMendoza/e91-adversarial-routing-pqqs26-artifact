#!/usr/bin/env python3
"""Compare flat and structured histogram learners on one shared payoff audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.learner_validation import (  # noqa: E402
    multi_case_paired_learner_validation,
    paired_learner_validation,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-bundle", type=Path, required=True)
    parser.add_argument("--structured-bundle", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "case slug to analyze; repeat for several. By default, analyze every "
            "case present in both bundles"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    cases = args.case or _common_cases(args.flat_bundle, args.structured_bundle)
    if not cases:
        raise ValueError("no common run cases found in flat and structured bundles")
    flat_by_case = {case: _load_runs(args.flat_bundle, case) for case in cases}
    structured_by_case = {
        case: _load_runs(args.structured_bundle, case) for case in cases
    }
    if len(cases) == 1:
        result = paired_learner_validation(
            flat_payloads=flat_by_case[cases[0]],
            structured_payloads=structured_by_case[cases[0]],
        )
    else:
        result = multi_case_paired_learner_validation(
            flat_payloads_by_case=flat_by_case,
            structured_payloads_by_case=structured_by_case,
        )
    output = args.out_dir or (
        args.structured_bundle / "analysis" /
        ("paired_validation" if len(cases) == 1 else "paired_validation_all_cases")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_learner_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if len(cases) == 1:
        _write_csv(output / "paired_learner_validation.csv", result)
        _write_figure(output / "paired_learner_validation", result)
    else:
        _write_multi_csv(output / "paired_learner_validation.csv", result)
        _write_multi_figure(output / "paired_learner_validation", result)
        for case, case_result in result["cases"].items():
            case_dir = output / "cases" / case
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(case_dir / "paired_learner_validation.csv", case_result)
            _write_figure(case_dir / "paired_learner_validation", case_result)
    print(
        f"wrote paired validation for {len(result['matched_seeds'])} seeds "
        f"across {len(cases)} case(s) to {output}",
        flush=True,
    )
    return result


def _load_runs(bundle: Path, case: str) -> list[dict[str, Any]]:
    paths = sorted((bundle / "runs" / case).glob("seed_*.json"))
    if not paths:
        raise ValueError(f"no runs found for {case!r} under {bundle}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _common_cases(flat_bundle: Path, structured_bundle: Path) -> list[str]:
    def cases(bundle: Path) -> set[str]:
        root = bundle / "runs"
        return {
            path.name for path in root.iterdir()
            if path.is_dir() and any(path.glob("seed_*.json"))
        }

    return sorted(cases(flat_bundle) & cases(structured_bundle))


def _write_csv(path: Path, result: dict[str, Any]) -> None:
    metric_names = list(result["pairs"][0]["structured_minus_flat"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *metric_names])
        writer.writeheader()
        for pair in result["pairs"]:
            writer.writerow({"seed": pair["seed"], **pair["structured_minus_flat"]})


def _write_figure(stem: Path, result: dict[str, Any]) -> None:
    pairs = result["pairs"]
    seeds = [pair["seed"] for pair in pairs]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for index, pair in enumerate(pairs):
        axes[0].plot(
            [0, 1],
            [pair["flat"]["reference_exploitability"],
             pair["structured"]["reference_exploitability"]],
            marker="o",
            label=f"seed {seeds[index]}",
        )
        common = pair["common_opponent_alice_regret"]["oracle_eve"]
        axes[1].plot(
            [0, 1],
            [common["flat_alice_regret"], common["structured_alice_regret"]],
            marker="o",
        )
    for axis, title, ylabel in (
        (axes[0], "Total exploitability", "Nash gap"),
        (axes[1], "Alice regret vs shared oracle Eve", "best-response gap"),
    ):
        axis.set_xticks([0, 1], ["flat Exp3", "linear Exp2"])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    metrics = [
        "reference_exploitability",
        "reference_alice_best_response_gap",
        "reference_eve_best_response_gap",
        "alice_regret_against_oracle_eve",
    ]
    labels = ["total", "Alice/joint", "Eve/joint", "Alice/oracle Eve"]
    summaries = result["paired_delta_summary"]
    means = np.asarray([summaries[name]["mean"] for name in metrics])
    intervals = [summaries[name]["confidence_interval_95"] for name in metrics]
    lower = means - np.asarray([interval[0] for interval in intervals])
    upper = np.asarray([interval[1] for interval in intervals]) - means
    axes[2].errorbar(
        np.arange(len(metrics)), means, yerr=np.vstack([lower, upper]),
        fmt="o", capsize=4, color="#355f8d",
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xticks(np.arange(len(metrics)), labels, rotation=25, ha="right")
    axes[2].set_title("Structured − flat, paired 95% CI")
    axes[2].set_ylabel("delta (negative favors structured)")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Shared independent audit: "
        f"{result['reference_audit']['samples_per_cell'][0]} samples/cell",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)


def _write_multi_csv(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for case, case_result in result["cases"].items():
        for pair in case_result["pairs"]:
            rows.append({
                "case": case,
                "seed": pair["seed"],
                **pair["structured_minus_flat"],
            })
    fields = ["case", "seed", *result["topology_averaged_paired_delta_summary"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_multi_figure(stem: Path, result: dict[str, Any]) -> None:
    cases = list(result["cases"])
    metrics = (
        "reference_exploitability",
        "alice_regret_against_oracle_eve",
    )
    titles = ("Total exploitability", "Alice regret vs shared oracle Eve")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    x = np.arange(len(cases) + 1)
    labels = [*cases, "topology\naverage"]
    for axis, metric, title in zip(axes, metrics, titles):
        summaries = [
            result["cases"][case]["paired_delta_summary"][metric]
            for case in cases
        ] + [result["topology_averaged_paired_delta_summary"][metric]]
        means = np.asarray([row["mean"] for row in summaries])
        intervals = [row["confidence_interval_95"] for row in summaries]
        lower = means - np.asarray([row[0] for row in intervals])
        upper = np.asarray([row[1] for row in intervals]) - means
        colors = ["#3b528b"] * len(cases) + ["#5ec962"]
        axis.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="none",
                      color="#202020", capsize=4, zorder=2)
        axis.scatter(x, means, c=colors, s=48, zorder=3)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_title(title)
        axis.set_ylabel("structured − flat (negative favors structured)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Shared independent audits; three matched seeds per predeclared topology",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={
        "CreationDate": None,
        "ModDate": None,
    })
    plt.close(fig)


if __name__ == "__main__":
    main()
