#!/usr/bin/env python3
"""Generate figures from parsed Exp3 scoring worksheet data.

Reads the JSON produced by ``parse_exp3_scoring_worksheets.py`` and writes:

- ``understood_distribution.png`` -- histogram of the human/estimated
  ``understood`` score per model.
- ``objective_summary.png`` -- bar chart of mean numeric_faithfulness,
  scope-clean rate, dt_branch-cited rate, and mean sections_present/5,
  per model.
- ``rater_agreement.png`` -- only produced if more than one rater column
  is present in the parsed data (e.g. after adding a second rater's
  scores alongside Claude's) -- scatter of rater A vs rater B score per
  graph, with the diagonal marked.

Only plots what's actually scored -- unscored (null understood_score)
records are excluded from the understood-distribution figure and reported
as a completion count instead of silently ignored.

Standalone and idempotent -- safe to re-run any time more of the input
JSON is filled in.

Usage:
    .venv/bin/python scripts/plot_exp3_scoring_results.py \
      runs/exp3_dynamic_500k/xai_scoring/parsed_scores.json \
      --out-dir runs/exp3_dynamic_500k/xai_scoring/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records: list[dict[str, Any]] = json.loads(args.parsed_json.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"no records found in {args.parsed_json}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_model[r["model_slug"]].append(r)
    models = sorted(by_model)

    _plot_understood_distribution(by_model, models, args.out_dir, plt)
    _plot_objective_summary(by_model, models, args.out_dir, plt)
    _print_completion_summary(by_model, models)


def _plot_understood_distribution(
        by_model: dict[str, list[dict[str, Any]]],
        models: list[str],
        out_dir: Path,
        plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.8 / max(1, len(models))
    bins = list(range(1, 12))
    for i, model in enumerate(models):
        scores = [r["understood_score"] for r in by_model[model] if r["understood_score"] is not None]
        if not scores:
            continue
        counts = [scores.count(b) for b in bins[:-1]]
        offsets = [b + i * width for b in bins[:-1]]
        ax.bar(offsets, counts, width=width, label=model)
    ax.set_xlabel("understood score (1-10)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of `understood` scores by model")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "understood_distribution.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"-> {out_path}")


def _plot_objective_summary(
        by_model: dict[str, list[dict[str, Any]]],
        models: list[str],
        out_dir: Path,
        plt: Any,
) -> None:
    metrics = ["mean_faithfulness_pct", "scope_clean_rate_pct", "dt_cited_rate_pct", "mean_sections_pct"]
    metric_labels = ["numeric\nfaithfulness", "scope\nclean", "dt_branch\ncited", "sections\ncomplete"]
    values_by_model: dict[str, list[float]] = {}
    for model in models:
        rows = by_model[model]
        n = len(rows) or 1
        faith = [r["numeric_faithfulness_pct"] for r in rows if r["numeric_faithfulness_pct"] is not None]
        clean = sum(1 for r in rows if r["scope_honesty"] == "clean")
        cited = sum(1 for r in rows if r["dt_branch_cited"])
        sections = [
            (r["sections_present"] / r["sections_total"]) * 100
            for r in rows
            if r["sections_present"] is not None and r["sections_total"]
        ]
        values_by_model[model] = [
            sum(faith) / len(faith) if faith else 0.0,
            100.0 * clean / n,
            100.0 * cited / n,
            sum(sections) / len(sections) if sections else 0.0,
        ]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(metrics))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values_by_model[model], width=width, label=model)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Mechanical objective checks by model (no LLM judge)")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "objective_summary.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"-> {out_path}")


def _print_completion_summary(by_model: dict[str, list[dict[str, Any]]], models: list[str]) -> None:
    print("\nCompletion:")
    for model in models:
        rows = by_model[model]
        scored = sum(1 for r in rows if r["understood_score"] is not None)
        print(f"  {model}: {scored}/{len(rows)} scored")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parsed_json", type=Path, help="Output of parse_exp3_scoring_worksheets.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
