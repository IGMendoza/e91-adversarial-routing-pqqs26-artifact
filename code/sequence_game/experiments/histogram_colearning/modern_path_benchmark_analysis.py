"""Analysis and figures for matched modern path-learner benchmarks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import t as student_t


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


LABEL_ORDER = (
    "flat_exp3",
    "linear_exp2_g_optimal",
    "maiti_dag_ftrl",
    "ito_dag_ftrl_tsallis",
    "ito_dag_ftrl_log_barrier",
)
LABEL_DISPLAY = {
    "flat_exp3": "Flat Exp3",
    "linear_exp2_g_optimal": "LinearExp2\nG-optimal",
    "maiti_dag_ftrl": "Maiti\nDAG FTRL",
    "ito_dag_ftrl_tsallis": "Ito\nTsallis",
    "ito_dag_ftrl_log_barrier": "Ito\nlog barrier",
}
COLORS = {
    "flat_exp3": "#7f7f7f",
    "linear_exp2_g_optimal": "#0072B2",
    "maiti_dag_ftrl": "#D55E00",
    "ito_dag_ftrl_tsallis": "#009E73",
    "ito_dag_ftrl_log_barrier": "#CC79A7",
}
METRICS = (
    "mean_alice_utility",
    "final_alice_utility",
    "zero_sum_equivalent_exploitability",
    "alice_false_signal_rate_all_turns",
    "alice_false_signal_policy_total_variation",
    "turns_per_second",
)


def analyze_modern_path_benchmark(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "histogram_modern_path_benchmark.v1":
        raise ValueError("unsupported modern path benchmark schema")
    rows = list(payload["comparison_rows"])
    grouped = {
        label: [row for row in rows if row["label"] == label]
        for label in LABEL_ORDER
    }
    if any(not group for group in grouped.values()):
        raise ValueError("benchmark is missing one or more declared learners")
    aggregates = []
    for label, group in grouped.items():
        metric_rows = {
            metric: _sample_summary([float(row[metric]) for row in group])
            for metric in METRICS
        }
        warmed_throughput = [
            float(row["turns_per_second"])
            for row in group
            if not (label == "flat_exp3" and row["seed"] == min(
                member["seed"] for member in group
            ))
        ]
        aggregates.append({
            "label": label,
            "display_label": LABEL_DISPLAY[label].replace("\n", " "),
            "seed_count": len(group),
            "metrics": metric_rows,
            "warmed_throughput": (
                _sample_summary(warmed_throughput)
                if warmed_throughput else None
            ),
        })
    paired = []
    for comparator in LABEL_ORDER[1:]:
        for baseline in ("flat_exp3", "linear_exp2_g_optimal"):
            if comparator == baseline:
                continue
            paired.append({
                "comparator": comparator,
                "baseline": baseline,
                "mean_alice_utility_delta": _paired_summary(
                    grouped[comparator], grouped[baseline],
                    "mean_alice_utility",
                ),
                "exploitability_delta": _paired_summary(
                    grouped[comparator], grouped[baseline],
                    "zero_sum_equivalent_exploitability",
                ),
            })
    return {
        "schema": "histogram_modern_path_benchmark_analysis.v1",
        "source_schema": payload["schema"],
        "backend": payload["backend"],
        "case": dict(payload["case"]),
        "seeds": list(payload["seeds"]),
        "turns": int(payload["base_config"]["turns"]),
        "posthoc_samples_per_cell": int(
            payload["base_config"]["posthoc_samples_per_cell"]
        ),
        "aggregates": aggregates,
        "paired_differences": paired,
        "interpretation_boundary": (
            "n=3 t intervals are descriptive and wide; positive utility delta "
            "favors the comparator, negative exploitability delta favors it"
        ),
    }


def write_modern_path_benchmark_analysis(
    payload: Mapping[str, Any],
    out_dir: str | Path,
) -> list[Path]:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze_modern_path_benchmark(payload)
    paths = []
    analysis_path = output / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.append(analysis_path)
    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "seed", *METRICS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames}
                         for row in payload["comparison_rows"])
    paths.append(csv_path)
    paths.extend(_write_figures(payload, analysis, output))
    report_path = output / "report.md"
    report_path.write_text(_report_markdown(analysis), encoding="utf-8")
    paths.append(report_path)
    return paths


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be non-empty and finite")
    mean = float(np.mean(array))
    if array.size == 1:
        sd = se = half_width = 0.0
    else:
        sd = float(np.std(array, ddof=1))
        se = sd / math.sqrt(array.size)
        half_width = float(
            student_t.ppf(0.975, df=array.size - 1) * se
        )
    return {
        "n": int(array.size),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": se,
        "t95_low": mean - half_width,
        "t95_high": mean + half_width,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _paired_summary(
    comparator: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    left = {int(row["seed"]): float(row[metric]) for row in comparator}
    right = {int(row["seed"]): float(row[metric]) for row in baseline}
    if set(left) != set(right):
        raise ValueError("paired benchmark seeds do not match")
    values = [left[seed] - right[seed] for seed in sorted(left)]
    return {**_sample_summary(values), "by_seed": dict(zip(sorted(left), values))}


def _write_figures(
    payload: Mapping[str, Any],
    analysis: Mapping[str, Any],
    output: Path,
) -> list[Path]:
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for label in LABEL_ORDER:
        curves = [
            row["summary"]["learning_curve"] for row in payload["runs"]
            if row["benchmark_spec"]["label"] == label
        ]
        turns = np.asarray([row["turn"] for row in curves[0]], dtype=float)
        utilities = np.asarray([
            [row["alice_utility_mean"] for row in curve] for curve in curves
        ])
        mean = utilities.mean(axis=0)
        se = utilities.std(axis=0, ddof=1) / math.sqrt(len(curves))
        axes[0].plot(turns, mean, color=COLORS[label], label=LABEL_DISPLAY[label])
        axes[0].fill_between(
            turns, mean - se, mean + se, color=COLORS[label], alpha=0.16
        )
    axes[0].set(
        xlabel="Online turn",
        ylabel="Cumulative mean Alice utility",
        title="Matched 100k co-learning trajectories",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    aggregates = {row["label"]: row for row in analysis["aggregates"]}
    x = np.arange(len(LABEL_ORDER))
    utility = [
        aggregates[label]["metrics"]["mean_alice_utility"]["mean"]
        for label in LABEL_ORDER
    ]
    utility_se = [
        aggregates[label]["metrics"]["mean_alice_utility"]["standard_error"]
        for label in LABEL_ORDER
    ]
    axes[1].bar(
        x, utility, yerr=utility_se,
        color=[COLORS[label] for label in LABEL_ORDER], capsize=3,
    )
    axes[1].set_xticks(x, [LABEL_DISPLAY[label] for label in LABEL_ORDER])
    axes[1].set(
        ylabel="Mean Alice utility (mean ± SE)",
        title="Three matched seeds",
        ylim=(0.25, 0.53),
    )
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"modern_path_learning_comparison.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    exploitability = [
        aggregates[label]["metrics"][
            "zero_sum_equivalent_exploitability"
        ]["mean"] for label in LABEL_ORDER
    ]
    exploitability_se = [
        aggregates[label]["metrics"][
            "zero_sum_equivalent_exploitability"
        ]["standard_error"] for label in LABEL_ORDER
    ]
    axes[0].bar(
        x, exploitability, yerr=exploitability_se,
        color=[COLORS[label] for label in LABEL_ORDER], capsize=3,
    )
    axes[0].set_xticks(x, [LABEL_DISPLAY[label] for label in LABEL_ORDER])
    axes[0].set(
        ylabel="Audited zero-sum-equivalent exploitability",
        title="Lower is better; 16 samples per cell",
    )
    axes[0].grid(axis="y", alpha=0.2)

    route_policies = []
    for label in LABEL_ORDER:
        strategies = np.asarray([
            row["summary"]["alice_strategy"] for row in payload["runs"]
            if row["benchmark_spec"]["label"] == label
        ])
        route_policies.append(strategies.mean(axis=0))
    image = axes[1].imshow(
        np.asarray(route_policies), aspect="auto", cmap="turbo", vmin=0.0,
    )
    axes[1].set_yticks(x, [LABEL_DISPLAY[label].replace("\n", " ")
                           for label in LABEL_ORDER])
    axes[1].set_xticks(np.arange(len(route_policies[0])))
    axes[1].set(
        xlabel="Declared route index",
        title="Mean empirical route strategy",
    )
    fig.colorbar(image, ax=axes[1], label="Probability")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"modern_path_strategy_audit.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _report_markdown(analysis: Mapping[str, Any]) -> str:
    rows = []
    for row in analysis["aggregates"]:
        metrics = row["metrics"]
        rows.append(
            f"| {row['display_label']} | "
            f"{metrics['mean_alice_utility']['mean']:.4f} ± "
            f"{metrics['mean_alice_utility']['standard_error']:.4f} | "
            f"{metrics['zero_sum_equivalent_exploitability']['mean']:.4f} ± "
            f"{metrics['zero_sum_equivalent_exploitability']['standard_error']:.4f} | "
            f"{metrics['alice_false_signal_rate_all_turns']['mean']:.4%} |"
        )
    return "\n".join([
        "# Modern path learner 100k comparison",
        "",
        "Three matched seeds on the 23-route long composable Topology Bench case; "
        "100,000 online turns and 16 independent audit samples per route--attack cell.",
        "",
        "| Learner | Mean Alice utility ± SE | Exploitability ± SE | False-signal rate |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "Ito--Tsallis and Ito--log-barrier preserve mean service utility at the "
        "G-optimal LinearExp2 level. Their three-seed exploitability does not "
        "establish superiority: Tsallis is statistically noisy and log barrier "
        "is slightly worse in point estimate. Maiti at the theorem-scale "
        "schedule is not competitive and remains highly exploitable.",
        "",
        "Maiti schedule, implicit-exploration, and valid signed-loss-baseline "
        "sweeps did not restore parity on disjoint calibration seeds. Retain it "
        "as a literature/scaling implementation, not as the active learner.",
        "",
        "The edge-additive theorem assumptions remain an approximation to the "
        "histogram utility matrix; simultaneous Eve learning uses the adversarial "
        "rather than stationary interpretation.",
        "",
    ])
