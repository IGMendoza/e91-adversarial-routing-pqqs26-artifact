"""Tables and figures for two-sided histogram learner cross-play."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .crossplay_validation import (
    CELL_ALICE_STRUCTURED,
    CELL_BASELINE,
    CELL_BOTH_STRUCTURED,
    CELL_EVE_STRUCTURED,
    CELL_ORDER,
    LIVE_SCALAR_METRICS,
    REFERENCE_METRICS,
)


CELL_LABELS = {
    CELL_BASELINE: "FF",
    CELL_ALICE_STRUCTURED: "SF",
    CELL_EVE_STRUCTURED: "FS",
    CELL_BOTH_STRUCTURED: "SS",
}


def write_two_sided_crossplay_artifacts(
    output_dir: str | Path,
    *,
    case_results: dict[str, dict[str, Any]],
    source_bundles: dict[str, str],
) -> dict[str, Any]:
    """Write a self-describing cross-play result bundle."""

    if not case_results:
        raise ValueError("at least one cross-play case result is required")
    output = Path(output_dir)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "schema": "histogram_two_sided_crossplay_bundle.v1",
        "case_count": len(case_results),
        "case_slugs": list(case_results),
        "source_bundles": dict(source_bundles),
        "cases": case_results,
    }
    result_path = output / "two_sided_crossplay.json"
    _write_json(result_path, combined)
    table_paths = [
        _write_rows(
            table_dir / "live_cell_metrics.csv",
            _live_cell_rows(case_results),
        ),
        _write_rows(
            table_dir / "factorial_contrasts.csv",
            _factorial_contrast_rows(case_results),
        ),
        _write_rows(
            table_dir / "full_source_crossplay.csv",
            _source_crossplay_rows(case_results),
        ),
        _write_rows(
            table_dir / "learner_family_crossplay.csv",
            _family_crossplay_rows(case_results),
        ),
        _write_rows(
            table_dir / "joint_live_vs_offline.csv",
            _joint_comparison_rows(case_results),
        ),
    ]
    figure_paths: list[Path] = []
    for case, result in case_results.items():
        case_dir = figure_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        figure_paths.extend(_write_live_grid(case_dir / "live_two_sided_grid", result))
        figure_paths.extend(
            _write_source_crossplay_grid(
                case_dir / "full_source_crossplay",
                result,
            )
        )
        figure_paths.extend(
            _write_joint_comparison(
                case_dir / "joint_live_vs_offline",
                result,
            )
        )
    paths = [result_path, *table_paths, *figure_paths]
    manifest = {
        "schema": "histogram_two_sided_crossplay_artifacts.v1",
        "source_bundles": dict(source_bundles),
        "case_slugs": list(case_results),
        "files": [str(path.relative_to(output)) for path in paths],
        "file_count": len(paths),
        "online_information_boundary": (
            "cross-play uses learned marginal policies and the independent audit "
            "only after all online runs have finished"
        ),
        "cross_seed_inference_boundary": (
            "cross-seed grids are descriptive; paired confidence intervals use "
            "the three live seeds"
        ),
    }
    manifest_path = output / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["result_path"] = str(result_path)
    return manifest


def _live_cell_rows(
    case_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case, result in case_results.items():
        modes = result["learner_modes"]
        for replicate in result["live_profiles"]:
            for cell in CELL_ORDER:
                profile = replicate["cells"][cell]
                rows.append({
                    "case": case,
                    "seed": replicate["seed"],
                    "cell": cell,
                    "cell_label": CELL_LABELS[cell],
                    "alice_learner": modes[cell]["alice"],
                    "eve_learner": modes[cell]["eve"],
                    **{metric: profile[metric] for metric in LIVE_SCALAR_METRICS},
                    "alice_entropy_nats": profile[
                        "alice_strategy_diagnostics"
                    ]["entropy_nats"],
                    "alice_effective_action_count": profile[
                        "alice_strategy_diagnostics"
                    ]["effective_action_count"],
                    "eve_entropy_nats": profile[
                        "eve_strategy_diagnostics"
                    ]["entropy_nats"],
                    "eve_effective_action_count": profile[
                        "eve_strategy_diagnostics"
                    ]["effective_action_count"],
                })
    return rows


def _factorial_contrast_rows(
    case_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case, result in case_results.items():
        for contrast, payload in result["paired_factorial_contrasts"].items():
            for metric, summary in payload["summary"].items():
                interval = summary["confidence_interval_95"]
                rows.append({
                    "case": case,
                    "contrast": contrast,
                    "definition": payload["definition"],
                    "metric": metric,
                    "mean": summary["mean"],
                    "sd": summary["sd"],
                    "confidence_interval_95_low": (
                        interval[0] if interval is not None else None
                    ),
                    "confidence_interval_95_high": (
                        interval[1] if interval is not None else None
                    ),
                    "sample_count": summary["sample_count"],
                })
    return rows


def _source_crossplay_rows(
    case_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case, result in case_results.items():
        crossplay = result["full_source_crossplay"]
        for alice_source in crossplay["alice_source_order"]:
            for eve_source in crossplay["eve_source_order"]:
                for profile in crossplay["cells"][alice_source][eve_source][
                    "profiles"
                ]:
                    rows.append({
                        "case": case,
                        "alice_source": alice_source,
                        "alice_source_label": CELL_LABELS[alice_source],
                        "eve_source": eve_source,
                        "eve_source_label": CELL_LABELS[eve_source],
                        **profile,
                    })
    return rows


def _family_crossplay_rows(
    case_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case, result in case_results.items():
        family = result["learner_family_crossplay"]
        for alice_family in family["alice_family_order"]:
            for eve_family in family["eve_family_order"]:
                cell = family["cells"][alice_family][eve_family]
                row = {
                    "case": case,
                    "alice_family": alice_family,
                    "eve_family": eve_family,
                    "profile_count": cell["profile_count"],
                }
                for metric, summary in cell["descriptive"].items():
                    row[f"{metric}_mean"] = summary["mean"]
                    row[f"{metric}_sd"] = summary["sd"]
                    row[f"{metric}_minimum"] = summary["minimum"]
                    row[f"{metric}_maximum"] = summary["maximum"]
                rows.append(row)
    return rows


def _joint_comparison_rows(
    case_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case, result in case_results.items():
        comparison = result["joint_live_vs_offline_factorial"]
        for profile in comparison["per_seed"]:
            row = {
                "case": case,
                "seed": profile["seed"],
                "alice_strategy_total_variation": profile[
                    "strategy_adaptation_total_variation"
                ]["alice"],
                "eve_strategy_total_variation": profile[
                    "strategy_adaptation_total_variation"
                ]["eve"],
            }
            for metric in REFERENCE_METRICS:
                row[f"offline_{metric}"] = profile["offline_factorial"][metric]
                row[f"live_{metric}"] = profile["live_both_structured"][metric]
                row[f"live_minus_offline_{metric}"] = profile[
                    "live_minus_offline"
                ][metric]
            rows.append(row)
    return rows


def _write_live_grid(stem: Path, result: dict[str, Any]) -> list[Path]:
    cell_grid = (
        (CELL_BASELINE, CELL_EVE_STRUCTURED),
        (CELL_ALICE_STRUCTURED, CELL_BOTH_STRUCTURED),
    )
    panels = (
        ("reference_exploitability", "Nash gap", ".4f"),
        ("reference_alice_best_response_gap", "Alice best-response gap", ".4f"),
        ("reference_eve_best_response_gap", "Eve best-response gap", ".4f"),
        (
            "reference_alice_equilibrium_set_total_variation",
            "Alice distance to equilibrium set",
            ".4f",
        ),
        (
            "reference_eve_equilibrium_set_total_variation",
            "Eve distance to equilibrium set",
            ".4f",
        ),
        (
            "mean_bell_qualified_rate_hz",
            "Bell-qualified rate (pairs/s)",
            ".2f",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
    fig.patch.set_facecolor("white")
    summary = result["live_cell_summary"]
    for axis, (metric, title, number_format) in zip(axes.ravel(), panels):
        matrix = np.asarray([
            [summary[cell][metric]["mean"] for cell in row]
            for row in cell_grid
        ])
        _heatmap(
            axis,
            matrix,
            row_labels=["Alice flat", "Alice structured"],
            column_labels=["Eve flat", "Eve structured"],
            title=title,
            number_format=number_format,
        )
    fig.suptitle("Live two-sided learner assignments", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _write_source_crossplay_grid(
    stem: Path,
    result: dict[str, Any],
) -> list[Path]:
    crossplay = result["full_source_crossplay"]
    panels = (
        ("reference_joint_equivalent_utility", "Equivalent Alice utility", ".4f"),
        ("reference_exploitability", "Nash gap", ".4f"),
        ("reference_alice_best_response_gap", "Alice best-response gap", ".4f"),
        ("reference_eve_best_response_gap", "Eve best-response gap", ".4f"),
        (
            "reference_alice_equilibrium_set_total_variation",
            "Alice distance to equilibrium set",
            ".4f",
        ),
        (
            "reference_eve_equilibrium_set_total_variation",
            "Eve distance to equilibrium set",
            ".4f",
        ),
    )
    labels = [CELL_LABELS[cell] for cell in CELL_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.0))
    fig.patch.set_facecolor("white")
    for axis, (metric, title, number_format) in zip(axes.ravel(), panels):
        _heatmap(
            axis,
            np.asarray(crossplay["mean_matrices"][metric], dtype=float),
            row_labels=labels,
            column_labels=labels,
            title=title,
            number_format=number_format,
        )
        axis.set_xlabel("Eve marginal source cell")
        axis.set_ylabel("Alice marginal source cell")
    fig.suptitle(
        "Full source-conditioned marginal cross-play\n"
        "FF=flat/flat, SF=structured Alice, FS=structured Eve, SS=both",
        fontsize=14,
    )
    fig.tight_layout()
    return _save_figure(fig, stem)


def _write_joint_comparison(
    stem: Path,
    result: dict[str, Any],
) -> list[Path]:
    rows = result["joint_live_vs_offline_factorial"]["per_seed"]
    live = result["live_profiles"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))
    fig.patch.set_facecolor("white")
    for row in rows:
        axes[0].plot(
            [0, 1],
            [
                row["offline_factorial"]["reference_exploitability"],
                row["live_both_structured"]["reference_exploitability"],
            ],
            marker="o",
            label=f"seed {row['seed']}",
        )
    axes[0].set_xticks([0, 1], ["offline factorial", "live SS"])
    axes[0].set_ylabel("Nash gap")
    axes[0].set_title("Joint structured profile")
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [row["strategy_adaptation_total_variation"]["alice"] for row in rows],
        width,
        label="Alice",
        color="#0072B2",
    )
    axes[1].bar(
        x + width / 2,
        [row["strategy_adaptation_total_variation"]["eve"] for row in rows],
        width,
        label="Eve",
        color="#D55E00",
    )
    axes[1].set_xticks(x, [f"seed {row['seed']}" for row in rows])
    axes[1].set_ylabel("total variation")
    axes[1].set_title("Live adaptation from unilateral marginals")
    cell_x = np.arange(len(CELL_ORDER), dtype=float)
    for replicate in live:
        axes[2].plot(
            cell_x,
            [
                replicate["cells"][cell]["reference_exploitability"]
                for cell in CELL_ORDER
            ],
            marker="o",
            alpha=0.8,
            label=f"seed {replicate['seed']}",
        )
    axes[2].set_xticks(cell_x, [CELL_LABELS[cell] for cell in CELL_ORDER])
    axes[2].set_ylabel("Nash gap")
    axes[2].set_title("Four live learner-assignment cells")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _heatmap(
    axis: Any,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    number_format: str,
) -> None:
    image = axis.imshow(matrix, cmap="turbo", aspect="auto")
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            rgba = image.cmap(image.norm(matrix[row, column]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                column,
                row,
                format(float(matrix[row, column]), number_format),
                ha="center",
                va="center",
                color="black" if luminance > 0.52 else "white",
                fontsize=9,
            )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"table {path.name} has inconsistent columns")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
