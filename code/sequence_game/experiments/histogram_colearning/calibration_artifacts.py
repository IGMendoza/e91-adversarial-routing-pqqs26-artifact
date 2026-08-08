"""Result tables and figures for Step 7 histogram learner calibration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .calibration_candidates import LEARNER_FAMILIES


FAMILY_LABELS = {
    "flat_exp3": "Exp3",
    "tsallis_inf": "Tsallis-INF",
    "exp3_plus_plus": "EXP3++",
    "linear_exp2": "LinearExp2",
}

FAMILY_COLORS = {
    "flat_exp3": "#555555",
    "tsallis_inf": "#0072B2",
    "exp3_plus_plus": "#009E73",
    "linear_exp2": "#D55E00",
}

TOPOLOGY_LABELS = {
    "cernet": "CERNET",
    "hiberniaireland": "Hibernia Ireland",
    "karen": "KAREN",
    "layer42": "Layer42",
    "memorex": "Memorex topology\n(1990s–early 2000s)",
}

STRATEGIC_METRICS = (
    ("exploitability", "Nash gap", ".4f"),
    ("alice_external_regret_per_turn", "Alice external regret", ".4f"),
    ("eve_external_regret_per_turn", "Eve external regret", ".4f"),
    (
        "alice_equilibrium_set_total_variation",
        "Alice equilibrium-set distance",
        ".3f",
    ),
    (
        "eve_equilibrium_set_total_variation",
        "Eve equilibrium-set distance",
        ".3f",
    ),
    ("mean_bell_qualified_rate_hz", "Accepted-service rate (pairs/s)", ".2f"),
)

SIGNAL_METRICS = (
    ("false_signal_rate", "Alice false-signal rate", ".4%"),
    (
        "false_signal_policy_total_variation",
        "Alice policy shift from false signals",
        ".4f",
    ),
    ("mean_alice_reward", "Alice public service reward", ".3f"),
    ("mean_bell_qualified_rate_hz", "Accepted-service rate (pairs/s)", ".2f"),
    ("turns_per_second", "Learner + transcript turns/s", ".0f"),
)


def write_step7_calibration_artifacts(
    output_dir: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    topology_cases: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Render a complete, manifest-checked Step 7 result bundle."""

    source = Path(output_dir)
    target = Path(artifact_dir) if artifact_dir is not None else source / "artifacts"
    plan = _read_json(source / "step7_plan.json")
    selection = _read_json(source / "screening_selection.json")
    confirmation = _read_json(source / "confirmation_summary.json")
    payoff = _read_json(
        source / "payoff_structure" / "payoff_structure_summary.json"
    )
    _validate_sources(plan, selection, confirmation, payoff)

    table_dir = target / "tables"
    figure_dir = target / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        _write_rows(
            table_dir / "screening_selections.csv",
            _screening_selection_rows(selection),
        ),
        _write_rows(
            table_dir / "confirmation_crossplay.csv",
            _confirmation_rows(confirmation),
        ),
        _write_rows(
            table_dir / "confirmation_by_topology.csv",
            _confirmation_case_rows(confirmation),
        ),
        _write_rows(
            table_dir / "payoff_structure.csv",
            list(payoff["cases"]),
        ),
    ]
    live_rows_path = source / "confirmation_rows.jsonl"
    schedule_rows: list[dict[str, Any]] = []
    if live_rows_path.exists():
        live_rows = _read_jsonl(live_rows_path)
        if len(live_rows) != int(confirmation["run_count"]):
            raise ValueError(
                "confirmation row count does not match confirmation summary"
            )
        schedule_rows = _confirmation_schedule_rows(live_rows)
        tables.append(_write_rows(
            table_dir / "confirmation_schedule_dynamics.csv",
            schedule_rows,
        ))
    figures = [
        *_plot_screening_profiles(
            selection,
            figure_dir / "screening_regret_profiles",
        ),
        *_plot_confirmation_heatmaps(
            confirmation,
            metrics=STRATEGIC_METRICS,
            stem=figure_dir / "confirmation_strategic_heatmaps",
        ),
        *_plot_confirmation_heatmaps(
            confirmation,
            metrics=SIGNAL_METRICS,
            stem=figure_dir / "confirmation_signal_heatmaps",
        ),
        *_plot_topology_exploitability(
            confirmation,
            figure_dir / "confirmation_topology_nash_gap",
        ),
        *_plot_payoff_structure(
            payoff,
            figure_dir / "payoff_structure_diagnostics",
        ),
    ]
    if schedule_rows:
        figures.extend(_plot_confirmation_schedule_dynamics(
            schedule_rows,
            figure_dir / "confirmation_schedule_dynamics",
        ))
    if topology_cases:
        from .calibration_topology_plots import (
            write_step7_calibration_topology_figures,
        )

        figures.extend(
            Path(path) for path in write_step7_calibration_topology_figures(
                figure_dir,
                topology_cases,
                confirmation,
            )
        )
    report_path = target / "STEP7_CALIBRATION_REPORT.md"
    report_path.write_text(
        _render_report(
            plan,
            selection,
            confirmation,
            payoff,
            schedule_dynamics_included=bool(schedule_rows),
            topology_figures_included=bool(topology_cases),
        ),
        encoding="utf-8",
    )
    files = [*tables, *figures, report_path]
    manifest = {
        "schema": "histogram_step7_calibration_artifacts.v1",
        "plan_schema": plan["schema"],
        "screening_schema": selection["schema"],
        "confirmation_schema": confirmation["schema"],
        "payoff_structure_schema": payoff["schema"],
        "screening_run_count": selection["run_count"],
        "confirmation_run_count": confirmation["run_count"],
        "evaluation_partition_loaded": False,
        "files": [str(path.relative_to(target)) for path in files],
    }
    manifest_path = target / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    return manifest


def _validate_sources(
    plan: dict[str, Any],
    selection: dict[str, Any],
    confirmation: dict[str, Any],
    payoff: dict[str, Any],
) -> None:
    if int(plan["case_count"]) != 5:
        raise ValueError("Step 7 plan must contain five fixed cases")
    if plan["information_boundary"].get("evaluation_partition_loaded") is not False:
        raise ValueError("Step 7 plan crossed the evaluation boundary")
    for name, payload in (
        ("screening", selection),
        ("confirmation", confirmation),
    ):
        if payload.get("complete") is not True:
            raise ValueError(f"{name} result is not complete")
        if int(payload["run_count"]) != int(payload["expected_run_count"]):
            raise ValueError(f"{name} run count does not match its contract")
        if payload.get("evaluation_partition_loaded") is not False:
            raise ValueError(f"{name} result crossed the evaluation boundary")
    if payoff.get("evaluation_partition_loaded") is not False:
        raise ValueError("payoff audit crossed the evaluation boundary")
    if payoff.get("selection_revisited_after_fit") is not False:
        raise ValueError("calibration case selection changed after payoff fitting")
    if int(payoff["case_count"]) != 5:
        raise ValueError("Step 7 payoff audit must contain five fixed cases")
    planned_case_ids = {row["case_id"] for row in plan["cases"]}
    confirmed_case_ids = set(
        confirmation["aggregate"][0]["case_means_exploitability"]
    )
    if planned_case_ids != confirmed_case_ids:
        raise ValueError("Step 7 plan and confirmation case identities differ")


def _screening_selection_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for player in ("alice", "eve"):
        for family in LEARNER_FAMILIES:
            result = selection[player]["selected_by_family"][family]
            selected = result["selected"]
            raw = result["raw_minimum"]
            rows.append({
                "player": player,
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "selected_label": selected["candidate"]["label"],
                "selected_mean_external_regret_per_turn": selected[
                    "mean_external_regret_per_turn"
                ],
                "selected_seed_block_standard_error": selected["standard_error"],
                "raw_minimum_label": raw["candidate"]["label"],
                "raw_minimum_external_regret_per_turn": raw[
                    "mean_external_regret_per_turn"
                ],
                "one_standard_error_threshold": result[
                    "one_standard_error_threshold"
                ],
                "eligible_labels": "|".join(result["eligible_labels"]),
            })
    return rows


def _confirmation_rows(confirmation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for aggregate in confirmation["aggregate"]:
        row = {
            "alice_family": aggregate["alice_family"],
            "eve_family": aggregate["eve_family"],
            "run_count": aggregate["run_count"],
        }
        for metric, _title, _number_format in (*STRATEGIC_METRICS, *SIGNAL_METRICS):
            field = f"mean_{metric}"
            if field in aggregate:
                row[field] = aggregate[field]
                error_field = (
                    "case_standard_error_turns_per_second"
                    if metric == "turns_per_second"
                    else f"seed_block_standard_error_{metric}"
                )
                row[f"standard_error_{metric}"] = aggregate[error_field]
        rows.append(row)
    return rows


def _confirmation_case_rows(
    confirmation: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for aggregate in confirmation["aggregate"]:
        case_ids = aggregate["case_means_exploitability"]
        for case_id in case_ids:
            row = {
                "case_id": case_id,
                "alice_family": aggregate["alice_family"],
                "eve_family": aggregate["eve_family"],
            }
            for metric, _title, _number_format in STRATEGIC_METRICS:
                row[metric] = aggregate[f"case_means_{metric}"][case_id]
            rows.append(row)
    return rows


def _plot_screening_profiles(
    selection: dict[str, Any],
    stem: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.0), sharex=False)
    fig.patch.set_facecolor("white")
    for row_index, player in enumerate(("alice", "eve")):
        summaries = selection[player]["candidate_summaries"]
        flat_regret = _candidate_summary(
            summaries,
            "flat_exp3_frozen",
        )["mean_external_regret_per_turn"]
        _plot_tsallis_panel(axes[row_index, 0], summaries, player)
        _plot_exp3pp_panel(axes[row_index, 1], summaries, player)
        _plot_linear_panel(axes[row_index, 2], summaries, player)
        for axis in axes[row_index]:
            axis.axhline(
                flat_regret,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                label="frozen Exp3" if axis is axes[row_index, 0] else None,
            )
            axis.grid(alpha=0.25)
            axis.set_xscale("log", base=2)
            if axis is axes[row_index, 0]:
                axis.set_ylabel(f"{player.title()} audit regret / turn")
    axes[0, 0].set_title("Tsallis-INF (3 eta x 2 estimators)")
    axes[0, 1].set_title("EXP3++ (5--7 eta x 3 beta values)")
    axes[0, 2].set_title("LinearExp2 (5 eta; G-optimal)")
    for axis in axes.ravel():
        axis.set_xlabel("eta multiplier")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "10k candidate screen — own-player regret on the reserved audit",
        fontsize=15,
    )
    fig.tight_layout()
    return _save_figure(fig, stem)


def _plot_tsallis_panel(
    axis: Any,
    summaries: Sequence[dict[str, Any]],
    player: str,
) -> None:
    field = f"{player}_tsallis_inf_eta_scale"
    for estimator, color in (("rv", "#0072B2"), ("iw", "#D55E00")):
        rows = [
            row for row in summaries
            if row["candidate"]["family"] == "tsallis_inf"
            and row["candidate"]["config_overrides"][
                f"{player}_tsallis_inf_estimator"
            ] == estimator
        ]
        _error_curve(axis, rows, x_field=field, label=estimator.upper(), color=color)


def _plot_exp3pp_panel(
    axis: Any,
    summaries: Sequence[dict[str, Any]],
    player: str,
) -> None:
    for beta, color, marker, linestyle in (
        (64.0, "#0072B2", "o", "-"),
        (256.0, "#009E73", "s", "--"),
        (1024.0, "#D55E00", "^", ":"),
    ):
        rows = [
            row for row in summaries
            if row["candidate"]["family"] == "exp3_plus_plus"
            and row["candidate"]["config_overrides"][
                f"{player}_exp3pp_exploration_beta"
            ] == beta
        ]
        _error_curve(
            axis,
            rows,
            x_field=f"{player}_exp3pp_eta_scale",
            label=f"beta {beta:g}",
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    axis.text(
        0.03,
        0.97,
        "beta curves coincide through eta 8",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#333333",
    )


def _plot_linear_panel(
    axis: Any,
    summaries: Sequence[dict[str, Any]],
    player: str,
) -> None:
    rows = [
        row for row in summaries
        if row["candidate"]["family"] == "linear_exp2"
    ]
    field = "alice_exp3_eta_c" if player == "alice" else "eve_exp3_eta_c"
    _error_curve(axis, rows, x_field=field, label="G-optimal", color="#CC79A7")


def _error_curve(
    axis: Any,
    rows: Sequence[dict[str, Any]],
    *,
    x_field: str,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: float(row["candidate"]["config_overrides"][x_field]),
    )
    x = [float(row["candidate"]["config_overrides"][x_field]) for row in ordered]
    y = [float(row["mean_external_regret_per_turn"]) for row in ordered]
    error = [float(row["standard_error"]) for row in ordered]
    axis.errorbar(
        x,
        y,
        yerr=error,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.8,
        capsize=3,
        color=color,
        label=label,
    )


def _confirmation_schedule_rows(
    live_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    for run in live_rows:
        trajectory = run.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("confirmation row has no strategic trajectory")
        for player in ("alice", "eve"):
            family = str(run[f"{player}_family"])
            action_count = int(run[f"{player}_runtime"]["action_count"])
            entropy_denominator = float(np.log(action_count))
            for point in trajectory:
                turn = int(point["turn"])
                metrics = {
                    "eta": point.get(f"{player}_eta"),
                    "explicit_exploration_mass": point.get(f"{player}_gamma"),
                    "uniform_coverage_mass": point.get(
                        f"{player}_policy_uniform_coverage_mass"
                    ),
                    "nonmodal_probability_mass": point.get(
                        f"{player}_policy_nonmodal_probability_mass"
                    ),
                    "normalized_policy_entropy": (
                        float(point[f"{player}_policy_entropy"])
                        / entropy_denominator
                        if action_count > 1 else 0.0
                    ),
                }
                for metric, value in metrics.items():
                    if value is None:
                        continue
                    grouped.setdefault(
                        (player, family, turn, metric),
                        [],
                    ).append(float(value))
    rows = []
    for (player, family, turn, metric), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        rows.append({
            "player": player,
            "family": family,
            "turn": turn,
            "metric": metric,
            "sample_count": int(array.size),
            "q25": float(np.quantile(array, 0.25)),
            "median": float(np.median(array)),
            "q75": float(np.quantile(array, 0.75)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        })
    return rows


def _plot_confirmation_schedule_dynamics(
    rows: Sequence[dict[str, Any]],
    stem: Path,
) -> list[Path]:
    available_metrics = {str(row["metric"]) for row in rows}
    metrics = [
        ("eta", "algorithm-internal learning rate", True),
        (
            "explicit_exploration_mass",
            r"explicitly mixed exploration mass $\gamma_t$",
            False,
        ),
    ]
    if "uniform_coverage_mass" in available_metrics:
        metrics.append((
            "uniform_coverage_mass",
            r"realized uniform coverage $K\min_i p_{t,i}$",
            False,
        ))
    if "nonmodal_probability_mass" in available_metrics:
        metrics.append((
            "nonmodal_probability_mass",
            r"realized nonmodal mass $1-\max_i p_{t,i}$",
            False,
        ))
    metrics.append((
        "normalized_policy_entropy",
        "normalized current-policy entropy",
        False,
    ))
    row_count = len(metrics)
    figure, axes = plt.subplots(
        row_count,
        2,
        figsize=(14.8, 3.55 * row_count + 1.15),
        sharex=True,
    )
    figure.patch.set_facecolor("white")
    for column, player in enumerate(("alice", "eve")):
        for row_index, (metric, title, log_y) in enumerate(metrics):
            axis = axes[row_index, column]
            for family in LEARNER_FAMILIES:
                points = [
                    row for row in rows
                    if row["player"] == player
                    and row["family"] == family
                    and row["metric"] == metric
                ]
                if not points:
                    continue
                points.sort(key=lambda row: int(row["turn"]))
                turns = np.asarray([row["turn"] for row in points], dtype=float)
                medians = np.asarray([row["median"] for row in points], dtype=float)
                q25 = np.asarray([row["q25"] for row in points], dtype=float)
                q75 = np.asarray([row["q75"] for row in points], dtype=float)
                color = FAMILY_COLORS[family]
                axis.plot(
                    turns,
                    medians,
                    color=color,
                    linewidth=1.8,
                    label=FAMILY_LABELS[family],
                )
                axis.fill_between(turns, q25, q75, color=color, alpha=0.13)
            axis.set_xscale("log")
            if log_y:
                axis.set_yscale("log")
            axis.grid(alpha=0.22)
            axis.set_title(f"{player.title()}: {title}", fontsize=10)
            if row_index == row_count - 1:
                axis.set_xlabel("turn")
            if metric == "explicit_exploration_mass":
                axis.plot(
                    [],
                    [],
                    color=FAMILY_COLORS["tsallis_inf"],
                    marker="x",
                    linestyle="none",
                    label=r"Tsallis-INF: no separate $\gamma_t$",
                )
                axis.text(
                    0.02,
                    0.05,
                    "Tsallis-INF exploration follows from its regularized "
                    "policy;\ncompare normalized policy entropy below.",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )
            if metric == "normalized_policy_entropy":
                axis.text(
                    0.02,
                    0.05,
                    "EXP3++ begins with randomized one-pass forced sampling",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )
            axis.legend(frameon=False, fontsize=7.5, loc="best")
    figure.suptitle(
        "Selected schedules and realized exploration during 100k confirmation",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.012,
        "Lines are medians across topology, opponent family, and seed; bands are interquartile ranges. "
        "Raw eta values are internal to different update equations and must not be compared as a common physical scale.",
        ha="center",
        fontsize=8.3,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    return _save_figure(figure, stem)


def _candidate_summary(
    summaries: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    matches = [row for row in summaries if row["candidate"]["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one candidate summary for {label}")
    return matches[0]


def _plot_confirmation_heatmaps(
    confirmation: dict[str, Any],
    *,
    metrics: Sequence[tuple[str, str, str]],
    stem: Path,
) -> list[Path]:
    columns = 3
    rows = int(np.ceil(len(metrics) / columns))
    paper_labels = {
        "flat_exp3": "Exp3",
        "tsallis_inf": "Tsallis",
        "exp3_plus_plus": "EXP3++",
        "linear_exp2": "Linear",
    }
    # The six diagnostics have different units.  Normalize each matrix to
    # [0, 1] for color only, retain the measured values in the annotations,
    # and use one shared colorbar.  A single explicit GridSpec keeps all six
    # heatmap panels exactly the same size.
    fig = plt.figure(figsize=(8.4, 2.18 * rows), layout="constrained")
    fig.patch.set_facecolor("white")
    outer_grid = fig.add_gridspec(
        rows,
        columns + 1,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        wspace=0.10,
        hspace=0.16,
    )
    shared_image = None
    for index, (metric, panel_title, number_format) in enumerate(metrics):
        row, column = divmod(index, columns)
        axis = fig.add_subplot(outer_grid[row, column])
        raw_matrix = _family_matrix(confirmation["aggregate"], metric)
        lower = float(np.min(raw_matrix))
        upper = float(np.max(raw_matrix))
        if np.isclose(lower, upper):
            color_matrix = np.full_like(raw_matrix, 0.5, dtype=float)
        else:
            color_matrix = (raw_matrix - lower) / (upper - lower)
        shared_image = _heatmap(
            axis,
            color_matrix,
            row_labels=[paper_labels[name] for name in LEARNER_FAMILIES],
            column_labels=[paper_labels[name] for name in LEARNER_FAMILIES],
            title=panel_title,
            number_format=number_format,
            annotation_matrix=raw_matrix,
            annotation_fontsize=8.0,
            add_colorbar=False,
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(panel_title, fontsize=9.8, pad=4.0)
        axis.set_xlabel(
            "Eve learner" if row == rows - 1 else "",
            fontsize=8.6,
            labelpad=2.5,
        )
        axis.set_ylabel(
            "Alice learner" if column == 0 else "",
            fontsize=8.6,
            labelpad=2.5,
        )
        axis.tick_params(
            axis="both",
            labelsize=7.9,
            length=2.0,
            pad=1.5,
            labelleft=column == 0,
        )
        axis.tick_params(axis="y", length=0, pad=6.0)
        plt.setp(
            axis.get_xticklabels(),
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
    for index in range(len(metrics), rows * columns):
        row, column = divmod(index, columns)
        axis = fig.add_subplot(outer_grid[row, column])
        axis.axis("off")
    if shared_image is not None:
        colorbar_axis = fig.add_subplot(outer_grid[:, columns])
        colorbar = fig.colorbar(
            shared_image,
            cax=colorbar_axis,
            orientation="vertical",
        )
        colorbar.set_ticks((0.0, 0.5, 1.0))
        colorbar.set_ticklabels(("panel min.", "midpoint", "panel max."))
        colorbar.set_label(
            "Within-panel normalized value",
            fontsize=8.2,
            labelpad=4.0,
        )
        colorbar.ax.tick_params(labelsize=7.0, length=2.0, pad=2.0)
    return _save_figure(fig, stem)


def _family_matrix(
    aggregate: Sequence[dict[str, Any]],
    metric: str,
) -> np.ndarray:
    lookup = {
        (row["alice_family"], row["eve_family"]): row
        for row in aggregate
    }
    field = f"mean_{metric}"
    return np.asarray([
        [float(lookup[(alice, eve)][field]) for eve in LEARNER_FAMILIES]
        for alice in LEARNER_FAMILIES
    ])


def _plot_topology_exploitability(
    confirmation: dict[str, Any],
    stem: Path,
) -> list[Path]:
    aggregate = confirmation["aggregate"]
    case_ids = sorted(aggregate[0]["case_means_exploitability"])
    columns = [
        (alice, eve)
        for alice in LEARNER_FAMILIES
        for eve in LEARNER_FAMILIES
    ]
    lookup = {
        (row["alice_family"], row["eve_family"]): row
        for row in aggregate
    }
    matrix = np.asarray([
        [
            float(lookup[cell]["case_means_exploitability"][case_id])
            for cell in columns
        ]
        for case_id in case_ids
    ])
    fig, axis = plt.subplots(figsize=(16.0, 5.4))
    fig.patch.set_facecolor("white")
    _heatmap(
        axis,
        matrix,
        row_labels=[_case_label(case_id) for case_id in case_ids],
        column_labels=[
            f"{FAMILY_LABELS[alice]}\n{FAMILY_LABELS[eve]}"
            for alice, eve in columns
        ],
        title="Nash gap by topology and Alice/Eve learner pair",
        number_format=".3f",
        annotate=False,
    )
    axis.set_xlabel("Alice learner / Eve learner")
    axis.tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _plot_payoff_structure(
    payoff: dict[str, Any],
    stem: Path,
) -> list[Path]:
    cases = payoff["cases"]
    labels = [
        TOPOLOGY_LABELS[row["family"]].replace(" topology", "")
        for row in cases
    ]
    x = np.arange(len(cases), dtype=float)
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7))
    fig.patch.set_facecolor("white")
    axes[0].bar(
        x - width,
        [row["bilinear_in_sample_rmse"] for row in cases],
        width,
        label="in-sample RMSE",
        color="#0072B2",
    )
    axes[0].bar(
        x,
        [row["bilinear_leave_route_out_rmse"] for row in cases],
        width,
        label="leave-route-out RMSE",
        color="#D55E00",
    )
    axes[0].bar(
        x + width,
        [row["bilinear_leave_target_out_rmse"] for row in cases],
        width,
        label="leave-target-out RMSE",
        color="#009E73",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Alice payoff RMSE")
    axes[0].set_title("Bilinear payoff representation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        x,
        [row["bilinear_game_value_absolute_error"] for row in cases],
        color="#CC79A7",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("absolute game-value error")
    axes[1].set_title("Induced zero-sum game value")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Audit-bank payoff structure", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _render_report(
    plan: dict[str, Any],
    selection: dict[str, Any],
    confirmation: dict[str, Any],
    payoff: dict[str, Any],
    *,
    schedule_dynamics_included: bool = False,
    topology_figures_included: bool = False,
) -> str:
    selections = _screening_selection_rows(selection)
    aggregates = confirmation["aggregate"]
    lowest_gap = min(aggregates, key=lambda row: row["mean_exploitability"])
    baseline = next(
        row for row in aggregates
        if row["alice_family"] == "flat_exp3"
        and row["eve_family"] == "flat_exp3"
    )
    false_values = [float(row["mean_false_signal_rate"]) for row in aggregates]
    policy_values = [
        float(row["mean_false_signal_policy_total_variation"])
        for row in aggregates
    ]
    alice_main_effects = _family_main_effects(aggregates, player="alice")
    eve_main_effects = _family_main_effects(aggregates, player="eve")
    topology_rows = _topology_comparison_rows(aggregates, plan["cases"])
    schedule_figure_lines = (
        [
            "",
            "![Selected schedules and realized exploration]"
            "(figures/confirmation_schedule_dynamics.png)",
        ]
        if schedule_dynamics_included else []
    )
    topology_figure_lines = (
        [
            "",
            "![Five frozen calibration networks]"
            "(figures/calibration_topology_overview.png)",
            "",
            "![Network-aligned topology comparison]"
            "(figures/confirmation_topology_network_context.png)",
        ]
        if topology_figures_included else []
    )
    lines = [
        "# Step 7 histogram learner calibration report",
        "",
        "## Result in one paragraph",
        "",
        "On these five calibration topologies, symmetric Tsallis-INF gives "
        "the lowest aggregate Nash gap: "
        f"{lowest_gap['mean_exploitability']:.6f} +/- "
        f"{lowest_gap['seed_block_standard_error_exploitability']:.6f}, "
        "compared with "
        f"{baseline['mean_exploitability']:.6f} +/- "
        f"{baseline['seed_block_standard_error_exploitability']:.6f} for "
        "Exp3/Exp3. Eve's learner family has the larger main effect: every "
        "Alice family paired with Tsallis-INF Eve reaches a lower gap than "
        "the corresponding pairing with Exp3 Eve. The finite structured "
        "learner is competitive but topology-dependent; its clearest result "
        "is KAREN, where LinearExp2 Alice against Tsallis-INF Eve gives the "
        "lowest gap. These measurements identify the leading settings in the "
        "declared screen, but the asymmetric eta/exploration coverage requires "
        "a calibration-only refinement before settings are frozen. They are "
        "not evaluation-set estimates.",
        "",
        "## Scope and information boundary",
        "",
        f"The screen contains {selection['run_count']:,} matched 10k-turn runs. "
        f"The confirmation contains {confirmation['run_count']:,} fresh "
        f"compiled-histogram 100k-turn runs over five fixed calibration "
        "topologies and three fresh seeds. The evaluation partition was not "
        "loaded. Learners received only their selected action's scalar reward; "
        "the reserved audit bank supplied post-run strategic scoring.",
        "",
        "## Selected settings",
        "",
        "| Player | Family | Selected | Mean own regret/turn | Raw minimum |",
        "|---|---|---|---:|---|",
    ]
    for row in selections:
        lines.append(
            f"| {row['player'].title()} | {row['family_label']} | "
            f"`{row['selected_label']}` | "
            f"{float(row['selected_mean_external_regret_per_turn']):.6f} | "
            f"`{row['raw_minimum_label']}` |"
        )
    lines.extend([
        "",
        "EXP3++ beta values 64, 256, and 1024 were behaviorally identical "
        "through eta 8 because the gap-dependent term did not become the "
        "active exploration floor. Alice's EXP3++ minimum is interior at eta "
        "4. Eve's raw minimum is eta 16; the declared one-standard-error rule "
        "selects eta 8.",
        "",
        "Calibration coverage is asymmetric: Tsallis-INF tests three eta "
        "multipliers per estimator, LinearExp2 tests five, and EXP3++ tests "
        "five per beta plus eta 16/32 at beta 256. The Tsallis-INF IW curve "
        "is still improving at eta 2 for both players. Therefore RV/eta 1 is "
        "the minimum of the declared Tsallis grid, not evidence that the full "
        "Tsallis schedule family has been optimized.",
        "",
        "Each eta label is a constant multiplier on a time-dependent schedule, "
        "not a constant learning rate. Tsallis-INF decays as 1/sqrt(t). "
        "Exp3, EXP3++, and LinearExp2 also use time-dependent rates. Explicit "
        "exploration is not independently tuned across all families: "
        "Tsallis-INF has no added gamma mixture; EXP3++ beta never becomes "
        "active in the tested range; and LinearExp2 fixes the G-optimal design "
        "while tying gamma_t to eta_t and the feature rank.",
        "",
        "| Family | Learning-rate schedule | Explicit exploration |",
        "|---|---|---|",
        "| Exp3 | `c sqrt(log(K_eff)/(K_eff (t+t0)))` | "
        "`gamma_t=min(0.2, K_eff eta_t)` |",
        "| Tsallis-INF RV/IW | `4c/sqrt(t)` / `2c/sqrt(t)` | "
        "none; exploration follows from the Tsallis-regularized policy |",
        "| EXP3++ | `0.5c sqrt(log(K)/(tK))` | per-arm "
        "`min(1/(2K), base_eta_t, beta log(t)/(t gap_lcb_t(a)^2))` |",
        "| LinearExp2 | `c sqrt(log(K)/(d (t+t0)))` | "
        "`gamma_t=min(0.2, d eta_t)` mixed through the G-optimal design |",
        "",
        "Here `K_eff` is the inverse minimum exploration-prior mass and `d` "
        "is the feature rank. These predictable schedules remain relevant to "
        "a Stackelberg prime-and-probe attack: time variation creates distinct "
        "probing, concentration, and reduced-exploration windows; it does not "
        "remove them.",
        "",
        "![Screening response curves](figures/screening_regret_profiles.png)",
        *schedule_figure_lines,
        "",
        "## Two-sided 100k confirmation",
        "",
        "Each cell below averages five topologies and three fresh seeds. The "
        "uncertainties are standard errors over matched seed blocks, not over "
        "individual turns.",
        "",
        "| Alice | Eve | Alice regret/turn | Eve regret/turn | Nash gap | "
        "Alice equilibrium-set TV | Eve equilibrium-set TV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in aggregates:
        lines.append(
            f"| {FAMILY_LABELS[row['alice_family']]} | "
            f"{FAMILY_LABELS[row['eve_family']]} | "
            f"{row['mean_alice_external_regret_per_turn']:.6f} | "
            f"{row['mean_eve_external_regret_per_turn']:.6f} | "
            f"{row['mean_exploitability']:.6f} +/- "
            f"{row['seed_block_standard_error_exploitability']:.6f} | "
            f"{row['mean_alice_equilibrium_set_total_variation']:.6f} | "
            f"{row['mean_eve_equilibrium_set_total_variation']:.6f} |"
        )
    lines.extend([
        "",
        "The Nash gap is the sum of Alice's and Eve's audit-bank best-response "
        "gaps. It measures exploitability of the learned pair. External "
        "regret instead scores each realized opponent sequence, so the two "
        "diagnostics need not order learner pairs identically.",
        "",
        "Eve's equilibrium-set total-variation distance remains between "
        "0.55 and 0.63 even in low-regret cells. Therefore the confirmation "
        "supports low exploitability at 100k turns, not convergence to a "
        "particular equilibrium marginal. Redundant and nearly equivalent "
        "attack actions make distributional convergence a stricter demand.",
        "",
        "![Strategic cross-play heatmaps](figures/confirmation_strategic_heatmaps.png)",
        "",
        "## Learner-family main effects",
        "",
        "These are descriptive averages over all four opposing learner "
        "families. They do not remove interaction effects.",
        "",
        "| Alice family | Mean Alice regret/turn | Mean Nash gap |",
        "|---|---:|---:|",
    ])
    for row in alice_main_effects:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | "
            f"{row['mean_own_regret']:.6f} | {row['mean_exploitability']:.6f} |"
        )
    lines.extend([
        "",
        "| Eve family | Mean Eve regret/turn | Mean Nash gap |",
        "|---|---:|---:|",
    ])
    for row in eve_main_effects:
        lines.append(
            f"| {FAMILY_LABELS[row['family']]} | "
            f"{row['mean_own_regret']:.6f} | {row['mean_exploitability']:.6f} |"
        )
    lines.extend([
        "",
        "Tsallis-INF has the lowest own-player regret in both roles. The "
        "structured LinearExp2 learner improves on Exp3 in the role-averaged "
        "comparison, but it is not uniformly best. This is evidence for "
        "retaining both independent-arm and structured families rather than "
        "declaring one universal winner.",
        "",
        "## Topology dependence",
        "",
        "| Topology | Routes | Eve actions | Best Alice/Eve pair | Best gap | "
        "Exp3/Exp3 gap | Reduction from Exp3/Exp3 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ])
    for row in topology_rows:
        lines.append(
            f"| {row['label']} | {row['route_count']} | "
            f"{row['eve_action_count']} | {row['best_pair']} | "
            f"{row['best_gap']:.6f} | {row['baseline_gap']:.6f} | "
            f"{row['relative_reduction']:.1%} |"
        )
    lines.extend([
        "",
        "KAREN is the clearest structured-Alice case: LinearExp2 Alice with "
        "Tsallis-INF Eve reaches a gap of 0.007108, compared with 0.080551 "
        "for Exp3/Exp3. Hibernia has only two routes; many learner pairs are "
        "already close in strategic value, so it provides little evidence "
        "for or against a high-dimensional representation.",
        "",
        "![Topology-resolved Nash gaps](figures/confirmation_topology_nash_gap.png)",
        *topology_figure_lines,
        "",
        "## Public signals, counterfactual attribution, and timing",
        "",
        f"Across the 16 family pairs, the mean false-signal rate ranges from "
        f"{min(false_values):.4%} to {max(false_values):.4%}; the associated "
        f"counterfactual Alice-policy total variation ranges from "
        f"{min(policy_values):.6f} to {max(policy_values):.6f}.",
        "",
        "The policy attribution holds selected actions and their factual "
        "sampling probabilities fixed, replaces false-signal rewards by their "
        "latent-threshold corrections, and replays Alice's estimator updates. "
        "It measures the policy displacement caused by those observations on "
        "the realized trajectory; it does not simulate the recursively changed "
        "actions of a second learning history.",
        "",
        f"The separate uncontended timing tranche measures 10k-turn "
        f"learner-plus-transcript throughput. Exp3/Exp3 reaches "
        f"{baseline['mean_turns_per_second']:.0f} turns/s; "
        f"Tsallis-INF/Tsallis-INF reaches "
        f"{lowest_gap['mean_turns_per_second']:.0f} turns/s. Strategic worker "
        "wall time is excluded from this comparison because those runs share "
        "CPU and memory bandwidth.",
        "",
        "![Service, signal, and timing heatmaps](figures/confirmation_signal_heatmaps.png)",
        "",
        "## Payoff-structure audit",
        "",
        "For route feature matrix `X`, Eve feature matrix `Z`, and audit payoff "
        "matrix `U`, the bilinear model is `U_hat = X W Z^T`. The "
        "minimum-Frobenius-norm least-squares coefficient is "
        "`W* = X^+ U (Z^T)^+`, where `+` denotes the Moore--Penrose "
        "pseudoinverse. Numerical rank is determined by truncated SVD with "
        "the explicit cutoff `sqrt(machine epsilon) * largest singular value`; "
        "the inverse is therefore the Moore--Penrose inverse of the retained "
        "effective-rank matrix. This cutoff prevents numerically null "
        "directions in held-out folds from amplifying bounded payoffs.",
        "",
        "| Topology | In-sample RMSE | Leave-route-out RMSE | "
        "Leave-target-out RMSE | Game-value absolute error |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in payoff["cases"]:
        label = TOPOLOGY_LABELS[row["family"]].replace("\n", " ")
        lines.append(
            f"| {label} | {row['bilinear_in_sample_rmse']:.6f} | "
            f"{row['bilinear_leave_route_out_rmse']:.6f} | "
            f"{row['bilinear_leave_target_out_rmse']:.6f} | "
            f"{row['bilinear_game_value_absolute_error']:.6f} |"
        )
    lines.extend([
        "",
        "KAREN and Memorex retain low held-axis errors, so the declared route "
        "and Eve features represent their payoff variation beyond the fitted "
        "cells. Hibernia and Layer42 fit almost exactly in-sample but have "
        "large held-route and held-target errors; their small route sets do "
        "not identify the missing feature directions. CERNET is intermediate. "
        "Game-value error is reported separately because modest entrywise "
        "error can still move the strategically relevant saddle value.",
        "",
        "![Bilinear payoff diagnostics](figures/payoff_structure_diagnostics.png)",
        "",
        "## Claim boundary and next decision",
        "",
        "This bundle establishes declared-grid settings and calibration-set "
        "comparisons for four symmetric two-player learner families. It does "
        "not estimate performance on the five frozen evaluation topologies, "
        "establish equilibrium-marginal convergence, or establish that the "
        "bilinear feature model transfers to every Topology Bench graph. The "
        "next controlled step is to refine eta and exploration on the same "
        "calibration partition, freeze the resulting settings, and only then "
        "run one 4-by-4 comparison on the evaluation partition with the same "
        "regret, exploitability, equilibrium-set, signal, service, and timing "
        "diagnostics.",
        "",
        "The CSV tables retain every family-pair and topology mean used in "
        "this report.",
        "",
    ])
    return "\n".join(lines)


def _family_main_effects(
    aggregates: Sequence[dict[str, Any]],
    *,
    player: str,
) -> list[dict[str, Any]]:
    family_field = f"{player}_family"
    regret_field = f"mean_{player}_external_regret_per_turn"
    rows = []
    for family in LEARNER_FAMILIES:
        matches = [row for row in aggregates if row[family_field] == family]
        if len(matches) != len(LEARNER_FAMILIES):
            raise ValueError(
                f"expected {len(LEARNER_FAMILIES)} cross-play rows for "
                f"{player} {family}"
            )
        rows.append({
            "family": family,
            "mean_own_regret": float(np.mean([
                row[regret_field] for row in matches
            ])),
            "mean_exploitability": float(np.mean([
                row["mean_exploitability"] for row in matches
            ])),
        })
    return rows


def _topology_comparison_rows(
    aggregates: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = sorted(aggregates[0]["case_means_exploitability"])
    case_by_id = {row["case_id"]: row for row in cases}
    baseline = next(
        row for row in aggregates
        if row["alice_family"] == "flat_exp3"
        and row["eve_family"] == "flat_exp3"
    )
    rows = []
    for case_id in case_ids:
        best = min(
            aggregates,
            key=lambda row: row["case_means_exploitability"][case_id],
        )
        baseline_gap = float(baseline["case_means_exploitability"][case_id])
        best_gap = float(best["case_means_exploitability"][case_id])
        case = case_by_id[case_id]
        rows.append({
            "label": _case_label(case_id).replace("\n", " "),
            "route_count": int(case["route_count"]),
            "eve_action_count": int(case["eve_action_count"]),
            "best_pair": (
                f"{FAMILY_LABELS[best['alice_family']]} / "
                f"{FAMILY_LABELS[best['eve_family']]}"
            ),
            "best_gap": best_gap,
            "baseline_gap": baseline_gap,
            "relative_reduction": (
                (baseline_gap - best_gap) / baseline_gap
                if baseline_gap > 0.0 else 0.0
            ),
        })
    return rows


def _heatmap(
    axis: Any,
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    number_format: str,
    annotate: bool = True,
    annotation_matrix: np.ndarray | None = None,
    annotation_fontsize: float = 8.0,
    add_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    color_values = np.asarray(matrix, dtype=float)
    annotations = (
        color_values
        if annotation_matrix is None
        else np.asarray(annotation_matrix, dtype=float)
    )
    if annotations.shape != color_values.shape:
        raise ValueError("annotation matrix must match heatmap shape")
    image = axis.imshow(
        color_values,
        cmap="turbo",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)
    if annotate:
        for row in range(color_values.shape[0]):
            for column in range(color_values.shape[1]):
                rgba = image.cmap(image.norm(color_values[row, column]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column,
                    row,
                    format(float(annotations[row, column]), number_format),
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.52 else "white",
                    fontsize=annotation_fontsize,
                )
    if add_colorbar:
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return image


def _case_label(case_id: str) -> str:
    family = case_id.split("_topology_bench_", 1)[1].split("_seed_", 1)[0]
    return TOPOLOGY_LABELS[family]


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = stem.with_suffix(f".{suffix}")
        save_options = {
            "dpi": 220,
            "bbox_inches": "tight",
            "facecolor": "white",
            "transparent": False,
        }
        if suffix == "pdf":
            save_options["metadata"] = {
                "CreationDate": None,
                "ModDate": None,
            }
        fig.savefig(path, **save_options)
        paths.append(path)
    plt.close(fig)
    return paths


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty table {path.name}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"table {path.name} has inconsistent columns")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
