"""Publication artifacts for the physical Prime-and-Probe study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.3,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.transparent": False,
})


_TSALLIS = "tsallis_inf_rv_eta1"
_LINEAR = "linear_exp2_goptimal_eta2"
_DEFENDER_LABEL = {
    _TSALLIS: "Tsallis-INF (RV, $\\eta{=}1$)",
    _LINEAR: "LinearExp2 (G-optimal, $\\eta{=}2$)",
}
_DEFENDER_COLOR = {
    _TSALLIS: "#0072B2",
    _LINEAR: "#D55E00",
}
_FAMILY_MARKER = {
    "memorex": "o",
    "hiberniaireland": "s",
    "layer42": "^",
    "karen": "D",
    "cernet": "P",
}
_FAMILY_DISPLAY = {
    "memorex": "Memorex",
    "hiberniaireland": "Hibernia Ireland",
    "layer42": "Layer42",
    "karen": "Karen",
    "cernet": "Cernet",
}
_PHASE_COLOR = {
    "warmup": "#E6E6E6",
    "prime": "#E8D99B",
    "probe": "#A9CCE3",
    "exploit": "#E3A17B",
    "recovery": "#B7D8CD",
}


def write_prime_probe_artifacts(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Render figures, a tidy table, and a self-contained result report."""

    source = Path(run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    runs = _load_jsonl(source / "results.jsonl")
    comparisons = _load_jsonl(source / "paired_comparisons.jsonl")
    checkpoints = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((source / "graph_checkpoints").glob("*.json"))
    }
    _validate_inputs(
        manifest,
        summary,
        runs,
        comparisons,
        checkpoints,
        require_complete=require_complete,
    )

    files: list[Path] = []
    renderers = (
        ("prime_probe_phase_protocol", _plot_prime_probe_phase_protocol),
        ("prime_probe_network_mechanism", _plot_prime_probe_network_mechanism),
        ("ordered_phase_dynamics", _plot_ordered_phase_dynamics),
        ("timing_matters", _plot_timing_comparison),
        ("probe_information", _plot_probe_information),
        ("recovery", _plot_recovery),
        ("topology_defender_outcomes", _plot_topology_defender_outcomes),
        ("selected_bottleneck_physics", _plot_plan_physics),
    )
    for stem, renderer in renderers:
        png_path = target / f"{stem}.png"
        renderer(manifest, runs, comparisons, checkpoints, png_path)
        files.extend(png_path.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))

    csv_path = target / "prime_probe_paired_results.csv"
    _write_comparison_csv(comparisons, csv_path)
    files.append(csv_path)
    report_path = target / "PRIME_PROBE_500K_REPORT.md"
    report_path.write_text(
        _report_markdown(manifest, runs, comparisons, checkpoints),
        encoding="utf-8",
    )
    files.append(report_path)

    artifact_manifest = {
        "schema": "histogram_prime_probe_artifacts.v1",
        "source_study_schema": manifest["schema"],
        "source_corpus_selection_digest_blake2b_128": manifest[
            "source_corpus_selection_digest_blake2b_128"
        ],
        "complete_run_required": bool(require_complete),
        "graph_count": len(checkpoints),
        "ordered_run_count": sum(
            row["condition"] == "ordered_prime_probe" for row in runs
        ),
        "paired_comparison_count": len(comparisons),
        "publication_rendering": {
            "formats": ["pdf", "svg", "png"],
            "png_dpi": 400,
            "vector_text": True,
            "pdf_font_embedding": "TrueType",
            "primary_figure_width_inches": 7.1,
            "color_encoding": "Okabe-Ito with marker and line-style redundancy",
            "uncertainty_boundary": (
                "three learner/outcome-resampling seeds conditional on one shared "
                "equal-coverage physical bank for each of five predeclared "
                "topologies; intervals and distributions are finite-study summaries, "
                "not device-calibration, population, or asymptotic confidence claims"
            ),
        },
        "scientific_boundary": {
            "interventions": "declared compiled-histogram Eve instruments",
            "reward_poisoning": False,
            "control": "identical physical action multiset under temporal permutation",
            "online_trigger_signal": "count-valued local probe record only",
            "audit_split": "scoring only",
            "claim_scope": "finite-horizon leader/follower experiment",
        },
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "blake2b_128": _file_digest(path),
            }
            for path in files
        ],
    }
    manifest_path = target / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact_manifest


def _validate_inputs(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool,
) -> None:
    expected_runs = int(manifest["expected_condition_runs"])
    expected_pairs = int(manifest["expected_paired_comparisons"])
    if require_complete and not bool(summary["complete"]):
        raise ValueError("Prime-and-Probe source study is not complete")
    if require_complete and len(runs) != expected_runs:
        raise ValueError(f"condition run count is {len(runs)}/{expected_runs}")
    if require_complete and len(comparisons) != expected_pairs:
        raise ValueError(f"paired comparison count is {len(comparisons)}/{expected_pairs}")
    if require_complete and len(checkpoints) != len(manifest["cases"]):
        raise ValueError("graph checkpoint count does not match the study manifest")
    for row in comparisons:
        if not bool(row["identical_action_multiset"]):
            raise ValueError("a paired comparison does not certify its action multiset")
    for row in runs:
        boundary = row["information_boundary"]
        if not bool(boundary["physical_interventions_only"]):
            raise ValueError("a condition includes a nonphysical intervention")
        if bool(boundary["reward_poisoning"]):
            raise ValueError("a condition includes reward poisoning")
        if row["condition"] == "ordered_prime_probe":
            if any(
                bool(boundary[field])
                for field in (
                    "probe_trigger_uses_route_identity",
                    "probe_trigger_uses_alice_policy_or_weights",
                    "probe_trigger_uses_target_active",
                    "probe_trigger_uses_audit_split",
                )
            ):
                raise ValueError("an ordered trigger crosses its declared information boundary")


def _plot_prime_probe_phase_protocol(
    manifest: Mapping[str, Any],
    _runs: Sequence[Mapping[str, Any]],
    _comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    """Render the exact phase schedule, controller action, and information boundary."""

    schedule = manifest["schedule"]
    phases = (
        (
            "warmup",
            int(schedule["warmup_turns"]),
            "No intervention",
            "Learns physical route service",
            "Late-warmup reference",
        ),
        (
            "prime",
            int(schedule["prime_turns"]),
            "Deny selected alternatives; spare $T$",
            "Policy mass can move toward $T$",
            "No local probe record",
        ),
        (
            "probe",
            int(schedule["probe_turns"]),
            "Measure target resource $T$",
            "Learner continues updating",
            "Integer local count $C$",
        ),
        (
            "exploit",
            int(schedule["exploit_turns"]),
            "Deny $T$ if triggered",
            "Target-family service is suppressed",
            "$L(C)>p_T^{\\mathrm{uniform}}$",
        ),
        (
            "recovery",
            int(schedule["recovery_turns"]),
            "Withdraw intervention",
            "Policy and service re-adapt",
            "Sustained return to reference",
        ),
    )
    total_turns = int(schedule["total_turns"])
    if sum(turns for _phase, turns, *_details in phases) != total_turns:
        raise ValueError("Prime-and-Probe phase lengths do not sum to total_turns")

    figure, axis = plt.subplots(figsize=(7.1, 5.25))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    timeline_y = 0.865
    timeline_height = 0.075
    start = 0
    boundaries = [0]
    for phase, turns, *_details in phases:
        left = start / total_turns
        width = turns / total_turns
        axis.add_patch(Rectangle(
            (left, timeline_y),
            width,
            timeline_height,
            facecolor=_PHASE_COLOR[phase],
            edgecolor="white",
            linewidth=0.8,
        ))
        if width < 0.055:
            axis.annotate(
                phase,
                xy=(left + width / 2.0, timeline_y + timeline_height),
                xytext=(left + width / 2.0, timeline_y + 0.145),
                ha="center",
                va="bottom",
                fontsize=6.8,
                arrowprops={
                    "arrowstyle": "-",
                    "linewidth": 0.7,
                    "color": "#555555",
                },
            )
        else:
            axis.text(
                left + width / 2.0,
                timeline_y + timeline_height / 2.0,
                phase,
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
            )
        start += turns
        boundaries.append(start)
    for boundary_index, boundary in enumerate(boundaries):
        x = boundary / total_turns
        label_y = timeline_y - 0.018
        if boundary_index == 2:
            horizontal_alignment = "right"
            x -= 0.003
        elif boundary_index == 3:
            horizontal_alignment = "left"
            x += 0.003
            label_y -= 0.032
        else:
            horizontal_alignment = (
                "left"
                if boundary_index == 0
                else "right"
                if boundary_index == len(boundaries) - 1
                else "center"
            )
        axis.text(
            x,
            label_y,
            _format_turns(boundary),
            ha=horizontal_alignment,
            va="top",
            fontsize=6.5,
        )
    axis.text(
        0.5,
        0.968,
        "Exact 500k-turn schedule",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    column_x = (0.075, 0.285, 0.59, 0.86)
    headers = ("PHASE", "EVE ACTION", "ALICE / NETWORK RESPONSE", "LOCAL RECORD / DECISION")
    for x, header in zip(column_x, headers):
        axis.text(
            x,
            0.705,
            header,
            ha="center",
            va="bottom",
            fontsize=6.65,
            fontweight="bold",
        )
    row_top = 0.68
    row_height = 0.079
    for index, (phase, turns, eve_action, response, record) in enumerate(phases):
        y = row_top - (index + 1) * row_height
        axis.add_patch(Rectangle(
            (0.012, y),
            0.976,
            row_height - 0.005,
            facecolor=_PHASE_COLOR[phase],
            alpha=0.16,
            edgecolor="white",
            linewidth=0.6,
        ))
        entries = (
            f"{phase.capitalize()}  ({_format_turns(turns)})",
            eve_action,
            response,
            record,
        )
        for x, entry in zip(column_x, entries):
            axis.text(
                x,
                y + (row_height - 0.005) / 2.0,
                entry,
                ha="center",
                va="center",
                fontsize=5.85,
                fontweight="bold" if x == column_x[0] else "normal",
            )
    for x in (0.17, 0.43, 0.75):
        axis.plot(
            (x, x),
            (row_top - 5 * row_height, row_top),
            color="#FFFFFF",
            linewidth=1.0,
        )

    boundary_y = 0.035
    boundary_height = 0.19
    axis.add_patch(FancyBboxPatch(
        (0.012, boundary_y),
        0.976,
        boundary_height,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        facecolor="#F7F7F7",
        edgecolor="#777777",
        linewidth=0.9,
    ))
    axis.plot(
        (0.5, 0.5),
        (boundary_y + 0.025, boundary_y + boundary_height - 0.025),
        color="#B0B0B0",
        linewidth=0.8,
    )
    axis.text(
        0.255,
        boundary_y + boundary_height - 0.035,
        "ONLINE EVE RECEIVES",
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="#0072B2",
    )
    axis.text(
        0.255,
        boundary_y + boundary_height - 0.078,
        "Topology and declared intervention bank\n"
        "Own selected actions and local probe count $C$\n"
        "Count-calibrated target-mass interval",
        ha="center",
        va="top",
        fontsize=6.25,
        linespacing=1.3,
    )
    axis.text(
        0.745,
        boundary_y + boundary_height - 0.035,
        "NOT AVAILABLE TO THE TRIGGER",
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="#D55E00",
    )
    axis.text(
        0.745,
        boundary_y + boundary_height - 0.078,
        "Alice's policy, weights, or route identity\n"
        "Target-active state or sampled route history\n"
        "Independent audit split",
        ha="center",
        va="top",
        fontsize=6.25,
        linespacing=1.3,
    )
    figure.suptitle(
        "Physical Prime-and-Probe controller: ordered phases and information boundary",
        y=0.995,
    )
    _save_publication_figure(figure, path)


def _plot_prime_probe_network_mechanism(
    manifest: Mapping[str, Any],
    _runs: Sequence[Mapping[str, Any]],
    _comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    """Show how the ordered intervention creates, measures, and exploits concentration."""

    schedule = manifest["schedule"]
    stages = (
        (
            "warmup",
            "Warmup",
            int(schedule["warmup_turns"]),
            "No intervention\nlearn ordinary service",
        ),
        (
            "prime",
            "Prime",
            int(schedule["prime_turns"]),
            "Deny alternatives\nspare target family",
        ),
        (
            "probe",
            "Probe",
            int(schedule["probe_turns"]),
            "Measure target\nobserve local count $C$",
        ),
        (
            "exploit",
            "Exploit",
            int(schedule["exploit_turns"]),
            "If triggered,\ndeny target resource",
        ),
        (
            "recovery",
            "Recovery",
            int(schedule["recovery_turns"]),
            "Withdraw attack\ntrack policy + service",
        ),
    )
    figure, axes = plt.subplots(1, 5, figsize=(7.1, 3.35))
    for axis, (stage, title, turns, caption) in zip(axes, stages):
        _draw_route_family_stage(axis, stage)
        axis.set_title(
            f"{title}\n{_format_turns(turns)}",
            fontsize=7.8,
            pad=5,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": _PHASE_COLOR[stage],
                "alpha": 0.25,
                "edgecolor": _PHASE_COLOR[stage],
                "linewidth": 0.7,
            },
        )
        axis.text(
            0.5,
            -0.12,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=6.3,
            linespacing=1.25,
        )
    figure.text(
        0.5,
        0.055,
        "Target $T$ denotes the selected route/resource family for each topology; "
        "the report records the exact compiled prime, probe, and exploit instruments.",
        ha="center",
        va="bottom",
        fontsize=6.25,
    )
    legend_handles = [
        Line2D([0], [0], color="#0072B2", linewidth=3.0, label="target-family route"),
        Line2D([0], [0], color="#8C8C8C", linewidth=1.8, label="alternative route"),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#D55E00",
            markeredgewidth=2.0,
            linestyle="none",
            label="physical denial",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#7B3294",
            markerfacecolor="none",
            linestyle="none",
            label="count-valued measurement",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        fontsize=6.25,
    )
    figure.suptitle(
        "How ordered physical interventions create and exploit a virtual bottleneck",
        y=0.995,
    )
    _save_publication_figure(figure, path)


def _draw_route_family_stage(axis: Any, stage: str) -> None:
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.0, 1.0)
    axis.set_aspect("equal")
    axis.axis("off")
    positions = {
        "A": (-0.95, 0.0),
        "B": (0.95, 0.0),
        "$q_1$": (0.0, 0.63),
        "$T$": (0.0, 0.0),
        "$q_2$": (0.0, -0.63),
    }
    if stage == "warmup":
        line_widths = (1.8, 1.8, 1.8)
    elif stage in {"prime", "probe", "exploit"}:
        line_widths = (1.0, 3.7, 1.0)
    else:
        line_widths = (1.65, 2.2, 1.65)
    route_specs = (
        ("$q_1$", "#8C8C8C", line_widths[0]),
        ("$T$", "#0072B2", line_widths[1]),
        ("$q_2$", "#8C8C8C", line_widths[2]),
    )
    for resource, color, linewidth in route_specs:
        middle = positions[resource]
        axis.plot(
            (positions["A"][0], middle[0], positions["B"][0]),
            (positions["A"][1], middle[1], positions["B"][1]),
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )
    for label, (x, y) in positions.items():
        radius = 0.105 if label in {"A", "B"} else 0.09
        facecolor = "#EAF4FB" if label == "$T$" else "white"
        edgecolor = "#0072B2" if label == "$T$" else "#333333"
        axis.add_patch(Circle(
            (x, y),
            radius=radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            zorder=3,
        ))
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            zorder=4,
        )
    if stage == "prime":
        _draw_denial_mark(axis, positions["$q_1$"])
        _draw_denial_mark(axis, positions["$q_2$"])
        axis.text(
            0.0,
            0.28,
            "$p_T\\,\\uparrow$",
            ha="center",
            va="center",
            color="#0072B2",
            fontsize=8.0,
            fontweight="bold",
        )
    elif stage == "probe":
        axis.add_patch(Circle(
            positions["$T$"],
            radius=0.165,
            facecolor="none",
            edgecolor="#7B3294",
            linewidth=2.0,
            zorder=5,
        ))
        axis.annotate(
            "$C$",
            xy=(0.1, 0.1),
            xytext=(0.48, 0.42),
            fontsize=8.0,
            color="#7B3294",
            fontweight="bold",
            arrowprops={
                "arrowstyle": "->",
                "color": "#7B3294",
                "linewidth": 1.1,
            },
        )
    elif stage == "exploit":
        _draw_denial_mark(axis, positions["$T$"])
        axis.text(
            0.0,
            0.34,
            "$L(C)>p_T^{\\mathrm{uniform}}$",
            ha="center",
            va="center",
            color="#7B3294",
            fontsize=6.2,
        )
    elif stage == "recovery":
        axis.annotate(
            "",
            xy=(-0.2, 0.52),
            xytext=(-0.38, 0.25),
            arrowprops={
                "arrowstyle": "->",
                "color": "#009E73",
                "linewidth": 1.0,
            },
        )
        axis.annotate(
            "",
            xy=(-0.2, -0.52),
            xytext=(-0.38, -0.25),
            arrowprops={
                "arrowstyle": "->",
                "color": "#009E73",
                "linewidth": 1.0,
            },
        )


def _draw_denial_mark(axis: Any, position: tuple[float, float]) -> None:
    x, y = position
    axis.plot(
        (x - 0.12, x + 0.12),
        (y - 0.12, y + 0.12),
        color="#D55E00",
        linewidth=2.2,
        zorder=6,
    )
    axis.plot(
        (x - 0.12, x + 0.12),
        (y + 0.12, y - 0.12),
        color="#D55E00",
        linewidth=2.2,
        zorder=6,
    )


def _format_turns(turns: int) -> str:
    if turns == 0:
        return "0"
    if turns % 1_000 == 0:
        return f"{turns // 1_000}k"
    return f"{turns:,}"


def _plot_ordered_phase_dynamics(
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    ordered = [row for row in runs if row["condition"] == "ordered_prime_probe"]
    figure = plt.figure(figsize=(7.1, 3.25))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(4.25, 1.35),
        height_ratios=(1.15, 1.0),
        hspace=0.13,
        wspace=0.32,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[1, 0]),
    )
    axes[1].sharex(axes[0])
    alarm_axis = figure.add_subplot(grid[:, 1])
    fields = (
        (
            "rolling_alice_reward",
            "(a)",
            "Alice reward\n(trailing 5k mean)",
        ),
        (
            "alice_target_family_policy_mass",
            "(b)",
            "Target-family\npolicy mass",
        ),
    )
    schedule = manifest["schedule"]
    warmup_stop = int(schedule["warmup_turns"])
    probe_start = warmup_stop + int(schedule["prime_turns"])
    exploit_start = probe_start + int(schedule["probe_turns"])
    withdrawal_start = (
        exploit_start + int(schedule["exploit_turns"])
    )
    total_turns = int(schedule["total_turns"])
    tick_turns = (0, warmup_stop, probe_start, withdrawal_start, total_turns)
    detector_summaries = _alice_endpoint_alarm_summaries(
        ordered,
        prime_start=warmup_stop,
        outcome_name="e91_measurement_counts_only",
    )
    service_detector_summaries = _alice_endpoint_alarm_summaries(
        ordered,
        prime_start=warmup_stop,
        outcome_name="classical_service",
    )
    for axis, (field, panel_label, ylabel) in zip(axes, fields):
        for defender, linestyle in ((_TSALLIS, "-"), (_LINEAR, "--")):
            selected = [row for row in ordered if row["defender_label"] == defender]
            if not selected:
                continue
            turns = np.asarray([point["turn"] for point in selected[0]["trajectory"]])
            values = np.asarray([
                [float(point[field]) for point in row["trajectory"]]
                for row in selected
            ])
            median = np.median(values, axis=0)
            lower = np.quantile(values, 0.25, axis=0)
            upper = np.quantile(values, 0.75, axis=0)
            axis.plot(
                turns,
                median,
                color=_DEFENDER_COLOR[defender],
                linestyle=linestyle,
                linewidth=1.9,
                label=(
                    "Tsallis-INF median"
                    if defender == _TSALLIS
                    else "LinearExp2 median"
                ),
                zorder=3,
            )
            axis.fill_between(
                turns,
                lower,
                upper,
                color=_DEFENDER_COLOR[defender],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                turns,
                lower,
                color=_DEFENDER_COLOR[defender],
                linewidth=0.55,
                alpha=0.62,
                zorder=2,
            )
            axis.plot(
                turns,
                upper,
                color=_DEFENDER_COLOR[defender],
                linewidth=0.55,
                alpha=0.62,
                zorder=2,
            )
        axis.set_ylabel(ylabel)
        axis.text(
            0.012,
            0.955,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=5,
        )
        axis.set_ylim(-0.025, 1.04)
        axis.set_xlim(0, total_turns)
        axis.set_xticks(
            tick_turns,
            tuple(_format_turns(turn) for turn in tick_turns),
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].tick_params(axis="x", labelbottom=False)
    _mark_prime_probe_phases(axes, schedule)
    for axis in axes:
        axis.axvline(
            exploit_start,
            color="#5D3A75",
            linestyle="-.",
            linewidth=1.15,
            alpha=0.95,
            zorder=4,
        )
    trigger_count, trigger_total = _eve_probe_trigger_count(
        ordered,
        comparisons,
    )
    trigger_fraction = (
        f"{trigger_count}/{trigger_total}"
        if trigger_total
        else "not reported"
    )
    axes[0].annotate(
        "Probe authorizes Exploit\n"
        f"in {trigger_fraction} runs",
        xy=(exploit_start, 0.02),
        xytext=(exploit_start + 0.025 * total_turns, 0.30),
        color="#5D3A75",
        fontsize=5.8,
        fontweight="bold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": "#5D3A75",
            "linewidth": 0.7,
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        zorder=5,
    )
    if detector_summaries:
        _draw_alice_endpoint_alarm_panel(
            alarm_axis,
            detector_summaries,
            service_detector_summaries,
        )
    else:
        _draw_missing_alice_alarm_panel(alarm_axis)
    axes[0].legend(
        frameon=True,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.018),
        handlelength=2.5,
        columnspacing=1.15,
        borderaxespad=0.0,
        borderpad=0.25,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    axes[1].set_xlabel("Game turn", labelpad=1.5)
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.145,
        top=0.905,
        hspace=0.13,
        wspace=0.32,
    )
    _save_publication_figure(figure, path, apply_tight_layout=False)


def _alice_endpoint_alarm_summaries(
    ordered_runs: Sequence[Mapping[str, Any]],
    *,
    prime_start: int,
    outcome_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize Alice alarms after Prime onset by learner.

    The detector output uses one-based absolute turns, whereas ``prime_start``
    is the number of completed warmup turns.  Consequently, an alarm on the
    first Prime turn has delay one.  Pre-Prime alarms are counted but excluded
    from the plotted distribution.  New runs use the E91-measurement-count-only
    arm; the older ``quantum_endpoint`` key is retained for compatible bundles.
    """

    instrumented_outcomes = [
        detector["outcomes"]
        for row in ordered_runs
        if isinstance(
            (detector := row.get("alice_endpoint_detector")),
            Mapping,
        )
        and bool(detector.get("enabled", False))
        and isinstance(detector.get("outcomes"), Mapping)
    ]
    use_measurement_counts = bool(instrumented_outcomes) and all(
        "e91_measurement_counts_only" in outcomes
        for outcomes in instrumented_outcomes
    )
    if outcome_name is None:
        outcome_name = (
            "e91_measurement_counts_only"
            if use_measurement_counts
            else "quantum_endpoint"
        )
    summaries: dict[str, dict[str, Any]] = {}
    for defender in (_TSALLIS, _LINEAR):
        instrumented = 0
        pre_prime_false_alarms = 0
        pre_prime_false_alarm_turns: list[int] = []
        alarm_turns: list[int] = []
        delays: list[int] = []
        for row in ordered_runs:
            if row.get("defender_label") != defender:
                continue
            detector = row.get("alice_endpoint_detector")
            if not isinstance(detector, Mapping) or not bool(
                detector.get("enabled", False)
            ):
                continue
            outcomes = detector.get("outcomes")
            if not isinstance(outcomes, Mapping):
                continue
            outcome = outcomes.get(outcome_name)
            if not isinstance(outcome, Mapping):
                continue
            instrumented += 1
            alarm_turn_raw = outcome.get("alarm_turn")
            if alarm_turn_raw is None:
                continue
            alarm_turn = int(alarm_turn_raw)
            reported_pre_prime = bool(
                outcome.get("pre_prime_false_alarm", False)
            )
            pre_prime = reported_pre_prime or alarm_turn <= prime_start
            if pre_prime:
                pre_prime_false_alarms += 1
                pre_prime_false_alarm_turns.append(alarm_turn)
                continue
            derived_delay = alarm_turn - prime_start
            reported_delay = outcome.get(
                "detection_delay_from_prime_start_turns"
            )
            if (
                reported_delay is not None
                and int(reported_delay) != derived_delay
            ):
                raise ValueError(
                    "Alice endpoint detector delay does not match its "
                    "absolute alarm turn"
                )
            alarm_turns.append(alarm_turn)
            delays.append(derived_delay)
        if instrumented:
            summaries[defender] = {
                "instrumented_run_count": instrumented,
                "post_prime_alarm_count": len(alarm_turns),
                "pre_prime_false_alarm_count": pre_prime_false_alarms,
                "pre_prime_false_alarm_turns": tuple(
                    pre_prime_false_alarm_turns
                ),
                "alarm_turns": tuple(alarm_turns),
                "delays": tuple(delays),
                "median_alarm_turn": (
                    float(np.median(alarm_turns))
                    if alarm_turns
                    else None
                ),
                "median_delay": (
                    float(np.median(delays))
                    if delays
                    else None
                ),
                "detector_outcome_name": outcome_name,
            }
    return summaries


def _eve_probe_trigger_count(
    ordered_runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Return Eve's recorded local-Probe decisions without assuming 30 runs."""

    decisions = []
    for row in ordered_runs:
        trigger = row.get("probe_trigger")
        if (
            isinstance(trigger, Mapping)
            and isinstance(trigger.get("triggered"), bool)
        ):
            decisions.append(bool(trigger["triggered"]))
    if not decisions:
        decisions = [
            bool(row["ordered_triggered"])
            for row in comparisons
            if isinstance(row.get("ordered_triggered"), bool)
        ]
    return sum(decisions), len(decisions)


def _draw_alice_endpoint_alarm_panel(
    axis: Any,
    summaries: Mapping[str, Mapping[str, Any]],
    service_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Plot run-level Alice alarm delays on a scale distinct from 500k turns."""

    lane_x = {_TSALLIS: 0.0, _LINEAR: 1.0}
    service_summaries = service_summaries or {}
    detector_groups = [
        ("E91 counts", summaries, "o", -0.11 if service_summaries else 0.0, True),
    ]
    if service_summaries:
        detector_groups.append(
            ("Service decision", service_summaries, "s", 0.11, False)
        )
    all_delays: list[int] = []
    for defender in (_TSALLIS, _LINEAR):
        for _label, group_summaries, marker, group_offset, filled in detector_groups:
            summary = group_summaries.get(defender)
            if summary is None:
                continue
            delays = [int(value) for value in summary["delays"]]
            all_delays.extend(delays)
            x = lane_x[defender] + group_offset
            if delays:
                offsets = np.linspace(-0.065, 0.065, num=len(delays))
                axis.scatter(
                    x + offsets,
                    delays,
                    s=12,
                    marker=marker,
                    facecolors=(
                        _DEFENDER_COLOR[defender] if filled else "none"
                    ),
                    edgecolors=_DEFENDER_COLOR[defender],
                    linewidths=0.45,
                    alpha=0.68,
                    zorder=3,
                )
                median_delay = float(summary["median_delay"])
                axis.plot(
                    (x - 0.07, x + 0.07),
                    (median_delay, median_delay),
                    color=_DEFENDER_COLOR[defender],
                    linestyle="-" if filled else (0, (2.2, 1.5)),
                    linewidth=1.0,
                    zorder=4,
                )
    maximum_delay = max(all_delays, default=1)
    upper_delay = max(
        10.0,
        10.0 ** np.ceil(np.log10(1.8 * maximum_delay)),
    )
    axis.set_yscale("log")
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(0.72, upper_delay)
    powers = tuple(
        10 ** exponent
        for exponent in range(
            int(np.floor(np.log10(upper_delay))) + 1
        )
        if 10 ** exponent <= upper_delay
    )
    axis.set_yticks(powers, tuple(str(value) for value in powers))
    axis.set_xticks(
        (lane_x[_TSALLIS], lane_x[_LINEAR]),
        ("Tsallis-INF", "LinearExp2"),
    )
    axis.tick_params(axis="both", which="major", labelsize=6.5, length=2.5)
    axis.tick_params(
        axis="y",
        which="minor",
        left=False,
        right=False,
        labelleft=False,
    )
    axis.set_ylabel(
        "Turns from Prime onset\n(log scale)",
        fontsize=6.8,
        labelpad=2.0,
    )
    axis.text(
        0.025,
        0.975,
        "(c)",
        transform=axis.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "square,pad=0.08",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=7,
    )
    axis.grid(axis="y", alpha=0.24, linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _draw_missing_alice_alarm_panel(axis: Any) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.text(
        0.025,
        0.975,
        "(c)",
        transform=axis.transAxes,
        color="#000000",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axis.text(
        0.5,
        0.5,
        "Alice detector not run",
        transform=axis.transAxes,
        color="#4D4D4D",
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#8C8C8C",
            "linewidth": 0.6,
        },
    )


def _plot_timing_comparison(
    _manifest: Mapping[str, Any],
    _runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    fields = (
        ("ordered_minus_shuffled_mean_alice_reward", "Realized 500k-turn reward"),
        ("ordered_minus_shuffled_audit_expected_alice_reward", "Independent audit expectation"),
    )
    rng = np.random.default_rng(20_260_716)
    for axis, (field, title) in zip(axes, fields):
        for defender_index, defender in enumerate((_TSALLIS, _LINEAR)):
            rows = [row for row in comparisons if row["defender_label"] == defender]
            values = np.asarray([float(row[field]) for row in rows])
            positions = defender_index + rng.uniform(-0.12, 0.12, size=values.size)
            markers = [_FAMILY_MARKER[_family_from_graph(row["graph_id"])] for row in rows]
            for x, value, marker in zip(positions, values, markers):
                axis.scatter(
                    x,
                    value,
                    marker=marker,
                    s=31,
                    facecolors="none",
                    edgecolors=_DEFENDER_COLOR[defender],
                    linewidths=0.9,
                    alpha=0.9,
                )
            axis.plot(
                [defender_index - 0.18, defender_index + 0.18],
                [np.median(values), np.median(values)],
                color=_DEFENDER_COLOR[defender],
                linewidth=2.6,
            )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks((0, 1), ("Tsallis-INF", "LinearExp2"))
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Ordered minus shuffled Alice reward\n(negative means timing helps Eve)")
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=_FAMILY_MARKER[family],
            color="black",
            markerfacecolor="none",
            linestyle="none",
            markersize=5.5,
            label=_FAMILY_DISPLAY[family],
        )
        for family in _FAMILY_MARKER
    ]
    figure.legend(
        handles=family_handles,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        fontsize=6.8,
    )
    figure.suptitle(
        "Temporal ordering under an exactly matched physical action budget",
        y=1.01,
    )
    _save_publication_figure(figure, path)


def _plot_probe_information(
    _manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    _comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    ordered = [row for row in runs if row["condition"] == "ordered_prime_probe"]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    for row in ordered:
        defender = row["defender_label"]
        exact = float(row["probe_information_regimes"]["live_policy_upper_bound_offline"])
        interval = row["probe_information_regimes"]["local_instrument_count_interval"]
        lower = float(interval["target_mass_lower"])
        upper = float(interval["target_mass_upper"])
        midpoint = (lower + upper) / 2.0
        axes[0].errorbar(
            exact,
            midpoint,
            yerr=np.asarray([[midpoint - lower], [upper - midpoint]]),
            marker="o" if defender == _TSALLIS else "^",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=_DEFENDER_COLOR[defender],
            ecolor=_DEFENDER_COLOR[defender],
            elinewidth=0.8,
            capsize=1.8,
            linestyle="none",
            alpha=0.82,
        )
    exact_values = np.asarray([
        float(row["probe_information_regimes"]["live_policy_upper_bound_offline"])
        for row in ordered
    ])
    interval_lowers = np.asarray([
        float(
            row["probe_information_regimes"]["local_instrument_count_interval"][
                "target_mass_lower"
            ]
        )
        for row in ordered
    ])
    zoom_lower = max(
        0.0,
        min(float(exact_values.min()), float(interval_lowers.min())) - 0.015,
    )
    axes[0].plot(
        (zoom_lower, 1.001),
        (zoom_lower, 1.001),
        color="black",
        linewidth=1.0,
    )
    axes[0].set(
        xlim=(zoom_lower, 1.001),
        ylim=(zoom_lower, 1.001),
        xlabel="Offline Alice target-family policy mass",
        ylabel="Local count-derived target-mass interval",
        title="(a) What Eve can infer locally",
    )
    regimes = (
        (
            "Local count\n(online Eve)",
            lambda row: _interval_midpoint(
                row["probe_information_regimes"]["local_instrument_count_interval"]
            ),
        ),
        (
            "Route history\n(offline bound)",
            lambda row: float(
                row["probe_information_regimes"][
                    "sampled_route_history_upper_bound_offline"
                ]
            ),
        ),
        (
            "$\\eta$ + rule\n(no actions)",
            lambda row: float(
                row["probe_information_regimes"][
                    "eta_and_update_rule_without_action_identity"
                ]
            ),
        ),
    )
    for regime_index, (_label, estimate) in enumerate(regimes):
        for defender, offset in ((_TSALLIS, -0.11), (_LINEAR, 0.11)):
            rows = [row for row in ordered if row["defender_label"] == defender]
            error = np.asarray([
                estimate(row)
                - float(
                    row["probe_information_regimes"][
                        "live_policy_upper_bound_offline"
                    ]
                )
                for row in rows
            ])
            axes[1].scatter(
                np.full(len(rows), regime_index + offset),
                error,
                marker="o" if defender == _TSALLIS else "^",
                s=27,
                facecolors=(
                    "none" if defender == _TSALLIS else _DEFENDER_COLOR[defender]
                ),
                edgecolors=_DEFENDER_COLOR[defender],
                linewidths=0.8,
                alpha=0.82,
                label=(
                    _DEFENDER_LABEL[defender] if regime_index == 0 else None
                ),
            )
            axes[1].plot(
                [regime_index + offset - 0.055, regime_index + offset + 0.055],
                [np.median(error), np.median(error)],
                color=_DEFENDER_COLOR[defender],
                linewidth=2.2,
            )
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set(
        xticks=np.arange(len(regimes)),
        xticklabels=[row[0] for row in regimes],
        ylabel="Estimate minus offline policy mass",
        title="(b) Information-regime estimation error",
    )
    axes[1].legend(frameon=False, loc="best")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        "The exploit trigger uses 5,000 count-valued local probe records—never Alice's weights",
        y=1.015,
    )
    _save_publication_figure(figure, path)


def _plot_recovery(
    _manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    _comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    ordered = [row for row in runs if row["condition"] == "ordered_prime_probe"]
    criteria = ("service", "target_mass", "policy_tv", "joint")
    labels = ("Service", "Target mass", "Policy TV", "Joint")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    x = np.arange(len(criteria))
    width = 0.34
    for defender_index, defender in enumerate((_TSALLIS, _LINEAR)):
        selected = [row for row in ordered if row["defender_label"] == defender]
        recovery_fraction = [
            np.mean([bool(row["recovery"][criterion]["recovered"]) for row in selected])
            for criterion in criteria
        ]
        axes[0].bar(
            x + (defender_index - 0.5) * width,
            recovery_fraction,
            width=width,
            color=_DEFENDER_COLOR[defender],
            alpha=0.82,
            hatch="//" if defender == _LINEAR else None,
            label=_DEFENDER_LABEL[defender],
        )
        joint = [row["recovery"]["joint"] for row in selected]
        latencies = np.asarray([float(value["latency_turns"]) for value in joint])
        censored = np.asarray([bool(value["right_censored"]) for value in joint])
        jitter = np.linspace(-0.12, 0.12, len(latencies))
        position = defender_index
        axes[1].scatter(
            position + jitter[~censored],
            latencies[~censored] / 1_000.0,
            marker="o",
            s=28,
            facecolors="none",
            edgecolors=_DEFENDER_COLOR[defender],
            linewidths=0.9,
        )
        axes[1].scatter(
            position + jitter[censored],
            latencies[censored] / 1_000.0,
            marker="^",
            s=36,
            color=_DEFENDER_COLOR[defender],
            label=("right-censored" if defender_index == 0 else None),
        )
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylim=(0.0, 1.05),
        ylabel="Fraction recovered within 145k turns",
        title="(a) Sustained recovery by criterion",
    )
    axes[1].set(
        xticks=(0, 1),
        xticklabels=("Tsallis-INF", "LinearExp2"),
        ylabel="Joint recovery latency (thousand turns)",
        title="(b) Joint recovery; triangles are censored",
    )
    axes[0].legend(frameon=False, fontsize=6.7)
    axes[1].legend(frameon=False, fontsize=6.7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Recovery to the learner's own late-warmup envelope", y=1.01)
    _save_publication_figure(figure, path)


def _plot_topology_defender_outcomes(
    _manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    _checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    families = ("memorex", "hiberniaireland", "layer42", "karen", "cernet")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    for defender, offset, marker in ((_TSALLIS, -0.08, "o"), (_LINEAR, 0.08, "^")):
        medians = []
        q1 = []
        q3 = []
        bottleneck_median = []
        bottleneck_q1 = []
        bottleneck_q3 = []
        for family in families:
            selected = [
                row for row in comparisons
                if row["defender_label"] == defender
                and _family_from_graph(row["graph_id"]) == family
            ]
            values = np.asarray([
                float(row["ordered_minus_shuffled_mean_alice_reward"])
                for row in selected
            ])
            medians.append(float(np.median(values)))
            q1.append(float(np.quantile(values, 0.25)))
            q3.append(float(np.quantile(values, 0.75)))
            ordered_rows = [
                row for row in runs
                if row["condition"] == "ordered_prime_probe"
                and row["defender_label"] == defender
                and str(row["family"]).lower() == family
            ]
            uplift = np.asarray([
                float(_virtual_bottleneck(row)["post_prime_minus_late_warmup_mean"])
                for row in ordered_rows
            ])
            bottleneck_median.append(float(np.median(uplift)))
            bottleneck_q1.append(float(np.quantile(uplift, 0.25)))
            bottleneck_q3.append(float(np.quantile(uplift, 0.75)))
        positions = np.arange(len(families)) + offset
        axes[0].errorbar(
            positions,
            medians,
            yerr=np.asarray([np.asarray(medians) - q1, np.asarray(q3) - medians]),
            marker=marker,
            markersize=5,
            color=_DEFENDER_COLOR[defender],
            markerfacecolor="white" if defender == _TSALLIS else _DEFENDER_COLOR[defender],
            capsize=2,
            linewidth=1.0,
            linestyle="none",
            label=_DEFENDER_LABEL[defender],
        )
        axes[1].errorbar(
            positions,
            bottleneck_median,
            yerr=np.asarray([
                np.asarray(bottleneck_median) - bottleneck_q1,
                np.asarray(bottleneck_q3) - bottleneck_median,
            ]),
            marker=marker,
            markersize=5,
            color=_DEFENDER_COLOR[defender],
            markerfacecolor="white" if defender == _TSALLIS else _DEFENDER_COLOR[defender],
            capsize=2,
            linewidth=1.0,
            linestyle="none",
        )
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_ylabel("Ordered minus shuffled Alice reward")
    axes[0].set_title("(a) Timing effect; median and seed IQR")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_ylabel("Post-prime target mass minus warmup mean")
    axes[1].set_title("(b) Priming uplift; zero can mean prior concentration")
    memorex_rows = [
        row
        for row in runs
        if row["condition"] == "ordered_prime_probe"
        and str(row["family"]).lower() == "memorex"
    ]
    memorex_baseline_mass = [
        float(value["late_warmup_target_mass_mean"])
        for row in memorex_rows
        for value in (_virtual_bottleneck(row),)
        if "late_warmup_target_mass_mean" in value
    ]
    memorex_exploit_service = [
        float(row["phase_metrics"]["exploit"]["mean_alice_reward"])
        for row in memorex_rows
        if "phase_metrics" in row
    ]
    if memorex_baseline_mass:
        exploit_text = (
            "\n"
            f"exploit service {min(memorex_exploit_service):.3f}–"
            f"{max(memorex_exploit_service):.3f}"
            if memorex_exploit_service
            else ""
        )
        axes[1].annotate(
            "Memorex already carried\n"
            f"{min(memorex_baseline_mass):.2%}–{max(memorex_baseline_mass):.2%}\n"
            f"target-family mass{exploit_text}",
            xy=(0.0, 0.01),
            xytext=(0.48, 0.24),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#333333",
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "linewidth": 0.8,
            },
        )
    for axis in axes:
        axis.set_xticks(
            np.arange(len(families)),
            [_FAMILY_DISPLAY[name] for name in families],
            rotation=28,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=6.8)
    figure.suptitle(
        "Attack timing and target-family concentration by normalized Real Topology",
        y=1.01,
    )
    _save_publication_figure(figure, path)


def _plot_plan_physics(
    _manifest: Mapping[str, Any],
    _runs: Sequence[Mapping[str, Any]],
    _comparisons: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> None:
    ordered = sorted(checkpoints.values(), key=lambda row: str(row["family"]).lower())
    families = [str(row["family"]).lower() for row in ordered]
    plans = [row["plan"] for row in ordered]
    x = np.arange(len(families))
    fields = (
        ("target_route_fraction_under_uniform_policy", "Target-family fraction"),
        ("topological_coverage_floor", "Prime max-min coverage floor"),
        ("physical_prime_loss_floor", "Minimum alternative-route prime loss"),
        ("physical_exploit_loss_mean", "Mean target-route exploit loss"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.15))
    for axis, (field, title) in zip(axes.flat, fields):
        values = np.asarray([float(plan[field]) for plan in plans])
        axis.bar(x, values, color="#56B4E9", edgecolor="#005A8D", linewidth=0.7)
        axis.set_xticks(
            x,
            [_FAMILY_DISPLAY[name] for name in families],
            rotation=25,
            ha="right",
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        axis.set_title(f"{label} {axis.get_title()}")
    figure.suptitle(
        "Selected steering plans: topology constraints and compiled physical losses",
        y=1.01,
    )
    _save_publication_figure(figure, path)


def _mark_prime_probe_phases(
    axes: Sequence[Any],
    schedule: Mapping[str, Any],
) -> None:
    boundaries = (
        (0, int(schedule["warmup_turns"]), "warmup"),
        (int(schedule["warmup_turns"]), int(schedule["warmup_turns"]) + int(schedule["prime_turns"]), "prime"),
        (int(schedule["warmup_turns"]) + int(schedule["prime_turns"]), int(schedule["warmup_turns"]) + int(schedule["prime_turns"]) + int(schedule["probe_turns"]), "probe"),
        (int(schedule["warmup_turns"]) + int(schedule["prime_turns"]) + int(schedule["probe_turns"]), int(schedule["warmup_turns"]) + int(schedule["prime_turns"]) + int(schedule["probe_turns"]) + int(schedule["exploit_turns"]), "exploit"),
        (int(schedule["warmup_turns"]) + int(schedule["prime_turns"]) + int(schedule["probe_turns"]) + int(schedule["exploit_turns"]), int(schedule["total_turns"]), "recovery"),
    )
    for axis in axes:
        for _start, stop, _phase in boundaries[:-1]:
            axis.axvline(stop, color="#777777", linewidth=0.6, alpha=0.72)

    top = axes[0]
    phase_display = {
        "warmup": "Warmup",
        "prime": "Prime",
        "exploit": "Exploit",
        "recovery": "Withdrawal",
    }
    for start, stop, phase in boundaries:
        top.add_patch(
            Rectangle(
                (start, 1.035),
                stop - start,
                0.07,
                transform=top.get_xaxis_transform(),
                facecolor=_PHASE_COLOR[phase],
                edgecolor="white",
                linewidth=0.6,
                clip_on=False,
            )
        )
        if phase != "probe":
            top.text(
                (start + stop) / 2.0,
                1.07,
                phase_display[phase],
                transform=top.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=6.3,
                fontweight="bold",
            )
    probe_start, probe_stop, _phase = boundaries[2]
    top.annotate(
        f"Probe\n{_format_turns(probe_start)}--{_format_turns(probe_stop)}",
        xy=((probe_start + probe_stop) / 2.0, 1.07),
        xycoords=top.get_xaxis_transform(),
        xytext=(probe_start - 0.012 * int(schedule["total_turns"]), 0.62),
        textcoords=top.get_xaxis_transform(),
        ha="right",
        va="center",
        fontsize=6.5,
        color="#24536B",
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": "#24536B",
            "linewidth": 0.75,
        },
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=6,
    )


def _write_comparison_csv(
    comparisons: Sequence[Mapping[str, Any]], path: Path
) -> None:
    fields = (
        "graph_id",
        "defender_label",
        "seed",
        "identical_action_multiset",
        "ordered_triggered",
        "ordered_minus_shuffled_mean_alice_reward",
        "ordered_minus_shuffled_audit_expected_alice_reward",
        "ordered_minus_shuffled_mean_eve_public_utility",
        "joint_recovered",
        "joint_recovery_latency_turns",
        "joint_recovery_right_censored",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            comparisons,
            key=lambda item: (str(item["graph_id"]), str(item["defender_label"]), int(item["seed"])),
        ):
            joint = row["ordered_recovery"]["joint"]
            writer.writerow({
                **{field: row[field] for field in fields[:8]},
                "joint_recovered": joint["recovered"],
                "joint_recovery_latency_turns": joint["latency_turns"],
                "joint_recovery_right_censored": joint["right_censored"],
            })


def _report_markdown(
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> str:
    ordered = [row for row in runs if row["condition"] == "ordered_prime_probe"]
    lines = [
        "# Physical Prime-and-Probe: 500k-turn defender study",
        "",
        f"Source corpus selection digest: `{manifest['source_corpus_selection_digest_blake2b_128']}`.",
        "",
        "The study contains five predeclared normalized Real Topologies, two Alice defenders, and three learner/outcome-resampling seeds conditional on one shared equal-coverage physical bank per topology. Each ordered intervention is paired with a temporal permutation of exactly the same physical Eve-action multiset.",
        "",
        "Eve acts only through declared compiled-histogram instruments. The exploit decision uses the count-valued local probe record; it never receives Alice's policy, weights, route identity, target-active flag, or audit outcomes. The independent audit split is used only for scoring.",
        "",
        "| Defender | Runs | Median post-prime target-mass uplift | Probe triggered | Median ordered−shuffled Alice reward | Median audit difference | Joint recovered | Median joint latency among recovered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for defender in (_TSALLIS, _LINEAR):
        rows = [row for row in comparisons if row["defender_label"] == defender]
        defender_ordered = [
            row for row in ordered if row["defender_label"] == defender
        ]
        joint = [row["ordered_recovery"]["joint"] for row in rows]
        recovered_latency = [
            float(value["latency_turns"]) for value in joint if value["recovered"]
        ]
        latency_text = (
            f"{np.median(recovered_latency):,.0f} turns" if recovered_latency else "none"
        )
        lines.append(
            f"| {_DEFENDER_LABEL[defender]} | {len(rows)} | "
            f"{np.median([float(_virtual_bottleneck(row)['post_prime_minus_late_warmup_mean']) for row in defender_ordered]):+.6f} | "
            f"{sum(bool(row['ordered_triggered']) for row in rows)}/{len(rows)} | "
            f"{np.median([float(row['ordered_minus_shuffled_mean_alice_reward']) for row in rows]):+.6f} | "
            f"{np.median([float(row['ordered_minus_shuffled_audit_expected_alice_reward']) for row in rows]):+.6f} | "
            f"{sum(bool(value['recovered']) for value in joint)}/{len(joint)} | {latency_text} |"
        )
    tsallis_comparisons = [
        row for row in comparisons if row["defender_label"] == _TSALLIS
    ]
    linear_comparisons = [
        row for row in comparisons if row["defender_label"] == _LINEAR
    ]
    tsallis_uplift = [
        float(_virtual_bottleneck(row)["post_prime_minus_late_warmup_mean"])
        for row in ordered if row["defender_label"] == _TSALLIS
    ]
    linear_uplift = [
        float(_virtual_bottleneck(row)["post_prime_minus_late_warmup_mean"])
        for row in ordered if row["defender_label"] == _LINEAR
    ]
    lines.extend([
        "",
        "## Main findings",
        "",
        f"Priming changes the target-family policy mass by a median {np.median(tsallis_uplift):+.6f} for Tsallis-INF and {np.median(linear_uplift):+.6f} for LinearExp2. Memorex is the important exception visible in the topology figure: the selected target family already carries almost all late-warmup mass, so its continuous uplift is near zero rather than evidence that priming created a new concentration.",
        "",
        f"Ordered timing lowers Alice's realized reward relative to the shuffled identical-budget control in {sum(float(row['ordered_minus_shuffled_mean_alice_reward']) < 0.0 for row in linear_comparisons)}/15 LinearExp2 runs, compared with {sum(float(row['ordered_minus_shuffled_mean_alice_reward']) < 0.0 for row in tsallis_comparisons)}/15 Tsallis-INF runs. The paired medians are {np.median([float(row['ordered_minus_shuffled_mean_alice_reward']) for row in linear_comparisons]):+.6f} and {np.median([float(row['ordered_minus_shuffled_mean_alice_reward']) for row in tsallis_comparisons]):+.6f}, respectively. On these five cases and this declared schedule, the result does not support the hypothesis that the LinearExp2 defender is more resistant to temporal steering.",
        "",
        f"Joint recovery occurs within 145,000 turns in {sum(bool(row['ordered_recovery']['joint']['recovered']) for row in tsallis_comparisons)}/15 Tsallis-INF runs and {sum(bool(row['ordered_recovery']['joint']['recovered']) for row in linear_comparisons)}/15 LinearExp2 runs. Right-censoring is therefore part of the result and is not converted into an inferred recovery time.",
    ])
    lines.extend([
        "",
        "## Selected physical intervention plans",
        "",
        "| Topology | Routes | Target routes | Exploit instrument | Prime actions | Coverage floor | Prime loss floor | Exploit loss mean |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ])
    for checkpoint in sorted(checkpoints.values(), key=lambda row: str(row["family"]).lower()):
        plan = checkpoint["plan"]
        lines.append(
            f"| {_FAMILY_DISPLAY[str(checkpoint['family']).lower()]} | {int(checkpoint['case']['route_count'])} | "
            f"{len(plan['target_route_indices'])} | `{plan['exploit_action_id']}` | "
            f"{len(plan['prime_action_indices'])} | {float(plan['topological_coverage_floor']):.4f} | "
            f"{float(plan['physical_prime_loss_floor']):.6f} | {float(plan['physical_exploit_loss_mean']):.6f} |"
        )
    local_errors = []
    for row in ordered:
        exact = float(row["probe_information_regimes"]["live_policy_upper_bound_offline"])
        interval = row["probe_information_regimes"]["local_instrument_count_interval"]
        local_errors.append(_interval_midpoint(interval) - exact)
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"Across the {len(ordered)} ordered runs, the median count-derived target-mass midpoint error relative to the offline policy diagnostic is {np.median(local_errors):+.6f}.",
        "",
        "This is a finite-horizon leader/follower experiment, not an assertion that an asymptotic Stackelberg theorem transfers unchanged to the nonstationary, general-sum physical network game. The shuffled control isolates temporal ordering at a fixed action economy; it does not remove stochastic transcript variation. Three resampling seeds per topology quantify finite-run repeatability conditional on the shared bank for these five cases, not device-calibration uncertainty or uncertainty over the full topology population.",
        "",
        "The local target-mass interval is conditional on the fixed 64-draw-per-cell feedback-bank calibration: its standard-error term measures the 5,000-turn probe record, while the route-cell minimum/maximum calibration bounds remain fixed. It is not a joint confidence interval over a newly estimated device model.",
        "",
        "## Artifact guide",
        "",
        "- `prime_probe_phase_protocol`: exact 500k phase schedule, controller decisions, and the online information boundary.",
        "- `prime_probe_network_mechanism`: route-family view of priming alternatives, locally probing the target, exploiting concentration, and withdrawing for recovery.",
        "- `ordered_phase_dynamics`: service and target-family policy response through warmup, prime, probe, exploit, and recovery.",
        "- `timing_matters`: ordered-versus-shuffled outcome under an exactly matched physical action multiset.",
        "- `probe_information`: locally count-identifiable target mass versus offline diagnostics unavailable to Eve.",
        "- `recovery`: sustained recovery to each run's late-warmup service/policy envelope.",
        "- `topology_defender_outcomes`: seed-resolved timing effects and continuous post-prime target-family uplift.",
        "- `selected_bottleneck_physics`: topological coverage and compiled physical loss of each selected plan.",
    ])
    return "\n".join(lines) + "\n"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _family_from_graph(graph_id: str) -> str:
    normalized = str(graph_id).lower()
    for family in _FAMILY_MARKER:
        if family in normalized:
            return family
    raise ValueError(f"cannot recover study family from graph id {graph_id!r}")


def _interval_midpoint(interval: Mapping[str, Any]) -> float:
    return (
        float(interval["target_mass_lower"])
        + float(interval["target_mass_upper"])
    ) / 2.0


def _virtual_bottleneck(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read stored telemetry or reconstruct it for the first production run."""

    if "virtual_bottleneck" in row:
        return dict(row["virtual_bottleneck"])
    schedule = row["schedule"]
    probe_start = int(schedule["warmup_turns"]) + int(schedule["prime_turns"])
    post_prime = next(
        point for point in row["trajectory"] if int(point["turn"]) == probe_start
    )
    baseline = row["baseline_envelope"]
    upper = float(baseline["target_mass_upper_97p5"])
    baseline_start = int(schedule["warmup_turns"]) - int(
        schedule["baseline_tail_turns"]
    )
    baseline_rows = [
        point
        for point in row["trajectory"]
        if baseline_start < int(point["turn"]) <= int(schedule["warmup_turns"])
    ]
    baseline_center = float(np.mean([
        float(point["alice_target_family_policy_mass"])
        for point in baseline_rows
    ]))
    mass = float(post_prime["alice_target_family_policy_mass"])
    return {
        "offline_scoring_only": True,
        "post_prime_target_family_policy_mass": mass,
        "late_warmup_target_mass_mean": baseline_center,
        "post_prime_minus_late_warmup_mean": mass - baseline_center,
        "post_prime_exceeds_late_warmup_upper_quantile": mass > upper,
        "used_by_eve_trigger": False,
        "reconstructed_from_recorded_late_warmup_rows": True,
    }


def _save_publication_figure(
    figure: Any,
    png_path: Path,
    *,
    apply_tight_layout: bool = True,
) -> None:
    if apply_tight_layout:
        figure.tight_layout()
    figure.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    svg_path = png_path.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
