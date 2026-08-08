"""Publication figures for the Cernet steering-detection study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from .calibration_bank import HistogramPayoffBank


_DEFENDER_LABELS = {
    "tsallis_inf_rv_eta1": r"Tsallis-INF (RV, $\eta{=}1$)",
    "linear_exp2_goptimal_eta2": r"LinearExp2 (G-optimal, $\eta{=}2$)",
}
_DEFENDER_COLORS = {
    "tsallis_inf_rv_eta1": "#0072B2",
    "linear_exp2_goptimal_eta2": "#D55E00",
}
_CONDITION_LABELS = {
    "clean_control": "Clean",
    "detect_only": "Detect only",
    "fixed_sentinel_05": "Fixed 5%",
    "adaptive_sentinel_02_20": "Adaptive 2→20%",
    "adaptive_sentinel_02_20_restart": "Adaptive + restart",
    "quantum_detect_freeze_policy": "4-outcome detect + freeze",
    "classical_binary_detect_freeze_policy": "Binary detect + freeze",
}
_CONDITION_COLORS = {
    "clean_control": "#777777",
    "detect_only": "#CC79A7",
    "fixed_sentinel_05": "#009E73",
    "adaptive_sentinel_02_20": "#56B4E9",
    "adaptive_sentinel_02_20_restart": "#E69F00",
    "quantum_detect_freeze_policy": "#0072B2",
    "classical_binary_detect_freeze_policy": "#6A3D9A",
}
_CONDITION_ORDER = tuple(_CONDITION_LABELS)
_CORE_CONDITION_ORDER = (
    "clean_control",
    "detect_only",
    "fixed_sentinel_05",
    "adaptive_sentinel_02_20",
    "adaptive_sentinel_02_20_restart",
)
_RESPONSE_CONDITION_ORDER = (
    "detect_only",
    "quantum_detect_freeze_policy",
    "classical_binary_detect_freeze_policy",
    "adaptive_sentinel_02_20_restart",
)
_DAMAGE_REGIMES = (
    "clean",
    "colearning",
    "strict",
    "quantum_freeze",
    "classical_freeze",
    "restart",
)
_DAMAGE_REGIME_LABELS = {
    "clean": "Clean",
    "colearning": "Co-learning\nfrom turn 1",
    "strict": "Strict commitment\nonset 25k",
    "quantum_freeze": "4-outcome\nfreeze",
    "classical_freeze": "Binary\nfreeze",
    "restart": "Monitoring\n+ restart",
}
_DAMAGE_REGIME_COLORS = {
    "clean": "#777777",
    "colearning": "#009E73",
    "strict": "#CC79A7",
    "quantum_freeze": "#0072B2",
    "classical_freeze": "#6A3D9A",
    "restart": "#E69F00",
}


def render_stackelberg_detection_artifacts(
    run_dir: str | Path,
) -> list[Path]:
    """Render PNG/PDF pairs from one complete study directory."""

    source = Path(run_dir)
    summary = json.loads(
        (source / "summary.json").read_text(encoding="utf-8")
    )
    results = [
        json.loads(line)
        for line in (source / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    controls = [
        json.loads(line)
        for line in (source / "colearning_controls.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not summary.get("complete"):
        raise ValueError("detection study is incomplete")
    if len(controls) != 6:
        raise ValueError("ordinary co-learning controls are incomplete")
    bank = HistogramPayoffBank.load(summary["bank_path"])
    output = source / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    written = []
    written.extend(_detection_figure(results, output))
    written.extend(_tradeoff_figure(results, output))
    written.extend(_steering_trajectory_figure(results, output))
    written.extend(_evidence_trajectory_figure(results, output))
    written.extend(
        _classical_quantum_comparison_figure(results, bank, output)
    )
    written.extend(
        _peak_damage_figure(results, controls, bank, output)
    )
    written.append(
        _write_damage_window_csv(results, controls, bank, output)
    )
    written.append(
        _write_rows_csv(
            output / "study_aggregates.csv",
            summary["aggregates"],
            (
                "defender_label",
                "condition",
                "seed_count",
                "pre_attack_alarm_rate",
                "post_attack_detection_rate",
                "mean_quantum_four_outcome_detection_delay_turns",
                "mean_classical_binary_service_detection_delay_turns",
                "mean_monitoring_request_fraction",
                "mean_clean_window_service_reward",
                "mean_attack_window_service_reward",
                "mean_application_reward_per_total_request",
                "mean_attack_window_eve_public_utility",
                "mean_final_intended_route_policy_mass",
                "mean_learner_restart_count",
                "service_policy_freeze_rate",
            ),
        )
    )
    written.append(
        _write_rows_csv(
            output / "colearning_controls.csv",
            controls,
            (
                "defender_label",
                "seed",
                "turns",
                "mean_alice_reward",
                "evaluation_start_turn",
                "mean_pre_evaluation_alice_reward",
                "mean_evaluation_interval_alice_reward",
                "audit_current_profile_alice_reward",
                "audit_current_best_response_reward",
                "audit_current_reward_retention",
                "audit_equilibrium_alice_reward",
            ),
        )
    )
    return written


def _detection_figure(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.7),
        constrained_layout=True,
    )
    x = np.arange(len(_CORE_CONDITION_ORDER), dtype=float)
    offsets = {
        "tsallis_inf_rv_eta1": -0.13,
        "linear_exp2_goptimal_eta2": 0.13,
    }
    for defender in _DEFENDER_LABELS:
        grouped = [
            _select(rows, defender=defender, condition=condition)
            for condition in _CORE_CONDITION_ORDER
        ]
        detection = []
        for condition, group in zip(_CORE_CONDITION_ORDER, grouped):
            field = (
                "pre_attack_alarm"
                if condition == "clean_control"
                else "detected_after_attack"
            )
            detection.append(
                np.mean([bool(row[field]) for row in group])
            )
        axes[0].plot(
            x + offsets[defender],
            detection,
            marker="o" if defender.startswith("tsallis") else "^",
            linestyle="none",
            markersize=7,
            color=_DEFENDER_COLORS[defender],
            label=_DEFENDER_LABELS[defender],
        )
        attacked_groups = grouped[1:]
        delay_centers = [np.nan]
        delay_lower = []
        delay_upper = []
        for group in attacked_groups:
            values = np.asarray([
                float(row["detection_delay_turns"])
                for row in group
                if row.get("detection_delay_turns") is not None
            ])
            if values.size:
                center = float(np.median(values))
                lower, upper = np.quantile(values, [0.25, 0.75])
                delay_centers.append(center)
                delay_lower.append(center - float(lower))
                delay_upper.append(float(upper) - center)
            else:
                delay_centers.append(np.nan)
                delay_lower.append(0.0)
                delay_upper.append(0.0)
        delay_lower.insert(0, 0.0)
        delay_upper.insert(0, 0.0)
        axes[1].errorbar(
            x + offsets[defender],
            delay_centers,
            yerr=np.asarray([delay_lower, delay_upper]),
            marker="o" if defender.startswith("tsallis") else "^",
            linestyle="none",
            capsize=3,
            markersize=7,
            color=_DEFENDER_COLORS[defender],
        )

    axes[0].set_title("(a) Alarm probability")
    axes[0].set_ylabel("Clean false alarm / attacked detection")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[0].legend(frameon=False, loc="center right")
    axes[1].set_title("(b) Delay conditional on detection")
    axes[1].set_ylabel("Detection delay (turns; median and seed IQR)")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [_CONDITION_LABELS[name] for name in _CORE_CONDITION_ORDER],
            rotation=24,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Cernet: endpoint-visible detection of strict Stackelberg steering",
        fontsize=13,
    )
    return _save_figure(fig, output / "detection_performance")


def _tradeoff_figure(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.8),
        constrained_layout=True,
    )
    for defender, marker in (
        ("tsallis_inf_rv_eta1", "o"),
        ("linear_exp2_goptimal_eta2", "^"),
    ):
        for condition in (
            *_CORE_CONDITION_ORDER,
            "quantum_detect_freeze_policy",
        ):
            group = _select(
                rows, defender=defender, condition=condition
            )
            if not group:
                continue
            monitor = np.asarray([
                row["monitoring_request_fraction"] for row in group
            ], dtype=float)
            application_yield = np.asarray([
                row["mean_service_reward"]
                * row["service_request_count"]
                / row["turns"]
                for row in group
            ], dtype=float)
            eve = np.asarray([
                row["mean_attack_window_eve_public_utility"]
                for row in group
                if row["mean_attack_window_eve_public_utility"] is not None
            ], dtype=float)
            target = np.asarray([
                row["final_target_mass"][
                    "intended_route_current_service_policy_mass"
                ]
                for row in group
            ], dtype=float)
            axes[0].scatter(
                np.mean(monitor),
                np.mean(application_yield),
                marker=marker,
                s=65,
                color=_CONDITION_COLORS[condition],
                edgecolor="black",
                linewidth=0.45,
            )
            if eve.size:
                axes[1].scatter(
                    np.mean(target),
                    np.mean(eve),
                    marker=marker,
                    s=65,
                    color=_CONDITION_COLORS[condition],
                    edgecolor="black",
                    linewidth=0.45,
                )
    axes[0].set_title("(a) Monitoring cost and application yield")
    axes[0].set_xlabel("Monitoring-request fraction")
    axes[0].set_ylabel("Application reward per total request budget")
    axes[1].set_title("(b) Residual steering and Eve utility")
    axes[1].set_xlabel("Final intended-route policy mass")
    axes[1].set_ylabel("Attack-window Eve public utility")
    for axis in axes:
        axis.grid(alpha=0.25)
    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=_CONDITION_COLORS[name],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=_CONDITION_LABELS[name],
        )
        for name in (
            *_CORE_CONDITION_ORDER,
            "quantum_detect_freeze_policy",
        )
    ]
    defender_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="black",
            label=_DEFENDER_LABELS[name],
        )
        for name, marker in (
            ("tsallis_inf_rv_eta1", "o"),
            ("linear_exp2_goptimal_eta2", "^"),
        )
    ]
    axes[0].legend(
        handles=condition_handles,
        frameon=False,
        fontsize=8,
        loc="best",
    )
    axes[1].legend(
        handles=defender_handles,
        frameon=False,
        fontsize=8,
        loc="best",
    )
    fig.suptitle(
        "Cernet mitigation trade-offs under the same physical commitment",
        fontsize=13,
    )
    return _save_figure(fig, output / "mitigation_tradeoffs")


def _steering_trajectory_figure(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.8),
        sharey=True,
        constrained_layout=True,
    )
    for axis, defender in zip(axes, _DEFENDER_LABELS):
        selected = _RESPONSE_CONDITION_ORDER
        for condition in selected:
            group = _select(
                rows, defender=defender, condition=condition
            )
            turns, matrix = _trajectory_matrix(
                group,
                "intended_route_current_service_policy_mass",
            )
            median = np.median(matrix, axis=0)
            lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
            axis.plot(
                turns,
                median,
                color=_CONDITION_COLORS[condition],
                linewidth=1.7,
                linestyle=(
                    "--"
                    if condition
                    == "classical_binary_detect_freeze_policy"
                    else "-"
                ),
                label=_CONDITION_LABELS[condition],
            )
            axis.fill_between(
                turns,
                lower,
                upper,
                color=_CONDITION_COLORS[condition],
                alpha=0.14,
                linewidth=0,
            )
        attack_start = int(group[0]["experiment_config"]["attack_start_turn"])
        axis.axvline(
            attack_start,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(_DEFENDER_LABELS[defender])
        axis.set_xlabel("Turn")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Intended-route service-policy mass")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Effect of alarm response on Eve's intended route "
        "(median and seed IQR)",
        fontsize=13,
    )
    return _save_figure(fig, output / "steering_policy_trajectories")


def _evidence_trajectory_figure(
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.8),
        sharey=True,
        constrained_layout=True,
    )
    for axis, defender in zip(axes, _DEFENDER_LABELS):
        for condition in (
            "clean_control",
            *_RESPONSE_CONDITION_ORDER,
        ):
            group = _select(
                rows, defender=defender, condition=condition
            )
            turns, matrix = _transition_matrix(
                group, "log_e_value"
            )
            threshold = -math.log(
                group[0]["experiment_config"][
                    "detector_false_alarm_level"
                ]
            )
            displayed = np.minimum(matrix, 1.5 * threshold)
            median = np.median(displayed, axis=0)
            lower, upper = np.quantile(
                displayed, [0.25, 0.75], axis=0
            )
            axis.plot(
                turns,
                median,
                color=_CONDITION_COLORS[condition],
                linewidth=1.55,
                linestyle=(
                    "--"
                    if condition
                    == "classical_binary_detect_freeze_policy"
                    else "-"
                ),
                label=_CONDITION_LABELS[condition],
            )
            axis.fill_between(
                turns,
                lower,
                upper,
                color=_CONDITION_COLORS[condition],
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(
            threshold,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="Alarm threshold" if axis is axes[0] else None,
        )
        axis.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_title(_DEFENDER_LABELS[defender])
        axis.set_xlabel("Turn relative to attack onset")
        axis.set_xlim(-20, 60)
        axis.set_ylim(-0.25, 1.5 * threshold + 0.25)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel(
        r"Detector evidence $\min(\log E_t,\,1.5\log(1/\alpha))$"
    )
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Endpoint-visible evidence at attack onset "
        "(median and seed IQR; display capped)",
        fontsize=13,
    )
    return _save_figure(fig, output / "detector_evidence_trajectories")


def _classical_quantum_comparison_figure(
    rows: Sequence[Mapping[str, Any]],
    bank: HistogramPayoffBank,
    output: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.2),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
        constrained_layout=True,
    )
    axis = axes[0]
    axis.set_axis_off()
    axis.set_title("(a) Matched observation and response boundary")
    _comparison_row(
        axis,
        y=0.68,
        title="Classical observation control",
        color="#6A3D9A",
        observations="route + accepted / rejected",
        evidence="same route-conditioned\nchange detector",
        response="freeze current\nrouting policy",
        boundary=(
            "No QBER/CHSH labels. This is a coarsened observation control,\n"
            "not a separately simulated classical transport network."
        ),
    )
    _comparison_row(
        axis,
        y=0.22,
        title="Quantum service observation",
        color="#0072B2",
        observations="route + delivery / QBER /\nCHSH / accepted",
        evidence="same route-conditioned\nchange detector",
        response="freeze current\nrouting policy",
        boundary=(
            "Quantum categories can distinguish integrity failures;\n"
            "active sentinel verification consumes entanglement requests."
        ),
    )
    axis.text(
        0.5,
        0.48,
        "same physical route/action trace",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )

    axis = axes[1]
    comparison_rows = [
        row
        for row in rows
        if row["condition"]["name"] == "detect_only"
    ]
    jitter = {
        "tsallis_inf_rv_eta1": -0.035,
        "linear_exp2_goptimal_eta2": 0.035,
    }
    for row in comparison_rows:
        defender = str(row["defender_label"])
        quantum = float(
            row["observer_comparison"][
                "quantum_four_outcome_detection_delay_turns"
            ]
        )
        classical = float(
            row["observer_comparison"][
                "classical_binary_service_detection_delay_turns"
            ]
        )
        xs = np.asarray([0.0, 1.0]) + jitter[defender]
        axis.plot(
            xs,
            [classical, quantum],
            color=_DEFENDER_COLORS[defender],
            alpha=0.45,
            linewidth=1.0,
        )
        axis.scatter(
            xs,
            [classical, quantum],
            color=_DEFENDER_COLORS[defender],
            marker="o" if defender.startswith("tsallis") else "^",
            s=42,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    for defender in _DEFENDER_LABELS:
        group = [
            row for row in comparison_rows
            if row["defender_label"] == defender
        ]
        classical = np.asarray([
            row["observer_comparison"][
                "classical_binary_service_detection_delay_turns"
            ]
            for row in group
        ], dtype=float)
        quantum = np.asarray([
            row["observer_comparison"][
                "quantum_four_outcome_detection_delay_turns"
            ]
            for row in group
        ], dtype=float)
        axis.plot(
            np.asarray([0.0, 1.0]) + jitter[defender],
            [np.mean(classical), np.mean(quantum)],
            color=_DEFENDER_COLORS[defender],
            linewidth=2.8,
        )
    axis.set_xticks([0.0, 1.0])
    axis.set_xticklabels([
        "Binary service\nobservation",
        "Four quantum\noutcomes",
    ])
    axis.set_ylabel("Detection delay after attack onset (turns)")
    axis.set_title("(b) Binary and four-way delays coincide here")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.5,
        6.86,
        "Clean false alarms: 0/6 in both observation regimes",
        ha="center",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=_DEFENDER_COLORS[name],
            marker="o" if name.startswith("tsallis") else "^",
            label=_DEFENDER_LABELS[name],
        )
        for name in _DEFENDER_LABELS
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )

    axis = axes[2]
    audit = bank.audit.alice_payoff_matrix()
    clean_ideal = float(
        np.max(audit[:, bank.action_ids.index("no_attack")])
    )
    freeze_regimes = ("quantum_freeze", "classical_freeze")
    freeze_labels = ("Four-outcome\nfreeze", "Binary-service\nfreeze")
    freeze_x = np.arange(2, dtype=float)
    freeze_jitter = {
        "tsallis_inf_rv_eta1": -0.045,
        "linear_exp2_goptimal_eta2": 0.045,
    }
    for defender in _DEFENDER_LABELS:
        values = []
        lower = []
        upper = []
        for regime in freeze_regimes:
            group = _damage_window_group(
                rows,
                (),
                defender=defender,
                regime=regime,
                window_turns=5_000,
            )
            losses = np.asarray([
                1.0 - entry["mean_reward"] / clean_ideal
                for entry in group
            ])
            center = float(np.median(losses))
            q1, q3 = np.quantile(losses, [0.25, 0.75])
            values.append(center)
            lower.append(center - float(q1))
            upper.append(float(q3) - center)
        axis.errorbar(
            freeze_x + freeze_jitter[defender],
            values,
            yerr=np.asarray([lower, upper]),
            marker="o" if defender.startswith("tsallis") else "^",
            linestyle="-",
            linewidth=1.1,
            capsize=3,
            color=_DEFENDER_COLORS[defender],
        )
    axis.set_xticks(freeze_x)
    axis.set_xticklabels(freeze_labels)
    axis.set_ylabel("Worst 5k-window loss of clean ideal")
    axis.set_title("(c) Worst 5k-window freeze loss")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.5,
        0.03,
        "Monitoring requests: none",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
    )
    fig.suptitle(
        "Matched classical-information and quantum-information detection",
        fontsize=13,
    )
    return _save_figure(fig, output / "classical_quantum_detection")


def _alice_vulnerability_figure(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    bank: HistogramPayoffBank,
    output: Path,
) -> list[Path]:
    regimes = (
        "clean",
        "colearning",
        "strict",
        "quantum_freeze",
        "classical_freeze",
        "restart",
    )
    labels = (
        "Clean",
        "Co-learned\nTsallis Eve",
        "Strict Eve",
        "Strict + freeze\n4-outcome",
        "Strict + freeze\nbinary",
        "Strict +\nrestart",
    )
    x = np.arange(len(regimes), dtype=float)
    audit = bank.audit.alice_payoff_matrix()
    no_attack = bank.action_ids.index("no_attack")
    clean_ideal = float(np.max(audit[:, no_attack]))
    strict_row = next(
        row for row in rows
        if row["condition"]["name"] == "detect_only"
    )
    commitment = np.asarray(
        strict_row["commitment"]["probabilities"], dtype=float
    )
    strict_ceiling = float(np.max(audit @ commitment))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.4, 4.1),
        constrained_layout=True,
    )
    offsets = {
        "tsallis_inf_rv_eta1": -0.14,
        "linear_exp2_goptimal_eta2": 0.14,
    }
    observed_by_defender: dict[str, np.ndarray] = {}
    target_by_defender: dict[str, np.ndarray] = {}
    for defender in _DEFENDER_LABELS:
        observed_groups = [
            _vulnerability_reward_group(
                rows,
                controls,
                defender=defender,
                regime=regime,
            )
            for regime in regimes
        ]
        target_groups = [
            _vulnerability_target_group(
                rows,
                controls,
                defender=defender,
                regime=regime,
            )
            for regime in regimes
        ]
        observed = np.asarray(
            [np.mean(group) / clean_ideal for group in observed_groups]
        )
        observed_by_defender[defender] = observed
        target = np.asarray([np.mean(group) for group in target_groups])
        target_by_defender[defender] = target
        lower = []
        upper = []
        target_lower = []
        target_upper = []
        for values, center in zip(observed_groups, observed):
            q1, q3 = np.quantile(
                np.asarray(values) / clean_ideal, [0.25, 0.75]
            )
            lower.append(center - q1)
            upper.append(q3 - center)
        for values, center in zip(target_groups, target):
            q1, q3 = np.quantile(values, [0.25, 0.75])
            target_lower.append(center - q1)
            target_upper.append(q3 - center)
        axes[0].errorbar(
            x + offsets[defender],
            observed,
            yerr=np.asarray([lower, upper]),
            marker="o" if defender.startswith("tsallis") else "^",
            linestyle="none",
            color=_DEFENDER_COLORS[defender],
            capsize=3,
            markersize=7,
            label=_DEFENDER_LABELS[defender],
        )
        axes[2].errorbar(
            x + offsets[defender],
            target,
            yerr=np.asarray([target_lower, target_upper]),
            marker="o" if defender.startswith("tsallis") else "^",
            linestyle="none",
            color=_DEFENDER_COLORS[defender],
            capsize=3,
            markersize=7,
        )

    strict_ratio = strict_ceiling / clean_ideal
    axes[0].scatter(
        x[[0, 2, 3, 4, 5]],
        [1.0, strict_ratio, strict_ratio, strict_ratio, strict_ratio],
        marker="D",
        facecolors="white",
        edgecolors="black",
        s=46,
        zorder=4,
        label="Full-information route ceiling",
    )
    axes[0].set_title("(a) Alice service retained")
    axes[0].set_ylabel("Post-onset reward / clean best-route reward")
    axes[0].set_ylim(0.5, 1.03)
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")

    for index, defender in enumerate(_DEFENDER_LABELS):
        observed = observed_by_defender[defender][2]
        retained = min(observed, strict_ratio)
        learning_gap = max(0.0, strict_ratio - observed)
        environment_loss = max(0.0, 1.0 - strict_ratio)
        axes[1].bar(
            index,
            retained,
            width=0.62,
            color=_DEFENDER_COLORS[defender],
            label="Retained service" if index == 0 else None,
        )
        axes[1].bar(
            index,
            learning_gap,
            bottom=retained,
            width=0.62,
            color="#E69F00",
            label="Alice adaptation gap" if index == 0 else None,
        )
        axes[1].bar(
            index,
            environment_loss,
            bottom=retained + learning_gap,
            width=0.62,
            color="#BDBDBD",
            label="Physical environment loss" if index == 0 else None,
        )
        axes[1].text(
            index,
            retained / 2,
            f"{100.0 * retained:.1f}% retained",
            ha="center",
            va="center",
            color="white",
            fontsize=7.5,
            rotation=90,
        )
        axes[1].text(
            index,
            strict_ratio + environment_loss / 2,
            f"{100.0 * environment_loss:.1f}% environment",
            ha="center",
            va="center",
            fontsize=7.2,
            rotation=90,
        )
        axes[1].annotate(
            f"{100.0 * learning_gap:.2f}% adaptation",
            xy=(index, retained + learning_gap / 2),
            xytext=(index, strict_ratio + 0.055),
            ha="center",
            va="center",
            fontsize=7.0,
            arrowprops={"arrowstyle": "-", "color": "#A05A00", "lw": 0.8},
        )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Tsallis-INF", "LinearExp2"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Fraction of clean best-route reward")
    axes[1].set_title("(b) Strict-Eve loss decomposition")
    axes[1].legend(
        frameon=False,
        fontsize=7.2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
    )

    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-5, 1.0)
    axes[2].set_ylabel("Final Stackelberg-target policy mass")
    axes[2].set_title("(c) Was Alice actually steered?")
    for axis in (axes[0], axes[2]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.24)
    fig.suptitle(
        "Alice vulnerability relative to clean service and ordinary "
        "Tsallis-Eve co-learning",
        fontsize=13,
    )
    return _save_figure(fig, output / "alice_vulnerability_comparison")


def _peak_damage_figure(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    bank: HistogramPayoffBank,
    output: Path,
) -> list[Path]:
    audit = bank.audit.alice_payoff_matrix()
    no_attack = bank.action_ids.index("no_attack")
    clean_ideal = float(np.max(audit[:, no_attack]))
    window_turns = tuple(
        map(
            int,
            rows[0]["experiment_config"]["damage_window_turns"],
        )
    )
    primary_window = (
        5_000 if 5_000 in window_turns else window_turns[len(window_turns) // 2]
    )
    window_colors = {
        window: color
        for window, color in zip(
            window_turns,
            ("#56B4E9", "#0072B2", "#D55E00", "#CC79A7"),
        )
    }
    offsets = np.linspace(-0.20, 0.20, len(window_turns))
    x = np.arange(len(_DAMAGE_REGIMES), dtype=float)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    for column, defender in enumerate(_DEFENDER_LABELS):
        damage_axis = axes[0, column]
        timing_axis = axes[1, column]
        for offset, window in zip(offsets, window_turns):
            groups = [
                _damage_window_group(
                    rows,
                    controls,
                    defender=defender,
                    regime=regime,
                    window_turns=window,
                )
                for regime in _DAMAGE_REGIMES
            ]
            peak_losses = [
                1.0
                - np.asarray(
                    [entry["mean_reward"] for entry in group],
                    dtype=float,
                )
                / clean_ideal
                for group in groups
            ]
            centers = np.asarray(
                [np.median(values) for values in peak_losses],
                dtype=float,
            )
            quantiles = np.asarray(
                [np.quantile(values, [0.25, 0.75]) for values in peak_losses],
                dtype=float,
            )
            damage_axis.errorbar(
                x + offset,
                centers,
                yerr=np.asarray([
                    centers - quantiles[:, 0],
                    quantiles[:, 1] - centers,
                ]),
                marker="o",
                linestyle="none",
                capsize=2.5,
                markersize=5.5,
                color=window_colors[window],
                label=f"{window // 1_000}k-turn window",
            )

        for regime_index, regime in enumerate(_DAMAGE_REGIMES):
            group = _damage_window_group(
                rows,
                controls,
                defender=defender,
                regime=regime,
                window_turns=primary_window,
            )
            jitter = np.linspace(-0.13, 0.13, len(group))
            for vertical_offset, entry in zip(jitter, group):
                y = regime_index + vertical_offset
                timing_axis.hlines(
                    y,
                    float(entry["window_start_turn"]) / 1_000.0,
                    float(entry["window_end_turn"]) / 1_000.0,
                    color=_DAMAGE_REGIME_COLORS[regime],
                    linewidth=3.0,
                    alpha=0.82,
                )
                timing_axis.plot(
                    float(entry["window_start_turn"]) / 1_000.0,
                    y,
                    marker="|",
                    color=_DAMAGE_REGIME_COLORS[regime],
                    markersize=8,
                )
        attack_start = int(rows[0]["experiment_config"]["attack_start_turn"])
        timing_axis.axvline(
            attack_start / 1_000.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Strict-attack onset",
        )

        damage_axis.set_title(_DEFENDER_LABELS[defender])
        damage_axis.set_xticks(x)
        damage_axis.set_xticklabels(
            [_DAMAGE_REGIME_LABELS[name] for name in _DAMAGE_REGIMES],
            rotation=14,
            ha="right",
            fontsize=8,
        )
        damage_axis.set_ylim(bottom=0.0)
        damage_axis.grid(axis="y", alpha=0.24)
        timing_axis.set_yticks(np.arange(len(_DAMAGE_REGIMES)))
        timing_axis.set_yticklabels([
            _DAMAGE_REGIME_LABELS[name].replace("\n", " ")
            for name in _DAMAGE_REGIMES
        ], fontsize=8)
        timing_axis.invert_yaxis()
        timing_axis.set_xlim(24.0, rows[0]["turns"] / 1_000.0 + 1.0)
        timing_axis.set_xlabel("Turn (thousands)")
        timing_axis.set_title(
            f"Location of each seed's worst {primary_window // 1_000}k window"
        )
        timing_axis.grid(axis="x", alpha=0.22)

    axes[0, 0].set_ylabel("Maximum application-service loss")
    axes[0, 1].legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Eve's maximum observed damage: exact sliding-window comparison\n"
        r"$D_w=1-\min_t\overline{R}_{t,w}/R^*_{\rm clean}$; "
        "window search restricted to turns 25,001–100,000",
        fontsize=14,
    )
    return _save_figure(fig, output / "eve_peak_damage_windows")


def _write_damage_window_csv(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    bank: HistogramPayoffBank,
    output: Path,
) -> Path:
    audit = bank.audit.alice_payoff_matrix()
    clean_ideal = float(
        np.max(audit[:, bank.action_ids.index("no_attack")])
    )
    records = []
    for defender in _DEFENDER_LABELS:
        for regime in _DAMAGE_REGIMES:
            window_values = (
                rows[0]["experiment_config"]["damage_window_turns"]
            )
            for window_turns in window_values:
                for entry in _damage_window_group(
                    rows,
                    controls,
                    defender=defender,
                    regime=regime,
                    window_turns=int(window_turns),
                ):
                    records.append({
                        "defender_label": defender,
                        "regime": regime,
                        "regime_label": _DAMAGE_REGIME_LABELS[regime].replace(
                            "\n", " "
                        ),
                        "seed": int(entry["seed"]),
                        "window_turns": int(entry["window_turns"]),
                        "window_start_turn": int(
                            entry["window_start_turn"]
                        ),
                        "window_end_turn": int(entry["window_end_turn"]),
                        "minimum_mean_application_reward": float(
                            entry["mean_reward"]
                        ),
                        "clean_audit_best_route_reward": clean_ideal,
                        "maximum_loss_of_clean_ideal": (
                            1.0
                            - float(entry["mean_reward"]) / clean_ideal
                        ),
                    })
    return _write_rows_csv(
        output / "peak_damage_windows.csv",
        records,
        (
            "defender_label",
            "regime",
            "regime_label",
            "seed",
            "window_turns",
            "window_start_turn",
            "window_end_turn",
            "minimum_mean_application_reward",
            "clean_audit_best_route_reward",
            "maximum_loss_of_clean_ideal",
        ),
    )


def _damage_window_group(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    defender: str,
    regime: str,
    window_turns: int,
) -> list[dict[str, Any]]:
    if regime == "colearning":
        source = [
            row for row in controls
            if row["defender_label"] == defender
        ]
        field = "alice_reward_window_minima"
    else:
        condition = {
            "clean": "clean_control",
            "strict": "detect_only",
            "quantum_freeze": "quantum_detect_freeze_policy",
            "classical_freeze": "classical_binary_detect_freeze_policy",
            "restart": "adaptive_sentinel_02_20_restart",
        }[regime]
        source = _select(rows, defender=defender, condition=condition)
        field = "application_reward_window_minima"
    entries = []
    for row in source:
        matches = [
            entry for entry in row[field]
            if int(entry["window_turns"]) == int(window_turns)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{regime} has incomplete {window_turns}-turn damage telemetry"
            )
        entries.append({
            **matches[0],
            "seed": int(row["seed"]),
        })
    if not entries:
        raise ValueError(f"{regime} damage group is empty")
    return sorted(entries, key=lambda entry: entry["seed"])


def _comparison_row(
    axis: plt.Axes,
    *,
    y: float,
    title: str,
    color: str,
    observations: str,
    evidence: str,
    response: str,
    boundary: str,
) -> None:
    axis.text(
        0.01,
        y + 0.18,
        title,
        color=color,
        fontsize=10,
        fontweight="bold",
        transform=axis.transAxes,
    )
    entries = (
        (0.02, observations),
        (0.36, evidence),
        (0.70, response),
    )
    for x, text in entries:
        box = patches.FancyBboxPatch(
            (x, y),
            0.27,
            0.14,
            boxstyle="round,pad=0.015",
            facecolor=color,
            edgecolor="none",
            alpha=0.14,
            transform=axis.transAxes,
        )
        axis.add_patch(box)
        axis.text(
            x + 0.135,
            y + 0.07,
            text,
            ha="center",
            va="center",
            fontsize=8.2,
            transform=axis.transAxes,
        )
    for start in (0.29, 0.63):
        axis.annotate(
            "",
            xy=(start + 0.055, y + 0.07),
            xytext=(start, y + 0.07),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4},
            xycoords=axis.transAxes,
        )
    axis.text(
        0.02,
        y - 0.055,
        boundary,
        fontsize=7.4,
        color="#444444",
        transform=axis.transAxes,
    )


def _vulnerability_reward_group(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    defender: str,
    regime: str,
) -> np.ndarray:
    condition = {
        "clean": "clean_control",
        "strict": "detect_only",
        "quantum_freeze": "quantum_detect_freeze_policy",
        "classical_freeze": "classical_binary_detect_freeze_policy",
        "restart": "adaptive_sentinel_02_20_restart",
    }.get(regime)
    if regime == "colearning":
        return np.asarray([
            row["mean_last_75k_alice_reward"]
            for row in controls
            if row["defender_label"] == defender
        ], dtype=float)
    return np.asarray([
        row["mean_attack_window_service_reward"]
        for row in _select(rows, defender=defender, condition=str(condition))
    ], dtype=float)


def _vulnerability_target_group(
    rows: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    defender: str,
    regime: str,
) -> np.ndarray:
    condition = {
        "clean": "clean_control",
        "strict": "detect_only",
        "quantum_freeze": "quantum_detect_freeze_policy",
        "classical_freeze": "classical_binary_detect_freeze_policy",
        "restart": "adaptive_sentinel_02_20_restart",
    }.get(regime)
    if regime == "colearning":
        return np.asarray([
            row["intended_route_current_policy_mass"]
            for row in controls
            if row["defender_label"] == defender
        ], dtype=float)
    return np.asarray([
        row["final_target_mass"][
            "intended_route_current_service_policy_mass"
        ]
        for row in _select(rows, defender=defender, condition=str(condition))
    ], dtype=float)


def _trajectory_matrix(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("trajectory group is empty")
    turns = np.asarray(
        [row["turn"] for row in rows[0]["trajectory"]],
        dtype=int,
    )
    matrix = []
    for row in rows:
        row_turns = np.asarray(
            [point["turn"] for point in row["trajectory"]],
            dtype=int,
        )
        if not np.array_equal(row_turns, turns):
            raise ValueError("trajectory checkpoints are not aligned")
        matrix.append([
            float(point[field]) for point in row["trajectory"]
        ])
    return turns, np.asarray(matrix, dtype=float)


def _transition_matrix(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("detector-transition group is empty")
    turns = np.asarray(
        [
            row["turn_relative_to_attack"]
            for row in rows[0]["detector_transition"]
        ],
        dtype=int,
    )
    matrix = []
    for row in rows:
        row_turns = np.asarray(
            [
                point["turn_relative_to_attack"]
                for point in row["detector_transition"]
            ],
            dtype=int,
        )
        if not np.array_equal(row_turns, turns):
            raise ValueError("detector-transition checkpoints are not aligned")
        matrix.append([
            float(point[field]) for point in row["detector_transition"]
        ])
    return turns, np.asarray(matrix, dtype=float)


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    defender: str,
    condition: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row["defender_label"]) == defender
        and str(row["condition"]["name"]) == condition
    ]


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    written = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return path
