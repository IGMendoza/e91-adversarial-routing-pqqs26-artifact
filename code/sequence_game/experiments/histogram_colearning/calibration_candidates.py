"""Declared two-player learner grids and uncertainty-aware selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Literal, Sequence

import numpy as np

from .config import (
    ALICE_LEARNER_EXP3_PLUS_PLUS,
    ALICE_LEARNER_FLAT_EXP3,
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_TSALLIS_INF,
    EVE_LEARNER_EXP3_PLUS_PLUS,
    EVE_LEARNER_FLAT_EXP3,
    EVE_LEARNER_LINEAR_ACTION_EXP2,
    EVE_LEARNER_TSALLIS_INF,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
    HistogramCoLearningConfig,
)


Player = Literal["alice", "eve"]
LEARNER_FAMILIES = ("flat_exp3", "tsallis_inf", "exp3_plus_plus", "linear_exp2")
TSALLIS_REFINEMENT_ETA_SCALES = tuple(
    2.0 ** (half_octave / 2.0) for half_octave in range(-2, 7)
)


@dataclass(frozen=True)
class LearnerCalibrationCandidate:
    """One predeclared role-specific learner setting."""

    player: Player
    family: str
    label: str
    config_overrides: dict[str, Any]
    literature_default: bool
    default_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "family": self.family,
            "label": self.label,
            "config_overrides": dict(self.config_overrides),
            "literature_default": self.literature_default,
            "default_distance": self.default_distance,
        }


def declared_calibration_candidates(
    player: Player,
) -> tuple[LearnerCalibrationCandidate, ...]:
    """Return the complete finite-action grid for one player."""

    if player not in {"alice", "eve"}:
        raise ValueError(f"unsupported player {player!r}")
    mode_field = f"{player}_learner_mode"
    tsallis_estimator_field = f"{player}_tsallis_inf_estimator"
    tsallis_eta_field = f"{player}_tsallis_inf_eta_scale"
    exp3pp_beta_field = f"{player}_exp3pp_exploration_beta"
    exp3pp_eta_field = f"{player}_exp3pp_eta_scale"
    linear_eta_field = f"{player}_exp3_eta_c"
    structured_exploration_field = (
        "structured_exploration_mode"
        if player == "alice" else "eve_structured_exploration_mode"
    )
    modes = {
        "flat_exp3": (
            ALICE_LEARNER_FLAT_EXP3
            if player == "alice" else EVE_LEARNER_FLAT_EXP3
        ),
        "tsallis_inf": (
            ALICE_LEARNER_TSALLIS_INF
            if player == "alice" else EVE_LEARNER_TSALLIS_INF
        ),
        "exp3_plus_plus": (
            ALICE_LEARNER_EXP3_PLUS_PLUS
            if player == "alice" else EVE_LEARNER_EXP3_PLUS_PLUS
        ),
        "linear_exp2": (
            ALICE_LEARNER_LINEAR_ROUTE_EXP2
            if player == "alice" else EVE_LEARNER_LINEAR_ACTION_EXP2
        ),
    }
    rows = [LearnerCalibrationCandidate(
        player=player,
        family="flat_exp3",
        label="flat_exp3_frozen",
        config_overrides={mode_field: modes["flat_exp3"]},
        literature_default=True,
        default_distance=0.0,
    )]
    for estimator in ("rv", "iw"):
        for eta_scale in (0.5, 1.0, 2.0):
            rows.append(LearnerCalibrationCandidate(
                player=player,
                family="tsallis_inf",
                label=f"tsallis_{estimator}_eta{eta_scale:g}",
                config_overrides={
                    mode_field: modes["tsallis_inf"],
                    tsallis_estimator_field: estimator,
                    tsallis_eta_field: eta_scale,
                },
                literature_default=estimator == "rv" and eta_scale == 1.0,
                default_distance=(
                    abs(math.log2(eta_scale)) + (0.0 if estimator == "rv" else 1.0)
                ),
            ))
    for exploration_beta in (64.0, 256.0, 1024.0):
        eta_scales = (
            (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
            if exploration_beta == 256.0
            else (0.5, 1.0, 2.0, 4.0, 8.0)
        )
        for eta_scale in eta_scales:
            rows.append(LearnerCalibrationCandidate(
                player=player,
                family="exp3_plus_plus",
                label=f"exp3pp_beta{exploration_beta:g}_eta{eta_scale:g}",
                config_overrides={
                    mode_field: modes["exp3_plus_plus"],
                    exp3pp_beta_field: exploration_beta,
                    exp3pp_eta_field: eta_scale,
                },
                literature_default=(
                    exploration_beta == 256.0 and eta_scale == 1.0
                ),
                default_distance=(
                    abs(math.log2(exploration_beta / 256.0))
                    + abs(math.log2(eta_scale))
                ),
            ))
    for eta_scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        rows.append(LearnerCalibrationCandidate(
            player=player,
            family="linear_exp2",
            label=f"linear_goptimal_eta{eta_scale:g}",
            config_overrides={
                mode_field: modes["linear_exp2"],
                linear_eta_field: eta_scale,
                structured_exploration_field: STRUCTURED_EXPLORATION_G_OPTIMAL,
            },
            literature_default=eta_scale == 1.0,
            default_distance=abs(math.log2(eta_scale)),
        ))
    return tuple(rows)


def default_anchor_candidates(
    player: Player,
) -> tuple[LearnerCalibrationCandidate, ...]:
    """One frozen/default opponent setting from each learner family."""

    candidates = declared_calibration_candidates(player)
    rows = tuple(row for row in candidates if row.literature_default)
    if {row.family for row in rows} != set(LEARNER_FAMILIES):
        raise RuntimeError("every learner family must have one default anchor")
    return rows


def tsallis_refinement_candidates(
    player: Player,
) -> tuple[LearnerCalibrationCandidate, ...]:
    """Return the dense two-estimator Tsallis-INF calibration grid.

    The eta multiplier is sampled every half octave from 0.5 through 8.0.
    This retains the literature RV/eta-1 setting, adds resolution around it,
    and extends beyond the previous IW/eta-2 calibration boundary.
    """

    if player not in {"alice", "eve"}:
        raise ValueError(f"unsupported player {player!r}")
    mode_field = f"{player}_learner_mode"
    estimator_field = f"{player}_tsallis_inf_estimator"
    eta_field = f"{player}_tsallis_inf_eta_scale"
    mode = (
        ALICE_LEARNER_TSALLIS_INF
        if player == "alice" else EVE_LEARNER_TSALLIS_INF
    )
    rows = []
    for estimator in ("rv", "iw"):
        for eta_scale in TSALLIS_REFINEMENT_ETA_SCALES:
            eta_label = f"{eta_scale:.6g}"
            rows.append(LearnerCalibrationCandidate(
                player=player,
                family="tsallis_inf",
                label=f"tsallis_{estimator}_eta{eta_label}",
                config_overrides={
                    mode_field: mode,
                    estimator_field: estimator,
                    eta_field: eta_scale,
                },
                literature_default=estimator == "rv" and eta_scale == 1.0,
                default_distance=(
                    abs(math.log2(eta_scale))
                    + (0.0 if estimator == "rv" else 1.0)
                ),
            ))
    return tuple(rows)


def apply_candidate_pair(
    base: HistogramCoLearningConfig,
    *,
    alice: LearnerCalibrationCandidate,
    eve: LearnerCalibrationCandidate,
    turns: int,
    seed: int,
) -> HistogramCoLearningConfig:
    """Create one valid joint config without altering unrelated physics fields."""

    if alice.player != "alice" or eve.player != "eve":
        raise ValueError("candidate pair has incorrect player roles")
    overlap = set(alice.config_overrides) & set(eve.config_overrides)
    if overlap:
        raise ValueError(f"candidate role overrides overlap: {sorted(overlap)}")
    return replace(
        base,
        turns=int(turns),
        seed=int(seed),
        **alice.config_overrides,
        **eve.config_overrides,
    )


def select_candidate_by_external_regret(
    candidates: Sequence[LearnerCalibrationCandidate],
    rows: Sequence[dict[str, Any]],
    *,
    player: Player,
) -> dict[str, Any]:
    """Select each family with a seed-block one-standard-error rule."""

    declared = tuple(candidates)
    if not declared or any(row.player != player for row in declared):
        raise ValueError("candidate declarations do not match the selected player")
    candidate_by_label = {row.label: row for row in declared}
    if len(candidate_by_label) != len(declared):
        raise ValueError("candidate labels are not unique")
    result_rows = tuple(rows)
    if not result_rows:
        raise ValueError("calibration rows are empty")
    unknown = {
        str(row["candidate_label"]) for row in result_rows
    } - set(candidate_by_label)
    if unknown:
        raise ValueError(f"results contain unknown candidates: {sorted(unknown)}")
    selected = {}
    summaries = []
    metric = f"{player}_external_regret_per_turn"
    for candidate in declared:
        observations = [
            row for row in result_rows
            if row["candidate_label"] == candidate.label
        ]
        if not observations:
            raise ValueError(f"candidate {candidate.label!r} has no results")
        seeds = sorted({int(row["seed"]) for row in observations})
        seed_means = [
            float(np.mean([
                float(row[metric]) for row in observations
                if int(row["seed"]) == seed
            ]))
            for seed in seeds
        ]
        case_means = {
            str(case_id): float(np.mean([
                float(row[metric]) for row in observations
                if str(row["case_id"]) == str(case_id)
            ]))
            for case_id in sorted({str(row["case_id"]) for row in observations})
        }
        summaries.append({
            "candidate": candidate.to_dict(),
            "observation_count": len(observations),
            "seed_block_means": seed_means,
            "mean_external_regret_per_turn": float(np.mean(seed_means)),
            "standard_error": (
                float(np.std(seed_means, ddof=1) / math.sqrt(len(seed_means)))
                if len(seed_means) > 1 else 0.0
            ),
            "case_mean_external_regret_per_turn": case_means,
            "worst_case_mean_external_regret_per_turn": max(case_means.values()),
            "mean_exploitability": float(np.mean([
                float(row["exploitability"]) for row in observations
            ])),
            "mean_equilibrium_set_total_variation": float(np.mean([
                float(row[f"{player}_equilibrium_set_total_variation"])
                for row in observations
                if row.get(f"{player}_equilibrium_set_total_variation") is not None
            ])) if any(
                row.get(f"{player}_equilibrium_set_total_variation") is not None
                for row in observations
            ) else None,
        })
    for family in LEARNER_FAMILIES:
        family_rows = [
            row for row in summaries
            if row["candidate"]["family"] == family
        ]
        raw = min(
            family_rows,
            key=lambda row: (
                row["mean_external_regret_per_turn"],
                row["worst_case_mean_external_regret_per_turn"],
                row["candidate"]["label"],
            ),
        )
        threshold = raw["mean_external_regret_per_turn"] + raw["standard_error"]
        eligible = [
            row for row in family_rows
            if row["mean_external_regret_per_turn"] <= threshold
        ]
        winner = min(
            eligible,
            key=lambda row: (
                row["candidate"]["default_distance"],
                row["mean_external_regret_per_turn"],
                row["worst_case_mean_external_regret_per_turn"],
                row["candidate"]["label"],
            ),
        )
        selected[family] = {
            "selected": winner,
            "raw_minimum": raw,
            "one_standard_error_threshold": threshold,
            "eligible_labels": [row["candidate"]["label"] for row in eligible],
        }
    return {
        "player": player,
        "selection_metric": metric,
        "seed_block_definition": (
            "for each seed, macro-average equally across calibration cases and "
            "the four default opponent-family anchors"
        ),
        "selection_rule": (
            "within each family, find the minimum mean own-player external "
            "regret; admit settings within one seed-block standard error of "
            "that minimum; choose the admitted setting with smallest declared "
            "distance from the literature/default schedule"
        ),
        "candidate_summaries": summaries,
        "selected_by_family": selected,
    }


def selected_candidates(
    selection: dict[str, Any],
) -> tuple[LearnerCalibrationCandidate, ...]:
    """Reconstruct the four frozen family settings from a selection artifact."""

    rows = []
    for family in LEARNER_FAMILIES:
        payload = selection["selected_by_family"][family]["selected"]["candidate"]
        rows.append(LearnerCalibrationCandidate(
            player=payload["player"],
            family=payload["family"],
            label=payload["label"],
            config_overrides=dict(payload["config_overrides"]),
            literature_default=bool(payload["literature_default"]),
            default_distance=float(payload["default_distance"]),
        ))
    return tuple(rows)
