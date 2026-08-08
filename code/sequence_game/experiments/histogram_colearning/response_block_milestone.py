"""Coupled finite-block and Tsallis alarm-response calibration contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .endpoint_response import (
    DETECTOR_CLASSICAL_SERVICE,
    DETECTOR_QUANTUM_ENDPOINT,
)
from .endpoint_response_calibration import (
    SequentialResponseCalibrationConfig,
    SequentialResponseCandidate,
)
from .paired_block_evidence import (
    PAIRED_BLOCK_DIAGNOSTIC_SIZES,
    PAIRED_BLOCK_PRIMARY_SIZES,
)
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
)
from .stackelberg_controller import ControllerPrimeExploitSchedule


RESPONSE_BLOCK_MILESTONE_SCHEMA = (
    "histogram_tsallis_response_block_milestone.v1"
)
RESPONSE_BLOCK_FREEZE_SCHEMA = (
    "histogram_tsallis_response_block_freeze.v1"
)
RESPONSE_BLOCK_SELECTION_SCHEMA = (
    "histogram_tsallis_response_block_selection.v1"
)


@dataclass(frozen=True)
class TsallisResponseBlockMilestoneConfig:
    """Frozen execution and information boundary for the staged milestone."""

    diagnostic_block_sizes: tuple[int, ...] = (
        PAIRED_BLOCK_DIAGNOSTIC_SIZES
    )
    primary_block_sizes: tuple[int, ...] = PAIRED_BLOCK_PRIMARY_SIZES
    endpoint_samples_per_cell_per_split: int = 96
    endpoint_root_seed: int = 20_260_720
    thinning_root_seed: int = 20_260_721
    maximum_follow_turns: int = 100
    follow_posterior_threshold: float = 0.99
    execution_seeds: tuple[int, ...] = (
        2_026_071_901,
        2_026_071_902,
        2_026_071_903,
    )
    prime_turns: int = 3_000
    exploit_turns: int = 1_000
    record_every: int = 100
    commissioning_draws_per_route: int = 32
    feedback_calibration_draw_start: int = 32
    feedback_calibration_draw_stop: int = 96
    audit_draw_start: int = 0
    audit_draw_stop: int = 96
    response_workers: int = 3
    detector_false_alarm_level: float = 0.01
    detector_commissioning_pseudocount: float = 0.5
    detector_change_fractions: tuple[float, ...] = (
        0.10,
        0.25,
        0.50,
    )
    active_detectors: tuple[str, ...] = (
        DETECTOR_CLASSICAL_SERVICE,
        DETECTOR_QUANTUM_ENDPOINT,
    )
    response_calibration: SequentialResponseCalibrationConfig = field(
        default_factory=SequentialResponseCalibrationConfig
    )

    def __post_init__(self) -> None:
        diagnostic = tuple(map(int, self.diagnostic_block_sizes))
        primary = tuple(map(int, self.primary_block_sizes))
        if (
            not diagnostic
            or not primary
            or min(*diagnostic, *primary) < 1
            or set(diagnostic) & set(primary)
        ):
            raise ValueError(
                "diagnostic and primary block sizes must be positive and "
                "disjoint"
            )
        if max(*diagnostic, *primary) != 1_200:
            raise ValueError("the paired source block must remain 1,200")
        if self.endpoint_samples_per_cell_per_split < 2:
            raise ValueError("endpoint bank needs at least two draws")
        if (
            self.commissioning_draws_per_route < 2
            or self.feedback_calibration_draw_start
            < self.commissioning_draws_per_route
            or self.feedback_calibration_draw_stop
            > self.endpoint_samples_per_cell_per_split
            or self.feedback_calibration_draw_stop
            <= self.feedback_calibration_draw_start
            or self.audit_draw_start < 0
            or self.audit_draw_stop
            > self.endpoint_samples_per_cell_per_split
            or self.audit_draw_stop <= self.audit_draw_start
        ):
            raise ValueError("endpoint evidence draw partitions are invalid")
        if (
            not self.execution_seeds
            or len(set(map(int, self.execution_seeds)))
            != len(self.execution_seeds)
        ):
            raise ValueError("execution seeds must be nonempty and unique")
        if self.maximum_follow_turns < 1:
            raise ValueError("maximum_follow_turns must be positive")
        if not 0.5 < self.follow_posterior_threshold < 1.0:
            raise ValueError(
                "follow_posterior_threshold must lie in (0.5, 1)"
            )
        if self.prime_turns < 1 or self.exploit_turns < 1:
            raise ValueError("Prime and Exploit turns must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if self.response_workers < 1:
            raise ValueError("response_workers must be positive")
        if tuple(self.active_detectors) != tuple(
            self.response_calibration.active_detectors
        ):
            raise ValueError(
                "milestone and response-calibration detectors differ"
            )
        self.detector_config

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(sorted({
            *map(int, self.diagnostic_block_sizes),
            *map(int, self.primary_block_sizes),
        }))

    @property
    def maximum_block_size(self) -> int:
        return max(self.block_sizes)

    @property
    def schedule(self) -> ControllerPrimeExploitSchedule:
        return ControllerPrimeExploitSchedule(
            prime_turns=self.prime_turns,
            exploit_turns=self.exploit_turns,
            record_every=self.record_every,
        )

    @property
    def detector_config(self) -> BinomialEvidenceDetectorConfig:
        return BinomialEvidenceDetectorConfig(
            horizon=self.schedule.total_turns,
            model_false_alarm_level=self.detector_false_alarm_level,
            commissioning_pseudocount=(
                self.detector_commissioning_pseudocount
            ),
            change_fractions=self.detector_change_fractions,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "diagnostic_block_sizes",
            "primary_block_sizes",
            "execution_seeds",
            "detector_change_fractions",
            "active_detectors",
        ):
            payload[key] = list(payload[key])
        payload["block_sizes"] = list(self.block_sizes)
        payload["schedule"] = self.schedule.to_dict()
        payload["detector_config"] = self.detector_config.to_dict()
        payload["response_calibration"] = (
            self.response_calibration.to_dict()
        )
        return payload


def select_feedback_response_profiles(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[SequentialResponseCandidate],
    *,
    primary_block_sizes: Sequence[int] = PAIRED_BLOCK_PRIMARY_SIZES,
) -> dict[str, Any]:
    """Select per-block and joint-primary choices from feedback rows only."""

    values = tuple(dict(row) for row in rows)
    if not values:
        raise ValueError("feedback calibration rows are empty")
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    if not candidate_by_id:
        raise ValueError("response candidate set is empty")
    detectors = sorted({str(row["active_detector"]) for row in values})
    block_sizes = sorted({int(row["block_size"]) for row in values})
    per_block: list[dict[str, Any]] = []
    selected_by_pair: dict[
        tuple[str, int],
        tuple[SequentialResponseCandidate, dict[str, Any]],
    ] = {}

    for detector in detectors:
        for block_size in block_sizes:
            selected_rows = [
                row for row in values
                if str(row["active_detector"]) == detector
                and int(row["block_size"]) == block_size
            ]
            summaries = _candidate_summaries(
                selected_rows,
                candidate_by_id,
            )
            selection = _one_standard_error_selection(summaries)
            candidate = candidate_by_id[
                str(selection["selected"]["candidate_id"])
            ]
            selected_by_pair[(detector, block_size)] = (
                candidate,
                selection,
            )
            per_block.append({
                "active_detector": detector,
                "block_size": block_size,
                "block_role": (
                    "primary"
                    if block_size in set(map(int, primary_block_sizes))
                    else "underpowered_diagnostic"
                ),
                "selected_candidate": candidate.to_dict(),
                **selection,
            })

    joint_primary: list[dict[str, Any]] = []
    primary = set(map(int, primary_block_sizes))
    for detector in detectors:
        summaries: list[dict[str, Any]] = []
        for block_size in sorted(primary):
            selected_rows = [
                row for row in values
                if str(row["active_detector"]) == detector
                and int(row["block_size"]) == block_size
            ]
            if not selected_rows:
                raise ValueError(
                    f"missing primary block {block_size} for {detector}"
                )
            summaries.extend(
                _candidate_summaries(
                    selected_rows,
                    candidate_by_id,
                    block_size=block_size,
                )
            )
        selection = _one_standard_error_selection(
            summaries,
            include_block_size=True,
        )
        selected_id = str(selection["selected"]["candidate_id"])
        selected_size = int(selection["selected"]["block_size"])
        joint_primary.append({
            "active_detector": detector,
            "selected_block_size": selected_size,
            "selected_candidate": candidate_by_id[
                selected_id
            ].to_dict(),
            **selection,
        })

    return {
        "schema": RESPONSE_BLOCK_SELECTION_SCHEMA,
        "selection_data_split": "feedback_only",
        "per_block": per_block,
        "joint_primary": joint_primary,
        "diagnostic_block_sizes_excluded_from_joint_selection": (
            sorted(set(block_sizes) - primary)
        ),
        "selection_rule": {
            "aggregation": (
                "equal-weight topology macro mean after averaging seeds "
                "within each topology"
            ),
            "uncertainty": (
                "one standard error over topology-level means around the "
                "raw maximum"
            ),
            "candidate_tie_break": (
                "smallest containment ceiling, smallest TV radius, longest "
                "arming horizon, longest posterior window, longest refresh"
            ),
            "joint_block_tie_break": (
                "candidate conservative order followed by fewer requested "
                "pairs per observation block"
            ),
        },
    }


def response_result_row(
    response: Mapping[str, Any],
    *,
    partition: str,
    case_id: str,
    family: str,
    block_size: int,
    active_detector: str,
    candidate: SequentialResponseCandidate,
    execution_role: str,
) -> dict[str, Any]:
    """Reduce one response run and report alarm latency in turns and pairs."""

    alarm_turn = response.get("response_alarm_turn")
    effective_turn = response.get("response_effective_turn")
    control = response["response_control"]
    return {
        "partition": str(partition),
        "case_id": str(case_id),
        "family": str(family),
        "block_size": int(block_size),
        "block_role": (
            "primary"
            if int(block_size) in PAIRED_BLOCK_PRIMARY_SIZES
            else "underpowered_diagnostic"
        ),
        "active_detector": str(active_detector),
        "seed": int(response["seed"]),
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "execution_role": str(execution_role),
        "execution_split": str(response["execution_split"]),
        "alarm_turn": alarm_turn,
        "alarm_requested_pairs": (
            int(alarm_turn) * int(block_size)
            if alarm_turn is not None
            else None
        ),
        "response_effective_turn": effective_turn,
        "response_effective_requested_pairs": (
            int(effective_turn) * int(block_size)
            if effective_turn is not None
            else None
        ),
        "mean_alice_reward": float(response["mean_alice_reward"]),
        "mean_response_window_reward": response[
            "mean_response_window_reward"
        ],
        "mean_prime_reward": float(
            response["phase_metrics"]["prime"]["mean_alice_reward"]
        ),
        "mean_exploit_reward": float(
            response["phase_metrics"]["exploit"]["mean_alice_reward"]
        ),
        "response_window_turns": int(response["response_window_turns"]),
        "behavior_mixture_turns": int(
            control["behavior_mixture_turns"]
        ),
        "learner_controlled_turns": int(
            control["learner_controlled_turns"]
        ),
        "containment_controlled_turns": int(
            control["containment_controlled_turns"]
        ),
        "mean_containment_probability": float(
            control["mean_containment_probability"]
        ),
        "final_containment_probability": float(
            control["final_containment_probability"]
        ),
        "response_policy_refreshes": int(
            control["response_policy_refreshes"]
        ),
        "final_behavior_policy_statistics": dict(
            response["final_behavior_policy_statistics"]
        ),
        "elapsed_seconds": float(response["elapsed_seconds"]),
    }


def freeze_digest(payload: Mapping[str, Any]) -> str:
    """Hash a freeze manifest without relying on filesystem timestamps."""

    content = {
        key: value
        for key, value in dict(payload).items()
        if key != "freeze_digest_blake2b_128"
    }
    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def validate_freeze_manifest(payload: Mapping[str, Any]) -> None:
    """Reject evaluation unless the calibration choices are immutable."""

    if payload.get("schema") != RESPONSE_BLOCK_FREEZE_SCHEMA:
        raise ValueError("unexpected response-block freeze schema")
    expected = str(payload.get("freeze_digest_blake2b_128") or "")
    if not expected or freeze_digest(payload) != expected:
        raise ValueError("response-block freeze digest mismatch")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("freeze manifest has no feedback selection")
    if selection.get("selection_data_split") != "feedback_only":
        raise ValueError("freeze selection was not feedback-only")
    if bool(payload.get("evaluation_partition_loaded_before_freeze")):
        raise ValueError("evaluation partition was loaded before freeze")


def _candidate_summaries(
    rows: Sequence[Mapping[str, Any]],
    candidate_by_id: Mapping[str, SequentialResponseCandidate],
    *,
    block_size: int | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    ids = sorted({str(row["candidate_id"]) for row in rows})
    if set(ids) - set(candidate_by_id):
        raise ValueError("feedback rows contain an unknown candidate")
    for candidate_id in ids:
        selected = [
            row for row in rows
            if str(row["candidate_id"]) == candidate_id
        ]
        case_means = []
        for case_id in sorted({str(row["case_id"]) for row in selected}):
            rewards = [
                float(row["mean_alice_reward"])
                for row in selected
                if str(row["case_id"]) == case_id
            ]
            case_means.append(float(np.mean(rewards)))
        values = np.asarray(case_means, dtype=float)
        summary = {
            "candidate_id": candidate_id,
            "candidate": candidate_by_id[candidate_id].to_dict(),
            "case_count": int(values.size),
            "run_count": len(selected),
            "topology_macro_mean_alice_reward": float(np.mean(values)),
            "topology_standard_error": (
                float(np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else 0.0
            ),
            "topology_mean_rewards": list(map(float, values)),
        }
        if block_size is not None:
            summary["block_size"] = int(block_size)
        summaries.append(summary)
    return summaries


def _one_standard_error_selection(
    summaries: Sequence[Mapping[str, Any]],
    *,
    include_block_size: bool = False,
) -> dict[str, Any]:
    values = tuple(dict(row) for row in summaries)
    if not values:
        raise ValueError("selection summaries are empty")
    raw = max(
        values,
        key=lambda row: (
            float(row["topology_macro_mean_alice_reward"]),
            str(row["candidate_id"]),
            -int(row.get("block_size", 0)),
        ),
    )
    threshold = (
        float(raw["topology_macro_mean_alice_reward"])
        - float(raw["topology_standard_error"])
    )
    eligible = [
        row for row in values
        if float(row["topology_macro_mean_alice_reward"])
        >= threshold
    ]

    def preference(row: Mapping[str, Any]) -> tuple[Any, ...]:
        candidate = SequentialResponseCandidate(**{
            key: row["candidate"][key]
            for key in (
                "maximum_total_variation",
                "arming_turns",
                "containment_ceiling",
                "posterior_window_turns",
                "posterior_refresh_turns",
            )
        })
        base = (
            candidate.containment_ceiling,
            candidate.maximum_total_variation,
            -candidate.arming_turns,
            -candidate.posterior_window_turns,
            -candidate.posterior_refresh_turns,
        )
        if include_block_size:
            return (*base, int(row["block_size"]), candidate.candidate_id)
        return (*base, candidate.candidate_id)

    selected = min(eligible, key=preference)
    return {
        "raw_best": raw,
        "selected": selected,
        "one_standard_error_threshold": threshold,
        "eligible": sorted(
            eligible,
            key=lambda row: (
                int(row.get("block_size", 0)),
                str(row["candidate_id"]),
            ),
        ),
        "candidate_summaries": sorted(
            values,
            key=lambda row: (
                int(row.get("block_size", 0)),
                str(row["candidate_id"]),
            ),
        ),
    }
