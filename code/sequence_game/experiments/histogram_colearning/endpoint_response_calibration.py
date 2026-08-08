"""Feedback-only calibration of sequential alarm-response controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import HistogramEndpointEvidenceBank
from .endpoint_response import (
    ACTIVE_RESPONSE_DETECTORS,
    DETECTOR_CLASSICAL_SERVICE,
    DETECTOR_QUANTUM_ENDPOINT,
    RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
    RESPONSE_UPDATE_MODES,
    RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW,
    AlarmResponseConfig,
    discover_alarm_response_record,
    run_alarm_conditioned_response_continuation,
)
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
    MatchedClassicalQuantumEndpointDetectors,
)
from .stackelberg_bandit import (
    RouteObservationDesign,
    StackelbergBanditFollowRuntime,
    run_stackelberg_bandit_follow_runtime,
)
from .stackelberg_controller import ControllerPrimeExploitSchedule
from .stackelberg_controller_plans import ControllerPrimeExploitPlan


SEQUENTIAL_RESPONSE_CALIBRATION_SCHEMA = (
    "histogram_sequential_endpoint_response_calibration.v1"
)


@dataclass(frozen=True)
class SequentialResponseCandidate:
    """One declared sequential response parameter tuple."""

    maximum_total_variation: float
    arming_turns: int
    containment_ceiling: float
    posterior_window_turns: int
    posterior_refresh_turns: int = 8

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_total_variation)
            or not 0.0 <= self.maximum_total_variation <= 1.0
        ):
            raise ValueError(
                "maximum_total_variation must lie in [0, 1]"
            )
        if self.arming_turns < 1:
            raise ValueError("arming_turns must be positive")
        if (
            not math.isfinite(self.containment_ceiling)
            or not 0.0 <= self.containment_ceiling <= 1.0
        ):
            raise ValueError(
                "containment_ceiling must lie in [0, 1]"
            )
        if self.posterior_window_turns < 1:
            raise ValueError(
                "posterior_window_turns must be positive"
            )
        if self.posterior_refresh_turns < 1:
            raise ValueError(
                "posterior_refresh_turns must be positive"
            )

    @property
    def candidate_id(self) -> str:
        return (
            "sequential:"
            f"tv={self.maximum_total_variation:.12g}:"
            f"arm={self.arming_turns}:"
            f"ceiling={self.containment_ceiling:.12g}:"
            f"window={self.posterior_window_turns}:"
            f"refresh={self.posterior_refresh_turns}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class SequentialResponseCalibrationConfig:
    """Disjoint feedback-model, feedback-calibration, and audit contract."""

    condition: str = "integrity_only_controller_attack"
    active_detectors: tuple[str, ...] = (
        DETECTOR_CLASSICAL_SERVICE,
        DETECTOR_QUANTUM_ENDPOINT,
    )
    maximum_total_variations: tuple[float, ...] = (0.01, 0.05)
    arming_turns: tuple[int, ...] = (256, 1_024)
    containment_ceilings: tuple[float, ...] = (0.01, 0.05, 0.10)
    posterior_window_turns: tuple[int, ...] = (256, 1_024)
    posterior_refresh_turns: int = 32
    independent_arm_update_mode: str = (
        RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW
    )
    include_monitor_only_candidate: bool = True
    switching_penalty: float = 0.05
    compatibility_effect_floor: float = 0.01
    posterior_pseudocount: float = 0.5
    feedback_model_draw_start: int = 0
    feedback_model_draw_stop: int = 32
    feedback_calibration_draw_start: int = 32
    feedback_calibration_draw_stop: int | None = None
    audit_draw_start: int = 0
    audit_draw_stop: int | None = None
    record_every: int = 100

    def __post_init__(self) -> None:
        if not str(self.condition).strip():
            raise ValueError("condition must be nonempty")
        if (
            not self.active_detectors
            or len(set(self.active_detectors))
            != len(self.active_detectors)
            or any(
                detector not in ACTIVE_RESPONSE_DETECTORS
                for detector in self.active_detectors
            )
        ):
            raise ValueError(
                "active_detectors are empty, duplicated, or invalid"
            )
        if self.independent_arm_update_mode not in RESPONSE_UPDATE_MODES:
            raise ValueError(
                "independent_arm_update_mode is not declared"
            )
        if (
            not self.maximum_total_variations
            or not self.arming_turns
            or not self.containment_ceilings
            or not self.posterior_window_turns
        ):
            raise ValueError("sequential response grid is empty")
        candidates = self.candidates()
        if len({candidate.candidate_id for candidate in candidates}) != len(
            candidates
        ):
            raise ValueError("sequential response grid contains duplicates")
        if (
            not math.isfinite(self.switching_penalty)
            or self.switching_penalty < 0.0
        ):
            raise ValueError(
                "switching_penalty must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.compatibility_effect_floor)
            or not 0.0 <= self.compatibility_effect_floor <= 1.0
        ):
            raise ValueError(
                "compatibility_effect_floor must lie in [0, 1]"
            )
        if (
            not math.isfinite(self.posterior_pseudocount)
            or self.posterior_pseudocount <= 0.0
        ):
            raise ValueError(
                "posterior_pseudocount must be finite and positive"
            )
        if self.feedback_model_draw_start < 0:
            raise ValueError("feedback model draw start is negative")
        if (
            self.feedback_model_draw_stop
            <= self.feedback_model_draw_start
        ):
            raise ValueError(
                "feedback model draw stop must exceed its start"
            )
        if (
            self.feedback_calibration_draw_start
            < self.feedback_model_draw_stop
        ):
            raise ValueError(
                "feedback calibration overlaps the model draw range"
            )
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def candidates(self) -> tuple[SequentialResponseCandidate, ...]:
        active = tuple(
            SequentialResponseCandidate(
                maximum_total_variation=float(tv),
                arming_turns=int(arming),
                containment_ceiling=float(ceiling),
                posterior_window_turns=int(window),
                posterior_refresh_turns=int(
                    self.posterior_refresh_turns
                ),
            )
            for tv in self.maximum_total_variations
            for arming in self.arming_turns
            for ceiling in self.containment_ceilings
            for window in self.posterior_window_turns
        )
        if not self.include_monitor_only_candidate:
            return active
        monitor_only = SequentialResponseCandidate(
            maximum_total_variation=0.0,
            arming_turns=max(map(int, self.arming_turns)),
            containment_ceiling=0.0,
            posterior_window_turns=max(
                map(int, self.posterior_window_turns)
            ),
            posterior_refresh_turns=int(
                self.posterior_refresh_turns
            ),
        )
        return (*active, monitor_only)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "active_detectors": list(self.active_detectors),
            "maximum_total_variations": list(
                self.maximum_total_variations
            ),
            "arming_turns": list(self.arming_turns),
            "containment_ceilings": list(
                self.containment_ceilings
            ),
            "posterior_window_turns": list(
                self.posterior_window_turns
            ),
            "candidates": [
                candidate.to_dict() for candidate in self.candidates()
            ],
        }


def run_sequential_response_calibration_from_detection(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    detection_study: Mapping[str, Any],
    calibration_config: SequentialResponseCalibrationConfig = (
        SequentialResponseCalibrationConfig()
    ),
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Calibrate on feedback draws and evaluate selected profiles on audit."""

    if payoff_bank.route_ids != evidence_bank.route_ids:
        raise ValueError("payoff and evidence route axes differ")
    if payoff_bank.action_ids != evidence_bank.action_ids:
        raise ValueError("payoff and evidence action axes differ")
    if str(detection_study.get("case_id")) != payoff_bank.case_id:
        raise ValueError("detection study and banks use different cases")
    _validate_draw_partition(evidence_bank, calibration_config)

    source_config = dict(detection_study["study_config"])
    schedule = ControllerPrimeExploitSchedule(
        **{
            name: source_config["schedule"][name]
            for name in (
                "prime_turns",
                "exploit_turns",
                "record_every",
                "trigger_confidence_z",
            )
        }
    )
    detector_config = BinomialEvidenceDetectorConfig(
        **{
            field.name: source_config["detector_config"][field.name]
            for field in fields(BinomialEvidenceDetectorConfig)
        }
    )
    commissioning = _commissioning_evidence_by_route(
        evidence_bank,
        stop=int(source_config["commissioning_draws_per_route"]),
    )
    source_rows = [
        dict(row)
        for row in detection_study["rows"]
        if str(row["condition"]) == calibration_config.condition
    ]
    if not source_rows:
        raise ValueError("no source rows match the calibration condition")
    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )

    follow_cache: dict[
        tuple[str, int],
        StackelbergBanditFollowRuntime,
    ] = {}
    feedback_alarm_records: dict[
        tuple[str, int, str],
        dict[str, Any],
    ] = {}
    audit_alarm_records: dict[
        tuple[str, int, str],
        dict[str, Any],
    ] = {}
    candidates = calibration_config.candidates()
    calibration_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_calibration_runs = (
        len(source_rows)
        * len(calibration_config.active_detectors)
        * len(candidates)
    )
    completed = 0

    for source_row in source_rows:
        label = str(source_row["true_hypothesis_label_for_scoring_only"])
        seed = int(source_row["seed"])
        if label not in learner_configs:
            raise KeyError(f"missing learner config {label!r}")
        follow = _follow_runtime(
            follow_cache,
            case,
            payoff_bank,
            learner_configs,
            label=label,
            seed=seed,
            maximum_follow_turns=int(
                source_config["maximum_follow_turns"]
            ),
            posterior_threshold=float(
                source_config["posterior_threshold"]
            ),
            selection_digest=selection_digest,
            record_every=calibration_config.record_every,
            observation_design=observation_design,
        )
        plan = _controller_plan_from_dict(source_row["plan"])
        for detector in calibration_config.active_detectors:
            key = (label, seed, detector)
            feedback_alarm = discover_alarm_response_record(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs[label],
                plan,
                schedule=schedule,
                alice=follow.clone_true_alice(
                    continuation_seed=seed
                ),
                observation_design=observation_design,
                detectors=_fresh_detectors(
                    commissioning,
                    detector_config,
                ),
                active_detector=detector,
                seed=seed,
                attack_enabled=bool(source_row["attack_enabled"]),
                execution_split="feedback",
                execution_draw_start=(
                    calibration_config.feedback_calibration_draw_start
                ),
                execution_draw_stop=(
                    calibration_config.feedback_calibration_draw_stop
                ),
                selection_digest_blake2b_128=selection_digest,
            )
            feedback_alarm_records[key] = feedback_alarm
            for candidate in candidates:
                response = run_alarm_conditioned_response_continuation(
                    case,
                    payoff_bank,
                    evidence_bank,
                    learner_configs[label],
                    plan,
                    schedule=schedule,
                    alice=follow.clone_true_alice(
                        continuation_seed=seed
                    ),
                    observation_design=observation_design,
                    detectors=_fresh_detectors(
                        commissioning,
                        detector_config,
                    ),
                    response_config=_response_config(
                        calibration_config,
                        candidate,
                        detector,
                    ),
                    seed=seed,
                    attack_enabled=bool(
                        source_row["attack_enabled"]
                    ),
                    condition=str(source_row["condition"]),
                    frozen_alarm_record=feedback_alarm,
                    execution_split="feedback",
                    execution_draw_start=(
                        calibration_config.feedback_calibration_draw_start
                    ),
                    execution_draw_stop=(
                        calibration_config.feedback_calibration_draw_stop
                    ),
                    selection_digest_blake2b_128=selection_digest,
                )
                calibration_rows.append(
                    _calibration_row(
                        response,
                        label=label,
                        detector=detector,
                        candidate=candidate,
                    )
                )
                completed += 1
                if progress_callback is not None and (
                    completed == 1
                    or completed == total_calibration_runs
                    or completed % max(1, len(candidates)) == 0
                ):
                    progress_callback(
                        "sequential response calibration "
                        f"{completed}/{total_calibration_runs}"
                    )

    selections = _select_candidates(calibration_rows, candidates)
    audit_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        label = str(source_row["true_hypothesis_label_for_scoring_only"])
        seed = int(source_row["seed"])
        follow = follow_cache[(label, seed)]
        plan = _controller_plan_from_dict(source_row["plan"])
        for detector in calibration_config.active_detectors:
            key = (label, seed, detector)
            audit_alarm = discover_alarm_response_record(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs[label],
                plan,
                schedule=schedule,
                alice=follow.clone_true_alice(
                    continuation_seed=seed
                ),
                observation_design=observation_design,
                detectors=_fresh_detectors(
                    commissioning,
                    detector_config,
                ),
                active_detector=detector,
                seed=seed,
                attack_enabled=bool(source_row["attack_enabled"]),
                execution_split="audit",
                execution_draw_start=calibration_config.audit_draw_start,
                execution_draw_stop=calibration_config.audit_draw_stop,
                selection_digest_blake2b_128=selection_digest,
            )
            expected_alarm = _source_alarm_turn(source_row, detector)
            if audit_alarm["alarm_turn"] != expected_alarm:
                raise AssertionError(
                    "audit alarm replay changed the source detector turn"
                )
            audit_alarm_records[key] = audit_alarm
            selected = selections[(label, detector)]
            candidate = selected["candidate"]
            response = run_alarm_conditioned_response_continuation(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs[label],
                plan,
                schedule=schedule,
                alice=follow.clone_true_alice(
                    continuation_seed=seed
                ),
                observation_design=observation_design,
                detectors=_fresh_detectors(
                    commissioning,
                    detector_config,
                ),
                response_config=_response_config(
                    calibration_config,
                    candidate,
                    detector,
                ),
                seed=seed,
                attack_enabled=bool(source_row["attack_enabled"]),
                condition=str(source_row["condition"]),
                frozen_alarm_record=audit_alarm,
                execution_split="audit",
                execution_draw_start=calibration_config.audit_draw_start,
                execution_draw_stop=calibration_config.audit_draw_stop,
                selection_digest_blake2b_128=selection_digest,
            )
            native_mean = _source_mean_reward(source_row)
            audit_rows.append({
                **_calibration_row(
                    response,
                    label=label,
                    detector=detector,
                    candidate=candidate,
                ),
                "native_mean_alice_reward": native_mean,
                "selected_response_minus_native_mean_reward": (
                    float(response["mean_alice_reward"]) - native_mean
                ),
                "selection_rule": selected["selection_rule"],
            })

    return {
        "schema": SEQUENTIAL_RESPONSE_CALIBRATION_SCHEMA,
        "case_id": payoff_bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "calibration_config": calibration_config.to_dict(),
        "source_detection_schema": detection_study["schema"],
        "source_study_config": source_config,
        "candidate_count": len(candidates),
        "calibration_run_count": len(calibration_rows),
        "audit_run_count": len(audit_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "calibration_rows": calibration_rows,
        "selections": [
            _selection_to_dict(label, detector, selection)
            for (label, detector), selection in sorted(
                selections.items()
            )
        ],
        "audit_rows": audit_rows,
        "audit_aggregates": _audit_aggregates(audit_rows),
        "alarm_records": {
            "feedback_count": len(feedback_alarm_records),
            "audit_count": len(audit_alarm_records),
        },
        "information_boundary": {
            "intervention_model": (
                "feedback draw range "
                f"[{calibration_config.feedback_model_draw_start}, "
                f"{calibration_config.feedback_model_draw_stop})"
            ),
            "parameter_calibration": (
                "feedback draw range "
                f"[{calibration_config.feedback_calibration_draw_start}, "
                f"{calibration_config.feedback_calibration_draw_stop})"
            ),
            "evaluation": "seed-disjoint endpoint evidence audit split",
            "excluded_from_selection": [
                "audit endpoint outcomes",
                "audit payoff matrix",
                "realized Eve action identity",
                "realized Eve target identity",
                "target-active simulator flag",
            ],
        },
        "claim_boundary": (
            "The selected response maximizes empirical full-horizon Alice "
            "reward under the declared feedback-bank grid and one-standard-"
            "error rule, including an explicit monitor-only candidate so "
            "active containment is not mandatory. Tsallis response turns use "
            "the declared behavior-policy importance correction, whose loss "
            "estimate is unbiased but whose controlled behavior is outside "
            "the native Tsallis-INF regret theorem. The audit split was held "
            "out from this calibration but had been used by earlier "
            "diagnostic response studies, so this is not a globally "
            "untouched confirmation."
        ),
    }


def sequential_response_config(
    calibration: SequentialResponseCalibrationConfig,
    candidate: SequentialResponseCandidate,
    detector: str,
) -> AlarmResponseConfig:
    return AlarmResponseConfig(
        strategy=RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
        active_detector=detector,
        switching_penalty=calibration.switching_penalty,
        compatibility_effect_floor=(
            calibration.compatibility_effect_floor
        ),
        topology_sharing_trust_region_tv=(
            candidate.maximum_total_variation
        ),
        independent_arm_trust_region_tv=(
            candidate.maximum_total_variation
        ),
        topology_sharing_arming_turns=candidate.arming_turns,
        independent_arm_arming_turns=candidate.arming_turns,
        topology_sharing_containment_ceiling=(
            candidate.containment_ceiling
        ),
        independent_arm_containment_ceiling=(
            candidate.containment_ceiling
        ),
        independent_arm_update_mode=(
            calibration.independent_arm_update_mode
        ),
        posterior_pseudocount=calibration.posterior_pseudocount,
        posterior_window_turns=candidate.posterior_window_turns,
        posterior_refresh_turns=candidate.posterior_refresh_turns,
        posterior_model_draw_start=(
            calibration.feedback_model_draw_start
        ),
        posterior_model_draw_stop=(
            calibration.feedback_model_draw_stop
        ),
        record_every=calibration.record_every,
    )


_response_config = sequential_response_config


def _follow_runtime(
    cache: dict[tuple[str, int], StackelbergBanditFollowRuntime],
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    label: str,
    seed: int,
    maximum_follow_turns: int,
    posterior_threshold: float,
    selection_digest: str,
    record_every: int,
    observation_design: RouteObservationDesign,
) -> StackelbergBanditFollowRuntime:
    key = (label, int(seed))
    cached = cache.get(key)
    if cached is not None:
        return cached
    runtime = run_stackelberg_bandit_follow_runtime(
        case,
        payoff_bank,
        true_alice_config=learner_configs[label],
        hypothesis_configs=learner_configs,
        observation_design=observation_design,
        turns=maximum_follow_turns,
        seed=int(seed),
        true_hypothesis_label=label,
        record_every=max(1, min(10, int(record_every))),
        decision_posterior_threshold=posterior_threshold,
        selection_digest_blake2b_128=selection_digest,
    )
    cache[key] = runtime
    return runtime


def _calibration_row(
    response: Mapping[str, Any],
    *,
    label: str,
    detector: str,
    candidate: SequentialResponseCandidate,
) -> dict[str, Any]:
    selection = response.get("response_selection") or {}
    posterior = selection.get("commissioned_intervention_posterior")
    return {
        "true_hypothesis_label": str(label),
        "active_detector": str(detector),
        "seed": int(response["seed"]),
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "alarm_turn": response["response_alarm_turn"],
        "mean_alice_reward": float(response["mean_alice_reward"]),
        "mean_prime_reward": float(
            response["phase_metrics"]["prime"]["mean_alice_reward"]
        ),
        "mean_exploit_reward": float(
            response["phase_metrics"]["exploit"]["mean_alice_reward"]
        ),
        "mean_response_window_reward": response[
            "mean_response_window_reward"
        ],
        "containment_controlled_turns": int(
            response["response_control"][
                "containment_controlled_turns"
            ]
        ),
        "learner_controlled_turns": int(
            response["response_control"]["learner_controlled_turns"]
        ),
        "behavior_mixture_turns": int(
            response["response_control"]["behavior_mixture_turns"]
        ),
        "mean_containment_probability": float(
            response["response_control"][
                "mean_containment_probability"
            ]
        ),
        "response_policy_refreshes": int(
            response["response_control"]["response_policy_refreshes"]
        ),
        "final_containment_probability": float(
            response["response_control"][
                "final_containment_probability"
            ]
        ),
        "final_posterior_attack_mass": (
            float(selection["posterior_attack_mass"])
            if "posterior_attack_mass" in selection
            else None
        ),
        "final_posterior_entropy_nats": (
            float(posterior["posterior_entropy_nats"])
            if isinstance(posterior, Mapping)
            else None
        ),
        "execution_split": str(response["execution_split"]),
        "execution_draw_start": int(
            response["execution_draw_start"]
        ),
        "execution_draw_stop": int(response["execution_draw_stop"]),
    }


def _select_candidates(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[SequentialResponseCandidate],
) -> dict[tuple[str, str], dict[str, Any]]:
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    selections: dict[tuple[str, str], dict[str, Any]] = {}
    pairs = sorted({
        (
            str(row["true_hypothesis_label"]),
            str(row["active_detector"]),
        )
        for row in rows
    })
    for label, detector in pairs:
        selected_rows = [
            row for row in rows
            if str(row["true_hypothesis_label"]) == label
            and str(row["active_detector"]) == detector
        ]
        summaries = []
        for candidate_id in sorted({
            str(row["candidate_id"]) for row in selected_rows
        }):
            candidate_rows = [
                row for row in selected_rows
                if str(row["candidate_id"]) == candidate_id
            ]
            rewards = np.asarray([
                float(row["mean_alice_reward"])
                for row in candidate_rows
            ])
            summaries.append({
                "candidate_id": candidate_id,
                "candidate": candidate_by_id[candidate_id],
                "runs": len(candidate_rows),
                "mean_alice_reward": float(np.mean(rewards)),
                "standard_error": (
                    float(np.std(rewards, ddof=1) / math.sqrt(rewards.size))
                    if rewards.size > 1
                    else 0.0
                ),
                "mean_prime_reward": float(np.mean([
                    float(row["mean_prime_reward"])
                    for row in candidate_rows
                ])),
                "mean_exploit_reward": float(np.mean([
                    float(row["mean_exploit_reward"])
                    for row in candidate_rows
                ])),
            })
        raw = max(
            summaries,
            key=lambda row: (
                row["mean_alice_reward"],
                row["candidate_id"],
            ),
        )
        threshold = (
            float(raw["mean_alice_reward"])
            - float(raw["standard_error"])
        )
        eligible = [
            row for row in summaries
            if float(row["mean_alice_reward"]) >= threshold
        ]
        chosen = min(
            eligible,
            key=lambda row: _conservative_preference(
                row["candidate"]
            ),
        )
        selections[(label, detector)] = {
            "candidate": chosen["candidate"],
            "raw_best": raw,
            "selected": chosen,
            "eligible_candidate_ids": [
                str(row["candidate_id"])
                for row in sorted(
                    eligible,
                    key=lambda row: str(row["candidate_id"]),
                )
            ],
            "selection_rule": {
                "objective": "maximize mean full-horizon Alice reward",
                "uncertainty": (
                    "one-standard-error set around the raw maximum"
                ),
                "tie_break": (
                    "prefer monitor-only when it lies in the one-standard-"
                    "error set; otherwise use the smallest ceiling, then "
                    "smallest TV radius, longest arming horizon, longest "
                    "posterior window, and longest refresh interval"
                ),
                "threshold": threshold,
            },
            "candidate_summaries": summaries,
        }
    return selections


def _conservative_preference(
    candidate: SequentialResponseCandidate,
) -> tuple[Any, ...]:
    return (
        candidate.containment_ceiling,
        candidate.maximum_total_variation,
        -candidate.arming_turns,
        -candidate.posterior_window_turns,
        -candidate.posterior_refresh_turns,
        candidate.candidate_id,
    )


def _selection_to_dict(
    label: str,
    detector: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "true_hypothesis_label": label,
        "active_detector": detector,
        "selected_candidate": selection["candidate"].to_dict(),
        "raw_best": _summary_to_dict(selection["raw_best"]),
        "selected": _summary_to_dict(selection["selected"]),
        "eligible_candidate_ids": list(
            selection["eligible_candidate_ids"]
        ),
        "selection_rule": dict(selection["selection_rule"]),
        "candidate_summaries": [
            _summary_to_dict(row)
            for row in selection["candidate_summaries"]
        ],
    }


def _summary_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.to_dict()
            if isinstance(value, SequentialResponseCandidate)
            else value
        )
        for key, value in row.items()
    }


def _audit_aggregates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = []
    keys = sorted({
        (
            str(row["true_hypothesis_label"]),
            str(row["active_detector"]),
        )
        for row in rows
    })
    for label, detector in keys:
        selected = [
            row for row in rows
            if str(row["true_hypothesis_label"]) == label
            and str(row["active_detector"]) == detector
        ]
        aggregates.append({
            "true_hypothesis_label": label,
            "active_detector": detector,
            "runs": len(selected),
            "selected_candidate_id": str(
                selected[0]["candidate_id"]
            ),
            "mean_alice_reward": float(np.mean([
                float(row["mean_alice_reward"]) for row in selected
            ])),
            "mean_native_alice_reward": float(np.mean([
                float(row["native_mean_alice_reward"])
                for row in selected
            ])),
            "mean_selected_minus_native_reward": float(np.mean([
                float(
                    row[
                        "selected_response_minus_native_mean_reward"
                    ]
                )
                for row in selected
            ])),
            "mean_prime_reward": float(np.mean([
                float(row["mean_prime_reward"]) for row in selected
            ])),
            "mean_exploit_reward": float(np.mean([
                float(row["mean_exploit_reward"]) for row in selected
            ])),
            "mean_containment_controlled_fraction": float(np.mean([
                (
                    int(row["containment_controlled_turns"])
                    / (
                        int(row["containment_controlled_turns"])
                        + int(row["learner_controlled_turns"])
                    )
                    if (
                        int(row["containment_controlled_turns"])
                        + int(row["learner_controlled_turns"])
                    )
                    else 0.0
                )
                for row in selected
            ])),
        })
    return aggregates


def _validate_draw_partition(
    bank: HistogramEndpointEvidenceBank,
    config: SequentialResponseCalibrationConfig,
) -> None:
    feedback_stop = (
        bank.draws_per_cell
        if config.feedback_calibration_draw_stop is None
        else int(config.feedback_calibration_draw_stop)
    )
    audit_stop = (
        bank.draws_per_cell
        if config.audit_draw_stop is None
        else int(config.audit_draw_stop)
    )
    if not (
        0
        <= config.feedback_model_draw_start
        < config.feedback_model_draw_stop
        <= config.feedback_calibration_draw_start
        < feedback_stop
        <= bank.draws_per_cell
    ):
        raise ValueError("feedback model/calibration draw partition is invalid")
    if not (
        0 <= config.audit_draw_start < audit_stop <= bank.draws_per_cell
    ):
        raise ValueError("audit draw range is invalid")


def _fresh_detectors(
    commissioning: Sequence[Sequence[Any]],
    config: BinomialEvidenceDetectorConfig,
) -> MatchedClassicalQuantumEndpointDetectors:
    return MatchedClassicalQuantumEndpointDetectors.from_commissioning(
        commissioning,
        config,
    )


def _commissioning_evidence_by_route(
    bank: HistogramEndpointEvidenceBank,
    *,
    stop: int,
) -> tuple[tuple[Any, ...], ...]:
    if not 2 <= stop <= bank.draws_per_cell:
        raise ValueError("commissioning stop is outside the evidence bank")
    no_attack = bank.action_ids.index("no_attack")
    return tuple(
        tuple(
            bank.feedback.evidence(
                route,
                no_attack,
                draw,
                requested_pair_count=bank.requested_pairs_per_draw,
            )
            for draw in range(stop)
        )
        for route in range(len(bank.route_ids))
    )


def _controller_plan_from_dict(
    payload: Mapping[str, Any],
) -> ControllerPrimeExploitPlan:
    tuple_fields = {
        "target_route_indices",
        "target_route_ids",
        "alternative_route_indices",
        "prime_action_indices",
        "prime_action_ids",
        "prime_attack_types",
        "prime_probabilities",
    }
    values = {}
    for field in fields(ControllerPrimeExploitPlan):
        value = payload[field.name]
        values[field.name] = (
            tuple(value) if field.name in tuple_fields else value
        )
    return ControllerPrimeExploitPlan(**values)


def _source_alarm_turn(
    row: Mapping[str, Any],
    detector: str,
) -> int | None:
    key = (
        "classical_alarm_turn"
        if detector == DETECTOR_CLASSICAL_SERVICE
        else "quantum_alarm_turn"
    )
    value = row["detector_outcomes"][key]
    return None if value is None else int(value)


def _source_mean_reward(row: Mapping[str, Any]) -> float:
    total_reward = 0.0
    total_turns = 0
    for phase in ("prime", "exploit"):
        phase_row = row["phase_metrics"][phase]
        turns = int(phase_row["turns"])
        total_reward += turns * float(phase_row["mean_alice_reward"])
        total_turns += turns
    return total_reward / total_turns
