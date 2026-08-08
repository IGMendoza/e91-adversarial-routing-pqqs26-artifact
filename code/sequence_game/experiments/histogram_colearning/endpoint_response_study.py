"""Matched response study downstream of a frozen endpoint detector study."""

from __future__ import annotations

from collections import defaultdict
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
    ALARM_RESPONSE_STRATEGIES,
    ACTIVE_RESPONSE_DETECTORS,
    DETECTOR_CLASSICAL_SERVICE,
    DETECTOR_QUANTUM_ENDPOINT,
    RESPONSE_COMMISSIONED_ROBUST_POLICY,
    RESPONSE_FREEZE_LIVE_POLICY,
    RESPONSE_NATIVE_CONTINUATION,
    RESPONSE_POSTERIOR_TRUST_REGION,
    RESPONSE_QUARANTINE_LEADING_ROUTE,
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


ENDPOINT_RESPONSE_STUDY_SCHEMA = (
    "histogram_stackelberg_endpoint_response_study.v1"
)


@dataclass(frozen=True)
class EndpointResponseStudyConfig:
    """Frozen downstream response comparison."""

    conditions: tuple[str, ...] = (
        "integrity_only_controller_attack",
    )
    active_detectors: tuple[str, ...] = (
        DETECTOR_CLASSICAL_SERVICE,
        DETECTOR_QUANTUM_ENDPOINT,
    )
    strategies: tuple[str, ...] = (
        RESPONSE_NATIVE_CONTINUATION,
        RESPONSE_FREEZE_LIVE_POLICY,
        RESPONSE_QUARANTINE_LEADING_ROUTE,
        RESPONSE_COMMISSIONED_ROBUST_POLICY,
        RESPONSE_POSTERIOR_TRUST_REGION,
    )
    switching_penalty: float = 0.05
    compatibility_effect_floor: float = 0.01
    topology_sharing_trust_region_tv: float = 0.25
    independent_arm_trust_region_tv: float = 0.10
    topology_sharing_arming_turns: int = 4
    independent_arm_arming_turns: int = 12
    topology_sharing_containment_ceiling: float = 0.50
    independent_arm_containment_ceiling: float = 0.25
    posterior_pseudocount: float = 0.5
    record_every: int = 100

    def __post_init__(self) -> None:
        if (
            not self.conditions
            or len(set(self.conditions)) != len(self.conditions)
        ):
            raise ValueError("conditions must be nonempty and unique")
        if (
            not self.active_detectors
            or len(set(self.active_detectors)) != len(self.active_detectors)
            or any(
                value not in ACTIVE_RESPONSE_DETECTORS
                for value in self.active_detectors
            )
        ):
            raise ValueError("active_detectors are empty, duplicated, or invalid")
        if (
            not self.strategies
            or len(set(self.strategies)) != len(self.strategies)
            or any(
                value not in ALARM_RESPONSE_STRATEGIES
                for value in self.strategies
            )
        ):
            raise ValueError("strategies are empty, duplicated, or invalid")
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
        for name, value in (
            (
                "topology_sharing_trust_region_tv",
                self.topology_sharing_trust_region_tv,
            ),
            (
                "independent_arm_trust_region_tv",
                self.independent_arm_trust_region_tv,
            ),
            (
                "topology_sharing_containment_ceiling",
                self.topology_sharing_containment_ceiling,
            ),
            (
                "independent_arm_containment_ceiling",
                self.independent_arm_containment_ceiling,
            ),
        ):
            if (
                not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        for name, value in (
            (
                "topology_sharing_arming_turns",
                self.topology_sharing_arming_turns,
            ),
            (
                "independent_arm_arming_turns",
                self.independent_arm_arming_turns,
            ),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not math.isfinite(self.posterior_pseudocount)
            or self.posterior_pseudocount <= 0.0
        ):
            raise ValueError(
                "posterior_pseudocount must be finite and positive"
            )
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "conditions": list(self.conditions),
            "active_detectors": list(self.active_detectors),
            "strategies": list(self.strategies),
        }


def run_endpoint_response_study_from_detection(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    detection_study: Mapping[str, Any],
    response_study_config: EndpointResponseStudyConfig = (
        EndpointResponseStudyConfig()
    ),
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Freeze each alarm once, then replay matched response branches."""

    if str(detection_study.get("case_id")) != payoff_bank.case_id:
        raise ValueError("detection study and payoff bank cases differ")
    if (
        str(
            detection_study["payoff_bank"][
                "content_digest_blake2b_128"
            ]
        )
        != payoff_bank.content_digest()
    ):
        raise ValueError("detection study payoff-bank digest changed")
    if (
        str(
            detection_study["endpoint_evidence_bank"][
                "content_digest_blake2b_128"
            ]
        )
        != evidence_bank.content_digest()
    ):
        raise ValueError("detection study endpoint-bank digest changed")

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
    selected_rows = [
        dict(row)
        for row in detection_study["rows"]
        if str(row["condition"]) in response_study_config.conditions
    ]
    if not selected_rows:
        raise ValueError("no detection rows match the response conditions")

    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )
    follow_cache: dict[
        tuple[str, int],
        StackelbergBanditFollowRuntime,
    ] = {}
    rows: list[dict[str, Any]] = []
    alarm_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_alarm_records = (
        len(selected_rows) * len(response_study_config.active_detectors)
    )
    completed_alarm_records = 0

    for source_row in selected_rows:
        label = str(source_row["true_hypothesis_label_for_scoring_only"])
        seed = int(source_row["seed"])
        if label not in learner_configs:
            raise KeyError(f"missing learner config {label!r}")
        follow_key = (label, seed)
        follow = follow_cache.get(follow_key)
        if follow is None:
            follow = run_stackelberg_bandit_follow_runtime(
                case,
                payoff_bank,
                true_alice_config=learner_configs[label],
                hypothesis_configs=learner_configs,
                observation_design=observation_design,
                turns=int(source_config["maximum_follow_turns"]),
                seed=seed,
                true_hypothesis_label=label,
                record_every=max(
                    1,
                    min(10, response_study_config.record_every),
                ),
                decision_posterior_threshold=float(
                    source_config["posterior_threshold"]
                ),
                selection_digest_blake2b_128=selection_digest,
            )
            follow_cache[follow_key] = follow
        plan = _controller_plan_from_dict(source_row["plan"])

        for active_detector in response_study_config.active_detectors:
            alarm_record = discover_alarm_response_record(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs[label],
                plan,
                schedule=schedule,
                alice=follow.clone_true_alice(continuation_seed=seed),
                observation_design=observation_design,
                detectors=_fresh_detectors(commissioning, detector_config),
                active_detector=active_detector,
                seed=seed,
                attack_enabled=bool(source_row["attack_enabled"]),
                selection_digest_blake2b_128=selection_digest,
            )
            expected_alarm_turn = _source_alarm_turn(
                source_row,
                active_detector,
            )
            if alarm_record["alarm_turn"] != expected_alarm_turn:
                raise AssertionError(
                    "response alarm replay changed the frozen detector turn: "
                    f"expected {expected_alarm_turn}, got "
                    f"{alarm_record['alarm_turn']}"
                )
            alarm_records.append({
                **alarm_record,
                "condition": str(source_row["condition"]),
                "true_hypothesis_label_for_scoring_only": label,
                "seed": seed,
                "plan_id": plan.plan_id,
            })

            for strategy in response_study_config.strategies:
                run = run_alarm_conditioned_response_continuation(
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
                    response_config=AlarmResponseConfig(
                        strategy=strategy,
                        active_detector=active_detector,
                        switching_penalty=(
                            response_study_config.switching_penalty
                        ),
                        compatibility_effect_floor=(
                            response_study_config.compatibility_effect_floor
                        ),
                        topology_sharing_trust_region_tv=(
                            response_study_config.topology_sharing_trust_region_tv
                        ),
                        independent_arm_trust_region_tv=(
                            response_study_config.independent_arm_trust_region_tv
                        ),
                        topology_sharing_arming_turns=(
                            response_study_config.topology_sharing_arming_turns
                        ),
                        independent_arm_arming_turns=(
                            response_study_config.independent_arm_arming_turns
                        ),
                        topology_sharing_containment_ceiling=(
                            response_study_config.topology_sharing_containment_ceiling
                        ),
                        independent_arm_containment_ceiling=(
                            response_study_config.independent_arm_containment_ceiling
                        ),
                        posterior_pseudocount=(
                            response_study_config.posterior_pseudocount
                        ),
                        record_every=response_study_config.record_every,
                    ),
                    seed=seed,
                    attack_enabled=bool(source_row["attack_enabled"]),
                    condition=str(source_row["condition"]),
                    frozen_alarm_record=alarm_record,
                    selection_digest_blake2b_128=selection_digest,
                )
                source_mean = _source_mean_reward(source_row)
                rows.append({
                    **run,
                    "true_hypothesis_label_for_scoring_only": label,
                    "source_detection_mean_alice_reward": source_mean,
                    "native_replay_minus_source_mean_reward": (
                        float(run["mean_alice_reward"]) - source_mean
                        if strategy == RESPONSE_NATIVE_CONTINUATION
                        else None
                    ),
                    "source_detector_outcomes": dict(
                        source_row["detector_outcomes"]
                    ),
                })

            completed_alarm_records += 1
            if progress_callback is not None:
                progress_callback(
                    "endpoint response alarm "
                    f"{completed_alarm_records}/{total_alarm_records}: "
                    f"{label}, seed={seed}, detector={active_detector}"
                )

    native_replay_errors = [
        abs(float(row["native_replay_minus_source_mean_reward"]))
        for row in rows
        if row["native_replay_minus_source_mean_reward"] is not None
    ]
    return {
        "schema": ENDPOINT_RESPONSE_STUDY_SCHEMA,
        "case_id": payoff_bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "response_study_config": response_study_config.to_dict(),
        "source_detection_schema": detection_study["schema"],
        "source_study_config": source_config,
        "learner_configs": {
            label: {
                "alice_learner_mode": config.alice_learner_mode,
                "structured_exploration_mode": (
                    config.structured_exploration_mode
                ),
                "alice_tsallis_inf_estimator": (
                    config.resolved_alice_tsallis_inf_estimator
                ),
                "alice_tsallis_inf_eta_scale": (
                    config.alice_tsallis_inf_eta_scale
                ),
                "alice_exp3_eta_c": config.alice_exp3_eta_c,
            }
            for label, config in learner_configs.items()
        },
        "run_count": len(rows),
        "alarm_record_count": len(alarm_records),
        "elapsed_seconds": time.perf_counter() - started,
        "alarm_records": alarm_records,
        "rows": rows,
        "aggregates": _aggregate_response_rows(rows),
        "replay_validation": {
            "native_replay_count": len(native_replay_errors),
            "maximum_absolute_mean_reward_error": max(
                native_replay_errors,
                default=0.0,
            ),
            "exact_within_1e_12": all(
                value <= 1e-12 for value in native_replay_errors
            ),
        },
        "information_boundary": {
            "alarm": (
                "one frozen held-out public detector record per learner, "
                "seed, condition, and active detector"
            ),
            "response_selection": (
                "exact live Alice state plus feedback-bank commissioned "
                "model; audit payoffs are scoring-only"
            ),
            "learner_difference": (
                "Tsallis-INF uses independent route-arm statistics; "
                "LinearExp2 uses topology features and G-optimal exploration"
            ),
            "excluded_online": [
                "realized Eve action identity",
                "realized Eve target identity",
                "audit payoff matrix",
                "counterfactual route rewards",
            ],
        },
        "claim_boundary": (
            "The study compares alarm timing and response control inside the "
            "same compiled E91 repeater-service simulation. It does not claim "
            "that LinearExp2 must recover faster or that a packet simulator is "
            "required for the repeated-game baseline."
        ),
    }


def _aggregate_response_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["true_hypothesis_label_for_scoring_only"]),
            str(row["condition"]),
            str(row["response_config"]["active_detector"]),
            str(row["response_config"]["strategy"]),
        )
        grouped[key].append(row)
    cells = []
    for key, cell_rows in sorted(grouped.items()):
        label, condition, detector, strategy = key
        cells.append({
            "true_hypothesis_label": label,
            "condition": condition,
            "active_detector": detector,
            "strategy": strategy,
            "runs": len(cell_rows),
            "mean_alarm_turn": _optional_mean(
                row["response_alarm_turn"] for row in cell_rows
            ),
            "mean_alice_reward": float(np.mean([
                float(row["mean_alice_reward"]) for row in cell_rows
            ])),
            "mean_prime_reward": float(np.mean([
                float(row["phase_metrics"]["prime"]["mean_alice_reward"])
                for row in cell_rows
            ])),
            "mean_exploit_reward": float(np.mean([
                float(row["phase_metrics"]["exploit"]["mean_alice_reward"])
                for row in cell_rows
            ])),
            "mean_response_window_reward": _optional_mean(
                row["mean_response_window_reward"] for row in cell_rows
            ),
            "mean_policy_switch_total_variation": _optional_mean(
                _response_total_variation(row) for row in cell_rows
            ),
            "mean_target_policy_total_variation_from_live_at_alarm": (
                _optional_mean(
                    _response_total_variation(row) for row in cell_rows
                )
            ),
            "mean_containment_controlled_fraction": _optional_mean(
                (
                    row["response_control"][
                        "containment_controlled_turns"
                    ]
                    / row["response_window_turns"]
                    if row["response_window_turns"]
                    else None
                )
                for row in cell_rows
            ),
            "mean_final_containment_probability": _optional_mean(
                row["response_control"][
                    "final_containment_probability"
                ]
                for row in cell_rows
            ),
            "exploit_trigger_count": sum(
                bool(row["passive_exploit_trigger"]["triggered"])
                for row in cell_rows
            ),
        })

    paired_quantum_minus_classical = []
    paired_strategy_minus_native = []
    by_branch = {
        (
            str(row["true_hypothesis_label_for_scoring_only"]),
            int(row["seed"]),
            str(row["condition"]),
            str(row["response_config"]["active_detector"]),
            str(row["response_config"]["strategy"]),
        ): row
        for row in rows
    }
    labels = sorted({
        str(row["true_hypothesis_label_for_scoring_only"])
        for row in rows
    })
    conditions = sorted({str(row["condition"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    strategies = sorted({
        str(row["response_config"]["strategy"]) for row in rows
    })
    detectors = sorted({
        str(row["response_config"]["active_detector"]) for row in rows
    })
    for label in labels:
        for condition in conditions:
            for strategy in strategies:
                differences = []
                exploit_differences = []
                for seed in seeds:
                    quantum = by_branch.get((
                        label,
                        seed,
                        condition,
                        DETECTOR_QUANTUM_ENDPOINT,
                        strategy,
                    ))
                    classical = by_branch.get((
                        label,
                        seed,
                        condition,
                        DETECTOR_CLASSICAL_SERVICE,
                        strategy,
                    ))
                    if quantum is None or classical is None:
                        continue
                    differences.append(
                        float(quantum["mean_alice_reward"])
                        - float(classical["mean_alice_reward"])
                    )
                    exploit_differences.append(
                        float(
                            quantum["phase_metrics"]["exploit"][
                                "mean_alice_reward"
                            ]
                        )
                        - float(
                            classical["phase_metrics"]["exploit"][
                                "mean_alice_reward"
                            ]
                        )
                    )
                if differences:
                    paired_quantum_minus_classical.append({
                        "true_hypothesis_label": label,
                        "condition": condition,
                        "strategy": strategy,
                        "paired_runs": len(differences),
                        "mean_full_horizon_reward_difference": float(
                            np.mean(differences)
                        ),
                        "mean_exploit_reward_difference": float(
                            np.mean(exploit_differences)
                        ),
                        "per_seed_full_horizon_reward_difference": differences,
                    })

            for detector in detectors:
                for strategy in strategies:
                    if strategy == RESPONSE_NATIVE_CONTINUATION:
                        continue
                    differences = []
                    for seed in seeds:
                        response = by_branch.get((
                            label,
                            seed,
                            condition,
                            detector,
                            strategy,
                        ))
                        native = by_branch.get((
                            label,
                            seed,
                            condition,
                            detector,
                            RESPONSE_NATIVE_CONTINUATION,
                        ))
                        if response is None or native is None:
                            continue
                        differences.append(
                            float(response["mean_alice_reward"])
                            - float(native["mean_alice_reward"])
                        )
                    if differences:
                        paired_strategy_minus_native.append({
                            "true_hypothesis_label": label,
                            "condition": condition,
                            "active_detector": detector,
                            "strategy": strategy,
                            "paired_runs": len(differences),
                            "mean_full_horizon_reward_difference": float(
                                np.mean(differences)
                            ),
                            "per_seed_full_horizon_reward_difference": (
                                differences
                            ),
                        })
    return {
        "cells": cells,
        "paired_quantum_minus_classical": (
            paired_quantum_minus_classical
        ),
        "paired_strategy_minus_native": paired_strategy_minus_native,
    }


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
    no_attack_index = bank.action_ids.index("no_attack")
    return tuple(
        tuple(
            bank.feedback.evidence(
                route_index,
                no_attack_index,
                draw_index,
                requested_pair_count=bank.requested_pairs_per_draw,
            )
            for draw_index in range(stop)
        )
        for route_index in range(len(bank.route_ids))
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
        values[field.name] = tuple(value) if field.name in tuple_fields else value
    return ControllerPrimeExploitPlan(**values)


def _source_alarm_turn(
    row: Mapping[str, Any],
    active_detector: str,
) -> int | None:
    key = (
        "classical_alarm_turn"
        if active_detector == DETECTOR_CLASSICAL_SERVICE
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


def _response_total_variation(row: Mapping[str, Any]) -> float | None:
    selection = row.get("response_selection")
    if not selection:
        return None
    if "total_variation_from_live_policy" in selection:
        return float(selection["total_variation_from_live_policy"])
    robust = selection.get("commissioned_robust_policy")
    if robust is not None:
        return float(robust["total_variation_from_live_policy"])
    posterior = selection.get("posterior_trust_region_policy")
    if posterior is not None:
        return float(posterior["total_variation_from_live_policy"])
    if selection.get("strategy") == RESPONSE_NATIVE_CONTINUATION:
        return 0.0
    return None


def _optional_mean(values: Sequence[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None
