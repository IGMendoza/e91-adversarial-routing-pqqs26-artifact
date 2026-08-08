"""Matched passive-controller study over one compiled histogram payoff bank."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .stackelberg_bandit import (
    LearnerConditionedPlanLibrary,
    RouteObservationDesign,
    run_stackelberg_bandit_follow_runtime,
)
from .stackelberg_controller import (
    ControllerPrimeExploitSchedule,
    evaluate_post_follow_controller_plans,
    execute_post_follow_controller_decision,
)
from .stackelberg_controller_plans import (
    ControllerPrimeExploitPlan,
    build_controller_prime_exploit_plans,
    controller_plan_family_census,
    controller_plan_screening_representatives,
    controller_target_family_representatives,
)


CONTROLLER_STUDY_SCHEMA = "histogram_stackelberg_controller_study.v1"
CLASSICAL_CONTROLLER_ATTACK_TYPES = ("edge_dos", "swap_denial")


@dataclass(frozen=True)
class ControllerMatchedStudyConfig:
    """Frozen planning, execution, and comparison contract."""

    maximum_follow_turns: int = 100
    posterior_threshold: float = 0.99
    target_family_count: int = 5
    planning_seeds: tuple[int, ...] = (
        2_026_072_901,
        2_026_072_902,
        2_026_072_903,
    )
    execution_seeds: tuple[int, ...] = (
        2_026_071_901,
        2_026_071_902,
        2_026_071_903,
    )
    prime_turns: int = 3_000
    exploit_turns: int = 1_000
    record_every: int = 500
    execution_source_split: str = "audit"

    def __post_init__(self) -> None:
        if self.maximum_follow_turns < 1:
            raise ValueError("maximum_follow_turns must be positive")
        if (
            not math.isfinite(self.posterior_threshold)
            or not 0.5 < self.posterior_threshold < 1.0
        ):
            raise ValueError("posterior_threshold must lie in (0.5, 1)")
        if self.target_family_count < 2:
            raise ValueError("target_family_count must be at least two")
        for name, seeds in (
            ("planning_seeds", self.planning_seeds),
            ("execution_seeds", self.execution_seeds),
        ):
            if len(seeds) < 2 or len(set(map(int, seeds))) != len(seeds):
                raise ValueError(f"{name} must contain at least two unique seeds")
        if self.prime_turns < 1 or self.exploit_turns < 1:
            raise ValueError("Prime and Exploit turns must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if self.execution_source_split not in {"audit", "fresh"}:
            raise ValueError("execution_source_split must be audit or fresh")

    @property
    def schedule(self) -> ControllerPrimeExploitSchedule:
        return ControllerPrimeExploitSchedule(
            prime_turns=self.prime_turns,
            exploit_turns=self.exploit_turns,
            record_every=self.record_every,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "planning_seeds": list(map(int, self.planning_seeds)),
            "execution_seeds": list(map(int, self.execution_seeds)),
            "schedule": self.schedule.to_dict(),
        }


def run_matched_controller_study(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    study_config: ControllerMatchedStudyConfig = (
        ControllerMatchedStudyConfig()
    ),
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run posterior, information, oracle, and classical-action controls."""

    labels = tuple(map(str, learner_configs))
    if len(labels) != 2:
        raise ValueError(
            "the matched controller study currently requires two hypotheses"
        )
    if observation_design.route_ids != bank.route_ids:
        raise ValueError("observation design and payoff bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError("matched study requires exact passive route separation")
    base_config = learner_configs[labels[0]]
    all_plans = build_controller_prime_exploit_plans(
        case,
        bank,
        base_config,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
    )
    target_representatives = controller_target_family_representatives(
        all_plans,
        maximum_families=study_config.target_family_count,
    )
    selected_target_families = {
        plan.target_family_key for plan in target_representatives
    }
    full_plans = controller_plan_screening_representatives(tuple(
        plan
        for plan in all_plans
        if plan.target_family_key in selected_target_families
    ))
    classical_all = build_controller_prime_exploit_plans(
        case,
        bank,
        base_config,
        prime_attack_types=CLASSICAL_CONTROLLER_ATTACK_TYPES,
        exploit_attack_types=CLASSICAL_CONTROLLER_ATTACK_TYPES,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
    )
    classical_plans = controller_plan_screening_representatives(tuple(
        plan
        for plan in classical_all
        if plan.target_family_key in selected_target_families
    ))
    if len(full_plans) < 2 or len(classical_plans) < 2:
        raise ValueError("screened study requires at least two plans per arm")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    planning_rows: list[dict[str, Any]] = []
    total_follow_runs = len(labels) * len(study_config.execution_seeds)
    completed_follow_runs = 0
    for true_label in labels:
        true_config = learner_configs[true_label]
        wrong_label = next(label for label in labels if label != true_label)
        for execution_seed in map(int, study_config.execution_seeds):
            follow = run_stackelberg_bandit_follow_runtime(
                case,
                bank,
                true_alice_config=true_config,
                hypothesis_configs=learner_configs,
                observation_design=observation_design,
                turns=study_config.maximum_follow_turns,
                seed=execution_seed,
                true_hypothesis_label=true_label,
                record_every=max(1, min(10, study_config.record_every)),
                decision_posterior_threshold=(
                    study_config.posterior_threshold
                ),
                selection_digest_blake2b_128=(
                    selection_digest_blake2b_128
                ),
            )
            full_library, full_plan_rows = (
                evaluate_post_follow_controller_plans(
                    case,
                    bank,
                    true_config,
                    follow_runtime=follow,
                    plans=full_plans,
                    schedule=study_config.schedule,
                    observation_design=observation_design,
                    seeds=study_config.planning_seeds,
                    selection_digest_blake2b_128=(
                        selection_digest_blake2b_128
                    ),
                )
            )
            classical_library = _subset_plan_library(
                full_library,
                tuple(plan.plan_id for plan in classical_plans),
            )
            classical_plan_ids = set(classical_library.plan_ids)
            classical_plan_rows = [
                row for row in full_plan_rows
                if str(row["plan_id"]) in classical_plan_ids
            ]
            planning_rows.extend(
                _tag_planning_rows(
                    full_plan_rows,
                    true_label=true_label,
                    execution_seed=execution_seed,
                    action_contract="full_quantum_network_controls",
                )
            )
            planning_rows.extend(
                _tag_planning_rows(
                    classical_plan_rows,
                    true_label=true_label,
                    execution_seed=execution_seed,
                    action_contract="matched_classical_controls",
                )
            )

            posterior = follow.tracker.posterior_by_label()
            condition_specs = (
                (
                    "posterior_selected",
                    full_library,
                    full_plans,
                    posterior,
                ),
                (
                    "uninformed_prior_control",
                    full_library,
                    full_plans,
                    {label: 1.0 / len(labels) for label in labels},
                ),
                (
                    "wrong_learner_control",
                    full_library,
                    full_plans,
                    {
                        label: float(label == wrong_label)
                        for label in labels
                    },
                ),
                (
                    "true_learner_upper_bound",
                    full_library,
                    full_plans,
                    {
                        label: float(label == true_label)
                        for label in labels
                    },
                ),
                (
                    "classical_action_ablation",
                    classical_library,
                    classical_plans,
                    posterior,
                ),
            )
            for (
                condition,
                library,
                plan_values,
                decision_posterior,
            ) in condition_specs:
                result = execute_post_follow_controller_decision(
                    case,
                    bank,
                    true_config,
                    follow_runtime=follow,
                    plan_library=library,
                    plans=plan_values,
                    schedule=study_config.schedule,
                    observation_design=observation_design,
                    seed=execution_seed,
                    source_split=study_config.execution_source_split,
                    posterior=decision_posterior,
                    condition=condition,
                    selection_digest_blake2b_128=(
                        selection_digest_blake2b_128
                    ),
                )
                rows.append(_execution_row(
                    result,
                    true_label=true_label,
                    execution_seed=execution_seed,
                    follow_report=follow.report,
                    online_posterior=posterior,
                ))

            completed_follow_runs += 1
            if progress_callback is not None:
                progress_callback(
                    f"controller follow/planning/execution "
                    f"{completed_follow_runs}/{total_follow_runs}: "
                    f"{true_label}, seed={execution_seed}"
                )

    elapsed = time.perf_counter() - started
    return {
        "schema": CONTROLLER_STUDY_SCHEMA,
        "case_id": bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "study_config": study_config.to_dict(),
        "learner_configs": {
            label: _learner_config_summary(learner_configs[label])
            for label in labels
        },
        "observation_design": observation_design.to_dict(),
        "bank": bank.to_summary(),
        "plan_surface": {
            "complete": controller_plan_family_census(all_plans),
            "selected_target_family_count": len(selected_target_families),
            "selected_target_route_families": [
                list(plan.target_family_key)
                for plan in target_representatives
            ],
            "full_plan_count": len(full_plans),
            "full_plan_ids": [plan.plan_id for plan in full_plans],
            "classical_plan_count": len(classical_plans),
            "classical_plan_ids": [
                plan.plan_id for plan in classical_plans
            ],
            "classical_action_ablation_semantics": (
                "same topology, learners, passive access, target families, "
                "phase schedule, and feedback planning; memory degradation "
                "is excluded while edge and swap service denial remain"
            ),
        },
        "run_count": len(rows),
        "planning_row_count": len(planning_rows),
        "elapsed_seconds": elapsed,
        "rows": rows,
        "planning_rows": planning_rows,
        "aggregates": _aggregate_execution_rows(rows),
        "paired_contrasts": _paired_contrasts(rows),
        "claim_boundary": (
            "The classical-action arm is an action-set ablation inside the "
            "same histogram quantum-network service model. It is not yet a "
            "complete classical packet-network backend comparison."
        ),
    }


def _tag_planning_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    true_label: str,
    execution_seed: int,
    action_contract: str,
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "true_hypothesis_label_for_scoring_only": true_label,
            "execution_seed": int(execution_seed),
            "action_contract": action_contract,
        }
        for row in rows
    ]


def _subset_plan_library(
    library: LearnerConditionedPlanLibrary,
    plan_ids: Sequence[str],
) -> LearnerConditionedPlanLibrary:
    requested = tuple(map(str, plan_ids))
    indices = []
    for plan_id in requested:
        try:
            indices.append(library.plan_ids.index(plan_id))
        except ValueError as exc:
            raise ValueError(
                f"classical plan {plan_id!r} is absent from the full library"
            ) from exc
    index = np.asarray(indices, dtype=np.int64)
    values = np.asarray(library.expected_eve_values)[:, index]
    errors = (
        np.asarray(library.evaluation_standard_errors)[:, index]
        if library.evaluation_standard_errors is not None
        else None
    )
    counts = (
        np.asarray(library.evaluation_counts)[:, index]
        if library.evaluation_counts is not None
        else None
    )
    return LearnerConditionedPlanLibrary(
        hypothesis_labels=library.hypothesis_labels,
        plan_ids=requested,
        expected_eve_values=values,
        evaluation_standard_errors=errors,
        evaluation_counts=counts,
        value_semantics=library.value_semantics,
        source_split=library.source_split,
    )


def _execution_row(
    result: Mapping[str, Any],
    *,
    true_label: str,
    execution_seed: int,
    follow_report: Mapping[str, Any],
    online_posterior: Mapping[str, float],
) -> dict[str, Any]:
    execution = result["execution"]
    exploit = execution["phase_metrics"]["exploit"]
    prime = execution["phase_metrics"]["prime"]
    return {
        "condition": str(result["condition"]),
        "true_hypothesis_label_for_scoring_only": true_label,
        "execution_seed": int(execution_seed),
        "follow_turns": int(follow_report["turns"]),
        "follow_decision": dict(follow_report["decision"] or {}),
        "online_posterior": {
            str(label): float(value)
            for label, value in online_posterior.items()
        },
        "planning_posterior": dict(
            result["planning_decision"]["posterior"]
        ),
        "selected_plan_id": str(
            result["planning_decision"]["selected_plan_id"]
        ),
        "selected_posterior_expected_eve_value": float(
            result["planning_decision"][
                "selected_posterior_expected_eve_value"
            ]
        ),
        "exploit_triggered": bool(
            execution["passive_exploit_trigger"]["triggered"]
        ),
        "prime_mean_alice_reward": float(prime["mean_alice_reward"]),
        "prime_mean_eve_public_utility": float(
            prime["mean_eve_public_utility"]
        ),
        "prime_target_selection_rate": float(
            prime["passive_target_selection_rate"]
        ),
        "exploit_mean_alice_reward": float(
            exploit["mean_alice_reward"]
        ),
        "exploit_mean_eve_public_utility": float(
            exploit["mean_eve_public_utility"]
        ),
        "exploit_target_selection_rate": float(
            exploit["passive_target_selection_rate"]
        ),
        "state_continuity": dict(execution["state_continuity"]),
        "information_boundary": dict(execution["information_boundary"]),
    }


def _aggregate_execution_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["true_hypothesis_label_for_scoring_only"]),
            str(row["condition"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for (true_label, condition), values in sorted(groups.items()):
        result.append({
            "true_hypothesis_label": true_label,
            "condition": condition,
            "runs": len(values),
            "mean_follow_turns": float(np.mean([
                int(row["follow_turns"]) for row in values
            ])),
            "trigger_rate": float(np.mean([
                bool(row["exploit_triggered"]) for row in values
            ])),
            "mean_exploit_alice_reward": float(np.mean([
                float(row["exploit_mean_alice_reward"])
                for row in values
            ])),
            "mean_exploit_eve_public_utility": float(np.mean([
                float(row["exploit_mean_eve_public_utility"])
                for row in values
            ])),
            "mean_exploit_target_selection_rate": float(np.mean([
                float(row["exploit_target_selection_rate"])
                for row in values
            ])),
            "selected_plan_counts": dict(Counter(
                str(row["selected_plan_id"]) for row in values
            )),
        })
    return result


def _paired_contrasts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["true_hypothesis_label_for_scoring_only"]),
            int(row["execution_seed"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    conditions = sorted({
        str(row["condition"])
        for row in rows
        if str(row["condition"]) != "posterior_selected"
    })
    result = []
    true_labels = sorted({
        str(row["true_hypothesis_label_for_scoring_only"])
        for row in rows
    })
    seeds = sorted({int(row["execution_seed"]) for row in rows})
    for true_label in true_labels:
        for condition in conditions:
            differences = []
            for seed in seeds:
                selected = by_key.get((
                    true_label,
                    seed,
                    "posterior_selected",
                ))
                control = by_key.get((true_label, seed, condition))
                if selected is None or control is None:
                    continue
                differences.append(
                    float(selected["exploit_mean_eve_public_utility"])
                    - float(control["exploit_mean_eve_public_utility"])
                )
            if differences:
                result.append({
                    "true_hypothesis_label": true_label,
                    "posterior_selected_minus_condition": condition,
                    "paired_runs": len(differences),
                    "mean_eve_public_utility_difference": float(
                        np.mean(differences)
                    ),
                    "differences": list(map(float, differences)),
                })
    return result


def _learner_config_summary(
    config: HistogramCoLearningConfig,
) -> dict[str, Any]:
    return {
        "alice_learner_mode": config.alice_learner_mode,
        "alice_tsallis_inf_estimator": (
            config.alice_tsallis_inf_estimator
        ),
        "alice_tsallis_inf_eta_scale": (
            config.alice_tsallis_inf_eta_scale
        ),
        "alice_exp3_eta_c": config.alice_exp3_eta_c,
        "structured_exploration_mode": config.structured_exploration_mode,
        "game_mode": config.game_mode,
        "requested_pairs_per_turn": config.requested_pairs_per_turn,
    }
