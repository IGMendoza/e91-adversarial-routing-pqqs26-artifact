"""Matched controller execution with classical and quantum endpoint detectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    validate_endpoint_evidence_bank_against_case,
)
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
    MatchedClassicalQuantumEndpointDetectors,
)
from .stackelberg_bandit import (
    RouteObservationDesign,
    run_stackelberg_bandit_follow_runtime,
)
from .stackelberg_controller import (
    ControllerPrimeExploitSchedule,
    evaluate_post_follow_controller_plans,
    passive_exploit_trigger,
)
from .stackelberg_controller_plans import (
    ControllerPrimeExploitPlan,
    build_controller_prime_exploit_plans,
    controller_plan_screening_representatives,
    controller_target_family_representatives,
)


ENDPOINT_DETECTION_STUDY_SCHEMA = (
    "histogram_stackelberg_endpoint_detection_study.v1"
)
ENDPOINT_DETECTION_RUN_SCHEMA = (
    "histogram_stackelberg_endpoint_detection_run.v1"
)
INTEGRITY_ONLY_ATTACK_TYPES = ("memory_degradation",)


@dataclass(frozen=True)
class EndpointDetectionStudyConfig:
    """Frozen planning, commissioning, and held-out detector contract."""

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
    record_every: int = 100
    commissioning_draws_per_route: int = 32
    detector_false_alarm_level: float = 0.01
    detector_commissioning_pseudocount: float = 0.5
    detector_change_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)

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
        if self.commissioning_draws_per_route < 2:
            raise ValueError(
                "commissioning_draws_per_route must be at least two"
            )
        BinomialEvidenceDetectorConfig(
            horizon=self.schedule.total_turns,
            model_false_alarm_level=self.detector_false_alarm_level,
            commissioning_pseudocount=(
                self.detector_commissioning_pseudocount
            ),
            change_fractions=self.detector_change_fractions,
        )

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
        return {
            **asdict(self),
            "planning_seeds": list(map(int, self.planning_seeds)),
            "execution_seeds": list(map(int, self.execution_seeds)),
            "detector_change_fractions": list(
                map(float, self.detector_change_fractions)
            ),
            "schedule": self.schedule.to_dict(),
            "detector_config": self.detector_config.to_dict(),
        }


def run_matched_endpoint_detection_study(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    study_config: EndpointDetectionStudyConfig = (
        EndpointDetectionStudyConfig()
    ),
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Select feedback-only controller plans and detect on held-out counts."""

    labels = tuple(map(str, learner_configs))
    if len(labels) != 2:
        raise ValueError(
            "the matched endpoint study currently requires two hypotheses"
        )
    if (
        payoff_bank.route_ids != evidence_bank.route_ids
        or payoff_bank.action_ids != evidence_bank.action_ids
    ):
        raise ValueError("payoff and endpoint evidence bank axes differ")
    if evidence_bank.draws_per_cell < (
        study_config.commissioning_draws_per_route
    ):
        raise ValueError(
            "endpoint evidence bank is too short for commissioning"
        )
    if observation_design.route_ids != payoff_bank.route_ids:
        raise ValueError("observation design and bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError(
            "controller study requires exact passive route separation"
        )
    base_config = learner_configs[labels[0]]
    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        payoff_bank,
        case,
        base_config,
        selection_digest_blake2b_128=selection_digest,
    )
    validate_endpoint_evidence_bank_against_case(
        evidence_bank,
        case,
        base_config,
        selection_digest_blake2b_128=selection_digest,
    )

    all_plans = build_controller_prime_exploit_plans(
        case,
        payoff_bank,
        base_config,
        selection_digest_blake2b_128=selection_digest,
    )
    target_representatives = controller_target_family_representatives(
        all_plans,
        maximum_families=study_config.target_family_count,
    )
    target_families = {
        plan.target_family_key for plan in target_representatives
    }
    plans = controller_plan_screening_representatives(tuple(
        plan
        for plan in all_plans
        if plan.target_family_key in target_families
    ))
    plan_by_id = {plan.plan_id: plan for plan in plans}
    integrity_all_plans = build_controller_prime_exploit_plans(
        case,
        payoff_bank,
        base_config,
        prime_attack_types=INTEGRITY_ONLY_ATTACK_TYPES,
        exploit_attack_types=INTEGRITY_ONLY_ATTACK_TYPES,
        selection_digest_blake2b_128=selection_digest,
    )
    integrity_plans = controller_plan_screening_representatives(tuple(
        plan
        for plan in integrity_all_plans
        if plan.target_family_key in target_families
    ))
    if not integrity_plans:
        raise ValueError(
            "selected target families have no integrity-only controller plan"
        )
    integrity_plan_by_id = {
        plan.plan_id: plan for plan in integrity_plans
    }
    commissioning = _commissioning_evidence_by_route(
        evidence_bank,
        stop=study_config.commissioning_draws_per_route,
    )

    rows: list[dict[str, Any]] = []
    planning_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(labels) * len(study_config.execution_seeds)
    completed = 0
    for true_label in labels:
        true_config = learner_configs[true_label]
        for execution_seed in map(int, study_config.execution_seeds):
            follow = run_stackelberg_bandit_follow_runtime(
                case,
                payoff_bank,
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
                selection_digest_blake2b_128=selection_digest,
            )
            library, plan_rows = evaluate_post_follow_controller_plans(
                case,
                payoff_bank,
                true_config,
                follow_runtime=follow,
                plans=plans,
                schedule=study_config.schedule,
                observation_design=observation_design,
                seeds=study_config.planning_seeds,
                selection_digest_blake2b_128=selection_digest,
            )
            integrity_library, integrity_plan_rows = (
                evaluate_post_follow_controller_plans(
                    case,
                    payoff_bank,
                    true_config,
                    follow_runtime=follow,
                    plans=integrity_plans,
                    schedule=study_config.schedule,
                    observation_design=observation_design,
                    seeds=study_config.planning_seeds,
                    selection_digest_blake2b_128=selection_digest,
                )
            )
            posterior = follow.tracker.posterior_by_label()
            decision = library.select(posterior)
            selected_plan = plan_by_id[str(decision["selected_plan_id"])]
            integrity_decision = integrity_library.select(posterior)
            integrity_selected_plan = integrity_plan_by_id[
                str(integrity_decision["selected_plan_id"])
            ]
            planning_rows.extend({
                **dict(row),
                "true_hypothesis_label_for_scoring_only": true_label,
                "execution_seed": execution_seed,
                "action_contract": "full_controller_actions",
            } for row in plan_rows)
            planning_rows.extend({
                **dict(row),
                "true_hypothesis_label_for_scoring_only": true_label,
                "execution_seed": execution_seed,
                "action_contract": "integrity_only_memory_degradation",
            } for row in integrity_plan_rows)

            for (
                condition,
                attack_enabled,
                execution_plan,
                execution_decision,
                action_contract,
            ) in (
                (
                    "controller_attack",
                    True,
                    selected_plan,
                    decision,
                    "full_controller_actions",
                ),
                (
                    "integrity_only_controller_attack",
                    True,
                    integrity_selected_plan,
                    integrity_decision,
                    "integrity_only_memory_degradation",
                ),
                (
                    "clean_no_attack_control",
                    False,
                    selected_plan,
                    decision,
                    "no_attack",
                ),
            ):
                detectors = (
                    MatchedClassicalQuantumEndpointDetectors.from_commissioning(
                        commissioning,
                        study_config.detector_config,
                    )
                )
                alice = follow.clone_true_alice(
                    continuation_seed=execution_seed
                )
                run = run_matched_endpoint_detector_continuation(
                    case,
                    payoff_bank,
                    evidence_bank,
                    true_config,
                    execution_plan,
                    schedule=study_config.schedule,
                    alice=alice,
                    observation_design=observation_design,
                    detectors=detectors,
                    seed=execution_seed,
                    attack_enabled=attack_enabled,
                    condition=condition,
                    selection_digest_blake2b_128=selection_digest,
                )
                rows.append({
                    **run,
                    "true_hypothesis_label_for_scoring_only": true_label,
                    "action_contract": action_contract,
                    "follow_turns": int(follow.report["turns"]),
                    "follow_decision": dict(follow.report["decision"]),
                    "online_posterior": posterior,
                    "selected_plan_decision": execution_decision,
                })

            completed += 1
            if progress_callback is not None:
                progress_callback(
                    f"endpoint detector {completed}/{total}: "
                    f"{true_label}, seed={execution_seed}"
                )

    return {
        "schema": ENDPOINT_DETECTION_STUDY_SCHEMA,
        "case_id": payoff_bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "study_config": study_config.to_dict(),
        "learner_configs": {
            label: _learner_config_summary(learner_configs[label])
            for label in labels
        },
        "observation_design": observation_design.to_dict(),
        "payoff_bank": payoff_bank.to_summary(),
        "endpoint_evidence_bank": evidence_bank.to_summary(),
        "plan_surface": {
            "complete_plan_count": len(all_plans),
            "screened_plan_count": len(plans),
            "screened_plan_ids": [plan.plan_id for plan in plans],
            "integrity_only_complete_plan_count": len(
                integrity_all_plans
            ),
            "integrity_only_screened_plan_count": len(integrity_plans),
            "integrity_only_screened_plan_ids": [
                plan.plan_id for plan in integrity_plans
            ],
            "integrity_only_semantics": (
                "Prime and Exploit use memory degradation only; physical "
                "delivery remains available while endpoint QBER/CHSH "
                "integrity changes"
            ),
            "selected_target_route_families": [
                list(plan.target_family_key)
                for plan in target_representatives
            ],
        },
        "run_count": len(rows),
        "planning_row_count": len(planning_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "planning_rows": planning_rows,
        "aggregates": _aggregate_rows(rows),
        "information_boundary": {
            "planning": (
                "passive route signatures and public scalar rewards; "
                "feedback payoff bank only"
            ),
            "detector_common": (
                "selected route, requested/delivered pair counts, and public "
                "accepted/rejected service outcome"
            ),
            "detector_quantum_increment": (
                "key errors conditioned on key trials and same/different "
                "outcomes conditioned on each CHSH setting-pair trial count"
            ),
            "execution": "held-out endpoint evidence audit split",
            "excluded_online": [
                "Eve action identity",
                "Eve target identity",
                "target-active simulator flag",
                "expected physics probabilities",
                "payoff-bank audit outcomes",
                "counterfactual route rewards",
            ],
        },
        "claim_boundary": (
            "The detector comparison is a matched endpoint feature ablation "
            "on compiled quantum-network traces. It measures the conditional "
            "value of QBER/CHSH counts beyond physical delivery counts; it is "
            "not a complete classical packet-network simulation."
        ),
    }


def run_matched_endpoint_detector_continuation(
    case: HistogramGameCase,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    config: HistogramCoLearningConfig,
    plan: ControllerPrimeExploitPlan,
    *,
    schedule: ControllerPrimeExploitSchedule,
    alice: PlayerLearner,
    observation_design: RouteObservationDesign,
    detectors: MatchedClassicalQuantumEndpointDetectors,
    seed: int,
    attack_enabled: bool,
    condition: str,
    selection_digest_blake2b_128: str | None = None,
) -> dict[str, Any]:
    """Continue one Alice state while both detectors observe the same trace."""

    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(payoff_bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        payoff_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    validate_endpoint_evidence_bank_against_case(
        evidence_bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    if (
        payoff_bank.route_ids != evidence_bank.route_ids
        or payoff_bank.action_ids != evidence_bank.action_ids
    ):
        raise ValueError("payoff and endpoint evidence bank axes differ")
    if alice.player != "alice" or alice.action_ids != payoff_bank.route_ids:
        raise ValueError("continuation Alice does not match the bank")
    if alice.sample_calls != alice.update_calls:
        raise RuntimeError(
            "continuation Alice has an outstanding selected action"
        )
    if observation_design.route_ids != payoff_bank.route_ids:
        raise ValueError("observation design and bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError(
            "passive controller execution requires exact route separation"
        )
    _validate_plan(plan, payoff_bank)

    no_attack_index = payoff_bank.action_ids.index("no_attack")
    stream = CommonCellOutcomeStream(
        route_count=len(payoff_bank.route_ids),
        action_count=len(payoff_bank.action_ids),
        draws_per_cell=evidence_bank.draws_per_cell,
        seed=int(seed),
    )
    prime_rng = np.random.default_rng([int(seed), 613])
    prime_indices = np.asarray(plan.prime_action_indices, dtype=np.int64)
    prime_probabilities = np.asarray(plan.prime_probabilities, dtype=float)
    target_set = set(plan.target_route_indices)
    passive_target_observations: list[int] = []
    trigger: dict[str, Any] | None = None
    phase_sums: dict[str, dict[str, float]] = {}
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    route_counts = np.zeros(len(payoff_bank.route_ids), dtype=np.int64)
    detector_trace: list[dict[str, Any]] = []
    starting_sample_calls = int(alice.sample_calls)
    starting_update_calls = int(alice.update_calls)
    started = time.perf_counter()

    for turn_index in range(schedule.total_turns):
        phase = "prime" if turn_index < schedule.prime_turns else "exploit"
        if phase == "prime":
            planned_action_index = int(
                prime_rng.choice(
                    prime_indices,
                    p=prime_probabilities,
                )
            )
        else:
            if trigger is None:
                trigger = passive_exploit_trigger(
                    passive_target_observations,
                    threshold=plan.profitable_target_mass_threshold,
                    z=schedule.trigger_confidence_z,
                )
            planned_action_index = (
                plan.exploit_action_index
                if bool(trigger["triggered"])
                else no_attack_index
            )
        action_index = (
            planned_action_index if attack_enabled else no_attack_index
        )

        route_index = alice.sample()
        draw_index = stream.next_draw(route_index, action_index)
        observation = evidence_bank.audit.draw(
            route_index,
            action_index,
            draw_index,
            requested_pair_count=evidence_bank.requested_pairs_per_draw,
        )
        reward = float(observation["alice_reward"])
        alice.update(reward)
        detector_rows = detectors.update(
            route_index,
            observation["endpoint_evidence"],
        )

        signature = observation_design.signature_for_route(route_index)
        observed_routes = observation_design.routes_for_signature(signature)
        if len(observed_routes) != 1:
            raise AssertionError(
                "declared exact reservation signature became ambiguous"
            )
        observed_route_index = int(observed_routes[0])
        observed_target = int(observed_route_index in target_set)
        if phase == "prime":
            passive_target_observations.append(observed_target)

        action_cost = (
            float(payoff_bank.action_costs[action_index])
            if attack_enabled
            else 0.0
        )
        values = phase_sums.setdefault(
            phase,
            {
                "alice_reward": 0.0,
                "eve_public_utility": 0.0,
                "target_selection": 0.0,
            },
        )
        values["alice_reward"] += reward
        values["eve_public_utility"] += 1.0 - reward - action_cost
        values["target_selection"] += observed_target
        phase_counts[phase] += 1
        action_counts[action_index] += 1
        route_counts[route_index] += 1

        step = turn_index + 1
        newly_alarmed = any(
            bool(row["newly_alarmed"])
            for row in detector_rows.values()
        )
        if (
            step == 1
            or step == schedule.prime_turns
            or step == schedule.total_turns
            or step % schedule.record_every == 0
            or newly_alarmed
        ):
            detector_trace.append({
                "continuation_turn": step,
                "phase": phase,
                "delivery_only_log_e_value": float(
                    detector_rows["classical_delivery_only"][
                        "log_e_value"
                    ]
                ),
                "classical_service_log_e_value": float(
                    detector_rows["classical_service_baseline"][
                        "log_e_value"
                    ]
                ),
                "quantum_log_e_value": float(
                    detector_rows["quantum_endpoint_integrity"][
                        "log_e_value"
                    ]
                ),
                "delivery_only_alarmed": bool(
                    detector_rows["classical_delivery_only"]["alarmed"]
                ),
                "classical_service_alarmed": bool(
                    detector_rows["classical_service_baseline"]["alarmed"]
                ),
                "quantum_alarmed": bool(
                    detector_rows["quantum_endpoint_integrity"]["alarmed"]
                ),
                "alice_target_policy_mass_offline": float(
                    np.sum(alice.policy()[list(target_set)])
                ),
            })

    if trigger is None:
        trigger = passive_exploit_trigger(
            passive_target_observations,
            threshold=plan.profitable_target_mass_threshold,
            z=schedule.trigger_confidence_z,
        )
    phase_metrics = {
        phase: {
            "turns": int(phase_counts[phase]),
            "mean_alice_reward": (
                values["alice_reward"] / phase_counts[phase]
            ),
            "mean_eve_public_utility": (
                values["eve_public_utility"] / phase_counts[phase]
            ),
            "target_selection_rate": (
                values["target_selection"] / phase_counts[phase]
            ),
        }
        for phase, values in phase_sums.items()
    }
    diagnostics = detectors.diagnostics()
    return {
        "schema": ENDPOINT_DETECTION_RUN_SCHEMA,
        "condition": str(condition),
        "attack_enabled": bool(attack_enabled),
        "seed": int(seed),
        "case_id": payoff_bank.case_id,
        "turns": schedule.total_turns,
        "schedule": schedule.to_dict(),
        "plan": plan.to_dict(),
        "phase_metrics": phase_metrics,
        "passive_exploit_trigger": trigger,
        "action_multiset": {
            payoff_bank.action_ids[index]: int(count)
            for index, count in sorted(action_counts.items())
        },
        "route_counts": route_counts.tolist(),
        "detectors": diagnostics,
        "detector_outcomes": {
            "delivery_only_alarm_turn": detectors.delivery_only.alarm_turn,
            "classical_alarm_turn": detectors.classical.alarm_turn,
            "quantum_alarm_turn": detectors.quantum.alarm_turn,
            "delivery_only_alarm_phase": _alarm_phase(
                detectors.delivery_only.alarm_turn,
                schedule,
            ),
            "classical_alarm_phase": _alarm_phase(
                detectors.classical.alarm_turn,
                schedule,
            ),
            "quantum_alarm_phase": _alarm_phase(
                detectors.quantum.alarm_turn,
                schedule,
            ),
            "quantum_minus_classical_alarm_turn": (
                detectors.quantum.alarm_turn
                - detectors.classical.alarm_turn
                if (
                    detectors.quantum.alarm_turn is not None
                    and detectors.classical.alarm_turn is not None
                )
                else None
            ),
            "quantum_minus_delivery_only_alarm_turn": (
                detectors.quantum.alarm_turn
                - detectors.delivery_only.alarm_turn
                if (
                    detectors.quantum.alarm_turn is not None
                    and detectors.delivery_only.alarm_turn is not None
                )
                else None
            ),
        },
        "detector_trace": detector_trace,
        "ending_alice_policy": alice.policy().tolist(),
        "ending_alice_policy_statistics": (
            policy_distribution_statistics(alice.policy())
        ),
        "state_continuity": {
            "starting_alice_sample_calls": starting_sample_calls,
            "starting_alice_update_calls": starting_update_calls,
            "continued_turns": schedule.total_turns,
            "ending_alice_sample_calls": int(alice.sample_calls),
            "ending_alice_update_calls": int(alice.update_calls),
            "exact_post_follow_state_continued": True,
            "rebuilt_from_uniform_prior": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "information_boundary": diagnostics["information_boundary"],
    }


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


def _validate_plan(
    plan: ControllerPrimeExploitPlan,
    bank: HistogramPayoffBank,
) -> None:
    if bank.action_ids[plan.exploit_action_index] != plan.exploit_action_id:
        raise ValueError("controller exploit action does not match the bank")
    if tuple(
        bank.action_ids[index] for index in plan.prime_action_indices
    ) != plan.prime_action_ids:
        raise ValueError("controller Prime actions do not match the bank")
    if tuple(
        bank.route_ids[index] for index in plan.target_route_indices
    ) != plan.target_route_ids:
        raise ValueError("controller target routes do not match the bank")


def _alarm_phase(
    alarm_turn: int | None,
    schedule: ControllerPrimeExploitSchedule,
) -> str | None:
    if alarm_turn is None:
        return None
    return "prime" if alarm_turn <= schedule.prime_turns else "exploit"


def _learner_config_summary(
    config: HistogramCoLearningConfig,
) -> dict[str, Any]:
    return {
        "alice_learner_mode": config.alice_learner_mode,
        "alice_tsallis_inf_estimator": (
            config.alice_tsallis_inf_estimator
        ),
        "alice_tsallis_inf_eta_scale": config.alice_tsallis_inf_eta_scale,
        "alice_exp3_eta_c": config.alice_exp3_eta_c,
        "structured_exploration_mode": config.structured_exploration_mode,
        "requested_pairs_per_turn": config.requested_pairs_per_turn,
        "game_mode": config.game_mode,
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    labels = sorted({
        str(row["true_hypothesis_label_for_scoring_only"])
        for row in rows
    })
    for label in labels:
        label_rows = [
            row
            for row in rows
            if row["true_hypothesis_label_for_scoring_only"] == label
        ]
        condition_rows = {
            condition: [
                row for row in label_rows
                if row["condition"] == condition
            ]
            for condition in (
                "controller_attack",
                "integrity_only_controller_attack",
                "clean_no_attack_control",
            )
        }
        clean = condition_rows["clean_no_attack_control"]
        attack_conditions: dict[str, Any] = {}
        for condition in (
            "controller_attack",
            "integrity_only_controller_attack",
        ):
            attacked = condition_rows[condition]
            attack_conditions[condition] = {
                "runs": len(attacked),
                "delivery_only_detection_count": sum(
                    row["detector_outcomes"][
                        "delivery_only_alarm_turn"
                    ] is not None
                    for row in attacked
                ),
                "classical_detection_count": sum(
                    row["detector_outcomes"][
                        "classical_alarm_turn"
                    ] is not None
                    for row in attacked
                ),
                "quantum_detection_count": sum(
                    row["detector_outcomes"][
                        "quantum_alarm_turn"
                    ] is not None
                    for row in attacked
                ),
                "quantum_only_detection_count": sum(
                    row["detector_outcomes"][
                        "quantum_alarm_turn"
                    ] is not None
                    and row["detector_outcomes"][
                        "classical_alarm_turn"
                    ] is None
                    for row in attacked
                ),
                "quantum_beyond_delivery_only_count": sum(
                    row["detector_outcomes"][
                        "quantum_alarm_turn"
                    ] is not None
                    and row["detector_outcomes"][
                        "delivery_only_alarm_turn"
                    ] is None
                    for row in attacked
                ),
                "mean_delivery_only_alarm_turn": _optional_mean(
                    row["detector_outcomes"]["delivery_only_alarm_turn"]
                    for row in attacked
                ),
                "mean_classical_alarm_turn": _optional_mean(
                    row["detector_outcomes"]["classical_alarm_turn"]
                    for row in attacked
                ),
                "mean_quantum_alarm_turn": _optional_mean(
                    row["detector_outcomes"]["quantum_alarm_turn"]
                    for row in attacked
                ),
                "quantum_leading_channel_counts": dict(Counter(
                    (
                        row["detectors"]["quantum_endpoint_integrity"]
                        ["leading_alternative"] or {}
                    ).get("channel", "none")
                    for row in attacked
                )),
                "selected_plan_counts": dict(Counter(
                    row["plan"]["plan_id"] for row in attacked
                )),
            }
        aggregates[label] = {
            "run_count": len(label_rows),
            "clean_delivery_only_false_alarm_count": sum(
                row["detector_outcomes"][
                    "delivery_only_alarm_turn"
                ] is not None
                for row in clean
            ),
            "clean_classical_false_alarm_count": sum(
                row["detector_outcomes"]["classical_alarm_turn"] is not None
                for row in clean
            ),
            "clean_quantum_false_alarm_count": sum(
                row["detector_outcomes"]["quantum_alarm_turn"] is not None
                for row in clean
            ),
            "attack_conditions": attack_conditions,
        }
    return aggregates


def _optional_mean(values: Sequence[Any] | Any) -> float | None:
    retained = [
        float(value)
        for value in values
        if value is not None
    ]
    return float(np.mean(retained)) if retained else None
