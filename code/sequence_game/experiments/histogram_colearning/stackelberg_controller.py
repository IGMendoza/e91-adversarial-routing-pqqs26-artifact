"""Passive-observation Stackelberg controller planning and execution.

The strong local-access controller follows Alice through reservation-resource
signatures. Active physical operations are reserved for priming alternative
routes and exploiting a target route family. No intercept/resend or projective
occupancy probe is used by this runtime.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .stackelberg_bandit import (
    LearnerConditionedPlanLibrary,
    RouteObservationDesign,
    StackelbergBanditFollowRuntime,
)
from .stackelberg_controller_plans import ControllerPrimeExploitPlan


CONTROLLER_RUN_SCHEMA = "histogram_stackelberg_controller_run.v1"
CONTROLLER_SCHEDULE_SCHEMA = "histogram_stackelberg_controller_schedule.v1"


@dataclass(frozen=True)
class ControllerPrimeExploitSchedule:
    """Finite continuation schedule after the passive follow decision."""

    prime_turns: int
    exploit_turns: int
    record_every: int = 100
    trigger_confidence_z: float = 1.959963984540054

    def __post_init__(self) -> None:
        if self.prime_turns < 1 or self.exploit_turns < 1:
            raise ValueError("Prime and Exploit phases must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if (
            not math.isfinite(self.trigger_confidence_z)
            or self.trigger_confidence_z <= 0.0
        ):
            raise ValueError(
                "trigger_confidence_z must be finite and positive"
            )

    @property
    def total_turns(self) -> int:
        return self.prime_turns + self.exploit_turns

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROLLER_SCHEDULE_SCHEMA,
            **asdict(self),
            "total_turns": self.total_turns,
        }


def run_controller_prime_exploit_continuation(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: ControllerPrimeExploitPlan,
    *,
    schedule: ControllerPrimeExploitSchedule,
    alice: PlayerLearner,
    observation_design: RouteObservationDesign,
    outcome_stream: CommonCellOutcomeStream,
    seed: int,
    source_split: str = "feedback",
    condition: str = "passive_controller_prime_exploit",
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Continue one exact post-follow Alice state through Prime and Exploit.

    ``alice`` is mutated in place. Planning branches should pass a cloned
    hypothesis state; the actual execution branch should pass the live Alice
    object returned by :func:`run_stackelberg_bandit_follow_runtime`.
    """

    active_seed = int(seed)
    active_config = replace(
        config,
        turns=schedule.total_turns,
        seed=active_seed,
    )
    validation_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        bank,
        case,
        active_config,
        selection_digest_blake2b_128=validation_digest,
    )
    if alice.player != "alice" or alice.action_ids != bank.route_ids:
        raise ValueError("continuation Alice does not match the payoff bank")
    if alice.sample_calls != alice.update_calls:
        raise RuntimeError(
            "continuation Alice has an outstanding selected action"
        )
    if observation_design.route_ids != bank.route_ids:
        raise ValueError("observation design and payoff bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError(
            "passive controller execution requires exact route separation"
        )
    if (
        outcome_stream.route_count != len(bank.route_ids)
        or outcome_stream.action_count != len(bank.action_ids)
        or outcome_stream.draws_per_cell != bank.draws_per_cell
    ):
        raise ValueError("outcome stream dimensions do not match the bank")
    outcome_block = _outcome_block(bank, source_split)
    _validate_plan_against_bank(plan, bank)

    no_attack_index = bank.action_ids.index("no_attack")
    prime_indices = np.asarray(plan.prime_action_indices, dtype=np.int64)
    prime_probabilities = np.asarray(plan.prime_probabilities, dtype=float)
    prime_rng = np.random.default_rng([active_seed, 613])
    starting_sample_calls = int(alice.sample_calls)
    starting_update_calls = int(alice.update_calls)
    starting_policy = alice.policy()
    target_set = set(plan.target_route_indices)
    joint_counts = np.zeros(
        (len(bank.route_ids), len(bank.action_ids)),
        dtype=np.int64,
    )
    phase_counts: Counter[str] = Counter()
    phase_sums: dict[str, dict[str, float]] = {}
    action_counts: Counter[int] = Counter()
    passive_target_observations: list[int] = []
    trajectory: list[dict[str, Any]] = []
    trigger: dict[str, Any] | None = None
    started = time.perf_counter()

    for turn_index in range(schedule.total_turns):
        phase = "prime" if turn_index < schedule.prime_turns else "exploit"
        if phase == "prime":
            action_index = int(
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
            action_index = (
                plan.exploit_action_index
                if bool(trigger["triggered"])
                else no_attack_index
            )

        route_index = alice.sample()
        draw_index = outcome_stream.next_draw(route_index, action_index)
        observation = outcome_block.draw(
            route_index,
            action_index,
            draw_index,
        )
        reward = float(observation["alice_reward"])
        alice.update(reward)

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

        action_cost = float(bank.action_costs[action_index])
        eve_public_utility = 1.0 - reward - action_cost
        phase_counts[phase] += 1
        phase_row = phase_sums.setdefault(
            phase,
            {
                "alice_reward": 0.0,
                "eve_public_utility": 0.0,
                "passive_target_observation": 0.0,
            },
        )
        phase_row["alice_reward"] += reward
        phase_row["eve_public_utility"] += eve_public_utility
        phase_row["passive_target_observation"] += observed_target
        action_counts[action_index] += 1
        joint_counts[route_index, action_index] += 1

        step = turn_index + 1
        if progress_callback is not None and (
            step == 1
            or step == schedule.total_turns
            or step % max(schedule.record_every, 10_000) == 0
        ):
            progress_callback(step, phase)
        if (
            step == 1
            or step == schedule.prime_turns
            or step == schedule.total_turns
            or step % schedule.record_every == 0
        ):
            policy = alice.policy()
            trajectory.append({
                "continuation_turn": step,
                "absolute_alice_update_count": int(alice.update_calls),
                "phase": phase,
                "alice_target_family_policy_mass_offline": float(
                    np.sum(policy[list(target_set)])
                ),
                "alice_policy_effective_action_fraction": (
                    policy_distribution_statistics(policy)[
                        "effective_action_fraction"
                    ]
                ),
                "passive_target_selection_rate": (
                    float(np.mean(passive_target_observations))
                    if passive_target_observations else None
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
            "passive_target_selection_rate": (
                values["passive_target_observation"] / phase_counts[phase]
            ),
        }
        for phase, values in phase_sums.items()
    }
    action_multiset = {
        bank.action_ids[index]: int(count)
        for index, count in sorted(action_counts.items())
    }
    elapsed = time.perf_counter() - started
    return {
        "schema": CONTROLLER_RUN_SCHEMA,
        "condition": str(condition),
        "case_id": bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "seed": active_seed,
        "source_split": source_split,
        "turns": schedule.total_turns,
        "elapsed_seconds": elapsed,
        "turns_per_second": (
            schedule.total_turns / elapsed if elapsed > 0.0 else None
        ),
        "schedule": schedule.to_dict(),
        "plan": plan.to_dict(),
        "alice_learner": dict(alice.metadata),
        "starting_alice_policy": starting_policy.tolist(),
        "ending_alice_policy": alice.policy().tolist(),
        "ending_alice_empirical_strategy": (
            alice.empirical_strategy().tolist()
        ),
        "phase_metrics": phase_metrics,
        "passive_exploit_trigger": trigger,
        "action_multiset": action_multiset,
        "joint_action_counts_offline": joint_counts.tolist(),
        "trajectory": trajectory,
        "state_continuity": {
            "starting_alice_sample_calls": starting_sample_calls,
            "starting_alice_update_calls": starting_update_calls,
            "continued_turns": schedule.total_turns,
            "ending_alice_sample_calls": int(alice.sample_calls),
            "ending_alice_update_calls": int(alice.update_calls),
            "rebuilt_from_uniform_prior": False,
            "exact_post_follow_state_continued": True,
        },
        "information_boundary": {
            "access_regime": "compromised_reservation_controller",
            "online_route_identity_source": (
                "unique passive local reservation-resource signature"
            ),
            "active_occupancy_probe_used": False,
            "projective_logical_state_probe_used": False,
            "intercept_resend_probe_used": False,
            "alice_policy_or_weights_disclosed_online": False,
            "simulator_target_active_used_online": False,
            "audit_split_used_online": source_split == "audit",
            "physical_prime_and_exploit_actions_only": True,
            "offline_diagnostics": [
                "Alice target-family policy mass",
                "joint route/action counts",
            ],
        },
    }


def evaluate_post_follow_controller_plans(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    follow_runtime: StackelbergBanditFollowRuntime,
    plans: Sequence[ControllerPrimeExploitPlan],
    schedule: ControllerPrimeExploitSchedule,
    observation_design: RouteObservationDesign,
    seeds: Sequence[int],
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[LearnerConditionedPlanLibrary, list[dict[str, Any]]]:
    """Evaluate plans from each exact post-follow hypothesis state.

    The feedback split supplies planning outcomes. Each branch clones the
    conditioned learner sufficient state and receives an independent future
    random stream; the actual Alice execution object and audit split are not
    consumed.
    """

    labels = tuple(follow_runtime.tracker.labels)
    plan_values = tuple(plans)
    seed_values = tuple(map(int, seeds))
    if len(labels) < 2:
        raise ValueError(
            "posterior controller planning requires at least two hypotheses"
        )
    if len(plan_values) < 2:
        raise ValueError(
            "posterior controller planning requires at least two plans"
        )
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("plan evaluation requires at least two unique seeds")
    plan_ids = tuple(plan.plan_id for plan in plan_values)
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("controller plan IDs must be unique")

    samples = np.empty(
        (len(labels), len(plan_values), len(seed_values)),
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    completed = 0
    total = samples.size
    for hypothesis_index, label in enumerate(labels):
        for plan_index, plan in enumerate(plan_values):
            for seed_index, seed in enumerate(seed_values):
                alice = follow_runtime.clone_hypothesis(
                    label,
                    continuation_seed=seed,
                )
                starting_updates = int(alice.update_calls)
                stream = CommonCellOutcomeStream(
                    route_count=len(bank.route_ids),
                    action_count=len(bank.action_ids),
                    draws_per_cell=bank.draws_per_cell,
                    seed=seed,
                )
                run = run_controller_prime_exploit_continuation(
                    case,
                    bank,
                    config,
                    plan,
                    schedule=schedule,
                    alice=alice,
                    observation_design=observation_design,
                    outcome_stream=stream,
                    seed=seed,
                    source_split="feedback",
                    condition="post_follow_feedback_planning",
                    selection_digest_blake2b_128=(
                        selection_digest_blake2b_128
                    ),
                )
                value = float(
                    run["phase_metrics"]["exploit"][
                        "mean_eve_public_utility"
                    ]
                )
                samples[hypothesis_index, plan_index, seed_index] = value
                rows.append({
                    "hypothesis_label": label,
                    "plan_id": plan.plan_id,
                    "plan": plan.to_dict(),
                    "seed": seed,
                    "planning_split": "feedback",
                    "audit_split_used": False,
                    "starting_post_follow_update_count": starting_updates,
                    "ending_update_count": int(alice.update_calls),
                    "exploit_triggered": bool(
                        run["passive_exploit_trigger"]["triggered"]
                    ),
                    "exploit_window_eve_public_utility": value,
                    "exploit_window_alice_public_reward": float(
                        run["phase_metrics"]["exploit"][
                            "mean_alice_reward"
                        ]
                    ),
                    "state_continuity": dict(run["state_continuity"]),
                })
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        f"controller plan evaluation {completed}/{total}: "
                        f"{label}, {plan.plan_id}, seed={seed}"
                    )

    means = np.mean(samples, axis=2)
    standard_errors = np.std(samples, axis=2, ddof=1) / math.sqrt(
        len(seed_values)
    )
    counts = np.full(means.shape, len(seed_values), dtype=np.int64)
    return (
        LearnerConditionedPlanLibrary(
            hypothesis_labels=labels,
            plan_ids=plan_ids,
            expected_eve_values=means,
            evaluation_standard_errors=standard_errors,
            evaluation_counts=counts,
            value_semantics=(
                "mean Eve public utility during the passive-controller "
                "Exploit window, including action cost"
            ),
            source_split="feedback",
        ),
        rows,
    )


def execute_post_follow_controller_decision(
    case: HistogramGameCase,
    execution_bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    follow_runtime: StackelbergBanditFollowRuntime,
    plan_library: LearnerConditionedPlanLibrary,
    plans: Sequence[ControllerPrimeExploitPlan],
    schedule: ControllerPrimeExploitSchedule,
    observation_design: RouteObservationDesign,
    seed: int,
    source_split: str,
    posterior: Mapping[str, float] | None = None,
    condition: str = "posterior_selected_controller",
    selection_digest_blake2b_128: str | None = None,
) -> dict[str, Any]:
    """Select from feedback-only values and execute on held-out/fresh draws.

    The actual post-follow Alice state is cloned exactly so matched execution
    controls can branch from the same cumulative state and future RNG state.
    The plan library must come from the feedback split; audit/fresh outcomes
    are not available until after the plan has been fixed.
    """

    if plan_library.source_split != "feedback":
        raise ValueError(
            "controller execution requires a feedback-only planning library"
        )
    active_posterior = (
        follow_runtime.tracker.posterior_by_label()
        if posterior is None
        else {str(label): float(value) for label, value in posterior.items()}
    )
    decision = plan_library.select(active_posterior)
    plan_values = tuple(plans)
    plan_by_id = {plan.plan_id: plan for plan in plan_values}
    if len(plan_by_id) != len(plan_values):
        raise ValueError("controller execution plans must have unique IDs")
    try:
        selected_plan = plan_by_id[str(decision["selected_plan_id"])]
    except KeyError as exc:
        raise ValueError(
            "selected controller plan is absent from the execution plan set"
        ) from exc

    alice = follow_runtime.clone_true_alice()
    stream = CommonCellOutcomeStream(
        route_count=len(execution_bank.route_ids),
        action_count=len(execution_bank.action_ids),
        draws_per_cell=execution_bank.draws_per_cell,
        seed=int(seed),
    )
    execution = run_controller_prime_exploit_continuation(
        case,
        execution_bank,
        config,
        selected_plan,
        schedule=schedule,
        alice=alice,
        observation_design=observation_design,
        outcome_stream=stream,
        seed=int(seed),
        source_split=source_split,
        condition=condition,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
    )
    return {
        "condition": str(condition),
        "planning_decision": decision,
        "execution": execution,
        "control_contract": {
            "planning_source_split": plan_library.source_split,
            "execution_source_split": source_split,
            "audit_or_fresh_outcomes_available_during_plan_selection": False,
            "actual_alice_weights_disclosed_to_controller": False,
            "exact_post_follow_alice_state_cloned_for_matched_branch": True,
            "future_alice_rng_state_matched_across_control_branches": True,
        },
    }


def _outcome_block(
    bank: HistogramPayoffBank,
    source_split: str,
) -> Any:
    if source_split == "feedback":
        return bank.feedback
    if source_split == "audit":
        return bank.audit
    if source_split == "fresh":
        return bank.feedback
    raise ValueError("source_split must be feedback, audit, or fresh")


def _validate_plan_against_bank(
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


def passive_exploit_trigger(
    target_observations: Sequence[int],
    *,
    threshold: float,
    z: float,
) -> dict[str, Any]:
    interval = _wilson_target_mass_interval(target_observations, z=z)
    return {
        "triggered": bool(interval["target_mass_lower"] > float(threshold)),
        "decision_rule": (
            "exploit iff the Wilson lower confidence bound for passive "
            "target-family reservation occupancy exceeds the plan's "
            "feedback-derived profitable target-mass threshold"
        ),
        "profitable_target_mass_threshold": float(threshold),
        "passive_target_mass_interval": interval,
    }


def _wilson_target_mass_interval(
    target_observations: Sequence[int],
    *,
    z: float,
) -> dict[str, float | int]:
    values = np.asarray(target_observations, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.all(np.isin(values, (0.0, 1.0)))
    ):
        raise ValueError(
            "passive target observations must be a nonempty binary vector"
        )
    count = int(values.size)
    successes = int(np.sum(values))
    proportion = successes / count
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / count
    center = (proportion + z2 / (2.0 * count)) / denominator
    half_width = (
        float(z)
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z2 / (4.0 * count * count)
        )
        / denominator
    )
    return {
        "samples": count,
        "target_observations": successes,
        "target_mass_mean": float(proportion),
        "target_mass_lower": float(max(0.0, center - half_width)),
        "target_mass_upper": float(min(1.0, center + half_width)),
        "confidence_z": float(z),
    }
