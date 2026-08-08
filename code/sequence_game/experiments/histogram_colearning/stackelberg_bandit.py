"""Observation and decision layer for a physically situated Stackelberg Eve.

This module does not disclose Alice's live weights or simulator target bit.
It formalizes what a set of compromised edge/node resources can distinguish,
updates finite learner hypotheses from observed route/reward behavior, and
selects among a declared library of leader plans by posterior expected value.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp

from .calibration_bank import (
    HistogramPayoffBank,
    validate_bank_against_case,
)
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import (
    ConditionableScalarFeedbackLearner,
    PlayerLearner,
)
from .learning import build_histogram_alice_learner
from .prime_probe import (
    PrimeProbePlan,
    PrimeProbeSchedule,
    run_ordered_prime_probe,
)
from .structure import route_resource_embedding


STACKELBERG_BANDIT_OBSERVATION_SCHEMA = (
    "histogram_stackelberg_bandit_observation.v1"
)
STACKELBERG_BANDIT_HYPOTHESIS_SCHEMA = (
    "histogram_stackelberg_bandit_hypotheses.v1"
)
STACKELBERG_BANDIT_DECISION_SCHEMA = (
    "histogram_stackelberg_bandit_decision.v1"
)
STACKELBERG_BANDIT_FOLLOW_SCHEMA = (
    "histogram_stackelberg_bandit_follow.v1"
)


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=float)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive)))


@dataclass(frozen=True)
class RouteObservationDesign:
    """Minimum local resource set separating all distinguishable route classes."""

    route_ids: tuple[str, ...]
    sensor_labels: tuple[str, ...]
    incidence: np.ndarray
    selected_sensor_indices: tuple[int, ...]
    route_class_ids: tuple[int, ...]
    route_classes: tuple[tuple[int, ...], ...]
    optimization_objective: float
    allowed_sensor_kinds: tuple[str, ...] = ("edge", "node")

    def __post_init__(self) -> None:
        matrix = np.asarray(self.incidence, dtype=bool)
        if matrix.shape != (len(self.sensor_labels), len(self.route_ids)):
            raise ValueError("route observation incidence has the wrong shape")
        if len(self.route_class_ids) != len(self.route_ids):
            raise ValueError("route class IDs do not match the route universe")
        if sorted(index for group in self.route_classes for index in group) != list(
            range(len(self.route_ids))
        ):
            raise ValueError("route classes do not partition the route universe")
        if any(
            not 0 <= index < len(self.sensor_labels)
            for index in self.selected_sensor_indices
        ):
            raise ValueError("selected sensor index is outside the sensor universe")

    @property
    def selected_sensor_labels(self) -> tuple[str, ...]:
        return tuple(
            self.sensor_labels[index]
            for index in self.selected_sensor_indices
        )

    @property
    def selected_incidence(self) -> np.ndarray:
        return np.asarray(
            self.incidence[list(self.selected_sensor_indices), :],
            dtype=bool,
        )

    @property
    def exact_route_observation(self) -> bool:
        return all(len(group) == 1 for group in self.route_classes)

    def signature_for_route(self, route_index: int) -> tuple[int, ...]:
        index = int(route_index)
        if not 0 <= index < len(self.route_ids):
            raise IndexError("route index is outside the observation design")
        return tuple(
            map(int, self.selected_incidence[:, index])
        )

    def routes_for_signature(
        self,
        signature: Sequence[int],
    ) -> tuple[int, ...]:
        observed = tuple(int(value) for value in signature)
        if len(observed) != len(self.selected_sensor_indices):
            raise ValueError("observation signature has the wrong length")
        if any(value not in {0, 1} for value in observed):
            raise ValueError("observation signature must be binary")
        return tuple(
            route_index
            for route_index in range(len(self.route_ids))
            if self.signature_for_route(route_index) == observed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACKELBERG_BANDIT_OBSERVATION_SCHEMA,
            "route_ids": list(self.route_ids),
            "sensor_labels": list(self.sensor_labels),
            "selected_sensor_indices": list(self.selected_sensor_indices),
            "selected_sensor_labels": list(self.selected_sensor_labels),
            "selected_sensor_count": len(self.selected_sensor_indices),
            "route_classes": [
                [self.route_ids[index] for index in group]
                for group in self.route_classes
            ],
            "route_class_count": len(self.route_classes),
            "exact_route_observation": self.exact_route_observation,
            "optimization": {
                "problem": (
                    "minimum binary edge/node sensor set that separates every "
                    "route pair distinguishable by the complete resource incidence"
                ),
                "solver": "scipy.optimize.milp",
                "objective": self.optimization_objective,
                "payoff_or_physics_used": False,
                "allowed_sensor_kinds": list(self.allowed_sensor_kinds),
            },
            "online_information": {
                "alice_weights_disclosed": False,
                "simulator_target_active_disclosed": False,
                "route_identity_directly_disclosed": False,
                "route_identity_inferred_only_when_sensor_signature_is_unique": True,
            },
        }


def minimum_route_observation_design(
    case: HistogramGameCase,
    *,
    allowed_sensor_kinds: Sequence[str] = ("edge", "node"),
) -> RouteObservationDesign:
    """Solve the exact minimum local-resource route-separation problem."""

    embedding = route_resource_embedding(case)
    kinds = tuple(dict.fromkeys(str(kind).lower() for kind in allowed_sensor_kinds))
    if not kinds or any(kind not in {"edge", "node"} for kind in kinds):
        raise ValueError(
            "allowed_sensor_kinds must contain 'edge', 'node', or both"
        )
    all_labels = tuple(embedding.raw_labels[1:])
    selected_columns = tuple(
        index
        for index, label in enumerate(all_labels)
        if label.split(":", 1)[0] in kinds
    )
    labels = tuple(all_labels[index] for index in selected_columns)
    incidence = np.asarray(
        embedding.raw_matrix[:, 1:][:, selected_columns].T,
        dtype=bool,
    )
    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    if incidence.shape != (len(labels), len(route_ids)):
        raise AssertionError("route-resource embedding shape changed")

    full_signatures = [
        tuple(map(int, incidence[:, route_index]))
        for route_index in range(len(route_ids))
    ]
    signature_to_routes: dict[tuple[int, ...], list[int]] = {}
    for route_index, signature in enumerate(full_signatures):
        signature_to_routes.setdefault(signature, []).append(route_index)
    route_classes = tuple(
        tuple(indices)
        for _signature, indices in sorted(
            signature_to_routes.items(),
            key=lambda row: row[1][0],
        )
    )
    route_class_ids = [0] * len(route_ids)
    for class_index, indices in enumerate(route_classes):
        for route_index in indices:
            route_class_ids[route_index] = class_index

    representatives = [group[0] for group in route_classes]
    pair_differences: list[np.ndarray] = []
    for left_position, left in enumerate(representatives):
        for right in representatives[left_position + 1:]:
            difference = np.logical_xor(
                incidence[:, left],
                incidence[:, right],
            ).astype(float)
            if not np.any(difference):
                raise AssertionError(
                    "distinct full resource signatures were not separable"
                )
            pair_differences.append(difference)

    if not pair_differences:
        selected: tuple[int, ...] = ()
        objective = 0.0
    else:
        constraints = np.stack(pair_differences, axis=0)
        sensor_count = len(labels)
        tie_scale = 1.0 / max(4.0, 4.0 * sensor_count * sensor_count)
        costs = np.ones(sensor_count, dtype=float) + tie_scale * np.arange(
            sensor_count,
            dtype=float,
        )
        result = milp(
            c=costs,
            integrality=np.ones(sensor_count, dtype=np.int8),
            bounds=Bounds(
                np.zeros(sensor_count, dtype=float),
                np.ones(sensor_count, dtype=float),
            ),
            constraints=LinearConstraint(
                constraints,
                np.ones(len(pair_differences), dtype=float),
                np.full(len(pair_differences), np.inf),
            ),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                "minimum route-observation sensor MILP failed: "
                f"{result.message}"
            )
        selected = tuple(
            map(int, np.flatnonzero(np.asarray(result.x) > 0.5))
        )
        objective = float(result.fun)

    design = RouteObservationDesign(
        route_ids=route_ids,
        sensor_labels=labels,
        incidence=incidence,
        selected_sensor_indices=selected,
        route_class_ids=tuple(route_class_ids),
        route_classes=route_classes,
        optimization_objective=objective,
        allowed_sensor_kinds=kinds,
    )
    selected_signatures = {
        design.signature_for_route(group[0])
        for group in route_classes
    }
    if len(selected_signatures) != len(route_classes):
        raise AssertionError(
            "minimum observation design failed to separate route classes"
        )
    return design


class LearnerHypothesisTracker:
    """Bayesian conditioning over declared Alice learner implementations.

    This exact finite tracker requires the selected route and Alice's public
    reward to be identifiable from Eve's declared observation channel. It does
    not read learner weights. Each hypothesis is advanced by the same
    selected-action/scalar-reward update that the hypothesized learner would
    perform, conditional on the observed action.
    """

    def __init__(
        self,
        hypotheses: Mapping[str, PlayerLearner],
        *,
        prior: Mapping[str, float] | None = None,
    ) -> None:
        if not hypotheses:
            raise ValueError("at least one Alice learner hypothesis is required")
        self.hypotheses = dict(hypotheses)
        if any(learner.player != "alice" for learner in self.hypotheses.values()):
            raise ValueError("all Stackelberg-bandit hypotheses must be Alice learners")
        action_id_sets = {
            learner.action_ids for learner in self.hypotheses.values()
        }
        if len(action_id_sets) != 1:
            raise ValueError("Alice hypotheses must share one route action universe")
        for learner in self.hypotheses.values():
            if not isinstance(
                learner.learner,
                ConditionableScalarFeedbackLearner,
            ):
                raise TypeError(
                    f"{learner.metadata['algorithm']} cannot be conditioned "
                    "on observed actions"
                )
        self.labels = tuple(self.hypotheses)
        self.action_ids = next(iter(action_id_sets))
        if prior is None:
            probabilities = np.full(
                len(self.labels),
                1.0 / len(self.labels),
                dtype=float,
            )
        else:
            if set(prior) != set(self.labels):
                raise ValueError("hypothesis prior labels do not match hypotheses")
            probabilities = np.asarray(
                [float(prior[label]) for label in self.labels],
                dtype=float,
            )
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
            ):
                raise ValueError(
                    "hypothesis prior probabilities must be finite and positive"
                )
            probabilities /= probabilities.sum()
        self._log_weights = np.log(probabilities)
        self.observations = 0
        self.cumulative_log_predictive_probability = 0.0

    @property
    def posterior(self) -> np.ndarray:
        normalized = self._log_weights - logsumexp(self._log_weights)
        return np.exp(normalized)

    def posterior_by_label(self) -> dict[str, float]:
        return {
            label: float(probability)
            for label, probability in zip(self.labels, self.posterior)
        }

    def predictive_route_policy(self) -> np.ndarray:
        policies = np.stack(
            [
                self.hypotheses[label].policy(
                    turn=self.observations + 1,
                )
                for label in self.labels
            ],
            axis=0,
        )
        return self.posterior @ policies

    def observe_route_reward(
        self,
        route_index: int,
        alice_reward: float,
    ) -> dict[str, Any]:
        route = int(route_index)
        reward = float(alice_reward)
        if not 0 <= route < len(self.action_ids):
            raise IndexError("observed route index is outside the route universe")
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("observed Alice reward must lie in [0, 1]")

        action_probabilities = np.asarray(
            [
                self.hypotheses[label].policy(
                    turn=self.observations + 1,
                )[route]
                for label in self.labels
            ],
            dtype=float,
        )
        if np.any(action_probabilities <= 0.0):
            raise FloatingPointError(
                "a learner hypothesis assigned zero mass to an observed route"
            )
        predictive = float(
            np.dot(self.posterior, action_probabilities)
        )
        self._log_weights += np.log(action_probabilities)
        self._log_weights -= logsumexp(self._log_weights)
        for label, expected_probability in zip(
            self.labels,
            action_probabilities,
        ):
            actual_probability = self.hypotheses[
                label
            ].condition_on_observation(route, reward)
            if not math.isclose(
                actual_probability,
                float(expected_probability),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise AssertionError(
                    "conditioned learner probability changed between "
                    "likelihood and update"
                )
        self.observations += 1
        self.cumulative_log_predictive_probability += math.log(predictive)
        return {
            "schema": STACKELBERG_BANDIT_HYPOTHESIS_SCHEMA,
            "observation": self.observations,
            "route_index": route,
            "route_id": self.action_ids[route],
            "alice_reward": reward,
            "predictive_probability": predictive,
            "hypothesis_action_probabilities": {
                label: float(probability)
                for label, probability in zip(
                    self.labels,
                    action_probabilities,
                )
            },
            "posterior": self.posterior_by_label(),
        }

    def observe_sensor_signature(
        self,
        design: RouteObservationDesign,
        signature: Sequence[int],
        alice_reward: float,
    ) -> dict[str, Any]:
        if design.route_ids != self.action_ids:
            raise ValueError(
                "route observation design and learner hypotheses differ"
            )
        matches = design.routes_for_signature(signature)
        if len(matches) != 1:
            raise ValueError(
                "the local resource observation identifies a route class of "
                f"size {len(matches)}, so exact learner-state conditioning is "
                "not available under this sensor set"
            )
        result = self.observe_route_reward(matches[0], alice_reward)
        return {
            **result,
            "sensor_signature": list(map(int, signature)),
            "selected_sensor_labels": list(
                design.selected_sensor_labels
            ),
            "route_identity_source": (
                "unique local resource-incidence signature"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACKELBERG_BANDIT_HYPOTHESIS_SCHEMA,
            "hypothesis_labels": list(self.labels),
            "route_ids": list(self.action_ids),
            "observations": self.observations,
            "posterior": self.posterior_by_label(),
            "predictive_route_policy": self.predictive_route_policy().tolist(),
            "cumulative_log_predictive_probability": (
                self.cumulative_log_predictive_probability
            ),
            "information_boundary": {
                "alice_weights_disclosed": False,
                "candidate_learner_state_maintained_by_eve": True,
                "requires_identifiable_selected_route": True,
                "requires_alice_public_reward": True,
                "simulator_audit_fields_used": False,
            },
        }


@dataclass
class StackelbergBanditFollowRuntime:
    """Private orchestration state after one passive follow phase.

    The serializable report is Eve's admissible observation record. The live
    Alice object and conditioned learner hypotheses remain private experiment
    state so a subsequent plan can continue from the exact post-follow
    cumulative learner state instead of rebuilding Alice from a uniform prior.
    """

    report: dict[str, Any]
    true_alice: PlayerLearner
    tracker: LearnerHypothesisTracker
    outcome_stream: CommonCellOutcomeStream

    def clone_true_alice(
        self,
        *,
        continuation_seed: int | None = None,
    ) -> PlayerLearner:
        """Clone the actual post-follow Alice for a matched control branch."""

        return _clone_post_follow_learner(
            self.true_alice,
            continuation_seed=continuation_seed,
        )

    def clone_hypothesis(
        self,
        label: str,
        *,
        continuation_seed: int | None = None,
    ) -> PlayerLearner:
        """Clone one Eve-maintained post-follow learner hypothesis."""

        if label not in self.tracker.hypotheses:
            raise KeyError(f"unknown learner hypothesis {label!r}")
        return _clone_post_follow_learner(
            self.tracker.hypotheses[label],
            continuation_seed=continuation_seed,
        )


def _clone_post_follow_learner(
    learner: PlayerLearner,
    *,
    continuation_seed: int | None,
) -> PlayerLearner:
    if learner.sample_calls != learner.update_calls:
        raise RuntimeError(
            "cannot branch a learner with an outstanding selected action"
        )
    cloned = deepcopy(learner)
    if continuation_seed is not None:
        if not hasattr(cloned.learner, "rng"):
            raise TypeError(
                f"{cloned.metadata['algorithm']} does not expose a continuation RNG"
            )
        cloned.learner.rng = np.random.default_rng(int(continuation_seed))
    return cloned


def run_stackelberg_bandit_follow_runtime(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    *,
    true_alice_config: HistogramCoLearningConfig,
    hypothesis_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    turns: int,
    seed: int,
    true_hypothesis_label: str | None = None,
    record_every: int = 100,
    posterior_thresholds: Sequence[float] = (0.9, 0.95, 0.99),
    decision_posterior_threshold: float | None = 0.99,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> StackelbergBanditFollowRuntime:
    """Run Eve's passive follow phase against one actual Alice learner.

    Alice receives ordinary no-attack public service rewards from the
    histogram feedback bank. Eve observes only the binary activity signature
    of the declared local edge/node sensors plus that public reward. The
    selected route is reconstructed only when that signature is unique.
    Candidate learner states are then advanced by exact conditional updates;
    Alice's live weights, policy, random draw, and simulator audit fields are
    never inputs to the tracker.
    """

    active_turns = int(turns)
    active_seed = int(seed)
    if active_turns < 1:
        raise ValueError("follow turns must be positive")
    if record_every < 1:
        raise ValueError("record_every must be positive")
    if observation_design.route_ids != bank.route_ids:
        raise ValueError("observation design and payoff bank routes differ")
    if not observation_design.exact_route_observation:
        raise ValueError(
            "exact finite learner-state tracking requires a sensor design "
            "that separates every route in the declared action universe"
        )
    thresholds = tuple(sorted(set(map(float, posterior_thresholds))))
    if any(
        not math.isfinite(value) or not 0.5 < value < 1.0
        for value in thresholds
    ):
        raise ValueError("posterior thresholds must lie strictly in (0.5, 1)")
    if (
        decision_posterior_threshold is not None
        and (
            not math.isfinite(float(decision_posterior_threshold))
            or not 0.5 < float(decision_posterior_threshold) < 1.0
        )
    ):
        raise ValueError(
            "decision_posterior_threshold must lie strictly in (0.5, 1)"
        )
    if not hypothesis_configs:
        raise ValueError("at least one learner hypothesis is required")
    if (
        true_hypothesis_label is not None
        and true_hypothesis_label not in hypothesis_configs
    ):
        raise ValueError(
            "true_hypothesis_label is outside the declared hypothesis set"
        )

    validation_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(bank.metadata["selection_digest_blake2b_128"])
    )
    active_true_config = replace(
        true_alice_config,
        turns=active_turns,
        seed=active_seed,
    )
    validate_bank_against_case(
        bank,
        case,
        active_true_config,
        selection_digest_blake2b_128=validation_digest,
    )
    prior = np.full(len(bank.route_ids), 1.0 / len(bank.route_ids), dtype=float)
    true_implementation, true_metadata = build_histogram_alice_learner(
        case=case,
        config=active_true_config,
        prior=prior,
    )
    true_alice = PlayerLearner(
        player="alice",
        action_ids=bank.route_ids,
        learner=true_implementation,
        metadata=true_metadata,
    )

    hypotheses: dict[str, PlayerLearner] = {}
    for label, config in hypothesis_configs.items():
        active_config = replace(config, turns=active_turns)
        validate_bank_against_case(
            bank,
            case,
            active_config,
            selection_digest_blake2b_128=validation_digest,
        )
        implementation, metadata = build_histogram_alice_learner(
            case=case,
            config=active_config,
            prior=prior,
        )
        hypotheses[str(label)] = PlayerLearner(
            player="alice",
            action_ids=bank.route_ids,
            learner=implementation,
            metadata=metadata,
        )
    tracker = LearnerHypothesisTracker(hypotheses)

    try:
        no_attack_index = bank.action_ids.index("no_attack")
    except ValueError as exc:
        raise ValueError(
            "Stackelberg-bandit following requires a no_attack action"
        ) from exc
    stream = CommonCellOutcomeStream(
        route_count=len(bank.route_ids),
        action_count=len(bank.action_ids),
        draws_per_cell=bank.draws_per_cell,
        seed=active_seed,
    )
    route_counts = np.zeros(len(bank.route_ids), dtype=np.int64)
    threshold_turns: dict[str, int | None] = {
        f"{value:g}": None for value in thresholds
    }
    trajectory: list[dict[str, Any]] = []
    reward_sum = 0.0
    started = time.perf_counter()
    decision: dict[str, Any] | None = None
    executed_turns = 0

    for turn_index in range(active_turns):
        step = turn_index + 1
        route_index = true_alice.sample()
        draw_index = stream.next_draw(route_index, no_attack_index)
        observation = bank.feedback.draw(
            route_index,
            no_attack_index,
            draw_index,
        )
        reward = float(observation["alice_reward"])
        true_alice.update(reward)
        signature = observation_design.signature_for_route(route_index)
        update = tracker.observe_sensor_signature(
            observation_design,
            signature,
            reward,
        )
        route_counts[route_index] += 1
        reward_sum += reward
        executed_turns = step
        posterior = update["posterior"]
        if true_hypothesis_label is not None:
            true_mass = float(posterior[true_hypothesis_label])
            for value in thresholds:
                key = f"{value:g}"
                if threshold_turns[key] is None and true_mass >= value:
                    threshold_turns[key] = step

        threshold_crossed = (
            true_hypothesis_label is not None
            and step in {
                value for value in threshold_turns.values()
                if value is not None
            }
        )
        if (
            step == 1
            or step == active_turns
            or step % record_every == 0
            or threshold_crossed
        ):
            trajectory.append({
                "turn": step,
                "mean_alice_public_reward": reward_sum / step,
                "posterior": dict(posterior),
                "posterior_entropy_nats": _entropy(
                    np.asarray(list(posterior.values()), dtype=float)
                ),
                "posterior_modal_hypothesis": max(
                    posterior,
                    key=posterior.__getitem__,
                ),
                "posterior_modal_mass": max(map(float, posterior.values())),
                "empirical_route_distribution": (
                    route_counts / step
                ).tolist(),
            })
        modal_label = max(posterior, key=posterior.__getitem__)
        modal_mass = float(posterior[modal_label])
        if (
            decision_posterior_threshold is not None
            and modal_mass >= float(decision_posterior_threshold)
        ):
            decision = {
                "turn": step,
                "selected_hypothesis_label": modal_label,
                "selected_hypothesis_posterior": modal_mass,
                "posterior_threshold": float(
                    decision_posterior_threshold
                ),
                "posterior_error_mass_within_declared_hypothesis_set": (
                    1.0 - modal_mass
                ),
                "stopping_rule": (
                    "first turn at which the modal learner hypothesis reaches "
                    "the declared posterior threshold"
                ),
            }
            if not trajectory or int(trajectory[-1]["turn"]) != step:
                trajectory.append({
                    "turn": step,
                    "mean_alice_public_reward": reward_sum / step,
                    "posterior": dict(posterior),
                    "posterior_entropy_nats": _entropy(
                        np.asarray(list(posterior.values()), dtype=float)
                    ),
                    "posterior_modal_hypothesis": modal_label,
                    "posterior_modal_mass": modal_mass,
                    "empirical_route_distribution": (
                        route_counts / step
                    ).tolist(),
                })
        if progress_callback is not None and (
            step == 1
            or step == active_turns
            or step % max(record_every, 10_000) == 0
        ):
            progress_callback(step)
        if decision is not None:
            break

    elapsed = time.perf_counter() - started
    result = {
        "schema": STACKELBERG_BANDIT_FOLLOW_SCHEMA,
        "case_id": bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "turns": executed_turns,
        "maximum_follow_turns": active_turns,
        "seed": active_seed,
        "elapsed_seconds": elapsed,
        "turns_per_second": (
            executed_turns / elapsed if elapsed > 0.0 else None
        ),
        "true_alice_learner": dict(true_alice.metadata),
        "true_hypothesis_label_for_scoring_only": true_hypothesis_label,
        "hypothesis_learner_metadata": {
            label: dict(learner.metadata)
            for label, learner in hypotheses.items()
        },
        "observation_design": observation_design.to_dict(),
        "no_attack_action_index": no_attack_index,
        "no_attack_action_id": bank.action_ids[no_attack_index],
        "decision": decision,
        "stopped_at_decision_threshold": decision is not None,
        "posterior_threshold_first_crossing_turn": threshold_turns,
        "tracker": tracker.to_dict(),
        "mean_alice_public_reward": reward_sum / executed_turns,
        "true_alice_empirical_strategy_for_scoring_only": (
            route_counts / executed_turns
        ).tolist(),
        "trajectory": trajectory,
        "information_boundary": {
            "follow_intervention": "no_attack",
            "eve_observes": [
                "declared local resource activity signature",
                "Alice public scalar service reward",
            ],
            "route_identity_source": (
                "unique signature from the declared compromised sensor set"
            ),
            "alice_live_policy_or_weights_disclosed": False,
            "alice_private_sampling_randomness_disclosed": False,
            "simulator_target_active_used": False,
            "simulator_eve_private_instrument_record_used": False,
            "audit_split_used": False,
            "feedback_split_used": True,
            "true_hypothesis_label_used_online": False,
        },
        "claim_boundary": (
            "This is a passive strong-local-access follow phase. It measures "
            "learner identifiability from declared controller/resource "
            "telemetry; it is not yet the subsequent prime/probe/exploit "
            "policy or a limited-sensor belief filter."
        ),
    }
    if true_hypothesis_label is not None:
        result["final_true_hypothesis_posterior"] = float(
            tracker.posterior_by_label()[true_hypothesis_label]
        )
        result["final_hypothesis_identified_correctly"] = bool(
            max(
                tracker.posterior_by_label(),
                key=tracker.posterior_by_label().__getitem__,
            )
            == true_hypothesis_label
        )
    return StackelbergBanditFollowRuntime(
        report=result,
        true_alice=true_alice,
        tracker=tracker,
        outcome_stream=stream,
    )


def run_stackelberg_bandit_follow_identification(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    *,
    true_alice_config: HistogramCoLearningConfig,
    hypothesis_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: RouteObservationDesign,
    turns: int,
    seed: int,
    true_hypothesis_label: str | None = None,
    record_every: int = 100,
    posterior_thresholds: Sequence[float] = (0.9, 0.95, 0.99),
    decision_posterior_threshold: float | None = 0.99,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run passive identification and return its serializable report."""

    return run_stackelberg_bandit_follow_runtime(
        case,
        bank,
        true_alice_config=true_alice_config,
        hypothesis_configs=hypothesis_configs,
        observation_design=observation_design,
        turns=turns,
        seed=seed,
        true_hypothesis_label=true_hypothesis_label,
        record_every=record_every,
        posterior_thresholds=posterior_thresholds,
        decision_posterior_threshold=decision_posterior_threshold,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
        progress_callback=progress_callback,
    ).report


@dataclass(frozen=True)
class LearnerConditionedPlanLibrary:
    """Finite leader-plan values evaluated under each learner hypothesis."""

    hypothesis_labels: tuple[str, ...]
    plan_ids: tuple[str, ...]
    expected_eve_values: np.ndarray
    evaluation_standard_errors: np.ndarray | None = None
    evaluation_counts: np.ndarray | None = None
    value_semantics: str = "expected Eve public utility"
    source_split: str = "feedback"

    def __post_init__(self) -> None:
        values = np.asarray(self.expected_eve_values, dtype=float)
        if values.shape != (
            len(self.hypothesis_labels),
            len(self.plan_ids),
        ):
            raise ValueError("learner-conditioned plan value table has wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("learner-conditioned plan values must be finite")
        if len(set(self.hypothesis_labels)) != len(self.hypothesis_labels):
            raise ValueError("hypothesis labels must be unique")
        if not self.plan_ids or len(set(self.plan_ids)) != len(self.plan_ids):
            raise ValueError("plan IDs must be nonempty and unique")
        if self.evaluation_standard_errors is not None:
            errors = np.asarray(self.evaluation_standard_errors, dtype=float)
            if (
                errors.shape != values.shape
                or not np.all(np.isfinite(errors))
                or np.any(errors < 0.0)
            ):
                raise ValueError(
                    "plan evaluation standard errors have the wrong shape "
                    "or range"
                )
        if self.evaluation_counts is not None:
            counts = np.asarray(self.evaluation_counts)
            if (
                counts.shape != values.shape
                or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 1)
            ):
                raise ValueError(
                    "plan evaluation counts have the wrong shape or range"
                )
        if not str(self.value_semantics).strip():
            raise ValueError("plan value semantics must be explicit")
        if self.source_split not in {"feedback", "audit", "fresh"}:
            raise ValueError("unknown plan-library source split")

    def select(
        self,
        posterior: Mapping[str, float],
    ) -> dict[str, Any]:
        if set(posterior) != set(self.hypothesis_labels):
            raise ValueError("posterior labels do not match the plan library")
        probabilities = np.asarray(
            [float(posterior[label]) for label in self.hypothesis_labels],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not math.isclose(
                float(probabilities.sum()),
                1.0,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        ):
            raise ValueError("hypothesis posterior must lie on the simplex")
        values = np.asarray(self.expected_eve_values, dtype=float)
        posterior_values = probabilities @ values
        selected = int(np.argmax(posterior_values))
        return {
            "schema": STACKELBERG_BANDIT_DECISION_SCHEMA,
            "selected_plan_index": selected,
            "selected_plan_id": self.plan_ids[selected],
            "selected_posterior_expected_eve_value": float(
                posterior_values[selected]
            ),
            "posterior_expected_eve_values": {
                plan_id: float(value)
                for plan_id, value in zip(self.plan_ids, posterior_values)
            },
            "posterior": {
                label: float(probability)
                for label, probability in zip(
                    self.hypothesis_labels,
                    probabilities,
                )
            },
            "selection_rule": (
                "maximize posterior expected Eve value over the declared "
                "finite plan library"
            ),
            "value_semantics": self.value_semantics,
            "source_split": self.source_split,
            "selected_plan_evaluation_standard_errors_by_hypothesis": (
                {
                    label: float(error)
                    for label, error in zip(
                        self.hypothesis_labels,
                        np.asarray(
                            self.evaluation_standard_errors,
                            dtype=float,
                        )[:, selected],
                    )
                }
                if self.evaluation_standard_errors is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACKELBERG_BANDIT_DECISION_SCHEMA,
            "hypothesis_labels": list(self.hypothesis_labels),
            "plan_ids": list(self.plan_ids),
            "expected_eve_values": np.asarray(
                self.expected_eve_values,
                dtype=float,
            ).tolist(),
            "evaluation_standard_errors": (
                np.asarray(
                    self.evaluation_standard_errors,
                    dtype=float,
                ).tolist()
                if self.evaluation_standard_errors is not None
                else None
            ),
            "evaluation_counts": (
                np.asarray(self.evaluation_counts, dtype=int).tolist()
                if self.evaluation_counts is not None
                else None
            ),
            "value_semantics": self.value_semantics,
            "source_split": self.source_split,
        }


def evaluate_learner_conditioned_prime_probe_plans(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    *,
    hypothesis_configs: Mapping[str, HistogramCoLearningConfig],
    plans: Sequence[PrimeProbePlan],
    schedule: PrimeProbeSchedule,
    seeds: Sequence[int],
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[LearnerConditionedPlanLibrary, list[dict[str, Any]]]:
    """Evaluate a finite Prime-and-Probe library for each Alice hypothesis.

    These are planning-model evaluations on the feedback split. The reserved
    audit split is not read. The returned table therefore supports Eve's
    posterior plan choice, while a subsequent held-out execution is still
    required to score the selected policy.
    """

    labels = tuple(map(str, hypothesis_configs))
    plan_values = tuple(plans)
    seed_values = tuple(map(int, seeds))
    if len(labels) < 2:
        raise ValueError(
            "learner-conditioned planning requires at least two hypotheses"
        )
    if len(plan_values) < 2:
        raise ValueError(
            "learner-conditioned planning requires at least two leader plans"
        )
    if len(seed_values) < 2 or len(set(seed_values)) != len(seed_values):
        raise ValueError("plan evaluation requires at least two unique seeds")
    plan_ids = tuple(plan.exploit_action_id for plan in plan_values)
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("Prime-and-Probe exploit action IDs must be unique")

    sample_values = np.empty(
        (len(labels), len(plan_values), len(seed_values)),
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    total = sample_values.size
    completed = 0
    for hypothesis_index, label in enumerate(labels):
        config = hypothesis_configs[label]
        for plan_index, plan in enumerate(plan_values):
            for seed_index, seed in enumerate(seed_values):
                active_config = replace(
                    config,
                    turns=schedule.total_turns,
                    seed=seed,
                )
                run = run_ordered_prime_probe(
                    case,
                    bank,
                    active_config,
                    plan,
                    schedule=schedule,
                    seed=seed,
                    selection_digest_blake2b_128=(
                        selection_digest_blake2b_128
                    ),
                    include_audit_scoring=False,
                )
                value = float(
                    run["phase_metrics"]["exploit"][
                        "mean_eve_public_utility"
                    ]
                )
                sample_values[
                    hypothesis_index,
                    plan_index,
                    seed_index,
                ] = value
                rows.append({
                    "hypothesis_label": label,
                    "alice_learner": dict(run["alice_learner"]),
                    "plan_id": plan.exploit_action_id,
                    "plan": plan.to_dict(),
                    "seed": seed,
                    "value_semantics": (
                        "mean Eve public utility during the declared exploit "
                        "window, including the selected action cost"
                    ),
                    "exploit_window_eve_public_utility": value,
                    "exploit_window_alice_public_reward": float(
                        run["phase_metrics"]["exploit"][
                            "mean_alice_reward"
                        ]
                    ),
                    "probe_triggered": bool(
                        run["probe_trigger"]["triggered"]
                    ),
                    "planning_split": "feedback",
                    "audit_expected_alice_reward": (
                        run["audit_expected_alice_reward"]
                    ),
                    "audit_split_used": bool(
                        run["information_boundary"][
                            "audit_split_read_for_offline_scoring"
                        ]
                    ),
                })
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        f"plan evaluation {completed}/{total}: "
                        f"{label}, {plan.exploit_action_id}, seed={seed}"
                    )

    means = np.mean(sample_values, axis=2)
    standard_errors = np.std(
        sample_values,
        axis=2,
        ddof=1,
    ) / math.sqrt(len(seed_values))
    counts = np.full(
        means.shape,
        len(seed_values),
        dtype=np.int64,
    )
    return (
        LearnerConditionedPlanLibrary(
            hypothesis_labels=labels,
            plan_ids=plan_ids,
            expected_eve_values=means,
            evaluation_standard_errors=standard_errors,
            evaluation_counts=counts,
            value_semantics=(
                "mean Eve public utility during the declared Prime-and-Probe "
                "exploit window, including action cost"
            ),
            source_split="feedback",
        ),
        rows,
    )
