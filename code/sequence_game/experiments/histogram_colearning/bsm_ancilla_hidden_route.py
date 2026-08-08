"""Hidden-route learner filtering from compiled BSM ancilla observations.

The two-route ancilla smoke can hard-decode a route because each route owns a
distinct repeater. Cernet routes overlap, so that reduction is unavailable.
This module instead maintains a sequential Monte Carlo approximation over
learner-labelled latent route histories. Eve's online record contains only her
chosen XX/ZZ repeater probe, a sampled ancilla hit/no-hit record when the
readout is available, and Alice's public scalar reward.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import combinations
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from sequence_game.corpus.compiled_histogram_backend import (
    eve_move_from_runtime_action,
)

from .bsm_ancilla_follower import (
    BSM_STABILIZERS,
    _ancilla_observation,
    _build_alice,
    _expected_service_invariance,
    _learner_config_summary,
    _probe_actions,
    _validate_matched_physics_configs,
)
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner
from .learning import bell_qualified_rate_utility
from .physics import (
    CACHE_ENTRY_RAM_ESTIMATE_BYTES,
    CACHE_ENTRY_RAM_ESTIMATE_SOURCE,
    HistogramPhysicsPool,
)
from .stackelberg_bandit import minimum_route_observation_design


BSM_ANCILLA_HIDDEN_ROUTE_SCHEMA = (
    "histogram_bsm_ancilla_hidden_route_filter.v1"
)


@dataclass(frozen=True)
class BSMAncillaObservationScenario:
    """Sparse probe and heralded readout-availability control."""

    name: str
    probe_probability: float
    readout_availability_probability: float
    probes_per_turn: int | None = 1

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("scenario name must be nonempty")
        for field_name, value in (
            ("probe_probability", self.probe_probability),
            (
                "readout_availability_probability",
                self.readout_availability_probability,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if (
            self.probes_per_turn is not None
            and self.probes_per_turn < 1
        ):
            raise ValueError(
                "probes_per_turn must be positive or None"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_BSM_ANCILLA_SCENARIOS = (
    BSMAncillaObservationScenario(
        "separating_panel_every_turn",
        1.0,
        1.0,
        None,
    ),
    BSMAncillaObservationScenario(
        "two_probe_panel_every_turn",
        1.0,
        1.0,
        2,
    ),
    BSMAncillaObservationScenario(
        "two_probe_panel_quarter_turns_missed",
        0.25,
        0.75,
        2,
    ),
)


@dataclass(frozen=True)
class BSMAncillaHiddenRouteConfig:
    """Cernet-scale hidden-route follower contract."""

    maximum_follow_turns: int = 80
    particles_per_hypothesis: int = 128
    commissioning_draws_per_route_probe: int = 8
    posterior_threshold: float = 0.99
    dephasing_theta: float = math.pi / 2.0
    stabilizers: tuple[str, ...] = BSM_STABILIZERS
    scenarios: tuple[BSMAncillaObservationScenario, ...] = (
        DEFAULT_BSM_ANCILLA_SCENARIOS
    )
    execution_seeds: tuple[int, ...] = (
        2_026_072_201,
        2_026_072_202,
        2_026_072_203,
    )
    record_every: int = 10

    def __post_init__(self) -> None:
        if self.maximum_follow_turns < 2:
            raise ValueError("maximum_follow_turns must be at least two")
        if self.particles_per_hypothesis < 8:
            raise ValueError(
                "particles_per_hypothesis must be at least eight"
            )
        if self.commissioning_draws_per_route_probe < 2:
            raise ValueError(
                "commissioning_draws_per_route_probe must be at least two"
            )
        if (
            not math.isfinite(self.posterior_threshold)
            or not 0.5 < self.posterior_threshold < 1.0
        ):
            raise ValueError("posterior_threshold must lie in (0.5, 1)")
        if (
            not math.isfinite(self.dephasing_theta)
            or not 0.0 <= self.dephasing_theta <= math.pi / 2.0
        ):
            raise ValueError(
                "dephasing_theta must lie in [0, pi/2]"
            )
        normalized = tuple(str(value).upper() for value in self.stabilizers)
        if (
            not normalized
            or len(set(normalized)) != len(normalized)
            or any(value not in BSM_STABILIZERS for value in normalized)
        ):
            raise ValueError("stabilizers must be unique XX/ZZ labels")
        if (
            not self.scenarios
            or len({row.name for row in self.scenarios})
            != len(self.scenarios)
        ):
            raise ValueError("observation scenarios must be unique")
        if (
            not self.execution_seeds
            or len(set(map(int, self.execution_seeds)))
            != len(self.execution_seeds)
        ):
            raise ValueError("execution_seeds must be nonempty and unique")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stabilizers": list(self.stabilizers),
            "scenarios": [row.to_dict() for row in self.scenarios],
            "execution_seeds": list(map(int, self.execution_seeds)),
        }


class HiddenRouteLearnerParticleFilter:
    """Fully adapted particle filter over learner-labelled route histories.

    Within each learner hypothesis, particles carry exact conditioned learner
    state. The current route is proposed from the learner policy conditioned
    on the available ancilla observation. Label evidence is the corresponding
    particle-mixture predictive probability. Public reward advances every
    sampled learner state, but no uncommissioned reward likelihood is invented.
    """

    def __init__(
        self,
        hypotheses: Mapping[str, PlayerLearner],
        *,
        particles_per_hypothesis: int,
        rng: np.random.Generator,
        prior: Mapping[str, float] | None = None,
    ) -> None:
        if len(hypotheses) < 2:
            raise ValueError(
                "hidden-route filtering requires at least two hypotheses"
            )
        if particles_per_hypothesis < 8:
            raise ValueError(
                "particles_per_hypothesis must be at least eight"
            )
        action_sets = {row.action_ids for row in hypotheses.values()}
        if len(action_sets) != 1:
            raise ValueError(
                "learner hypotheses must share one route universe"
            )
        self.labels = tuple(map(str, hypotheses))
        self.action_ids = next(iter(action_sets))
        self.particles_per_hypothesis = int(
            particles_per_hypothesis
        )
        self.rng = rng
        if prior is None:
            probabilities = np.full(
                len(self.labels),
                1.0 / len(self.labels),
                dtype=float,
            )
        else:
            if set(prior) != set(self.labels):
                raise ValueError("hypothesis prior labels do not match")
            probabilities = np.asarray(
                [float(prior[label]) for label in self.labels],
                dtype=float,
            )
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
            ):
                raise ValueError(
                    "hypothesis prior must be finite and positive"
                )
            probabilities /= probabilities.sum()
        self._log_label_weights = np.log(probabilities)
        self.particles = {
            label: [
                deepcopy(hypotheses[label])
                for _ in range(self.particles_per_hypothesis)
            ]
            for label in self.labels
        }
        self.turns = 0
        self.informative_observations = 0
        self.cumulative_log_measurement_predictive_probability = 0.0

    @property
    def posterior(self) -> np.ndarray:
        normalized = (
            self._log_label_weights
            - logsumexp(self._log_label_weights)
        )
        return np.exp(normalized)

    def posterior_by_label(self) -> dict[str, float]:
        return {
            label: float(probability)
            for label, probability in zip(self.labels, self.posterior)
        }

    def predictive_route_policy(self) -> np.ndarray:
        label_policies = self._label_predictive_policies()
        mixture = np.zeros(len(self.action_ids), dtype=float)
        for label_probability, label in zip(
            self.posterior,
            self.labels,
        ):
            mixture += label_probability * np.mean(
                label_policies[label],
                axis=0,
            )
        return _normalized(mixture, "predictive route policy")

    def observe(
        self,
        *,
        action_id: str | None,
        observed_hit: bool | None,
        commissioning: Mapping[str, Any],
        alice_public_reward: float,
    ) -> dict[str, Any]:
        if observed_hit is not None and action_id is None:
            raise ValueError("an observed hit requires a probe action")
        if observed_hit is None:
            likelihood = np.ones(len(self.action_ids), dtype=float)
        else:
            probabilities = np.asarray(
                commissioning[
                    "binary_hit_probability_by_action_route"
                ][str(action_id)],
                dtype=float,
            )
            if probabilities.shape != (len(self.action_ids),):
                raise ValueError(
                    "commissioned emission vector has the wrong shape"
                )
            likelihood = (
                probabilities if observed_hit else 1.0 - probabilities
            )
        return self.observe_route_likelihood(
            route_likelihood=likelihood,
            observation={
                "action_id": action_id,
                "observed_hit": observed_hit,
            },
            informative=observed_hit is not None,
            alice_public_reward=alice_public_reward,
        )

    def observe_route_likelihood(
        self,
        *,
        route_likelihood: Sequence[float] | np.ndarray,
        observation: Mapping[str, Any],
        informative: bool,
        alice_public_reward: float,
    ) -> dict[str, Any]:
        """Advance once from one commissioned route-likelihood vector."""

        reward = float(alice_public_reward)
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("Alice public reward must lie in [0, 1]")
        likelihood = np.asarray(route_likelihood, dtype=float)
        if (
            likelihood.shape != (len(self.action_ids),)
            or not np.all(np.isfinite(likelihood))
            or np.any(likelihood < 0.0)
            or not np.any(likelihood > 0.0)
        ):
            raise ValueError(
                "route likelihood must be finite, nonnegative, and nonzero"
            )

        old_label_posterior = self.posterior
        label_policies = self._label_predictive_policies()
        label_evidence = np.empty(len(self.labels), dtype=float)
        route_joint = np.zeros(len(self.action_ids), dtype=float)
        particle_evidence_by_label: dict[str, np.ndarray] = {}
        for label_index, label in enumerate(self.labels):
            policies = label_policies[label]
            particle_evidence = policies @ likelihood
            if (
                not np.all(np.isfinite(particle_evidence))
                or np.any(particle_evidence <= 0.0)
            ):
                raise FloatingPointError(
                    "particle measurement evidence is nonpositive"
                )
            particle_evidence_by_label[label] = particle_evidence
            label_evidence[label_index] = float(
                np.mean(particle_evidence)
            )
            route_joint += (
                old_label_posterior[label_index]
                * np.mean(policies, axis=0)
                * likelihood
            )
        measurement_predictive = float(np.sum(route_joint))
        if (
            not math.isfinite(measurement_predictive)
            or measurement_predictive <= 0.0
        ):
            raise FloatingPointError(
                "measurement predictive probability is nonpositive"
            )
        route_posterior = route_joint / measurement_predictive

        self._log_label_weights += np.log(label_evidence)
        self._log_label_weights -= logsumexp(
            self._log_label_weights
        )
        resampling: dict[str, Any] = {}
        for label in self.labels:
            old_particles = self.particles[label]
            policies = label_policies[label]
            particle_evidence = particle_evidence_by_label[label]
            if not informative:
                parent_indices = np.arange(
                    self.particles_per_hypothesis,
                    dtype=np.int64,
                )
            else:
                parent_indices = _systematic_resample(
                    particle_evidence / np.sum(particle_evidence),
                    self.rng,
                )
            new_particles: list[PlayerLearner] = []
            for parent_index in parent_indices:
                index = int(parent_index)
                route_distribution = (
                    policies[index] * likelihood
                    / particle_evidence[index]
                )
                route_index = int(
                    self.rng.choice(
                        len(self.action_ids),
                        p=route_distribution,
                    )
                )
                particle = deepcopy(old_particles[index])
                conditioned_probability = (
                    particle.condition_on_observation(
                        route_index,
                        reward,
                    )
                )
                if not math.isclose(
                    conditioned_probability,
                    float(policies[index, route_index]),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise AssertionError(
                        "particle learner policy changed between proposal "
                        "and conditioning"
                    )
                new_particles.append(particle)
            self.particles[label] = new_particles
            resampling[label] = {
                "unique_parent_fraction": (
                    len(set(map(int, parent_indices)))
                    / self.particles_per_hypothesis
                ),
                "mean_particle_measurement_evidence": float(
                    np.mean(particle_evidence)
                ),
            }

        self.turns += 1
        if informative:
            self.informative_observations += 1
        self.cumulative_log_measurement_predictive_probability += (
            math.log(measurement_predictive)
        )
        return {
            "turn": self.turns,
            "observation": dict(observation),
            "measurement_predictive_probability": (
                measurement_predictive
            ),
            "route_posterior": route_posterior,
            "posterior": self.posterior_by_label(),
            "label_measurement_evidence": {
                label: float(value)
                for label, value in zip(
                    self.labels,
                    label_evidence,
                )
            },
            "resampling": resampling,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "filter": "fully_adapted_sequential_monte_carlo",
            "particles_per_hypothesis": (
                self.particles_per_hypothesis
            ),
            "hypothesis_count": len(self.labels),
            "turns": self.turns,
            "informative_observations": (
                self.informative_observations
            ),
            "posterior": self.posterior_by_label(),
            "predictive_route_policy": (
                self.predictive_route_policy().tolist()
            ),
            "cumulative_log_measurement_predictive_probability": (
                self.cumulative_log_measurement_predictive_probability
            ),
            "approximation_boundary": (
                "learner state is exact within each conditioned particle; "
                "the posterior over latent route histories is a finite-"
                "particle Monte Carlo approximation"
            ),
            "reward_likelihood": (
                "not modeled; public reward advances conditioned learner "
                "states but contributes no invented route-emission density"
            ),
        }

    def _label_predictive_policies(
        self,
    ) -> dict[str, np.ndarray]:
        return {
            label: np.stack(
                [
                    particle.policy(
                        turn=particle.update_calls + 1,
                    )
                    for particle in self.particles[label]
                ],
                axis=0,
            )
            for label in self.labels
        }


def run_bsm_ancilla_hidden_route_study(
    case: HistogramGameCase,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    study_config: BSMAncillaHiddenRouteConfig = (
        BSMAncillaHiddenRouteConfig()
    ),
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run matched dense, sparse, and missed-readout follower conditions."""

    labels = tuple(map(str, learner_configs))
    if len(labels) != 2:
        raise ValueError(
            "the declared Stackelberg follower study requires two learners"
        )
    _validate_matched_physics_configs(learner_configs)
    base_config = learner_configs[labels[0]]
    sensor_nodes = _all_internal_sensor_nodes(case)
    observation_design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node",),
    )
    separating_panel_nodes = tuple(
        str(label).split(":", 1)[1]
        for label in observation_design.selected_sensor_labels
    )
    actions = bsm_probe_actions_for_case(
        case,
        stabilizers=study_config.stabilizers,
        dephasing_theta=study_config.dephasing_theta,
    )
    physics = HistogramPhysicsPool(case, base_config)
    started = time.perf_counter()
    commissioning = _commission_probe_emissions(
        case,
        physics,
        actions,
        draws_per_route_probe=(
            study_config.commissioning_draws_per_route_probe
        ),
    )
    invariance = _expected_service_invariance(
        case,
        physics,
        actions,
    )
    rows: list[dict[str, Any]] = []
    total = (
        len(study_config.scenarios)
        * len(labels)
        * len(study_config.execution_seeds)
    )
    completed = 0
    for scenario in study_config.scenarios:
        for label in labels:
            for seed in map(int, study_config.execution_seeds):
                rows.append(
                    _run_one_hidden_route_filter(
                        case,
                        physics,
                        learner_configs,
                        true_label=label,
                        seed=seed,
                        actions=actions,
                        separating_panel_nodes=separating_panel_nodes,
                        commissioning=commissioning,
                        scenario=scenario,
                        study_config=study_config,
                    )
                )
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        "BSM hidden-route follower "
                        f"{completed}/{total}: {scenario.name}, "
                        f"{label}, seed={seed}"
                    )

    return {
        "schema": BSM_ANCILLA_HIDDEN_ROUTE_SCHEMA,
        "case": case.to_dict(),
        "study_config": study_config.to_dict(),
        "learner_configs": {
            label: _learner_config_summary(config)
            for label, config in learner_configs.items()
        },
        "physics": physics.diagnostics(),
        "probe_cache_plan": bsm_probe_cache_plan(
            case,
            base_config,
            actions,
        ),
        "probe_contract": {
            "sensor_nodes": list(sensor_nodes),
            "minimum_separating_panel_nodes": list(
                separating_panel_nodes
            ),
            "minimum_separating_panel_size": len(
                separating_panel_nodes
            ),
            "action_count": len(actions),
            "actions": [dict(action) for action in actions],
            "allocation": (
                "Bernoulli probe-turn control; the strong-access condition "
                "uses the topology-only minimum separating node panel, while "
                "bounded panels maximize deterministic pattern information "
                "under Eve's current predictive route distribution; XX and "
                "ZZ panels alternate"
            ),
            "observation_states": [
                "not_probed",
                "heralded_missing_readout",
                "observed_hit",
                "observed_no_hit",
            ],
            "route_inference": (
                "particle posterior over latent route histories; no route "
                "is hard-decoded before learner-state conditioning"
            ),
            "panel_likelihood": (
                "exact compatibility with commissioned deterministic local "
                "ancilla-presence patterns after marginalizing heralded "
                "missing panel entries; no product of independent channel "
                "likelihoods"
            ),
        },
        "commissioning": commissioning,
        "expected_service_invariance": invariance,
        "rows": rows,
        "aggregates": _aggregate_rows(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "information_boundary": {
            "eve_observes": [
                "chosen compromised repeater",
                "chosen XX or ZZ stabilizer",
                "whether a scheduled readout is heralded missing",
                "sampled ancilla hit or no-hit when readout is available",
                "Alice public scalar service reward",
            ],
            "audit_only": [
                "Alice selected route",
                "true learner label",
                "counterfactual clean Alice route and reward",
                "physical target-active flag implied by the true route",
            ],
            "excluded_online": [
                "deterministic reservation signature",
                "Alice live policy or weights",
                "Alice private sampling randomness",
                "hard-decoded true route",
                "payoff audit matrix",
            ],
        },
        "scientific_scope": {
            "instrument": "ideal_commuting_bsm_stabilizer_ancilla",
            "physics_backend": "compiled_histogram_transcript",
            "hidden_state_filter": (
                "finite-particle sequential Monte Carlo approximation"
            ),
            "missed_observation_control": (
                "modeled heralded readout availability, not a demonstrated "
                "device-efficiency value"
            ),
            "device_costs": (
                "parallel ancilla preparation, gates, unheralded detector "
                "loss, readout error, and added memory latency are not yet "
                "applied"
            ),
            "security_claim": "none",
        },
        "claim_boundary": (
            "This study tests whether the validated ideal XX/ZZ ancilla "
            "signal remains useful when Cernet routes overlap and probing is "
            "sparse or heralded records are missing. It is not an exact "
            "Bayesian filter over all route histories or a near-term "
            "hardware demonstration."
        ),
    }


def bsm_probe_cache_plan(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exact canonical cache-key count for the declared ancilla surface."""

    unique_by_route = []
    for route in case.routes:
        canonical = {
            json.dumps(
                eve_move_from_runtime_action(
                    dict(action),
                    route,
                )[0].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for action in actions
        }
        unique_by_route.append(len(canonical))
    total = int(sum(unique_by_route))
    maximum = max(unique_by_route, default=0)
    estimated_bytes = total * CACHE_ENTRY_RAM_ESTIMATE_BYTES
    configured = int(config.evaluated_cache_size)
    return {
        "cache_ownership": "one bounded evaluated-histogram LRU per route",
        "route_count": len(case.routes),
        "declared_probe_action_count": len(actions),
        "nominal_route_probe_cells": len(case.routes) * len(actions),
        "canonical_unique_move_count_total": total,
        "canonical_unique_move_count_per_route": unique_by_route,
        "maximum_canonical_moves_on_one_route": maximum,
        "configured_entries_per_route": configured,
        "configured_entry_capacity_all_routes": (
            configured * len(case.routes)
        ),
        "sufficient_without_eviction": configured >= maximum,
        "estimated_resident_bytes": estimated_bytes,
        "estimated_resident_mib": estimated_bytes / 1024**2,
        "estimated_bytes_per_canonical_move": (
            CACHE_ENTRY_RAM_ESTIMATE_BYTES
        ),
        "ram_estimate_source": CACHE_ENTRY_RAM_ESTIMATE_SOURCE,
    }


def bsm_probe_actions_for_case(
    case: HistogramGameCase,
    *,
    stabilizers: Sequence[str] = BSM_STABILIZERS,
    dephasing_theta: float = math.pi / 2.0,
) -> tuple[dict[str, Any], ...]:
    """Declare every internal-repeater XX/ZZ action for one route universe."""

    return _probe_actions(
        _all_internal_sensor_nodes(case),
        stabilizers,
        dephasing_theta,
    )


def _run_one_hidden_route_filter(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    true_label: str,
    seed: int,
    actions: Sequence[dict[str, Any]],
    separating_panel_nodes: Sequence[str],
    commissioning: Mapping[str, Any],
    scenario: BSMAncillaObservationScenario,
    study_config: BSMAncillaHiddenRouteConfig,
) -> dict[str, Any]:
    turns = study_config.maximum_follow_turns
    true_alice = _build_alice(
        case,
        replace(
            learner_configs[true_label],
            turns=turns,
            seed=int(seed),
        ),
    )
    clean_alice = _build_alice(
        case,
        replace(
            learner_configs[true_label],
            turns=turns,
            seed=int(seed),
        ),
    )
    hypotheses = {
        label: _build_alice(
            case,
            replace(config, turns=turns),
        )
        for label, config in learner_configs.items()
    }
    particle_filter = HiddenRouteLearnerParticleFilter(
        hypotheses,
        particles_per_hypothesis=(
            study_config.particles_per_hypothesis
        ),
        rng=np.random.default_rng(int(seed) + 1_300_003),
    )
    schedule_rng = np.random.default_rng(int(seed) + 1_400_003)
    readout_rng = np.random.default_rng(int(seed) + 1_500_003)
    no_attack = {
        "id": "no_attack",
        "kind": "none",
        "target": "",
        "attack_type": "no_attack",
        "cost": 0.0,
    }
    route_ids = tuple(str(row["route_id"]) for row in case.routes)
    probe_turns = 0
    probe_attempts = 0
    valid_readouts = 0
    missing_readouts = 0
    observed_hits = 0
    correct_modal_routes = 0
    true_route_probability_sum = 0.0
    route_sequence_mismatches = 0
    reward_mismatches = 0
    within_panel_reward_mismatches = 0
    reward_sum = 0.0
    axis_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    observation_counts: Counter[str] = Counter()
    trajectory: list[dict[str, Any]] = []
    decision: dict[str, Any] | None = None

    for turn_index in range(turns):
        step = turn_index + 1
        predictive_route_policy = (
            particle_filter.predictive_route_policy()
        )
        if (
            schedule_rng.random()
            < scenario.probe_probability
        ):
            desired_stabilizer = study_config.stabilizers[
                probe_turns % len(study_config.stabilizers)
            ]
            panel_actions, information_gain = _select_probe_panel(
                predictive_route_policy,
                actions,
                commissioning,
                separating_panel_nodes=separating_panel_nodes,
                stabilizer=desired_stabilizer,
                probes_per_turn=scenario.probes_per_turn,
            )
            probe_turns += 1
            probe_attempts += len(panel_actions)
            for action in panel_actions:
                axis_counts[str(action["bsm_stabilizer"])] += 1
                node_counts[str(action["target"])] += 1
        else:
            panel_actions = []
            information_gain = 0.0

        true_route = true_alice.sample()
        clean_route = clean_alice.sample()
        route_sequence_mismatches += int(true_route != clean_route)
        if panel_actions:
            panel_results = [
                physics.step(
                    true_route,
                    action,
                    seed=int(seed) + 1_600_003,
                    step_index=turn_index,
                )
                for action in panel_actions
            ]
        else:
            panel_results = [
                physics.step(
                    true_route,
                    no_attack,
                    seed=int(seed) + 1_600_003,
                    step_index=turn_index,
                )
            ]
        clean_result = physics.step(
            clean_route,
            no_attack,
            seed=int(seed) + 1_600_003,
            step_index=turn_index,
        )
        panel_rewards = [
            bell_qualified_rate_utility(
                result=result,
                calibration=physics.calibrations[true_route],
                reference_service_rate_hz=(
                    physics.reference_service_rate_hz
                ),
            ).reward
            for result in panel_results
        ]
        probed_reward = panel_rewards[0]
        within_panel_reward_mismatches += sum(
            not math.isclose(
                reward,
                probed_reward,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for reward in panel_rewards[1:]
        )
        clean_reward = bell_qualified_rate_utility(
            result=clean_result,
            calibration=physics.calibrations[clean_route],
            reference_service_rate_hz=(
                physics.reference_service_rate_hz
            ),
        ).reward
        true_alice.update(probed_reward)
        clean_alice.update(clean_reward)
        reward_sum += probed_reward
        reward_mismatches += int(
            not math.isclose(
                probed_reward,
                clean_reward,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

        if not panel_actions:
            observation_status = "not_probed"
            panel_observations: list[dict[str, Any]] = []
            route_likelihood = np.ones(
                len(case.routes),
                dtype=float,
            )
            informative = False
        else:
            panel_observations = []
            observed_by_action: dict[str, bool | None] = {}
            valid_this_turn = 0
            missing_this_turn = 0
            for action, result in zip(
                panel_actions,
                panel_results,
            ):
                ancilla = _ancilla_observation(result)
                action_id = str(action["id"])
                if (
                    readout_rng.random()
                    > scenario.readout_availability_probability
                ):
                    status = "heralded_missing_readout"
                    observed_hit = None
                    missing_readouts += 1
                    missing_this_turn += 1
                else:
                    valid_readouts += 1
                    valid_this_turn += 1
                    observed_hit = bool(ancilla["hit"])
                    if observed_hit:
                        status = "observed_hit"
                        observed_hits += 1
                    else:
                        status = "observed_no_hit"
                observation_counts[status] += 1
                observed_by_action[action_id] = observed_hit
                panel_observations.append({
                    "action_id": action_id,
                    "target": str(action["target"]),
                    "stabilizer": str(action["bsm_stabilizer"]),
                    "status": status,
                    "observed_hit": observed_hit,
                    "ancilla_total_count": int(
                        ancilla["total_count"]
                    ),
                    "ancilla_one_count": int(
                        ancilla["one_count"]
                    ),
                })
            if valid_this_turn == 0:
                observation_status = "panel_all_readouts_missing"
            elif missing_this_turn:
                observation_status = "panel_partially_observed"
            else:
                observation_status = "panel_fully_observed"
            observation_counts[observation_status] += 1
            informative = valid_this_turn > 0
            route_likelihood = (
                _commissioned_panel_compatibility_likelihood(
                    panel_actions,
                    observed_by_action,
                    commissioning,
                    route_count=len(case.routes),
                )
            )
        if not panel_actions:
            observation_counts[observation_status] += 1
        update = particle_filter.observe_route_likelihood(
            route_likelihood=route_likelihood,
            observation={
                "status": observation_status,
                "panel": panel_observations,
            },
            informative=informative,
            alice_public_reward=probed_reward,
        )
        route_posterior = np.asarray(
            update["route_posterior"],
            dtype=float,
        )
        modal_route = int(np.argmax(route_posterior))
        correct_modal_routes += int(modal_route == true_route)
        true_route_probability_sum += float(
            route_posterior[true_route]
        )
        posterior = update["posterior"]
        modal_label = max(posterior, key=posterior.__getitem__)
        modal_mass = float(posterior[modal_label])
        if (
            step == 1
            or step % study_config.record_every == 0
            or modal_mass >= study_config.posterior_threshold
            or step == turns
        ):
            trajectory.append({
                "turn": step,
                "probe_action_ids": [
                    str(action["id"]) for action in panel_actions
                ],
                "probe_stabilizer": (
                    None
                    if not panel_actions
                    else str(panel_actions[0]["bsm_stabilizer"])
                ),
                "probe_targets": [
                    str(action["target"]) for action in panel_actions
                ],
                "probe_information_gain_nats": information_gain,
                "observation_status": observation_status,
                "panel_observations": panel_observations,
                "posterior": dict(posterior),
                "posterior_modal_hypothesis": modal_label,
                "posterior_modal_mass": modal_mass,
                "modal_route_id_for_scoring_only": (
                    route_ids[modal_route]
                ),
                "true_route_id_for_scoring_only": (
                    route_ids[true_route]
                ),
                "true_route_posterior_for_scoring_only": float(
                    route_posterior[true_route]
                ),
                "route_inference_accuracy_for_scoring_only": (
                    correct_modal_routes / step
                ),
            })
        if modal_mass >= study_config.posterior_threshold:
            decision = {
                "turn": step,
                "selected_hypothesis_label": modal_label,
                "selected_hypothesis_posterior": modal_mass,
                "posterior_threshold": (
                    study_config.posterior_threshold
                ),
            }
            break

    executed = int(true_alice.update_calls)
    final_posterior = particle_filter.posterior_by_label()
    probe_policy = true_alice.policy()
    clean_policy = clean_alice.policy()
    return {
        "scenario": scenario.to_dict(),
        "true_hypothesis_label_for_scoring_only": true_label,
        "seed": int(seed),
        "turns": executed,
        "decision": decision,
        "identified_correctly_for_scoring_only": bool(
            decision is not None
            and decision["selected_hypothesis_label"] == true_label
        ),
        "final_posterior": final_posterior,
        "true_hypothesis_posterior_for_scoring_only": float(
            final_posterior[true_label]
        ),
        "route_inference_accuracy_for_scoring_only": (
            correct_modal_routes / executed
        ),
        "mean_true_route_posterior_for_scoring_only": (
            true_route_probability_sum / executed
        ),
        "mean_alice_public_reward": reward_sum / executed,
        "probe_turns": probe_turns,
        "probe_attempts": probe_attempts,
        "probe_fraction": probe_turns / executed,
        "mean_probe_interactions_per_turn": (
            probe_attempts / executed
        ),
        "valid_readouts": valid_readouts,
        "missing_readouts": missing_readouts,
        "valid_readout_fraction_of_probes": (
            valid_readouts / probe_attempts if probe_attempts else None
        ),
        "observed_hits": observed_hits,
        "observation_counts": dict(observation_counts),
        "axis_counts": dict(axis_counts),
        "node_counts": dict(node_counts),
        "matched_clean_presteering_audit": {
            "route_sequence_mismatches": route_sequence_mismatches,
            "reward_mismatches_beyond_1e-15": reward_mismatches,
            "within_panel_reward_mismatches_beyond_1e-15": (
                within_panel_reward_mismatches
            ),
            "final_policy_total_variation": (
                0.5
                * float(np.sum(np.abs(probe_policy - clean_policy)))
            ),
            "counterfactual_disclosed_online": False,
        },
        "filter_diagnostics": particle_filter.diagnostics(),
        "trajectory": trajectory,
    }


def _commission_probe_emissions(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    actions: Sequence[dict[str, Any]],
    *,
    draws_per_route_probe: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    probabilities: dict[str, list[float]] = {}
    deterministic_hits: dict[str, list[bool]] = {}
    root_seed = 2_026_072_177
    for action_index, action in enumerate(actions):
        hit_probabilities = []
        action_deterministic_hits = []
        for route_index in range(len(case.routes)):
            hits = 0
            total_observations = 0
            one_count = 0
            for draw_index in range(draws_per_route_probe):
                result = physics.step(
                    route_index,
                    action,
                    seed=(
                        root_seed
                        + action_index * 101
                        + route_index * 17
                    ),
                    step_index=draw_index,
                )
                ancilla = _ancilla_observation(result)
                hits += int(ancilla["hit"])
                total_observations += int(ancilla["total_count"])
                one_count += int(ancilla["one_count"])
            probability = (
                hits + 0.5
            ) / (draws_per_route_probe + 1.0)
            if hits not in {0, draws_per_route_probe}:
                raise ValueError(
                    "parallel panel compatibility requires deterministic "
                    "commissioned ancilla presence in the declared ideal "
                    "instrument"
                )
            deterministic_hit = hits == draws_per_route_probe
            hit_probabilities.append(float(probability))
            action_deterministic_hits.append(deterministic_hit)
            rows.append({
                "action_id": str(action["id"]),
                "target_node": str(action["target"]),
                "stabilizer": str(action["bsm_stabilizer"]),
                "route_index": route_index,
                "route_id": str(case.routes[route_index]["route_id"]),
                "draws": draws_per_route_probe,
                "hit_count": hits,
                "deterministic_hit": deterministic_hit,
                "beta_half_posterior_mean_hit_probability": (
                    probability
                ),
                "mean_ancilla_count": (
                    total_observations / draws_per_route_probe
                ),
                "ancilla_one_fraction": (
                    one_count / total_observations
                    if total_observations
                    else None
                ),
            })
        probabilities[str(action["id"])] = hit_probabilities
        deterministic_hits[str(action["id"])] = (
            action_deterministic_hits
        )
    return {
        "draws_per_route_probe": draws_per_route_probe,
        "binary_hit_probability_by_action_route": probabilities,
        "deterministic_hit_by_action_route": deterministic_hits,
        "all_local_presence_patterns_deterministic": True,
        "rows": rows,
        "source": "compiled sampled XX/ZZ ancilla records",
        "pseudocount": "Beta(0.5,0.5) posterior mean",
    }


def _select_probe_panel(
    route_prior: Sequence[float] | np.ndarray,
    actions: Sequence[Mapping[str, Any]],
    commissioning: Mapping[str, Any],
    *,
    separating_panel_nodes: Sequence[str],
    stabilizer: str,
    probes_per_turn: int | None,
) -> tuple[list[dict[str, Any]], float]:
    prior = _normalized(
        np.asarray(route_prior, dtype=float),
        "probe route prior",
    )
    desired = str(stabilizer).upper()
    action_by_node = {
        str(action["target"]): dict(action)
        for action in actions
        if str(action["bsm_stabilizer"]).upper() == desired
    }
    panel_nodes = tuple(map(str, separating_panel_nodes))
    if any(node not in action_by_node for node in panel_nodes):
        raise ValueError(
            f"separating panel lacks {desired} actions"
        )
    budget = (
        len(panel_nodes)
        if probes_per_turn is None
        else min(int(probes_per_turn), len(panel_nodes))
    )
    candidate_panels = (
        (panel_nodes,)
        if budget == len(panel_nodes)
        else combinations(panel_nodes, budget)
    )
    scored = []
    deterministic = commissioning[
        "deterministic_hit_by_action_route"
    ]
    for nodes in candidate_panels:
        active_nodes = tuple(nodes)
        pattern_mass: dict[tuple[bool, ...], float] = {}
        for route_index, route_mass in enumerate(prior):
            pattern = tuple(
                bool(
                    deterministic[
                        str(action_by_node[node]["id"])
                    ][route_index]
                )
                for node in active_nodes
            )
            pattern_mass[pattern] = (
                pattern_mass.get(pattern, 0.0)
                + float(route_mass)
            )
        information_gain = _categorical_entropy(
            tuple(pattern_mass.values())
        )
        scored.append((
            information_gain,
            active_nodes,
        ))
    information_gain, selected_nodes = max(
        scored,
        key=lambda row: (row[0], row[1]),
    )
    return (
        [action_by_node[node] for node in selected_nodes],
        float(information_gain),
    )


def _commissioned_panel_compatibility_likelihood(
    panel_actions: Sequence[Mapping[str, Any]],
    observed_by_action: Mapping[str, bool | None],
    commissioning: Mapping[str, Any],
    *,
    route_count: int,
) -> np.ndarray:
    deterministic = commissioning[
        "deterministic_hit_by_action_route"
    ]
    likelihood = np.ones(route_count, dtype=float)
    for action in panel_actions:
        action_id = str(action["id"])
        observed = observed_by_action[action_id]
        if observed is None:
            continue
        expected = np.asarray(
            deterministic[action_id],
            dtype=bool,
        )
        likelihood *= expected == bool(observed)
    if not np.any(likelihood > 0.0):
        raise FloatingPointError(
            "observed panel pattern is incompatible with every "
            "commissioned route"
        )
    return likelihood


def _binary_mutual_information(
    prior: np.ndarray,
    hit_probabilities: np.ndarray,
) -> float:
    if prior.shape != hit_probabilities.shape:
        raise ValueError("prior and hit probabilities differ in shape")
    marginal_hit = float(np.dot(prior, hit_probabilities))
    conditional_entropy = float(np.dot(
        prior,
        np.asarray([
            _binary_entropy(value)
            for value in hit_probabilities
        ]),
    ))
    return max(
        0.0,
        _binary_entropy(marginal_hit) - conditional_entropy,
    )


def _categorical_entropy(
    probabilities: Sequence[float] | np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    positive = values[values > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 0.0), 1.0)
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return -value * math.log(value) - (1.0 - value) * math.log(
        1.0 - value
    )


def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized = _normalized(weights, "particle resampling weights")
    count = normalized.size
    positions = (
        rng.random() + np.arange(count, dtype=float)
    ) / count
    cumulative = np.cumsum(normalized)
    cumulative[-1] = 1.0
    return np.searchsorted(
        cumulative,
        positions,
        side="right",
    ).astype(np.int64)


def _all_internal_sensor_nodes(
    case: HistogramGameCase,
) -> tuple[str, ...]:
    nodes = sorted({
        str(node)
        for route in case.routes
        for node in route.get("internal_nodes") or ()
    })
    if not nodes:
        raise ValueError("case has no internal repeater nodes")
    return tuple(nodes)


def _normalized(
    values: Sequence[float] | np.ndarray,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    total = float(np.sum(array))
    if (
        array.ndim != 1
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or not math.isfinite(total)
        or total <= 0.0
    ):
        raise FloatingPointError(f"{label} is not normalizable")
    return array / total


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregates = []
    keys = sorted({
        (
            str(row["scenario"]["name"]),
            str(row["true_hypothesis_label_for_scoring_only"]),
        )
        for row in rows
    })
    for scenario, label in keys:
        selected = [
            row for row in rows
            if str(row["scenario"]["name"]) == scenario
            and str(
                row["true_hypothesis_label_for_scoring_only"]
            ) == label
        ]
        aggregates.append({
            "scenario": scenario,
            "true_hypothesis_label": label,
            "runs": len(selected),
            "correct_identification_count": sum(
                bool(row["identified_correctly_for_scoring_only"])
                for row in selected
            ),
            "mean_turns": float(np.mean([
                int(row["turns"]) for row in selected
            ])),
            "mean_true_hypothesis_posterior": float(np.mean([
                float(
                    row[
                        "true_hypothesis_posterior_for_scoring_only"
                    ]
                )
                for row in selected
            ])),
            "mean_route_inference_accuracy": float(np.mean([
                float(
                    row[
                        "route_inference_accuracy_for_scoring_only"
                    ]
                )
                for row in selected
            ])),
            "mean_true_route_posterior": float(np.mean([
                float(
                    row[
                        "mean_true_route_posterior_for_scoring_only"
                    ]
                )
                for row in selected
            ])),
            "mean_probe_fraction": float(np.mean([
                float(row["probe_fraction"]) for row in selected
            ])),
            "mean_probe_interactions_per_turn": float(np.mean([
                float(row["mean_probe_interactions_per_turn"])
                for row in selected
            ])),
            "mean_valid_readout_fraction_of_probes": float(
                np.mean([
                    float(
                        row["valid_readout_fraction_of_probes"]
                    )
                    for row in selected
                    if row["valid_readout_fraction_of_probes"]
                    is not None
                ])
            ),
            "total_route_sequence_mismatches_vs_clean": sum(
                int(
                    row["matched_clean_presteering_audit"][
                        "route_sequence_mismatches"
                    ]
                )
                for row in selected
            ),
            "total_reward_mismatches_vs_clean": sum(
                int(
                    row["matched_clean_presteering_audit"][
                        "reward_mismatches_beyond_1e-15"
                    ]
                )
                for row in selected
            ),
            "total_within_panel_reward_mismatches": sum(
                int(
                    row["matched_clean_presteering_audit"][
                        "within_panel_reward_mismatches_beyond_1e-15"
                    ]
                )
                for row in selected
            ),
            "maximum_final_policy_tv_vs_clean": max(
                float(
                    row["matched_clean_presteering_audit"][
                        "final_policy_total_variation"
                    ]
                )
                for row in selected
            ),
        })
    return aggregates
