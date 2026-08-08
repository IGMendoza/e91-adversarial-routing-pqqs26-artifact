"""Quantum-interaction follower smoke test using BSM stabilizer ancillas.

This module does not use the deterministic reservation-signature follower.
Eve selects a physical repeater and an ``XX`` or ``ZZ`` Bell-stabilizer
ancilla interaction. Her route estimate is constructed from the sampled
ancilla record emitted by the compiled quantum instrument. The selected Alice
route remains audit-only and is used solely to score inference accuracy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner
from .learning import (
    bell_qualified_rate_utility,
    build_histogram_alice_learner,
)
from .physics import HistogramPhysicsPool
from .stackelberg_bandit import LearnerHypothesisTracker


BSM_ANCILLA_FOLLOWER_SMOKE_SCHEMA = (
    "histogram_bsm_ancilla_follower_smoke.v1"
)
BSM_STABILIZERS = ("XX", "ZZ")


@dataclass(frozen=True)
class BSMAncillaFollowerSmokeConfig:
    """Small compiled-physics identification and disturbance contract."""

    maximum_follow_turns: int = 80
    commissioning_draws_per_route_probe: int = 8
    posterior_threshold: float = 0.99
    dephasing_theta: float = math.pi / 2.0
    stabilizers: tuple[str, ...] = BSM_STABILIZERS
    execution_seeds: tuple[int, ...] = (91, 93, 95)
    presteering_seeds: tuple[int, ...] = tuple(range(201, 217))
    record_every: int = 10

    def __post_init__(self) -> None:
        if self.maximum_follow_turns < 2:
            raise ValueError("maximum_follow_turns must be at least two")
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
        for name, seeds in (
            ("execution_seeds", self.execution_seeds),
            ("presteering_seeds", self.presteering_seeds),
        ):
            if not seeds or len(set(map(int, seeds))) != len(seeds):
                raise ValueError(f"{name} must be nonempty and unique")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stabilizers": list(self.stabilizers),
            "execution_seeds": list(map(int, self.execution_seeds)),
            "presteering_seeds": list(map(int, self.presteering_seeds)),
        }


def run_bsm_ancilla_follower_smoke(
    case: HistogramGameCase,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    smoke_config: BSMAncillaFollowerSmokeConfig = (
        BSMAncillaFollowerSmokeConfig()
    ),
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Identify Alice from sampled ancilla records and measure policy drift."""

    labels = tuple(map(str, learner_configs))
    if len(labels) != 2:
        raise ValueError("ancilla follower smoke requires two learner hypotheses")
    base_config = learner_configs[labels[0]]
    _validate_matched_physics_configs(learner_configs)
    route_sensor_nodes = _two_route_sensor_nodes(case)
    actions = _probe_actions(
        route_sensor_nodes,
        smoke_config.stabilizers,
        smoke_config.dephasing_theta,
    )
    physics = HistogramPhysicsPool(case, base_config)
    started = time.perf_counter()
    commissioning = _commission_probe_emissions(
        case,
        physics,
        actions,
        smoke_config=smoke_config,
    )
    invariance = _expected_service_invariance(
        case,
        physics,
        actions,
    )

    identification_rows: list[dict[str, Any]] = []
    total_identification = (
        len(labels) * len(smoke_config.execution_seeds)
    )
    completed = 0
    for true_label in labels:
        for seed in map(int, smoke_config.execution_seeds):
            identification_rows.append(
                _run_one_identification(
                    case,
                    physics,
                    learner_configs,
                    true_label=true_label,
                    seed=seed,
                    actions=actions,
                    commissioning=commissioning,
                    smoke_config=smoke_config,
                )
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    f"BSM ancilla identification {completed}/"
                    f"{total_identification}: {true_label}, seed={seed}"
                )

    presteering_rows: list[dict[str, Any]] = []
    total_presteering = (
        len(labels) * len(smoke_config.presteering_seeds)
    )
    completed = 0
    for label in labels:
        for seed in map(int, smoke_config.presteering_seeds):
            presteering_rows.append(
                _run_one_presteering_comparison(
                    case,
                    physics,
                    learner_configs[label],
                    label=label,
                    seed=seed,
                    actions=actions,
                    turns=smoke_config.maximum_follow_turns,
                )
            )
            completed += 1
            if progress_callback is not None and (
                completed == 1
                or completed == total_presteering
                or completed % 8 == 0
            ):
                progress_callback(
                    f"BSM ancilla pre-steering {completed}/"
                    f"{total_presteering}"
                )

    return {
        "schema": BSM_ANCILLA_FOLLOWER_SMOKE_SCHEMA,
        "case": case.to_dict(),
        "smoke_config": smoke_config.to_dict(),
        "learner_configs": {
            label: _learner_config_summary(config)
            for label, config in learner_configs.items()
        },
        "physics": physics.diagnostics(),
        "probe_contract": {
            "actions": [dict(action) for action in actions],
            "route_sensor_nodes": list(route_sensor_nodes),
            "observation": (
                "sampled local XX/ZZ stabilizer-ancilla zero/one counts; "
                "no reservation signature or true route label"
            ),
            "route_inference": (
                "Bayes update from commissioned binary ancilla-presence "
                "emission probabilities and the follower's predictive "
                "route policy"
            ),
            "learner_filter": (
                "hard-decoded route is supplied to the exact finite learner "
                "hypotheses; true route is used for scoring only"
            ),
        },
        "commissioning": commissioning,
        "expected_service_invariance": invariance,
        "identification_rows": identification_rows,
        "presteering_rows": presteering_rows,
        "aggregates": {
            "identification": _aggregate_identification(
                identification_rows
            ),
            "presteering": _aggregate_presteering(presteering_rows),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "information_boundary": {
            "eve_observes": [
                "chosen compromised repeater",
                "chosen XX or ZZ stabilizer",
                "sampled ancilla zero/one counts",
                "Alice public scalar service reward",
            ],
            "audit_only": [
                "Alice selected route",
                "true learner label",
                "probe target-active flag",
                "counterfactual clean outcome",
            ],
            "excluded_online": [
                "deterministic reservation signature",
                "Alice live policy or weights",
                "Alice private sampling randomness",
                "simulator target-active flag",
                "payoff audit matrix",
            ],
        },
        "scientific_scope": {
            "instrument": "ideal_commuting_bsm_stabilizer_ancilla",
            "physics_backend": "compiled_histogram_transcript",
            "device_costs": (
                "ancilla preparation, gates, readout, loss, and added memory "
                "latency are not yet applied"
            ),
            "security_claim": "none",
        },
        "claim_boundary": (
            "This smoke test asks whether the ideal compiled XX/ZZ BSM "
            "ancilla produces usable follower observations and whether its "
            "marginal service effect exceeds ordinary finite-transcript "
            "variation. It is not a near-term hardware demonstration."
        ),
    }


def _run_one_identification(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    true_label: str,
    seed: int,
    actions: Sequence[dict[str, Any]],
    commissioning: Mapping[str, Any],
    smoke_config: BSMAncillaFollowerSmokeConfig,
) -> dict[str, Any]:
    true_alice = _build_alice(
        case,
        replace(
            learner_configs[true_label],
            turns=smoke_config.maximum_follow_turns,
            seed=int(seed),
        ),
    )
    hypotheses = {
        label: _build_alice(
            case,
            replace(config, turns=smoke_config.maximum_follow_turns),
        )
        for label, config in learner_configs.items()
    }
    tracker = LearnerHypothesisTracker(hypotheses)
    correct_routes = 0
    confidence_sum = 0.0
    reward_sum = 0.0
    total_observation_counts = 0
    axis_counts: Counter[str] = Counter()
    trajectory: list[dict[str, Any]] = []
    decision: dict[str, Any] | None = None
    route_ids = tuple(str(route["route_id"]) for route in case.routes)

    for turn_index in range(smoke_config.maximum_follow_turns):
        step = turn_index + 1
        action = actions[turn_index % len(actions)]
        route_prior = tracker.predictive_route_policy()
        route_index = true_alice.sample()
        result = physics.step(
            route_index,
            action,
            seed=int(seed) + 700_001,
            step_index=turn_index,
        )
        reward = bell_qualified_rate_utility(
            result=result,
            calibration=physics.calibrations[route_index],
            reference_service_rate_hz=physics.reference_service_rate_hz,
        ).reward
        true_alice.update(reward)
        ancilla = _ancilla_observation(result)
        inferred_route, route_posterior = _infer_route(
            route_prior,
            action_id=str(action["id"]),
            hit=bool(ancilla["hit"]),
            commissioning=commissioning,
        )
        tracker_row = tracker.observe_route_reward(inferred_route, reward)
        correct_routes += int(inferred_route == route_index)
        confidence_sum += float(route_posterior[inferred_route])
        reward_sum += reward
        total_observation_counts += int(ancilla["total_count"])
        axis_counts[str(action["bsm_stabilizer"])] += 1
        posterior = tracker.posterior_by_label()
        modal_label = max(posterior, key=posterior.__getitem__)
        modal_mass = float(posterior[modal_label])

        if (
            step == 1
            or step % smoke_config.record_every == 0
            or modal_mass >= smoke_config.posterior_threshold
            or step == smoke_config.maximum_follow_turns
        ):
            trajectory.append({
                "turn": step,
                "probe_action_id": str(action["id"]),
                "probe_stabilizer": str(action["bsm_stabilizer"]),
                "ancilla_total_count": int(ancilla["total_count"]),
                "ancilla_one_count": int(ancilla["one_count"]),
                "inferred_route_id": route_ids[inferred_route],
                "inferred_route_confidence": float(
                    route_posterior[inferred_route]
                ),
                "posterior": posterior,
                "posterior_modal_hypothesis": modal_label,
                "posterior_modal_mass": modal_mass,
                "route_inference_accuracy_for_scoring_only": (
                    correct_routes / step
                ),
            })
        if modal_mass >= smoke_config.posterior_threshold:
            decision = {
                "turn": step,
                "selected_hypothesis_label": modal_label,
                "selected_hypothesis_posterior": modal_mass,
                "posterior_threshold": smoke_config.posterior_threshold,
            }
            break

    turns = int(true_alice.update_calls)
    final_posterior = tracker.posterior_by_label()
    return {
        "true_hypothesis_label_for_scoring_only": true_label,
        "seed": int(seed),
        "turns": turns,
        "decision": decision,
        "identified_correctly_for_scoring_only": bool(
            decision is not None
            and decision["selected_hypothesis_label"] == true_label
        ),
        "final_posterior": final_posterior,
        "route_inference_accuracy_for_scoring_only": (
            correct_routes / turns
        ),
        "mean_inferred_route_confidence": confidence_sum / turns,
        "mean_alice_public_reward": reward_sum / turns,
        "mean_ancilla_observation_count": (
            total_observation_counts / turns
        ),
        "axis_counts": dict(axis_counts),
        "trajectory": trajectory,
        "information_boundary": (
            "learner tracker receives only the route hard-decoded from the "
            "sampled quantum ancilla signal plus public reward"
        ),
    }


def _run_one_presteering_comparison(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    *,
    label: str,
    seed: int,
    actions: Sequence[dict[str, Any]],
    turns: int,
) -> dict[str, Any]:
    probe = _run_alice_trace(
        case,
        physics,
        config,
        seed=seed,
        physics_seed=seed + 800_001,
        turns=turns,
        actions=actions,
    )
    clean = _run_alice_trace(
        case,
        physics,
        config,
        seed=seed,
        physics_seed=seed + 800_001,
        turns=turns,
        actions=None,
    )
    clean_reference = _run_alice_trace(
        case,
        physics,
        config,
        seed=seed,
        physics_seed=seed + 900_001,
        turns=turns,
        actions=None,
    )
    return {
        "learner_label": label,
        "seed": int(seed),
        "turns": int(turns),
        "probe_vs_clean": _trace_comparison(probe, clean),
        "clean_vs_independent_clean": _trace_comparison(
            clean,
            clean_reference,
        ),
        "probe": _trace_summary(probe),
        "clean": _trace_summary(clean),
        "independent_clean": _trace_summary(clean_reference),
        "semantics": (
            "probe and first clean control share learner and physics seeds; "
            "the independent clean control estimates ordinary finite-"
            "transcript trajectory variation"
        ),
    }


def _run_alice_trace(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    *,
    seed: int,
    physics_seed: int,
    turns: int,
    actions: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    alice = _build_alice(
        case,
        replace(config, turns=turns, seed=int(seed)),
    )
    no_attack = {
        "id": "no_attack",
        "kind": "none",
        "target": "",
        "attack_type": "no_attack",
        "cost": 0.0,
    }
    route_indices: list[int] = []
    rewards: list[float] = []
    delivered_fractions: list[float] = []
    qber_values: list[float] = []
    chsh_values: list[float] = []
    outcomes: Counter[str] = Counter()
    ancilla_counts = 0
    for turn_index in range(turns):
        route_index = alice.sample()
        action = (
            no_attack
            if actions is None
            else actions[turn_index % len(actions)]
        )
        result = physics.step(
            route_index,
            action,
            seed=int(physics_seed),
            step_index=turn_index,
        )
        reward = bell_qualified_rate_utility(
            result=result,
            calibration=physics.calibrations[route_index],
            reference_service_rate_hz=physics.reference_service_rate_hz,
        ).reward
        alice.update(reward)
        route_indices.append(route_index)
        rewards.append(reward)
        delivered_fractions.append(
            result.delivered_pair_count / result.requested_pair_count
        )
        if result.qber_key is not None:
            qber_values.append(float(result.qber_key))
        if result.chsh_S_observed is not None:
            chsh_values.append(float(result.chsh_S_observed))
        outcomes[str(result.public_outcome)] += 1
        ancilla_counts += int(result.eve_observation_count)
    return {
        "route_indices": route_indices,
        "rewards": rewards,
        "delivered_fractions": delivered_fractions,
        "qber_values": qber_values,
        "chsh_values": chsh_values,
        "outcomes": dict(outcomes),
        "ancilla_counts": ancilla_counts,
        "final_policy": alice.policy(),
    }


def _commission_probe_emissions(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    actions: Sequence[dict[str, Any]],
    *,
    smoke_config: BSMAncillaFollowerSmokeConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    probabilities: dict[str, list[float]] = {}
    root_seed = 2_026_071_777
    for action_index, action in enumerate(actions):
        hit_probabilities = []
        for route_index in range(len(case.routes)):
            hits = 0
            total_observations = 0
            one_count = 0
            delivered = 0
            for draw_index in range(
                smoke_config.commissioning_draws_per_route_probe
            ):
                result = physics.step(
                    route_index,
                    action,
                    seed=root_seed + action_index * 101 + route_index * 17,
                    step_index=draw_index,
                )
                ancilla = _ancilla_observation(result)
                hits += int(ancilla["hit"])
                total_observations += int(ancilla["total_count"])
                one_count += int(ancilla["one_count"])
                delivered += int(result.delivered_pair_count)
            draws = smoke_config.commissioning_draws_per_route_probe
            probability = (hits + 0.5) / (draws + 1.0)
            hit_probabilities.append(float(probability))
            rows.append({
                "action_id": str(action["id"]),
                "target_node": str(action["target"]),
                "stabilizer": str(action["bsm_stabilizer"]),
                "route_index": route_index,
                "route_id": str(case.routes[route_index]["route_id"]),
                "draws": draws,
                "hit_count": hits,
                "beta_half_posterior_mean_hit_probability": probability,
                "mean_ancilla_count": total_observations / draws,
                "ancilla_one_fraction": (
                    one_count / total_observations
                    if total_observations
                    else None
                ),
                "mean_delivered_pair_count": delivered / draws,
            })
        probabilities[str(action["id"])] = hit_probabilities
    return {
        "draws_per_route_probe": (
            smoke_config.commissioning_draws_per_route_probe
        ),
        "binary_hit_probability_by_action_route": probabilities,
        "rows": rows,
        "source": "compiled sampled XX/ZZ ancilla records",
        "pseudocount": "Beta(0.5,0.5) posterior mean",
    }


def _expected_service_invariance(
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    no_attack = {
        "id": "no_attack",
        "kind": "none",
        "target": "",
        "attack_type": "no_attack",
        "cost": 0.0,
    }
    rows = []
    maximum = {
        "delivery_probability": 0.0,
        "expected_qber": 0.0,
        "expected_chsh_s": 0.0,
    }
    for route_index, route in enumerate(case.routes):
        baseline = physics.step(
            route_index,
            no_attack,
            seed=2_026_071_881,
            step_index=0,
        )
        for action in actions:
            if str(action["target"]) not in set(
                map(str, route.get("internal_nodes") or [])
            ):
                continue
            probed = physics.step(
                route_index,
                action,
                seed=2_026_071_881,
                step_index=0,
            )
            deltas = {
                "delivery_probability": abs(
                    float(probed.delivery_probability_expected)
                    - float(baseline.delivery_probability_expected)
                ),
                "expected_qber": abs(
                    float(probed.expected_qber_from_histogram)
                    - float(baseline.expected_qber_from_histogram)
                ),
                "expected_chsh_s": abs(
                    float(probed.chsh_S_expected)
                    - float(baseline.chsh_S_expected)
                ),
            }
            for key, value in deltas.items():
                maximum[key] = max(maximum[key], value)
            rows.append({
                "route_id": str(route["route_id"]),
                "action_id": str(action["id"]),
                "stabilizer": str(action["bsm_stabilizer"]),
                "expected_metric_absolute_deltas": deltas,
                "sampled_ancilla_observation_count": int(
                    probed.eve_observation_count
                ),
                "eve_guess_probability": float(
                    probed.eve_guess_probability
                ),
                "information_exposed": bool(probed.information_exposed),
            })
    return {
        "rows": rows,
        "maximum_expected_metric_absolute_deltas": maximum,
        "exact_within_1e_12": all(
            value <= 1e-12 for value in maximum.values()
        ),
        "semantics": (
            "expected Alice delivery/QBER/CHSH marginals under the compiled "
            "probe versus no_attack; Alice's sampled transcript remains on "
            "the exact no-move histogram and Eve's record is sampled from "
            "the parallel conditional mini-histogram"
        ),
    }


def _infer_route(
    route_prior: Sequence[float] | np.ndarray,
    *,
    action_id: str,
    hit: bool,
    commissioning: Mapping[str, Any],
) -> tuple[int, np.ndarray]:
    prior = np.asarray(route_prior, dtype=float)
    hit_probabilities = np.asarray(
        commissioning["binary_hit_probability_by_action_route"][action_id],
        dtype=float,
    )
    likelihood = hit_probabilities if hit else 1.0 - hit_probabilities
    posterior = prior * likelihood
    total = float(np.sum(posterior))
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("ancilla route posterior has zero mass")
    posterior /= total
    return int(np.argmax(posterior)), posterior


def _ancilla_observation(result: Any) -> dict[str, Any]:
    zero_count = 0
    one_count = 0
    for label, count in result.eve_observation_distribution.items():
        if not str(label).startswith("repeater_bsm_stabilizer_probe:"):
            continue
        if str(label).endswith(":ancilla=0"):
            zero_count += int(count)
        elif str(label).endswith(":ancilla=1"):
            one_count += int(count)
    total = zero_count + one_count
    return {
        "hit": total > 0,
        "zero_count": zero_count,
        "one_count": one_count,
        "total_count": total,
    }


def _probe_actions(
    sensor_nodes: Sequence[str],
    stabilizers: Sequence[str],
    theta: float,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "id": (
                "repeater_bsm_stabilizer_probe:"
                f"{node}:{str(stabilizer).upper()}"
            ),
            "kind": "node",
            "target": str(node),
            "attack_type": "repeater_bsm_stabilizer_probe",
            "cost": 0.0,
            "bsm_stabilizer": str(stabilizer).upper(),
            "dephasing_theta": float(theta),
        }
        for node in sensor_nodes
        for stabilizer in stabilizers
    )


def _two_route_sensor_nodes(
    case: HistogramGameCase,
) -> tuple[str, str]:
    if len(case.routes) != 2:
        raise ValueError("ancilla smoke currently requires exactly two routes")
    internal_sets = [
        set(map(str, route.get("internal_nodes") or []))
        for route in case.routes
    ]
    selected = []
    for index, nodes in enumerate(internal_sets):
        unique = sorted(nodes - internal_sets[1 - index])
        if not unique:
            raise ValueError(
                "each smoke route requires a distinct internal BSM node"
            )
        selected.append(unique[0])
    return tuple(selected)


def _build_alice(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
) -> PlayerLearner:
    implementation, metadata = build_histogram_alice_learner(
        case=case,
        config=config,
        prior=np.full(len(case.routes), 1.0 / len(case.routes)),
    )
    return PlayerLearner(
        player="alice",
        action_ids=tuple(
            str(route["route_id"]) for route in case.routes
        ),
        learner=implementation,
        metadata=metadata,
    )


def _trace_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_policy = np.asarray(left["final_policy"], dtype=float)
    right_policy = np.asarray(right["final_policy"], dtype=float)
    left_routes = np.asarray(left["route_indices"], dtype=np.int64)
    right_routes = np.asarray(right["route_indices"], dtype=np.int64)
    return {
        "final_policy_total_variation": (
            0.5 * float(np.sum(np.abs(left_policy - right_policy)))
        ),
        "route_sequence_agreement": float(
            np.mean(left_routes == right_routes)
        ),
        "mean_reward_difference": (
            float(np.mean(left["rewards"]))
            - float(np.mean(right["rewards"]))
        ),
        "mean_delivered_fraction_difference": (
            float(np.mean(left["delivered_fractions"]))
            - float(np.mean(right["delivered_fractions"]))
        ),
        "mean_qber_difference": _optional_mean_difference(
            left["qber_values"],
            right["qber_values"],
        ),
        "mean_chsh_s_difference": _optional_mean_difference(
            left["chsh_values"],
            right["chsh_values"],
        ),
    }


def _trace_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_reward": float(np.mean(trace["rewards"])),
        "mean_delivered_fraction": float(
            np.mean(trace["delivered_fractions"])
        ),
        "mean_qber": _optional_mean(trace["qber_values"]),
        "mean_chsh_s": _optional_mean(trace["chsh_values"]),
        "public_outcome_counts": dict(trace["outcomes"]),
        "ancilla_observation_count": int(trace["ancilla_counts"]),
        "final_policy": np.asarray(
            trace["final_policy"], dtype=float
        ).tolist(),
    }


def _aggregate_identification(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_label = {}
    for label in sorted({
        str(row["true_hypothesis_label_for_scoring_only"])
        for row in rows
    }):
        selected = [
            row for row in rows
            if row["true_hypothesis_label_for_scoring_only"] == label
        ]
        by_label[label] = {
            "runs": len(selected),
            "correct_identification_count": sum(
                bool(row["identified_correctly_for_scoring_only"])
                for row in selected
            ),
            "mean_turns": float(np.mean([
                int(row["turns"]) for row in selected
            ])),
            "mean_route_inference_accuracy": float(np.mean([
                float(row["route_inference_accuracy_for_scoring_only"])
                for row in selected
            ])),
            "mean_inferred_route_confidence": float(np.mean([
                float(row["mean_inferred_route_confidence"])
                for row in selected
            ])),
        }
    return by_label


def _aggregate_presteering(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_label = {}
    for label in sorted({str(row["learner_label"]) for row in rows}):
        selected = [
            row for row in rows if str(row["learner_label"]) == label
        ]
        probe_tv = np.asarray([
            row["probe_vs_clean"]["final_policy_total_variation"]
            for row in selected
        ], dtype=float)
        clean_tv = np.asarray([
            row["clean_vs_independent_clean"][
                "final_policy_total_variation"
            ]
            for row in selected
        ], dtype=float)
        by_label[label] = {
            "runs": len(selected),
            "mean_probe_vs_clean_policy_tv": float(np.mean(probe_tv)),
            "mean_clean_vs_independent_clean_policy_tv": float(
                np.mean(clean_tv)
            ),
            "mean_probe_excess_policy_tv_over_clean_noise": float(
                np.mean(probe_tv - clean_tv)
            ),
            "mean_probe_vs_clean_reward_difference": float(np.mean([
                row["probe_vs_clean"]["mean_reward_difference"]
                for row in selected
            ])),
            "mean_probe_vs_clean_route_sequence_agreement": float(np.mean([
                row["probe_vs_clean"]["route_sequence_agreement"]
                for row in selected
            ])),
            "per_seed_probe_vs_clean_policy_tv": probe_tv.tolist(),
            "per_seed_clean_noise_policy_tv": clean_tv.tolist(),
        }
    return by_label


def _validate_matched_physics_configs(
    configs: Mapping[str, HistogramCoLearningConfig],
) -> None:
    fields = (
        "requested_pairs_per_turn",
        "game_mode",
        "repeater_service_profile_path",
        "accepted_alive_time_s",
        "age_points_per_edge",
        "allow_known_1550_to_1558_detector_approximation",
        "fock_cutoff",
        "evaluated_cache_size",
    )
    rows = list(configs.values())
    reference = rows[0]
    for config in rows[1:]:
        changed = [
            name for name in fields
            if getattr(config, name) != getattr(reference, name)
        ]
        if changed:
            raise ValueError(
                "learner smoke configs changed physics fields: "
                + ", ".join(changed)
            )


def _learner_config_summary(
    config: HistogramCoLearningConfig,
) -> dict[str, Any]:
    return {
        "alice_learner_mode": config.alice_learner_mode,
        "structured_exploration_mode": config.structured_exploration_mode,
        "alice_tsallis_inf_estimator": (
            config.resolved_alice_tsallis_inf_estimator
        ),
        "alice_tsallis_inf_eta_scale": config.alice_tsallis_inf_eta_scale,
        "alice_exp3_eta_c": config.alice_exp3_eta_c,
        "requested_pairs_per_turn": config.requested_pairs_per_turn,
    }


def _optional_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _optional_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if not left or not right:
        return None
    return float(np.mean(left) - np.mean(right))
