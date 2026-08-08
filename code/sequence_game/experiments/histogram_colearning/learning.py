"""Partial-feedback Alice/Eve bandit co-learning on histogram transcripts."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence

import numpy as np

from sequence_game.eve.feedback import (
    EVE_FEEDBACK_MODEL,
    eve_feedback_from_public_utility,
)
from sequence_game.game_theory import (
    Exp3,
    Exp3PlusPlus,
    LinearExp2,
    TsallisINF,
    solve_zero_sum_lp,
)

from .config import (
    ALICE_LEARNER_DAG_PATH_COMBAND,
    ALICE_LEARNER_DAG_PATH_EXP2,
    ALICE_LEARNER_EXP3_PLUS_PLUS,
    ALICE_LEARNER_FLAT_EXP3,
    ALICE_LEARNER_ITO_DAG_FTRL,
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_MAITI_DAG_FTRL,
    ALICE_LEARNER_TSALLIS_INF,
    EVE_LEARNER_EXP3_PLUS_PLUS,
    EVE_LEARNER_FLAT_EXP3,
    EVE_LEARNER_LINEAR_ACTION_EXP2,
    EVE_LEARNER_TSALLIS_INF,
    HistogramCoLearningConfig,
)
from .corpus import HistogramGameCase
from .learner_protocol import (
    PlayerLearner,
    ScalarFeedbackLearner,
    policy_distribution_statistics,
)
from .physics import HistogramPhysicsPool, RoutePhysicsCalibration
from .path_polytope import HistogramDAGComBand, HistogramDAGPathExp2
from .modern_path_learners import (
    HistogramItoPathFTRL,
    HistogramMaitiPathFTRL,
)
from .structure import eve_reward_embedding, route_resource_embedding


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PublicServiceUtility:
    """Bell-qualified saturated-demand service normalized to the best route."""

    reward: float
    bell_qualified: bool
    route_rate_ratio: float
    observed_delivery_fraction: float
    clean_delivery_probability: float
    relative_delivery_yield: float
    estimated_bell_qualified_rate_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "bell_qualified_saturated_service_rate_v1",
            "reward": self.reward,
            "bell_qualified": self.bell_qualified,
            "route_rate_ratio": self.route_rate_ratio,
            "observed_delivery_fraction": self.observed_delivery_fraction,
            "clean_delivery_probability": self.clean_delivery_probability,
            "relative_delivery_yield": self.relative_delivery_yield,
            "estimated_bell_qualified_rate_hz": (
                self.estimated_bell_qualified_rate_hz
            ),
            "demand_model": "saturated_pair_demand",
        }


@dataclass(frozen=True)
class AlicePublicSignalAssessment:
    """Observed finite-transcript decision versus latent histogram quality."""

    classification: str
    classifiable_quality_signal: bool
    false_signal: bool
    observed_bell_qualified: bool
    latent_bell_qualified: bool | None
    expected_qber: float | None
    expected_chsh_s: float | None
    qber_threshold: float | None
    chsh_threshold: float | None


@dataclass(frozen=True)
class HistogramCoLearningSummary:
    case: dict[str, Any]
    config: dict[str, Any]
    turns: int
    elapsed_seconds: float
    actions: tuple[dict[str, Any], ...]
    route_calibrations: tuple[dict[str, Any], ...]
    alice_prior: np.ndarray
    eve_prior: np.ndarray
    alice_learner: dict[str, Any]
    eve_learner: dict[str, Any]
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray
    alice_current_strategy: np.ndarray
    eve_current_strategy: np.ndarray
    mean_alice_utility: float
    final_alice_utility: float
    mean_bell_qualified_rate_hz: float
    final_bell_qualified_rate_hz: float
    mean_eve_public_utility: float
    mean_eve_bandit_reward: float
    target_overlap_rate_diagnostic: float
    information_observation_rate: float
    alice_false_signal_telemetry: dict[str, Any]
    outcome_counts: dict[str, int]
    learning_curve: tuple[dict[str, Any], ...]
    empirical_game: dict[str, Any]
    online_learning_telemetry: dict[str, Any]
    physics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "compiled_histogram_transcript",
            "runtime_engines": ["compiled_histogram_transcript_step"],
            "deprecated_sequence_backend_executed": False,
            "scientific_scope": {
                "alice_network_physics": "empirical_fit",
                "alice_network_claim_tier": "near_term_composite",
                "eve_attack_instruments": "modeled_attack",
                "eve_action_cost": "modeled_control_cost",
                "service_utility": "derived_saturated_demand_control_payoff",
                "security_claim": "none",
            },
            "case": dict(self.case),
            "config": dict(self.config),
            "turns": self.turns,
            "elapsed_seconds": self.elapsed_seconds,
            "turns_per_second": (
                self.turns / self.elapsed_seconds
                if self.elapsed_seconds > 0.0 else None
            ),
            "actions": [dict(row) for row in self.actions],
            "route_calibrations": [
                dict(row) for row in self.route_calibrations
            ],
            "alice_prior": self.alice_prior.tolist(),
            "eve_prior": self.eve_prior.tolist(),
            "alice_learner": dict(self.alice_learner),
            "eve_learner": dict(self.eve_learner),
            "alice_strategy": self.alice_strategy.tolist(),
            "eve_strategy": self.eve_strategy.tolist(),
            "alice_current_strategy": self.alice_current_strategy.tolist(),
            "eve_current_strategy": self.eve_current_strategy.tolist(),
            "mean_alice_utility": self.mean_alice_utility,
            "final_alice_utility": self.final_alice_utility,
            "mean_bell_qualified_rate_hz": self.mean_bell_qualified_rate_hz,
            "final_bell_qualified_rate_hz": self.final_bell_qualified_rate_hz,
            "mean_eve_public_utility": self.mean_eve_public_utility,
            "mean_eve_bandit_reward": self.mean_eve_bandit_reward,
            "target_overlap_rate_diagnostic": (
                self.target_overlap_rate_diagnostic
            ),
            "information_observation_rate": self.information_observation_rate,
            "alice_false_signal_telemetry": dict(
                self.alice_false_signal_telemetry
            ),
            "outcome_counts": dict(self.outcome_counts),
            "learning_curve": [dict(row) for row in self.learning_curve],
            "empirical_game": dict(self.empirical_game),
            "online_learning_telemetry": dict(
                self.online_learning_telemetry
            ),
            "physics": dict(self.physics),
            "feedback_boundary": {
                "eve_feedback_model": EVE_FEEDBACK_MODEL,
                "alice_feedback_model": (
                    "finite public delivery/QBER/CHSH transcript decision"
                ),
                "alice_false_signal_counterfactual_disclosed_to_learner": False,
                "learner_signal": (
                    "affine-normalized public service complement minus declared "
                    "action cost"
                ),
                "target_overlap_disclosed_to_learner": False,
                "information_gain_rewarded": False,
                "local_instrument_records_rewarded": False,
            },
        }


def run_histogram_colearning(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    physics: HistogramPhysicsPool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> HistogramCoLearningSummary:
    """Run simultaneous partial-feedback route and attack bandit learners."""

    started = time.perf_counter()
    active_physics = physics or HistogramPhysicsPool(case, config)
    physics_before = active_physics.diagnostics()
    calibrations = active_physics.calibrations
    actions = case.actions(config.game_mode)
    if not actions:
        raise ValueError("co-learning requires at least one Eve action")
    alice_prior = capacity_weighted_route_prior(
        [row.service_rate_hz for row in calibrations],
        mix=config.alice_physics_prior_mix,
    )
    eve_prior = (
        family_balanced_action_prior(actions)
        if config.eve_learner_mode == EVE_LEARNER_FLAT_EXP3
        else np.full(len(actions), 1.0 / len(actions), dtype=float)
    )
    alice_implementation, alice_learner = build_histogram_alice_learner(
        case=case,
        config=config,
        prior=alice_prior,
    )
    eve_implementation, eve_learner = build_histogram_eve_learner(
        case=case,
        actions=actions,
        config=config,
        prior=eve_prior,
    )
    alice = PlayerLearner(
        player="alice",
        action_ids=tuple(str(route["route_id"]) for route in case.routes),
        learner=alice_implementation,
        metadata=alice_learner,
    )
    eve = PlayerLearner(
        player="eve",
        action_ids=tuple(str(action["id"]) for action in actions),
        learner=eve_implementation,
        metadata=eve_learner,
    )
    reference_rate = active_physics.reference_service_rate_hz
    costs = np.asarray(
        [float(action.get("cost", 0.0)) for action in actions],
        dtype=float,
    )
    maximum_cost = float(costs.max(initial=0.0))
    matrix = _EmpiricalUtilityMatrix(len(case.routes), len(actions))
    outcomes: Counter[str] = Counter()
    curve: list[dict[str, Any]] = []
    final_alice: deque[float] = deque(maxlen=min(config.final_window, config.turns))
    final_rates: deque[float] = deque(maxlen=min(config.final_window, config.turns))
    alice_sum = 0.0
    rate_sum = 0.0
    eve_public_sum = 0.0
    eve_bandit_sum = 0.0
    overlap_sum = 0.0
    information_observation_sum = 0.0
    signal_counts: Counter[str] = Counter()
    false_signal_reward_delta_sum = 0.0
    false_signal_reward_delta_abs_sum = 0.0
    counterfactual_log_weight_offset = np.zeros(len(case.routes), dtype=float)
    counterfactual_policy = alice.policy(turn=1)
    progress_every = max(1, min(10_000, config.turns // 10 or 1))

    for turn in range(config.turns):
        route_index = alice.sample()
        action_index = eve.sample()
        action = actions[action_index]
        result = active_physics.step(
            route_index,
            action,
            seed=config.seed + 100_000,
            step_index=turn,
        )
        utility = bell_qualified_rate_utility(
            result=result,
            calibration=calibrations[route_index],
            reference_service_rate_hz=reference_rate,
        )
        signal = assess_alice_public_signal(result)
        corrected_utility = bell_qualified_rate_utility(
            result=result,
            calibration=calibrations[route_index],
            reference_service_rate_hz=reference_rate,
            bell_qualified_override=signal.latent_bell_qualified,
        )
        cost = costs[action_index]
        feedback = eve_feedback_from_public_utility(
            public_outcome=result.public_outcome,
            alice_utility=utility.reward,
            attack_cost=float(cost),
            local_observation_kind=(
                "attack_instrument_private_record"
                if result.eve_observation_count > 0 else "none"
            ),
            local_observation_count=result.eve_observation_count,
        )
        eve_bandit_reward = normalize_eve_bandit_reward(
            feedback.reward,
            maximum_cost=maximum_cost,
        )
        reward_delta = corrected_utility.reward - utility.reward
        counterfactual_state_replay = isinstance(
            alice.learner,
            (
                Exp3PlusPlus,
                HistogramMaitiPathFTRL,
                HistogramItoPathFTRL,
                TsallisINF,
            ),
        )
        if counterfactual_state_replay:
            alice.learner.update_counterfactual(
                utility.reward,
                corrected_utility.reward,
            )
        else:
            counterfactual_log_weight_offset += (
                alice.learner.counterfactual_log_weight_correction(reward_delta)
            )
        alice.update(utility.reward)
        eve.update(eve_bandit_reward)

        matrix.observe(route_index, action_index, utility.reward)
        outcomes[str(result.public_outcome)] += 1
        alice_sum += utility.reward
        rate_sum += utility.estimated_bell_qualified_rate_hz
        eve_public_sum += feedback.reward
        eve_bandit_sum += eve_bandit_reward
        # These two quantities are offline diagnostics only and never enter
        # either learner's update.
        overlap_sum += float(result.target_active and action["attack_type"] != "no_attack")
        information_observation_sum += float(result.eve_observation_count > 0)
        signal_counts[signal.classification] += 1
        signal_counts["classifiable"] += int(signal.classifiable_quality_signal)
        signal_counts["false"] += int(signal.false_signal)
        false_signal_reward_delta_sum += reward_delta
        false_signal_reward_delta_abs_sum += abs(reward_delta)
        final_alice.append(utility.reward)
        final_rates.append(utility.estimated_bell_qualified_rate_hz)

        step_number = turn + 1
        if (
            step_number == 1
            or step_number == config.turns
            or step_number % config.record_every == 0
        ):
            # Counterfactual false-signal attribution is offline telemetry, not
            # learner feedback.  Solving its FTRL state on unrecorded turns
            # doubled modern-learner optimization work without changing any
            # reported value or policy update.
            if counterfactual_state_replay:
                counterfactual_policy = alice.learner.counterfactual_policy(
                    turn=step_number + 1
                )
            else:
                counterfactual_policy = alice.learner.policy_with_log_weight_offset(
                    counterfactual_log_weight_offset,
                    turn=step_number + 1,
                )
            curve.append(_curve_row(
                step_number=step_number,
                alice=alice,
                eve=eve,
                alice_mean=alice_sum / step_number,
                rate_mean=rate_sum / step_number,
                eve_public_mean=eve_public_sum / step_number,
                target_overlap_mean=overlap_sum / step_number,
                outcome_counts=outcomes,
                signal_counts=signal_counts,
                factual_alice_policy=alice.policy(turn=step_number + 1),
                counterfactual_alice_policy=counterfactual_policy,
                corrected_reward_delta_mean=(
                    false_signal_reward_delta_sum / step_number
                ),
            ))
        if progress_callback and (
            step_number == 1
            or step_number == config.turns
            or step_number % progress_every == 0
        ):
            progress_callback(
                f"turn {step_number}/{config.turns}; "
                f"Alice utility={alice_sum / step_number:.4f}; "
                f"Bell-qualified rate={rate_sum / step_number:.3f} pairs/s; "
                f"Eve public utility={eve_public_sum / step_number:.4f}; "
                f"accepted={outcomes['accepted'] / step_number:.2%}"
            )

    alice_strategy = alice.empirical_strategy()
    eve_strategy = eve.empirical_strategy()
    empirical_game = _empirical_game_diagnostics(
        learning_matrix=matrix,
        physics=active_physics,
        calibrations=calibrations,
        actions=actions,
        action_costs=costs,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        reference_service_rate_hz=reference_rate,
        config=config,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - started
    if isinstance(
        alice.learner, (HistogramMaitiPathFTRL, HistogramItoPathFTRL)
    ):
        alice.metadata["runtime_diagnostics"] = (
            alice.learner.full_diagnostics()
        )
    elif isinstance(alice.learner, (Exp3PlusPlus, TsallisINF)):
        alice.metadata["runtime_diagnostics"] = alice.learner.diagnostics()
    if isinstance(eve.learner, (Exp3PlusPlus, TsallisINF)):
        eve.metadata["runtime_diagnostics"] = eve.learner.diagnostics()
    online_learning_telemetry = _online_learning_telemetry(
        alice=alice,
        eve=eve,
        joint_action_counts=matrix.count,
        turns=config.turns,
    )
    return HistogramCoLearningSummary(
        case=case.to_dict(),
        config=config.to_dict(),
        turns=config.turns,
        elapsed_seconds=elapsed,
        actions=actions,
        route_calibrations=tuple(row.to_dict() for row in calibrations),
        alice_prior=alice_prior,
        eve_prior=eve_prior,
        alice_learner=alice.metadata,
        eve_learner=eve.metadata,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        alice_current_strategy=alice.policy(turn=config.turns + 1),
        eve_current_strategy=eve.policy(turn=config.turns + 1),
        mean_alice_utility=alice_sum / config.turns,
        final_alice_utility=float(np.mean(final_alice)),
        mean_bell_qualified_rate_hz=rate_sum / config.turns,
        final_bell_qualified_rate_hz=float(np.mean(final_rates)),
        mean_eve_public_utility=eve_public_sum / config.turns,
        mean_eve_bandit_reward=eve_bandit_sum / config.turns,
        target_overlap_rate_diagnostic=overlap_sum / config.turns,
        information_observation_rate=(
            information_observation_sum / config.turns
        ),
        alice_false_signal_telemetry=_false_signal_summary(
            signal_counts=signal_counts,
            turns=config.turns,
            factual_policy=alice.policy(turn=config.turns + 1),
            counterfactual_policy=counterfactual_policy,
            log_weight_offset=counterfactual_log_weight_offset,
            reward_delta_sum=false_signal_reward_delta_sum,
            reward_delta_abs_sum=false_signal_reward_delta_abs_sum,
            route_ids=[str(row["route_id"]) for row in case.routes],
            learner_mode=config.alice_learner_mode,
            counterfactual_state=(
                alice.learner.full_diagnostics().get("counterfactual_solver")
                if isinstance(
                    alice.learner,
                    (HistogramMaitiPathFTRL, HistogramItoPathFTRL),
                )
                else None
            ),
        ),
        outcome_counts=dict(outcomes),
        learning_curve=tuple(curve),
        empirical_game=empirical_game,
        online_learning_telemetry=online_learning_telemetry,
        physics=_physics_run_diagnostics(
            physics_before,
            active_physics.diagnostics(),
        ),
    )


def build_histogram_alice_learner(
    *,
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    prior: np.ndarray,
) -> tuple[
    ScalarFeedbackLearner,
    dict[str, Any],
]:
    common = {
        "gamma": config.alice_exp3_gamma,
        "rng": np.random.default_rng([config.seed, 11]),
        "schedule_mode": config.exp3_schedule_mode,
        "eta_c": config.alice_exp3_eta_c,
        "t0": config.exp3_t0,
        "gamma_max": config.exp3_gamma_max,
    }
    if config.alice_learner_mode == ALICE_LEARNER_FLAT_EXP3:
        return (
            Exp3(len(case.routes), prior=prior, **common),
            {
                "mode": ALICE_LEARNER_FLAT_EXP3,
                "algorithm": "Exp3",
                "feedback": "selected-route scalar public service reward only",
                "cross_route_information_sharing": False,
                "initial_route_prior": "uniform" if np.allclose(
                    prior, np.full(len(prior), 1.0 / len(prior))
                ) else "explicit_physics_prior_sensitivity",
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_TSALLIS_INF:
        learner = TsallisINF(
            len(case.routes),
            estimator=config.resolved_alice_tsallis_inf_estimator,
            eta_scale=config.alice_tsallis_inf_eta_scale,
            rng=np.random.default_rng([config.seed, 11]),
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_TSALLIS_INF,
                "algorithm": "Tsallis-INF_alpha_half",
                "literature_source": {
                    "citation": (
                        "Zimmert and Seldin, Tsallis-INF: An Optimal "
                        "Algorithm for Stochastic and Adversarial Bandits, "
                        "JMLR 22(28), 2021"
                    ),
                    "arxiv": "1807.07623",
                    "mapped_sections": ["4.1", "4.2", "5.1"],
                },
                "feedback": "selected-route scalar public service loss only",
                "reward_to_loss_mapping": "loss = 1 - public_service_reward",
                "cross_route_information_sharing": False,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "uniform_action_symmetry",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.alice_tsallis_inf_eta_scale == 1.0
                        else "configured_nondefault_sensitivity"
                    ),
                    "estimator": config.resolved_alice_tsallis_inf_estimator,
                    "eta_scale": config.alice_tsallis_inf_eta_scale,
                    "eta_schedule": (
                        "4/sqrt(t) for reduced-variance estimator; "
                        "2/sqrt(t) for importance-weighted estimator"
                    ),
                },
                "theorem_boundary": (
                    "the source proves adaptive-adversary pseudo-regret and "
                    "best-of-both-worlds rates; this run measures a "
                    "simultaneous stochastic two-learner process and makes no "
                    "finite-run equilibrium theorem claim"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_EXP3_PLUS_PLUS:
        learner = Exp3PlusPlus(
            len(case.routes),
            confidence_alpha=config.exp3pp_confidence_alpha,
            exploration_beta=config.alice_exp3pp_exploration_beta,
            eta_scale=config.alice_exp3pp_eta_scale,
            rng=np.random.default_rng([config.seed, 11]),
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_EXP3_PLUS_PLUS,
                "algorithm": "EXP3++_improved_LCB_gap",
                "literature_sources": [
                    {
                        "citation": (
                            "Seldin and Slivkins, One Practical Algorithm for "
                            "Both Stochastic and Adversarial Bandits, ICML 2014"
                        ),
                        "mapped_sections": ["Algorithm 1"],
                    },
                    {
                        "citation": (
                            "Seldin and Lugosi, An Improved Parametrization "
                            "and Analysis of the EXP3++ Algorithm, COLT 2017"
                        ),
                        "arxiv": "1702.06103",
                        "mapped_sections": ["Algorithms 1--2", "Theorem 3"],
                    },
                ],
                "feedback": "selected-route scalar public service loss only",
                "reward_to_loss_mapping": "loss = 1 - public_service_reward",
                "cross_route_information_sharing": False,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": (
                    "uniform_action_symmetry_with_randomized_one_pass_initialization"
                ),
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.exp3pp_confidence_alpha == 3.0
                        and config.alice_exp3pp_exploration_beta == 256.0
                        and config.alice_exp3pp_eta_scale == 1.0
                        else "configured_nondefault_sensitivity"
                    ),
                    "confidence_alpha": config.exp3pp_confidence_alpha,
                    "exploration_beta": (
                        config.alice_exp3pp_exploration_beta
                    ),
                    "eta_scale": config.alice_exp3pp_eta_scale,
                    "arm_floor": (
                        "epsilon_t(a)=min(1/(2K), "
                        "0.5*sqrt(log(K)/(tK)), "
                        "beta*log(t)/(t*gap_lcb_t(a)^2))"
                    ),
                },
                "theorem_boundary": (
                    "the improved source's adversarial guarantee assumes an "
                    "oblivious loss sequence; simultaneous co-learning is an "
                    "empirical control outside that theorem's stated scope"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_LINEAR_ROUTE_EXP2:
        embedding = route_resource_embedding(case)
        learner = LinearExp2(
            embedding.embedding,
            exploration_mode=config.structured_exploration_mode,
            **common,
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_LINEAR_ROUTE_EXP2,
                "algorithm": "adversarial_linear_Exp2",
                "feedback": "selected-route scalar public service reward only",
                "cross_route_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "uniform_reward_weights",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "route_features": embedding.to_dict(),
                "exploration_design": learner.exploration_design.to_dict(),
                "scope": (
                    "structured linear bandit; topology features and the "
                    "exploration design contain no route-quality or physics "
                    "ranking"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_DAG_PATH_COMBAND:
        learner = HistogramDAGComBand(
            case,
            horizon=config.turns,
            rng=np.random.default_rng([config.seed, 11]),
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_DAG_PATH_COMBAND,
                "algorithm": "COMBAND_bounded_DAG_path_polytope",
                "literature_source": {
                    "citation": (
                        "Cesa-Bianchi and Lugosi, Combinatorial Bandits, "
                        "JCSS 78(5), 2012, Algorithm 1 and Theorem 1"
                    ),
                    "doi": "10.1016/j.jcss.2012.01.001",
                },
                "feedback": "selected-route scalar public service loss only",
                "reward_to_loss_mapping": "loss = 1 - public_service_reward",
                "cross_route_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "uniform_path_distribution",
                "initial_sampling_distribution": learner.policy(
                    turn=1
                ).tolist(),
                "schedule": learner.schedule.to_dict(),
                "path_polytope": learner.adapter.diagnostics(
                    horizon=config.turns
                ),
                "route_features": {
                    "rank": learner.adapter.polytope.dimension,
                    "route_count": len(case.routes),
                    "representation": "compressed_edge_incidence_span",
                },
                "scope": (
                    "non-enumerative bounded-DAG path sampling and COMBAND "
                    "updates; finite route features are retained only for the "
                    "compiled-physics lookup and comparable diagnostics"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_DAG_PATH_EXP2:
        learner = HistogramDAGPathExp2(
            case,
            eta_c=config.alice_exp3_eta_c,
            t0=config.exp3_t0,
            gamma_max=config.exp3_gamma_max,
            rng=np.random.default_rng([config.seed, 11]),
        )
        initial = learner.learner.schedule_at_turn(1)
        final = learner.learner.schedule_at_turn(config.turns)
        return (
            learner,
            {
                "mode": ALICE_LEARNER_DAG_PATH_EXP2,
                "algorithm": "anytime_LinearExp2_bounded_DAG_path_polytope",
                "algorithmic_sources": [
                    {
                        "citation": (
                            "Cesa-Bianchi and Lugosi, Combinatorial Bandits, "
                            "JCSS 78(5), 2012"
                        ),
                        "doi": "10.1016/j.jcss.2012.01.001",
                    },
                    {
                        "citation": (
                            "Combes et al., Combinatorial Bandits Revisited, "
                            "NeurIPS 2015"
                        ),
                        "arxiv": "1502.03475",
                    },
                ],
                "feedback": "selected-route scalar public service reward only",
                "action_independent_baseline": (
                    "maximizes reward directly; equivalent path regret to "
                    "minimizing 1-reward without estimating the common unit "
                    "baseline"
                ),
                "cross_route_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "uniform_path_distribution",
                "initial_sampling_distribution": learner.policy(
                    turn=1
                ).tolist(),
                "schedule": {
                    "mode": "topology_only_anytime",
                    "formula": (
                        "eta_t=eta_c*sqrt(log(N)/(d*(t+t0))); "
                        "gamma_t=min(gamma_max,d*eta_t)"
                    ),
                    "eta_c": config.alice_exp3_eta_c,
                    "t0": config.exp3_t0,
                    "gamma_max": config.exp3_gamma_max,
                    "initial": {
                        "eta": initial.eta,
                        "gamma": initial.gamma,
                    },
                    "horizon": {
                        "turn": config.turns,
                        "eta": final.eta,
                        "gamma": final.gamma,
                    },
                },
                "path_polytope": learner.adapter.diagnostics(
                    horizon=config.turns
                ),
                "route_features": {
                    "rank": learner.adapter.polytope.dimension,
                    "route_count": len(case.routes),
                    "representation": "compressed_edge_incidence_span",
                },
                "scope": (
                    "non-enumerative bounded-DAG normalization, sampling, and "
                    "moment estimation with the same topology-only anytime "
                    "schedule as the finite LinearExp2 reference"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_MAITI_DAG_FTRL:
        learner = HistogramMaitiPathFTRL(
            case,
            horizon=(config.maiti_schedule_horizon or config.turns),
            failure_probability=config.maiti_failure_probability,
            eta_scale=config.maiti_eta_scale,
            implicit_exploration_scale=(
                config.maiti_implicit_exploration_scale
            ),
            reward_loss_baseline=config.maiti_reward_loss_baseline,
            rng=np.random.default_rng([config.seed, 11]),
            solver_tolerance=config.path_ftrl_solver_tolerance,
            solver_max_iterations=config.path_ftrl_solver_max_iterations,
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_MAITI_DAG_FTRL,
                "algorithm": "Maiti_COLT2025_centroid_Tsallis_FTRL",
                "literature_source": {
                    "citation": (
                        "Maiti, Fan, Jamieson, Ratliff, and Farina, Efficient "
                        "Near-Optimal Algorithm for Online Shortest Paths in "
                        "Directed Acyclic Graphs with Bandit Feedback Against "
                        "Adaptive Adversaries, COLT 2025"
                    ),
                    "url": "https://proceedings.mlr.press/v291/maiti25a.html",
                    "mapped_sections": ["3.2", "3.3", "Algorithm 2"],
                },
                "feedback": "selected-route scalar public service loss only",
                "reward_to_loss_mapping": (
                    "loss = baseline - public_service_reward"
                ),
                "loss_range_mapping": (
                    "[baseline-1,baseline] subset of paper's [-1,1] range"
                ),
                "cross_route_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "topology_only_FTRL_analytic_center",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.maiti_eta_scale == 1.0
                        and config.maiti_implicit_exploration_scale == 1.0
                        and config.maiti_reward_loss_baseline == 1.0
                        else "configured_nondefault_sensitivity"
                    ),
                    "calibration_artifact": (
                        "docs/modern_path_learner_calibration.json"
                    ),
                    "eta_scale": config.maiti_eta_scale,
                    "implicit_exploration_scale": (
                        config.maiti_implicit_exploration_scale
                    ),
                    "failure_probability": config.maiti_failure_probability,
                    "reward_loss_baseline": (
                        config.maiti_reward_loss_baseline
                    ),
                    "schedule_horizon": (
                        config.maiti_schedule_horizon or config.turns
                    ),
                },
                "path_geometry": learner.full_diagnostics(),
                "additive_loss_claim_boundary": (
                    "the regret theorem requires an edge-additive path loss; "
                    "histogram public utility is an empirically tested "
                    "structured approximation rather than a declared identity"
                ),
            },
        )
    if config.alice_learner_mode == ALICE_LEARNER_ITO_DAG_FTRL:
        learner = HistogramItoPathFTRL(
            case,
            horizon=config.turns,
            regularizer=config.ito_regularizer,
            eta_scale=config.ito_eta_scale,
            beta=config.ito_beta,
            rng=np.random.default_rng([config.seed, 11]),
            solver_tolerance=config.path_ftrl_solver_tolerance,
            solver_max_iterations=config.path_ftrl_solver_max_iterations,
        )
        return (
            learner,
            {
                "mode": ALICE_LEARNER_ITO_DAG_FTRL,
                "algorithm": "Ito_NeurIPS2025_BOBW_flow_FTRL",
                "literature_source": {
                    "citation": (
                        "Ito, Jamieson, Luo, Maiti, and Tsuchiya, Adapting to "
                        "Stochastic and Adversarial Losses in Episodic MDPs with "
                        "Aggregate Bandit Feedback, NeurIPS 2025"
                    ),
                    "url": (
                        "https://proceedings.neurips.cc/paper_files/paper/2025/"
                        "hash/0a50497b94c379b16dfb9b0c959a708e-"
                        "Abstract-Conference.html"
                    ),
                    "mapped_sections": ["2.3", "3.1", "4.1", "Appendix C"],
                },
                "feedback": "selected-route scalar public service loss only",
                "reward_to_loss_mapping": "loss = 1 - public_service_reward",
                "cross_route_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "initial_route_prior": "topology_only_FTRL_analytic_center",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.ito_eta_scale == 1.0
                        and config.ito_beta == 2.0
                        else "synthetic_edge_loss_calibrated"
                    ),
                    "calibration_artifact": (
                        "docs/modern_path_learner_calibration.json"
                    ),
                    "regularizer": config.ito_regularizer,
                    "eta_scale": config.ito_eta_scale,
                    "beta": config.ito_beta,
                },
                "path_geometry": learner.full_diagnostics(),
                "additive_loss_claim_boundary": (
                    "the BOBW theorem requires aggregate feedback whose "
                    "conditional mean is an edge-additive path loss; simultaneous "
                    "Eve learning is covered by the adversarial rather than the "
                    "stationary stochastic interpretation"
                ),
            },
        )
    raise ValueError(f"unsupported Alice learner {config.alice_learner_mode!r}")


def build_histogram_eve_learner(
    *,
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
    config: HistogramCoLearningConfig,
    prior: np.ndarray,
) -> tuple[ScalarFeedbackLearner, dict[str, Any]]:
    """Construct Eve behind the same scalar-feedback boundary as Alice.

    Independent-arm controls are symmetric across players.  None receives
    Alice's route, target overlap, unplayed rewards, or private simulator state.
    """

    if config.eve_learner_mode == EVE_LEARNER_FLAT_EXP3:
        learner = Exp3(
            len(actions),
            gamma=config.eve_exp3_gamma,
            rng=np.random.default_rng([config.seed, 23]),
            schedule_mode=config.exp3_schedule_mode,
            eta_c=config.eve_exp3_eta_c,
            t0=config.exp3_t0,
            gamma_max=config.exp3_gamma_max,
            prior=prior,
        )
        return (
            learner,
            {
                "mode": EVE_LEARNER_FLAT_EXP3,
                "algorithm": "Exp3",
                "feedback": (
                    "selected-attack scalar affine-normalized public service "
                    "complement minus declared action cost only"
                ),
                "cross_action_information_sharing": False,
                "initial_action_prior": "family_balanced_attack_exploration",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "target_overlap_disclosed": False,
                "private_instrument_record_disclosed": False,
            },
        )
    if config.eve_learner_mode == EVE_LEARNER_TSALLIS_INF:
        learner = TsallisINF(
            len(actions),
            estimator=config.resolved_eve_tsallis_inf_estimator,
            eta_scale=config.eve_tsallis_inf_eta_scale,
            rng=np.random.default_rng([config.seed, 23]),
        )
        return (
            learner,
            {
                "mode": EVE_LEARNER_TSALLIS_INF,
                "algorithm": "Tsallis-INF_alpha_half",
                "literature_source": {
                    "citation": (
                        "Zimmert and Seldin, Tsallis-INF: An Optimal "
                        "Algorithm for Stochastic and Adversarial Bandits, "
                        "JMLR 22(28), 2021"
                    ),
                    "arxiv": "1807.07623",
                    "mapped_sections": ["4.1", "4.2", "5.1"],
                },
                "feedback": (
                    "selected-attack scalar affine-normalized public service "
                    "complement minus declared action cost only"
                ),
                "reward_to_loss_mapping": "loss = 1 - eve_bandit_reward",
                "cross_action_information_sharing": False,
                "initial_action_prior": "uniform_action_symmetry",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.eve_tsallis_inf_eta_scale == 1.0
                        else "configured_nondefault_sensitivity"
                    ),
                    "estimator": config.resolved_eve_tsallis_inf_estimator,
                    "eta_scale": config.eve_tsallis_inf_eta_scale,
                },
                "theorem_boundary": (
                    "adaptive-adversary pseudo-regret does not itself prove "
                    "finite-turn equilibrium convergence for this stochastic "
                    "two-learner process"
                ),
                "target_overlap_disclosed": False,
                "private_instrument_record_disclosed": False,
            },
        )
    if config.eve_learner_mode == EVE_LEARNER_EXP3_PLUS_PLUS:
        learner = Exp3PlusPlus(
            len(actions),
            confidence_alpha=config.exp3pp_confidence_alpha,
            exploration_beta=config.eve_exp3pp_exploration_beta,
            eta_scale=config.eve_exp3pp_eta_scale,
            rng=np.random.default_rng([config.seed, 23]),
        )
        return (
            learner,
            {
                "mode": EVE_LEARNER_EXP3_PLUS_PLUS,
                "algorithm": "EXP3++_improved_LCB_gap",
                "literature_sources": [
                    {
                        "citation": (
                            "Seldin and Slivkins, One Practical Algorithm for "
                            "Both Stochastic and Adversarial Bandits, ICML 2014"
                        ),
                        "mapped_sections": ["Algorithm 1"],
                    },
                    {
                        "citation": (
                            "Seldin and Lugosi, An Improved Parametrization "
                            "and Analysis of the EXP3++ Algorithm, COLT 2017"
                        ),
                        "arxiv": "1702.06103",
                        "mapped_sections": ["Algorithms 1--2", "Theorem 3"],
                    },
                ],
                "feedback": (
                    "selected-attack scalar affine-normalized public service "
                    "complement minus declared action cost only"
                ),
                "reward_to_loss_mapping": "loss = 1 - eve_bandit_reward",
                "cross_action_information_sharing": False,
                "initial_action_prior": (
                    "uniform_action_symmetry_with_randomized_one_pass_initialization"
                ),
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "tuning": {
                    "status": (
                        "literature_default"
                        if config.exp3pp_confidence_alpha == 3.0
                        and config.eve_exp3pp_exploration_beta == 256.0
                        and config.eve_exp3pp_eta_scale == 1.0
                        else "configured_nondefault_sensitivity"
                    ),
                    "confidence_alpha": config.exp3pp_confidence_alpha,
                    "exploration_beta": config.eve_exp3pp_exploration_beta,
                    "eta_scale": config.eve_exp3pp_eta_scale,
                },
                "theorem_boundary": (
                    "the improved EXP3++ guarantee assumes an oblivious loss "
                    "sequence; simultaneous co-learning is an empirical "
                    "control outside that theorem's stated scope"
                ),
                "target_overlap_disclosed": False,
                "private_instrument_record_disclosed": False,
            },
        )
    if config.eve_learner_mode == EVE_LEARNER_LINEAR_ACTION_EXP2:
        embedding = eve_reward_embedding(
            case,
            actions,
            mode=config.eve_structured_feature_mode,
        )
        learner = LinearExp2(
            embedding.embedding,
            gamma=config.eve_exp3_gamma,
            rng=np.random.default_rng([config.seed, 23]),
            schedule_mode=config.exp3_schedule_mode,
            eta_c=config.eve_exp3_eta_c,
            t0=config.exp3_t0,
            gamma_max=config.exp3_gamma_max,
            exploration_mode=config.eve_structured_exploration_mode,
        )
        reward_terms = _eve_known_reward_term_span(
            embedding.embedding,
            actions,
        )
        return (
            learner,
            {
                "mode": EVE_LEARNER_LINEAR_ACTION_EXP2,
                "algorithm": "adversarial_linear_Exp2",
                "algorithmic_sources": [
                    {
                        "citation": (
                            "Cesa-Bianchi and Lugosi, Combinatorial Bandits, "
                            "JCSS 78(5), 2012"
                        ),
                        "doi": "10.1016/j.jcss.2012.01.001",
                        "mapped_sections": ["Algorithm 1", "Theorem 1"],
                    },
                    {
                        "citation": (
                            "Combes et al., Combinatorial Bandits Revisited, "
                            "NeurIPS 2015"
                        ),
                        "arxiv": "1502.03475",
                        "mapped_sections": ["Section 2", "Section 3"],
                    },
                ],
                "feedback": (
                    "selected-attack scalar affine-normalized public service "
                    "complement minus declared action cost only"
                ),
                "cross_action_information_sharing": True,
                "opponent_action_disclosed": False,
                "counterfactual_payoff_cells_disclosed": False,
                "target_overlap_disclosed": False,
                "private_instrument_record_disclosed": False,
                "initial_action_prior": "uniform_reward_weights",
                "initial_sampling_distribution": learner.policy(turn=1).tolist(),
                "action_features": embedding.to_dict(),
                "known_reward_term_span": reward_terms,
                "exploration_design": learner.exploration_design.to_dict(),
                "feature_selection_audit": {
                    "document": "docs/eve_bilinear_payoff_audit.md",
                    "mode": config.eve_structured_feature_mode,
                    "payoff_used_to_construct_online_features": False,
                    "long_case_public_utility_rmse": 0.005040,
                    "long_case_held_target_rmse": 0.005260,
                    "theorem_boundary": (
                        "the finite-action update is exact for the declared "
                        "feature span; the histogram public-utility matrix has "
                        "the separately reported model residual and is not "
                        "claimed to lie exactly in that span"
                    ),
                },
            },
        )
    raise ValueError(f"unsupported Eve learner {config.eve_learner_mode!r}")


def _eve_known_reward_term_span(
    features: np.ndarray,
    actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Certify that Eve's known affine reward terms lie in her feature span."""

    matrix = np.asarray(features, dtype=float)
    projection = matrix @ matrix.T
    vectors = {
        "constant": np.ones(len(actions), dtype=float),
        "declared_action_cost": np.asarray(
            [float(action.get("cost", 0.0)) for action in actions],
            dtype=float,
        ),
    }
    residuals = {
        name: float(np.max(np.abs(vector - projection @ vector)))
        for name, vector in vectors.items()
    }
    tolerance = 1e-10
    if any(residual > tolerance for residual in residuals.values()):
        raise ValueError(
            "Eve's known affine reward terms are outside the declared feature span"
        )
    return {
        "maximum_absolute_residuals": residuals,
        "numerical_tolerance": tolerance,
        "all_terms_represented": True,
        "normalization": (
            "(1 - Alice public utility - action cost + maximum cost) / "
            "(1 + maximum cost)"
        ),
    }


def bell_qualified_rate_utility(
    *,
    result: Any,
    calibration: RoutePhysicsCalibration,
    reference_service_rate_hz: float,
    bell_qualified_override: bool | None = None,
) -> PublicServiceUtility:
    """Public payoff for saturated demand with finite transcript monitoring.

    The controller's route renewal rate supplies timing. The sampled delivered
    fraction is divided by that route's clean random-time availability so clean
    availability is not counted twice. A failed Bell/QBER decision has zero
    usable quantum service even when physical pairs arrived.
    """

    reference = float(reference_service_rate_hz)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_service_rate_hz must be finite and > 0")
    clean_delivery = float(calibration.clean_delivery_probability)
    if not 0.0 < clean_delivery <= 1.0:
        raise ValueError("clean_delivery_probability must be in (0, 1]")
    requested = int(result.requested_pair_count)
    delivered = int(result.delivered_pair_count)
    if requested < 1 or delivered < 0 or delivered > requested:
        raise ValueError("invalid requested/delivered pair counts")
    observed = delivered / requested
    relative_yield = min(1.0, max(0.0, observed / clean_delivery))
    rate_ratio = min(1.0, max(0.0, calibration.service_rate_hz / reference))
    bell_qualified = (
        str(result.public_outcome) == "accepted"
        if bell_qualified_override is None
        else bool(bell_qualified_override)
    )
    reward = rate_ratio * relative_yield if bell_qualified else 0.0
    return PublicServiceUtility(
        reward=reward,
        bell_qualified=bell_qualified,
        route_rate_ratio=rate_ratio,
        observed_delivery_fraction=observed,
        clean_delivery_probability=clean_delivery,
        relative_delivery_yield=relative_yield,
        estimated_bell_qualified_rate_hz=reward * reference,
    )


def assess_alice_public_signal(result: Any) -> AlicePublicSignalAssessment:
    """Classify finite-transcript Alice feedback against latent state quality.

    Delivery failures are real service failures rather than false quality
    decisions, so they remain explicitly unclassified.  For a delivered
    transcript, the latent comparator applies the same public QBER/CHSH
    thresholds to the histogram expectations.  A false positive is a public
    acceptance of threshold-failing latent quality; a false negative is an
    abort of threshold-passing latent quality.
    """

    observed = str(getattr(result, "public_outcome", "")) == "accepted"
    expected_qber = _optional_finite_float(
        getattr(result, "expected_qber_from_histogram", None)
    )
    expected_chsh = _optional_finite_float(
        getattr(result, "chsh_S_expected", None)
    )
    qber_threshold = _optional_finite_float(
        getattr(result, "qber_abort_threshold", None)
    )
    chsh_threshold = _optional_finite_float(
        getattr(result, "chsh_abort_threshold", None)
    )
    if int(getattr(result, "delivered_pair_count", 0)) <= 0:
        return AlicePublicSignalAssessment(
            classification="delivery_failure_not_quality_signal",
            classifiable_quality_signal=False,
            false_signal=False,
            observed_bell_qualified=observed,
            latent_bell_qualified=None,
            expected_qber=expected_qber,
            expected_chsh_s=expected_chsh,
            qber_threshold=qber_threshold,
            chsh_threshold=chsh_threshold,
        )
    if None in (
        expected_qber,
        expected_chsh,
        qber_threshold,
        chsh_threshold,
    ):
        return AlicePublicSignalAssessment(
            classification="missing_latent_quality_reference",
            classifiable_quality_signal=False,
            false_signal=False,
            observed_bell_qualified=observed,
            latent_bell_qualified=None,
            expected_qber=expected_qber,
            expected_chsh_s=expected_chsh,
            qber_threshold=qber_threshold,
            chsh_threshold=chsh_threshold,
        )
    latent = bool(
        expected_qber <= qber_threshold
        and expected_chsh >= chsh_threshold
    )
    if observed and latent:
        classification = "true_accept"
    elif observed:
        classification = "false_accept"
    elif latent:
        classification = "false_abort"
    else:
        classification = "true_abort"
    return AlicePublicSignalAssessment(
        classification=classification,
        classifiable_quality_signal=True,
        false_signal=observed != latent,
        observed_bell_qualified=observed,
        latent_bell_qualified=latent,
        expected_qber=expected_qber,
        expected_chsh_s=expected_chsh,
        qber_threshold=qber_threshold,
        chsh_threshold=chsh_threshold,
    )


def capacity_weighted_route_prior(
    service_rates_hz: Sequence[float],
    *,
    mix: float,
) -> np.ndarray:
    rates = np.asarray(service_rates_hz, dtype=float)
    if rates.ndim != 1 or rates.size < 1:
        raise ValueError("service_rates_hz must be a non-empty vector")
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("service rates must be finite and > 0")
    if not 0.0 <= float(mix) < 1.0:
        raise ValueError("mix must be in [0, 1)")
    physical = rates / rates.sum()
    uniform = np.full(rates.size, 1.0 / rates.size)
    return (1.0 - float(mix)) * uniform + float(mix) * physical


def family_balanced_action_prior(
    actions: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Equal prior mass per physical attack family, then per target."""

    if not actions:
        raise ValueError("actions must be non-empty")
    groups: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        family = str(action.get("attack_type") or "unknown")
        groups.setdefault(family, []).append(index)
    prior = np.zeros(len(actions), dtype=float)
    family_mass = 1.0 / len(groups)
    for indices in groups.values():
        target_mass = family_mass / len(indices)
        prior[indices] = target_mass
    return prior


def normalize_eve_bandit_reward(
    raw_reward: float,
    *,
    maximum_cost: float,
) -> float:
    """Affine-map ``1 - Alice utility - cost`` into Exp3's [0, 1] range.

    Using one experiment-wide affine map preserves every Eve action ordering;
    clipping negative rewards at zero would erase the distinction between a
    costly ineffective attack and the cost-free no-attack control.
    """

    reward = float(raw_reward)
    cost = float(maximum_cost)
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError("maximum_cost must be finite and >= 0")
    if not math.isfinite(reward) or reward < -cost - 1e-12 or reward > 1.0 + 1e-12:
        raise ValueError("raw Eve reward is outside [-maximum_cost, 1]")
    return min(1.0, max(0.0, (reward + cost) / (1.0 + cost)))


def _empirical_game_diagnostics(
    *,
    learning_matrix: _EmpiricalUtilityMatrix,
    physics: HistogramPhysicsPool,
    calibrations: Sequence[RoutePhysicsCalibration],
    actions: Sequence[dict[str, Any]],
    action_costs: np.ndarray,
    alice_strategy: np.ndarray,
    eve_strategy: np.ndarray,
    reference_service_rate_hz: float,
    config: HistogramCoLearningConfig,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    adaptive = learning_matrix.game_diagnostics(
        action_costs=action_costs,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        behavior_joint_counts=learning_matrix.count,
        sampling_design="adaptive_joint_bandit_play",
        used_for_learning=True,
    )
    samples = int(config.posthoc_samples_per_cell)
    if samples <= 0:
        return adaptive
    if progress_callback is not None:
        progress_callback(
            "post-hoc stratified payoff audit: "
            f"{len(calibrations)} routes x {len(actions)} actions x "
            f"{samples} independent samples"
        )
    audit = _EmpiricalUtilityMatrix(len(calibrations), len(actions))
    step_index = 0
    for route_index, calibration in enumerate(calibrations):
        for action_index, action in enumerate(actions):
            for _ in range(samples):
                result = physics.step(
                    route_index,
                    action,
                    seed=config.seed + 900_000_000,
                    step_index=step_index,
                )
                step_index += 1
                utility = bell_qualified_rate_utility(
                    result=result,
                    calibration=calibration,
                    reference_service_rate_hz=reference_service_rate_hz,
                )
                audit.observe(route_index, action_index, utility.reward)
    diagnostics = audit.game_diagnostics(
        action_costs=action_costs,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        behavior_joint_counts=learning_matrix.count,
        sampling_design="stratified_independent_posthoc",
        used_for_learning=False,
    )
    diagnostics["posthoc_samples_per_cell"] = samples
    diagnostics["posthoc_total_samples"] = step_index
    diagnostics["adaptive_learning_sample_coverage"] = {
        "cell_counts": adaptive["cell_counts"],
        "all_cells_observed": adaptive["all_cells_observed"],
        "minimum_cell_samples": adaptive["minimum_cell_samples"],
        "maximum_cell_standard_error": adaptive[
            "maximum_cell_standard_error"
        ],
        "sampling_design": adaptive["sampling_design"],
        "note": (
            "retained for behavior-coverage audit only; post-hoc LP and "
            "exploitability use the independent stratified matrix"
        ),
    }
    return diagnostics


class _EmpiricalUtilityMatrix:
    def __init__(self, routes: int, actions: int) -> None:
        self.count = np.zeros((routes, actions), dtype=np.int64)
        self.total = np.zeros((routes, actions), dtype=float)
        self.total_squared = np.zeros((routes, actions), dtype=float)

    def observe(self, route: int, action: int, utility: float) -> None:
        value = float(utility)
        self.count[route, action] += 1
        self.total[route, action] += value
        self.total_squared[route, action] += value * value

    def game_diagnostics(
        self,
        *,
        action_costs: np.ndarray,
        alice_strategy: np.ndarray,
        eve_strategy: np.ndarray,
        behavior_joint_counts: np.ndarray,
        sampling_design: str,
        used_for_learning: bool,
    ) -> dict[str, Any]:
        observed = self.count > 0
        means = np.divide(
            self.total,
            self.count,
            out=np.full_like(self.total, np.nan),
            where=observed,
        )
        standard_errors = np.full_like(self.total, np.nan)
        enough = self.count > 1
        variance = np.zeros_like(self.total)
        variance[enough] = (
            self.total_squared[enough]
            - self.total[enough] ** 2 / self.count[enough]
        ) / (self.count[enough] - 1)
        standard_errors[enough] = np.sqrt(
            np.maximum(0.0, variance[enough]) / self.count[enough]
        )
        payload: dict[str, Any] = {
            "cell_counts": self.count.tolist(),
            "alice_public_utility_matrix": _json_matrix(means),
            "cell_standard_errors": _json_matrix(standard_errors),
            "all_cells_observed": bool(np.all(observed)),
            "minimum_cell_samples": int(self.count.min()),
            "maximum_cell_standard_error": (
                float(np.nanmax(standard_errors))
                if np.any(np.isfinite(standard_errors)) else None
            ),
            "sampling_design": sampling_design,
            "used_for_learning": bool(used_for_learning),
            "sampling_note": (
                "cell means are conditional sample means collected under the "
                "bandits' adaptive joint policy"
                if sampling_design == "adaptive_joint_bandit_play"
                else "every route/action cell receives the same number of "
                "independent post-learning histogram samples"
            ),
        }
        if not np.all(observed):
            payload["oracle"] = None
            return payload
        behavior_counts = np.asarray(behavior_joint_counts, dtype=np.int64)
        if behavior_counts.shape != means.shape or np.any(behavior_counts < 0):
            raise ValueError("behavior_joint_counts has the wrong shape or values")
        behavior_turns = int(np.sum(behavior_counts))
        if behavior_turns < 1:
            raise ValueError("behavior_joint_counts must contain at least one turn")
        behavior_frequency = behavior_counts / behavior_turns
        behavior_alice = np.sum(behavior_frequency, axis=1)
        behavior_eve = np.sum(behavior_frequency, axis=0)
        if not np.allclose(behavior_alice, alice_strategy, atol=1e-12):
            raise ValueError("Alice strategy disagrees with behavior joint counts")
        if not np.allclose(behavior_eve, eve_strategy, atol=1e-12):
            raise ValueError("Eve strategy disagrees with behavior joint counts")
        equivalent = means + action_costs[np.newaxis, :]
        value, oracle_alice, oracle_eve = solve_zero_sum_lp(equivalent)
        learned_alice_utility = float(alice_strategy @ means @ eve_strategy)
        learned_eve_utility = float(
            1.0
            - learned_alice_utility
            - eve_strategy @ action_costs
        )
        oracle_alice_utility = float(oracle_alice @ means @ oracle_eve)
        product_equivalent_utility = float(
            alice_strategy @ equivalent @ eve_strategy
        )
        realized_joint_equivalent_utility = float(
            np.sum(behavior_frequency * equivalent)
        )
        alice_best_fixed = float(np.max(equivalent @ eve_strategy))
        eve_best_fixed = float(np.min(alice_strategy @ equivalent))
        alice_external_regret = (
            alice_best_fixed - realized_joint_equivalent_utility
        )
        eve_external_regret = (
            realized_joint_equivalent_utility - eve_best_fixed
        )
        alice_best_response_gap = (
            alice_best_fixed - product_equivalent_utility
        )
        eve_best_response_gap = (
            product_equivalent_utility - eve_best_fixed
        )
        saddle_gap = alice_best_fixed - eve_best_fixed
        payload["zero_sum_equivalent_matrix"] = equivalent.tolist()
        payload["oracle"] = {
            "equivalence": (
                "Eve maximizes 1-A-c; the strategically equivalent zero-sum "
                "Alice matrix is A+c because c depends only on Eve's action"
            ),
            "equivalent_value": float(value),
            "alice_strategy": oracle_alice.tolist(),
            "eve_strategy": oracle_eve.tolist(),
            "alice_public_utility": oracle_alice_utility,
            "eve_public_utility": float(
                1.0
                - oracle_alice_utility
                - oracle_eve @ action_costs
            ),
        }
        payload["learned_profile"] = {
            "alice_public_utility": learned_alice_utility,
            "eve_public_utility": learned_eve_utility,
            "zero_sum_equivalent_exploitability": float(saddle_gap),
            "best_response_gap_decomposition": {
                "alice": float(alice_best_response_gap),
                "eve": float(eve_best_response_gap),
                "sum": float(alice_best_response_gap + eve_best_response_gap),
                "product_profile_equivalent_utility": (
                    product_equivalent_utility
                ),
            },
            "external_regret_decomposition": {
                "alice_per_turn": float(alice_external_regret),
                "eve_per_turn": float(eve_external_regret),
                "sum_per_turn": float(
                    alice_external_regret + eve_external_regret
                ),
                "audit_expected_realized_joint_equivalent_utility": (
                    realized_joint_equivalent_utility
                ),
                "behavior_turns": behavior_turns,
                "interpretation": (
                    "external regret against each player's best fixed action "
                    "for the realized opponent-action sequence, evaluated on "
                    "this empirical payoff matrix; individual regret may be "
                    "negative while their sum equals the saddle-point gap"
                ),
            },
            "decomposition_identity_error": float(
                abs(
                    saddle_gap
                    - alice_external_regret
                    - eve_external_regret
                )
            ),
        }
        return payload


def _curve_row(
    *,
    step_number: int,
    alice: PlayerLearner,
    eve: PlayerLearner,
    alice_mean: float,
    rate_mean: float,
    eve_public_mean: float,
    target_overlap_mean: float,
    outcome_counts: Counter[str],
    signal_counts: Counter[str],
    factual_alice_policy: np.ndarray,
    counterfactual_alice_policy: np.ndarray,
    corrected_reward_delta_mean: float,
) -> dict[str, Any]:
    classifiable = int(signal_counts["classifiable"])
    false_count = int(signal_counts["false"])
    alice_policy_statistics = policy_distribution_statistics(
        alice.policy(turn=step_number + 1)
    )
    eve_policy_statistics = policy_distribution_statistics(
        eve.policy(turn=step_number + 1)
    )
    return {
        "turn": step_number,
        "alice_utility_mean": alice_mean,
        "bell_qualified_rate_hz_mean": rate_mean,
        "eve_public_utility_mean": eve_public_mean,
        "target_overlap_rate_diagnostic": target_overlap_mean,
        "accepted_rate": outcome_counts["accepted"] / step_number,
        "alice_strategy": alice.empirical_strategy().tolist(),
        "eve_strategy": eve.empirical_strategy().tolist(),
        "alice_policy_entropy": alice_policy_statistics["entropy_nats"],
        "eve_policy_entropy": eve_policy_statistics["entropy_nats"],
        "alice_policy_minimum_action_probability": alice_policy_statistics[
            "minimum_action_probability"
        ],
        "eve_policy_minimum_action_probability": eve_policy_statistics[
            "minimum_action_probability"
        ],
        "alice_policy_uniform_coverage_mass": alice_policy_statistics[
            "uniform_coverage_mass"
        ],
        "eve_policy_uniform_coverage_mass": eve_policy_statistics[
            "uniform_coverage_mass"
        ],
        "alice_policy_nonmodal_probability_mass": alice_policy_statistics[
            "nonmodal_probability_mass"
        ],
        "eve_policy_nonmodal_probability_mass": eve_policy_statistics[
            "nonmodal_probability_mass"
        ],
        "alice_policy_effective_action_count": alice_policy_statistics[
            "effective_action_count"
        ],
        "eve_policy_effective_action_count": eve_policy_statistics[
            "effective_action_count"
        ],
        "alice_policy_effective_action_fraction": alice_policy_statistics[
            "effective_action_fraction"
        ],
        "eve_policy_effective_action_fraction": eve_policy_statistics[
            "effective_action_fraction"
        ],
        "alice_eta": _schedule_value(alice, "eta"),
        "alice_gamma": _schedule_value(alice, "gamma"),
        "eve_eta": _schedule_value(eve, "eta"),
        "eve_gamma": _schedule_value(eve, "gamma"),
        "alice_false_signal_rate_all_turns": false_count / step_number,
        "alice_false_signal_rate_classifiable": (
            false_count / classifiable if classifiable else None
        ),
        "alice_false_accept_rate_all_turns": (
            signal_counts["false_accept"] / step_number
        ),
        "alice_false_abort_rate_all_turns": (
            signal_counts["false_abort"] / step_number
        ),
        "alice_false_signal_corrected_reward_delta_mean": (
            corrected_reward_delta_mean
        ),
        "alice_false_signal_policy_total_variation": _total_variation(
            factual_alice_policy,
            counterfactual_alice_policy,
        ),
        "alice_counterfactual_corrected_policy": (
            counterfactual_alice_policy.tolist()
        ),
    }


def _schedule_value(
    learner: PlayerLearner,
    name: str,
) -> float | None:
    point = learner.last_schedule
    if point is None:
        return None
    value = getattr(point, name, None)
    return None if value is None else float(value)


def _online_learning_telemetry(
    *,
    alice: PlayerLearner,
    eve: PlayerLearner,
    joint_action_counts: np.ndarray,
    turns: int,
) -> dict[str, Any]:
    counts = np.asarray(joint_action_counts, dtype=np.int64)
    expected_shape = (len(alice.action_ids), len(eve.action_ids))
    if counts.shape != expected_shape or np.any(counts < 0):
        raise ValueError("joint action counts do not match the player action sets")
    if int(np.sum(counts)) != int(turns):
        raise ValueError("joint action counts do not sum to the run horizon")
    alice_counts = np.sum(counts, axis=1)
    eve_counts = np.sum(counts, axis=0)
    alice_strategy = alice.empirical_strategy()
    eve_strategy = eve.empirical_strategy()
    if not np.allclose(alice_counts / turns, alice_strategy, atol=1e-12):
        raise ValueError("Alice empirical strategy disagrees with joint counts")
    if not np.allclose(eve_counts / turns, eve_strategy, atol=1e-12):
        raise ValueError("Eve empirical strategy disagrees with joint counts")
    return {
        "schema": "histogram_two_player_learning_telemetry.v1",
        "feedback_protocol": "selected_action_scalar_reward_v1",
        "turns": int(turns),
        "alice": {
            **alice.runtime_telemetry(turn=turns + 1),
            "action_counts": alice_counts.tolist(),
        },
        "eve": {
            **eve.runtime_telemetry(turn=turns + 1),
            "action_counts": eve_counts.tolist(),
        },
        "joint_action_counts": counts.tolist(),
        "joint_action_frequency": (counts / turns).tolist(),
        "observed_joint_cell_count": int(np.count_nonzero(counts)),
        "declared_joint_cell_count": int(counts.size),
        "joint_cell_coverage_fraction": float(np.count_nonzero(counts) / counts.size),
        "semantics": (
            "joint action identities and target overlap remain offline audit "
            "telemetry; each online update received only its player's selected "
            "action scalar reward"
        ),
    }


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _json_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if math.isfinite(float(value)) else None for value in row]
        for row in values
    ]


def _false_signal_summary(
    *,
    signal_counts: Counter[str],
    turns: int,
    factual_policy: np.ndarray,
    counterfactual_policy: np.ndarray,
    log_weight_offset: np.ndarray,
    reward_delta_sum: float,
    reward_delta_abs_sum: float,
    route_ids: Sequence[str],
    learner_mode: str,
    counterfactual_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classifiable = int(signal_counts["classifiable"])
    false_count = int(signal_counts["false"])
    factual = np.asarray(factual_policy, dtype=float)
    counterfactual = np.asarray(counterfactual_policy, dtype=float)
    state_replay = learner_mode in {
        ALICE_LEARNER_EXP3_PLUS_PLUS,
        ALICE_LEARNER_MAITI_DAG_FTRL,
        ALICE_LEARNER_ITO_DAG_FTRL,
        ALICE_LEARNER_TSALLIS_INF,
    }
    attribution_methods = {
        ALICE_LEARNER_FLAT_EXP3: (
            "fixed_behavior_exp3_reward_correction_v1"
        ),
        ALICE_LEARNER_TSALLIS_INF: (
            "fixed_behavior_tsallis_inf_reward_replay_v1"
        ),
        ALICE_LEARNER_EXP3_PLUS_PLUS: (
            "fixed_behavior_exp3_plus_plus_reward_replay_v1"
        ),
        ALICE_LEARNER_DAG_PATH_COMBAND: (
            "fixed_behavior_dag_comband_reward_correction_v1"
        ),
        ALICE_LEARNER_DAG_PATH_EXP2: (
            "fixed_behavior_dag_path_exp2_reward_correction_v1"
        ),
        ALICE_LEARNER_LINEAR_ROUTE_EXP2: (
            "fixed_behavior_linear_exp2_reward_correction_v1"
        ),
        ALICE_LEARNER_MAITI_DAG_FTRL: (
            "fixed_behavior_maiti_ftrl_reward_correction_v1"
        ),
        ALICE_LEARNER_ITO_DAG_FTRL: (
            "fixed_behavior_ito_ftrl_reward_correction_v1"
        ),
    }
    return {
        "definition": (
            "A false public signal is a delivered step whose finite-transcript "
            "accept/abort decision disagrees with the histogram expected QBER "
            "and CHSH classification under the same declared thresholds."
        ),
        "delivery_failure_semantics": (
            "delivery failures are real service outcomes and are excluded from "
            "the latent-quality false-signal denominator"
        ),
        "counts": {
            key: int(value)
            for key, value in sorted(signal_counts.items())
        },
        "false_signal_rate_all_turns": false_count / turns,
        "false_signal_rate_classifiable_quality_signals": (
            false_count / classifiable if classifiable else None
        ),
        "false_accept_rate_all_turns": signal_counts["false_accept"] / turns,
        "false_abort_rate_all_turns": signal_counts["false_abort"] / turns,
        "corrected_minus_observed_reward_mean": reward_delta_sum / turns,
        "absolute_reward_correction_mean": reward_delta_abs_sum / turns,
        "policy_attribution": {
            "method": attribution_methods[learner_mode],
            "interpretation": (
                "The selected routes, their factual selection probabilities, "
                "and Eve trajectory are held fixed. Only false accept/abort "
                "rewards are replaced by latent-threshold rewards. Nonlinear "
                "learners replay their exact factual-probability estimator "
                "state; additive-score learners apply their exact score "
                "correction."
            ),
            "causal_boundary": (
                "This is local sensitivity along the realized behavior path, "
                "not a fully re-simulated joint Alice/Eve counterfactual."
            ),
            "factual_current_policy": factual.tolist(),
            "false_signal_corrected_current_policy": counterfactual.tolist(),
            "policy_total_variation": _total_variation(factual, counterfactual),
            "log_weight_correction": (
                None if state_replay else log_weight_offset.tolist()
            ),
            "ftrl_counterfactual_solver": counterfactual_state,
            "route_probability_shift": [
                {
                    "route_id": str(route_id),
                    "factual_probability": float(factual[index]),
                    "corrected_probability": float(counterfactual[index]),
                    "corrected_minus_factual": float(
                        counterfactual[index] - factual[index]
                    ),
                }
                for index, route_id in enumerate(route_ids)
            ],
        },
    }


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        0.5
        * np.abs(
            np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        ).sum()
    )


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _physics_run_diagnostics(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Report per-run cache/runtime counters when a physics pool is reused.

    Paper ensembles intentionally reuse an already compiled route pool across
    independent learner seeds.  The backend counters are cumulative, so
    copying the final diagnostics into every replicate would overstate later
    runs.  Static adherence/configuration fields remain untouched while the
    public ``stats`` fields become this run's deltas; cumulative values are
    retained explicitly for audit.
    """

    payload = dict(after)
    payload["stats_cumulative"] = dict(after.get("stats") or {})
    payload["stats"] = _numeric_delta(
        before.get("stats") or {},
        after.get("stats") or {},
    )
    before_routes = before.get("routes") or {}
    route_rows: dict[str, Any] = {}
    for route_id, row in (after.get("routes") or {}).items():
        route = dict(row)
        route["stats_cumulative"] = dict(row.get("stats") or {})
        route["stats"] = _numeric_delta(
            (before_routes.get(route_id) or {}).get("stats") or {},
            row.get("stats") or {},
        )
        route_rows[str(route_id)] = route
    payload["routes"] = route_rows
    payload["counter_semantics"] = (
        "stats are deltas for this co-learning run; stats_cumulative records "
        "the shared route-pool counters after the run"
    )
    return payload


def _numeric_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        str(key): float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in sorted(keys)
        if isinstance(before.get(key, 0.0), (int, float))
        and isinstance(after.get(key, 0.0), (int, float))
    }
