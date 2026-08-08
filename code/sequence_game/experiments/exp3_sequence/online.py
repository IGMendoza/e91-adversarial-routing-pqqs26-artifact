"""Online Exp3/oracle runs over SeQUeNCe-derived payoff matrices."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from sequence_game.game_theory import exploitability
from sequence_game.eve.feedback import (
    EVE_FEEDBACK_MODEL,
    eve_feedback_from_public_utility,
)

from .backend import ActionSpec, TurnResult, _action_hits_route
from .attack_cache import (
    attack_route_cache_key,
    cached_attack_sample_count,
    load_cached_attack_samples,
)
from .acceptance import alice_accepts_turn, alice_reward_for_turn
from .baseline_cache import (
    cached_baseline_sample_count,
    load_cached_baseline_results,
    route_physics_cache_key,
)
from .config import ConditionConfig, Exp3SequenceConfig
from .corpus import GraphCase
from .learners import make_policy
from .oracle import OracleSummary
from .payoff import PayoffEstimate


@dataclass(frozen=True)
class OnlineRunSummary:
    graph_id: str
    condition: str
    alice_mode: str
    eve_mode: str
    turns: int
    final_key_rate: float
    final_retention: float | None
    total_key_rate: float
    total_retention: float | None
    eve_feedback_mean: float
    hit_rate: float
    exploitability_vs_payoff: float
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray
    outcome_counts: dict[str, int]
    steps: tuple[dict[str, Any], ...]
    learning_curve: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "condition": self.condition,
            "alice_mode": self.alice_mode,
            "eve_mode": self.eve_mode,
            "turns": self.turns,
            "final_key_rate": self.final_key_rate,
            "final_retention": self.final_retention,
            "total_key_rate": self.total_key_rate,
            "total_retention": self.total_retention,
            "eve_feedback_mean": self.eve_feedback_mean,
            "hit_rate": self.hit_rate,
            "exploitability_vs_payoff": self.exploitability_vs_payoff,
            "alice_strategy": self.alice_strategy.tolist(),
            "eve_strategy": self.eve_strategy.tolist(),
            "outcome_counts": dict(self.outcome_counts),
            "steps": list(self.steps),
            "learning_curve": list(self.learning_curve),
            "metadata": dict(self.metadata),
        }


def run_online_condition(case: GraphCase, actions: list[ActionSpec],
                         payoff: PayoffEstimate, oracle: OracleSummary,
                         condition: ConditionConfig,
                         config: Exp3SequenceConfig, *, seed: int,
                         progress_callback: Any = None) -> OnlineRunSummary:
    rng_alice = np.random.default_rng([seed, 11])
    rng_eve = np.random.default_rng([seed, 23])
    alice_policy = make_policy(
        condition.alice,
        len(case.routes),
        oracle_strategy=oracle.alice_strategy,
        gamma=config.alice_exp3_gamma,
        rng=rng_alice,
        route_scores=_no_attack_route_scores(actions, payoff),
        avoidance_horizon=config.cautious_greedy_avoidance_horizon,
        schedule_mode=config.exp3_schedule_mode,
        eta_c=config.alice_exp3_eta_c,
        t0=config.alice_exp3_t0,
        gamma_max=config.alice_exp3_gamma_max,
    )
    eve_policy = make_policy(
        condition.eve,
        len(actions),
        oracle_strategy=oracle.eve_strategy,
        gamma=config.eve_exp3_gamma,
        rng=rng_eve,
        schedule_mode=config.exp3_schedule_mode,
        eta_c=config.eve_exp3_eta_c,
        t0=config.eve_exp3_t0,
        gamma_max=config.eve_exp3_gamma_max,
    )
    stored_steps: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    rewards: list[float] = []
    eve_feedbacks: list[float] = []
    hits: list[float] = []
    outcome_counts: Counter[str] = Counter()
    rng_outcome = np.random.default_rng([seed, 37])
    progress_every = max(1, min(250, config.online_turns // 10 or 1))
    final_window_rewards: deque[float] = deque(
        maxlen=max(1, config.online_final_window))
    for turn in range(config.online_turns):
        route_index = alice_policy.sample()
        action_index = eve_policy.sample()
        action = actions[action_index]
        result = payoff_turn_result(
            case,
            actions,
            payoff,
            config,
            route_index,
            action_index,
            seed=int(rng_outcome.integers(0, 2**63 - 1)),
        )
        alice_policy.update(result.alice_reward)
        _observe_alice_public_outcome(alice_policy, result.public_outcome)
        if result.eve_feedback_reward is None:
            raise RuntimeError("online turn did not provide Eve-visible feedback")
        eve_policy.update(result.eve_feedback_reward)
        rewards.append(float(result.alice_reward))
        eve_feedbacks.append(float(result.eve_feedback_reward))
        # Structural target overlap is retained as an offline diagnostic only.
        hits.append(float(result.eve_hit_reward))
        final_window_rewards.append(float(result.alice_reward))
        outcome_counts[str(result.public_outcome)] += 1
        alice_schedule = _last_policy_schedule(alice_policy)
        eve_schedule = _last_policy_schedule(eve_policy)
        row = {
            "turn": turn,
            "route_index": route_index,
            "route_id": case.routes[route_index]["route_id"],
            "action_index": action_index,
            "action_id": action.action_id,
            **result.to_dict(),
            "gamma_A_t": _schedule_value(alice_schedule, "gamma"),
            "gamma_E_t": _schedule_value(eve_schedule, "gamma"),
            "eta_A_t": _schedule_value(alice_schedule, "eta"),
            "eta_E_t": _schedule_value(eve_schedule, "eta"),
            "alice_strategy": alice_policy.empirical_strategy().tolist(),
            "eve_strategy": eve_policy.empirical_strategy().tolist(),
        }
        if condition.alice == "cautious_greedy":
            row["alice_cautious_fallback"] = bool(
                getattr(alice_policy, "last_fallback", False)
            )
        if _record_online_step(turn, config.online_turns,
                               config.online_step_record_stride):
            stored_steps.append(row)
        if (turn + 1) % max(1, min(25, config.online_turns or 1)) == 0 or turn + 1 == config.online_turns:
            x = alice_policy.empirical_strategy()
            y = eve_policy.empirical_strategy()
            key_rate_so_far = float(np.mean(rewards))
            eve_feedback_so_far = float(np.mean(eve_feedbacks))
            hit_rate_so_far = float(np.mean(hits))
            matrix_value = float(x @ payoff.payoff @ y)
            baseline = float(oracle.baseline_rate)
            gap = float(exploitability(payoff.payoff, x, y))
            curve.append({
                "turn": turn + 1,
                "t": turn + 1,
                "matrix_value": matrix_value,
                "matrix_retention": (
                    None if baseline <= 0 else matrix_value / baseline
                ),
                "exploitability": gap,
                "empirical_nash_gap": gap,
                "key_rate_so_far": key_rate_so_far,
                "retention_so_far": (
                    None if baseline <= 0 else key_rate_so_far / baseline
                ),
                "alice_sampled_win_rate": key_rate_so_far,
                "alice_sampled_retention_rate": (
                    None if baseline <= 0 else key_rate_so_far / baseline
                ),
                "eve_sampled_win_rate": eve_feedback_so_far,
                "eve_feedback_mean_so_far": eve_feedback_so_far,
                "hit_rate_so_far": hit_rate_so_far,
                "gamma_A_t": _schedule_value(alice_schedule, "gamma"),
                "gamma_E_t": _schedule_value(eve_schedule, "gamma"),
                "eta_A_t": _schedule_value(alice_schedule, "eta"),
                "eta_E_t": _schedule_value(eve_schedule, "eta"),
                "alice_policy_entropy": _policy_entropy(alice_policy),
                "eve_policy_entropy": _policy_entropy(eve_policy),
                "oracle_value": oracle.value,
                "oracle_retention": oracle.retention,
            })
        if progress_callback and (
            turn + 1 == 1
            or turn + 1 == config.online_turns
            or (turn + 1) % progress_every == 0
        ):
            progress_callback(
                f"{condition.key} turn {turn + 1}/{config.online_turns}; "
                f"key_rate={float(np.mean(rewards)):.3f} "
                f"eve_feedback={float(np.mean(eve_feedbacks)):.3f} "
                f"target_overlap={float(np.mean(hits)):.3f}"
            )
    return summarize_online(
        case=case,
        payoff=payoff,
        oracle=oracle,
        condition=condition,
        steps=stored_steps,
        curve=curve,
        rewards=rewards,
        eve_feedbacks=eve_feedbacks,
        hits=hits,
        outcome_counts=dict(outcome_counts),
        final_window_rewards=list(final_window_rewards),
        alice_strategy=alice_policy.empirical_strategy(),
        eve_strategy=eve_policy.empirical_strategy(),
        metadata=_online_run_metadata(
            config,
            alice_action_count=len(case.routes),
            eve_action_count=len(actions),
            turns=config.online_turns,
        ),
    )


def _no_attack_route_scores(
        actions: list[ActionSpec],
        payoff: PayoffEstimate,
) -> np.ndarray:
    try:
        no_attack_index = next(
            index for index, action in enumerate(actions)
            if action.attack_type == "no_attack"
        )
    except StopIteration:
        no_attack_index = 0
    return np.asarray(payoff.payoff[:, no_attack_index], dtype=float)


def _observe_alice_public_outcome(policy: Any, public_outcome: str) -> None:
    observer = getattr(policy, "observe_public_outcome", None)
    if observer is not None:
        observer(public_outcome)


def _record_online_step(turn: int, total_turns: int, stride: int) -> bool:
    return (
        turn == 0
        or turn + 1 == total_turns
        or (turn + 1) % stride == 0
    )


def payoff_turn_result(case: GraphCase, actions: list[ActionSpec],
                       payoff: PayoffEstimate,
                       config: Exp3SequenceConfig,
                       route_index: int,
                       action_index: int,
                       *,
                       seed: int) -> TurnResult:
    action = actions[action_index]
    route = case.routes[route_index]
    cell = _cell_for(payoff, route_index, action_index, len(actions))
    target_active = _action_hits_route(action, route)
    if target_active and action.attack_type != "no_attack":
        return _sample_attack_turn(
            route=route,
            action=action,
            cell=cell,
            config=config,
            seed=seed,
        )
    return _sample_clean_turn(
        route=route,
        action=action,
        cell=cell,
        config=config,
        seed=seed,
    )


def _sample_attack_turn(
        *,
        route: dict[str, Any],
        action: ActionSpec,
        cell: Any,
        config: Exp3SequenceConfig,
        seed: int,
) -> TurnResult:
    cache_key = attack_route_cache_key(route, action.attack_type)
    samples = load_cached_attack_samples(
        config.attack_cache_db_path,
        cache_key=cache_key,
        sample_count=1,
        seed=seed,
    )
    if samples is None:
        available = cached_attack_sample_count(
            config.attack_cache_db_path,
            cache_key=cache_key,
        )
        raise RuntimeError(
            "online attack turn cache missing or undersampled: "
            f"route_id={route.get('route_id')} action_id={action.action_id} "
            f"cache_key={cache_key!r} required=1 available={available} "
            f"db={config.attack_cache_db_path}"
        )
    sample = samples[0]
    return _copy_turn_result(
        sample.result,
        config=config,
        active_route_attacked=True,
        runtime_attack_applied=action.to_dict(),
        sequence_timing_extra={
            **cell.sequence_timing,
            "online_model": "cached_sequence_game_turn_sample",
            "online_sample_source": "attack_route_profile_cache",
            "online_cache_key": list(cache_key),
            "online_cache_seed": seed,
            "online_cache_sample_index": sample.sample_index,
            "online_cache_target_id": sample.target_id,
            "online_cache_target_kind": sample.target_kind,
            "requested_action_id": action.action_id,
            "pooled_attack_location_sampling": True,
        },
    )


def _sample_clean_turn(
        *,
        route: dict[str, Any],
        action: ActionSpec,
        cell: Any,
        config: Exp3SequenceConfig,
        seed: int,
) -> TurnResult:
    cache_key = route_physics_cache_key(route)
    results = load_cached_baseline_results(
        config.baseline_cache_db_path,
        cache_key=cache_key,
        sample_count=1,
        seed=seed,
    )
    if results is None:
        available = cached_baseline_sample_count(
            config.baseline_cache_db_path,
            cache_key=cache_key,
        )
        raise RuntimeError(
            "online clean turn cache missing or undersampled: "
            f"route_id={route.get('route_id')} action_id={action.action_id} "
            f"cache_key={cache_key!r} required=1 available={available} "
            f"db={config.baseline_cache_db_path}"
        )
    result = results[0]
    return _copy_turn_result(
        result,
        config=config,
        active_route_attacked=False,
        runtime_attack_applied=action.to_dict(),
        sequence_timing_extra={
            **cell.sequence_timing,
            "online_model": "cached_sequence_game_turn_sample",
            "online_sample_source": "clean_route_profile_cache",
            "online_cache_key": list(cache_key),
            "online_cache_seed": seed,
            "requested_action_id": action.action_id,
            "selected_action_missed_route": action.attack_type != "no_attack",
        },
    )


def _copy_turn_result(
    result: TurnResult,
    *,
    config: Exp3SequenceConfig,
    active_route_attacked: bool,
    runtime_attack_applied: dict[str, Any],
    sequence_timing_extra: dict[str, Any],
) -> TurnResult:
    accepted = alice_accepts_turn(result, config)
    alice_reward = alice_reward_for_turn(result, config)
    local_observation = (
        dict(result.eve_feedback_observation)
        if active_route_attacked
        else {}
    )
    feedback = eve_feedback_from_public_utility(
        public_outcome=result.public_outcome,
        alice_utility=alice_reward,
        local_observation_kind=str(
            local_observation.get("local_observation_kind", "none")
        ),
        local_observation_count=int(
            local_observation.get("local_observation_count", 0)
        ),
    )
    attack_type = str(runtime_attack_applied.get("attack_type", "no_attack"))
    target_overlap = active_route_attacked and attack_type != "no_attack"
    return TurnResult(
        public_outcome=result.public_outcome,
        alice_reward=float(alice_reward),
        eve_hit_reward=1.0 if target_overlap else 0.0,
        active_route_attacked=bool(active_route_attacked),
        accepted=bool(accepted),
        qber=result.qber,
        chsh_s=result.chsh_s,
        chsh_adequately_sampled=result.chsh_adequately_sampled,
        delivered_count=result.delivered_count,
        sifted_count=result.sifted_count,
        fidelity=result.fidelity,
        runtime_engine=result.runtime_engine,
        runtime_attack_applied=runtime_attack_applied,
        sequence_timing={
            **result.sequence_timing,
            **sequence_timing_extra,
            "cached_public_outcome": result.public_outcome,
            "cached_protocol_accepted": bool(result.accepted),
            "cached_protocol_alice_reward": float(result.alice_reward),
            "alice_acceptance_rule": config.alice_acceptance_rule,
            "alice_effective_accepted": bool(accepted),
            "alice_key_rate_shaping_weight": config.alice_key_rate_shaping_weight,
            "qber_is_diagnostic_not_hard_veto": (
                config.alice_acceptance_rule == "chsh_only"
            ),
        },
        eve_feedback_reward=feedback.reward,
        eve_feedback_model=EVE_FEEDBACK_MODEL,
        eve_feedback_observation=feedback.to_dict(),
    )


def _cell_for(payoff: PayoffEstimate, route_index: int, action_index: int,
              action_count: int):
    return payoff.cells[route_index * action_count + action_index]


def summarize_online(case: GraphCase, payoff: PayoffEstimate, oracle: OracleSummary,
                     condition: ConditionConfig, steps: list[dict[str, Any]],
                     curve: list[dict[str, Any]], rewards: list[float],
                     eve_feedbacks: list[float],
                     hits: list[float], outcome_counts: dict[str, int],
                     final_window_rewards: list[float],
                     alice_strategy: np.ndarray,
                     eve_strategy: np.ndarray,
                     metadata: dict[str, Any] | None = None) -> OnlineRunSummary:
    window = final_window_rewards
    final_key_rate = float(np.mean(window)) if window else 0.0
    total_key_rate = float(np.mean(rewards)) if rewards else 0.0
    baseline = oracle.baseline_rate
    final_retention = None if baseline <= 0 else final_key_rate / baseline
    total_retention = None if baseline <= 0 else total_key_rate / baseline
    return OnlineRunSummary(
        graph_id=case.graph_id,
        condition=condition.key,
        alice_mode=condition.alice,
        eve_mode=condition.eve,
        turns=len(rewards),
        final_key_rate=final_key_rate,
        final_retention=final_retention,
        total_key_rate=total_key_rate,
        total_retention=total_retention,
        eve_feedback_mean=(
            float(np.mean(eve_feedbacks)) if eve_feedbacks else 0.0
        ),
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        exploitability_vs_payoff=float(exploitability(payoff.payoff, alice_strategy, eve_strategy)),
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        outcome_counts=dict(outcome_counts),
        steps=tuple(steps),
        learning_curve=tuple(curve),
        metadata=dict(metadata or {}),
    )


def _last_policy_schedule(policy: Any) -> Any:
    return getattr(policy, "last_schedule", None)


def _schedule_value(schedule: Any, key: str) -> float | None:
    if schedule is None:
        return None
    return float(getattr(schedule, key))


def _policy_entropy(policy: Any) -> float:
    strategy = np.asarray(policy.current_strategy(), dtype=float)
    positive = strategy[strategy > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _online_run_metadata(
        config: Exp3SequenceConfig,
        *,
        alice_action_count: int,
        eve_action_count: int,
        turns: int,
) -> dict[str, Any]:
    initial_a = _schedule_point_from_config(
        config.exp3_schedule_mode,
        alice_action_count,
        gamma=config.alice_exp3_gamma,
        eta_c=config.alice_exp3_eta_c,
        t0=config.alice_exp3_t0,
        gamma_max=config.alice_exp3_gamma_max,
        turn=1,
    )
    initial_e = _schedule_point_from_config(
        config.exp3_schedule_mode,
        eve_action_count,
        gamma=config.eve_exp3_gamma,
        eta_c=config.eve_exp3_eta_c,
        t0=config.eve_exp3_t0,
        gamma_max=config.eve_exp3_gamma_max,
        turn=1,
    )
    final_turn = max(1, int(turns))
    final_a = _schedule_point_from_config(
        config.exp3_schedule_mode,
        alice_action_count,
        gamma=config.alice_exp3_gamma,
        eta_c=config.alice_exp3_eta_c,
        t0=config.alice_exp3_t0,
        gamma_max=config.alice_exp3_gamma_max,
        turn=final_turn,
    )
    final_e = _schedule_point_from_config(
        config.exp3_schedule_mode,
        eve_action_count,
        gamma=config.eve_exp3_gamma,
        eta_c=config.eve_exp3_eta_c,
        t0=config.eve_exp3_t0,
        gamma_max=config.eve_exp3_gamma_max,
        turn=final_turn,
    )
    return {
        "schedule_mode": config.exp3_schedule_mode,
        "K_A": int(alice_action_count),
        "K_E": int(eve_action_count),
        "c_eta_A": config.alice_exp3_eta_c,
        "c_eta_E": config.eve_exp3_eta_c,
        "t0_A": config.alice_exp3_t0,
        "t0_E": config.eve_exp3_t0,
        "gamma_max_A": config.alice_exp3_gamma_max,
        "gamma_max_E": config.eve_exp3_gamma_max,
        "initial_gamma_A": initial_a["gamma"],
        "initial_gamma_E": initial_e["gamma"],
        "final_gamma_A": final_a["gamma"],
        "final_gamma_E": final_e["gamma"],
        "initial_eta_A": initial_a["eta"],
        "initial_eta_E": initial_e["eta"],
        "final_eta_A": final_a["eta"],
        "final_eta_E": final_e["eta"],
        "alice_acceptance_rule": config.alice_acceptance_rule,
        "alice_key_rate_shaping_weight": config.alice_key_rate_shaping_weight,
        "eve_feedback": {
            "model": EVE_FEEDBACK_MODEL,
            "learner_scalar": "1 - Alice public per-turn utility",
            "public_evidence": "protocol outcome and configured public utility rule",
            "local_instrument_records": "reported separately; not added to denial reward",
            "target_overlap_disclosed_to_learner": False,
            "single_turn_causal_attribution": False,
        },
        "reward_convention": (
            "reward maximization; EXP3 update is "
            "log_weight[action] += eta_t * reward / p_t[action]"
        ),
    }


def _schedule_point_from_config(
        mode: str,
        action_count: int,
        *,
        gamma: float,
        eta_c: float,
        t0: float,
        gamma_max: float,
        turn: int,
) -> dict[str, float]:
    if action_count < 1:
        raise ValueError("action_count must be >= 1")
    if mode == "constant":
        return {"eta": float(gamma / action_count), "gamma": float(gamma)}
    if mode == "anytime":
        eta = float(eta_c) * math.sqrt(
            math.log(action_count)
            / (float(action_count) * (float(turn) + float(t0)))
        )
        return {"eta": float(eta), "gamma": float(min(gamma_max, action_count * eta))}
    raise ValueError(f"unsupported EXP3 schedule mode {mode!r}")
