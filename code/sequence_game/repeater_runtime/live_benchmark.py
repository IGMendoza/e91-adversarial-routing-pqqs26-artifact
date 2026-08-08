"""Feasibility benchmark for live selected-path repeater transitions.

This module measures direct SeQUeNCe selected-path calls for the two-route
runtime surface.  It does not train Q-learning and does not use cached
transition tables.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..claims import (
    BLOCKED,
    BLOCKED_PENDING_GRAPH_INTERFACE,
    IMPLEMENTED_AND_TESTABLE,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
)
from .two_route_control import (
    BASELINE_ACTION,
    EVE_ACTIONS,
    ROUTE_IDS,
    ROUTE_PATHS,
    _action_to_runtime_attack,
    _avoidance_summary,
    _blocked_claims,
    _numeric_summary,
    _select_alice_route,
)
from .two_route_reference import (
    REWARD_SCOPE,
    TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
    TwoRouteRuntimeCase,
    build_two_route_runtime_config,
    run_two_route_runtime_case,
)

LIVE_TRAINING_FEASIBLE_FOR_DEMO = "LIVE_TRAINING_FEASIBLE_FOR_DEMO"
LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN = "LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN"
LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE = "LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE"
LIVE_TRAINING_NOT_RECOMMENDED_FOR_LOOP = "LIVE_TRAINING_NOT_RECOMMENDED_FOR_LOOP"
ALLOWED_FEASIBILITY_LABELS = (
    LIVE_TRAINING_FEASIBLE_FOR_DEMO,
    LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN,
    LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE,
    LIVE_TRAINING_NOT_RECOMMENDED_FOR_LOOP,
)
PROJECTION_STEPS = (100, 500, 1000, 2000, 5000)


@dataclass(frozen=True)
class LiveTransitionBenchmarkConfig:
    measured_calls: int = 5
    warmup_calls: int = 1
    seed: int = 1201

    def __post_init__(self) -> None:
        if self.measured_calls < 1:
            raise ValueError("measured_calls must be >= 1")
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_calls": self.measured_calls,
            "warmup_calls": self.warmup_calls,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class LiveEpisodeBenchmarkConfig:
    seed: int = 1301
    avoidance_horizon: int = 1
    requested_step_counts: tuple[int, ...] = (5, 20, 50)
    policies: tuple[str, ...] = ("no_attack", "random_eve", "route_aware_oracle")
    max_20_step_seconds_for_50: float = 120.0

    def __post_init__(self) -> None:
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if not self.requested_step_counts:
            raise ValueError("requested_step_counts must be non-empty")
        if any(steps < 1 for steps in self.requested_step_counts):
            raise ValueError("requested_step_counts must be positive")
        allowed = {"no_attack", "random_eve", "route_aware_oracle"}
        bad = [policy for policy in self.policies if policy not in allowed]
        if bad:
            raise ValueError(f"unsupported live episode benchmark policies {bad}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "avoidance_horizon": self.avoidance_horizon,
            "requested_step_counts": list(self.requested_step_counts),
            "policies": list(self.policies),
            "max_20_step_seconds_for_50": self.max_20_step_seconds_for_50,
        }


@dataclass(frozen=True)
class LiveQLearningFeasibilityConfig:
    transition: LiveTransitionBenchmarkConfig = LiveTransitionBenchmarkConfig()
    episode: LiveEpisodeBenchmarkConfig = LiveEpisodeBenchmarkConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition": self.transition.to_dict(),
            "episode": self.episode.to_dict(),
        }


def default_transition_cases() -> list[dict[str, str]]:
    return [
        {"case_id": "route_r0_no_attack", "alice_route_id": "route_r0", "eve_action": BASELINE_ACTION},
        {"case_id": "route_r1_no_attack", "alice_route_id": "route_r1", "eve_action": BASELINE_ACTION},
        {"case_id": "route_r0_swap_denial_r0", "alice_route_id": "route_r0", "eve_action": "swap_denial_r0"},
        {"case_id": "route_r0_swap_denial_r1", "alice_route_id": "route_r0", "eve_action": "swap_denial_r1"},
        {"case_id": "route_r1_swap_denial_r0", "alice_route_id": "route_r1", "eve_action": "swap_denial_r0"},
        {"case_id": "route_r1_swap_denial_r1", "alice_route_id": "route_r1", "eve_action": "swap_denial_r1"},
        {"case_id": "route_r0_memory_degradation_r0", "alice_route_id": "route_r0", "eve_action": "memory_degradation_r0"},
        {"case_id": "route_r1_memory_degradation_r1", "alice_route_id": "route_r1", "eve_action": "memory_degradation_r1"},
        {"case_id": "route_r0_memory_measure_x_r0", "alice_route_id": "route_r0", "eve_action": "repeater_memory_measure_X_r0"},
        {"case_id": "route_r1_memory_measure_x_r1", "alice_route_id": "route_r1", "eve_action": "repeater_memory_measure_X_r1"},
        {"case_id": "route_r0_dephasing_probe_pi2_r0", "alice_route_id": "route_r0", "eve_action": "repeater_memory_dephasing_probe_pi2_r0"},
        {"case_id": "route_r1_dephasing_probe_pi2_r1", "alice_route_id": "route_r1", "eve_action": "repeater_memory_dephasing_probe_pi2_r1"},
    ]


def benchmark_live_transition_case(case: dict[str, str],
                                   *,
                                   config: LiveTransitionBenchmarkConfig | None = None
                                   ) -> dict[str, Any]:
    config = config or LiveTransitionBenchmarkConfig()
    route_id = case["alice_route_id"]
    action_id = case["eve_action"]
    if route_id not in ROUTE_IDS:
        raise ValueError(f"unsupported route {route_id!r}")
    if action_id != BASELINE_ACTION and action_id not in EVE_ACTIONS:
        raise ValueError(f"unsupported Eve action {action_id!r}")
    runtime_config = build_two_route_runtime_config()
    route_index = ROUTE_IDS.index(route_id)
    attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
    avoided_until: dict[str, int] = {}

    for warmup_idx in range(config.warmup_calls):
        run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"{case['case_id']}_warmup_{warmup_idx}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=config.seed + warmup_idx,
            ),
        )

    records: list[dict[str, Any]] = []
    for call_index in range(config.measured_calls):
        seed = config.seed + config.warmup_calls + call_index
        started = time.perf_counter()
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"{case['case_id']}_measured_{call_index}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=seed,
            ),
        )
        elapsed = time.perf_counter() - started
        public_outcome = runtime_record.get("public_outcome", "blocked")
        if public_outcome in {"delivery_failure", "qber_abort", "blocked"}:
            avoided_until[route_id] = call_index + 1
        records.append(_public_benchmark_step(
            runtime_record,
            step=call_index,
            route_id=route_id,
            action_id=action_id,
            policy="fixed_transition_benchmark",
            elapsed_seconds=elapsed,
            avoidance_summary=_avoidance_summary(
                list(ROUTE_IDS), avoided_until, call_index + 1),
            fallback_used=False,
            seed=seed,
        ))

    summary = _transition_case_summary(case, records, config)
    return {
        "case": dict(case),
        "summary": summary,
        "records": records,
    }


def run_live_transition_benchmark(
        config: LiveTransitionBenchmarkConfig | None = None,
        *,
        cases: Optional[list[dict[str, str]]] = None) -> dict[str, Any]:
    config = config or LiveTransitionBenchmarkConfig()
    results = [
        benchmark_live_transition_case(case, config=config)
        for case in (cases or default_transition_cases())
    ]
    measured_seconds = [
        record["wall_clock_seconds"]
        for result in results
        for record in result["records"]
        if record["scope_label"] == REPEATER_RUNTIME
    ]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "benchmark_kind": "live_transition",
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "config": config.to_dict(),
        "case_count": len(results),
        "measured_transition_count": len(measured_seconds),
        "timing_seconds": _timing_summary(measured_seconds),
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(summary)
    return {
        "summary": summary,
        "cases": results,
    }


def run_short_live_episode_benchmark(
        config: LiveEpisodeBenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or LiveEpisodeBenchmarkConfig()
    results: list[dict[str, Any]] = []
    initial_counts = tuple(
        step_count for step_count in config.requested_step_counts
        if step_count != 50
    )
    for step_count in initial_counts:
        for policy in config.policies:
            results.append(_run_timed_episode(policy, step_count, config=config))

    if 50 in config.requested_step_counts:
        twenty_second_runs = [
            result["summary"]["total_wall_clock_seconds"]
            for result in results
            if (
                result["scope_label"] == REPEATER_RUNTIME
                and result["summary"]["step_count"] == 20
            )
        ]
        allow_50 = (
            not twenty_second_runs
            or max(twenty_second_runs) <= config.max_20_step_seconds_for_50
        )
        for policy in config.policies:
            if not allow_50:
                results.append(_skipped_episode_result(policy, 50))
                continue
            results.append(_run_timed_episode(policy, 50, config=config))
    measured = [
        seconds
        for result in results
        if result["scope_label"] == REPEATER_RUNTIME
        for seconds in result["summary"]["step_wall_clock_seconds"]
    ]
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "benchmark_kind": "short_live_episode",
        "config": config.to_dict(),
        "timing_seconds": _timing_summary(measured),
        "episodes": results,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def run_live_q_learning_feasibility_benchmark(
        config: LiveQLearningFeasibilityConfig | None = None) -> dict[str, Any]:
    config = config or LiveQLearningFeasibilityConfig()
    transition = run_live_transition_benchmark(config.transition)
    episodes = run_short_live_episode_benchmark(config.episode)
    transition_seconds = [
        record["wall_clock_seconds"]
        for case in transition["cases"]
        for record in case["records"]
        if record["scope_label"] == REPEATER_RUNTIME
    ]
    projections = project_live_training_costs(transition_seconds)
    recommendation = _recommendation(projections["feasibility_classification"])
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "benchmark_kind": "live_q_learning_feasibility",
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "config": config.to_dict(),
        "transition_benchmark": transition,
        "episode_benchmark": episodes,
        "projection": projections,
        "recommendation": recommendation,
        "cached_training_results_emitted": False,
        "information_gain_reward_enabled": False,
        "arbitrary_graph_runtime": {
            "scope_label": BLOCKED,
            "claim_status": BLOCKED_PENDING_GRAPH_INTERFACE,
            "blocked": True,
            "blocked_reason": "arbitrary graph-to-SeQUeNCe runtime remains blocked",
        },
        "blocked_claims": _blocked_claims(),
    }
    assert_no_private_public_fields(payload)
    return payload


def project_live_training_costs(seconds: list[float]) -> dict[str, Any]:
    timing = _timing_summary(seconds)
    median = float(timing["p50_seconds"] or 0.0)
    p90 = float(timing["p90_seconds"] or median)
    rows = []
    for steps in PROJECTION_STEPS:
        rows.append({
            "scope_label": REPEATER_RUNTIME,
            "updates": steps,
            "median_estimate_seconds": median * steps,
            "median_estimate_minutes": median * steps / 60.0,
            "p90_estimate_seconds": p90 * steps,
            "p90_estimate_minutes": p90 * steps / 60.0,
        })
    classification = classify_live_training_feasibility(
        estimate_500_seconds=p90 * 500,
        estimate_2000_seconds=p90 * 2000,
    )
    return {
        "scope_label": REPEATER_RUNTIME,
        "basis": "measured live selected-path transition p50 and p90",
        "timing_seconds": timing,
        "rows": rows,
        "feasibility_classification": classification,
        "threshold_note": "engineering guidance only; not a scientific claim",
    }


def classify_live_training_feasibility(*,
                                       estimate_500_seconds: float,
                                       estimate_2000_seconds: float) -> str:
    if estimate_500_seconds < 10 * 60:
        return LIVE_TRAINING_FEASIBLE_FOR_DEMO
    if estimate_2000_seconds < 60 * 60:
        return LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN
    if estimate_2000_seconds < 3 * 60 * 60:
        return LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE
    return LIVE_TRAINING_NOT_RECOMMENDED_FOR_LOOP


def write_live_q_learning_feasibility_artifacts(json_path: Path,
                                                csv_path: Path,
                                                *,
                                                config: LiveQLearningFeasibilityConfig | None = None
                                                ) -> dict[str, Any]:
    payload = run_live_q_learning_feasibility_benchmark(config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    _write_benchmark_csv(payload, csv_path)
    return payload


def _run_timed_episode(policy: str,
                       step_count: int,
                       *,
                       config: LiveEpisodeBenchmarkConfig) -> dict[str, Any]:
    runtime_config = build_two_route_runtime_config()
    ranking = list(ROUTE_IDS)
    rng = np.random.default_rng(config.seed + step_count + len(policy))
    avoided_until: dict[str, int] = {}
    steps: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    total_started = time.perf_counter()
    for step_idx in range(step_count):
        route_id, route_index, fallback_used = _select_alice_route(
            ranking, avoided_until, step_idx)
        action_id = _episode_policy_action(policy, route_id, rng)
        attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
        seed = config.seed + step_count * 10_000 + step_idx
        started = time.perf_counter()
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"live_episode_{policy}_{step_count}_{step_idx}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=seed,
            ),
        )
        elapsed = time.perf_counter() - started
        step_seconds.append(elapsed)
        public_outcome = runtime_record.get("public_outcome", "blocked")
        if public_outcome in {"delivery_failure", "qber_abort", "blocked"}:
            avoided_until[route_id] = step_idx + config.avoidance_horizon
        steps.append(_public_benchmark_step(
            runtime_record,
            step=step_idx,
            route_id=route_id,
            action_id=action_id,
            policy=policy,
            elapsed_seconds=elapsed,
            avoidance_summary=_avoidance_summary(
                ranking, avoided_until, step_idx + 1),
            fallback_used=fallback_used,
            seed=seed,
        ))
    total_elapsed = time.perf_counter() - total_started
    summary = _episode_summary(
        policy, step_count, steps, step_seconds, total_elapsed)
    return {
        "scope_label": REPEATER_RUNTIME,
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
    }


def _episode_policy_action(policy: str,
                           route_id: str,
                           rng: np.random.Generator) -> str:
    if policy == "no_attack":
        return BASELINE_ACTION
    if policy == "random_eve":
        return str(EVE_ACTIONS[int(rng.integers(len(EVE_ACTIONS)))])
    if policy == "route_aware_oracle":
        return f"swap_denial_{route_id[-2:]}"
    raise ValueError(f"unsupported episode benchmark policy {policy!r}")


def _public_benchmark_step(runtime_record: dict[str, Any],
                           *,
                           step: int,
                           route_id: str,
                           action_id: str,
                           policy: str,
                           elapsed_seconds: float,
                           avoidance_summary: dict[str, Any],
                           fallback_used: bool,
                           seed: int) -> dict[str, Any]:
    record = {
        "step": step,
        "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
        "claim_status": runtime_record.get("claim_status", IMPLEMENTED_AND_TESTABLE),
        "benchmark_kind": "live_selected_path_transition",
        "seed": seed,
        "alice_policy": "deterministic_cautious_greedy",
        "alice_route_id": route_id,
        "alice_path": ROUTE_PATHS[route_id],
        "eve_policy": policy,
        "eve_action": action_id,
        "public_outcome": runtime_record.get("public_outcome", "blocked"),
        "delivery_success": bool(runtime_record.get("delivery_success", False)),
        "delivered_count": runtime_record.get("delivered_count"),
        "quality_metric": runtime_record.get("fidelity"),
        "fidelity": runtime_record.get("fidelity"),
        "qber": runtime_record.get("qber"),
        "active_route_attacked": bool(runtime_record.get("active_route_attacked", False)),
        "alice_avoidance": avoidance_summary,
        "fallback_used": bool(fallback_used),
        "wall_clock_seconds": float(elapsed_seconds),
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
    }
    if runtime_record.get("scope_label") == BLOCKED:
        record["blocked"] = True
        record["blocked_reason"] = runtime_record.get("blocked_reason")
    assert_no_private_public_fields(record)
    return record


def _transition_case_summary(case: dict[str, str],
                             records: list[dict[str, Any]],
                             config: LiveTransitionBenchmarkConfig) -> dict[str, Any]:
    seconds = [record["wall_clock_seconds"] for record in records]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "case_id": case["case_id"],
        "alice_route_id": case["alice_route_id"],
        "eve_action": case["eve_action"],
        "warmup_calls": config.warmup_calls,
        "measured_calls": len(records),
        "timing_seconds": _timing_summary(seconds),
        "public_outcome_counts": dict(sorted(
            Counter(record["public_outcome"] for record in records).items())),
        "delivery_failure_rate": sum(
            1 for record in records if not record.get("delivery_success", False)
        ) / len(records),
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(summary)
    return summary


def _episode_summary(policy: str,
                     step_count: int,
                     steps: list[dict[str, Any]],
                     step_seconds: list[float],
                     total_elapsed: float) -> dict[str, Any]:
    outcomes = Counter(step["public_outcome"] for step in steps)
    fidelities = [
        float(step["fidelity"]) for step in steps if step.get("fidelity") is not None
    ]
    qbers = [
        float(step["qber"]) for step in steps if step.get("qber") is not None
    ]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "policy": policy,
        "step_count": step_count,
        "total_wall_clock_seconds": float(total_elapsed),
        "mean_seconds_per_step": float(total_elapsed / len(steps)),
        "step_wall_clock_seconds": [float(v) for v in step_seconds],
        "timing_seconds": _timing_summary(step_seconds),
        "delivery_failure_rate": sum(
            1 for step in steps if not step.get("delivery_success", False)
        ) / len(steps),
        "public_outcome_counts": dict(sorted(outcomes.items())),
        "quality_fidelity_summary": _numeric_summary(fidelities),
        "qber_summary": _numeric_summary(qbers),
        "final_20_step_replay_available": bool(steps),
        "final_20_step_replay_window": steps[-min(20, len(steps)):],
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(summary)
    return summary


def _skipped_episode_result(policy: str, step_count: int) -> dict[str, Any]:
    return {
        "scope_label": BLOCKED,
        "blocked": True,
        "blocked_reason": (
            "50-step benchmark skipped because 20-step benchmark exceeded "
            "configured local runtime threshold"
        ),
        "policy": policy,
        "step_count": step_count,
        "information_gain_reward_enabled": False,
    }


def _timing_summary(seconds: list[float]) -> dict[str, Optional[float]]:
    if not seconds:
        return {
            "count": 0,
            "min_seconds": None,
            "mean_seconds": None,
            "p50_seconds": None,
            "p90_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
        }
    sorted_values = sorted(float(value) for value in seconds)
    return {
        "count": len(sorted_values),
        "min_seconds": min(sorted_values),
        "mean_seconds": statistics.fmean(sorted_values),
        "p50_seconds": _quantile(sorted_values, 0.50),
        "p90_seconds": _quantile(sorted_values, 0.90),
        "p95_seconds": _quantile(sorted_values, 0.95),
        "max_seconds": max(sorted_values),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _recommendation(feasibility: str) -> dict[str, Any]:
    if feasibility in {
        LIVE_TRAINING_FEASIBLE_FOR_DEMO,
        LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN,
    }:
        action = "implement live Q-tabular Eve next"
    elif feasibility == LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE:
        action = "benchmark more before live Q-tabular Eve training"
    else:
        action = "keep cached tables as debugging/control fallback only"
    return {
        "scope_label": REPEATER_RUNTIME,
        "recommended_next_step": action,
        "cached_tables_role": "debugging/control fallback only",
        "classification": feasibility,
    }


def _write_benchmark_csv(payload: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in payload["transition_benchmark"]["cases"]:
        for record in case["records"]:
            rows.append({
                "record_type": "transition",
                "case_id": case["case"]["case_id"],
                "policy": record["eve_policy"],
                "step_count": "",
                "step": record["step"],
                "alice_route_id": record["alice_route_id"],
                "eve_action": record["eve_action"],
                "public_outcome": record["public_outcome"],
                "delivery_success": record["delivery_success"],
                "wall_clock_seconds": record["wall_clock_seconds"],
                "scope_label": record["scope_label"],
            })
    for episode in payload["episode_benchmark"]["episodes"]:
        if episode["scope_label"] != REPEATER_RUNTIME:
            rows.append({
                "record_type": "episode_skipped",
                "case_id": "",
                "policy": episode["policy"],
                "step_count": episode["step_count"],
                "step": "",
                "alice_route_id": "",
                "eve_action": "",
                "public_outcome": episode["blocked_reason"],
                "delivery_success": "",
                "wall_clock_seconds": "",
                "scope_label": episode["scope_label"],
            })
            continue
        summary = episode["summary"]
        rows.append({
            "record_type": "episode_summary",
            "case_id": "",
            "policy": summary["policy"],
            "step_count": summary["step_count"],
            "step": "",
            "alice_route_id": "",
            "eve_action": "",
            "public_outcome": json.dumps(summary["public_outcome_counts"], sort_keys=True),
            "delivery_success": "",
            "wall_clock_seconds": summary["total_wall_clock_seconds"],
            "scope_label": episode["scope_label"],
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_type",
                "case_id",
                "policy",
                "step_count",
                "step",
                "alice_route_id",
                "eve_action",
                "public_outcome",
                "delivery_success",
                "wall_clock_seconds",
                "scope_label",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "ALLOWED_FEASIBILITY_LABELS",
    "LIVE_TRAINING_EXPENSIVE_BUT_POSSIBLE",
    "LIVE_TRAINING_FEASIBLE_FOR_DEMO",
    "LIVE_TRAINING_FEASIBLE_FOR_SMALL_RUN",
    "LIVE_TRAINING_NOT_RECOMMENDED_FOR_LOOP",
    "LiveEpisodeBenchmarkConfig",
    "LiveQLearningFeasibilityConfig",
    "LiveTransitionBenchmarkConfig",
    "benchmark_live_transition_case",
    "classify_live_training_feasibility",
    "default_transition_cases",
    "project_live_training_costs",
    "run_live_q_learning_feasibility_benchmark",
    "run_live_transition_benchmark",
    "run_short_live_episode_benchmark",
    "write_live_q_learning_feasibility_artifacts",
]
