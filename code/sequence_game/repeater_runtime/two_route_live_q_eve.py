"""Live Q-tabular Eve training over selected-path repeater runtime.

Every training update in this module calls the actual selected-path SeQUeNCe
runtime wrapper. Cached transition tables are not used here.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..analysis.periods import detect_route_period
from ..claims import (
    BLOCKED,
    CONTROL_GAME,
    IMPLEMENTED_AND_TESTABLE,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
    assert_public_eve_observation,
)
from ..rl.q_learning import (
    QLearningConfig,
    QTable,
    epsilon_greedy,
    linear_epsilon_schedule,
)
from .live_benchmark import _timing_summary
from .two_route_control import (
    BASELINE_ACTION,
    DEFERRED_RUNTIME_CLAIMS,
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

LIVE_BACKEND = "live_selected_path_sequence_runtime"
LIVE_Q_EVE_POLICY = "live_q_tabular_eve"
DEFAULT_LIVE_TRAINING_PREFIX = "results/repeater_runtime/q_eve_two_route_live_training"
DEFAULT_LIVE_Q_TABLE_PATH = "results/repeater_runtime/q_eve_two_route_live_q_table.json"


@dataclass(frozen=True)
class LiveQEveTrainingConfig:
    target_steps: int = 500
    max_wall_clock_seconds: float = 1800.0
    checkpoint_every: int = 25
    output_prefix: Optional[str] = None
    resume_from: Optional[str] = None
    seed: int = 1701
    attack_cost: float = 0.05
    alpha: float = 0.45
    gamma: float = 0.85
    epsilon_start: float = 0.6
    epsilon_end: float = 0.02
    avoidance_horizon: int = 1
    quality_reward_enabled: bool = False
    quality_reward_weight: float = 0.0
    final_window_size: int = 20

    def __post_init__(self) -> None:
        if self.target_steps < 1:
            raise ValueError("target_steps must be >= 1")
        if self.max_wall_clock_seconds < 0:
            raise ValueError("max_wall_clock_seconds must be >= 0")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if self.attack_cost < 0:
            raise ValueError("attack_cost must be >= 0")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be in [0, 1]")
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if self.quality_reward_weight < 0:
            raise ValueError("quality_reward_weight must be >= 0")
        if self.final_window_size < 1:
            raise ValueError("final_window_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_steps": self.target_steps,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "checkpoint_every": self.checkpoint_every,
            "output_prefix": self.output_prefix,
            "resume_from": self.resume_from,
            "seed": self.seed,
            "attack_cost": self.attack_cost,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "avoidance_horizon": self.avoidance_horizon,
            "quality_reward_enabled": self.quality_reward_enabled,
            "quality_reward_weight": self.quality_reward_weight,
            "final_window_size": self.final_window_size,
        }


@dataclass(frozen=True)
class LiveQEveControlsConfig:
    seed: int = 2701
    random_steps: int = 50
    fixed_steps: int = 20
    oracle_steps: int = 20
    attack_cost: float = 0.05
    avoidance_horizon: int = 1
    final_window_size: int = 20

    def __post_init__(self) -> None:
        if min(self.random_steps, self.fixed_steps, self.oracle_steps) < 1:
            raise ValueError("control step counts must be >= 1")
        if self.attack_cost < 0:
            raise ValueError("attack_cost must be >= 0")
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if self.final_window_size < 1:
            raise ValueError("final_window_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "random_steps": self.random_steps,
            "fixed_steps": self.fixed_steps,
            "oracle_steps": self.oracle_steps,
            "attack_cost": self.attack_cost,
            "avoidance_horizon": self.avoidance_horizon,
            "final_window_size": self.final_window_size,
        }


def train_live_q_eve_two_route(
        config: LiveQEveTrainingConfig | None = None,
        *,
        summary_path: Optional[Path] = None,
        steps_path: Optional[Path] = None,
        q_table_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None) -> dict[str, Any]:
    """Train Q Eve with one live selected-path runtime call per Q update."""

    config = config or LiveQEveTrainingConfig()
    paths = _training_paths(config, summary_path, steps_path, q_table_path, checkpoint_path)
    runtime_config = build_two_route_runtime_config()
    calibration = calibrate_live_q_eve_routes(seed=config.seed + 10_000)
    ranking = list(calibration["ranked_route_ids"])
    q_table = QTable(num_actions=len(EVE_ACTIONS))
    q_config = QLearningConfig(alpha=config.alpha, gamma=config.gamma)
    schedule = linear_epsilon_schedule(
        config.epsilon_start, config.epsilon_end, config.target_steps)
    resume_state = _load_checkpoint(Path(config.resume_from)) if config.resume_from else {}
    if resume_state:
        q_table = QTable.from_dict(resume_state["q_table"])
    previous_action = str(resume_state.get("previous_action", "start"))
    previous_public_outcome = str(resume_state.get("previous_public_outcome", "start"))
    avoided_until = {
        str(route_id): int(until)
        for route_id, until in dict(resume_state.get("avoided_until", {})).items()
    }
    start_step = int(resume_state.get("completed_steps", 0))
    steps: list[dict[str, Any]] = []
    action_histogram = {action: 0 for action in EVE_ACTIONS}
    reward_trace: list[float] = []
    step_seconds: list[float] = []
    started = time.perf_counter()
    stopped_reason = "target_steps_completed"

    steps_file = None
    try:
        if paths["steps"] is not None:
            paths["steps"].parent.mkdir(parents=True, exist_ok=True)
            steps_file = paths["steps"].open("a" if resume_state else "w", encoding="utf-8")

        for step_idx in range(start_step, config.target_steps):
            elapsed_total = time.perf_counter() - started
            if elapsed_total >= config.max_wall_clock_seconds:
                stopped_reason = "time_budget_reached"
                break
            route_id, route_index, fallback_used = _select_alice_route(
                ranking, avoided_until, step_idx)
            state = (previous_action, previous_public_outcome)
            observation = {
                "scope_label": REPEATER_RUNTIME,
                "previous_action": previous_action,
                "previous_public_outcome": previous_public_outcome,
            }
            assert_public_eve_observation(observation)
            epsilon = schedule(step_idx)
            action_index = epsilon_greedy(
                q_table, state, epsilon, np.random.default_rng(config.seed + step_idx))
            action_id = EVE_ACTIONS[action_index]
            action_histogram[action_id] += 1
            attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
            step_started = time.perf_counter()
            runtime_record = run_two_route_runtime_case(
                runtime_config,
                TwoRouteRuntimeCase(
                    case_id=f"live_q_eve_train_step_{step_idx}_{route_id}_{action_id}",
                    selected_route_index=route_index,
                    configured_attack_route_index=attack_route_index,
                    attack_kind=attack_kind,
                    seed=config.seed + 100_000 + step_idx,
                ),
            )
            step_elapsed = time.perf_counter() - step_started
            public_outcome = runtime_record.get("public_outcome", "blocked")
            reward, reward_components = compute_live_eve_reward(
                public_outcome=public_outcome,
                action_id=action_id,
                fidelity=runtime_record.get("fidelity"),
                attack_cost=config.attack_cost,
                quality_reward_enabled=config.quality_reward_enabled,
                quality_reward_weight=config.quality_reward_weight,
            )
            if public_outcome in {"delivery_failure", "qber_abort", "blocked"}:
                avoided_until[route_id] = step_idx + config.avoidance_horizon
            next_state = (action_id, str(public_outcome))
            new_q = q_table.update(state, action_index, reward, next_state, q_config)
            record = _live_training_step_record(
                runtime_record,
                step=step_idx,
                route_id=route_id,
                action_id=action_id,
                observation=observation,
                epsilon=epsilon,
                reward=reward,
                reward_components=reward_components,
                q_update={
                    "scope_label": REPEATER_RUNTIME,
                    "state": list(state),
                    "action": action_id,
                    "new_q": round(float(new_q), 6),
                },
                avoidance_summary=_avoidance_summary(ranking, avoided_until, step_idx + 1),
                fallback_used=fallback_used,
                wall_clock_seconds=step_elapsed,
            )
            steps.append(record)
            reward_trace.append(reward)
            step_seconds.append(step_elapsed)
            previous_action = action_id
            previous_public_outcome = str(public_outcome)
            if steps_file is not None:
                steps_file.write(json.dumps(record, sort_keys=True) + "\n")
                if (len(steps) % config.checkpoint_every) == 0:
                    steps_file.flush()
            if ((step_idx + 1) % config.checkpoint_every) == 0 and paths["checkpoint"] is not None:
                _write_checkpoint(
                    paths["checkpoint"],
                    config=config,
                    q_table=q_table,
                    completed_steps=step_idx + 1,
                    previous_action=previous_action,
                    previous_public_outcome=previous_public_outcome,
                    avoided_until=avoided_until,
                )
        else:
            stopped_reason = "target_steps_completed"
    finally:
        if steps_file is not None:
            steps_file.close()

    wall_clock_total = time.perf_counter() - started
    summary = _live_training_summary(
        config=config,
        calibration=calibration,
        q_table=q_table,
        steps=steps,
        reward_trace=reward_trace,
        step_seconds=step_seconds,
        action_histogram=action_histogram,
        completed_steps=start_step + len(steps),
        wall_clock_total=wall_clock_total,
        stopped_reason=stopped_reason,
    )
    _write_live_training_outputs(paths, summary, q_table, config,
                                 previous_action, previous_public_outcome,
                                 avoided_until)
    return {
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
        "q_table": q_table,
    }


def calibrate_live_q_eve_routes(seed: int = 1) -> dict[str, Any]:
    runtime_config = build_two_route_runtime_config()
    rows = []
    for index, route_id in enumerate(ROUTE_IDS):
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"live_q_eve_calibration_{route_id}",
                selected_route_index=index,
                configured_attack_route_index=None,
                attack_kind=BASELINE_ACTION,
                seed=seed + index,
            ),
        )
        rows.append({
            "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
            "claim_status": runtime_record.get("claim_status", IMPLEMENTED_AND_TESTABLE),
            "selected_route": route_id,
            "selected_path": ROUTE_PATHS[route_id],
            "public_outcome": runtime_record.get("public_outcome", "blocked"),
            "delivery_success": bool(runtime_record.get("delivery_success", False)),
            "delivered_count": runtime_record.get("delivered_count"),
            "qber": runtime_record.get("qber"),
            "fidelity": runtime_record.get("fidelity"),
            "information_gain_reward_enabled": False,
        })

    def key(row: dict[str, Any]) -> tuple[int, int, float, str]:
        accepted = 1 if row["public_outcome"] == "accepted" else 0
        delivered = 1 if row["delivery_success"] else 0
        fidelity = row.get("fidelity")
        quality = float(fidelity) if fidelity is not None else -1.0
        return (-accepted, -delivered, -quality, row["selected_route"])

    ranked = sorted(rows, key=key)
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "backend": LIVE_BACKEND,
        "ranking_basis": (
            "honest no-attack live runtime accepted outcome, delivery success, "
            "fidelity diagnostic, deterministic route-id tie-break"
        ),
        "routes": rows,
        "ranked_route_ids": [row["selected_route"] for row in ranked],
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def compute_live_eve_reward(*,
                            public_outcome: str,
                            action_id: str,
                            fidelity: Optional[float],
                            attack_cost: float,
                            quality_reward_enabled: bool = False,
                            quality_reward_weight: float = 0.0) -> tuple[float, dict[str, Any]]:
    public_reward = 1.0 if public_outcome == "delivery_failure" else 0.0
    cost = 0.0 if action_id == BASELINE_ACTION else float(attack_cost)
    quality_reward = 0.0
    if quality_reward_enabled and fidelity is not None:
        quality_reward = float(quality_reward_weight) * max(0.0, 1.0 - float(fidelity))
    reward = public_reward + quality_reward - cost
    components = {
        "scope_label": REPEATER_RUNTIME,
        "public_outcome_reward": public_reward,
        "attack_cost": cost,
        "quality_reward_enabled": bool(quality_reward_enabled),
        "quality_reward": quality_reward,
        "information_gain_reward": 0.0,
    }
    return float(reward), components


def run_live_q_eve_controls(config: LiveQEveControlsConfig | None = None) -> dict[str, Any]:
    config = config or LiveQEveControlsConfig()
    calibration = calibrate_live_q_eve_routes(seed=config.seed + 10_000)
    ranking = list(calibration["ranked_route_ids"])
    policies = [
        ("random_eve", config.random_steps),
        ("always_attack_r0", config.fixed_steps),
        ("always_attack_r1", config.fixed_steps),
        ("route_aware_oracle", config.oracle_steps),
    ]
    results = [
        _run_live_control_policy(policy, steps, ranking=ranking, config=config)
        for policy, steps in policies
    ]
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "backend": LIVE_BACKEND,
        "config": config.to_dict(),
        "controls": results,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def write_live_q_eve_controls(path: Path,
                              config: LiveQEveControlsConfig | None = None
                              ) -> dict[str, Any]:
    payload = run_live_q_eve_controls(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return payload


def write_live_q_eve_training_artifacts(summary_path: Path,
                                        steps_path: Path,
                                        q_table_path: Path,
                                        *,
                                        config: LiveQEveTrainingConfig | None = None,
                                        checkpoint_path: Optional[Path] = None
                                        ) -> dict[str, Any]:
    return train_live_q_eve_two_route(
        config or LiveQEveTrainingConfig(),
        summary_path=summary_path,
        steps_path=steps_path,
        q_table_path=q_table_path,
        checkpoint_path=checkpoint_path,
    )


def _run_live_control_policy(policy: str,
                             steps_count: int,
                             *,
                             ranking: list[str],
                             config: LiveQEveControlsConfig) -> dict[str, Any]:
    runtime_config = build_two_route_runtime_config()
    rng = np.random.default_rng(config.seed + len(policy))
    avoided_until: dict[str, int] = {}
    steps: list[dict[str, Any]] = []
    rewards: list[float] = []
    step_seconds: list[float] = []
    total_started = time.perf_counter()
    for step_idx in range(steps_count):
        route_id, route_index, fallback_used = _select_alice_route(
            ranking, avoided_until, step_idx)
        action_id = _control_action(policy, route_id, rng)
        attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
        started = time.perf_counter()
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"live_q_eve_control_{policy}_{step_idx}_{route_id}_{action_id}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=config.seed + 100_000 + step_idx,
            ),
        )
        elapsed = time.perf_counter() - started
        public_outcome = runtime_record.get("public_outcome", "blocked")
        if public_outcome in {"delivery_failure", "qber_abort", "blocked"}:
            avoided_until[route_id] = step_idx + config.avoidance_horizon
        reward, reward_components = compute_live_eve_reward(
            public_outcome=public_outcome,
            action_id=action_id,
            fidelity=runtime_record.get("fidelity"),
            attack_cost=config.attack_cost,
        )
        record = _live_control_step_record(
            runtime_record,
            step=step_idx,
            route_id=route_id,
            action_id=action_id,
            policy=policy,
            reward=reward,
            reward_components=reward_components,
            avoidance_summary=_avoidance_summary(ranking, avoided_until, step_idx + 1),
            fallback_used=fallback_used,
            wall_clock_seconds=elapsed,
        )
        steps.append(record)
        rewards.append(reward)
        step_seconds.append(elapsed)
    total_elapsed = time.perf_counter() - total_started
    route_ids = [step["alice_route_id"] for step in steps]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "backend": LIVE_BACKEND,
        "policy": policy,
        "eve_policy_metadata": _control_policy_metadata(policy),
        "steps": len(steps),
        "delivery_failure_rate": sum(
            1 for step in steps if step["public_outcome"] == "delivery_failure"
        ) / len(steps),
        "mean_reward": sum(rewards) / len(rewards),
        "attack_cost_total": sum(
            step["reward_components"]["attack_cost"] for step in steps
        ),
        "wall_clock_seconds_total": total_elapsed,
        "timing_seconds": _timing_summary(step_seconds),
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(
            Counter(step["public_outcome"] for step in steps).items())),
        "final_20_step_replay_window": steps[-min(config.final_window_size, len(steps)):],
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(summary)
    return {
        "scope_label": REPEATER_RUNTIME,
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
    }


def _control_action(policy: str, route_id: str, rng: np.random.Generator) -> str:
    if policy == "random_eve":
        return str(EVE_ACTIONS[int(rng.integers(len(EVE_ACTIONS)))])
    if policy == "always_attack_r0":
        return "swap_denial_r0"
    if policy == "always_attack_r1":
        return "swap_denial_r1"
    if policy == "route_aware_oracle":
        return f"swap_denial_{route_id[-2:]}"
    raise ValueError(f"unsupported live control policy {policy!r}")


def _control_policy_metadata(policy: str) -> dict[str, Any]:
    return {
        "scope_label": CONTROL_GAME if policy == "route_aware_oracle" else REPEATER_RUNTIME,
        "policy": policy,
        "privileged_control": policy == "route_aware_oracle",
        "control_label": (
            "privileged/control" if policy == "route_aware_oracle" else "runtime_control"
        ),
    }


def _live_training_step_record(runtime_record: dict[str, Any],
                               *,
                               step: int,
                               route_id: str,
                               action_id: str,
                               observation: dict[str, Any],
                               epsilon: float,
                               reward: float,
                               reward_components: dict[str, Any],
                               q_update: dict[str, Any],
                               avoidance_summary: dict[str, Any],
                               fallback_used: bool,
                               wall_clock_seconds: float) -> dict[str, Any]:
    record = {
        "step": step,
        "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
        "claim_status": runtime_record.get("claim_status", IMPLEMENTED_AND_TESTABLE),
        "backend": LIVE_BACKEND,
        "alice_policy": "deterministic_cautious_greedy",
        "alice_route_id": route_id,
        "alice_path": ROUTE_PATHS[route_id],
        "eve_policy": LIVE_Q_EVE_POLICY,
        "eve_observation": observation,
        "eve_action": action_id,
        "epsilon": epsilon,
        "public_outcome": runtime_record.get("public_outcome", "blocked"),
        "delivery_success": bool(runtime_record.get("delivery_success", False)),
        "delivered_count": runtime_record.get("delivered_count"),
        "quality_metric": runtime_record.get("fidelity"),
        "fidelity": runtime_record.get("fidelity"),
        "qber": runtime_record.get("qber"),
        "active_route_attacked": bool(runtime_record.get("active_route_attacked", False)),
        "alice_avoidance": avoidance_summary,
        "fallback_used": bool(fallback_used),
        "eve_reward": reward,
        "reward_components": reward_components,
        "reward_input_fields": ["public_outcome", "attack_cost", "public_quality_if_enabled"],
        "q_update": q_update,
        "wall_clock_seconds": wall_clock_seconds,
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
    }
    if runtime_record.get("scope_label") == BLOCKED:
        record["blocked"] = True
        record["blocked_reason"] = runtime_record.get("blocked_reason")
    assert_no_private_public_fields(record)
    return record


def _live_control_step_record(runtime_record: dict[str, Any],
                              *,
                              step: int,
                              route_id: str,
                              action_id: str,
                              policy: str,
                              reward: float,
                              reward_components: dict[str, Any],
                              avoidance_summary: dict[str, Any],
                              fallback_used: bool,
                              wall_clock_seconds: float) -> dict[str, Any]:
    record = {
        "step": step,
        "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
        "claim_status": runtime_record.get("claim_status", IMPLEMENTED_AND_TESTABLE),
        "backend": LIVE_BACKEND,
        "alice_policy": "deterministic_cautious_greedy",
        "alice_route_id": route_id,
        "alice_path": ROUTE_PATHS[route_id],
        "eve_policy": policy,
        "eve_policy_metadata": _control_policy_metadata(policy),
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
        "eve_reward": reward,
        "reward_components": reward_components,
        "wall_clock_seconds": wall_clock_seconds,
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
    }
    assert_no_private_public_fields(record)
    return record


def _live_training_summary(config: LiveQEveTrainingConfig,
                           calibration: dict[str, Any],
                           q_table: QTable,
                           steps: list[dict[str, Any]],
                           reward_trace: list[float],
                           step_seconds: list[float],
                           action_histogram: dict[str, int],
                           completed_steps: int,
                           wall_clock_total: float,
                           stopped_reason: str) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps]
    outcomes = Counter(step["public_outcome"] for step in steps)
    final_window = steps[-min(config.final_window_size, len(steps)):] if steps else []
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "backend": LIVE_BACKEND,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "config": config.to_dict(),
        "completed_steps": completed_steps,
        "target_steps": config.target_steps,
        "stopped_reason": stopped_reason,
        "wall_clock_seconds_total": wall_clock_total,
        "timing_seconds": _timing_summary(step_seconds),
        "mean_seconds_per_step": (
            sum(step_seconds) / len(step_seconds) if step_seconds else None
        ),
        "action_space": list(EVE_ACTIONS),
        "observation_space": ["previous_action", "previous_public_outcome"],
        "learning_rate": config.alpha,
        "discount": config.gamma,
        "epsilon_schedule": {
            "scope_label": REPEATER_RUNTIME,
            "start": config.epsilon_start,
            "end": config.epsilon_end,
            "target_steps": config.target_steps,
        },
        "attack_cost": config.attack_cost,
        "action_histogram": dict(sorted(action_histogram.items())),
        "reward_trace_summary": _numeric_summary(reward_trace),
        "reward_trace_tail": reward_trace[-20:],
        "delivery_failure_rate": (
            outcomes.get("delivery_failure", 0) / len(steps) if steps else 0.0
        ),
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(outcomes.items())),
        "top_q_actions": _top_q_actions(q_table),
        "final_20_step_replay_window": final_window,
        "detected_periodic_suffix": detect_route_period(route_ids),
        "route_calibration_summary": calibration,
        "convergence_claim": None,
        "optimality_claim": None,
        "information_gain_reward_enabled": False,
        "cached_training_used": False,
        "blocked_claims": _live_blocked_claims(),
    }
    assert_no_private_public_fields(summary)
    return summary


def _live_blocked_claims() -> dict[str, dict[str, Any]]:
    blocked = _blocked_claims()
    blocked["cached_training_as_primary_result"] = {
        "scope_label": BLOCKED,
        "blocked": True,
        "reason": "cached Q-Eve remains CONTROL_GAME comparison only",
    }
    for key in DEFERRED_RUNTIME_CLAIMS:
        blocked.setdefault(key, {
            "scope_label": BLOCKED,
            "blocked": True,
            "reason": "deferred by live Q-Eve selected-path scope",
        })
    return blocked


def _top_q_actions(q_table: QTable, limit: int = 10) -> list[dict[str, Any]]:
    ranked = []
    for entry in q_table.to_dict()["entries"]:
        for index, value in enumerate(entry["q"]):
            ranked.append({
                "scope_label": REPEATER_RUNTIME,
                "state": entry["state"],
                "action": EVE_ACTIONS[index],
                "q": round(float(value), 6),
            })
    ranked.sort(key=lambda row: (-row["q"], row["state"], row["action"]))
    return ranked[:limit]


def _training_paths(config: LiveQEveTrainingConfig,
                    summary_path: Optional[Path],
                    steps_path: Optional[Path],
                    q_table_path: Optional[Path],
                    checkpoint_path: Optional[Path]) -> dict[str, Optional[Path]]:
    if config.output_prefix:
        prefix = Path(config.output_prefix)
        summary_path = summary_path or Path(f"{prefix}_summary.json")
        steps_path = steps_path or Path(f"{prefix}_steps.jsonl")
        q_table_path = q_table_path or Path(f"{prefix}_q_table.json")
        checkpoint_path = checkpoint_path or Path(f"{prefix}_checkpoint.json")
    return {
        "summary": summary_path,
        "steps": steps_path,
        "q_table": q_table_path,
        "checkpoint": checkpoint_path,
    }


def _write_live_training_outputs(paths: dict[str, Optional[Path]],
                                 summary: dict[str, Any],
                                 q_table: QTable,
                                 config: LiveQEveTrainingConfig,
                                 previous_action: str,
                                 previous_public_outcome: str,
                                 avoided_until: dict[str, int]) -> None:
    if paths["summary"] is not None:
        paths["summary"].parent.mkdir(parents=True, exist_ok=True)
        paths["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if paths["q_table"] is not None:
        paths["q_table"].parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope_label": REPEATER_RUNTIME,
            "backend": LIVE_BACKEND,
            "q_table": q_table.to_dict(),
            "information_gain_reward_enabled": False,
        }
        paths["q_table"].write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if paths["checkpoint"] is not None:
        _write_checkpoint(
            paths["checkpoint"],
            config=config,
            q_table=q_table,
            completed_steps=summary["completed_steps"],
            previous_action=previous_action,
            previous_public_outcome=previous_public_outcome,
            avoided_until=avoided_until,
        )


def _write_checkpoint(path: Path,
                      *,
                      config: LiveQEveTrainingConfig,
                      q_table: QTable,
                      completed_steps: int,
                      previous_action: str,
                      previous_public_outcome: str,
                      avoided_until: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "backend": LIVE_BACKEND,
        "config": config.to_dict(),
        "completed_steps": completed_steps,
        "previous_action": previous_action,
        "previous_public_outcome": previous_public_outcome,
        "avoided_until": dict(sorted(avoided_until.items())),
        "q_table": q_table.to_dict(),
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_LIVE_Q_TABLE_PATH",
    "DEFAULT_LIVE_TRAINING_PREFIX",
    "LIVE_BACKEND",
    "LIVE_Q_EVE_POLICY",
    "LiveQEveControlsConfig",
    "LiveQEveTrainingConfig",
    "calibrate_live_q_eve_routes",
    "compute_live_eve_reward",
    "run_live_q_eve_controls",
    "train_live_q_eve_two_route",
    "write_live_q_eve_controls",
    "write_live_q_eve_training_artifacts",
]
