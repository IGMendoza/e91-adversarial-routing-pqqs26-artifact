"""Q-learning Eve over two-route repeater-control public outcomes.

High-volume training here is a cached CONTROL_GAME layer calibrated from
selected-path runtime rows.  Live validation remains a small REPEATER_RUNTIME
check that calls the actual selected-path runtime.
"""

from __future__ import annotations

import json
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
    IMPLEMENTED_CONTROL_ONLY,
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

EVIDENCE_SOURCE = "runtime_calibrated_two_route_table"
ACTUAL_RUNTIME_SOURCE = "actual_selected_path_runtime"
LIVE_RUNTIME_BLOCK_REASON = "LIVE_RUNTIME_VALIDATION_TOO_EXPENSIVE_OR_UNSTABLE"
Q_EVE_POLICY = "q_learning_eve"
CONTROL_POLICIES = (
    "no_attack",
    "random_eve",
    "always_attack_r0",
    "always_attack_r1",
    "route_aware_oracle",
    "learned_q_eve",
)


@dataclass(frozen=True)
class TwoRouteQEveTrainingConfig:
    training_steps: int = 2000
    seed: int = 301
    avoidance_horizon: int = 1
    alpha: float = 0.45
    gamma: float = 0.85
    epsilon_start: float = 0.6
    epsilon_end: float = 0.02
    attack_cost: float = 0.05
    quality_penalty_weight: float = 0.0
    final_window_size: int = 20
    control_steps: int = 200

    def __post_init__(self) -> None:
        if self.training_steps < 1:
            raise ValueError("training_steps must be >= 1")
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be in [0, 1]")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be in [0, 1]")
        if self.attack_cost < 0:
            raise ValueError("attack_cost must be >= 0")
        if self.quality_penalty_weight < 0:
            raise ValueError("quality_penalty_weight must be >= 0")
        if self.final_window_size < 1:
            raise ValueError("final_window_size must be >= 1")
        if self.control_steps < 1:
            raise ValueError("control_steps must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_steps": self.training_steps,
            "seed": self.seed,
            "avoidance_horizon": self.avoidance_horizon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "attack_cost": self.attack_cost,
            "quality_penalty_weight": self.quality_penalty_weight,
            "final_window_size": self.final_window_size,
            "control_steps": self.control_steps,
        }


@dataclass(frozen=True)
class TwoRouteLiveValidationConfig:
    validation_steps: int = 10
    seed: int = 901
    avoidance_horizon: int = 1
    learned_policy_source_artifact: str = (
        "results/repeater_runtime/q_eve_two_route_training_summary.json"
    )
    final_window_size: int = 20

    def __post_init__(self) -> None:
        if self.validation_steps < 1:
            raise ValueError("validation_steps must be >= 1")
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if self.final_window_size < 1:
            raise ValueError("final_window_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_steps": self.validation_steps,
            "seed": self.seed,
            "avoidance_horizon": self.avoidance_horizon,
            "learned_policy_source_artifact": self.learned_policy_source_artifact,
            "final_window_size": self.final_window_size,
        }


class CachedTwoRouteRepeaterControlEnv:
    """Cached public-outcome environment for Q Eve.

    The environment has access to Alice's selected route so it can query the
    runtime-calibrated table, but Eve's observation is only
    ``previous_action, previous_public_outcome``.
    """

    def __init__(self,
                 outcome_table: dict[str, Any],
                 *,
                 avoidance_horizon: int,
                 attack_cost: float,
                 quality_penalty_weight: float = 0.0):
        if avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if attack_cost < 0:
            raise ValueError("attack_cost must be >= 0")
        if quality_penalty_weight < 0:
            raise ValueError("quality_penalty_weight must be >= 0")
        self.outcome_table = outcome_table
        self.avoidance_horizon = avoidance_horizon
        self.attack_cost = attack_cost
        self.quality_penalty_weight = quality_penalty_weight
        self.ranking = _rank_routes_from_outcome_table(outcome_table)
        self.lookup = _outcome_lookup(outcome_table)
        self.reset()

    def reset(self) -> tuple[str, str]:
        self.step_index = 0
        self.previous_action = "start"
        self.previous_public_outcome = "start"
        self.avoided_until: dict[str, int] = {}
        return self.observation()

    def observation(self) -> tuple[str, str]:
        observation = (
            str(self.previous_action),
            str(self.previous_public_outcome),
        )
        assert_public_eve_observation({
            "scope_label": CONTROL_GAME,
            "previous_action": observation[0],
            "previous_public_outcome": observation[1],
        })
        return observation

    def current_alice_choice(self) -> tuple[str, int, bool]:
        return _select_alice_route(
            self.ranking, self.avoided_until, self.step_index)

    def step(self, action_id: str, *, eve_policy: str) -> dict[str, Any]:
        if action_id != BASELINE_ACTION and action_id not in EVE_ACTIONS:
            raise ValueError(f"unsupported Eve action {action_id!r}")
        observation = {
            "scope_label": CONTROL_GAME,
            "previous_action": self.previous_action,
            "previous_public_outcome": self.previous_public_outcome,
        }
        assert_public_eve_observation(observation)
        route_id, route_index, fallback_used = self.current_alice_choice()
        table_row = self.lookup[(route_id, action_id)]
        public_outcome = str(table_row["public_outcome"])
        delivery_success = bool(table_row["delivery_success"])
        failure = public_outcome in {"delivery_failure", "qber_abort", "blocked"}
        if failure:
            self.avoided_until[route_id] = self.step_index + self.avoidance_horizon
        reward, reward_components = _cached_reward(
            table_row,
            action_id,
            attack_cost=self.attack_cost,
            quality_penalty_weight=self.quality_penalty_weight,
        )
        avoidance = _avoidance_summary(
            self.ranking, self.avoided_until, self.step_index + 1)
        avoidance["scope_label"] = CONTROL_GAME
        record = {
            "step": self.step_index,
            "scope_label": CONTROL_GAME,
            "claim_status": IMPLEMENTED_CONTROL_ONLY,
            "evidence_source": EVIDENCE_SOURCE,
            "alice_policy": "deterministic_cautious_greedy",
            "alice_route_id": route_id,
            "alice_path": ROUTE_PATHS[route_id],
            "eve_policy": eve_policy,
            "eve_observation": observation,
            "eve_action": action_id,
            "public_outcome": public_outcome,
            "delivery_success": delivery_success,
            "quality_metric": table_row.get("quality_metric"),
            "fidelity": table_row.get("fidelity"),
            "qber": table_row.get("qber"),
            "active_route_attacked": bool(table_row["active_route_attacked"]),
            "selected_route_hit": bool(table_row["active_route_attacked"]),
            "alice_avoidance": avoidance,
            "fallback_used": bool(fallback_used),
            "eve_reward": reward,
            "reward_components": reward_components,
            "reward_input_fields": [
                "public_outcome",
                "quality_degradation",
                "attack_cost",
            ],
            "information_gain_reward_enabled": False,
            "reward_scope": REWARD_SCOPE,
        }
        assert_no_private_public_fields(record)
        self.previous_action = action_id
        self.previous_public_outcome = public_outcome
        self.step_index += 1
        return record


def build_two_route_runtime_outcome_table(*, seed: int = 701) -> dict[str, Any]:
    """Build the 2-route/action table from actual selected-path runtime cases."""

    runtime_config = build_two_route_runtime_config()
    rows: list[dict[str, Any]] = []
    case_index = 0
    for route_index, route_id in enumerate(ROUTE_IDS):
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"outcome_table_{route_id}_{BASELINE_ACTION}",
                selected_route_index=route_index,
                configured_attack_route_index=None,
                attack_kind=BASELINE_ACTION,
                seed=seed + case_index,
            ),
        )
        rows.append(_outcome_table_row(runtime_record, route_id, BASELINE_ACTION))
        case_index += 1
    for route_index, route_id in enumerate(ROUTE_IDS):
        for action_id in EVE_ACTIONS:
            attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
            runtime_record = run_two_route_runtime_case(
                runtime_config,
                TwoRouteRuntimeCase(
                    case_id=f"outcome_table_{route_id}_{action_id}",
                    selected_route_index=route_index,
                    configured_attack_route_index=attack_route_index,
                    attack_kind=attack_kind,
                    seed=seed + case_index,
                ),
            )
            rows.append(_outcome_table_row(runtime_record, route_id, action_id))
            case_index += 1

    rows = _add_quality_degradation(rows)
    table = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "source": ACTUAL_RUNTIME_SOURCE,
        "evidence_for": EVIDENCE_SOURCE,
        "route_ids": list(ROUTE_IDS),
        "action_space": list(EVE_ACTIONS),
        "calibration_action": BASELINE_ACTION,
        "rows": rows,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(table)
    return table


def write_two_route_runtime_outcome_table(path: Path,
                                          *,
                                          seed: int = 701) -> dict[str, Any]:
    table = build_two_route_runtime_outcome_table(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return table


def load_two_route_runtime_outcome_table(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_q_eve_two_route(
        config: TwoRouteQEveTrainingConfig | None = None,
        *,
        outcome_table: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    config = config or TwoRouteQEveTrainingConfig()
    table_payload = outcome_table or build_two_route_runtime_outcome_table()
    env = CachedTwoRouteRepeaterControlEnv(
        table_payload,
        avoidance_horizon=config.avoidance_horizon,
        attack_cost=config.attack_cost,
        quality_penalty_weight=config.quality_penalty_weight,
    )
    q_table = QTable(num_actions=len(EVE_ACTIONS))
    q_config = QLearningConfig(alpha=config.alpha, gamma=config.gamma)
    epsilon_schedule = linear_epsilon_schedule(
        config.epsilon_start, config.epsilon_end, config.training_steps)
    rng = np.random.default_rng(config.seed)
    state = env.reset()
    steps: list[dict[str, Any]] = []
    action_histogram = {action_id: 0 for action_id in EVE_ACTIONS}

    for step_idx in range(config.training_steps):
        epsilon = epsilon_schedule(step_idx)
        action_index = epsilon_greedy(q_table, state, epsilon, rng)
        action_id = EVE_ACTIONS[action_index]
        action_histogram[action_id] += 1
        record = env.step(action_id, eve_policy=Q_EVE_POLICY)
        next_state = env.observation()
        new_q = q_table.update(
            state, action_index, float(record["eve_reward"]), next_state, q_config)
        record["epsilon"] = epsilon
        record["q_update"] = {
            "scope_label": CONTROL_GAME,
            "state": list(state),
            "action": action_id,
            "new_q": round(float(new_q), 6),
        }
        steps.append(record)
        state = next_state

    controls = evaluate_q_eve_controls(
        q_table, table_payload, config=config, seed=config.seed + 50_000)
    summary = _training_summary(
        config, table_payload, q_table, steps, action_histogram, controls)
    return {
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
        "q_table": q_table,
        "outcome_table": table_payload,
    }


def evaluate_q_eve_controls(q_table: QTable,
                            outcome_table: dict[str, Any],
                            *,
                            config: TwoRouteQEveTrainingConfig,
                            seed: int) -> dict[str, Any]:
    controls = {
        "scope_label": CONTROL_GAME,
        "evidence_source": EVIDENCE_SOURCE,
        "policies": {},
    }
    for offset, policy in enumerate(CONTROL_POLICIES):
        controls["policies"][policy] = _evaluate_control_policy(
            policy,
            q_table,
            outcome_table,
            config=config,
            seed=seed + offset,
        )
    return controls


def run_q_eve_live_validation(q_table: QTable,
                              outcome_table: dict[str, Any],
                              *,
                              config: TwoRouteLiveValidationConfig | None = None
                              ) -> dict[str, Any]:
    config = config or TwoRouteLiveValidationConfig()
    runtime_config = build_two_route_runtime_config()
    ranking = _rank_routes_from_outcome_table(outcome_table)
    avoided_until: dict[str, int] = {}
    previous_action = "start"
    previous_public_outcome = "start"
    steps: list[dict[str, Any]] = []

    for step_idx in range(config.validation_steps):
        route_id, route_index, fallback_used = _select_alice_route(
            ranking, avoided_until, step_idx)
        state = (previous_action, previous_public_outcome)
        action_index = q_table.best_action(state)
        action_id = EVE_ACTIONS[action_index]
        attack_route_index, attack_kind = _action_to_runtime_attack(action_id)
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"q_eve_live_validation_{step_idx}_{route_id}_{action_id}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=config.seed + step_idx,
            ),
        )
        if runtime_record.get("scope_label") == BLOCKED:
            return _blocked_live_validation(runtime_record)

        public_outcome = runtime_record.get("public_outcome", "blocked")
        failure = public_outcome in {"delivery_failure", "qber_abort", "blocked"}
        if failure:
            avoided_until[route_id] = step_idx + config.avoidance_horizon
        avoidance = _avoidance_summary(ranking, avoided_until, step_idx + 1)
        record = {
            "step": step_idx,
            "scope_label": REPEATER_RUNTIME,
            "claim_status": IMPLEMENTED_AND_TESTABLE,
            "learned_policy_source_artifact": config.learned_policy_source_artifact,
            "alice_policy": "deterministic_cautious_greedy",
            "alice_route_id": route_id,
            "alice_path": ROUTE_PATHS[route_id],
            "eve_policy": "learned_q_eve",
            "eve_action": action_id,
            "public_outcome": public_outcome,
            "delivery_success": bool(runtime_record.get("delivery_success", False)),
            "delivered_count": runtime_record.get("delivered_count"),
            "quality_metric": runtime_record.get("fidelity"),
            "fidelity": runtime_record.get("fidelity"),
            "qber": runtime_record.get("qber"),
            "active_route_attacked": bool(runtime_record.get("active_route_attacked", False)),
            "alice_avoidance": avoidance,
            "fallback_used": bool(fallback_used),
            "information_gain_reward_enabled": False,
            "reward_scope": REWARD_SCOPE,
        }
        assert_no_private_public_fields(record)
        steps.append(record)
        previous_action = action_id
        previous_public_outcome = str(public_outcome)

    summary = _live_validation_summary(config, steps)
    return {
        "scope_label": REPEATER_RUNTIME,
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
    }


def write_q_eve_two_route_artifacts(
        outcome_table_path: Path,
        summary_path: Path,
        steps_path: Path,
        live_validation_path: Path,
        *,
        training_config: TwoRouteQEveTrainingConfig | None = None,
        live_config: TwoRouteLiveValidationConfig | None = None) -> dict[str, Any]:
    outcome_table = write_two_route_runtime_outcome_table(outcome_table_path)
    training = train_q_eve_two_route(
        training_config or TwoRouteQEveTrainingConfig(),
        outcome_table=outcome_table,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(training["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with steps_path.open("w", encoding="utf-8") as f:
        for step in training["steps"]:
            f.write(json.dumps(step, sort_keys=True) + "\n")

    live = run_q_eve_live_validation(
        training["q_table"],
        outcome_table,
        config=live_config or TwoRouteLiveValidationConfig(),
    )
    live_validation_path.parent.mkdir(parents=True, exist_ok=True)
    live_validation_path.write_text(
        json.dumps(_json_public(live), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "outcome_table": outcome_table,
        "training": training,
        "live_validation": live,
    }


def _outcome_table_row(runtime_record: dict[str, Any],
                       route_id: str,
                       action_id: str) -> dict[str, Any]:
    row = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": runtime_record.get("claim_status", IMPLEMENTED_AND_TESTABLE),
        "selected_route": route_id,
        "selected_path": ROUTE_PATHS[route_id],
        "attack_action": action_id,
        "public_outcome": runtime_record.get("public_outcome", "blocked"),
        "delivery_success": bool(runtime_record.get("delivery_success", False)),
        "delivered_count": runtime_record.get("delivered_count"),
        "quality_metric": runtime_record.get("fidelity"),
        "fidelity": runtime_record.get("fidelity"),
        "qber": runtime_record.get("qber"),
        "active_route_attacked": bool(runtime_record.get("active_route_attacked", False)),
        "runtime_attack_applied": runtime_record.get("runtime_attack_applied"),
        "source": ACTUAL_RUNTIME_SOURCE,
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
    }
    if runtime_record.get("scope_label") == BLOCKED:
        row["scope_label"] = BLOCKED
        row["blocked"] = True
        row["blocked_reason"] = runtime_record.get("blocked_reason")
    assert_no_private_public_fields(row)
    return row


def _add_quality_degradation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline: dict[str, Optional[float]] = {}
    for row in rows:
        if row["attack_action"] == BASELINE_ACTION:
            fidelity = row.get("fidelity")
            baseline[row["selected_route"]] = (
                float(fidelity) if fidelity is not None else None
            )
    enriched = []
    for row in rows:
        copied = dict(row)
        base = baseline.get(row["selected_route"])
        fidelity = row.get("fidelity")
        if base is not None and fidelity is not None:
            copied["quality_degradation"] = max(0.0, base - float(fidelity))
        else:
            copied["quality_degradation"] = 0.0
        enriched.append(copied)
    return enriched


def _outcome_lookup(outcome_table: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {
        (str(row["selected_route"]), str(row["attack_action"])): row
        for row in outcome_table["rows"]
    }
    expected = {
        (route_id, action_id)
        for route_id in ROUTE_IDS
        for action_id in EVE_ACTIONS + (BASELINE_ACTION,)
    }
    missing = expected - set(lookup)
    if missing:
        raise ValueError(f"outcome table missing rows: {sorted(missing)}")
    return lookup


def _rank_routes_from_outcome_table(outcome_table: dict[str, Any]) -> list[str]:
    rows = [
        row for row in outcome_table["rows"]
        if row["attack_action"] == BASELINE_ACTION
    ]

    def key(row: dict[str, Any]) -> tuple[int, int, float, str]:
        accepted = 1 if row.get("public_outcome") == "accepted" else 0
        delivered = 1 if row.get("delivery_success") else 0
        fidelity = row.get("fidelity")
        quality = float(fidelity) if fidelity is not None else -1.0
        return (-accepted, -delivered, -quality, str(row["selected_route"]))

    return [str(row["selected_route"]) for row in sorted(rows, key=key)]


def _cached_reward(row: dict[str, Any],
                   action_id: str,
                   *,
                   attack_cost: float,
                   quality_penalty_weight: float) -> tuple[float, dict[str, Any]]:
    public_failure_reward = 1.0 if row["public_outcome"] == "delivery_failure" else 0.0
    quality_degradation = float(row.get("quality_degradation", 0.0) or 0.0)
    cost = 0.0 if action_id == BASELINE_ACTION else float(attack_cost)
    quality_reward = float(quality_penalty_weight) * quality_degradation
    reward = public_failure_reward + quality_reward - cost
    components = {
        "scope_label": CONTROL_GAME,
        "public_outcome_reward": public_failure_reward,
        "quality_penalty_weight": float(quality_penalty_weight),
        "quality_degradation": quality_degradation,
        "quality_reward": quality_reward,
        "attack_cost": cost,
        "information_gain_reward": 0.0,
    }
    return float(reward), components


def _evaluate_control_policy(policy: str,
                             q_table: QTable,
                             outcome_table: dict[str, Any],
                             *,
                             config: TwoRouteQEveTrainingConfig,
                             seed: int) -> dict[str, Any]:
    env = CachedTwoRouteRepeaterControlEnv(
        outcome_table,
        avoidance_horizon=config.avoidance_horizon,
        attack_cost=config.attack_cost,
        quality_penalty_weight=config.quality_penalty_weight,
    )
    rng = np.random.default_rng(seed)
    env.reset()
    steps: list[dict[str, Any]] = []
    for _ in range(config.control_steps):
        action_id = _control_action(policy, q_table, env, rng)
        steps.append(env.step(action_id, eve_policy=policy))
    return _control_summary(policy, steps)


def _control_action(policy: str,
                    q_table: QTable,
                    env: CachedTwoRouteRepeaterControlEnv,
                    rng: np.random.Generator) -> str:
    if policy == "no_attack":
        return BASELINE_ACTION
    if policy == "random_eve":
        return str(EVE_ACTIONS[int(rng.integers(len(EVE_ACTIONS)))])
    if policy == "always_attack_r0":
        return "swap_denial_r0"
    if policy == "always_attack_r1":
        return "swap_denial_r1"
    if policy == "route_aware_oracle":
        route_id, _route_index, _fallback = env.current_alice_choice()
        return f"swap_denial_{route_id[-2:]}"
    if policy == "learned_q_eve":
        return EVE_ACTIONS[q_table.best_action(env.observation())]
    raise ValueError(f"unsupported control policy {policy!r}")


def _control_summary(policy: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(step["eve_reward"]) for step in steps]
    route_ids = [step["alice_route_id"] for step in steps]
    failures = sum(1 for step in steps if step["public_outcome"] == "delivery_failure")
    active_hits = sum(1 for step in steps if step["selected_route_hit"])
    attack_cost = sum(float(step["reward_components"]["attack_cost"]) for step in steps)
    summary = {
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "evidence_source": EVIDENCE_SOURCE,
        "policy": policy,
        "privileged_control": policy == "route_aware_oracle",
        "control_label": (
            "privileged/control" if policy == "route_aware_oracle" else "control"
        ),
        "steps": len(steps),
        "delivery_failure_rate": failures / len(steps),
        "selected_route_hit_rate": active_hits / len(steps),
        "attack_cost_total": attack_cost,
        "reward_total": sum(rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(
            Counter(step["public_outcome"] for step in steps).items())),
        "final_20_step_replay_window": steps[-min(20, len(steps)):],
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(summary)
    return summary


def _training_summary(config: TwoRouteQEveTrainingConfig,
                      outcome_table: dict[str, Any],
                      q_table: QTable,
                      steps: list[dict[str, Any]],
                      action_histogram: dict[str, int],
                      controls: dict[str, Any]) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps]
    rewards = [float(step["eve_reward"]) for step in steps]
    final_window = steps[-min(config.final_window_size, len(steps)):]
    summary = {
        "scope_label": CONTROL_GAME,
        "claim_status": IMPLEMENTED_CONTROL_ONLY,
        "evidence_source": EVIDENCE_SOURCE,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "seed": config.seed,
        "training_steps": len(steps),
        "action_space": list(EVE_ACTIONS),
        "observation_space": ["previous_action", "previous_public_outcome"],
        "learning_rate": config.alpha,
        "discount": config.gamma,
        "epsilon_schedule": {
            "scope_label": CONTROL_GAME,
            "start": config.epsilon_start,
            "end": config.epsilon_end,
            "steps": config.training_steps,
        },
        "attack_cost": config.attack_cost,
        "quality_penalty_weight": config.quality_penalty_weight,
        "reward_definition": (
            "public delivery_failure reward plus optional public quality "
            "degradation diagnostic minus non-calibration action cost"
        ),
        "reward_trace": rewards,
        "action_histogram": dict(sorted(action_histogram.items())),
        "top_q_actions": _top_q_actions(q_table),
        "q_table_entries": _q_table_public_entries(q_table),
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(
            Counter(step["public_outcome"] for step in steps).items())),
        "delivery_failure_rate": sum(
            1 for step in steps if step["public_outcome"] == "delivery_failure"
        ) / len(steps),
        "selected_route_hit_rate": sum(
            1 for step in steps if step["selected_route_hit"]
        ) / len(steps),
        "mean_reward": sum(rewards) / len(rewards),
        "final_20_step_replay_window": final_window,
        "detected_periodic_suffix": detect_route_period(route_ids),
        "controls": controls,
        "outcome_table_source": {
            "scope_label": REPEATER_RUNTIME,
            "source": outcome_table["source"],
            "row_count": len(outcome_table["rows"]),
        },
        "convergence_claim": None,
        "optimality_claim": None,
        "information_gain_reward_enabled": False,
        "blocked_claims": _blocked_claims(),
    }
    assert_no_private_public_fields(summary)
    return summary


def _top_q_actions(q_table: QTable, limit: int = 10) -> list[dict[str, Any]]:
    ranked = []
    for entry in q_table.to_dict()["entries"]:
        state = entry["state"]
        for index, value in enumerate(entry["q"]):
            ranked.append({
                "scope_label": CONTROL_GAME,
                "state": state,
                "action": EVE_ACTIONS[index],
                "q": round(float(value), 6),
            })
    ranked.sort(key=lambda row: (-row["q"], row["state"], row["action"]))
    return ranked[:limit]


def _q_table_public_entries(q_table: QTable) -> dict[str, Any]:
    return {
        "scope_label": CONTROL_GAME,
        "entries": q_table.to_dict()["entries"],
    }


def _live_validation_summary(config: TwoRouteLiveValidationConfig,
                             steps: list[dict[str, Any]]) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps]
    fidelities = [
        float(step["fidelity"]) for step in steps if step.get("fidelity") is not None
    ]
    qbers = [
        float(step["qber"]) for step in steps if step.get("qber") is not None
    ]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "seed": config.seed,
        "validation_steps": len(steps),
        "learned_policy_source_artifact": config.learned_policy_source_artifact,
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(
            Counter(step["public_outcome"] for step in steps).items())),
        "delivery_failure_rate": sum(
            1 for step in steps if step["public_outcome"] == "delivery_failure"
        ) / len(steps),
        "quality_fidelity_summary": _numeric_summary(fidelities),
        "qber_summary": _numeric_summary(qbers),
        "final_20_step_replay_window": steps[-min(config.final_window_size, len(steps)):],
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
    }
    assert_no_private_public_fields(summary)
    return summary


def _blocked_live_validation(runtime_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope_label": BLOCKED,
        "blocked": True,
        "blocked_reason": (
            f"{LIVE_RUNTIME_BLOCK_REASON}: "
            f"{runtime_record.get('blocked_reason', 'runtime unavailable')}"
        ),
        "information_gain_reward_enabled": False,
    }


def _json_public(payload: Any) -> Any:
    if isinstance(payload, QTable):
        return _q_table_public_entries(payload)
    if isinstance(payload, dict):
        return {
            key: _json_public(value)
            for key, value in payload.items()
            if key != "q_table"
        }
    if isinstance(payload, list):
        return [_json_public(value) for value in payload]
    return payload


__all__ = [
    "ACTUAL_RUNTIME_SOURCE",
    "CONTROL_POLICIES",
    "EVIDENCE_SOURCE",
    "LIVE_RUNTIME_BLOCK_REASON",
    "Q_EVE_POLICY",
    "CachedTwoRouteRepeaterControlEnv",
    "TwoRouteLiveValidationConfig",
    "TwoRouteQEveTrainingConfig",
    "build_two_route_runtime_outcome_table",
    "evaluate_q_eve_controls",
    "load_two_route_runtime_outcome_table",
    "run_q_eve_live_validation",
    "train_q_eve_two_route",
    "write_q_eve_two_route_artifacts",
    "write_two_route_runtime_outcome_table",
]
