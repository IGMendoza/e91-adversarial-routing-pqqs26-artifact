"""Deterministic two-route episodes over actual selected-path runtime.

This is the non-learning runtime analogue of the binary collision control game.
It runs only the explicit selected paths ``alice-r0-bob`` and ``alice-r1-bob``.
Arbitrary graph-to-SeQUeNCe execution remains blocked elsewhere.
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
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
)
from .two_route_reference import (
    REWARD_SCOPE,
    TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
    TwoRouteRuntimeCase,
    build_two_route_runtime_config,
    run_two_route_runtime_case,
)

ROUTE_IDS = ("route_r0", "route_r1")
ROUTE_PATHS = {
    "route_r0": ["alice", "r0", "bob"],
    "route_r1": ["alice", "r1", "bob"],
}
EVE_POLICIES = (
    "no_attack",
    "always_attack_r0",
    "always_attack_r1",
    "random_attack",
    "route_aware_oracle",
)
EVE_ACTIONS = (
    "swap_denial_r0",
    "swap_denial_r1",
    "memory_degradation_r0",
    "memory_degradation_r1",
    "repeater_memory_measure_X_r0",
    "repeater_memory_measure_X_r1",
    "repeater_memory_dephasing_probe_pi2_r0",
    "repeater_memory_dephasing_probe_pi2_r1",
)
BASELINE_ACTION = "no_attack"
DEFERRED_RUNTIME_CLAIMS = (
    "arbitrary_graph_runtime",
    "q_learning_on_repeater_runtime",
    "information_gain_reward",
    "harkness_security_coupling",
    "detector_side_channels",
    "chsh_endpoint_monitor",
    "q_neat",
    "graph_objective",
)


@dataclass(frozen=True)
class TwoRouteEpisodeConfig:
    num_steps: int = 12
    seed: int = 1
    eve_policy: str = "no_attack"
    avoidance_horizon: int = 2
    attack_kind: str = "swap_denial"
    final_window_size: int = 20

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.eve_policy not in EVE_POLICIES:
            raise ValueError(f"unsupported Eve policy {self.eve_policy!r}")
        if self.avoidance_horizon < 0:
            raise ValueError("avoidance_horizon must be >= 0")
        if self.attack_kind not in {"swap_denial", "memory_degradation"}:
            raise ValueError("attack_kind must be swap_denial or memory_degradation")
        if self.final_window_size < 1:
            raise ValueError("final_window_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "seed": self.seed,
            "eve_policy": self.eve_policy,
            "avoidance_horizon": self.avoidance_horizon,
            "attack_kind": self.attack_kind,
            "final_window_size": self.final_window_size,
        }


def calibrate_two_route_runtime(seed: int = 1) -> dict[str, Any]:
    """Run honest selected-path calibration and rank routes deterministically."""

    runtime_config = build_two_route_runtime_config()
    calibration_records = []
    for index, route_id in enumerate(ROUTE_IDS):
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"calibration_{route_id}",
                selected_route_index=index,
                configured_attack_route_index=None,
                attack_kind="no_attack",
                seed=seed + index,
            ),
        )
        calibration_records.append(_public_calibration_record(
            runtime_record, route_id))

    def rank_key(record: dict[str, Any]) -> tuple[int, int, float, str]:
        accepted = 1 if record.get("public_outcome") == "accepted" else 0
        delivered = 1 if record.get("delivery_success") else 0
        fidelity = record.get("fidelity")
        quality = float(fidelity) if fidelity is not None else -1.0
        return (-accepted, -delivered, -quality, record["selected_route"])

    ranked = sorted(calibration_records, key=rank_key)
    payload = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "ranking_basis": (
            "accepted runtime outcome, delivery success, fidelity diagnostic, "
            "deterministic route-id tie-break"
        ),
        "routes": calibration_records,
        "ranked_route_ids": [record["selected_route"] for record in ranked],
    }
    assert_no_private_public_fields(payload)
    return payload


def run_two_route_deterministic_episode(
        config: TwoRouteEpisodeConfig | None = None) -> dict[str, Any]:
    config = config or TwoRouteEpisodeConfig()
    runtime_config = build_two_route_runtime_config()
    calibration = calibrate_two_route_runtime(seed=config.seed + 10_000)
    ranking = list(calibration["ranked_route_ids"])
    rng = np.random.default_rng(config.seed)
    avoided_until: dict[str, int] = {}
    steps: list[dict[str, Any]] = []

    for step_idx in range(config.num_steps):
        route_id, route_index, fallback_used = _select_alice_route(
            ranking, avoided_until, step_idx)
        eve_action, privileged = _select_eve_action(
            config.eve_policy, route_index, config.attack_kind, rng)
        attack_route_index, attack_kind = _action_to_runtime_attack(eve_action)
        runtime_record = run_two_route_runtime_case(
            runtime_config,
            TwoRouteRuntimeCase(
                case_id=f"episode_step_{step_idx}_{route_id}_{eve_action}",
                selected_route_index=route_index,
                configured_attack_route_index=attack_route_index,
                attack_kind=attack_kind,
                seed=config.seed + 20_000 + step_idx,
            ),
        )
        public_outcome = runtime_record.get("public_outcome", "blocked")
        failure = public_outcome in {"delivery_failure", "qber_abort", "blocked"}
        if failure:
            avoided_until[route_id] = step_idx + config.avoidance_horizon
        avoidance_summary = _avoidance_summary(ranking, avoided_until, step_idx + 1)
        record = {
            "step": step_idx,
            "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
            "alice_route_id": route_id,
            "alice_path": ROUTE_PATHS[route_id],
            "alice_policy": "deterministic_cautious_greedy",
            "eve_policy": config.eve_policy,
            "eve_policy_metadata": _eve_policy_metadata(config.eve_policy, privileged),
            "eve_action": eve_action,
            "public_outcome": public_outcome,
            "delivery_success": bool(runtime_record.get("delivery_success", False)),
            "delivered_count": runtime_record.get("delivered_count"),
            "quality_metric": runtime_record.get("fidelity"),
            "fidelity": runtime_record.get("fidelity"),
            "qber": runtime_record.get("qber"),
            "alice_avoidance": avoidance_summary,
            "fallback_used": bool(fallback_used),
            "active_route_attacked": bool(runtime_record.get("active_route_attacked", False)),
            "information_gain_reward_enabled": False,
            "reward_scope": REWARD_SCOPE,
        }
        if runtime_record.get("scope_label") == BLOCKED:
            record["blocked_reason"] = runtime_record.get("blocked_reason")
        assert_no_private_public_fields(record)
        steps.append(record)

    summary = _episode_summary(config, calibration, steps)
    return {
        "summary": summary,
        "steps": steps,
        "final_window": summary["final_20_step_replay_window"],
    }


def write_two_route_deterministic_episode_artifacts(summary_path: Path,
                                                    steps_path: Path,
                                                    *,
                                                    config: TwoRouteEpisodeConfig | None = None
                                                    ) -> dict[str, Any]:
    result = run_two_route_deterministic_episode(config or TwoRouteEpisodeConfig(
        num_steps=20,
        seed=77,
        eve_policy="route_aware_oracle",
        avoidance_horizon=2,
        attack_kind="swap_denial",
    ))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with steps_path.open("w", encoding="utf-8") as f:
        for step in result["steps"]:
            f.write(json.dumps(step, sort_keys=True) + "\n")
    return result


def _select_alice_route(ranking: list[str], avoided_until: dict[str, int],
                        step_idx: int) -> tuple[str, int, bool]:
    available = [
        route_id for route_id in ranking
        if step_idx > avoided_until.get(route_id, -1)
    ]
    fallback = not available
    route_id = (available or ranking)[0]
    return route_id, ROUTE_IDS.index(route_id), fallback


def _select_eve_action(eve_policy: str, selected_route_index: int,
                       attack_kind: str, rng: np.random.Generator) -> tuple[str, bool]:
    if eve_policy == "no_attack":
        return BASELINE_ACTION, False
    if eve_policy == "always_attack_r0":
        return f"{attack_kind}_r0", False
    if eve_policy == "always_attack_r1":
        return f"{attack_kind}_r1", False
    if eve_policy == "random_attack":
        return str(EVE_ACTIONS[int(rng.integers(len(EVE_ACTIONS)))]), False
    if eve_policy == "route_aware_oracle":
        return f"{attack_kind}_r{selected_route_index}", True
    raise ValueError(f"unsupported Eve policy {eve_policy!r}")


def _eve_policy_metadata(eve_policy: str, privileged: bool) -> dict[str, Any]:
    return {
        "scope_label": CONTROL_GAME,
        "policy": eve_policy,
        "privileged_control": privileged,
        "control_type": (
            "route_aware_oracle" if privileged else "deterministic_runtime_control"
        ),
    }


def _action_to_runtime_attack(action: str) -> tuple[Optional[int], str]:
    if action == BASELINE_ACTION:
        return None, "no_attack"
    if action.endswith("_r0"):
        route_index = 0
        attack_name = action[:-3]
    elif action.endswith("_r1"):
        route_index = 1
        attack_name = action[:-3]
    else:
        raise ValueError(f"unsupported Eve action {action!r}")
    if attack_name not in {
        "swap_denial",
        "memory_degradation",
        "repeater_memory_measure_X",
        "repeater_memory_dephasing_probe_pi2",
    }:
        raise ValueError(f"unsupported Eve action {action!r}")
    return route_index, attack_name


def _avoidance_summary(ranking: list[str], avoided_until: dict[str, int],
                       next_step: int) -> dict[str, Any]:
    avoided = [
        route_id for route_id in ranking
        if next_step <= avoided_until.get(route_id, -1)
    ]
    return {
        "scope_label": REPEATER_RUNTIME,
        "avoided_routes": avoided,
        "num_avoided_routes": len(avoided),
    }


def _public_calibration_record(runtime_record: dict[str, Any],
                               route_id: str) -> dict[str, Any]:
    return {
        "scope_label": runtime_record.get("scope_label", REPEATER_RUNTIME),
        "claim_status": runtime_record.get("claim_status"),
        "selected_route": route_id,
        "selected_path": ROUTE_PATHS[route_id],
        "public_outcome": runtime_record.get("public_outcome", "blocked"),
        "delivery_success": bool(runtime_record.get("delivery_success", False)),
        "delivered_count": runtime_record.get("delivered_count"),
        "qber": runtime_record.get("qber"),
        "fidelity": runtime_record.get("fidelity"),
        "reward_scope": REWARD_SCOPE,
        "information_gain_reward_enabled": False,
        "blocked": bool(runtime_record.get("blocked", False)),
        "blocked_reason": runtime_record.get("blocked_reason"),
    }


def _episode_summary(config: TwoRouteEpisodeConfig,
                     calibration: dict[str, Any],
                     steps: list[dict[str, Any]]) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps]
    outcomes = Counter(step["public_outcome"] for step in steps)
    fidelities = [
        float(step["fidelity"]) for step in steps
        if step.get("fidelity") is not None
    ]
    qbers = [
        float(step["qber"]) for step in steps
        if step.get("qber") is not None
    ]
    final_window = steps[-min(config.final_window_size, len(steps)):]
    summary = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "seed": config.seed,
        "num_steps": len(steps),
        "alice_policy": "deterministic_cautious_greedy",
        "eve_policy": config.eve_policy,
        "eve_policy_metadata": _eve_policy_metadata(
            config.eve_policy, config.eve_policy == "route_aware_oracle"),
        "route_calibration_summary": calibration,
        "route_visit_counts": dict(sorted(Counter(route_ids).items())),
        "public_outcome_counts": dict(sorted(outcomes.items())),
        "delivery_failure_rate": sum(
            1 for step in steps if not step.get("delivery_success", False)
        ) / len(steps),
        "quality_fidelity_summary": _numeric_summary(fidelities),
        "qber_summary": _numeric_summary(qbers),
        "route_switch_count": sum(1 for a, b in zip(route_ids, route_ids[1:]) if a != b),
        "final_20_step_replay_window": final_window,
        "detected_periodic_suffix": detect_route_period(route_ids),
        "information_gain_reward_enabled": False,
        "reward_scope": REWARD_SCOPE,
        "blocked_claims": _blocked_claims(),
    }
    assert_no_private_public_fields(summary)
    return summary


def _blocked_claims() -> dict[str, dict[str, Any]]:
    return {
        claim: {
            "scope_label": BLOCKED,
            "blocked": True,
            "reason": _blocked_reason(claim),
        }
        for claim in DEFERRED_RUNTIME_CLAIMS
    }


def _blocked_reason(claim: str) -> str:
    reasons = {
        "arbitrary_graph_runtime": (
            "graph-to-SeQUeNCe route interface is not implemented"
        ),
        "q_learning_on_repeater_runtime": (
            "this milestone is deterministic control only"
        ),
        "information_gain_reward": (
            "no cited information metric or public-transcript mapping is enabled"
        ),
        "harkness_security_coupling": (
            "security-reference layer is not coupled to runtime diagnostics"
        ),
        "detector_side_channels": (
            "detector attack surfaces are deferred"
        ),
        "chsh_endpoint_monitor": (
            "CHSH/S endpoint monitoring is deferred"
        ),
        "q_neat": (
            "graph/Q-NEAT objective layer is not implemented"
        ),
        "graph_objective": (
            "hardness or operational graph objective remains unmapped"
        ),
    }
    return reasons[claim]


def _numeric_summary(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }
