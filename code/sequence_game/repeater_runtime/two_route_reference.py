"""Actual-runtime reference for two selected one-repeater paths.

This module deliberately does not map arbitrary generated graphs into
SeQUeNCe.  It runs only the explicit selected paths:

``alice -- r0 -- bob`` and ``alice -- r1 -- bob``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..claims import (
    BLOCKED,
    BLOCKED_PENDING_MAPPING,
    IMPLEMENTED_AND_TESTABLE,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
)
from ..eve.repeater_attacks import RepeaterAttackSpec
from ..game.repeater_game import TwoRouteRepeaterControlConfig
from ..physical.registry import PhysicalModel
from ..protocol.postprocessing import SiftingConfig
from ..protocol.repeater_trial import RepeaterE91RunConfig, run_fixed_repeater_e91_trial
from ..sequence_build.repeater_e91_builder import FixedRepeaterPath

TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL = "selected_two_route_reference_alice_r0_r1_bob"
REWARD_SCOPE = "availability_quality_only"


@dataclass(frozen=True)
class TwoRouteRuntimeCase:
    case_id: str
    selected_route_index: int
    configured_attack_route_index: Optional[int]
    attack_kind: str
    seed: int

    def __post_init__(self) -> None:
        if self.selected_route_index not in (0, 1):
            raise ValueError("selected_route_index must be 0 or 1")
        if self.configured_attack_route_index is not None and self.configured_attack_route_index not in (0, 1):
            raise ValueError("configured_attack_route_index must be 0 or 1 or None")
        if self.attack_kind not in {
            "no_attack",
            "swap_denial",
            "memory_degradation",
            "repeater_memory_measure_X",
            "repeater_memory_dephasing_probe_pi2",
        }:
            raise ValueError(f"unsupported reference attack {self.attack_kind!r}")


def _toy_memory_model() -> PhysicalModel:
    return PhysicalModel("two_route_reference_memory", "memory", "toy", "test-only high-delivery memory", {
        "fidelity": 1.0,
        "frequency": 8e7,
        "efficiency": 1.0,
        "coherence_time": -1,
        "wavelength": 500.0,
    })


def _toy_fiber_model() -> PhysicalModel:
    return PhysicalModel("two_route_reference_fiber", "fiber_channel", "toy", "test-only high-delivery fiber", {
        "attenuation": 0.0,
        "polarization_fidelity": 1.0,
        "light_speed": 2e-4,
        "frequency": 8e7,
    })


def build_two_route_runtime_config() -> TwoRouteRepeaterControlConfig:
    """Build the explicit two-route runtime fixture.

    Parameters are chosen for stable, fast tests only.  They are not hardware
    claims.
    """

    run_config = RepeaterE91RunConfig(
        memory_pairs=8,
        sifting=SiftingConfig(qber_threshold=0.5, min_sifted_samples=1),
        request_fidelity=0.01,
        start_time_ps=1_000_000,
        end_time_ps=80_000_000,
        stop_time_ps=100_000_000,
        swapping_success_prob=1.0,
        swapping_degradation=1.0,
    )
    return TwoRouteRepeaterControlConfig(
        memory_model=_toy_memory_model(),
        fiber_model=_toy_fiber_model(),
        run_config=run_config,
        routes=(
            FixedRepeaterPath(nodes=("alice", "r0", "bob"), edge_lengths_m=(10.0, 10.0)),
            FixedRepeaterPath(nodes=("alice", "r1", "bob"), edge_lengths_m=(10.0, 10.0)),
        ),
    )


def default_two_route_runtime_cases() -> list[TwoRouteRuntimeCase]:
    return [
        TwoRouteRuntimeCase("no_attack_route_r0", 0, None, "no_attack", 101),
        TwoRouteRuntimeCase("no_attack_route_r1", 1, None, "no_attack", 102),
        TwoRouteRuntimeCase("swap_denial_active_r0", 0, 0, "swap_denial", 103),
        TwoRouteRuntimeCase("swap_denial_inactive_r0_route_r1", 1, 0, "swap_denial", 104),
        TwoRouteRuntimeCase("memory_degradation_active_r0", 0, 0, "memory_degradation", 105),
        TwoRouteRuntimeCase("memory_degradation_inactive_r0_route_r1", 1, 0, "memory_degradation", 106),
    ]


def run_two_route_runtime_case(config: TwoRouteRepeaterControlConfig,
                               case: TwoRouteRuntimeCase) -> dict[str, Any]:
    selected_path = config.routes[case.selected_route_index]
    active_attack = (
        case.configured_attack_route_index == case.selected_route_index
        and case.attack_kind != "no_attack"
    )
    configured_target_node = (
        config.routes[case.configured_attack_route_index].nodes[1]
        if case.configured_attack_route_index is not None else None
    )
    runtime_attack = RepeaterAttackSpec(kind="none")
    if active_attack:
        runtime_attack = _runtime_attack_spec(case.attack_kind, selected_path.nodes[1])

    try:
        transcript, _built = run_fixed_repeater_e91_trial(
            selected_path,
            memory_model=config.memory_model,
            fiber_model=config.fiber_model,
            run_config=config.run_config,
            trial_id=case.case_id,
            seed=case.seed,
            attack=runtime_attack,
        )
    except Exception as exc:  # pragma: no cover - fail-closed fallback
        return {
            "case_id": case.case_id,
            "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
            "blocked": True,
            "blocked_reason": f"two-route selected-path runtime failed: {exc}",
            "scope_label": BLOCKED,
            "claim_status": BLOCKED_PENDING_MAPPING,
        }

    record = {
        "case_id": case.case_id,
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "selected_route": "route_r0" if case.selected_route_index == 0 else "route_r1",
        "selected_path": list(selected_path.nodes),
        "configured_attack": {
            "kind": case.attack_kind,
            "target_route": (
                None if case.configured_attack_route_index is None
                else ("route_r0" if case.configured_attack_route_index == 0 else "route_r1")
            ),
            "target_node": configured_target_node,
        },
        "runtime_attack_applied": runtime_attack.to_dict(),
        "active_route_attacked": active_attack,
        "attack_condition": _attack_condition(case, active_attack),
        "delivery_success": transcript.generation_successes > 0,
        "delivered_count": transcript.generation_successes,
        "qber": transcript.qber_estimate,
        "fidelity": transcript.extra.get("mean_actual_fidelity"),
        "amer_style_d_eff_mean": transcript.extra.get("amer_style_d_eff_mean"),
        "public_outcome": _public_outcome(transcript),
        "abort_reason": transcript.abort_reason,
        "reward_scope": REWARD_SCOPE,
        "information_gain_reward_enabled": False,
        "eve_information_metric": None,
    }
    assert_no_private_public_fields(record)
    return record


def run_two_route_runtime_reference(
        cases: Optional[list[TwoRouteRuntimeCase]] = None) -> dict[str, Any]:
    config = build_two_route_runtime_config()
    selected_paths = {
        "route_r0": list(config.routes[0].nodes),
        "route_r1": list(config.routes[1].nodes),
    }
    records = [
        run_two_route_runtime_case(config, case)
        for case in (cases or default_two_route_runtime_cases())
    ]
    payload = {
        "topology_label": TWO_ROUTE_REFERENCE_TOPOLOGY_LABEL,
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "selected_paths": selected_paths,
        "runtime_parameters": {
            "parameter_scope": "high_delivery_small_test_not_hardware_claim",
            "memory_pairs": config.run_config.memory_pairs,
            "request_fidelity": config.run_config.request_fidelity,
            "swapping_success_prob": config.run_config.swapping_success_prob,
            "swapping_degradation": config.run_config.swapping_degradation,
        },
        "reward_scope": REWARD_SCOPE,
        "information_gain_reward_enabled": False,
        "eve_information_metric": None,
        "arbitrary_graph_runtime": {
            "scope_label": BLOCKED,
            "blocked": True,
            "blocked_reason": "arbitrary graph-to-SeQUeNCe route execution is not implemented",
        },
        "cases": records,
    }
    assert_no_private_public_fields(payload)
    return payload


def write_two_route_runtime_reference(path: Path) -> dict[str, Any]:
    payload = run_two_route_runtime_reference()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _public_outcome(transcript) -> str:
    if transcript.accepted:
        return "accepted"
    if transcript.abort_reason == "qber_above_threshold":
        return "qber_abort"
    return "delivery_failure"


def _runtime_attack_spec(attack_kind: str, target_node: str) -> RepeaterAttackSpec:
    if attack_kind == "memory_degradation":
        return RepeaterAttackSpec(
            kind="memory_degradation",
            target_node=target_node,
            memory_fidelity_multiplier=0.5,
        )
    if attack_kind == "swap_denial":
        return RepeaterAttackSpec(
            kind="swap_denial",
            target_node=target_node,
        )
    if attack_kind == "repeater_memory_measure_X":
        return RepeaterAttackSpec(
            kind="repeater_memory_measurement",
            target_node=target_node,
            basis="X",
            attack_id=f"repeater_memory_measure_X:{target_node}",
        )
    if attack_kind == "repeater_memory_dephasing_probe_pi2":
        return RepeaterAttackSpec(
            kind="repeater_memory_dephasing_probe",
            target_node=target_node,
            theta=math.pi / 2,
            theta_label="pi/2",
            attack_id=f"repeater_memory_dephasing_probe:{target_node}:pi/2",
        )
    raise ValueError(f"unsupported reference attack {attack_kind!r}")


def _attack_condition(case: TwoRouteRuntimeCase, active_attack: bool) -> str:
    if case.attack_kind == "no_attack":
        return "no_attack"
    if active_attack:
        return f"{case.attack_kind}_on_active_route"
    return f"{case.attack_kind}_configured_on_inactive_route"
