"""Capability checks and reference cases for repeater-memory attack surfaces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..claims import (
    BLOCKED,
    BLOCKED_PENDING_MAPPING,
    HARDWARE_DIAGNOSTIC,
    IMPLEMENTED_AND_TESTABLE,
    IMPLEMENTED_DIAGNOSTIC_ONLY,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
)
from ..eve.attack_capabilities import (
    BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
    BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
    BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
    BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC,
    BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
    IMPLEMENTED_MEMORY_DEGRADATION_ATTACK,
    IMPLEMENTED_NO_ATTACK,
    IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK,
    IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK,
    IMPLEMENTED_SWAP_DENIAL_ATTACK,
    blocked_attack_response,
)
from ..eve.repeater_attacks import RepeaterAttackSpec
from .two_route_reference import (
    REWARD_SCOPE,
    TwoRouteRuntimeCase,
    build_two_route_runtime_config,
    run_two_route_runtime_case,
)

REPEATER_MEMORY_MEASUREMENT_ATTACK_IDS = (
    "repeater_memory_measure_Z:r0",
    "repeater_memory_measure_X:r0",
    "repeater_memory_measure_Z:r1",
    "repeater_memory_measure_X:r1",
)
REPEATER_MEMORY_DEPHASING_PROBE_ATTACK_IDS = (
    "repeater_memory_dephasing_probe:r0:0",
    "repeater_memory_dephasing_probe:r0:pi/6",
    "repeater_memory_dephasing_probe:r0:pi/4",
    "repeater_memory_dephasing_probe:r0:pi/2",
    "repeater_memory_dephasing_probe:r1:0",
    "repeater_memory_dephasing_probe:r1:pi/6",
    "repeater_memory_dephasing_probe:r1:pi/4",
    "repeater_memory_dephasing_probe:r1:pi/2",
)
REPEATER_ANCILLA_COUPLING_THETAS = ("0", "pi/6", "pi/4", "pi/2")
ALLOWED_THETA_VALUES = {
    "0": 0.0,
    "pi/6": math.pi / 6,
    "pi/4": math.pi / 4,
    "pi/2": math.pi / 2,
}
FIDELITY_DIAGNOSTIC_ROLE = (
    "delivered_memory_request_metadata_not_post_attack_endpoint_state_fidelity"
)
ENDPOINT_QUALITY_PRIMARY_METRIC = "qber"


@dataclass(frozen=True)
class StructuredRepeaterAttack:
    attack_id: str
    family: str
    target_repeater: str = ""
    basis: str = ""
    theta_label: str = ""
    theta: float | None = None

    @property
    def is_memory_measurement(self) -> bool:
        return self.family == "repeater_memory_measurement"

    @property
    def is_dephasing_probe(self) -> bool:
        return self.family == "repeater_memory_dephasing_probe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "family": self.family,
            "target_repeater": self.target_repeater,
            "basis": self.basis,
            "theta_label": self.theta_label,
            "theta": self.theta,
        }


def repeater_memory_measurement_hook_status() -> dict[str, Any]:
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "blocked": False,
        "capability_code": IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK,
        "hook_exists": True,
        "target_memory_side": "left",
        "measurement_mode": "selective_projective_measurement",
        "required_runtime_hook": (
            "pre-swap repeater memory qstate_key measurement hook before "
            "EntanglementSwappingA consumes the memories"
        ),
        "current_runtime_observation": "selected-route runtime registers this hook through RSVP swapping rules",
        "private_eve_record_public": False,
        "information_gain_reward_enabled": False,
    }


def repeater_memory_dephasing_probe_hook_status() -> dict[str, Any]:
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "blocked": False,
        "capability_code": IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK,
        "hook_exists": True,
        "target_memory_side": "left",
        "implementation": "exact_kraus_sampled_reduced_dephasing_channel",
        "required_runtime_hook": "pre-swap repeater memory qstate circuit or density-matrix state update",
        "current_runtime_observation": (
            "selected-route probe uses a Kraus-sampled phase-flip representation "
            "of the reduced dephasing channel in the ket-vector runtime"
        ),
        "private_eve_record_public": False,
        "information_gain_reward_enabled": False,
    }


def repeater_ancilla_coupling_hook_status() -> dict[str, Any]:
    return {
        "scope_label": BLOCKED,
        "blocked": True,
        "capability_code": BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
        "hook_exists": False,
        "required_runtime_hook": (
            "pre-swap repeater memory plus Eve ancilla two-qubit circuit hook"
        ),
        "current_runtime_observation": (
            "selected-path runtime does not expose an ancilla allocation and "
            "controlled-unitary insertion point before swapping"
        ),
        "information_gain_reward_enabled": False,
    }


def repeater_memory_attack_hook_report() -> dict[str, Any]:
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "claim_status": IMPLEMENTED_DIAGNOSTIC_ONLY,
        "measurement": repeater_memory_measurement_hook_status(),
        "dephasing_probe": repeater_memory_dephasing_probe_hook_status(),
        "ancilla_coupling": repeater_ancilla_coupling_hook_status(),
        "information_gain_reward_enabled": False,
    }


def parse_structured_repeater_attack(attack_id: str) -> StructuredRepeaterAttack:
    if attack_id in {"none", "no_attack"}:
        return StructuredRepeaterAttack(attack_id="no_attack", family="no_attack")
    if attack_id.startswith("repeater_memory_measure_"):
        head, _, target = attack_id.partition(":")
        if not target:
            # Backward-compatible parser for older underscore variants.
            parts = attack_id.rsplit("_", 1)
            if len(parts) == 2:
                head, target = parts
        basis = head.removeprefix("repeater_memory_measure_")
        if basis not in {"Z", "X"} or not target:
            raise ValueError(f"invalid repeater memory measurement attack {attack_id!r}")
        return StructuredRepeaterAttack(
            attack_id=f"repeater_memory_measure_{basis}:{target}",
            family="repeater_memory_measurement",
            target_repeater=target,
            basis=basis,
        )
    if attack_id.startswith("repeater_memory_dephasing_probe:"):
        parts = attack_id.split(":")
        if len(parts) != 3 or parts[2] not in ALLOWED_THETA_VALUES:
            raise ValueError(f"invalid repeater memory dephasing probe {attack_id!r}")
        return StructuredRepeaterAttack(
            attack_id=attack_id,
            family="repeater_memory_dephasing_probe",
            target_repeater=parts[1],
            theta_label=parts[2],
            theta=ALLOWED_THETA_VALUES[parts[2]],
        )
    if attack_id.startswith("repeater_ancilla_coupling:"):
        parts = attack_id.split(":")
        if len(parts) != 3 or parts[2] not in ALLOWED_THETA_VALUES:
            raise ValueError(f"invalid repeater ancilla coupling attack {attack_id!r}")
        return StructuredRepeaterAttack(
            attack_id=attack_id,
            family="repeater_ancilla_coupling",
            target_repeater=parts[1],
            theta_label=parts[2],
            theta=ALLOWED_THETA_VALUES[parts[2]],
        )
    raise ValueError(f"unsupported structured repeater attack {attack_id!r}")


def is_repeater_memory_attack_id(attack_id: str) -> bool:
    return (
        attack_id.startswith("repeater_memory_measure_")
        or attack_id.startswith("repeater_memory_dephasing_probe:")
        or attack_id.startswith("repeater_ancilla_coupling:")
    )


def structured_attack_runtime_blocker(
        attack: StructuredRepeaterAttack,
        *,
        selected_path: tuple[str, ...] = ()) -> dict[str, Any]:
    active = not selected_path or attack.target_repeater in set(selected_path[1:-1])
    if not active:
        payload = {
            "scope_label": REPEATER_RUNTIME,
            "blocked": False,
            "attack_id": attack.attack_id,
            "active_route_attacked": False,
            "attack_applied": False,
            "public_outcome": "inactive_target_no_effect",
            "information_gain_reward_enabled": False,
        }
        assert_no_private_public_fields(payload)
        return payload
    if attack.family in {"repeater_memory_measurement", "repeater_memory_dephasing_probe"}:
        capability = (
            IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK
            if attack.family == "repeater_memory_measurement"
            else IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK
        )
        payload = {
            "scope_label": REPEATER_RUNTIME,
            "blocked": False,
            "attack_id": attack.attack_id,
            "capability_code": capability,
            "active_route_attacked": True,
            "attack_applied": True,
            "runtime_execution_required": True,
            "target_memory_side": "left",
            "information_gain_reward_enabled": False,
        }
        assert_no_private_public_fields(payload)
        return payload
    if attack.family == "repeater_ancilla_coupling":
        return _blocked_runtime_attack(
            attack,
            capability_code=BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
            modeled_capability_code=BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
            reason="pre-swap memory plus Eve ancilla two-qubit circuit hook is absent",
        )
    raise ValueError(f"unsupported structured attack family {attack.family!r}")


def repeater_memory_measurement_blocker(attack_id: str) -> dict[str, Any]:
    if attack_id not in REPEATER_MEMORY_MEASUREMENT_ATTACK_IDS:
        raise ValueError(f"unsupported repeater memory measurement attack {attack_id!r}")
    attack = parse_structured_repeater_attack(attack_id)
    return structured_attack_runtime_blocker(
        attack,
        selected_path=("alice", attack.target_repeater, "bob"),
    )


def repeater_ancilla_coupling_attack_ids() -> tuple[str, ...]:
    return tuple(
        f"repeater_ancilla_coupling:{route}:{theta}"
        for route in ("r0", "r1")
        for theta in REPEATER_ANCILLA_COUPLING_THETAS
    )


def repeater_ancilla_coupling_blocker(attack_id: str) -> dict[str, Any]:
    if attack_id not in repeater_ancilla_coupling_attack_ids():
        raise ValueError(f"unsupported repeater ancilla attack {attack_id!r}")
    return blocked_attack_response(attack_id)


def run_repeater_memory_attack_reference_cases() -> dict[str, Any]:
    """Run supported selected-path cases and emit exact blockers for unsupported hooks."""

    config = build_two_route_runtime_config()
    cases: list[dict[str, Any]] = [
        _public_reference_record(
            run_two_route_runtime_case(config, TwoRouteRuntimeCase(
                "no_attack", 0, None, "no_attack", 3101)),
            capability_code=IMPLEMENTED_NO_ATTACK,
        ),
        _public_reference_record(
            run_two_route_runtime_case(config, TwoRouteRuntimeCase(
                "swap_denial", 0, 0, "swap_denial", 3102)),
            capability_code=IMPLEMENTED_SWAP_DENIAL_ATTACK,
        ),
        _public_reference_record(
            run_two_route_runtime_case(config, TwoRouteRuntimeCase(
                "memory_degradation", 0, 0, "memory_degradation", 3103)),
            capability_code=IMPLEMENTED_MEMORY_DEGRADATION_ATTACK,
        ),
    ]
    active_path = tuple(config.routes[0].nodes)
    inactive_path = tuple(config.routes[1].nodes)
    for attack_id in (
        "repeater_memory_measure_Z:r0",
        "repeater_memory_measure_X:r0",
        "repeater_memory_dephasing_probe:r0:0",
        "repeater_memory_dephasing_probe:r0:pi/6",
        "repeater_memory_dephasing_probe:r0:pi/2",
    ):
        cases.append(_run_structured_reference_case(
            config,
            attack_id,
            seed=321,
            case_id=f"{attack_id}_active",
        ))
    for attack_id in ("repeater_memory_measure_Z:r0", "repeater_memory_measure_X:r0"):
        attack = parse_structured_repeater_attack(attack_id)
        case = structured_attack_runtime_blocker(attack, selected_path=inactive_path)
        case["case_id"] = f"{attack_id}_inactive"
        cases.append(case)
    cases.extend([
        _unsupported_reference_case(
            "endpoint_qnd_resend",
            BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
            "endpoint attack transcript with Eve private record is absent",
        ),
        _unsupported_reference_case(
            "detector_time_shift",
            BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
            "eta_j(delta_t) detector-efficiency mismatch profile is required",
        ),
        _unsupported_reference_case(
            "detector_blinding",
            BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
            "classical detector-control model is required",
        ),
    ])
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "claim_status": IMPLEMENTED_DIAGNOSTIC_ONLY,
        "pre_swap_memory_hook_exists": True,
        "reduced_dephasing_channel_used": True,
        "explicit_ancilla_hook_exists": False,
        "target_memory_side": "left",
        "selective_measurement_used": True,
        "fidelity_diagnostic_role": FIDELITY_DIAGNOSTIC_ROLE,
        "cases": cases,
        "information_gain_reward_enabled": False,
    }


def _run_structured_reference_case(
        config,
        attack_id: str,
        *,
        seed: int,
        case_id: str) -> dict[str, Any]:
    from ..protocol.repeater_trial import run_fixed_repeater_e91_trial

    selected_path = config.routes[0]
    structured = parse_structured_repeater_attack(attack_id)
    if structured.family == "repeater_memory_measurement":
        runtime_attack = RepeaterAttackSpec(
            kind="repeater_memory_measurement",
            target_node=structured.target_repeater,
            basis=structured.basis,
            attack_id=structured.attack_id,
        )
        capability = IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK
    elif structured.family == "repeater_memory_dephasing_probe":
        runtime_attack = RepeaterAttackSpec(
            kind="repeater_memory_dephasing_probe",
            target_node=structured.target_repeater,
            theta=structured.theta,
            theta_label=structured.theta_label,
            attack_id=structured.attack_id,
        )
        capability = IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK
    else:
        raise ValueError(f"unsupported reference attack {attack_id!r}")

    transcript, built = run_fixed_repeater_e91_trial(
        selected_path,
        memory_model=config.memory_model,
        fiber_model=config.fiber_model,
        run_config=config.run_config,
        trial_id=case_id,
        seed=seed,
        attack=runtime_attack,
    )
    record = {
        "case_id": case_id,
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "attack_id": structured.attack_id,
        "capability_code": capability,
        "selected_route": "route_r0",
        "selected_path": list(selected_path.nodes),
        "runtime_attack_applied": runtime_attack.to_dict(),
        "active_route_attacked": True,
        "pre_swap_memory_hook_used": True,
        "target_memory_side": "left",
        "memory_intervention_public": {
            "records": len(built.memory_intervention_records),
            "target_memory_side": "left",
            "selective_measurement_used": structured.family == "repeater_memory_measurement",
            "reduced_dephasing_channel_used": structured.family == "repeater_memory_dephasing_probe",
            "dephasing_probe_implementation": (
                "exact_kraus_sampled_reduced_dephasing_channel"
                if structured.family == "repeater_memory_dephasing_probe"
                else "none"
            ),
            "private_record_public": False,
        },
        "delivery_success": transcript.generation_successes > 0,
        "delivered_count": transcript.generation_successes,
        "qber": transcript.qber_estimate,
        "fidelity": transcript.extra.get("mean_actual_fidelity"),
        "fidelity_diagnostic_role": FIDELITY_DIAGNOSTIC_ROLE,
        "endpoint_quality_primary_metric": ENDPOINT_QUALITY_PRIMARY_METRIC,
        "amer_style_d_eff_mean": transcript.extra.get("amer_style_d_eff_mean"),
        "public_outcome": _public_outcome(transcript),
        "abort_reason": transcript.abort_reason,
        "reward_scope": REWARD_SCOPE,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(record)
    return record


def _public_outcome(transcript) -> str:
    if transcript.accepted:
        return "accepted"
    if transcript.abort_reason == "qber_above_threshold":
        return "qber_abort"
    return "delivery_failure"


def write_repeater_memory_attack_reference(path: Path) -> dict[str, Any]:
    payload = run_repeater_memory_attack_reference_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return payload


def _blocked_runtime_attack(
        attack: StructuredRepeaterAttack,
        *,
        capability_code: str,
        modeled_capability_code: str,
        reason: str) -> dict[str, Any]:
    payload = {
        "scope_label": BLOCKED,
        "claim_status": BLOCKED_PENDING_MAPPING,
        "blocked": True,
        "attack_id": attack.attack_id,
        "capability_code": capability_code,
        "modeled_capability_code": modeled_capability_code,
        "blocked_reason": reason,
        "active_route_attacked": True,
        "private_eve_record_possible": True,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def _unsupported_reference_case(attack_id: str, capability_code: str,
                                reason: str) -> dict[str, Any]:
    payload = {
        "scope_label": BLOCKED,
        "claim_status": BLOCKED_PENDING_MAPPING,
        "blocked": True,
        "case_id": attack_id,
        "attack_id": attack_id,
        "capability_code": capability_code,
        "blocked_reason": reason,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def _public_reference_record(record: dict[str, Any], *,
                             capability_code: str) -> dict[str, Any]:
    copied = dict(record)
    copied["capability_code"] = capability_code
    if "fidelity" in copied:
        copied["fidelity_diagnostic_role"] = FIDELITY_DIAGNOSTIC_ROLE
    applied = copied.get("runtime_attack_applied")
    if isinstance(applied, dict) and applied.get("attack_id") == "":
        copied["runtime_attack_applied"] = dict(applied)
        copied["runtime_attack_applied"].pop("attack_id")
    copied.pop("eve_information_metric", None)
    assert_no_private_public_fields(copied)
    return copied


__all__ = [
    "ALLOWED_THETA_VALUES",
    "ENDPOINT_QUALITY_PRIMARY_METRIC",
    "FIDELITY_DIAGNOSTIC_ROLE",
    "REPEATER_ANCILLA_COUPLING_THETAS",
    "REPEATER_MEMORY_DEPHASING_PROBE_ATTACK_IDS",
    "REPEATER_MEMORY_MEASUREMENT_ATTACK_IDS",
    "StructuredRepeaterAttack",
    "is_repeater_memory_attack_id",
    "parse_structured_repeater_attack",
    "repeater_ancilla_coupling_attack_ids",
    "repeater_ancilla_coupling_blocker",
    "repeater_ancilla_coupling_hook_status",
    "repeater_memory_dephasing_probe_hook_status",
    "repeater_memory_attack_hook_report",
    "repeater_memory_measurement_blocker",
    "repeater_memory_measurement_hook_status",
    "run_repeater_memory_attack_reference_cases",
    "structured_attack_runtime_blocker",
    "write_repeater_memory_attack_reference",
]
