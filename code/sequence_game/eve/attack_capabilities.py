"""Source-scoped Eve attack capability registry.

The registry separates implemented runtime controls from source-motivated
attack surfaces that still lack the runtime hook or transcript needed for live
execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..claims import (
    BLOCKED,
    HARDWARE_DIAGNOSTIC,
    IMPLEMENTED_AND_TESTABLE,
    IMPLEMENTED_DIAGNOSTIC_ONLY,
    assert_no_private_public_fields,
)

IMPLEMENTED_NO_ATTACK = "IMPLEMENTED_NO_ATTACK"
IMPLEMENTED_SWAP_DENIAL_ATTACK = "IMPLEMENTED_SWAP_DENIAL_ATTACK"
IMPLEMENTED_MEMORY_DEGRADATION_ATTACK = "IMPLEMENTED_MEMORY_DEGRADATION_ATTACK"
BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT = (
    "BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT"
)
READY_REPEATER_MEMORY_MEASUREMENT_ATTACK = "READY_REPEATER_MEMORY_MEASUREMENT_ATTACK"
IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK = "IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK"
IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK = (
    "IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK"
)
BLOCKED_REPEATER_MEMORY_MEASUREMENT_REQUIRES_PRE_SWAP_MEMORY_HOOK = (
    "BLOCKED_REPEATER_MEMORY_MEASUREMENT_REQUIRES_PRE_SWAP_MEMORY_HOOK"
)
BLOCKED_REPEATER_MEMORY_PROBE_REQUIRES_STATE_UPDATE_HOOK = (
    "BLOCKED_REPEATER_MEMORY_PROBE_REQUIRES_STATE_UPDATE_HOOK"
)
BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK = (
    "BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK"
)
IMPLEMENTED_REPEATER_ANCILLA_COUPLING_ATTACK = "IMPLEMENTED_REPEATER_ANCILLA_COUPLING_ATTACK"
BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE = (
    "BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE"
)
BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL = (
    "BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL"
)
BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC = (
    "BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC"
)

ALLOWED_CAPABILITY_CODES = frozenset({
    IMPLEMENTED_NO_ATTACK,
    IMPLEMENTED_SWAP_DENIAL_ATTACK,
    IMPLEMENTED_MEMORY_DEGRADATION_ATTACK,
    BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
    READY_REPEATER_MEMORY_MEASUREMENT_ATTACK,
    IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK,
    IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK,
    BLOCKED_REPEATER_MEMORY_MEASUREMENT_REQUIRES_PRE_SWAP_MEMORY_HOOK,
    BLOCKED_REPEATER_MEMORY_PROBE_REQUIRES_STATE_UPDATE_HOOK,
    BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
    IMPLEMENTED_REPEATER_ANCILLA_COUPLING_ATTACK,
    BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
    BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
    BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC,
})
ALLOWED_STATUSES = frozenset({"implemented", "implemented_if_hook_available", "blocked"})
IMPLEMENTED_RUNTIME_ATTACKS = frozenset({
    "no_attack",
    "swap_denial",
    "memory_degradation",
    "repeater_memory_measure_Z",
    "repeater_memory_measure_X",
    "repeater_memory_dephasing_probe_theta",
})


class AttackCapabilityError(ValueError):
    """Attack capability was requested outside its supported runtime scope."""


@dataclass(frozen=True)
class AttackCapability:
    attack_id: str
    status: str
    capability_code: str
    scope_label: str
    primary_reference: str
    source_role: str
    modeled_operation: str
    required_runtime_hook: str
    allowed_targets: tuple[str, ...]
    public_effect: str
    private_eve_record_possible: bool
    reward_allowed: tuple[str, ...]
    reward_blocked: tuple[str, ...]
    not_claimed: str
    runtime_blocker_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise AttackCapabilityError(f"unsupported status {self.status!r}")
        if self.capability_code not in ALLOWED_CAPABILITY_CODES:
            raise AttackCapabilityError(
                f"unsupported capability_code {self.capability_code!r}")
        if self.status == "blocked" and self.scope_label != BLOCKED:
            raise AttackCapabilityError("blocked capabilities must use BLOCKED scope")
        if self.runtime_blocker_code is not None and self.runtime_blocker_code not in ALLOWED_CAPABILITY_CODES:
            raise AttackCapabilityError(
                f"unsupported runtime_blocker_code {self.runtime_blocker_code!r}")
        if "information_gain" not in self.reward_blocked:
            raise AttackCapabilityError("information_gain must stay reward-blocked")
        if "secret_key" not in self.reward_blocked:
            raise AttackCapabilityError("secret_key reward must stay blocked")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "status": self.status,
            "capability_code": self.capability_code,
            "scope_label": self.scope_label,
            "primary_reference": self.primary_reference,
            "source_role": self.source_role,
            "modeled_operation": self.modeled_operation,
            "required_runtime_hook": self.required_runtime_hook,
            "allowed_targets": list(self.allowed_targets),
            "public_effect": self.public_effect,
            "private_eve_record_possible": self.private_eve_record_possible,
            "reward_allowed": list(self.reward_allowed),
            "reward_blocked": list(self.reward_blocked),
            "not_claimed": self.not_claimed,
            "runtime_blocker_code": self.runtime_blocker_code,
        }


def capability_registry() -> list[AttackCapability]:
    blocked_rewards = ("information_gain", "secret_key")
    return [
        AttackCapability(
            attack_id="no_attack",
            status="implemented",
            capability_code=IMPLEMENTED_NO_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="baseline",
            source_role="control baseline with no Eve intervention",
            modeled_operation="identity channel / no runtime corruption",
            required_runtime_hook="none",
            allowed_targets=("none",),
            public_effect="none",
            private_eve_record_possible=False,
            reward_allowed=("none",),
            reward_blocked=blocked_rewards,
            not_claimed="not an attack or security result",
        ),
        AttackCapability(
            attack_id="swap_denial",
            status="implemented",
            capability_code=IMPLEMENTED_SWAP_DENIAL_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang",
            source_role="attack-surface motivation: corrupted repeater infrastructure",
            modeled_operation="force EntanglementSwappingA success probability to zero at targeted active repeater",
            required_runtime_hook="EntanglementSwappingA success probability control",
            allowed_targets=("selected-route repeater",),
            public_effect="delivery failure / abort",
            private_eve_record_possible=False,
            reward_allowed=("availability",),
            reward_blocked=blocked_rewards,
            not_claimed="does not model Eve key knowledge",
        ),
        AttackCapability(
            attack_id="memory_degradation",
            status="implemented",
            capability_code=IMPLEMENTED_MEMORY_DEGRADATION_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang",
            source_role="attack-surface motivation: corrupted repeater infrastructure",
            modeled_operation="lower target repeater MemoryArray initial fidelity template",
            required_runtime_hook="MemoryArray fidelity template control",
            allowed_targets=("selected-route repeater",),
            public_effect="quality/fidelity diagnostic change where measurable",
            private_eve_record_possible=False,
            reward_allowed=("quality",),
            reward_blocked=blocked_rewards,
            not_claimed="not a confidentiality or information-gain attack",
        ),
        AttackCapability(
            attack_id="repeater_memory_measure_Z",
            status="implemented",
            capability_code=IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang",
            source_role="formal model specified in this implementation; source papers motivate attack surface only",
            modeled_operation="selective or averaged Z-basis projective measurement on one pre-swap repeater memory",
            required_runtime_hook="pre-swap repeater memory qstate_key measurement hook",
            allowed_targets=("selected-route repeater memory",),
            public_effect="selected-route runtime applies selective measurement before swapping",
            private_eve_record_possible=True,
            reward_allowed=("availability", "quality"),
            reward_blocked=blocked_rewards,
            not_claimed="not a formula copied from Harkness/Satoh/Naik and not an information-gain reward",
        ),
        AttackCapability(
            attack_id="repeater_memory_measure_X",
            status="implemented",
            capability_code=IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang",
            source_role="formal model specified in this implementation; source papers motivate attack surface only",
            modeled_operation="selective or averaged X-basis projective measurement on one pre-swap repeater memory",
            required_runtime_hook="pre-swap repeater memory qstate_key measurement hook",
            allowed_targets=("selected-route repeater memory",),
            public_effect="selected-route runtime applies selective measurement before swapping",
            private_eve_record_possible=True,
            reward_allowed=("availability", "quality"),
            reward_blocked=blocked_rewards,
            not_claimed="not a formula copied from Harkness/Satoh/Naik and not an information-gain reward",
        ),
        AttackCapability(
            attack_id="repeater_memory_dephasing_probe_theta",
            status="implemented",
            capability_code=IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK,
            scope_label=HARDWARE_DIAGNOSTIC,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang; Naik et al. motivation only",
            source_role="formal model specified in this implementation; sources motivate attack surface/eavesdropping only",
            modeled_operation="controlled-Ry ancilla model reduced to exact Z-dephasing channel with parameter theta",
            required_runtime_hook="pre-swap repeater memory qstate_key density-matrix update hook",
            allowed_targets=("selected-route repeater memory",),
            public_effect="selected-route runtime applies exact reduced Z-dephasing channel before swapping",
            private_eve_record_possible=True,
            reward_allowed=("availability", "quality"),
            reward_blocked=blocked_rewards,
            not_claimed="P_guess diagnostic is not a reward and the unitary is our specified model",
        ),
        AttackCapability(
            attack_id="repeater_ancilla_coupling_theta",
            status="blocked",
            capability_code=BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK,
            scope_label=BLOCKED,
            primary_reference="Satoh et al.; Harkness/Krawec/Wang",
            source_role="attack-surface motivation; explicit ancilla circuit hook absent",
            modeled_operation="explicit Eve ancilla allocation plus controlled memory-ancilla unitary",
            required_runtime_hook="pre-swap memory plus Eve ancilla two-qubit circuit hook",
            allowed_targets=("selected-route repeater memory",),
            public_effect="blocked; no surrogate memory degradation emitted",
            private_eve_record_possible=True,
            reward_allowed=("availability", "quality"),
            reward_blocked=blocked_rewards,
            not_claimed="do not emulate explicit ancilla coupling by lowering memory fidelity",
        ),
        AttackCapability(
            attack_id="endpoint_qnd_resend",
            status="blocked",
            capability_code=BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT,
            scope_label=BLOCKED,
            primary_reference="Naik et al.",
            source_role="protocol/eavesdropping motivation only",
            modeled_operation="endpoint subsystem dephasing/intercept-resend formal channel with attack fraction",
            required_runtime_hook="endpoint attack transcript with Eve private record and public basis/sifting",
            allowed_targets=("Alice/Bob endpoint subsystem",),
            public_effect="blocked; endpoint attack transcript absent",
            private_eve_record_possible=True,
            reward_allowed=("quality",),
            reward_blocked=blocked_rewards,
            not_claimed="does not validate repeater placement or runtime hook",
        ),
        AttackCapability(
            attack_id="detector_time_shift",
            status="blocked",
            capability_code=BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE,
            scope_label=BLOCKED,
            primary_reference="Qi/Fung/Lo/Ma; Zhao/Fung/Qi/Chen/Lo; Korzh et al. timing reference only",
            source_role="detector-side-channel reference; Korzh timing alone is not a POVM model",
            modeled_operation="validate E_j(delta_t)=eta_j(delta_t)E_j and positive no-click operator",
            required_runtime_hook="detector efficiency eta_j(delta_t) mismatch profile",
            allowed_targets=("endpoint detector", "BSM heralding detector"),
            public_effect="blocked; no detector behavior changed",
            private_eve_record_possible=False,
            reward_allowed=("quality",),
            reward_blocked=blocked_rewards,
            not_claimed="Korzh timing/FWHM does not unlock time shift by itself",
        ),
        AttackCapability(
            attack_id="detector_blinding",
            status="blocked",
            capability_code=BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL,
            scope_label=BLOCKED,
            primary_reference="Lydersen et al.",
            source_role="detector-side-channel reference",
            modeled_operation="blocked classical detector-control model requirements only",
            required_runtime_hook="classical blinded-mode detector control model",
            allowed_targets=("endpoint detector", "BSM heralding detector"),
            public_effect="blocked; no detector behavior changed",
            private_eve_record_possible=False,
            reward_allowed=("availability", "quality"),
            reward_blocked=blocked_rewards,
            not_claimed="does not unlock runtime blinding without classical detector-control model",
        ),
        AttackCapability(
            attack_id="information_gain_reward",
            status="blocked",
            capability_code=BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC,
            scope_label=BLOCKED,
            primary_reference="future metric layer; Harkness/Krawec/Wang context remains uncoupled",
            source_role="future key-transcript metric blocker",
            modeled_operation="document P_guess(K|E,P) and I(K;E|P) formulas without reward use",
            required_runtime_hook="raw key K, Eve private record E, public transcript P, approved metric",
            allowed_targets=("reward layer",),
            public_effect="blocked; no reward contribution",
            private_eve_record_possible=True,
            reward_allowed=("none",),
            reward_blocked=blocked_rewards,
            not_claimed="no information-gain, key-guessing, or security claim",
        ),
    ]


VARIANT_PARENT = {
    "repeater_memory_measure_Z:r0": "repeater_memory_measure_Z",
    "repeater_memory_measure_X:r0": "repeater_memory_measure_X",
    "repeater_memory_measure_Z:r1": "repeater_memory_measure_Z",
    "repeater_memory_measure_X:r1": "repeater_memory_measure_X",
    "repeater_memory_measure_Z_r0": "repeater_memory_measure_Z",
    "repeater_memory_measure_X_r0": "repeater_memory_measure_X",
    "repeater_memory_measure_Z_r1": "repeater_memory_measure_Z",
    "repeater_memory_measure_X_r1": "repeater_memory_measure_X",
    "repeater_memory_dephasing_probe:r0:0": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r0:pi/6": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r0:pi/4": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r0:pi/2": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r1:0": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r1:pi/6": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r1:pi/4": "repeater_memory_dephasing_probe_theta",
    "repeater_memory_dephasing_probe:r1:pi/2": "repeater_memory_dephasing_probe_theta",
    "repeater_ancilla_coupling:r0:0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r0:pi/6": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r0:pi/4": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r0:pi/2": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r1:0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r1:pi/6": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r1:pi/4": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling:r1:pi/2": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_0_r0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/6_r0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/4_r0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/2_r0": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_0_r1": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/6_r1": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/4_r1": "repeater_ancilla_coupling_theta",
    "repeater_ancilla_coupling_theta_pi/2_r1": "repeater_ancilla_coupling_theta",
    "endpoint_qnd_resend_Z": "endpoint_qnd_resend",
    "endpoint_qnd_resend_X": "endpoint_qnd_resend",
    "endpoint_intercept_resend_random_basis": "endpoint_qnd_resend",
    "detector_time_shift_endpoint": "detector_time_shift",
    "detector_time_shift_bsm_heralding": "detector_time_shift",
    "detector_time_shift_swap_stage": "detector_time_shift",
    "endpoint_detector_blinding": "detector_blinding",
    "bsm_detector_blinding": "detector_blinding",
}


def capability_by_attack_id() -> dict[str, AttackCapability]:
    return {row.attack_id: row for row in capability_registry()}


def normalize_attack_id(attack_id: str) -> str:
    if attack_id in {"swap_denial_r0", "swap_denial_r1"}:
        return "swap_denial"
    if attack_id in {"memory_degradation_r0", "memory_degradation_r1"}:
        return "memory_degradation"
    if attack_id.startswith("repeater_memory_measure_Z:"):
        return "repeater_memory_measure_Z"
    if attack_id.startswith("repeater_memory_measure_X:"):
        return "repeater_memory_measure_X"
    if attack_id.startswith("repeater_memory_dephasing_probe:"):
        return "repeater_memory_dephasing_probe_theta"
    if attack_id.startswith("repeater_ancilla_coupling:"):
        return "repeater_ancilla_coupling_theta"
    return VARIANT_PARENT.get(attack_id, attack_id)


def capability_for_attack(attack_id: str) -> AttackCapability:
    normalized = normalize_attack_id(attack_id)
    rows = capability_by_attack_id()
    if normalized not in rows:
        raise AttackCapabilityError(f"unknown attack_id {attack_id!r}")
    return rows[normalized]


def is_live_runtime_selectable(attack_id: str) -> bool:
    return capability_for_attack(attack_id).status == "implemented"


def require_live_runtime_selectable(attack_id: str) -> AttackCapability:
    row = capability_for_attack(attack_id)
    if row.status != "implemented":
        raise AttackCapabilityError(
            f"{attack_id} is not selectable for live runtime: {row.capability_code}")
    return row


def blocked_attack_response(attack_id: str) -> dict[str, Any]:
    row = capability_for_attack(attack_id)
    if row.status == "implemented":
        raise AttackCapabilityError(f"{attack_id!r} is not blocked")
    capability_code = row.runtime_blocker_code or row.capability_code
    response = {
        "scope_label": BLOCKED,
        "blocked": True,
        "attack_id": attack_id,
        "capability_code": capability_code,
        "blocked_reason": row.required_runtime_hook,
        "public_effect": row.public_effect,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(response)
    return response


def attack_capability_report() -> dict[str, Any]:
    rows = [row.to_dict() for row in capability_registry()]
    variants = [
        {
            "attack_id": variant,
            "parent_attack_id": parent,
            "capability_code": capability_for_attack(variant).capability_code,
            "runtime_blocker_code": capability_for_attack(variant).runtime_blocker_code,
            "scope_label": capability_for_attack(variant).scope_label,
            "selectable_for_live_runtime": is_live_runtime_selectable(variant),
        }
        for variant, parent in sorted(VARIANT_PARENT.items())
    ]
    report = {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "claim_status": IMPLEMENTED_DIAGNOSTIC_ONLY,
        "rows": rows,
        "variants": variants,
        "blocked_attack_count": sum(1 for row in rows if row["status"] == "blocked"),
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(report)
    return report


def attack_vector_blockers() -> dict[str, Any]:
    blocked = [
        blocked_attack_response(row.attack_id)
        for row in capability_registry()
        if row.status != "implemented"
    ]
    variant_blockers = [
        blocked_attack_response(variant)
        for variant in sorted(VARIANT_PARENT)
        if capability_for_attack(variant).status != "implemented"
    ]
    payload = {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "blocked_attacks": blocked,
        "blocked_variants": variant_blockers,
        "information_gain_reward_enabled": False,
    }
    assert_no_private_public_fields(payload)
    return payload


def write_attack_capability_report(path: Path) -> dict[str, Any]:
    payload = attack_capability_report()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return payload


def write_attack_vector_blockers(path: Path) -> dict[str, Any]:
    payload = attack_vector_blockers()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return payload


def validate_capability_report(report: Mapping[str, Any]) -> None:
    for row in report["rows"]:
        if row["capability_code"] not in ALLOWED_CAPABILITY_CODES:
            raise AttackCapabilityError(f"bad capability code {row['capability_code']!r}")
        if row["status"] == "blocked" and row["scope_label"] != BLOCKED:
            raise AttackCapabilityError("blocked row must use BLOCKED scope")


__all__ = [
    "ALLOWED_CAPABILITY_CODES",
    "ALLOWED_STATUSES",
    "AttackCapability",
    "AttackCapabilityError",
    "BLOCKED_ANCILLA_ATTACK_REQUIRES_MEMORY_CIRCUIT_HOOK",
    "BLOCKED_DETECTOR_BLINDING_REQUIRES_CLASSICAL_DETECTOR_CONTROL_MODEL",
    "BLOCKED_ENDPOINT_QND_REQUIRES_ENDPOINT_ATTACK_TRANSCRIPT",
    "BLOCKED_INFORMATION_GAIN_REQUIRES_KEY_TRANSCRIPT_METRIC",
    "BLOCKED_REPEATER_MEMORY_MEASUREMENT_REQUIRES_PRE_SWAP_MEMORY_HOOK",
    "BLOCKED_REPEATER_MEMORY_PROBE_REQUIRES_STATE_UPDATE_HOOK",
    "BLOCKED_TIME_SHIFT_REQUIRES_DETECTOR_EFFICIENCY_MISMATCH_PROFILE",
    "IMPLEMENTED_MEMORY_DEGRADATION_ATTACK",
    "IMPLEMENTED_NO_ATTACK",
    "IMPLEMENTED_REPEATER_ANCILLA_COUPLING_ATTACK",
    "IMPLEMENTED_REPEATER_MEMORY_DEPHASING_PROBE_ATTACK",
    "IMPLEMENTED_REPEATER_MEMORY_MEASUREMENT_ATTACK",
    "IMPLEMENTED_SWAP_DENIAL_ATTACK",
    "READY_REPEATER_MEMORY_MEASUREMENT_ATTACK",
    "VARIANT_PARENT",
    "attack_capability_report",
    "attack_vector_blockers",
    "blocked_attack_response",
    "capability_for_attack",
    "capability_registry",
    "is_live_runtime_selectable",
    "normalize_attack_id",
    "require_live_runtime_selectable",
    "validate_capability_report",
    "write_attack_capability_report",
    "write_attack_vector_blockers",
]
