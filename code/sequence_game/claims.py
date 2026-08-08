"""Claim and scope guardrails for graph-game outputs.

These helpers are intentionally small and conservative.  They do not define
physics or security quantities; they only enforce that emitted records carry one
of the approved scope labels and fail closed when a requested claim needs a
source, public transcript mapping, or graph-runtime interface that is not
present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CONTROL_GAME = "CONTROL_GAME"
REPEATER_RUNTIME = "REPEATER_RUNTIME"
HARDWARE_DIAGNOSTIC = "HARDWARE_DIAGNOSTIC"
SECURITY_REFERENCE = "SECURITY_REFERENCE"
GRAPH_OBJECTIVE_CANDIDATE = "GRAPH_OBJECTIVE_CANDIDATE"
BLOCKED = "BLOCKED"

IMPLEMENTED_AND_TESTABLE = "IMPLEMENTED_AND_TESTABLE"
IMPLEMENTED_DIAGNOSTIC_ONLY = "IMPLEMENTED_DIAGNOSTIC_ONLY"
IMPLEMENTED_CONTROL_ONLY = "IMPLEMENTED_CONTROL_ONLY"
DESIGNED_NOT_IMPLEMENTED = "DESIGNED_NOT_IMPLEMENTED"
BLOCKED_PENDING_SOURCE = "BLOCKED_PENDING_SOURCE"
BLOCKED_PENDING_MAPPING = "BLOCKED_PENDING_MAPPING"
BLOCKED_PENDING_PUBLIC_TRANSCRIPT = "BLOCKED_PENDING_PUBLIC_TRANSCRIPT"
BLOCKED_PENDING_SECURITY_QUANTITY = "BLOCKED_PENDING_SECURITY_QUANTITY"
BLOCKED_PENDING_GRAPH_INTERFACE = "BLOCKED_PENDING_GRAPH_INTERFACE"

ALLOWED_SCOPE_LABELS = frozenset({
    CONTROL_GAME,
    REPEATER_RUNTIME,
    HARDWARE_DIAGNOSTIC,
    SECURITY_REFERENCE,
    GRAPH_OBJECTIVE_CANDIDATE,
    BLOCKED,
})

ALLOWED_CLAIM_STATUS_LABELS = frozenset({
    IMPLEMENTED_AND_TESTABLE,
    IMPLEMENTED_DIAGNOSTIC_ONLY,
    IMPLEMENTED_CONTROL_ONLY,
    DESIGNED_NOT_IMPLEMENTED,
    BLOCKED_PENDING_SOURCE,
    BLOCKED_PENDING_MAPPING,
    BLOCKED_PENDING_PUBLIC_TRANSCRIPT,
    BLOCKED_PENDING_SECURITY_QUANTITY,
    BLOCKED_PENDING_GRAPH_INTERFACE,
})

PRIVATE_PUBLIC_KEYS = frozenset({
    "alice_bases",
    "bob_bases",
    "alice_outcomes",
    "bob_outcomes",
    "basis_choices",
    "endpoint_outcomes",
    "private_debug",
    "private_key",
    "raw_key",
    "raw_key_bits",
    "sifted_key",
    "sifted_indices",
})

EVE_PRIVATE_OBSERVATION_KEYS = PRIVATE_PUBLIC_KEYS | frozenset({
    "alice_path",
    "alice_private_ranking",
    "alice_route",
    "alice_route_id",
    "current_route",
    "route_path",
})

SECURITY_REFERENCE_TERMS = frozenset({
    "harkness",
    "finite-key",
    "finite key",
    "composable",
    "security proof",
})


class ClaimGuardError(ValueError):
    """A result or claim crossed an explicit scope boundary."""


@dataclass(frozen=True)
class ScopedResult:
    """Minimal labelled result envelope for fail-closed call sites."""

    scope_label: str
    claim_status: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_scope_label(self.scope_label)
        validate_claim_status(self.claim_status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_label": self.scope_label,
            "claim_status": self.claim_status,
            **dict(self.payload),
        }


def validate_scope_label(scope_label: str) -> str:
    if scope_label not in ALLOWED_SCOPE_LABELS:
        raise ClaimGuardError(f"unsupported scope label {scope_label!r}")
    return scope_label


def validate_claim_status(claim_status: str) -> str:
    if claim_status not in ALLOWED_CLAIM_STATUS_LABELS:
        raise ClaimGuardError(f"unsupported claim status {claim_status!r}")
    return claim_status


def require_scope_label(result: Mapping[str, Any]) -> str:
    """Return a valid scope label or fail if a result is unlabeled."""

    scope_label = result.get("scope_label")
    if scope_label is None and isinstance(result.get("metrics"), Mapping):
        scope_label = result["metrics"].get("scope_label")
    if not isinstance(scope_label, str):
        raise ClaimGuardError("result is missing scope_label")
    return validate_scope_label(scope_label)


def blocked_result(blocked_reason: str,
                   *,
                   claim_status: str = BLOCKED_PENDING_SOURCE,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a blocked response without synthetic runtime metrics."""

    validate_claim_status(claim_status)
    payload = {
        "blocked": True,
        "blocked_reason": blocked_reason,
        "scope_label": BLOCKED,
        "claim_status": claim_status,
    }
    if extra:
        payload.update(dict(extra))
    if "metrics" in payload:
        raise ClaimGuardError("blocked_result must not include synthetic metrics")
    return payload


def validate_blocked_response(result: Mapping[str, Any]) -> None:
    require_scope_label(result)
    if result.get("blocked") is not True:
        raise ClaimGuardError("unsupported physics response must set blocked=True")
    if result.get("scope_label") != BLOCKED:
        raise ClaimGuardError("blocked physics response must use BLOCKED scope")
    if result.get("metrics"):
        raise ClaimGuardError("blocked physics response must not emit metrics")


def validate_information_gain_reward(weight: float,
                                     *,
                                     metric_source: str | None = None,
                                     public_transcript_mapping: str | None = None) -> None:
    """Reject nonzero information-gain reward without source and mapping."""

    if weight == 0:
        return
    if metric_source and public_transcript_mapping:
        return
    raise ClaimGuardError(
        "information-gain reward requires a cited metric and public transcript mapping"
    )


def assert_no_private_public_fields(payload: Any,
                                    *,
                                    forbidden_keys: frozenset[str] = PRIVATE_PUBLIC_KEYS) -> None:
    """Recursively reject private keys in a public payload."""

    if isinstance(payload, Mapping):
        leaked = forbidden_keys & set(payload)
        if leaked:
            raise ClaimGuardError(f"private fields leaked in public payload: {sorted(leaked)}")
        for value in payload.values():
            assert_no_private_public_fields(value, forbidden_keys=forbidden_keys)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_private_public_fields(value, forbidden_keys=forbidden_keys)


def assert_public_eve_observation(payload: Any, *, oracle_control: bool = False) -> None:
    """Reject Eve observations that carry Alice-private route or key material."""

    forbidden = PRIVATE_PUBLIC_KEYS if oracle_control else EVE_PRIVATE_OBSERVATION_KEYS
    assert_no_private_public_fields(payload, forbidden_keys=forbidden)


def assert_no_security_reference_claim(payload: Any) -> None:
    """Prevent security-reference language from diagnostic/runtime payloads."""

    if isinstance(payload, str):
        lowered = payload.lower()
        found = [term for term in SECURITY_REFERENCE_TERMS if term in lowered]
        if found:
            raise ClaimGuardError(
                f"security-reference claim requires mapping tests: {sorted(found)}"
            )
    elif isinstance(payload, Mapping):
        for value in payload.values():
            assert_no_security_reference_claim(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_security_reference_claim(value)


def require_graph_objective_test(result: Mapping[str, Any]) -> None:
    """Block graph-objective claims until the bound/objective test is named."""

    require_scope_label(result)
    if result.get("scope_label") != GRAPH_OBJECTIVE_CANDIDATE:
        return
    if not result.get("graph_objective_test_id"):
        raise ClaimGuardError(
            "graph objective candidate claim requires graph_objective_test_id"
        )
