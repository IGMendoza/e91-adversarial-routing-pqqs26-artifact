"""Runtime-facing source-scoped Eve attack-surface report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..claims import HARDWARE_DIAGNOSTIC, IMPLEMENTED_DIAGNOSTIC_ONLY
from ..eve.attack_capabilities import (
    attack_capability_report,
    attack_vector_blockers,
    blocked_attack_response,
    is_live_runtime_selectable,
    require_live_runtime_selectable,
    write_attack_capability_report,
    write_attack_vector_blockers,
)
from .repeater_memory_attacks import repeater_memory_attack_hook_report


def build_attack_surface_report() -> dict[str, Any]:
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "claim_status": IMPLEMENTED_DIAGNOSTIC_ONLY,
        "capability_report": attack_capability_report(),
        "memory_hook_report": repeater_memory_attack_hook_report(),
        "information_gain_reward_enabled": False,
    }


def assert_attack_selectable_for_live_runtime(attack_id: str) -> dict[str, Any]:
    row = require_live_runtime_selectable(attack_id)
    return row.to_dict()


def runtime_blocker_for_attack(attack_id: str) -> dict[str, Any]:
    if is_live_runtime_selectable(attack_id):
        raise ValueError(f"{attack_id!r} is already selectable for live runtime")
    return blocked_attack_response(attack_id)


def write_attack_surface_artifacts(report_path: Path,
                                   blockers_path: Path) -> dict[str, Any]:
    report = write_attack_capability_report(report_path)
    blockers = write_attack_vector_blockers(blockers_path)
    return {
        "scope_label": HARDWARE_DIAGNOSTIC,
        "capability_report": report,
        "blockers": blockers,
        "information_gain_reward_enabled": False,
    }


__all__ = [
    "assert_attack_selectable_for_live_runtime",
    "build_attack_surface_report",
    "runtime_blocker_for_attack",
    "write_attack_surface_artifacts",
]
