#!/usr/bin/env python3
"""Tiny numerical sanity table for the direct E91 transcript backend."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "sequence_matplotlib"),
)

from sequence_game.protocol.direct_e91_hardware_adapter import (  # noqa: E402
    direct_e91_route_from_topology_ir,
)
from sequence_game.protocol.direct_e91_transcript import (  # noqa: E402
    DirectE91AttackSpec,
    DirectE91HardwareConfig,
    DirectE91HardwareProvenance,
    DirectE91ProtocolConfig,
    DirectE91StepInput,
    run_direct_e91_transcript_step,
)


def _topology() -> dict[str, Any]:
    return {
        "nodes": {
            "alice": {"node_id": "alice", "roles": ["alice"]},
            "r0": {"node_id": "r0", "roles": ["swap_candidate"]},
            "bob": {"node_id": "bob", "roles": ["bob"]},
            "off_path": {"node_id": "off_path", "roles": ["swap_candidate"]},
        },
        "edges": [
            {"edge_id": "e_alice_r0", "u": "alice", "v": "r0", "length_m": 10.0},
            {"edge_id": "e_r0_bob", "u": "r0", "v": "bob", "length_m": 10.0},
            {"edge_id": "e_off_path", "u": "r0", "v": "off_path", "length_m": 10.0},
        ],
        "metadata": {"generator": "direct_e91_sanity"},
    }


def _hardware(**overrides: Any) -> DirectE91HardwareConfig:
    values = {
        "fiber_loss_db_per_km": 0.0,
        "source_pair_success_probability": 1.0,
        "detector_efficiency": 1.0,
        "dark_count_probability": 0.0,
        "misalignment_probability": 0.0,
        "elementary_pair_fidelity": 1.0,
        "swap_success_probability": 1.0,
        "swap_depolarizing_probability": 0.0,
        "memory_depolarizing_probability": 0.0,
        "memory_dephasing_probability": 0.0,
        "provenance": DirectE91HardwareProvenance(hardware_profile_id="sanity"),
    }
    values.update(overrides)
    return DirectE91HardwareConfig(**values)


def _protocol(
    requested_pairs: int,
    *,
    qber_abort_threshold: float = 1.0,
    chsh_abort_threshold: float = 0.0,
) -> DirectE91ProtocolConfig:
    return DirectE91ProtocolConfig(
        requested_pairs=requested_pairs,
        qber_abort_threshold=qber_abort_threshold,
        chsh_abort_threshold=chsh_abort_threshold,
        min_key_pairs_for_qber_abort=1,
        min_chsh_pairs_for_chsh_abort=1,
    )


def _run(
    *,
    requested_pairs: int,
    hardware: DirectE91HardwareConfig,
    attack: DirectE91AttackSpec | None = None,
    seed: int = 20260629,
    step_index: int = 0,
    qber_abort_threshold: float = 1.0,
    chsh_abort_threshold: float = 0.0,
):
    topology = _topology()
    route = direct_e91_route_from_topology_ir(
        topology,
        ["alice", "r0", "bob"],
        route_id="sanity_route",
    )
    return run_direct_e91_transcript_step(
        DirectE91StepInput(
            topology=topology,
            alice_node_id="alice",
            bob_node_id="bob",
            selected_route=route,
            eve_attack=attack or DirectE91AttackSpec(),
            hardware=hardware,
            protocol=_protocol(
                requested_pairs,
                qber_abort_threshold=qber_abort_threshold,
                chsh_abort_threshold=chsh_abort_threshold,
            ),
            seed=seed,
            step_index=step_index,
        )
    )


def _row(label: str, result, *, reference=None, note: str = "") -> dict[str, Any]:
    row = {
        "case": label,
        "requested": result.requested_pair_count,
        "delivered": result.delivered_pair_count,
        "key": result.key_pair_count,
        "chsh": result.chsh_pair_count,
        "discarded": result.discarded_setting_pair_count,
        "qber": result.qber_key,
        "chsh_expected": result.chsh_S_expected,
        "chsh_observed": result.chsh_S_observed,
        "public_outcome": result.public_outcome,
        "information_exposed": result.information_exposed,
        "target_active": result.target_active,
        "actual_final_bell_fidelity": result.actual_final_bell_fidelity,
        "link_fidelity_product_diagnostic": result.link_fidelity_product_diagnostic,
        "source_pair_success_application": result.source_pair_success_application,
        "route_state_model": result.route_state_model,
        "elementary_link_count": result.elementary_link_count,
        "bsm_swap_count": result.bsm_swap_count,
        "bsm_outcome_probabilities": result.bsm_outcome_probabilities,
        "pauli_frame_corrections_applied": result.pauli_frame_corrections_applied,
        "pauli_frame_correction_model": result.pauli_frame_correction_model,
        "target_memory_side_effective": result.target_memory_side_effective,
        "edge_attack_subsystem": result.edge_attack_subsystem,
        "edge_attack_edge_id": result.edge_attack_edge_id,
        "eve_guess_model": result.eve_guess_model,
        "note": note,
    }
    if reference is not None:
        row["same_as_reference"] = {
            "delivery_probability": result.delivery_probability == reference.delivery_probability,
            "qber": result.qber_key == reference.qber_key,
            "chsh_expected": result.chsh_S_expected == reference.chsh_S_expected,
            "chsh_observed": result.chsh_S_observed == reference.chsh_S_observed,
            "finite_key": (
                result.finite_key_estimate_bits
                == reference.finite_key_estimate_bits
            ),
        }
    return row


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "case",
        "requested",
        "delivered",
        "key",
        "chsh",
        "discarded",
        "qber",
        "chsh_expected",
        "chsh_observed",
        "public_outcome",
        "information_exposed",
        "target_active",
        "actual_final_bell_fidelity",
        "link_fidelity_product_diagnostic",
        "elementary_link_count",
        "bsm_swap_count",
        "target_memory_side_effective",
        "edge_attack_edge_id",
        "eve_guess_model",
        "note",
    ]
    widths = {
        column: max(len(column), *(len(_fmt(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(_fmt(row.get(column, "")).ljust(widths[column]) for column in columns))


def main() -> None:
    ideal_256 = _run(requested_pairs=256, hardware=_hardware(), step_index=0)
    ideal_4096 = _run(requested_pairs=4096, hardware=_hardware(), step_index=1)
    ideal_65536 = _run(requested_pairs=65_536, hardware=_hardware(), step_index=2)
    nonideal = _run(
        requested_pairs=65_536,
        hardware=_hardware(elementary_pair_fidelity=0.95),
        step_index=3,
    )
    dark_low = _run(
        requested_pairs=65_536,
        hardware=_hardware(detector_efficiency=0.5, dark_count_probability=0.0),
        step_index=4,
    )
    dark_high = _run(
        requested_pairs=65_536,
        hardware=_hardware(detector_efficiency=0.5, dark_count_probability=1.0),
        step_index=4,
    )
    baseline = _run(requested_pairs=4096, hardware=_hardware(), step_index=5)
    inactive_edge = _run(
        requested_pairs=4096,
        hardware=_hardware(),
        attack=DirectE91AttackSpec(
            attack_id="inactive_edge_added_loss",
            attack_kind="edge_added_loss",
            target_kind="edge",
            target_id="e_off_path",
            loss_multiplier=0.01,
        ),
        step_index=5,
    )
    inactive_node = _run(
        requested_pairs=4096,
        hardware=_hardware(),
        attack=DirectE91AttackSpec(
            attack_id="inactive_memory_measure",
            attack_kind="repeater_memory_measure_Z",
            target_kind="memory",
            target_id="off_path",
        ),
        step_index=5,
    )
    active_info = _run(
        requested_pairs=4096,
        hardware=_hardware(),
        attack=DirectE91AttackSpec(
            attack_id="active_edge_intercept_resend",
            attack_kind="edge_intercept_resend",
            target_kind="edge",
            target_id="e_alice_r0",
            intercept_basis_label="Z",
            intercept_probability=1.0,
        ),
        step_index=6,
        qber_abort_threshold=1.0,
        chsh_abort_threshold=0.0,
    )

    rows = [
        _row("ideal_256", ideal_256, note="QBER near 0; finite CHSH sample"),
        _row("ideal_4096", ideal_4096, note="observed CHSH closer"),
        _row("ideal_65536", ideal_65536, note="observed CHSH approaches expected"),
        _row("nonideal_fidelity", nonideal, note="QBER > 0; CHSH lower"),
        _row("dark_counts_low", dark_low, note="dark count baseline"),
        _row("dark_counts_high", dark_high, note="QBER up; observed CHSH down"),
        _row("inactive_edge", inactive_edge, reference=baseline, note="same seed/config as baseline"),
        _row("inactive_node", inactive_node, reference=baseline, note="same seed/config as baseline"),
        _row("active_information", active_info, note="accepted with private information"),
    ]
    checks = {
        "ideal_qber_near_zero": ideal_65536.qber_key is not None and ideal_65536.qber_key < 0.01,
        "ideal_expected_chsh_near_tsirelson": abs(ideal_65536.chsh_S_expected - 2.8284271247461903) < 1e-9,
        "ideal_observed_chsh_converges": (
            abs(ideal_65536.chsh_S_observed - ideal_65536.chsh_S_expected)
            < abs(ideal_256.chsh_S_observed - ideal_256.chsh_S_expected)
        ),
        "nonideal_qber_positive": nonideal.qber_key is not None and nonideal.qber_key > 0.0,
        "nonideal_chsh_lower": nonideal.chsh_S_expected < ideal_65536.chsh_S_expected,
        "dark_counts_qber_increases": dark_high.qber_key > dark_low.qber_key,
        "dark_counts_chsh_decreases": dark_high.chsh_S_observed < dark_low.chsh_S_observed,
        "inactive_edge_identical": all(_row("", inactive_edge, reference=baseline)["same_as_reference"].values()),
        "inactive_node_identical": all(_row("", inactive_node, reference=baseline)["same_as_reference"].values()),
        "active_information_accepted": active_info.public_outcome == "accepted",
        "active_information_exposed": active_info.information_exposed is True,
    }

    payload = {
        "backend": "direct_e91_transcript",
        "rows": rows,
        "checks": checks,
    }
    out_dir = ROOT / "results" / "direct_e91_transcript_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sanity_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print_table(rows)
    print()
    print(json.dumps({"summary_path": str(out_path), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
