#!/usr/bin/env python3
"""Smoke run for the direct E91 transcript backend."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "sequence_matplotlib"))

from sequence_game.protocol.direct_e91_hardware_adapter import (  # noqa: E402
    direct_e91_route_from_topology_ir,
    load_direct_e91_hardware_config_from_existing,
)
from sequence_game.protocol.direct_e91_transcript import (  # noqa: E402
    DirectE91AttackSpec,
    DirectE91ProtocolConfig,
    DirectE91StepInput,
    run_direct_e91_transcript_step,
)


def _diamond_topology() -> dict:
    return {
        "nodes": {
            "alice": {"node_id": "alice", "roles": ["alice"]},
            "r0": {"node_id": "r0", "roles": ["swap_candidate"]},
            "r1": {"node_id": "r1", "roles": ["swap_candidate"]},
            "bob": {"node_id": "bob", "roles": ["bob"]},
            "off_path": {"node_id": "off_path", "roles": ["swap_candidate"]},
        },
        "edges": [
            {"edge_id": "e_alice_r0", "u": "alice", "v": "r0", "length_m": 100.0},
            {"edge_id": "e_r0_bob", "u": "r0", "v": "bob", "length_m": 100.0},
            {"edge_id": "e_alice_r1", "u": "alice", "v": "r1", "length_m": 120.0},
            {"edge_id": "e_r1_bob", "u": "r1", "v": "bob", "length_m": 120.0},
            {"edge_id": "e_off_path", "u": "r0", "v": "off_path", "length_m": 50.0},
        ],
        "metadata": {"generator": "manual_direct_e91_smoke"},
    }


def _summary_row(result) -> dict:
    return {
        "attack_id": result.attack_id,
        "attack_kind": result.attack_kind,
        "target_active": result.target_active,
        "public_outcome": result.public_outcome,
        "delivery_probability": round(result.delivery_probability, 6),
        "delivered_pair_count": result.delivered_pair_count,
        "key_pair_count": result.key_pair_count,
        "chsh_pair_count": result.chsh_pair_count,
        "discarded_setting_pair_count": result.discarded_setting_pair_count,
        "qber_key": result.qber_key,
        "chsh_S_expected": result.chsh_S_expected,
        "chsh_S_observed": result.chsh_S_observed,
        "information_exposed": result.information_exposed,
        "secure_delivery": result.secure_delivery,
        "eve_observation_count": result.eve_observation_count,
        "eve_guess_probability": result.eve_guess_probability,
        "eve_guess_model": result.eve_guess_model,
        "effective_end_to_end_fidelity_model": result.effective_end_to_end_fidelity_model,
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
    }


def main() -> None:
    topology = _diamond_topology()
    route0 = direct_e91_route_from_topology_ir(
        topology,
        ["alice", "r0", "bob"],
        route_id="diamond_r0",
    )
    route1 = direct_e91_route_from_topology_ir(
        topology,
        ["alice", "r1", "bob"],
        route_id="diamond_r1",
    )
    hardware = load_direct_e91_hardware_config_from_existing()
    protocol = DirectE91ProtocolConfig(
        requested_pairs=2048,
        qber_abort_threshold=0.50,
        chsh_abort_threshold=0.0,
        min_key_pairs_for_qber_abort=8,
        min_chsh_pairs_for_chsh_abort=8,
    )
    cases = [
        ("none", route0, DirectE91AttackSpec()),
        ("edge_added_loss_active", route0, DirectE91AttackSpec(
            attack_id="edge_added_loss_active",
            attack_kind="edge_added_loss",
            target_kind="edge",
            target_id="e_alice_r0",
            loss_multiplier=0.1,
        )),
        ("edge_added_loss_inactive", route0, DirectE91AttackSpec(
            attack_id="edge_added_loss_inactive",
            attack_kind="edge_added_loss",
            target_kind="edge",
            target_id="e_alice_r1",
            loss_multiplier=0.1,
        )),
        ("swap_denial_active", route0, DirectE91AttackSpec(
            attack_id="swap_denial_active",
            attack_kind="swap_denial",
            target_kind="node",
            target_id="r0",
        )),
        ("swap_denial_inactive", route0, DirectE91AttackSpec(
            attack_id="swap_denial_inactive",
            attack_kind="swap_denial",
            target_kind="node",
            target_id="r1",
        )),
        ("memory_degradation_active", route0, DirectE91AttackSpec(
            attack_id="memory_degradation_active",
            attack_kind="memory_degradation",
            target_kind="node",
            target_id="r0",
            memory_depolarizing_probability=0.25,
            memory_dephasing_probability=0.10,
        )),
        ("memory_degradation_inactive", route0, DirectE91AttackSpec(
            attack_id="memory_degradation_inactive",
            attack_kind="memory_degradation",
            target_kind="node",
            target_id="r1",
            memory_depolarizing_probability=0.25,
            memory_dephasing_probability=0.10,
        )),
        ("edge_intercept_resend_active", route0, DirectE91AttackSpec(
            attack_id="edge_intercept_resend_active",
            attack_kind="edge_intercept_resend",
            target_kind="edge",
            target_id="e_alice_r0",
            intercept_basis_label="Z",
            intercept_probability=1.0,
        )),
        ("edge_intercept_resend_inactive", route0, DirectE91AttackSpec(
            attack_id="edge_intercept_resend_inactive",
            attack_kind="edge_intercept_resend",
            target_kind="edge",
            target_id="e_alice_r1",
            intercept_basis_label="Z",
            intercept_probability=1.0,
        )),
        ("repeater_memory_measure_Z_active", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_measure_Z_active",
            attack_kind="repeater_memory_measure_Z",
            target_kind="memory",
            target_id="r0",
        )),
        ("repeater_memory_measure_Z_inactive", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_measure_Z_inactive",
            attack_kind="repeater_memory_measure_Z",
            target_kind="memory",
            target_id="r1",
        )),
        ("repeater_memory_measure_X_active", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_measure_X_active",
            attack_kind="repeater_memory_measure_X",
            target_kind="memory",
            target_id="r0",
        )),
        ("repeater_memory_measure_X_inactive", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_measure_X_inactive",
            attack_kind="repeater_memory_measure_X",
            target_kind="memory",
            target_id="r1",
        )),
        ("repeater_memory_dephasing_probe_theta_0", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_dephasing_probe_theta_0",
            attack_kind="repeater_memory_dephasing_probe",
            target_kind="memory",
            target_id="r0",
            dephasing_theta=0.0,
        )),
        ("repeater_memory_dephasing_probe_theta_pi_2", route0, DirectE91AttackSpec(
            attack_id="repeater_memory_dephasing_probe_theta_pi_2",
            attack_kind="repeater_memory_dephasing_probe",
            target_kind="memory",
            target_id="r0",
            dephasing_theta=math.pi / 2.0,
        )),
        ("none_second_route", route1, DirectE91AttackSpec(attack_id="none_second_route")),
    ]
    rows = []
    full_results = []
    for index, (_label, route, attack) in enumerate(cases):
        result = run_direct_e91_transcript_step(
            DirectE91StepInput(
                topology=topology,
                alice_node_id="alice",
                bob_node_id="bob",
                selected_route=route,
                eve_attack=attack,
                hardware=hardware,
                protocol=protocol,
                seed=20260629,
                step_index=index,
            )
        )
        rows.append(_summary_row(result))
        full_results.append(result.to_dict())

    payload = {
        "backend": "direct_e91_transcript",
        "case_count": len(rows),
        "compact_results": rows,
        "results": full_results,
    }
    out_dir = ROOT / "results" / "direct_e91_transcript_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_path": str(out_path), "compact_results": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
