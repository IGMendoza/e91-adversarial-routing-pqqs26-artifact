#!/usr/bin/env python
"""Check whether public BSM labels affect Eve's exact key posterior."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from dataclasses import replace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.protocol.direct_e91_transcript import (
    DirectE91AttackSpec,
    DirectE91HardwareConfig,
    DirectE91ProtocolConfig,
    DirectE91Route,
    _apply_dark_count_outcome_model,
    _apply_measurement_noise,
    _first_key_pair,
    joint_outcome_probabilities,
    route_density_matrix_from_elementary_swaps,
)


def make_route(
    hops: int,
    *,
    length_m: float = 10_000.0,
    fidelity_mode: str = "uniform",
    base_fidelity: float = 0.96,
) -> DirectE91Route:
    """Create A--R1--...--B with `hops` edges."""
    if hops < 1:
        raise ValueError("hops must be >= 1")

    if hops == 1:
        nodes = ("A", "B")
    else:
        nodes = ("A",) + tuple(f"R{i}" for i in range(1, hops)) + ("B",)

    edges = tuple(f"e{i}" for i in range(hops))
    lengths = {edge: float(length_m) for edge in edges}

    if fidelity_mode == "uniform":
        fidelities = {edge: float(base_fidelity) for edge in edges}
    elif fidelity_mode == "graded":
        # Deliberately asymmetric route, to test whether BSM labels become relevant.
        fidelities = {
            edge: max(0.50, min(1.0, base_fidelity - 0.015 * i))
            for i, edge in enumerate(edges)
        }
    elif fidelity_mode == "alternating":
        fidelities = {
            edge: (base_fidelity if i % 2 == 0 else max(0.50, base_fidelity - 0.08))
            for i, edge in enumerate(edges)
        }
    else:
        raise ValueError(f"unknown fidelity_mode {fidelity_mode!r}")

    return DirectE91Route(
        route_id=f"route_h{hops}_{fidelity_mode}",
        node_path=nodes,
        edge_ids=edges,
        edge_lengths_m=lengths,
        total_length_m=sum(lengths.values()),
        edge_fidelities=fidelities,
    )


def hardware_regimes() -> dict[str, DirectE91HardwareConfig]:
    return {
        "ideal": DirectE91HardwareConfig(
            fiber_loss_db_per_km=0.0,
            source_pair_success_probability=1.0,
            detector_efficiency=1.0,
            dark_count_probability=0.0,
            misalignment_probability=0.0,
            elementary_pair_fidelity=1.0,
            swap_success_probability=1.0,
            swap_depolarizing_probability=0.0,
            memory_depolarizing_probability=0.0,
            memory_dephasing_probability=0.0,
        ),
        "bell_diagonal_noisy": DirectE91HardwareConfig(
            fiber_loss_db_per_km=0.2,
            source_pair_success_probability=0.92,
            detector_efficiency=0.88,
            dark_count_probability=1e-5,
            misalignment_probability=0.01,
            elementary_pair_fidelity=0.96,
            swap_success_probability=0.91,
            swap_depolarizing_probability=0.02,
            memory_depolarizing_probability=0.01,
            memory_dephasing_probability=0.006,
        ),
        "asymmetric_noisy": DirectE91HardwareConfig(
            fiber_loss_db_per_km=0.35,
            source_pair_success_probability=0.85,
            detector_efficiency=0.80,
            dark_count_probability=5e-5,
            misalignment_probability=0.025,
            elementary_pair_fidelity=0.93,
            swap_success_probability=0.85,
            swap_depolarizing_probability=0.04,
            memory_depolarizing_probability=0.025,
            memory_dephasing_probability=0.015,
        ),
    }


def projective_attacks(route: DirectE91Route) -> list[DirectE91AttackSpec]:
    attacks: list[DirectE91AttackSpec] = []

    # Edge intercept/resend on each elementary link, in Z and X.
    for edge_id in route.edge_ids:
        for basis in ("Z", "X"):
            attacks.append(
                DirectE91AttackSpec(
                    attack_id=f"edge_intercept_{basis}_{edge_id}",
                    attack_kind="edge_intercept_resend",
                    target_kind="edge",
                    target_id=edge_id,
                    intercept_basis_label=basis,
                    intercept_probability=1.0,
                    information_probability=1.0,
                )
            )

    # Repeater memory measurements on each internal node.
    for node_id in route.internal_nodes:
        for kind in ("repeater_memory_measure_Z", "repeater_memory_measure_X"):
            for side in ("left", "right", "both"):
                attacks.append(
                    DirectE91AttackSpec(
                        attack_id=f"{kind}_{node_id}_{side}",
                        attack_kind=kind,
                        target_kind="node",
                        target_id=node_id,
                        measurement_probability=1.0,
                        information_probability=1.0,
                        target_memory_side=side,
                    )
                )

    return attacks


def tuple_labels(branch: dict[str, Any], key_tuple: str, key_joined: str) -> tuple[str, ...]:
    labels = branch.get(key_tuple)
    if labels is not None:
        return tuple(str(x) for x in labels)

    joined = str(branch.get(key_joined, "none"))
    if joined in {"", "none"}:
        return ()
    return tuple(part for part in joined.split("|") if part and part != "none")


def posterior_guess(
    branch_model: dict[str, Any],
    protocol: DirectE91ProtocolConfig,
    hardware: DirectE91HardwareConfig,
    *,
    condition_on_public_bsm: bool,
    include_dark_counts: bool,
) -> float:
    """Compute sum_z max_b P(Alice raw key bit=b, Z=z)."""
    key_pair = _first_key_pair(protocol)
    if key_pair is None:
        raise RuntimeError("protocol has no key setting pair")
    alice_setting, bob_setting = key_pair

    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], list[float]] = {}

    for branch in branch_model["branches"]:
        p_branch = float(branch["probability"])
        if p_branch <= 0.0:
            continue

        eve_labels = tuple_labels(
            branch,
            "eve_observation_labels",
            "eve_observation_label",
        )
        public_bsm_labels = tuple_labels(
            branch,
            "public_bsm_labels",
            "public_bsm_label",
        )
        condition = (
            eve_labels,
            public_bsm_labels if condition_on_public_bsm else (),
        )

        if condition not in grouped:
            grouped[condition] = [0.0, 0.0]

        # Mirror the backend posterior path: endpoint misalignment/noise first.
        branch_state = _apply_measurement_noise(branch["state"], hardware)

        probs = joint_outcome_probabilities(
            branch_state,
            alice_setting.axis,
            bob_setting.axis,
        )

        # Optional stricter audit: include endpoint dark-count randomization in key bit.
        # The current backend Eve posterior historically did not include this, so default off.
        if include_dark_counts:
            probs = _apply_dark_count_outcome_model(probs, hardware)

        grouped[condition][0] += p_branch * float(probs[0, :].sum())
        grouped[condition][1] += p_branch * float(probs[1, :].sum())

    total = sum(sum(v) for v in grouped.values())
    if total <= 0.0:
        return 0.5
    return max(0.0, min(1.0, sum(max(v) for v in grouped.values()) / total))


def run_case(
    *,
    hops: int,
    fidelity_mode: str,
    hardware_name: str,
    hardware: DirectE91HardwareConfig,
    attack: DirectE91AttackSpec,
    protocol: DirectE91ProtocolConfig,
    include_dark_counts: bool,
) -> dict[str, Any] | None:
    route = make_route(
        hops,
        fidelity_mode=fidelity_mode,
        base_fidelity=hardware.elementary_pair_fidelity,
    )
    # Rebuild attack against the newly created route if needed.
    # Caller should pass attacks created from this route, so this is a sanity check.
    target_active = (
        attack.target_id in set(route.edge_ids)
        or attack.target_id in set(route.internal_nodes)
    )
    if not target_active:
        return None

    state = route_density_matrix_from_elementary_swaps(
        route,
        hardware,
        protocol,
        attack,
        target_active=True,
    )
    branch_model = state.branch_model
    if branch_model is None:
        return None

    full = posterior_guess(
        branch_model,
        protocol,
        hardware,
        condition_on_public_bsm=True,
        include_dark_counts=include_dark_counts,
    )
    eve_only = posterior_guess(
        branch_model,
        protocol,
        hardware,
        condition_on_public_bsm=False,
        include_dark_counts=include_dark_counts,
    )

    branches = branch_model.get("branches", [])
    public_bsm_records = {
        str(branch.get("public_bsm_label", "none"))
        for branch in branches
    }
    eve_records = {
        str(branch.get("eve_observation_label", "none"))
        for branch in branches
    }

    return {
        "hops": hops,
        "repeaters": max(0, hops - 1),
        "fidelity_mode": fidelity_mode,
        "hardware": hardware_name,
        "attack_id": attack.attack_id,
        "attack_kind": attack.attack_kind,
        "target_kind": attack.target_kind,
        "target_id": attack.target_id,
        "target_memory_side": getattr(attack, "target_memory_side", ""),
        "full_public_bsm_guess": full,
        "eve_only_guess": eve_only,
        "delta": full - eve_only,
        "branch_count": len(branches),
        "public_bsm_record_count": len(public_bsm_records),
        "eve_record_count": len(eve_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--include-dark-counts", action="store_true")
    parser.add_argument("--fail-on-delta", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    protocol = DirectE91ProtocolConfig(requested_pairs=0)
    regimes = hardware_regimes()
    fidelity_modes = ("uniform", "graded", "alternating")

    rows: list[dict[str, Any]] = []
    skipped = 0

    for hops in range(args.min_hops, args.max_hops + 1):
        for fidelity_mode, (hardware_name, hardware) in itertools.product(
            fidelity_modes,
            regimes.items(),
        ):
            route = make_route(
                hops,
                fidelity_mode=fidelity_mode,
                base_fidelity=hardware.elementary_pair_fidelity,
            )
            for attack in projective_attacks(route):
                row = run_case(
                    hops=hops,
                    fidelity_mode=fidelity_mode,
                    hardware_name=hardware_name,
                    hardware=hardware,
                    attack=attack,
                    protocol=protocol,
                    include_dark_counts=args.include_dark_counts,
                )
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)

    rows.sort(key=lambda r: abs(float(r["delta"])), reverse=True)

    max_abs_delta = max((abs(float(r["delta"])) for r in rows), default=0.0)
    max_delta = max((float(r["delta"]) for r in rows), default=0.0)
    min_delta = min((float(r["delta"]) for r in rows), default=0.0)

    print()
    print("=== Public BSM posterior relevance audit ===")
    print(f"cases evaluated:       {len(rows)}")
    print(f"cases skipped:         {skipped}")
    print(f"hops range:            {args.min_hops}..{args.max_hops}")
    print(f"include dark counts:   {args.include_dark_counts}")
    print(f"tolerance:             {args.tol:g}")
    print(f"max |delta|:           {max_abs_delta:.16g}")
    print(f"max delta:             {max_delta:.16g}")
    print(f"min delta:             {min_delta:.16g}")
    print()

    if rows:
        print(f"Top {min(args.top, len(rows))} cases by |delta|:")
        print(
            "delta                 full_guess          eve_only_guess      "
            "hops reps branches bsm_records eve_records hardware fidelity attack"
        )
        for row in rows[: args.top]:
            print(
                f"{row['delta']:+.16e} "
                f"{row['full_public_bsm_guess']:.12f} "
                f"{row['eve_only_guess']:.12f} "
                f"{row['hops']:>4} "
                f"{row['repeaters']:>4} "
                f"{row['branch_count']:>8} "
                f"{row['public_bsm_record_count']:>11} "
                f"{row['eve_record_count']:>10} "
                f"{row['hardware']} "
                f"{row['fidelity_mode']} "
                f"{row['attack_id']}"
            )

    if max_abs_delta <= args.tol:
        print()
        print(
            "RESULT: public BSM conditioning appears posterior-irrelevant "
            "within tolerance for the tested cases."
        )
        return 0

    print()
    print(
        "RESULT: public BSM conditioning changes Eve's posterior in at least "
        "one tested case."
    )
    print(
        "Interpretation: do not remove public-BSM conditioning from exact "
        "posterior mode unless those cases are outside the intended model."
    )

    return 1 if args.fail_on_delta else 0


if __name__ == "__main__":
    raise SystemExit(main())