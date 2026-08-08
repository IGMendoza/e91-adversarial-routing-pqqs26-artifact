"""Elementary-link construction and repeater-route branch composition."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .devices import *
from .fock import *
from .models import *
from .quantum import *
from .timing import *
from .timing import _apply_storage_channel_params_to_density
def route_quantum_branches(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool = True,
    context: ModelContext | None = None,
) -> list[Branch]:
    context = context or ModelContext()
    edge_ids = list(route["edge_ids"])
    use_fock_storage = context.physical_path == "fock_pnr_snspd"
    timing_policy = deterministic_storage_timing_policy(route, front) if use_fock_storage else None

    if use_fock_storage and normalized_waiting_protocol(context) == "rolling_alive_pair":
        return route_quantum_branches_rolling_alive_pair(
            route,
            front,
            attack,
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
            context=context,
        )

    if use_fock_storage and route["hops"] == 2:
        return route_quantum_branches_two_link_waiting(
            route,
            front,
            attack,
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
            context=context,
            timing_policy=timing_policy,
        )
    if (
        use_fock_storage
        and route["hops"] >= 3
        and memory_lifetime_parameters(front)["storage_model_enters_probability"]
    ):
        # With an active storage model the stochastic waits change branch
        # states, so the grouped swap-asap fold is required. Without one,
        # waits have no state effect and the deterministic composer below is
        # branch-exact; waiting still enters delivery mass separately.
        return route_quantum_branches_multi_link_waiting(
            route,
            front,
            attack,
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
            context=context,
            timing_policy=timing_policy,
        )

    def edge_branches_after_deterministic_storage(edge_id: str) -> list[Branch]:
        edge_branches = elementary_edge_branches(edge_id, route, front, attack, context)
        if not use_fock_storage or timing_policy is None:
            return edge_branches
        storage_time = timing_policy["edge_herald_ready_time_ps"][edge_id]
        return apply_memory_storage_channel_to_branches(
            edge_branches,
            front,
            qubits=(0, 1),
            storage_time_ps=storage_time,
        )

    branches = edge_branches_after_deterministic_storage(edge_ids[0])
    left_ready_time = (
        timing_policy["edge_herald_ready_time_ps"][edge_ids[0]]
        if timing_policy is not None else
        0
    )
    for swap_index, repeater in enumerate(route["nodes"][1:-1]):
        right_edge_id = edge_ids[swap_index + 1]
        right = edge_branches_after_deterministic_storage(right_edge_id)
        if use_fock_storage and timing_policy is not None:
            right_ready_time = timing_policy["edge_herald_ready_time_ps"][right_edge_id]
            operation_time = max(left_ready_time, right_ready_time)
            branches = apply_memory_storage_channel_to_branches(
                branches,
                front,
                qubits=(0, 1),
                storage_time_ps=operation_time - left_ready_time,
                elapsed_storage_time_ps=left_ready_time,
            )
            right = apply_memory_storage_channel_to_branches(
                right,
                front,
                qubits=(0, 1),
                storage_time_ps=operation_time - right_ready_time,
                elapsed_storage_time_ps=right_ready_time,
            )
        else:
            operation_time = 0
        combined: list[Branch] = []
        for left_branch in branches:
            for right_branch in right:
                eve = join_labels(left_branch.eve_observation, right_branch.eve_observation)
                combined.append(Branch(
                    probability=left_branch.probability * right_branch.probability,
                    rho=np.kron(left_branch.rho, right_branch.rho),
                    frame=compose_frame(left_branch.frame, right_branch.frame),
                    eve_observation=eve,
                    public_bsm_record=join_labels(left_branch.public_bsm_record, right_branch.public_bsm_record),
                    detector_click_record=join_detector_click_records(
                        left_branch.detector_click_record,
                        right_branch.detector_click_record,
                        keep_detector_click_records=keep_detector_click_records,
                    ),
                ))
        combined = apply_repeater_attack(combined, repeater, attack)
        next_branches: list[Branch] = []
        for branch in combined:
            next_branches.extend(swap_branch_bell_outcomes(branch, repeater, front, context))
        branches = aggregate_branches(
            normalize_branches(next_branches),
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
        )
        if use_fock_storage:
            left_ready_time = operation_time
    return aggregate_branches(
        normalize_branches(branches),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def route_quantum_branches_two_link_waiting(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool = True,
    context: ModelContext,
    timing_policy: dict[str, Any] | None,
) -> list[Branch]:
    edge_ids = list(route["edge_ids"])
    if len(edge_ids) != 2:
        raise ValueError("two-link waiting branch composer requires exactly two elementary links")
    timing_policy = timing_policy or deterministic_storage_timing_policy(route, front)

    def edge_branches_after_deterministic_storage(edge_id: str) -> list[Branch]:
        edge_branches = elementary_edge_branches(edge_id, route, front, attack, context)
        storage_time = timing_policy["edge_herald_ready_time_ps"][edge_id]
        return apply_memory_storage_channel_to_branches(
            edge_branches,
            front,
            qubits=(0, 1),
            storage_time_ps=storage_time,
        )

    left_edge, right_edge = edge_ids
    left_base = edge_branches_after_deterministic_storage(left_edge)
    right_base = edge_branches_after_deterministic_storage(right_edge)
    waiting_policy = stochastic_waiting_time_policy(route, front, attack, context)
    if not waiting_policy.get("supported", False):
        return []
    edge_success = waiting_policy["edge_success_probabilities"]
    attempt_policy = waiting_policy["attempt_period_policy"]["edge_attempt_periods"]
    edge_periods = {
        edge_id: float(attempt_policy[edge_id]["attempt_period_ps"])
        for edge_id in edge_ids
    }
    budgets, _budget_meta = waiting_attempt_budgets(context, edge_periods)
    histories, _distribution_meta = two_link_waiting_histories(
        p0=edge_success[left_edge],
        p1=edge_success[right_edge],
        period0_ps=edge_periods[left_edge],
        period1_ps=edge_periods[right_edge],
        context=context,
        budget0=budgets[left_edge],
        budget1=budgets[right_edge],
    )
    if not histories:
        return []
    left_ready = timing_policy["edge_herald_ready_time_ps"][left_edge]
    right_ready = timing_policy["edge_herald_ready_time_ps"][right_edge]
    wait_profile = endpoint_classical_wait_profile(route, front)
    endpoint_wait_alice, endpoint_wait_bob = endpoint_extra_waits_for_swap_tails(wait_profile, [0.0])

    # Stale-pair cutoff and detected-decay survival reweight the history
    # mixture classically (exact: survival/cutoff are state-independent
    # per-history factors); delivery mass bookkeeping happens in
    # delivery_stage_probabilities from the same distribution.
    weighted_ages = [
        (
            float(row["probability"]),
            two_link_history_memory_ages(
                row,
                herald_left_ps=left_ready,
                herald_right_ps=right_ready,
                endpoint_wait_alice_ps=endpoint_wait_alice,
                endpoint_wait_bob_ps=endpoint_wait_bob,
            ),
        )
        for row in histories
    ]
    usable_weights, _loss_stats = apply_memory_loss_policies_to_histories(weighted_ages, front, context)
    usable_mass = math.fsum(usable_weights)
    if usable_mass <= 0.0:
        return []
    histories = [
        {**row, "probability": weight / usable_mass}
        for row, weight in zip(histories, usable_weights)
        if weight > 0.0
    ]

    def waited_average(
        base: list[Branch],
        waits: list[tuple[float, float]],
        elapsed_storage_time_ps: float,
        extra_durations_ps: tuple[float, float],
    ) -> tuple[list[Branch], float]:
        """Probability-weighted storage-channel average of branches over waits.

        Exact: every downstream instrument step (attack channels, projective
        expansions, Bell projections, endpoint measurement rows) is linear in
        probability * rho, and branch record labels do not depend on the wait,
        so aggregating waiting histories before composition equals evaluating
        each history separately and summing. extra_durations_ps adds the
        endpoint classical-confirmation wait per qubit; the endpoint qubit's
        storage commutes with the swap/attack (disjoint qubits), so folding it
        into one accumulated-time application here is exact.
        """
        mass = math.fsum(prob for _wait_ps, prob in waits)
        if mass <= 0.0 or not base:
            return [], 0.0
        sums = [np.zeros_like(branch.rho) for branch in base]
        for wait_ps, prob in waits:
            weight = prob / mass
            waited_rhos = [np.array(branch.rho, dtype=complex, copy=True) for branch in base]
            for qubit, extra_ps in enumerate(extra_durations_ps):
                duration = wait_ps + extra_ps
                if duration <= 0.0:
                    continue
                params = memory_storage_channel_probabilities(
                    front,
                    duration,
                    elapsed_storage_time_ps=elapsed_storage_time_ps,
                )
                waited_rhos = [
                    _apply_storage_channel_params_to_density(rho, params, (qubit,))
                    for rho in waited_rhos
                ]
            for index, rho in enumerate(waited_rhos):
                sums[index] = sums[index] + weight * rho
        averaged = [
            Branch(
                probability=branch.probability,
                rho=clean_density(rho_sum),
                frame=branch.frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=branch.public_bsm_record,
                detector_click_record=branch.detector_click_record,
            )
            for branch, rho_sum in zip(base, sums)
        ]
        return averaged, mass

    left_extras = (endpoint_wait_alice, 0.0)
    right_extras = (0.0, endpoint_wait_bob)
    # Every generated history has min(left_wait, right_wait) == 0 (the later
    # link finishes exactly at the swap operation time), so the history mixture
    # splits into two one-sided groups plus a defensive general remainder.
    left_waiting = [
        (float(row["left_wait_ps"]), float(row["probability"]))
        for row in histories
        if row["right_wait_ps"] <= 0.0
    ]
    right_waiting = [
        (float(row["right_wait_ps"]), float(row["probability"]))
        for row in histories
        if row["right_wait_ps"] > 0.0 and row["left_wait_ps"] <= 0.0
    ]
    both_waiting = [
        row for row in histories
        if row["right_wait_ps"] > 0.0 and row["left_wait_ps"] > 0.0
    ]
    left_base_ready, _ = waited_average(left_base, [(0.0, 1.0)], left_ready, left_extras)
    right_base_ready, _ = waited_average(right_base, [(0.0, 1.0)], right_ready, right_extras)
    grouped: list[tuple[float, list[Branch], list[Branch]]] = []
    left_avg, left_mass = waited_average(left_base, left_waiting, left_ready, left_extras)
    if left_mass > 0.0:
        grouped.append((left_mass, left_avg, right_base_ready))
    right_avg, right_mass = waited_average(right_base, right_waiting, right_ready, right_extras)
    if right_mass > 0.0:
        grouped.append((right_mass, left_base_ready, right_avg))
    for row in both_waiting:
        left_one, _ = waited_average(left_base, [(float(row["left_wait_ps"]), 1.0)], left_ready, left_extras)
        right_one, _ = waited_average(right_base, [(float(row["right_wait_ps"]), 1.0)], right_ready, right_extras)
        grouped.append((float(row["probability"]), left_one, right_one))

    repeater = route["nodes"][1]
    next_branches: list[Branch] = []
    for group_probability, left_waited, right_waited in grouped:
        combined: list[Branch] = []
        for left_branch in left_waited:
            for right_branch in right_waited:
                eve = join_labels(left_branch.eve_observation, right_branch.eve_observation)
                combined.append(Branch(
                    probability=group_probability * left_branch.probability * right_branch.probability,
                    rho=np.kron(left_branch.rho, right_branch.rho),
                    frame=compose_frame(left_branch.frame, right_branch.frame),
                    eve_observation=eve,
                    public_bsm_record=join_labels(left_branch.public_bsm_record, right_branch.public_bsm_record),
                    detector_click_record=join_detector_click_records(
                        left_branch.detector_click_record,
                        right_branch.detector_click_record,
                        keep_detector_click_records=keep_detector_click_records,
                    ),
                ))
        combined = apply_repeater_attack(combined, repeater, attack)
        for branch in combined:
            next_branches.extend(swap_branch_bell_outcomes(branch, repeater, front, context))
    return aggregate_branches(
        normalize_branches(next_branches),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def _combine_attack_swap(
    left: list[Branch],
    right: list[Branch],
    repeater: str,
    attack: HistogramAttack,
    front: dict[str, Any],
    context: ModelContext,
    *,
    keep_detector_click_records: bool = True,
) -> list[Branch]:
    """One selected-chain fold step: kron segment with edge, apply the targeted
    repeater move, and take the (possibly label-confused) swap Bell outcomes."""
    combined: list[Branch] = []
    for left_branch in left:
        for right_branch in right:
            combined.append(Branch(
                probability=left_branch.probability * right_branch.probability,
                rho=np.kron(left_branch.rho, right_branch.rho),
                frame=compose_frame(left_branch.frame, right_branch.frame),
                eve_observation=join_labels(left_branch.eve_observation, right_branch.eve_observation),
                public_bsm_record=join_labels(left_branch.public_bsm_record, right_branch.public_bsm_record),
                detector_click_record=join_detector_click_records(
                    left_branch.detector_click_record,
                    right_branch.detector_click_record,
                    keep_detector_click_records=keep_detector_click_records,
                ),
            ))
    combined = apply_repeater_attack(combined, repeater, attack)
    out: list[Branch] = []
    for branch in combined:
        out.extend(swap_branch_bell_outcomes(branch, repeater, front, context))
    return out


def _rolling_edge_branches_for_age(
    *,
    edge_index: int,
    edge_id: str,
    hop_count: int,
    age_s: float,
    base: list[Branch],
    front: dict[str, Any],
    keep_full_bsm: bool,
    keep_detector_click_records: bool,
    endpoint_wait_alice_ps: float,
    endpoint_wait_bob_ps: float,
) -> list[Branch]:
    times = rolling_edge_storage_times_ps(
        edge_index=edge_index,
        hop_count=hop_count,
        age_s=age_s,
        endpoint_wait_alice_ps=endpoint_wait_alice_ps,
        endpoint_wait_bob_ps=endpoint_wait_bob_ps,
    )
    aged = apply_memory_storage_channel_to_branches_by_qubit(
        base,
        front,
        storage_time_by_qubit_ps=times,
    )
    return aggregate_branches(
        normalize_branches(aged),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def _fold_rolling_edge_branch_tables(
    *,
    route: dict[str, Any],
    edge_tables: list[list[Branch]],
    attack: HistogramAttack,
    front: dict[str, Any],
    context: ModelContext,
    keep_full_bsm: bool,
    keep_detector_click_records: bool,
) -> list[Branch]:
    branches = edge_tables[0]
    for swap_index, repeater in enumerate(route["nodes"][1:-1], start=1):
        swapped = _combine_attack_swap(
            branches,
            edge_tables[swap_index],
            repeater,
            attack,
            front,
            context,
            keep_detector_click_records=keep_detector_click_records,
        )
        branches = aggregate_branches(
            normalize_branches(swapped),
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
        )
    return aggregate_branches(
        normalize_branches(branches),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def route_quantum_branches_rolling_alive_pair_cartesian(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool,
    context: ModelContext,
) -> list[Branch]:
    """Explicit finite product over the configured per-edge age support."""
    edge_ids = list(route["edge_ids"])
    hops = len(edge_ids)
    policy = rolling_live_pair_policy(route, front, context)
    age_distributions, age_meta = rolling_age_distributions_for_route(route, context)
    support_product = int(age_meta["support_product"])
    limit = int(context.rolling_age_support_product_limit)
    if support_product > limit:
        raise NotImplementedError(
            "rolling_alive_pair cartesian age evaluation support product exceeds configured exact limit: "
            f"{support_product} > {limit}; use rolling_age_evaluation='moment_reduced' only if its "
            "exactness assumptions hold for the active physics"
        )
    endpoint_waits = policy["endpoint_classical_confirmation"]
    endpoint_wait_alice = float(endpoint_waits["alice_extra_wait_ps"])
    endpoint_wait_bob = float(endpoint_waits["bob_extra_wait_ps"])
    base_by_edge = [
        elementary_edge_branches(edge_id, route, front, attack, context)
        for edge_id in edge_ids
    ]
    edge_age_tables: list[list[list[Branch]]] = []
    for edge_index, edge_id in enumerate(edge_ids):
        tables_for_edge: list[list[Branch]] = []
        for age_s, _age_weight in age_distributions[edge_index]:
            tables_for_edge.append(_rolling_edge_branches_for_age(
                edge_index=edge_index,
                edge_id=edge_id,
                hop_count=hops,
                age_s=age_s,
                base=base_by_edge[edge_index],
                front=front,
                keep_full_bsm=keep_full_bsm,
                keep_detector_click_records=keep_detector_click_records,
                endpoint_wait_alice_ps=endpoint_wait_alice,
                endpoint_wait_bob_ps=endpoint_wait_bob,
            ))
        edge_age_tables.append(tables_for_edge)
    weighted: list[Branch] = []
    for combo in itertools.product(*(range(len(distribution)) for distribution in age_distributions)):
        combo_weight = 1.0
        tables: list[list[Branch]] = []
        for edge_index, point_index in enumerate(combo):
            _age_s, point_weight = age_distributions[edge_index][point_index]
            combo_weight *= point_weight
            tables.append(edge_age_tables[edge_index][point_index])
        folded = _fold_rolling_edge_branch_tables(
            route=route,
            edge_tables=tables,
            attack=attack,
            front=front,
            context=context,
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
        )
        for branch in folded:
            weighted.append(Branch(
                probability=combo_weight * branch.probability,
                rho=branch.rho,
                frame=branch.frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=branch.public_bsm_record,
                detector_click_record=branch.detector_click_record,
            ))
    return aggregate_branches(
        normalize_branches(weighted),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def route_quantum_branches_rolling_alive_pair_moment_reduced(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool,
    context: ModelContext,
) -> list[Branch]:
    """Exact edge-local age marginal for the configured rolling protocol.

    Because selected age is hidden and only memory-local storage channels depend
    on it, each edge's finite selected-age distribution can be averaged into the
    detector-conditioned edge branch state before the linear route fold. This
    preserves same-edge age correlations (for example E[C(t)^2]) and is exactly
    equal to explicit Cartesian marginalization under the metadata assumptions.
    """
    edge_ids = list(route["edge_ids"])
    hops = len(edge_ids)
    policy = rolling_live_pair_policy(route, front, context)
    age_distributions, _age_meta = rolling_age_distributions_for_route(route, context)
    endpoint_waits = policy["endpoint_classical_confirmation"]
    endpoint_wait_alice = float(endpoint_waits["alice_extra_wait_ps"])
    endpoint_wait_bob = float(endpoint_waits["bob_extra_wait_ps"])

    def edge_branches(edge_index: int, edge_id: str) -> list[Branch]:
        base = elementary_edge_branches(edge_id, route, front, attack, context)
        weighted: list[Branch] = []
        for age_s, age_weight in age_distributions[edge_index]:
            aged = _rolling_edge_branches_for_age(
                edge_index=edge_index,
                edge_id=edge_id,
                hop_count=hops,
                age_s=age_s,
                base=base,
                front=front,
                keep_full_bsm=keep_full_bsm,
                keep_detector_click_records=keep_detector_click_records,
                endpoint_wait_alice_ps=endpoint_wait_alice,
                endpoint_wait_bob_ps=endpoint_wait_bob,
            )
            for branch in aged:
                weighted.append(Branch(
                    probability=age_weight * branch.probability,
                    rho=branch.rho,
                    frame=branch.frame,
                    eve_observation=branch.eve_observation,
                    public_bsm_record=branch.public_bsm_record,
                    detector_click_record=branch.detector_click_record,
                ))
        return aggregate_branches(
            normalize_branches(weighted),
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
        )

    edge_tables = [
        edge_branches(edge_index, edge_id)
        for edge_index, edge_id in enumerate(edge_ids)
    ]
    return _fold_rolling_edge_branch_tables(
        route=route,
        edge_tables=edge_tables,
        attack=attack,
        front=front,
        context=context,
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def route_quantum_branches_rolling_alive_pair(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool = True,
    context: ModelContext,
) -> list[Branch]:
    """Rolling live-pair route composer.

    This is a distinct protocol model from single-buffer swap-asap waiting:
    every edge contributes an already accepted live elementary pair with a
    configured age distribution. The age distribution is folded into the edge's
    detector-conditioned memory state before swaps. Availability affects
    delivery mass separately in delivery_stage_probabilities.
    """
    mode = normalized_rolling_age_evaluation(context)
    if mode == "cartesian":
        return route_quantum_branches_rolling_alive_pair_cartesian(
            route,
            front,
            attack,
            keep_full_bsm=keep_full_bsm,
            keep_detector_click_records=keep_detector_click_records,
            context=context,
        )
    return route_quantum_branches_rolling_alive_pair_moment_reduced(
        route,
        front,
        attack,
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
        context=context,
    )


def route_quantum_branches_multi_link_waiting(
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    *,
    keep_full_bsm: bool,
    keep_detector_click_records: bool = True,
    context: ModelContext,
    timing_policy: dict[str, Any] | None,
) -> list[Branch]:
    """Selected-chain h>=3 composer with grouped swap-asap stochastic waiting.

    Each wait signature from multi_link_wait_signature_distribution is
    evaluated through the full sequential composition with exact per-memory
    storage applications: every memory qubit's (duration, elapsed) pair is
    known exactly from the signature, so Gaussian/exponential/T1-T2 storage
    composition is exact within the recorded attempt cutoffs. Signatures are
    probability-weighted and summed, which is exact because every downstream
    instrument step is linear in probability * rho. A prefix cache reuses
    partial-chain segments shared between signatures.
    """
    edge_ids = list(route["edge_ids"])
    hops = len(edge_ids)
    if hops < 3:
        raise ValueError("multi-link waiting branch composer requires at least three elementary links")
    timing_policy = timing_policy or deterministic_storage_timing_policy(route, front)
    herald_age = {
        edge_id: float(timing_policy["edge_herald_ready_time_ps"][edge_id])
        for edge_id in edge_ids
    }

    def edge_base(edge_id: str) -> list[Branch]:
        edge_branches = elementary_edge_branches(edge_id, route, front, attack, context)
        return apply_memory_storage_channel_to_branches(
            edge_branches,
            front,
            qubits=(0, 1),
            storage_time_ps=herald_age[edge_id],
        )

    edge_success = {}
    for edge_id in edge_ids:
        probs, _meta = fock_elementary_edge_outcome_probabilities(edge_id, route, front, attack, context)
        edge_success[edge_id], _round_meta = multiplexed_edge_round(
            single_mode_success_probability=(
                probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0)
            ),
            edge_id=edge_id,
            route=route,
            front=front,
            context=context,
        )
    attempt_policy = attempt_period_policy(route, front, context)
    edge_periods = {
        edge_id: float(attempt_policy["edge_attempt_periods"][edge_id]["attempt_period_ps"])
        for edge_id in edge_ids
    }
    budgets, _budget_meta = waiting_attempt_budgets(context, edge_periods)
    signatures, _meta = multi_link_wait_signature_distribution(
        edge_success=[edge_success[edge_id] for edge_id in edge_ids],
        edge_periods_ps=[edge_periods[edge_id] for edge_id in edge_ids],
        attempt_budgets=[budgets[edge_id] for edge_id in edge_ids],
        context=context,
    )
    if not signatures:
        return []
    bases = {edge_id: edge_base(edge_id) for edge_id in edge_ids}
    repeaters = list(route["nodes"][1:-1])
    prefix_cache: dict[tuple[float, ...], tuple[list[Branch], float]] = {}

    def waited(
        branches: list[Branch],
        *,
        qubits: tuple[int, ...],
        duration_ps: float,
        elapsed_ps: float,
    ) -> list[Branch]:
        if duration_ps <= 0.0:
            return branches
        return apply_memory_storage_channel_to_branches(
            branches,
            front,
            qubits=qubits,
            storage_time_ps=duration_ps,
            elapsed_storage_time_ps=elapsed_ps,
        )

    def segment_for_prefix(prefix: tuple[float, ...]) -> tuple[list[Branch], float]:
        """Post-swap segment branches and the Alice-memory age for a signature prefix."""
        cached = prefix_cache.get(prefix)
        if cached is not None:
            return cached
        step = len(prefix) // 2
        if step == 1:
            a_1, b_1 = prefix
            left = waited(bases[edge_ids[0]], qubits=(0, 1), duration_ps=a_1, elapsed_ps=herald_age[edge_ids[0]])
            right = waited(bases[edge_ids[1]], qubits=(0, 1), duration_ps=b_1, elapsed_ps=herald_age[edge_ids[1]])
            alice_age = herald_age[edge_ids[0]] + a_1
        else:
            parent = prefix[:-2]
            c_i, b_i = prefix[-2], prefix[-1]
            b_prev = parent[-1]
            segment, alice_age = segment_for_prefix(parent)
            previous_edge = edge_ids[step - 1]
            left = waited(segment, qubits=(0,), duration_ps=c_i, elapsed_ps=alice_age)
            left = waited(left, qubits=(1,), duration_ps=c_i, elapsed_ps=herald_age[previous_edge] + b_prev)
            right = waited(bases[edge_ids[step]], qubits=(0, 1), duration_ps=b_i, elapsed_ps=herald_age[edge_ids[step]])
            alice_age += c_i
        swapped = _combine_attack_swap(
            left,
            right,
            repeaters[step - 1],
            attack,
            front,
            context,
            keep_detector_click_records=keep_detector_click_records,
        )
        result = (
            aggregate_branches(
                normalize_branches(swapped),
                keep_full_bsm=keep_full_bsm,
                keep_detector_click_records=keep_detector_click_records,
            ),
            alice_age,
        )
        if len(prefix) < 2 * (hops - 1):
            prefix_cache[prefix] = result
        return result

    wait_profile = endpoint_classical_wait_profile(route, front)
    herald_list = [herald_age[edge_id] for edge_id in edge_ids]
    signature_items = list(signatures.items())
    weighted_ages = []
    endpoint_waits = []
    for signature, weight in signature_items:
        ages, alice_wait, bob_wait = signature_memory_ages(
            signature,
            herald_ages_ps=herald_list,
            wait_profile=wait_profile,
        )
        weighted_ages.append((weight, ages))
        endpoint_waits.append((alice_wait, bob_wait))
    usable_weights, _loss_stats = apply_memory_loss_policies_to_histories(weighted_ages, front, context)
    usable_mass = math.fsum(usable_weights)
    if usable_mass <= 0.0:
        return []

    accumulated: list[Branch] = []
    for (signature, _weight), usable_weight, (alice_wait, bob_wait) in zip(
        signature_items, usable_weights, endpoint_waits
    ):
        if usable_weight <= 0.0:
            continue
        weight = usable_weight / usable_mass
        branches, alice_age = segment_for_prefix(signature)
        # Endpoint classical-confirmation storage: the endpoint memories keep
        # storing until every swap outcome has classically reached them.
        branches = waited(branches, qubits=(0,), duration_ps=alice_wait, elapsed_ps=alice_age)
        bob_elapsed = herald_age[edge_ids[-1]] + float(signature[-1])
        branches = waited(branches, qubits=(1,), duration_ps=bob_wait, elapsed_ps=bob_elapsed)
        for branch in branches:
            accumulated.append(Branch(
                probability=weight * branch.probability,
                rho=branch.rho,
                frame=branch.frame,
                eve_observation=branch.eve_observation,
                public_bsm_record=branch.public_bsm_record,
                detector_click_record=branch.detector_click_record,
            ))
    return aggregate_branches(
        normalize_branches(accumulated),
        keep_full_bsm=keep_full_bsm,
        keep_detector_click_records=keep_detector_click_records,
    )


def elementary_edge_branches(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext | None = None,
) -> list[Branch]:
    context = context or ModelContext()
    if context.physical_path == "fock_pnr_snspd":
        return fock_elementary_edge_branches(edge_id, route, front, attack, context)
    fidelity_multiplier = elementary_edge_memory_fidelity_multiplier(edge_id, route, attack)
    # SeQUeNCe Barrett-Kok feed-forward targets phi_plus, while this generator's
    # E91 endpoint protocol scores the psi_minus (anti-correlated) convention.
    # Express the elementary pair in the endpoint convention through the same
    # recorded public Pauli-frame bookkeeping the Fock path uses; (I x Y) maps
    # the SeQUeNCe Bell-diagonal phi_plus assignment onto its psi_minus twin.
    # Without this frame, odd hop counts score the wrong outcome relation.
    convention_frame = BK_REFERENCE_ENDPOINT_CONVENTION_FRAME
    base = [Branch(
        1.0,
        apply_frame_correction(
            elementary_memory_pair_density(front, fidelity_multiplier=fidelity_multiplier),
            convention_frame,
        ),
        frame=convention_frame,
        public_bsm_record=f"{edge_id}:barrett_kok_heralds_marginalized",
    )]
    if attack.target_active and attack.move_kind == "edge_measure_resend" and attack.target_id == edge_id:
        expanded: list[Branch] = []
        basis_choice = attack.edge_measure_basis_label if attack.edge_measure_basis_label != "random" else attack.basis_choice
        basis_choices = ("Z", "X") if basis_choice == "random" else (basis_choice,)
        basis_weight = 1.0 / len(basis_choices)
        for basis in basis_choices:
            measured = projective_expand(
                base,
                qubit=1,
                basis=basis,
                label_prefix=f"edge_measure_resend:edge={edge_id}",
                probability=attack.edge_measure_probability,
                information_probability=attack.information_probability,
                retain_observation=True,
            )
            for branch in measured:
                expanded.append(Branch(
                    probability=basis_weight * branch.probability,
                    rho=branch.rho,
                    frame=branch.frame,
                    eve_observation=branch.eve_observation,
                    public_bsm_record=branch.public_bsm_record,
                    detector_click_record=branch.detector_click_record,
                ))
        return normalize_branches(expanded)
    return base


def elementary_edge_memory_fidelity_multiplier(
    edge_id: str,
    route: dict[str, Any],
    attack: HistogramAttack,
) -> float:
    return 1.0


def fock_elementary_edge_outcome_probabilities(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> tuple[dict[str, float], dict[str, Any]]:
    data = fock_memory_emission_elementary_link_data(edge_id, route, front, attack, context)
    return dict(data["bsm_outcome_probabilities"]), dict(data["metadata"])


def fock_elementary_edge_branches(
    edge_id: str,
    route: dict[str, Any],
    front: dict[str, Any],
    attack: HistogramAttack,
    context: ModelContext,
) -> list[Branch]:
    return fock_memory_emission_elementary_link_instrument(edge_id, route, front, attack, context)


def _rescale_branch_mass(branches: list[Branch], total_mass: float) -> list[Branch]:
    if total_mass <= 0.0:
        return []
    return [
        Branch(
            probability=branch.probability * total_mass,
            rho=branch.rho,
            frame=branch.frame,
            eve_observation=branch.eve_observation,
            public_bsm_record=branch.public_bsm_record,
            detector_click_record=branch.detector_click_record,
        )
        for branch in branches
    ]


def apply_repeater_attack(
    branches: list[Branch],
    repeater: str,
    attack: HistogramAttack,
) -> list[Branch]:
    """Apply a targeted repeater move; preserves the total input branch mass.

    Callers may pass branch sets whose probabilities sum to a group/history
    weight rather than 1 (the two-link waiting composer does), so the trace-
    preserving move channels must not renormalize that mass away.
    """
    if not (
        attack.target_active
        and attack.target_kind in {"node", "memory"}
        and attack.target_id == repeater
    ):
        return branches
    input_mass = sum(max(0.0, branch.probability) for branch in branches)
    target_side = attack.target_memory_side if attack.target_memory_side in {"left", "right", "both"} else attack.memory_side
    qubits = memory_side_qubits(target_side)
    if attack.move_kind == "memory_degradation":
        operated = branches
        depol = clip_probability(attack.memory_depolarizing_probability)
        dephase = clip_probability(attack.memory_dephasing_probability)
        for qubit in qubits:
            if depol > 0.0:
                operated = apply_branch_channel(operated, qubit=qubit, ops=depolarizing_kraus(depol))
            if dephase > 0.0:
                operated = apply_branch_channel(operated, qubit=qubit, ops=dephasing_kraus(dephase, "Z"))
        return _rescale_branch_mass(normalize_branches(operated), input_mass)
    if attack.move_kind in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
        basis = "Z" if attack.move_kind.endswith("_Z") else "X"
        expanded = projective_expand_joint(
            branches,
            qubits=qubits,
            basis=basis,
            label_prefixes=tuple(
                f"{attack.move_kind}:node={repeater}:side={'left' if qubit == 1 else 'right'}"
                for qubit in qubits
            ),
            probability=attack.measurement_probability,
            information_probability=attack.information_probability,
        )
        return _rescale_branch_mass(expanded, input_mass)
    if attack.move_kind == "repeater_memory_dephasing_probe":
        operated = []
        theta = attack.dephasing_theta
        for branch in branches:
            rho = branch.rho
            for qubit in qubits:
                rho = reduced_dephasing_probe_channel(rho, theta, target_qubit=qubit)
            operated.append(Branch(
                branch.probability,
                clean_density(rho),
                branch.frame,
                join_labels(branch.eve_observation, f"repeater_memory_dephasing_probe:node={repeater}:theta={theta:.12g}"),
                branch.public_bsm_record,
                branch.detector_click_record,
            ))
        return _rescale_branch_mass(normalize_branches(operated), input_mass)
    return branches


def reduced_dephasing_probe_channel(
    rho: np.ndarray,
    theta: float,
    *,
    target_qubit: int,
) -> np.ndarray:
    """Reduced channel of the existing repeater-memory dephasing probe formalism.

    The probe attenuates coherences between branches that differ in the targeted
    computational-basis bit by cos(theta), matching the repository's first-party
    memory-probe formalism without importing the SeQUeNCe runtime package graph.

    The input density matrix is copied before the in-place coherence scaling so
    callers may pass branch states that alias cached instrument output.
    """
    n_qubits = int(round(math.log2(rho.shape[0])))
    tensor = np.array(rho, dtype=complex, copy=True).reshape((2,) * (2 * n_qubits))
    row_axis = target_qubit
    col_axis = target_qubit + n_qubits
    factor = math.cos(theta)
    for row_bit in (0, 1):
        for col_bit in (0, 1):
            if row_bit == col_bit:
                continue
            index = [slice(None)] * (2 * n_qubits)
            index[row_axis] = row_bit
            index[col_axis] = col_bit
            tensor[tuple(index)] *= factor
    return clean_density(tensor.reshape(rho.shape))
